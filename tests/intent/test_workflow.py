# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest

from llamafactory.intent.export import freeze_dataset, get_export
from llamafactory.intent.jobs import JobRunner
from llamafactory.intent.models import IntentError, fingerprint, validate_intents
from llamafactory.intent.prompts import parse_samples
from llamafactory.intent.providers import DEFAULT_CONFIG, ProviderError
from llamafactory.intent.resources import ResourceGate
from llamafactory.intent.service import WorkspaceLock
from llamafactory.intent.storage import Store


def intents(target=17):
    return [
        {
            "label": "refund",
            "name": "退款查询",
            "description": "询问已申请退款的进度。",
            "examples": ["退款什么时候到账？"],
            "target": target,
        }
    ]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "intent")


def project(store, target=17, definitions=None):
    return store.save_project("客服分类", definitions or intents(target))


def records(store, version_id):
    with store.connect() as db:
        return [dict(x) for x in db.execute("SELECT * FROM samples WHERE version_id=? ORDER BY rowid", (version_id,))]


def approve_all(store, version_id):
    rows = records(store, version_id)
    store.review(
        version_id, [{"id": x["id"], "revision": x["revision"]} for x in rows if not x["deleted"]], "approved"
    )


class FakeProvider:
    config = dict(DEFAULT_CONFIG)
    descriptor = {"kind": "test", "model": "deterministic-test"}

    def __init__(self):
        self.calls = []
        self.counter = 0

    def session(self):
        return nullcontext()

    def generate(self, messages):
        self.calls.append(messages)
        body = json.loads(messages[1]["content"])
        result = [f"测试用户表达 {self.counter + i}：请查询这笔退款的进展。" for i in range(body["生成数量"])]
        self.counter += len(result)
        return json.dumps({"samples": result}, ensure_ascii=False)


def start(store, provider, version_id, targets=None):
    job_id = store.create_job(version_id, targets or {"refund": 17}, provider.descriptor)
    runner = JobRunner(store, retry_delay=0)
    runner.start(job_id, provider)
    return runner, job_id


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "nan", "inf", "oops"])
def test_invalid_quota(bad):
    value = intents()
    value[0]["target"] = bad
    with pytest.raises(IntentError):
        validate_intents(value)


def test_config_validation_and_single_seed():
    assert len(validate_intents(intents())[0]["examples"]) == 1
    duplicated = intents() + [{**intents()[0], "label": "REFUND"}]
    with pytest.raises(IntentError, match="重复"):
        validate_intents(duplicated)
    value = intents()
    value[0]["examples"] = []
    with pytest.raises(IntentError, match="样例"):
        validate_intents(value)
    assert fingerprint("ＡＢＣ   退款") == fingerprint("ABC 退款")
    assert fingerprint("退款 100 元") != fingerprint("退款 200 元")


def test_configuration_versions_and_persistence(store):
    first = project(store)
    assert store.save_project("改名", intents(), first["id"])["version_id"] == first["version_id"]
    approve_all(store, first["version_id"])
    changed = intents()
    changed[0]["description"] = "仅包括到账时间。"
    second = store.save_project("改名", changed, first["id"])
    assert second["version_id"] != first["version_id"]
    reopened = Store(store.root)
    assert reopened.project(first["id"])["current_version"] == second["version_id"]
    assert records(reopened, first["version_id"])[0]["status"] == "approved"
    assert records(reopened, second["version_id"])[0]["status"] == "pending"


def test_review_edit_delete_restore_and_audit(store):
    version = project(store)["version_id"]
    row = records(store, version)[0]
    store.review(version, [row], "approved")
    with pytest.raises(IntentError, match="已被修改"):
        store.review(version, [row], "rejected")
    row = records(store, version)[0]
    store.review(version, [row], "edit", text="什么时候才能收到退款？", label="refund")
    assert records(store, version)[0]["status"] == "pending"
    store.review(version, records(store, version), "delete")
    assert store.samples(version)[1] == 0
    assert store.samples(version, deleted=True)[1] == 1
    store.review(version, records(store, version), "restore")
    assert records(store, version)[0]["status"] == "pending"
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM review_events").fetchone()[0] == 4


def test_bulk_edit_is_atomic_on_stale_row(store):
    version = project(store)["version_id"]
    store.add_sample(version, "refund", "退款申请处理得怎么样？")
    rows = records(store, version)
    store.review(version, [rows[1]], "approved")
    with pytest.raises(IntentError):
        store.review(version, rows, "approved")
    assert records(store, version)[0]["status"] == "pending"


def test_duplicate_edit_and_cross_version_review_rejected(store):
    version = project(store)["version_id"]
    store.add_sample(version, "refund", "第二种问法")
    rows = records(store, version)
    with pytest.raises(IntentError, match="重复"):
        store.review(version, [rows[1]], "edit", text=rows[0]["text"], label="refund")
    other = store.save_project("另一个项目", intents())["version_id"]
    with pytest.raises(IntentError):
        store.review(other, [rows[0]], "approved")


def test_conflicts_and_frozen_content(store):
    definitions = intents() + [
        {
            "label": "cancel",
            "name": "取消订单",
            "description": "要求取消订单。",
            "examples": ["我要取消订单。"],
            "target": 10,
        }
    ]
    version = project(store, definitions=definitions)["version_id"]
    with pytest.raises(IntentError, match="待审核"):
        freeze_dataset(store, version)
    conflict_id = store.add_sample(version, "cancel", definitions[0]["examples"][0])
    approve_all(store, version)
    with pytest.raises(IntentError, match="冲突"):
        freeze_dataset(store, version)
    conflict = next(x for x in records(store, version) if x["id"] == conflict_id)
    store.review(version, [conflict], "rejected")
    manifest = freeze_dataset(store, version)
    export = get_export(store, manifest["id"])
    assert manifest["count"] == 2
    data = json.loads((Path(export["path"]) / "train.json").read_text(encoding="utf-8"))
    assert {x["output"] for x in data} == {"refund", "cancel"}
    assert len({x["instruction"] for x in data}) == 1
    first = records(store, version)[0]
    store.review(version, [first], "edit", text="修改后的问法", label=first["label"])
    assert get_export(store, manifest["id"])["manifest"] == manifest
    assert first["text"] in {x["input"] for x in data}
    (Path(export["path"]) / "train.json").write_text("[]", encoding="utf-8")
    with pytest.raises(IntentError, match="被修改"):
        get_export(store, manifest["id"])


def test_missing_category_blocks_export(store):
    version = project(store)["version_id"]
    store.review(version, records(store, version), "rejected")
    with pytest.raises(IntentError, match="没有已通过"):
        freeze_dataset(store, version)


def test_1000_samples_have_bounded_independent_context(store):
    version = project(store, 1000)["version_id"]
    provider = FakeProvider()
    runner, job_id = start(store, provider, version, {"refund": 1000})
    job = runner.wait(job_id, 30)
    assert job["status"] == "completed", job
    assert job["accepted"] == 1000 and len(provider.calls) == 100
    assert len(records(store, version)) == 1001  # Seed is not counted as generated output.
    lengths = []
    for call in provider.calls:
        assert [x["role"] for x in call] == ["system", "user"]
        body = json.loads(call[1]["content"])
        assert len(body["参考样例"]) == 1
        assert "测试用户表达" not in json.dumps(call, ensure_ascii=False)
        lengths.append(len(json.dumps(call, ensure_ascii=False)))
    assert max(lengths) - min(lengths) < 20


def test_tail_batch_and_idempotent_save(store):
    version = project(store)["version_id"]
    provider = FakeProvider()
    runner, job_id = start(store, provider, version)
    job = runner.wait(job_id)
    assert job["accepted"] == 17 and len(provider.calls) == 2
    assert json.loads(provider.calls[-1][1]["content"])["生成数量"] == 7
    with store.connect() as db:
        batch = dict(db.execute("SELECT * FROM batches WHERE job_id=? LIMIT 1", (job_id,)).fetchone())
    assert store.commit_batch(job_id, batch["id"], "refund", [], [], 0, 0) == batch["accepted"]
    assert store.job(job_id)["accepted"] == 17


def test_pause_resume_and_cancel_preserve_completed_batch(store):
    class Blocking(FakeProvider):
        entered, release = threading.Event(), threading.Event()

        def generate(self, messages):
            self.entered.set()
            assert self.release.wait(5)
            return super().generate(messages)

    version = project(store)["version_id"]
    provider = Blocking()
    runner, job_id = start(store, provider, version)
    assert provider.entered.wait(5)
    runner.pause(job_id)
    provider.release.set()
    assert runner.wait(job_id)["status"] == "paused"
    assert store.job(job_id)["accepted"] == 10
    runner.start(job_id, provider)
    assert runner.wait(job_id)["status"] == "completed"
    provider2 = Blocking()
    provider2.entered, provider2.release = threading.Event(), threading.Event()
    runner2, job2 = start(store, provider2, version)
    assert provider2.entered.wait(5)
    runner2.cancel(job2)
    provider2.release.set()
    assert runner2.wait(job2)["status"] == "cancelled"


def test_retry_and_non_retryable_error(store):
    class Flaky(FakeProvider):
        failures = 0

        def generate(self, messages):
            self.failures += 1
            if self.failures <= 2:
                raise ProviderError("限流", retryable=True)
            return super().generate(messages)

    version = project(store)["version_id"]
    provider = Flaky()
    runner, job_id = start(store, provider, version)
    assert runner.wait(job_id)["status"] == "completed"
    assert store.job(job_id)["attempts"] == 4

    class BadKey(FakeProvider):
        def generate(self, messages):
            raise ProviderError("认证失败")

    runner, job_id = start(store, BadKey(), version)
    assert runner.wait(job_id)["status"] == "failed"
    assert store.job(job_id)["attempts"] == 1


def test_duplicates_stop_and_restart_recovery(store):
    class Duplicate(FakeProvider):
        def generate(self, messages):
            return json.dumps({"samples": ["退款什么时候到账？"] * 10})

    version = project(store)["version_id"]
    runner, job_id = start(store, Duplicate(), version)
    result = runner.wait(job_id)
    assert result["status"] == "failed" and result["accepted"] == 0
    assert result["attempts"] == 5 and result["duplicates"] == 50
    store.transition(job_id, ("failed",), "running")
    reopened = Store(store.root)
    reopened.recover_jobs()
    assert reopened.job(job_id)["status"] == "interrupted"
    runner = JobRunner(reopened, retry_delay=0)
    runner.start(job_id, FakeProvider())
    assert runner.wait(job_id)["accepted"] == 17


def test_generated_samples_never_become_unreviewed_seeds(store):
    version = project(store)["version_id"]
    runner, job_id = start(store, FakeProvider(), version)
    runner.wait(job_id)
    assert len(store.seeds(version, "refund")) == 1
    rows = records(store, version)
    store.review(version, [rows[1]], "approved")
    assert len(store.seeds(version, "refund")) == 2


def test_provider_secrets_are_not_persisted(store):
    version = project(store)["version_id"]
    job = store.create_job(version, {"refund": 1}, {"kind": "test", "model": "mock", "api_key": "SECRET_SENTINEL"})
    assert "api_key" not in store.job(job)["provider"]
    with store.connect() as db:
        assert "SECRET_SENTINEL" not in "\n".join(db.iterdump())


@pytest.mark.parametrize("raw", ['{"samples": ["有效表达", "", null, 123]}', '```json\n{"samples":["有效表达"]}\n```'])
def test_parse_partial_and_fenced_json(raw):
    texts, produced, invalid = parse_samples(raw)
    assert texts == ["有效表达"]
    assert invalid == produced - 1


@pytest.mark.parametrize("raw", ["", "not json", '{"samples":', '{"samples":"bad"}', "123"])
def test_bad_generation_structure(raw):
    with pytest.raises(IntentError):
        parse_samples(raw)


def test_resource_lock_released_on_error():
    gate = ResourceGate()
    with pytest.raises(RuntimeError):
        with gate.reserve("生成"):
            with pytest.raises(IntentError):
                with gate.reserve("Chat"):
                    pass
            raise RuntimeError("test")
    with gate.reserve("新任务"):
        assert gate.owner == "新任务"


def test_workspace_has_only_one_owner(tmp_path):
    first = WorkspaceLock(tmp_path)
    try:
        with pytest.raises(IntentError):
            WorkspaceLock(tmp_path)
    finally:
        first.close()
    second = WorkspaceLock(tmp_path)
    second.close()


def test_only_one_job_can_claim_a_project(store):
    version = project(store)["version_id"]
    descriptor = FakeProvider.descriptor
    first = store.create_job(version, {"refund": 1}, descriptor)
    store.transition(first, ("queued",), "paused")
    second = store.create_job(version, {"refund": 1}, descriptor)
    with pytest.raises(IntentError, match="其他正在执行"):
        store.claim_job(first, descriptor)
    store.transition(second, ("queued",), "cancelled")
    store.claim_job(first, descriptor)
    with pytest.raises(IntentError, match="正在执行"):
        store.create_job(version, {"refund": 1}, descriptor)


def test_resume_rejects_changed_config_and_old_versions(store):
    item = project(store)
    job_id = store.create_job(item["version_id"], {"refund": 1}, FakeProvider.descriptor)
    store.transition(job_id, ("queued",), "paused")
    with pytest.raises(IntentError, match="配置与原任务不一致"):
        store.claim_job(job_id, {"kind": "test", "model": "different"})
    changed = intents()
    changed[0]["description"] = "新的意图定义"
    store.save_project("新版本", changed, item["id"])
    with pytest.raises(IntentError, match="旧生成任务"):
        store.claim_job(job_id, FakeProvider.descriptor)


def test_small_batch_size_has_sufficient_attempt_budget(store):
    version = project(store, 40)["version_id"]
    provider = FakeProvider()
    provider.config = {**DEFAULT_CONFIG, "batch_size": 1}
    provider.descriptor = {**provider.descriptor, "batch_size": 1}
    runner, job_id = start(store, provider, version, {"refund": 40})
    result = runner.wait(job_id)
    assert result["status"] == "completed" and result["attempts"] == 40


def test_filter_and_pagination_keep_selection_scoped(store):
    version = project(store)["version_id"]
    for index in range(35):
        store.add_sample(version, "refund", f"人工样本 {index}")
    page1, count, _ = store.samples(version, page=1)
    page2, _, page = store.samples(version, page=2)
    assert count == 36 and len(page1) == 25 and len(page2) == 11 and page == 2
    store.review(version, page2, "approved")
    assert store.samples(version, status="approved")[1] == 11
    assert store.samples(version, query="人工样本 34")[1] == 1
    assert store.samples(version, page=999)[2] == 2
