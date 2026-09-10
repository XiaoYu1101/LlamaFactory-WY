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

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.export import freeze_dataset
from llamafactory.intent.models import IntentError, validate_intents
from llamafactory.intent.prompts import build_messages
from llamafactory.intent.service import IntentService
from llamafactory.intent.storage import Store


def scene(label="task"):
    return dict(
        label=label,
        name="样例任务",
        description="生成常识问题",
        format="json",
        target=10,
        examples=[dict(instruction="回答问题", input=f"参考问题 {i}", output="参考答案") for i in range(4)],
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_at_least_four_distinct_references(count):
    value = scene()
    value["examples"] = value["examples"][:count]
    with pytest.raises(IntentError, match="至少需要 4"):
        validate_intents([value])
    value["examples"] = [value["examples"][0]] * 4
    with pytest.raises(IntentError, match="至少需要 4"):
        validate_intents([value])


@pytest.mark.parametrize("n", [4, 6, 7, 100])
def test_three_references_are_balanced_and_spread(n):
    definition = scene()
    seeds = [{"id": str(i), "record": {**definition["examples"][0], "input": f"seed {i}"}} for i in range(n)]
    usage, previous = {}, []
    covered = set()
    for batch in range(n * 2):
        messages, selected = build_messages(
            [definition], definition, seeds, 10, batch, usage=usage, previous=previous, job_id="stable-job"
        )
        assert len(selected) == len(set(selected)) == 3
        assert len(json.loads(messages[-1]["content"])["参考样例"]) == 3
        if n >= 6:
            assert not set(selected) & set(previous)
        for key in selected:
            usage[key] = usage.get(key, 0) + 1
        covered.update(selected)
        previous = selected
        assert max(usage.get(str(i), 0) for i in range(n)) - min(usage.get(str(i), 0) for i in range(n)) <= 1
    assert len(covered) == n


def test_cross_page_approval_and_download_preserve_rejected_deleted(tmp_path):
    service = IntentService(tmp_path)
    try:
        store = service.store
        version = store.save_project("bulk", [scene()])["version_id"]
        for i in range(57):
            store.add_sample(version, "task", record=dict(instruction="回答问题", input=f"new {i}", output="answer"))
        rows = store.samples(version, page_size=100)[0]
        store.review(version, [rows[0]], "rejected")
        store.review(version, [rows[1]], "delete")
        store.review(version, [rows[2]], "approved")
        with TestClient(create_app(service=service)) as client:
            result = client.post(f"/intent-api/versions/{version}/approve-export")
            assert result.status_code == 200, result.text
            manifest = result.json()
            assert manifest["approved_now"] == 54 and manifest["count"] == 55
            data = client.get(f"/intent-api/exports/{manifest['id']}/json").json()
            assert len(data) == 55
        with store.connect() as db:
            assert db.execute("SELECT count(*) FROM samples WHERE status='rejected'").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM samples WHERE deleted=1").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM review_events WHERE action='approved'").fetchone()[0] == 55
    finally:
        service.lock.close()


def test_approval_rolls_back_when_export_cannot_finish(tmp_path):
    store = Store(tmp_path)
    version = store.save_project("rollback", [scene(), scene("missing")])["version_id"]
    store.add_sample(version, "task", record=dict(instruction="回答问题", input="new", output="answer"))
    with pytest.raises(IntentError, match="没有已通过"):
        freeze_dataset(store, version, approve_pending=True)
    assert store.samples(version)[0][0]["status"] == "pending"
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM review_events").fetchone()[0] == 0


def test_one_click_approval_blocked_during_generation(tmp_path):
    store = Store(tmp_path)
    version = store.save_project("active", [scene()])["version_id"]
    store.create_job(version, {"task": 10}, {"model": "test", "kind": "test"})
    with pytest.raises(IntentError, match="暂停"):
        store.approve_all(version)
