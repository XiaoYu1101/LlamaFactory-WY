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
import sqlite3
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.export import freeze_dataset, get_export
from llamafactory.intent.jobs import JobRunner
from llamafactory.intent.models import IntentError, validate_intents
from llamafactory.intent.prompts import parse_samples
from llamafactory.intent.providers import DEFAULT_CONFIG
from llamafactory.intent.records import parse_examples, suggest_labels
from llamafactory.intent.service import IntentService
from llamafactory.intent.storage import SCHEMA, Store


INSTRUCTION = "对监测问句进行多标签分类。标签：数值查询、变化分析、内容生成。输出多个标签时用顿号分隔。"
EXAMPLES = [
    dict(instruction=INSTRUCTION, input="查询监测记录并生成报告", output="数值查询、内容生成"),
    dict(instruction=INSTRUCTION, input="分析近期变化", output="变化分析"),
    dict(instruction=INSTRUCTION, input="查看本周沉降数据", output="数值查询"),
    dict(instruction=INSTRUCTION, input="生成监测报告", output="内容生成"),
]


def scenario(target=1000):
    return dict(
        label="monitoring",
        name="监测数据扩写",
        description="生成自然问句和相应多标签答案",
        format="alpaca",
        examples=EXAMPLES,
        target=target,
        output_labels=["数值查询", "变化分析", "内容生成"],
    )


class JSONProvider:
    config = dict(DEFAULT_CONFIG)
    descriptor = {"kind": "test", "model": "json-fixture", **config}

    def __init__(self):
        self.calls = []
        self.index = 0

    def session(self):
        return nullcontext()

    def generate(self, messages):
        self.calls.append(messages)
        body = json.loads(messages[-1]["content"])
        rows = []
        for _ in range(body["生成数量"]):
            self.index += 1
            rows.append(
                dict(instruction=INSTRUCTION, input=f"模拟接口测试问句 {self.index}", output="数值查询、内容生成")
            )
        return json.dumps({"samples": rows}, ensure_ascii=False)


@pytest.mark.parametrize("payload", [EXAMPLES, json.dumps(EXAMPLES), "\n".join(json.dumps(x) for x in EXAMPLES)])
def test_parse_json_array_and_jsonl(payload):
    assert parse_examples(payload) == EXAMPLES
    assert suggest_labels(EXAMPLES) == ["数值查询", "变化分析", "内容生成"]


@pytest.mark.parametrize("payload", ["plain question", "{}", "[1]", "[]"])
def test_bad_reference_json_reports_error(payload):
    with pytest.raises(IntentError):
        parse_examples(payload)


def test_duplicate_seed_conflicts_and_unknown_answer():
    with pytest.raises(IntentError, match="不同答案"):
        parse_examples([EXAMPLES[0], {**EXAMPLES[0], "output": "变化分析"}])
    with pytest.raises(IntentError, match="答案标签"):
        validate_intents([{**scenario(), "output_labels": ["错误标签"]}])


def test_response_rejects_plain_strings_wrong_instruction_and_unknown_labels():
    payload = [
        "old text",
        {**EXAMPLES[0], "instruction": "rewritten"},
        {**EXAMPLES[0], "output": "monitoring"},
        EXAMPLES[0],
    ]
    rows, produced, invalid = parse_samples(json.dumps({"samples": payload}), 10, scenario())
    assert rows == [EXAMPLES[0]]
    assert produced == 4 and invalid == 3


def test_1000_full_records_review_export_and_reload(tmp_path):
    store = Store(tmp_path)
    version = store.save_project("JSON 工作流", [scenario()])["version_id"]
    assert store.samples(version)[1] == 0  # References do not silently become training data.
    provider = JSONProvider()
    job = store.create_job(version, {"monitoring": 1000}, provider.descriptor)
    runner = JobRunner(store)
    runner.start(job, provider)
    result = runner.wait(job, 60)
    assert result["status"] == "completed", result
    assert result["accepted"] == 1000 and len(provider.calls) == 100
    assert store.samples(version)[1] == 1000
    assert max(len(json.dumps(x)) for x in provider.calls) - min(len(json.dumps(x)) for x in provider.calls) < 1000
    for messages in provider.calls:
        refs = json.loads(messages[-1]["content"])["参考样例"]
        assert len(refs) == 3
        assert all(set(x) == {"instruction", "input", "output"} for x in refs)
        assert all("模拟接口" not in x["input"] for x in refs)
    with pytest.raises(IntentError, match="待审核"):
        freeze_dataset(store, version)
    for page in range(1, 11):
        rows = store.samples(version, page=page, page_size=100)[0]
        store.review(version, rows, "approved")
    manifest = freeze_dataset(store, version)
    export = get_export(store, manifest["id"])
    training = json.loads((store.root / "datasets" / manifest["id"] / "train.json").read_text(encoding="utf-8"))
    assert len(training) == manifest["count"] == 1000
    assert all(x["instruction"] == INSTRUCTION and x["output"] == "数值查询、内容生成" for x in training)
    assert all(set(x) == {"instruction", "input", "output"} for x in training)
    assert Store(tmp_path).samples(version)[1] == 1000
    assert export["manifest"]["reference_samples_included"] is False


def test_json_rest_edit_delete_restore_conflicts_and_download(tmp_path):
    service = IntentService(tmp_path)
    try:
        with TestClient(create_app(service=service)) as client:
            parsed = client.post("/intent-api/examples/parse", json={"examples": json.dumps(EXAMPLES)}).json()
            assert parsed["count"] == 4
            response = client.post("/intent-api/projects", json={"name": "测试", "intents": [scenario(2)]})
            assert response.status_code == 200
            version = response.json()["version_id"]
            base = f"/intent-api/versions/{version}"
            record = dict(label="monitoring", text="查询监测资料", instruction=INSTRUCTION, output="数值查询")
            assert client.post(base + "/samples", json={**record, "output": "monitoring"}).status_code == 400
            assert client.post(base + "/samples", json=record).status_code == 200
            assert client.post(base + "/samples", json={**record, "output": "内容生成"}).status_code == 400

            def row():
                return client.get(base + "/samples").json()["rows"][0]

            def review(action, selected=None, **extra):
                selected = selected or row()
                return client.post(
                    base + "/review",
                    json={
                        "selected": [{"id": selected["id"], "revision": selected["revision"]}],
                        "action": action,
                        **extra,
                    },
                )

            old = row()
            assert review("approved").status_code == 200
            assert review("edit", **{**record, "output": "数值查询、内容生成"}).status_code == 200
            assert row()["status"] == "pending" and row()["original_output"] == "数值查询"
            assert review("approved", selected=old).status_code == 400
            assert review("delete").status_code == 200
            deleted = client.get(base + "/samples?deleted=true").json()["rows"][0]
            assert review("restore", selected=deleted).status_code == 200
            assert row()["status"] == "pending"
            assert review("approved").status_code == 200
            exported = client.post(base + "/exports", json={})
            assert exported.status_code == 200
            exported_id = exported.json()["id"]
            data = client.get(f"/intent-api/exports/{exported_id}/json").json()
            assert data == [{"instruction": INSTRUCTION, "input": record["text"], "output": "数值查询、内容生成"}]
            assert (
                review("edit", **{**record, "instruction": "人工修订后的指令", "output": "内容生成"}).status_code
                == 200
            )
            assert client.get(f"/intent-api/exports/{exported_id}/json").json() == data
    finally:
        service.lock.close()


def test_migration_does_not_overwrite_existing_workspaces(tmp_path):
    with sqlite3.connect(tmp_path / "intent.sqlite3") as db:
        db.executescript(SCHEMA)
    store = Store(tmp_path)
    version = store.save_project(
        "legacy", [dict(label="legacy", name="旧类别", description="旧说明", examples=["旧文本"], target=1)]
    )["version_id"]
    row = store.samples(version)[0][0]
    assert row["text"] == "旧文本" and row["instruction"] == row["output"] == ""
    store.review(version, [row], "approved")
    assert freeze_dataset(store, version)["count"] == 1
