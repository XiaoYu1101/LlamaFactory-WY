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

import copy
import json
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.export import freeze_dataset, get_export
from llamafactory.intent.jobs import JobRunner
from llamafactory.intent.models import IntentError, validate_intents
from llamafactory.intent.prompts import parse_samples
from llamafactory.intent.providers import DEFAULT_CONFIG
from llamafactory.intent.records import parse_examples, record_fingerprint, validate_record
from llamafactory.intent.service import IntentService
from llamafactory.intent.storage import Store
from llamafactory.intent.training import training_values


FORMATS = [
    {"system": "保留此系统提示。", "instruction": "执行任务", "input": "种子问题", "output": "种子答案"},
    {
        "system": "回答问题",
        "question": "种子问题",
        "answer": "答案",
        "metadata": {"tags": ["test"], "score": 2.5, "active": True, "extra": None},
    },
    {"prompt": "种子问题", "completion": "答案", "id": 7},
    {
        "system": "外部上下文",
        "messages": [{"role": "user", "content": "种子问题"}, {"role": "assistant", "content": "答案"}],
    },
    {
        "messages": [
            {"role": "system", "content": "不要改写系统消息"},
            {"role": "user", "content": "种子问题"},
            {"role": "assistant", "content": "答案"},
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": "保留系统上下文"},
            {"from": "human", "value": "种子问题"},
            {"from": "gpt", "value": "答案"},
        ]
    },
    {"sys_context": "系统上下文", "request_text": "种子问题", "result_text": "答案", "flags": [1, 2], "empty": []},
]


def scenario(example, label="generic", target=17, mapping=None):
    return {
        "label": label,
        "name": "通用测试",
        "description": "扩写同类记录，保留所有字段",
        "format": "json",
        "examples": [example] + [varied(example, 1000000 + i) for i in range(3)],
        "output_labels": [],
        "target": target,
        "training_mapping": mapping,
    }


def varied(record, index=1):
    row = copy.deepcopy(record)
    for key in ("input", "question", "prompt", "request_text"):
        if key in row:
            row[key] = f"独立测试输入 {index}"
            return row
    field = "messages" if "messages" in row else "conversations"
    for message in row[field]:
        if message.get("role", message.get("from")) in {"user", "human"}:
            message["content" if "content" in message else "value"] = f"独立测试输入 {index}"
            break
    return row


class Provider:
    config = dict(DEFAULT_CONFIG)
    descriptor = {"kind": "test", "model": "generic-fixture", "batch_size": 10}

    def __init__(self):
        self.index = 0
        self.calls = []

    def session(self):
        return nullcontext()

    def generate(self, messages):
        body = json.loads(messages[-1]["content"])
        self.calls.append(body)
        rows = []
        for _ in range(body["生成数量"]):
            self.index += 1
            rows.append(varied(body["参考样例"][0], self.index))
        return json.dumps({"samples": rows}, ensure_ascii=False)


@pytest.mark.parametrize("example", FORMATS)
def test_generic_generate_review_export_roundtrip(example, tmp_path):
    store = Store(tmp_path)
    mapping = (
        {"prompt": "request_text", "response": "result_text", "system": "sys_context"}
        if "request_text" in example
        else None
    )
    definition = scenario(example, mapping=mapping)
    version = store.save_project("generic", [definition])["version_id"]
    provider = Provider()
    runner = JobRunner(store)
    job = store.create_job(version, {"generic": 17}, provider.descriptor)
    runner.start(job, provider)
    result = runner.wait(job)
    assert result["status"] == "completed", result
    assert result["accepted"] == 17 and len(provider.calls) == 2
    assert len(provider.calls[0]["参考样例"]) == 3
    assert all(x in definition["examples"] for x in provider.calls[0]["参考样例"])
    rows = store.samples(version)[0]
    for row in rows:
        assert set(row["record"]) == set(example)
        if "system" in example:
            assert row["record"]["system"] == example["system"]
        if "metadata" in example:
            assert row["record"]["metadata"] == example["metadata"]
    store.review(version, rows, "approved")
    original = rows[0]["record"]
    edited = varied(original, 500)
    if "system" in edited:
        edited["system"] = "人工审核修订系统提示"
    current = store.samples(version)[0][0]
    store.review(version, [current], "edit", label="generic", record=edited)
    current = store.samples(version)[0][0]
    assert current["status"] == "pending" and current["record"] == edited
    assert current["original_record"] == original
    store.review(version, [current], "approved")
    manifest = freeze_dataset(store, version)
    frozen = get_export(store, manifest["id"])
    assert manifest["training_ready"]
    output = json.loads((store.root / "datasets" / manifest["id"] / "train.json").read_text(encoding="utf-8"))
    assert len(output) == 17 and edited in output
    assert all(set(row) == set(example) for row in output)
    registry = json.loads((store.root / "datasets" / manifest["id"] / "dataset_info.json").read_text())
    assert registry["intent_train"]["columns"].get("system") == (
        "system" if "system" in example else "sys_context" if mapping else None
    )
    assert Store(tmp_path).samples(version)[0][0]["record"] == edited
    assert frozen["manifest"]["count"] == 17


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("system"),
        lambda r: r.update(extra="bad"),
        lambda r: r.update(system="changed"),
        lambda r: r.update(input=123),
    ],
)
def test_generation_cannot_drop_add_retype_or_rewrite_context(mutate):
    row = varied(FORMATS[0])
    mutate(row)
    with pytest.raises(IntentError):
        validate_record(row, scenario(FORMATS[0]), generated=True)
    records, _, invalid = parse_samples(json.dumps({"samples": [row]}), 10, scenario(FORMATS[0]))
    assert records == [] and invalid == 1


def test_system_instruction_pairs_and_system_message_are_preserved():
    first = FORMATS[0]
    second = {**first, "system": "另一个系统", "instruction": "另一个任务", "input": "另一个问题"}
    definition = {**scenario(first), "examples": [first, second]}
    with pytest.raises(IntentError, match="system"):
        validate_record({**varied(first), "system": second["system"]}, definition, generated=True)
    bad = varied(FORMATS[4])
    bad["messages"][0]["content"] = "系统被改写"
    with pytest.raises(IntentError):
        validate_record(bad, scenario(FORMATS[4]), generated=True)
    assert record_fingerprint(first) != record_fingerprint({**first, "system": "不同系统"})


@pytest.mark.parametrize(
    "raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '[{"x":"text"},{"x":1}]', '[{"x":1},{"x":1,"y":2}]']
)
def test_invalid_or_inconsistent_reference_schema(raw):
    with pytest.raises(IntentError):
        parse_examples(raw)


def test_empty_input_and_primitive_custom_fields_remain_unchanged():
    row = {
        "system": "",
        "instruction": "原文\n保留换行  ",
        "input": "",
        "output": "答案",
        "enabled": False,
        "score": 2.5,
        "optional": None,
    }
    assert parse_examples(json.dumps([row], ensure_ascii=False)) == [row]
    assert parse_examples('{"any_field":123}') == [{"any_field": 123}]
    with pytest.raises(IntentError, match="类型"):
        validate_record({**row, "enabled": 0}, scenario(row))


def test_unmapped_json_download_and_custom_training_mapping(tmp_path):
    store = Store(tmp_path)
    example = FORMATS[-1]
    version = store.save_project("custom", [scenario(example)])["version_id"]
    store.add_sample(version, "generic", record=varied(example))
    store.review(version, store.samples(version)[0], "approved")
    manifest = freeze_dataset(store, version)
    assert manifest["training_ready"] is False and manifest["training_datasets"] == []
    assert get_export(store, manifest["id"])["manifest"]["count"] == 1
    with pytest.raises(IntentError):
        validate_intents([scenario(example, mapping={"prompt": "flags", "response": "result_text"})])


def test_mixed_scenario_training_registry_preserves_original_shapes(tmp_path):
    store = Store(tmp_path)
    version = store.save_project("mixed", [scenario(FORMATS[0], "text"), scenario(FORMATS[4], "chat")])["version_id"]
    store.add_sample(version, "text", record=varied(FORMATS[0]))
    store.add_sample(version, "chat", record=varied(FORMATS[4]))
    store.review(version, store.samples(version)[0], "approved")
    manifest = freeze_dataset(store, version)
    export = get_export(store, manifest["id"])
    assert len(manifest["training_datasets"]) == 2
    assert training_values(export, {"SFT": "sft"}, "fp32")["train.dataset"] == manifest["training_datasets"]


def test_rest_keeps_extra_fields_and_rejects_schema_changing_edits(tmp_path):
    service = IntentService(tmp_path)
    try:
        with TestClient(create_app(service=service)) as client:
            example = FORMATS[1]
            parsed = client.post("/intent-api/examples/parse", json={"examples": json.dumps([example])}).json()
            assert parsed["schema"]["properties"]["metadata"]["type"] == "object"
            assert parsed["training_mapping"]["system"] == "system"
            project = client.post(
                "/intent-api/projects", json={"name": "generic", "intents": [scenario(example)]}
            ).json()
            base = f"/intent-api/versions/{project['version_id']}"
            row = varied(example)
            assert client.post(base + "/samples", json={"label": "generic", "record": row}).status_code == 200
            saved = client.get(base + "/samples").json()["rows"][0]
            assert saved["record"] == saved["original_record"] == row
            payload = {
                "action": "edit",
                "label": "generic",
                "selected": [{"id": saved["id"], "revision": saved["revision"]}],
                "record": {k: v for k, v in row.items() if k != "system"},
            }
            assert client.post(base + "/review", json=payload).status_code == 400
            assert client.get(base + "/samples").json()["rows"][0]["record"] == row
    finally:
        service.lock.close()


def test_1000_system_records_have_exact_count_and_full_export(tmp_path):
    store = Store(tmp_path)
    version = store.save_project("system 1000", [scenario(FORMATS[0], target=1000)])["version_id"]
    provider = Provider()
    job = store.create_job(version, {"generic": 1000}, provider.descriptor)
    runner = JobRunner(store)
    runner.start(job, provider)
    assert runner.wait(job, 60)["accepted"] == 1000
    for page in range(1, 11):
        store.review(version, store.samples(version, page=page, page_size=100)[0], "approved")
    manifest = freeze_dataset(store, version)
    export = get_export(store, manifest["id"])
    assert manifest["count"] == 1000
    data = json.loads((store.root / "datasets" / manifest["id"] / "train.json").read_text(encoding="utf-8"))
    assert len(data) == 1000 and all(row["system"] == FORMATS[0]["system"] for row in data)
    assert export["manifest"]["training_ready"]


def test_custom_system_mapping_is_protected_during_generation():
    reference = FORMATS[-1]
    definition = scenario(
        reference, mapping={"prompt": "request_text", "response": "result_text", "system": "sys_context"}
    )
    assert validate_record(varied(reference), definition, generated=True)["sys_context"] == reference["sys_context"]
    with pytest.raises(IntentError, match="system"):
        validate_record({**varied(reference), "sys_context": "模型改写了系统上下文"}, definition, generated=True)


@pytest.mark.parametrize("field", ["system", "system_prompt"])
def test_chat_system_mapping_does_not_silently_discard_context(field):
    from llamafactory.intent.dataset_mapping import dataset_mapping

    record = {
        field: "root context",
        "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    }
    assert dataset_mapping(record)["columns"]["system"] == field
    record["messages"].insert(0, {"role": "system", "content": "conflicting context"})
    assert dataset_mapping(record) is None
    record["messages"][0]["content"] = "root context"
    assert dataset_mapping(record)["columns"]["system"] == field
