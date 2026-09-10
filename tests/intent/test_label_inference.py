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
import time
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.label_inference import LabelInference, parse_result
from llamafactory.intent.models import IntentError
from llamafactory.intent.service import IntentService
from llamafactory.intent.storage import Store


EXAMPLES = [
    {
        "system": "分类标签只是测试",
        "question": f"问题 {i}",
        "answer": "A" if i % 2 else "B",
        "metadata": {"index": i, "enabled": True},
    }
    for i in range(4)
]
RESULT = {"classification": True, "labels": ["B", "A"], "explanation": "根据完整样例推断出两类，仍需人工核对。"}


class Provider:
    descriptor = {"kind": "test", "model": "label-model", "api_key": "secret-not-for-storage"}

    def __init__(self, gate=None, failure=None):
        self.gate, self.failure = gate, failure
        self.calls = []

    def session(self):
        return nullcontext()

    def generate(self, messages):
        self.calls.append(messages)
        if self.gate:
            assert self.gate.wait(5)
        if self.failure:
            raise self.failure
        return json.dumps(RESULT, ensure_ascii=False)


def payload():
    return {
        "examples": EXAMPLES,
        "description": "判断分类标签",
        "snapshot": {
            "project_name": "尚未保存的项目",
            "project_id": "",
            "version_id": "",
            "intents": [],
            "index": 0,
            "form": {
                "label": "test",
                "name": "待恢复场景",
                "description": "判断分类标签",
                "target": 1000,
                "examples": json.dumps(EXAMPLES),
            },
        },
    }


def wait(manager, task_id, status="completed"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        task = manager.get(task_id)
        if task["status"] == status:
            return task
        time.sleep(0.01)
    pytest.fail(str(manager.get(task_id)))


def test_background_task_survives_client_close_restores_draft_and_user_labels(tmp_path):
    service = IntentService(tmp_path)
    gate = threading.Event()
    provider = Provider(gate)
    service.provider = lambda *args: provider
    try:
        with TestClient(create_app(service=service)) as client:
            response = client.post("/intent-api/label-tasks", json=payload())
            assert response.status_code == 200
            task_id = response.json()["id"]
            assert response.json()["status"] in {"queued", "running"}
            again = client.post("/intent-api/label-tasks", json=payload()).json()
            assert again["id"] == task_id
        # HTTP client/browser closes while model request remains active.
        gate.set()
        task = wait(service.label_tasks, task_id)
        assert len(provider.calls) == 1
        assert json.loads(provider.calls[0][-1]["content"])["examples"] == EXAMPLES
        assert task["snapshot"] == payload()["snapshot"]
        assert task["result"] == RESULT
        assert "secret-not-for-storage" not in json.dumps(task)
        with TestClient(create_app(service=service)) as client:
            tasks = client.get("/intent-api/label-tasks").json()
            assert tasks["total"] == 1 and tasks["active"] == 0
            assert tasks["rows"][0]["name"] == "待恢复场景"
            scenario = {
                "label": "test",
                "name": "待恢复场景",
                "description": payload()["description"],
                "format": "json",
                "target": 1000,
                "examples": EXAMPLES,
                "output_labels": ["B", "A", "用户新增"],
                "label_task_id": task_id,
            }
            saved = client.post("/intent-api/projects", json={"name": "用户项目", "intents": [scenario]})
            assert saved.status_code == 200, saved.text
            version = service.store.version(saved.json()["version_id"])
            assert version["config"][0]["output_labels"] == ["B", "A", "用户新增"]
            scenario["examples"] = [{**EXAMPLES[0], "answer": "A、B"}, *EXAMPLES[1:]]
            assert client.post("/intent-api/scenarios/validate", json=scenario).status_code == 400
        # Completed tasks remain available after service re-creation.
        service.close()
        recovered = IntentService(tmp_path)
        try:
            assert recovered.label_tasks.get(task_id)["result"] == RESULT
        finally:
            recovered.close()
    finally:
        gate.set()
        service.close()


def test_pending_and_stale_result_cannot_be_saved(tmp_path):
    service = IntentService(tmp_path)
    gate = threading.Event()
    service.provider = lambda *args: Provider(gate)
    try:
        body = payload()
        with TestClient(create_app(service=service)) as client:
            task = client.post("/intent-api/label-tasks", json=body).json()
            scene = {
                "label": "test",
                "name": "test",
                "description": body["description"],
                "format": "json",
                "target": 10,
                "examples": EXAMPLES,
                "output_labels": [],
                "label_task_id": task["id"],
            }
            assert client.post("/intent-api/scenarios/validate", json=scene).status_code == 400
            gate.set()
            wait(service.label_tasks, task["id"])
            assert client.post("/intent-api/scenarios/validate", json=scene).status_code == 200
            scene["description"] = "新的生成要求"
            assert client.post("/intent-api/projects", json={"name": "test", "intents": [scene]}).status_code == 400
    finally:
        gate.set()
        service.close()


def test_failure_is_persisted_and_explicit_retry_recovers(tmp_path):
    service = IntentService(tmp_path)
    service.provider = lambda *args: Provider(failure=RuntimeError("secret-not-for-storage"))
    try:
        with TestClient(create_app(service=service)) as client:
            task = client.post("/intent-api/label-tasks", json=payload()).json()
            failed = wait(service.label_tasks, task["id"], "failed")
            assert "secret-not-for-storage" not in json.dumps(failed)
            service.provider = lambda *args: Provider()
            assert client.post(f"/intent-api/label-tasks/{task['id']}/retry").status_code == 200
            assert wait(service.label_tasks, task["id"])["attempts"] == 2
            assert client.post(f"/intent-api/label-tasks/{task['id']}/retry").status_code == 400
    finally:
        service.close()


def test_restart_marks_pending_task_interrupted_without_model_call(tmp_path, monkeypatch):
    store = Store(tmp_path)
    manager = LabelInference(store)
    monkeypatch.setattr(manager, "_launch", lambda *args: None)
    body = payload()
    task = manager.create(
        body["examples"], body["description"], body["snapshot"], {}, lambda _: pytest.fail("must not call")
    )
    recovered = LabelInference(Store(tmp_path))
    assert recovered.get(task["id"])["status"] == "interrupted"
    assert recovered.get(task["id"])["snapshot"] == body["snapshot"]


@pytest.mark.parametrize(
    "result",
    [
        {"classification": False, "labels": [], "explanation": "自由问答，不限定标签"},
        {"classification": True, "labels": ["一个类别"], "explanation": "样例仅支持这一类别"},
    ],
)
def test_nonclassification_and_single_label_are_supported(result):
    assert parse_result(json.dumps(result)) == result


@pytest.mark.parametrize(
    "result",
    [
        "not json",
        {},
        {"classification": False, "labels": ["A"], "explanation": "bad"},
        {"classification": True, "labels": ["A", "A"], "explanation": "bad"},
        {"classification": True, "labels": ["A、B"], "explanation": "bad"},
    ],
)
def test_invalid_inference_does_not_unlock_fields(result):
    with pytest.raises(IntentError):
        parse_result(json.dumps(result))


def test_reference_answer_order_is_not_rewritten_by_inferred_label_order(tmp_path):
    service = IntentService(tmp_path)
    service.provider = lambda *args: Provider()
    try:
        data = payload()
        data["examples"] = [{**row, "answer": "A、B"} for row in EXAMPLES]
        with TestClient(create_app(service=service)) as client:
            task = client.post("/intent-api/label-tasks", json=data).json()
            wait(service.label_tasks, task["id"])
            scene = {
                "label": "test",
                "name": "test",
                "description": data["description"],
                "format": "json",
                "target": 10,
                "examples": data["examples"],
                "output_labels": ["B", "A"],
                "label_task_id": task["id"],
            }
            result = client.post("/intent-api/scenarios/validate", json=scene)
            assert result.status_code == 200, result.text
            assert result.json()["examples"][0]["answer"] == "A、B"
    finally:
        service.close()


def test_restored_old_draft_cannot_overwrite_new_project_version(tmp_path):
    store = Store(tmp_path)
    scene = {
        "label": "test",
        "name": "test",
        "description": "old",
        "format": "json",
        "target": 10,
        "examples": EXAMPLES,
    }
    first = store.save_project("project", [scene])
    newer = store.save_project("project", [{**scene, "description": "new"}], first["id"])
    with pytest.raises(IntentError, match="不能用旧任务草稿覆盖"):
        store.save_project("project", [scene], first["id"], first["version_id"])
    assert store.project(first["id"])["current_version"] == newer["version_id"]
