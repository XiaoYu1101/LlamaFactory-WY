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

import io
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app, install_routes
from llamafactory.intent.providers import DEFAULT_CONFIG
from llamafactory.intent.service import IntentService


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
    service = IntentService(tmp_path / "workspace", config)
    app = create_app(service=service)
    with TestClient(app) as client:
        client.service = service
        yield client
    service.lock.close()


def create_project(client, name="售后分类"):
    response = client.post(
        "/intent-api/projects",
        json={
            "name": name,
            "intents": [
                {
                    "label": "refund",
                    "name": "退款",
                    "description": "查询退款进度",
                    "examples": ["退款到哪了"],
                    "target": 17,
                }
            ],
        },
    )
    assert response.status_code == 200
    return response.json()


def rows(client, version):
    return client.get(f"/intent-api/versions/{version}/samples").json()["rows"]


def review(client, version, sample, action, **extra):
    return client.post(
        f"/intent-api/versions/{version}/review",
        json={"action": action, "selected": [{"id": sample["id"], "revision": sample["revision"]}], **extra},
    )


def test_ant_static_and_public_settings(client, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-expose-test-secret")
    home = client.get("/")
    assert home.status_code == 200
    assert "意图工作台" in home.text
    assert "/intent/assets/" in home.text
    settings = client.get("/intent-api/settings")
    assert settings.json()["batch_size"] == 10
    assert not settings.json()["integrated"]
    assert "do-not-expose" not in settings.text
    assert settings.headers["cache-control"] == "no-store"


def test_project_history_and_export_download(client):
    project = create_project(client)
    version = project["version_id"]
    assert client.post(f"/intent-api/versions/{version}/exports", json={}).status_code == 400
    sample = rows(client, version)[0]
    assert review(client, version, sample, "approved").status_code == 200
    manifest = client.post(f"/intent-api/versions/{version}/exports", json={}).json()
    response = client.get(f"/intent-api/exports/{manifest['id']}/download")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        training = json.loads(archive.read("train.json"))
        assert training[0]["input"] == "退款到哪了"
        assert training[0]["output"] == "refund"
    data = client.get(f"/intent-api/projects/{project['id']}").json()
    definition = data["version"]["config"]
    definition[0]["target"] = 18
    changed = client.post(
        "/intent-api/projects", json={"name": "新版", "project_id": project["id"], "intents": definition}
    ).json()
    assert changed["version_id"] != version
    assert len(client.get(f"/intent-api/projects/{project['id']}").json()["versions"]) == 2
    assert client.get(f"/intent-api/versions/{version}").json()["exports"][0]["manifest"]["count"] == 1


def test_review_edit_delete_restore_and_stale_rejection(client):
    version = create_project(client)["version_id"]
    old = rows(client, version)[0]
    assert review(client, version, old, "approved").status_code == 200
    assert review(client, version, old, "delete").status_code == 400
    sample = rows(client, version)[0]
    assert review(client, version, sample, "edit", label="refund", text="我的退款什么时候到账").status_code == 200
    edited = rows(client, version)[0]
    assert edited["status"] == "pending" and edited["original"] == old["text"]
    assert review(client, version, edited, "delete").status_code == 200
    assert not rows(client, version)
    deleted = client.get(f"/intent-api/versions/{version}/samples?deleted=true").json()["rows"][0]
    assert review(client, version, deleted, "restore").status_code == 200
    assert rows(client, version)[0]["status"] == "pending"


def test_selection_cannot_cross_versions_and_manual_duplicates(client):
    first, second = create_project(client), create_project(client, "第二个项目")
    sample = rows(client, first["version_id"])[0]
    assert review(client, second["version_id"], sample, "approved").status_code == 400
    url = f"/intent-api/versions/{first['version_id']}/samples"
    body = {"label": "refund", "text": "新样本"}
    assert client.post(url, json=body).status_code == 200
    assert client.post(url, json=body).status_code == 400
    assert client.get(url + "?query=新样本&status=pending").json()["total"] == 1
    assert client.get(url + "?page_size=101").status_code == 422


def test_cross_site_mutation_blocked(client):
    result = client.post("/intent-api/projects", json={}, headers={"origin": "https://another-site.example"})
    assert result.status_code == 403
    assert client.get("/intent-api/projects").json() == []


def test_damaged_export_does_not_block_workspace(client):
    version = create_project(client)["version_id"]
    review(client, version, rows(client, version)[0], "approved")
    manifest = client.post(f"/intent-api/versions/{version}/exports", json={}).json()
    detail = client.get(f"/intent-api/versions/{version}").json()
    (Path(detail["exports"][0]["path"]) / "train.json").write_text("[]", encoding="utf-8")
    assert client.get(f"/intent-api/versions/{version}").status_code == 200
    assert client.get(f"/intent-api/exports/{manifest['id']}/download").status_code == 400


def test_rest_job_uses_existing_worker_and_exact_quota(client, monkeypatch):
    class Provider:
        config = dict(DEFAULT_CONFIG)
        descriptor = {"kind": "test", "model": "fake"}
        count = 0

        def session(self):
            return nullcontext()

        def generate(self, messages):
            amount = json.loads(messages[1]["content"])["生成数量"]
            result = [f"退款表达 {self.count + i}" for i in range(amount)]
            self.count += amount
            return json.dumps({"samples": result})

    provider = Provider()
    monkeypatch.setattr(client.service, "provider", lambda chatter=None: provider)
    version = create_project(client)["version_id"]
    result = client.post(f"/intent-api/versions/{version}/jobs", json={})
    assert result.status_code == 200
    job = client.service.jobs.wait(result.json()["id"])
    assert job["status"] == "completed" and job["accepted"] == 17 and job["attempts"] == 2
    detail = client.get(f"/intent-api/versions/{version}").json()
    assert detail["jobs"][0]["accepted"] == 17
    assert len(rows(client, version)) == 18
    assert client.post(f"/intent-api/versions/{version}/jobs", json={"label": "refund"}).status_code == 400


def test_mount_preserves_gradio_routes(client):
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown("original training interface")
    install_routes(demo.app, client.service)
    with TestClient(demo.app) as mounted:
        assert mounted.get("/").status_code == 200
        assert mounted.get("/config").status_code == 200
        assert mounted.get("/intent/").status_code == 200
        assert mounted.get("/intent-api/settings").status_code == 200
