import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.export import freeze_dataset
from llamafactory.intent.models import IntentError
from llamafactory.intent.service import IntentService


def scene(label):
    return dict(
        label=label,
        name=label,
        description="生成",
        format="json",
        target=10,
        examples=[dict(instruction="分类", input=f"样例{i}", output="A") for i in range(4)],
    )


@pytest.fixture
def service(tmp_path):
    item = IntentService(tmp_path)
    yield item
    item.close()


def add(store, version, label, n=1):
    for i in range(n):
        store.add_sample(version, label, record=dict(instruction="分类", input=f"新样本{i}", output="A"))


def test_delete_scenario_persists_and_removes_related_history_samples(service):
    store = service.store
    first = store.save_project("project", [scene("a"), scene("b")])
    add(store, first["version_id"], "a", 2)
    changed = scene("b")
    changed["target"] = 20
    second = store.save_project("project", [scene("a"), changed], first["id"])
    add(store, second["version_id"], "a", 2)
    add(store, second["version_id"], "b", 3)
    with TestClient(create_app(service=service)) as client:
        response = client.post(f"/intent-api/versions/{second['version_id']}/delete-scenario", json={"label": "a"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["deleted"] == 4
        project = client.get(f"/intent-api/projects/{first['id']}").json()
        assert project["current_version"] == result["version_id"]
        assert [s["label"] for s in store.version(result["version_id"])["config"]] == ["b"]
    assert store.samples(result["version_id"])[1] == 3
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM samples WHERE label='a'").fetchone()[0] == 0
    with pytest.raises(IntentError, match="更新"):
        store.save_project("project", [scene("a"), scene("b")], first["id"], base_version_id=second["version_id"])
    assert store.version(first["version_id"])["config"][0]["label"] == "b"


def test_clear_is_cross_page_includes_recycle_bin_and_keeps_references(service):
    store = service.store
    version = store.save_project("clear", [scene("a")])["version_id"]
    add(store, version, "a", 55)
    rows = store.samples(version, page_size=100)[0]
    store.review(version, [rows[0]], "approved")
    store.review(version, [rows[1]], "rejected")
    store.review(version, [rows[2]], "delete")
    job = store.create_job(version, {"a": 10}, {"model": "test"})
    store.transition(job, ("queued",), "paused")
    with TestClient(create_app(service=service)) as client:
        response = client.post(f"/intent-api/versions/{version}/clear-samples", json={})
        assert response.status_code == 200, response.text
        assert response.json()["deleted"] == 55
        assert client.get(f"/intent-api/versions/{version}/samples").json()["total"] == 0
    assert store.samples(version, deleted=True)[1] == 0
    assert len(store.version(version)["config"][0]["examples"]) == 4
    assert store.job(job)["status"] == "cancelled"
    add(store, version, "a")  # Same text can be generated/added again after a real clear.


def test_active_generation_prevents_partial_deletion(service):
    store = service.store
    version = store.save_project("busy", [scene("a")])["version_id"]
    add(store, version, "a")
    store.create_job(version, {"a": 10}, {"model": "test"})
    for action in (lambda: store.clear_samples(version), lambda: store.delete_scenario(version, "a")):
        with pytest.raises(IntentError, match="暂停"):
            action()
    assert store.samples(version)[1] == 1 and len(store.version(version)["config"]) == 1


def test_delete_last_scene_and_frozen_export(service):
    store = service.store
    version = store.save_project("last", [scene("a")])["version_id"]
    add(store, version, "a")
    frozen = freeze_dataset(store, version, approve_pending=True)
    result = store.delete_scenario(version, "a")
    assert store.version(result["version_id"])["config"] == []
    with pytest.raises(IntentError, match="场景"):
        service.start(result["version_id"])
    with pytest.raises(IntentError, match="场景"):
        freeze_dataset(store, result["version_id"])
    assert (store.root / "datasets" / frozen["id"] / "train.json").is_file()
