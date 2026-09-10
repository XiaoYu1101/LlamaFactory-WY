import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llamafactory.intent.api import create_app
from llamafactory.intent.export import freeze_dataset
from llamafactory.intent.service import IntentService
from llamafactory.intent.training_config import LoraConfig, get_plan, plan_bundle, save_plan, templates


@pytest.fixture
def prepared(tmp_path):
    service = IntentService(tmp_path / "workspace")
    scene = dict(
        label="task",
        name="训练",
        description="扩写",
        format="json",
        target=20,
        examples=[dict(system="保持系统字段", instruction="回答", input=f"种子{i}", output="答案") for i in range(4)],
    )
    version = service.store.save_project("迁移", [scene])["version_id"]
    for i in range(20):
        service.store.add_sample(
            version, "task", record=dict(system="保持系统字段", instruction="回答", input=f"新问题{i}", output="答案")
        )
    export = freeze_dataset(service.store, version, approve_pending=True)
    try:
        yield service, export
    finally:
        service.close()


def test_portable_bundle_api_and_persistence(prepared, tmp_path):
    service, export = prepared
    with TestClient(create_app(service=service)) as client:
        options = client.get("/intent-api/training/options").json()
        assert "qwen" in options["templates"] and "qwen3_nothink" in options["templates"]
        response = client.post(
            f"/intent-api/exports/{export['id']}/training-config",
            json=dict(model_name_or_path="/new machine/model", template="qwen", lora_rank=16, val_size=0.1),
        )
        assert response.status_code == 200, response.text
        plan = response.json()
        assert client.get("/intent-api/training/configs").json()[0] == plan
        archive = client.get(f"/intent-api/training/configs/{plan['id']}/download")
        assert archive.status_code == 200
    assert get_plan(service.store, plan["id"]) == plan
    destination = tmp_path / "other machine" / "bundle"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
        package.extractall(destination)
    config = json.loads((destination / "train_config.json").read_text(encoding="utf-8"))
    assert config["dataset_dir"] == "data" and not Path(config["output_dir"]).is_absolute()
    assert config["lora_rank"] == 16 and "lora_alpha" not in config
    assert config["eval_strategy"] == "epoch" and config["val_size"] == 0.1
    assert "bf16" not in config and "quantization_bit" not in config
    registry = json.loads((destination / "data/dataset_info.json").read_text(encoding="utf-8"))
    for item in registry.values():
        rows = json.loads((destination / "data" / item["file_name"]).read_text(encoding="utf-8"))
        assert len(rows) == 20 and rows[0]["system"] == "保持系统字段"
    result = subprocess.run(
        [sys.executable, str(destination / "run_training.py"), "--dry-run", "--model", "/replacement/model"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = json.loads(result.stdout)
    assert resolved["dataset_dir"] == str(destination / "data")
    assert resolved["model_name_or_path"] == "/replacement/model"
    assert not (destination / "train_resolved.json").exists()
    assert not (destination / "output").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"lora_rank": 0},
        {"lora_rank": 1.5},
        {"learning_rate": 0},
        {"lora_dropout": 1},
        {"precision": "int4"},
        {"val_size": 0.5},
        {"model_name_or_path": " "},
        {"unexpected": True},
        {"lora_target": "q_proj;bad"},
    ],
)
def test_invalid_training_parameters_rejected(prepared, changes):
    service, export = prepared
    with TestClient(create_app(service=service)) as client:
        response = client.post(
            f"/intent-api/exports/{export['id']}/training-config",
            json={"model_name_or_path": "model", "template": "qwen", **changes},
        )
        assert response.status_code == 422


def test_template_and_export_integrity_guard(prepared):
    service, export = prepared
    from llamafactory.intent.models import IntentError

    with pytest.raises(IntentError, match="模板"):
        save_plan(service.store, export["id"], LoraConfig(model_name_or_path="model", template="not_registered"))
    plan = save_plan(
        service.store, export["id"], LoraConfig(model_name_or_path="model", template="qwen", precision="fp32")
    )
    with zipfile.ZipFile(io.BytesIO(plan_bundle(service.store, plan["id"]))) as package:
        config = json.loads(package.read("train_config.json"))
        assert config["bf16"] is False and config["fp16"] is False
        assert "eval_strategy" not in config
    (service.store.root / "datasets" / export["id"] / "train.json").write_text("[]")
    with pytest.raises(IntentError, match="修改"):
        plan_bundle(service.store, plan["id"])


def test_templates_read_without_training_dependencies():
    assert "qwen3_nothink" in templates()


@pytest.mark.parametrize("bf16_supported, expected", [(True, (True, False)), (False, (False, True))])
def test_launcher_selects_precision_and_calls_original_cli_only_when_executed(
    prepared, tmp_path, monkeypatch, bf16_supported, expected
):
    import runpy
    from types import SimpleNamespace

    service, export = prepared
    plan = save_plan(service.store, export["id"], LoraConfig(model_name_or_path="model", template="qwen"))
    destination = tmp_path / "launcher"
    with zipfile.ZipFile(io.BytesIO(plan_bundle(service.store, plan["id"]))) as package:
        package.extractall(destination)
    namespace = runpy.run_path(str(destination / "run_training.py"), run_name="isolated_launcher_test")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: bf16_supported)),
    )
    monkeypatch.setattr(sys, "argv", [str(destination / "run_training.py")])
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace["main"]()
    resolved = json.loads((destination / "train_resolved.json").read_text(encoding="utf-8"))
    assert (resolved["bf16"], resolved["fp16"]) == expected
    assert calls[0][0][0][:4] == [sys.executable, "-m", "llamafactory.cli", "train"]
    assert calls[0][1]["cwd"] == destination and calls[0][1]["check"]
    output = Path(resolved["output_dir"])
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        namespace["main"]()
    assert len(calls) == 1
