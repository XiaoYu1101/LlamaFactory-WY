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

"""Check real WebUI parameter/monitor code with fake hardware; never start a trainer."""

import ast
import json
import os
import sys
from pathlib import Path
from subprocess import TimeoutExpired
from types import ModuleType, SimpleNamespace

import gradio as gr
import pytest

from llamafactory.intent.export import freeze_dataset
from llamafactory.intent.models import IntentError
from llamafactory.intent.resources import exclusive_resource
from llamafactory.intent.storage import Store
from llamafactory.intent.training import attach_training, training_outputs
from llamafactory.webui.locales import ALERTS
from llamafactory.webui.manager import Manager


ROOT = Path(__file__).resolve().parents[2] / "src/llamafactory"


def definitions(path, names, namespace):
    """Load the source definitions without importing GPU libraries or duplicating their bodies."""
    source = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [x for x in source.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(ROOT / path), "exec"
        ),
        namespace,
    )


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    constants = ModuleType("llamafactory.extras.constants")
    source = ast.parse((ROOT / "extras/constants.py").read_text(encoding="utf-8"))
    assignment = next(
        x
        for x in source.body
        if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAINING_STAGES" for t in x.targets)
    )
    constants.TRAINING_STAGES = ast.literal_eval(assignment.value)
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def noop(*args):
        return None

    def no_training(*args, **kwargs):
        raise AssertionError("A real training process must never be launched in these checks")

    namespace = dict(
        gr=gr,
        os=os,
        json=json,
        ALERTS=ALERTS,
        TRAINING_STAGES=constants.TRAINING_STAGES,
        PEFT_METHODS={"lora"},
        MULTIMODAL_SUPPORTED_MODELS=set(),
        DEFAULT_DATA_DIR="data",
        DEFAULT_SAVE_DIR=str(tmp_path / "saves"),
        DEFAULT_CACHE_DIR=str(tmp_path / "cache"),
        load_config=lambda: {},
        is_torch_npu_available=lambda: False,
        is_accelerator_available=lambda: False,
        exclusive_resource=exclusive_resource,
        torch_gc=noop,
        Popen=no_training,
        TimeoutExpired=TimeoutExpired,
        get_device_count=lambda: 0,
        SchedulerType=[SimpleNamespace(value="cosine")],
        create_preview_box=lambda *args: {},
        logger=SimpleNamespace(warning_rank0=noop),
    )
    for name in ("list_checkpoints", "list_config_paths", "list_datasets", "list_output_dirs"):
        namespace[name] = noop
    definitions("webui/common.py", {"get_save_dir", "load_eval_results"}, namespace)
    definitions("webui/control.py", {"change_stage"}, namespace)
    definitions("webui/runner.py", {"Runner", "_parse_seed"}, namespace)
    definitions("webui/components/train.py", {"create_train_tab"}, namespace)
    definitions("webui/components/eval.py", {"create_eval_tab"}, namespace)
    manager = Manager()
    engine = SimpleNamespace(manager=manager)
    engine.runner = namespace["Runner"](manager)
    with gr.Blocks() as ui:
        manager.add_elems(
            "top",
            {
                "lang": gr.Dropdown(choices=["en"], value="en"),
                "model_name": gr.Dropdown(choices=["Custom"], value="Custom"),
                "model_path": gr.Textbox(value="/models/test"),
                "finetuning_type": gr.Dropdown(choices=["full", "lora"], value="full"),
                "checkpoint_path": gr.Dropdown(choices=["old"], value="old"),
                "quantization_bit": gr.Dropdown(choices=["none"], value="none"),
                "quantization_method": gr.Dropdown(choices=["bnb"], value="bnb"),
                "template": gr.Dropdown(choices=["qwen"], value="qwen"),
                "rope_scaling": gr.Dropdown(choices=["none"], value="none"),
                "booster": gr.Dropdown(choices=["auto"], value="auto"),
            },
        )
        manager.add_elems("train", namespace["create_train_tab"](engine))
        manager.add_elems("eval", namespace["create_eval_tab"](engine))
    store = Store(tmp_path / "intent")
    version = store.save_project(
        "bridge",
        [{"label": "refund", "name": "退款", "description": "退款进度", "examples": ["我的退款到哪了"], "target": 1}],
    )["version_id"]
    sample = store.samples(version)[0][0]
    store.review(version, [{"id": sample["id"], "revision": sample["revision"]}], "approved")
    export = freeze_dataset(store, version)
    return engine, ui, store, export, namespace


def test_intent_preset_matches_real_runner_and_visible_controls(bridge):
    engine, ui, store, export, namespace = bridge
    captured = []

    def capture(data):
        captured.append(engine.runner._parse_train_args(data))
        yield {engine.manager.get_elem_by_id("train.output_box"): "checked without launching"}

    engine.runner.run_train = capture
    data = {elem: elem.value for elem in engine.runner.train_input_elems}
    data[engine.manager.get_elem_by_id("train.extra_args")] = '{"learning_rate":999}'
    updates, result = list(attach_training(engine, store, export["id"], data))
    args = captured[0]
    assert args["stage"] == "sft" and args["finetuning_type"] == "lora"
    assert args["dataset"] == "intent_train" and args["dataset_dir"].endswith(export["id"])
    assert args["model_name_or_path"] == "/models/test" and args["template"] == "qwen"
    assert args["learning_rate"] == 5e-5 and args["num_train_epochs"] == 3
    assert "adapter_name_or_path" not in args and args["report_to"] == "none"
    assert not args["fp16"] and not args["bf16"]
    assert set(updates) <= set(training_outputs(engine))
    assert set(result) <= set(training_outputs(engine))
    for name in ("training_stage", "report_to", "compute_type"):
        element = engine.manager.get_elem_by_id("train." + name)
        element.postprocess(updates[element]["value"])
    assert updates[engine.manager.get_elem_by_id("top.checkpoint_path")]["multiselect"]
    assert args["output_dir"].endswith(updates[engine.manager.get_elem_by_id("train.output_dir")]["value"])
    change = next(fn for fn in ui.fns.values() if fn.fn is namespace["change_stage"])
    assert change.targets[0][1] == "input"


def test_invalid_model_does_not_publish_training_link(bridge):
    engine, _, store, export, _ = bridge
    data = {elem: elem.value for elem in engine.runner.train_input_elems}
    data[engine.manager.get_elem_by_id("top.model_path")] = ""
    with pytest.raises(IntentError, match="模型"):
        list(attach_training(engine, store, export["id"], data))
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM training_links").fetchone()[0] == 0


@pytest.mark.parametrize("outcome", ["completed", "aborted", "missing", "invalid"])
def test_evaluation_monitor_results_and_cleanup_without_model(bridge, outcome):
    engine, _, _, _, namespace = bridge
    runner, manager = engine.runner, engine.manager
    runner.do_train = False
    runner.running_data = {elem: elem.value for elem in manager.get_elem_list() if hasattr(elem, "value")}
    for key, value in {
        "top.finetuning_type": "lora",
        "top.checkpoint_path": ["intent_adapter"],
        "eval.dataset": ["independent_eval"],
        "eval.output_dir": "eval_run",
    }.items():
        runner.running_data[manager.get_elem_by_id(key)] = value
    args = runner._parse_eval_args(runner.running_data)
    assert args["adapter_name_or_path"].endswith("intent_adapter")
    assert args["eval_dataset"] == "independent_eval" and args["do_predict"]
    output = Path(args["output_dir"])
    output.mkdir(parents=True)
    if outcome in {"completed", "invalid"}:
        (output / "all_results.json").write_text(
            json.dumps({"predict_loss": 0.25}) if outcome == "completed" else "broken", encoding="utf-8"
        )

    class FinishedProcess:
        returncode = 1 if outcome == "aborted" else 0

        def communicate(self, timeout):
            runner.aborted = outcome == "aborted"
            return None, None

    runner.trainer = FinishedProcess()
    namespace["get_trainer_info"] = lambda *args: ("progress log", gr.Slider(value=50), {})
    frames = list(runner.monitor())
    text = frames[-1][manager.get_elem_by_id("eval.output_box")]
    if outcome == "completed":
        assert "0.25" in text
    elif outcome == "aborted":
        assert text.startswith(ALERTS["info_aborted"]["en"])
    else:
        assert "missing or invalid" in text
    assert not runner.running and runner.trainer is None and runner.running_data is None


def test_training_monitor_keeps_original_progress_and_loss_outputs(bridge):
    engine, _, _, _, namespace = bridge
    runner, manager = engine.runner, engine.manager
    runner.running_data = {elem: elem.value for elem in runner.train_input_elems}
    runner.running_data[manager.get_elem_by_id("train.output_dir")] = "checked_run"
    runner.trainer = SimpleNamespace(returncode=0, communicate=lambda timeout: (None, None))
    loss = gr.Plot()
    namespace["get_trainer_info"] = lambda *args: ("training log", gr.Slider(value=75), {"loss_viewer": loss})
    frames = list(runner.monitor())
    assert frames[0][manager.get_elem_by_id("train.progress_bar")].value == 75
    assert frames[0][manager.get_elem_by_id("train.loss_viewer")] is loss
    assert "training log" in frames[-1][manager.get_elem_by_id("train.output_box")]
    assert not runner.running and runner.trainer is None


def test_prepare_dataset_updates_native_training_controls_without_launch(bridge, monkeypatch):
    from llamafactory.intent import ui as intent_ui

    engine, ui, store, export, _ = bridge
    monkeypatch.setattr(intent_ui, "get_service", lambda: SimpleNamespace(store=store))
    with ui:
        with gr.Tabs() as tabs:
            engine.intent_tabs = tabs
            with gr.Tab("Train", id="train"):
                gr.Markdown("native controls already registered")
            with gr.Tab("intent", id="intent"):
                intent_ui.create_ant_page(engine)
    handler = next(fn.fn for fn in ui.fns.values() if fn.fn and fn.fn.__name__ == "prepare_training")
    updates = handler(export["id"])
    assert updates[tabs]["selected"] == "train"
    assert updates[engine.manager.get_elem_by_id("train.dataset")]["value"] == ["intent_train"]
    assert updates[engine.manager.get_elem_by_id("train.dataset_dir")]["value"].endswith(export["id"])
    assert updates[engine.manager.get_elem_by_id("top.checkpoint_path")]["multiselect"]
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM training_links").fetchone()[0] == 0
    with pytest.raises(gr.Error):
        handler(None)


def test_saved_lora_config_reaches_original_runner(bridge, monkeypatch):
    from llamafactory.intent import ui as intent_ui
    from llamafactory.intent.training_config import LoraConfig, save_plan

    engine, ui, store, export, _ = bridge
    plan = save_plan(
        store,
        export["id"],
        LoraConfig(
            model_name_or_path="/target/model",
            template="qwen",
            lora_rank=32,
            lora_alpha=48,
            lora_dropout=0.1,
            lora_target="q_proj,v_proj",
            learning_rate=0.0002,
            num_train_epochs=2,
            gradient_accumulation_steps=16,
            cutoff_len=4096,
            precision="bf16",
        ),
    )
    monkeypatch.setattr(intent_ui, "get_service", lambda: SimpleNamespace(store=store))
    with ui:
        with gr.Tabs() as tabs:
            engine.intent_tabs = tabs
            with gr.Tab("intent", id="intent"):
                intent_ui.create_ant_page(engine)
    handler = next(fn.fn for fn in ui.fns.values() if fn.fn and fn.fn.__name__ == "prepare_saved_plan")
    updates = handler(plan["id"], "Custom")
    manager = engine.manager
    values = {elem: elem.value for elem in engine.runner.train_input_elems}
    values.update({elem: update["value"] for elem, update in updates.items() if "value" in update})
    args = engine.runner._parse_train_args(values)
    assert args["model_name_or_path"] == "/target/model" and args["finetuning_type"] == "lora"
    assert args["lora_rank"] == 32 and args["lora_alpha"] == 48 and args["lora_dropout"] == 0.1
    assert args["lora_target"] == "q_proj,v_proj" and args["learning_rate"] == 0.0002
    assert args["num_train_epochs"] == 2 and args["gradient_accumulation_steps"] == 16
    assert args["cutoff_len"] == 4096 and args["bf16"] and not args["fp16"]
    assert args["dataset"] == "intent_train" and "adapter_name_or_path" not in args
    assert values[manager.get_elem_by_id("top.checkpoint_path")] == []
    assert updates[tabs]["selected"] == "train" and not engine.runner.running
    engine.runner.running = True
    with pytest.raises(gr.Error, match="运行"):
        handler(plan["id"], "Custom")


def test_evaluation_data_transfer_preserves_selected_model_and_adapter(bridge, monkeypatch):
    from llamafactory.intent import ui as intent_ui

    engine, ui, store, export, _ = bridge
    monkeypatch.setattr(intent_ui, "get_service", lambda: SimpleNamespace(store=store))
    with ui:
        with gr.Tabs() as tabs:
            engine.intent_tabs = tabs
            with gr.Tab("intent", id="intent"):
                intent_ui.create_ant_page(engine)
    handler = next(fn.fn for fn in ui.fns.values() if fn.fn and fn.fn.__name__ == "prepare_evaluation")
    updates = handler(export["id"])
    assert not any(elem in updates for elem in engine.manager.get_base_elems())
    values = {elem: elem.value for elem in engine.manager.get_elem_list() if hasattr(elem, "value")}
    values[engine.manager.get_elem_by_id("top.finetuning_type")] = "lora"
    values[engine.manager.get_elem_by_id("top.checkpoint_path")] = ["trained_adapter"]
    values.update({elem: update["value"] for elem, update in updates.items() if "value" in update})
    args = engine.runner._parse_eval_args(values)
    assert args["eval_dataset"] == "intent_train"
    assert args["dataset_dir"].endswith(export["id"])
    assert args["model_name_or_path"] == "/models/test"
    assert "trained_adapter" in args["adapter_name_or_path"]
    assert args["do_predict"] and args["max_samples"] == export["count"]
    assert updates[tabs]["selected"] == "eval"
    with pytest.raises(gr.Error):
        handler(None)
    engine.runner.running = True
    with pytest.raises(gr.Error, match="运行"):
        handler(export["id"])
    train_handler = next(fn.fn for fn in ui.fns.values() if fn.fn and fn.fn.__name__ == "prepare_training")
    with pytest.raises(gr.Error, match="运行"):
        train_handler(export["id"])


def test_native_dataset_registry_reads_utf8_columns(tmp_path):
    import builtins
    from dataclasses import dataclass

    folder = tmp_path / "中文数据"
    folder.mkdir()
    (folder / "dataset_info.json").write_text(
        json.dumps(
            {"test": {"file_name": "train.json", "columns": {"prompt": "问题", "response": "答案", "system": "系统"}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def legacy_locale_open(file, *args, **kwargs):
        kwargs.setdefault("encoding", "ascii")
        return builtins.open(file, *args, **kwargs)

    namespace = dict(dataclass=dataclass, os=os, json=json, DATA_CONFIG="dataset_info.json", open=legacy_locale_open)
    definitions("data/parser.py", {"DatasetAttr", "get_dataset_list"}, namespace)
    result = namespace["get_dataset_list"](["test"], str(folder))[0]
    assert (result.prompt, result.response, result.system) == ("问题", "答案", "系统")
