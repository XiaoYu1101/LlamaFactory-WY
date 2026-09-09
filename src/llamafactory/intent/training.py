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

from .export import get_export
from .models import IntentError, dump, now, uid


def training_values(export: dict, stages: dict, compute_type: str) -> dict:
    """Map a frozen dataset to existing WebUI inputs; all training remains in the original Runner."""
    stage_name = next(name for name, value in stages.items() if value == "sft")
    return {
        "top.finetuning_type": "lora",
        "top.checkpoint_path": [],
        "train.training_stage": stage_name,
        "train.dataset_dir": export["path"],
        "train.dataset": ["wy_intent_train"],
        "train.output_dir": f"wy_{export['id'][:8]}_{uid()[:8]}",
        "train.learning_rate": "5e-5",
        "train.num_train_epochs": "3.0",
        "train.batch_size": 1,
        "train.gradient_accumulation_steps": 8,
        "train.cutoff_len": 2048,
        "train.lora_rank": 8,
        "train.val_size": 0,
        "train.enable_thinking": False,
        "train.create_new_adapter": True,
        "train.compute_type": compute_type,
        "train.report_to": ["none"],
        "train.extra_args": '{"optim":"adamw_torch"}',
    }


def attach_training(engine, store, export_id, data):
    """Yield original training outputs with a saved dataset-to-training association."""
    import torch

    from ..extras.constants import TRAINING_STAGES

    export = get_export(store, export_id)
    get = lambda name: data.get(engine.manager.get_elem_by_id(name))
    if not get("top.model_path") or not get("top.model_name"):
        raise IntentError("请先在顶部选择训练模型和路径。API 仅生成数据，不替代本地训练模型。")
    values = {elem: elem.value for elem in engine.runner.train_input_elems}
    values.update({elem: data[elem] for elem in engine.manager.get_base_elems()})
    precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
    if torch.cuda.is_available() and precision != "bf16":
        precision = "fp16"
    changes = training_values(export, TRAINING_STAGES, precision)
    for name, value in changes.items():
        values[engine.manager.get_elem_by_id(name)] = value
    if engine.runner.running:
        raise IntentError("已有训练或评测任务正在运行。")
    # The original Runner acquires the shared device gate and unloads Chat before starting.
    with store.connect(True) as db:
        db.execute("INSERT INTO training_links VALUES (?,?,?,?)", (uid(), export_id, dump(changes), now()))
    yield from engine.runner.run_train(values)
