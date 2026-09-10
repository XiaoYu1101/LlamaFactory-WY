"""Portable LoRA plans; importing this module never imports the training engine."""

import ast
import io
import json
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .export import get_export
from .models import IntentError, dump, now, uid


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_name_or_path: str = Field(min_length=1, max_length=1000)
    template: str = Field(min_length=1, max_length=100)
    lora_rank: int = Field(default=8, ge=1, le=256, strict=True)
    lora_alpha: int | None = Field(default=None, ge=1, le=2048, strict=True)
    lora_dropout: float = Field(default=0, ge=0, lt=1)
    lora_target: str = Field(default="all", min_length=1, max_length=1000, pattern=r"^[A-Za-z0-9_.,]+$")
    learning_rate: float = Field(default=5e-5, gt=0, le=1)
    num_train_epochs: float = Field(default=3, gt=0, le=1000)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=1024, strict=True)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=4096, strict=True)
    cutoff_len: int = Field(default=2048, ge=32, le=131072, strict=True)
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    val_size: float = Field(default=0, ge=0, lt=0.5)

    @field_validator("model_name_or_path", "template")
    @classmethod
    def clean(cls, value):
        value = value.strip()
        if not value or any(c in value for c in "\x00\r\n"):
            raise ValueError("请填写有效的模型路径或模板名称")
        return value


def templates():
    """Read the local template registry without loading transformers/torch."""
    tree = ast.parse((Path(__file__).parents[1] / "data" / "template.py").read_text(encoding="utf-8"))
    return sorted(
        {
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_template"
            for kw in node.keywords
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
        }
    )


def validate_plan(store, export_id, config):
    export = get_export(store, export_id)
    if not export["manifest"].get("training_ready") or not export["manifest"].get("training_datasets"):
        raise IntentError("数据尚未完成训练字段映射，请先配置映射并重新导出。")
    if config.template not in templates():
        raise IntentError("模板不在当前 LlamaFactory 注册列表中，请选择与基座模型匹配的模板。")
    if config.val_size and export["manifest"]["count"] * config.val_size < 1:
        raise IntentError("样本过少，验证比例至少需分出一条记录；也可设为 0。")
    return export


def save_plan(store, export_id, config):
    validate_plan(store, export_id, config)
    record = dict(id=uid(), export_id=export_id, config=config.model_dump(), created=now())
    with store.connect(True) as db:
        db.execute(
            "INSERT INTO training_configs VALUES (?,?,?,?)",
            (record["id"], export_id, dump(record["config"]), record["created"]),
        )
    return record


def get_plan(store, plan_id):
    with store.connect() as db:
        row = db.execute("SELECT * FROM training_configs WHERE id=?", (plan_id,)).fetchone()
    if not row:
        raise IntentError("训练配置不存在。")
    return {**dict(row), "config": json.loads(row["config"])}


def list_plans(store):
    with store.connect() as db:
        return [
            {**dict(row), "config": json.loads(row["config"])}
            for row in db.execute("SELECT * FROM training_configs ORDER BY created DESC,rowid DESC LIMIT 100")
        ]


def train_args(config, manifest, output_dir):
    args = config.model_dump(exclude={"precision"}, exclude_none=True)
    args.update(
        stage="sft",
        do_train=True,
        finetuning_type="lora",
        dataset_dir="data",
        dataset=",".join(manifest["training_datasets"]),
        output_dir=output_dir,
        logging_steps=10,
        save_strategy="epoch",
        plot_loss=True,
        report_to="none",
        preprocessing_num_workers=1,
        dataloader_num_workers=0,
        optim="adamw_torch",
        overwrite_output_dir=False,
    )
    if config.precision != "auto":
        args.update(bf16=config.precision == "bf16", fp16=config.precision == "fp16")
    if config.val_size:
        args.update(eval_strategy="epoch", per_device_eval_batch_size=1)
    return args


# The launcher stays standard-library-only until the user actually executes it.
LAUNCHER = r'''"""Run from any directory: python run_training.py [--model /path/to/model] [--dry-run]."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Override the model directory or model ID on this machine")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without importing torch or starting training")
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / "train_config.json").read_text(encoding="utf-8"))
    if options.model:
        config["model_name_or_path"] = options.model
    config["dataset_dir"] = str(root / config["dataset_dir"])
    config["output_dir"] = str(root / config["output_dir"])
    if options.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if "bf16" not in config and "fp16" not in config:
        import torch
        config["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        config["fp16"] = bool(torch.cuda.is_available() and not config["bf16"])
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty. Change output_dir in train_config.json for a new run.")
    resolved = root / "train_resolved.json"
    resolved.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "llamafactory.cli", "train", str(resolved)], cwd=root, check=True)


if __name__ == "__main__":
    main()
'''


def plan_bundle(store, plan_id):
    plan = get_plan(store, plan_id)
    config = LoraConfig(**plan["config"])
    export = validate_plan(store, plan["export_id"], config)
    args = train_args(config, export["manifest"], f"output/lora-{plan_id[:8]}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in export["manifest"]["files"]:
            if name == "dataset_info.json" or name.startswith("train") and name.endswith(".json"):
                archive.write(Path(export["path"]) / name, "data/" + name)
        archive.writestr("train_config.json", json.dumps(args, ensure_ascii=False, indent=2))
        archive.writestr("run_training.py", LAUNCHER)
        archive.writestr("training_plan.json", dump(plan))
        archive.writestr(
            "README.txt",
            "在训练机器安装本项目及匹配硬件的 PyTorch：python -m pip install -e .\n"
            "解压后先检查配置：python run_training.py --dry-run\n"
            "开始训练：python run_training.py --model /实际模型路径\n"
            "模型也可填写仓库 ID。API 生成服务地址不能作为本地训练模型。\n"
            "模板必须匹配模型；可编辑 train_config.json。自动精度在启动时按 CUDA 能力选择 bf16/fp16，否则 fp32。\n"
            "输出位于 output/，包含 LoRA 适配器、检查点、训练日志与损失图，使用时仍需要基座模型。\n"
            "验证比例为 0 时不划分验证集。验证 loss 不等同于分类准确率或 F1，最终测试请另备独立数据。\n"
            "重复运行不会覆盖已有输出，请修改 output_dir 后再开始。\n",
        )
    return buffer.getvalue()
