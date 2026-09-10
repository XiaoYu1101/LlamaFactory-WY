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

import hashlib
import json
import os
import zipfile
from pathlib import Path

from .dataset_mapping import dataset_mapping
from .models import IntentError, classification_prompt, dump, now, uid
from .records import is_json_scenario, stored_record, validate_record


def freeze_dataset(store, version_id: str, approve_pending=False) -> dict:
    """Freeze exactly the reviewed content in one transaction, then publish an immutable export."""
    export_id = uid()
    parent = store.root / "datasets"
    parent.mkdir(exist_ok=True)
    staging, destination = parent / f".tmp-{export_id}", parent / export_id
    with store.connect(True) as db:
        approved_now = store.approve_pending(db, version_id) if approve_pending else 0
        version = store.require(db, "versions", version_id)
        intents = json.loads(version["config"])
        if db.execute(
            "SELECT 1 FROM samples WHERE version_id=? AND deleted=0 AND status='pending' LIMIT 1", (version_id,)
        ).fetchone():
            raise IntentError("还有待审核的样本，请先通过、驳回或删除后再完成审核。")
        if store.conflicts(db, version_id):
            raise IntentError("存在相同文本对应多个类别的冲突，请修改或驳回后重试。")
        if db.execute(
            "SELECT 1 FROM jobs WHERE version_id=? AND status IN ('queued','running','pausing','cancelling')",
            (version_id,),
        ).fetchone():
            raise IntentError("生成任务尚未停止，请先暂停或等待完成。")
        samples = [
            dict(x)
            for x in db.execute(
                "SELECT id,label,instruction,text,output,record_json,source,revision,seed_ids FROM samples "
                "WHERE version_id=? AND deleted=0 AND status='approved' ORDER BY label,id",
                (version_id,),
            )
        ]
        missing = {x["label"] for x in intents} - {x["label"] for x in samples}
        if missing:
            raise IntentError("以下类别没有已通过样本：" + "、".join(sorted(missing)))
        prompt = classification_prompt([x for x in intents if not is_json_scenario(x)])
        scenarios = {x["label"]: x for x in intents}
        training, mappings, grouped = [], [], {}
        for row in samples:
            scenario = scenarios[row["label"]]
            record = (
                validate_record(stored_record(row), scenario)
                if is_json_scenario(scenario)
                else {"instruction": prompt, "input": row["text"], "output": row["label"]}
            )
            training.append(record)
            grouped.setdefault(row["label"], []).append(record)
            mappings.append(dataset_mapping(record, scenario.get("training_mapping")))
        training_ready = all(x is not None for x in mappings)
        contents = {
            "train.json": json.dumps(training, ensure_ascii=False, indent=2),
            "reviewed_samples.json": dump([{**x, "record": stored_record(x)} for x in samples]),
        }
        registry = {}
        if training_ready and all(x == mappings[0] for x in mappings):
            registry["wy_intent_train"] = {"file_name": "train.json", **mappings[0]}
        elif training_ready:
            # Different scenarios can use different original schemas without flattening them.
            for index, (label, rows) in enumerate(grouped.items(), 1):
                local_mappings = [dataset_mapping(x, scenarios[label].get("training_mapping")) for x in rows]
                if not all(x == local_mappings[0] for x in local_mappings):
                    training_ready = False
                    break
                file_name = f"train_{index}.json"
                contents[file_name] = json.dumps(rows, ensure_ascii=False, indent=2)
                registry[f"wy_intent_train_{index}"] = {"file_name": file_name, **local_mappings[0]}
        if not training_ready:
            registry = {}
        contents["dataset_info.json"] = dump(registry)
        hashes = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in contents.items()}
        manifest = {
            "id": export_id,
            "version_id": version_id,
            "project_id": version["project_id"],
            "created": now(),
            "count": len(samples),
            "approved_now": approved_now,
            "labels": intents,
            "classification_prompt": prompt if any(not is_json_scenario(x) for x in intents) else None,
            "instructions": list(
                dict.fromkeys(x["instruction"] for x in training if isinstance(x.get("instruction"), str))
            ),
            "training_ready": training_ready,
            "training_datasets": list(registry),
            "training_note": (
                "训练字段已映射，system 列和对话角色按原数据登记。"
                if training_ready
                else "部分记录未识别为可训练格式或缺少一致的字段映射。JSON 已完整保留；请在场景中配置训练字段映射，或自行按 LlamaFactory 格式准备训练配置。"
            ),
            "reference_samples_included": False if all(is_json_scenario(x) for x in intents) else None,
            "files": hashes,
            "evaluation": "未自动划分评测集；请另行准备独立测试数据。",
        }
        staging.mkdir()
        for name, content in {**contents, "manifest.json": dump(manifest)}.items():
            (staging / name).write_bytes(content.encode("utf-8"))
        with zipfile.ZipFile(staging / "dataset.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in [*contents, "manifest.json"]:
                archive.write(staging / name, arcname=name)
        os.replace(staging, destination)
        db.execute(
            "INSERT INTO exports VALUES (?,?,?,?,?)", (export_id, version_id, str(destination), dump(manifest), now())
        )
    return manifest


def get_export(store, export_id: str) -> dict:
    with store.connect() as db:
        row = store.require(db, "exports", export_id)
    path = Path(row["path"]).resolve()
    if not path.is_relative_to(store.root / "datasets"):
        raise IntentError("数据版本路径不在工作区中。")
    manifest = json.loads(row["manifest"])
    for name, expected in manifest["files"].items():
        file = path / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise IntentError("已冻结的数据文件丢失或被修改，请重新导出。")
    row["manifest"] = manifest
    return row
