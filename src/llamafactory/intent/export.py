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

from .models import IntentError, classification_prompt, dump, now, uid
from .records import validate_record


def freeze_dataset(store, version_id: str) -> dict:
    """Freeze exactly the reviewed content in one transaction, then publish an immutable export."""
    export_id = uid()
    parent = store.root / "datasets"
    parent.mkdir(exist_ok=True)
    staging, destination = parent / f".tmp-{export_id}", parent / export_id
    with store.connect(True) as db:
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
                "SELECT id,label,instruction,text,output,source,revision,seed_ids FROM samples "
                "WHERE version_id=? AND deleted=0 AND status='approved' ORDER BY label,id",
                (version_id,),
            )
        ]
        missing = {x["label"] for x in intents} - {x["label"] for x in samples}
        if missing:
            raise IntentError("以下类别没有已通过样本：" + "、".join(sorted(missing)))
        prompt = classification_prompt([x for x in intents if x.get("format") != "alpaca"])
        scenarios = {x["label"]: x for x in intents}
        training = [
            validate_record(
                dict(instruction=x["instruction"], input=x["text"], output=x["output"]), scenarios[x["label"]]
            )
            if scenarios[x["label"]].get("format") == "alpaca"
            else {"instruction": prompt, "input": x["text"], "output": x["label"]}
            for x in samples
        ]
        contents = {
            "train.json": dump(training),
            "dataset_info.json": dump({"wy_intent_train": {"file_name": "train.json", "formatting": "alpaca"}}),
            "reviewed_samples.json": dump(samples),
        }
        hashes = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in contents.items()}
        manifest = {
            "id": export_id,
            "version_id": version_id,
            "project_id": version["project_id"],
            "created": now(),
            "count": len(samples),
            "labels": intents,
            "classification_prompt": prompt if any(x.get("format") != "alpaca" for x in intents) else None,
            "instructions": list(dict.fromkeys(x["instruction"] for x in training)),
            "reference_samples_included": False if all(x.get("format") == "alpaca" for x in intents) else None,
            "files": hashes,
            "evaluation": "未自动划分评测集；请另行准备独立测试数据。",
        }
        staging.mkdir()
        for name, content in {**contents, "manifest.json": dump(manifest)}.items():
            (staging / name).write_text(content, encoding="utf-8")
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
