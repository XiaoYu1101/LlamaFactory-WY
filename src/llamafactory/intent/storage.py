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
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import IntentError, clean_text, dump, fingerprint, integer, now, uid, validate_intents
from .records import is_json_scenario, record_fingerprint, record_projection, stored_record, validate_record


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, current_version TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), config TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(id), label TEXT NOT NULL,
 original TEXT NOT NULL, text TEXT NOT NULL, fingerprint TEXT NOT NULL, source TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
 deleted INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 1,
 job_id TEXT, batch_id TEXT, seed_ids TEXT NOT NULL DEFAULT '[]', created TEXT NOT NULL,
 UNIQUE(version_id,label,fingerprint)
);
CREATE INDEX IF NOT EXISTS samples_query ON samples(version_id,deleted,status,label,created);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(id), status TEXT NOT NULL,
 targets TEXT NOT NULL, provider TEXT NOT NULL, accepted INTEGER NOT NULL DEFAULT 0,
 produced INTEGER NOT NULL DEFAULT 0, duplicates INTEGER NOT NULL DEFAULT 0, invalid INTEGER NOT NULL DEFAULT 0,
 attempts INTEGER NOT NULL DEFAULT 0, empty_batches INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL, error TEXT NOT NULL DEFAULT '', created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), label TEXT NOT NULL,
 accepted INTEGER NOT NULL, produced INTEGER NOT NULL, duplicates INTEGER NOT NULL, invalid INTEGER NOT NULL,
 seed_ids TEXT NOT NULL, error TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_events (
 id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id), action TEXT NOT NULL,
 before_json TEXT NOT NULL, after_json TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exports (
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(id), path TEXT NOT NULL,
 manifest TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS label_tasks (
 id TEXT PRIMARY KEY, status TEXT NOT NULL, input_hash TEXT NOT NULL, request_hash TEXT NOT NULL,
 input_json TEXT NOT NULL, snapshot_json TEXT NOT NULL, result_json TEXT,
 provider_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '', attempts INTEGER NOT NULL DEFAULT 1,
 created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS label_tasks_status ON label_tasks(status,created);
CREATE TABLE IF NOT EXISTS training_configs (
 id TEXT PRIMARY KEY, export_id TEXT NOT NULL REFERENCES exports(id), config TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_links (
 id TEXT PRIMARY KEY, export_id TEXT NOT NULL REFERENCES exports(id), config TEXT NOT NULL, created TEXT NOT NULL
);
"""
ACTIVE = ("queued", "running", "pausing", "cancelling")


class Store:
    """Transactional, versioned storage shared by background tasks and UI requests."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "intent.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            # Additive migration keeps existing workspaces, review history and exports intact.
            columns = {row[1] for row in db.execute("PRAGMA table_info(samples)")}
            for column in (
                "instruction",
                "output",
                "original_instruction",
                "original_output",
                "record_json",
                "original_record_json",
            ):
                if column not in columns:
                    db.execute(f"ALTER TABLE samples ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            # Upgrade older three-field records without discarding their review state.
            for row in db.execute("SELECT * FROM samples WHERE record_json='' AND instruction<>''").fetchall():
                row = dict(row)
                record, original = stored_record(row), stored_record(row, original=True)
                db.execute(
                    "UPDATE samples SET record_json=?,original_record_json=?,fingerprint=? WHERE id=?",
                    (
                        json.dumps(record, ensure_ascii=False),
                        json.dumps(original, ensure_ascii=False),
                        record_fingerprint(record),
                        row["id"],
                    ),
                )

    @contextmanager
    def connect(self, write: bool = False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def require(db, table: str, item_id: str):
        # Table names are fixed internal identifiers, never user input.
        if table not in {"projects", "versions", "jobs", "samples", "exports", "label_tasks"}:
            raise ValueError("Unknown internal table")
        row = db.execute(f"SELECT * FROM {table} WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise IntentError("记录不存在，请刷新页面后重试。")
        return dict(row)

    def projects(self) -> list[dict]:
        with self.connect() as db:
            return [dict(x) for x in db.execute("SELECT * FROM projects ORDER BY created DESC, rowid DESC")]

    def version(self, version_id: str) -> dict:
        with self.connect() as db:
            row = self.require(db, "versions", version_id)
        row["config"] = json.loads(row["config"])
        return row

    def project(self, project_id: str) -> dict:
        with self.connect() as db:
            row = self.require(db, "projects", project_id)
        row["version"] = self.version(row["current_version"])
        return row

    def save_project(
        self, name: str, intents: list[dict], project_id: str | None = None, base_version_id: str | None = None
    ) -> dict:
        name, intents = clean_text(name, "项目名称", 100), validate_intents(intents)
        config = dump(intents)
        from .label_inference import verify_label_task

        with self.connect(True) as db:
            for intent in intents:
                verify_label_task(db, intent)
            if project_id:
                project = self.require(db, "projects", project_id)
                if base_version_id is not None and project["current_version"] != base_version_id:
                    raise IntentError(
                        "项目已有更新的配置，不能用旧任务草稿覆盖。请先切换到最新项目配置，再合并本次修改。"
                    )
                previous = self.require(db, "versions", project["current_version"])
                if previous["config"] == config:
                    db.execute("UPDATE projects SET name=? WHERE id=?", (name, project_id))
                    return {"id": project_id, "version_id": previous["id"]}
                busy = db.execute(
                    "SELECT 1 FROM jobs WHERE version_id=? AND status IN ('queued','running','pausing','cancelling')",
                    (previous["id"],),
                ).fetchone()
                if busy:
                    raise IntentError("请先暂停或停止生成，再修改类别配置。")
            else:
                project_id = uid()
            version_id = uid()
            if not db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone():
                db.execute("INSERT INTO projects VALUES (?,?,?,?)", (project_id, name, version_id, now()))
            db.execute("INSERT INTO versions VALUES (?,?,?,?)", (version_id, project_id, config, now()))
            db.execute("UPDATE projects SET name=?,current_version=? WHERE id=?", (name, version_id, project_id))
            for intent in intents:
                if is_json_scenario(intent):
                    continue  # Reference JSON guides generation; it is not training output.
                for seed in intent["examples"]:
                    self._insert_sample(db, version_id, intent["label"], seed, "user")
        return {"id": project_id, "version_id": version_id}

    @staticmethod
    def _guard_purge(db, version_ids):
        for version_id in version_ids:
            if db.execute(
                "SELECT 1 FROM jobs WHERE version_id=? AND status IN ('queued','running','pausing','cancelling')",
                (version_id,),
            ).fetchone():
                raise IntentError("请先暂停或停止生成，再删除场景或清空样本。")
        for version_id in version_ids:
            db.execute(
                "UPDATE jobs SET status='cancelled' WHERE version_id=? AND status IN ('paused','interrupted','failed')",
                (version_id,),
            )

    @staticmethod
    def _purge_samples(db, version_id, label=None):
        where, params = "version_id=?", [version_id]
        if label is not None:
            where += " AND label=?"
            params.append(label)
        db.execute(f"DELETE FROM review_events WHERE sample_id IN (SELECT id FROM samples WHERE {where})", params)
        return db.execute(f"DELETE FROM samples WHERE {where}", params).rowcount

    def clear_samples(self, version_id):
        with self.connect(True) as db:
            self.require(db, "versions", version_id)
            self._guard_purge(db, [version_id])
            count = self._purge_samples(db, version_id)
        return {"deleted": count}

    def delete_scenario(self, version_id, label):
        with self.connect(True) as db:
            current = self.require(db, "versions", version_id)
            project = self.require(db, "projects", current["project_id"])
            if project["current_version"] != version_id:
                raise IntentError("只能从最新项目配置删除场景，请刷新后重试。")
            config = json.loads(current["config"])
            if label not in {item["label"] for item in config}:
                raise IntentError("场景不存在，请刷新后重试。")
            versions = db.execute("SELECT * FROM versions WHERE project_id=?", (project["id"],)).fetchall()
            affected = [row for row in versions if label in {item["label"] for item in json.loads(row["config"])}]
            self._guard_purge(db, [row["id"] for row in affected])
            count = 0
            for row in affected:
                remaining = [item for item in json.loads(row["config"]) if item["label"] != label]
                db.execute("UPDATE versions SET config=? WHERE id=?", (dump(remaining), row["id"]))
                count += self._purge_samples(db, row["id"], label)
            # A new current version prevents stale browser drafts from resurrecting the removed scene.
            new_id = uid()
            remaining = [item for item in config if item["label"] != label]
            db.execute("INSERT INTO versions VALUES (?,?,?,?)", (new_id, project["id"], dump(remaining), now()))
            db.execute("UPDATE projects SET current_version=? WHERE id=?", (new_id, project["id"]))
            rows = db.execute("SELECT * FROM samples WHERE version_id=?", (version_id,)).fetchall()
            for row in rows:
                item = dict(row)
                item.update(id=uid(), version_id=new_id)
                columns = list(item)
                db.execute(
                    f"INSERT INTO samples ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    list(item.values()),
                )
                for event in db.execute("SELECT * FROM review_events WHERE sample_id=?", (row["id"],)).fetchall():
                    db.execute(
                        "INSERT INTO review_events VALUES (?,?,?,?,?,?)",
                        (
                            uid(),
                            item["id"],
                            event["action"],
                            event["before_json"],
                            event["after_json"],
                            event["created"],
                        ),
                    )
        return {"version_id": new_id, "config": remaining, "deleted": count}

    @staticmethod
    def _insert_sample(db, version_id, label, text, source, job_id=None, batch_id=None, seed_ids=None):
        record = validate_record(text) if isinstance(text, dict) else None
        instruction, text, output = record_projection(record) if record else ("", text, "")
        record_json = json.dumps(record, ensure_ascii=False) if record else ""
        sample_id = uid()
        cursor = db.execute(
            "INSERT OR IGNORE INTO samples "
            "(id,version_id,label,original,text,fingerprint,source,job_id,batch_id,seed_ids,created,instruction,output,original_instruction,original_output,record_json,original_record_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sample_id,
                version_id,
                label,
                text,
                text,
                record_fingerprint(record) if record else fingerprint(text),
                source,
                job_id,
                batch_id,
                dump(seed_ids or []),
                now(),
                instruction,
                output,
                instruction,
                output,
                record_json,
                record_json,
            ),
        )
        return sample_id if cursor.rowcount else None

    def add_sample(
        self, version_id: str, label: str, text: str = "", instruction=None, output=None, record=None
    ) -> str:
        with self.connect(True) as db:
            config = json.loads(self.require(db, "versions", version_id)["config"])
            if label not in {x["label"] for x in config}:
                raise IntentError("请选择该配置版本中的类别。")
            scenario = next(x for x in config if x["label"] == label)
            if is_json_scenario(scenario):
                text = validate_record(
                    record if record is not None else dict(instruction=instruction, input=text, output=output),
                    scenario,
                )
            else:
                text = clean_text(text, "样本文本", 4000)
            sample_id = self._insert_sample(db, version_id, label, text, "manual")
            if sample_id is None:
                raise IntentError("同类别已有相同文本，包括已删除或驳回的记录。")
            return sample_id

    def seeds(self, version_id: str, label: str) -> list[dict]:
        scenario = next(x for x in self.version(version_id)["config"] if x["label"] == label)
        if is_json_scenario(scenario):
            # Wrapping protects a user-provided "id" field from internal bookkeeping.
            return [{"id": f"reference:{label}:{i}", "record": x} for i, x in enumerate(scenario["examples"])]
        with self.connect() as db:
            return [
                dict(x)
                for x in db.execute(
                    "SELECT id,text FROM samples WHERE version_id=? AND label=? AND deleted=0 "
                    "AND (status='approved' OR (source='user' AND status<>'rejected')) ORDER BY id LIMIT 500",
                    (version_id, label),
                )
            ]

    @staticmethod
    def conflicts(db, version_id: str) -> set[str]:
        return {
            x[0]
            for x in db.execute(
                "SELECT fingerprint FROM samples WHERE version_id=? AND deleted=0 AND status<>'rejected' "
                "GROUP BY fingerprint HAVING COUNT(DISTINCT CASE WHEN record_json='' AND instruction='' THEN label ELSE output END)>1",
                (version_id,),
            )
        }

    def samples(self, version_id: str, label="", status="", query="", page=1, page_size=25, deleted=False):
        page = integer(page, "页码")
        page_size = integer(page_size, "每页数量", maximum=100)
        clause, values = "version_id=? AND deleted=?", [version_id, int(deleted)]
        for column, value in (("label", label), ("status", status)):
            if value:
                clause += f" AND {column}=?"
                values.append(value)
        if query:
            clause += " AND (instr(text,?)>0 OR instr(instruction,?)>0 OR instr(output,?)>0 OR instr(record_json,?)>0)"
            values.extend([str(query)] * 4)
        with self.connect() as db:
            count = db.execute(f"SELECT COUNT(*) FROM samples WHERE {clause}", values).fetchone()[0]
            page = min(page, max(1, (count + page_size - 1) // page_size))
            rows = [
                dict(x)
                for x in db.execute(
                    f"SELECT * FROM samples WHERE {clause} ORDER BY created,rowid LIMIT ? OFFSET ?",
                    values + [page_size, (page - 1) * page_size],
                )
            ]
            conflicts = self.conflicts(db, version_id)
        for row in rows:
            row["record"] = stored_record(row)
            row["original_record"] = stored_record(row, original=True)
            row["conflict"] = row["fingerprint"] in conflicts and row["status"] != "rejected" and not row["deleted"]
        return rows, count, page

    def counts(self, version_id: str) -> dict:
        with self.connect() as db:
            rows = db.execute(
                "SELECT label,status,deleted,COUNT(*) AS n FROM samples WHERE version_id=? GROUP BY label,status,deleted",
                (version_id,),
            ).fetchall()
            conflicts = len(self.conflicts(db, version_id))
        return {"groups": [dict(x) for x in rows], "conflicts": conflicts}

    def review(
        self,
        version_id: str,
        selected: list[dict],
        action: str,
        text=None,
        label=None,
        instruction=None,
        output=None,
        record=None,
    ):
        if action not in {"approved", "rejected", "delete", "restore", "edit"} or not selected:
            raise IntentError("请先选择需要操作的样本。")
        if action == "edit" and len(selected) != 1:
            raise IntentError("编辑时请只选择一条样本。")
        if len({x["id"] for x in selected}) != len(selected):
            raise IntentError("选择的样本重复，请刷新后重试。")
        with self.connect(True) as db:
            config = json.loads(self.require(db, "versions", version_id)["config"])
            valid_labels = {x["label"] for x in config}
            for item in selected:
                before = self.require(db, "samples", item["id"])
                if before["version_id"] != version_id or before["revision"] != item.get("revision"):
                    raise IntentError("样本已被修改或属于其他版本，请刷新列表后重新操作。")
                after = dict(before)
                if action == "edit":
                    if before["deleted"]:
                        raise IntentError("请先恢复已删除的样本。")
                    if label not in valid_labels:
                        raise IntentError("类别不存在。")
                    scenario = next(x for x in config if x["label"] == label)
                    after.update(label=label, status="pending")
                    if is_json_scenario(scenario):
                        candidate = record
                        if candidate is None:
                            candidate = stored_record(before) or {}
                            candidate.update(instruction=instruction, input=text, output=output)
                        candidate = validate_record(candidate, scenario)
                        instr, content, answer = record_projection(candidate)
                        after.update(
                            text=content,
                            instruction=instr,
                            output=answer,
                            record_json=json.dumps(candidate, ensure_ascii=False),
                        )
                        after["fingerprint"] = record_fingerprint(candidate)
                    else:
                        if before["record_json"] or before["instruction"]:
                            raise IntentError("不能将 JSON 训练记录移入旧版单标签类别。")
                        after["text"] = clean_text(text, "样本文本", 4000)
                        after["fingerprint"] = fingerprint(after["text"])
                elif action in {"delete", "restore"}:
                    after.update(deleted=int(action == "delete"), status="pending")
                else:
                    if before["deleted"]:
                        raise IntentError("已删除样本需先恢复再审核。")
                    after["status"] = action
                after["revision"] += 1
                try:
                    db.execute(
                        "UPDATE samples SET text=?,label=?,fingerprint=?,status=?,deleted=?,revision=?,instruction=?,output=?,record_json=? WHERE id=?",
                        tuple(
                            after[x]
                            for x in (
                                "text",
                                "label",
                                "fingerprint",
                                "status",
                                "deleted",
                                "revision",
                                "instruction",
                                "output",
                                "record_json",
                                "id",
                            )
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise IntentError("修改后的文本与该类别已有样本重复。") from None
                db.execute(
                    "INSERT INTO review_events VALUES (?,?,?,?,?,?)",
                    (uid(), before["id"], action, dump(before), dump(after), now()),
                )

    def seed_usage(self, job_id, label):
        usage, previous = {}, []
        with self.connect() as db:
            for row in db.execute(
                "SELECT seed_ids FROM batches WHERE job_id=? AND label=? ORDER BY rowid", (job_id, label)
            ):
                previous = json.loads(row["seed_ids"])
                for seed_id in previous:
                    usage[seed_id] = usage.get(seed_id, 0) + 1
        return usage, previous

    def approve_pending(self, db, version_id):
        self.require(db, "versions", version_id)
        if db.execute(
            "SELECT 1 FROM jobs WHERE version_id=? AND status IN ('queued','running','pausing','cancelling')",
            (version_id,),
        ).fetchone():
            raise IntentError("请先暂停生成或等待任务完成，再一键审核。")
        if self.conflicts(db, version_id):
            raise IntentError("存在答案冲突，请先处理后再一键审核。")
        rows = db.execute(
            "SELECT * FROM samples WHERE version_id=? AND status='pending' AND deleted=0", (version_id,)
        ).fetchall()
        for row in rows:
            before = dict(row)
            after = {**before, "status": "approved", "revision": before["revision"] + 1}
            db.execute("UPDATE samples SET status='approved',revision=revision+1 WHERE id=?", (before["id"],))
            db.execute(
                "INSERT INTO review_events VALUES (?,?,?,?,?,?)",
                (uid(), before["id"], "approved", dump(before), dump(after), now()),
            )
        return len(rows)

    def approve_all(self, version_id):
        with self.connect(True) as db:
            return self.approve_pending(db, version_id)

    def create_job(self, version_id: str, targets: dict[str, int], provider: dict) -> str:
        with self.connect(True) as db:
            version = self.require(db, "versions", version_id)
            project = self.require(db, "projects", version["project_id"])
            if project["current_version"] != version_id:
                raise IntentError("请在项目的最新配置版本中生成数据。")
            labels = {x["label"] for x in json.loads(version["config"])}
            if not targets or not set(targets) <= labels:
                raise IntentError("生成配额包含未知类别。")
            targets = {k: integer(v, "生成数量") for k, v in targets.items()}
            if sum(targets.values()) > 100000:
                raise IntentError("单次任务不能超过 100000 条。")
            busy = db.execute(
                "SELECT 1 FROM jobs WHERE version_id=? AND status IN ('queued','running','pausing','cancelling')",
                (version_id,),
            ).fetchone()
            if busy:
                raise IntentError("该项目已有正在执行的生成任务。")
            # Only non-secret provider metadata may enter SQLite.
            public_keys = (
                "kind",
                "provider",
                "model",
                "template",
                "adapter",
                "base_url",
                "timeout_seconds",
                "max_tokens",
                "temperature",
                "json_mode",
                "thinking",
                "batch_size",
                "seed_count",
            )
            safe_provider = {k: provider[k] for k in public_keys if k in provider}
            job_id = uid()
            batch_size = integer(provider.get("batch_size", 10), "批大小", maximum=10)
            attempts = sum((x + batch_size - 1) // batch_size for x in targets.values()) * 5 + 10
            db.execute(
                "INSERT INTO jobs (id,version_id,status,targets,provider,max_attempts,created,updated) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, version_id, "queued", dump(targets), dump(safe_provider), attempts, now(), now()),
            )
            return job_id

    def job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = self.require(db, "jobs", job_id)
            row["by_label"] = {
                x[0]: x[1]
                for x in db.execute(
                    "SELECT label,SUM(accepted) FROM batches WHERE job_id=? GROUP BY label",
                    (job_id,),
                )
            }
        row["targets"], row["provider"] = json.loads(row["targets"]), json.loads(row["provider"])
        return row

    def jobs(self, version_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(x)
                for x in db.execute(
                    "SELECT id,status,accepted,created FROM jobs WHERE version_id=? ORDER BY created DESC,rowid DESC",
                    (version_id,),
                )
            ]

    def transition(self, job_id: str, old: tuple[str, ...], new: str, error: str = "") -> bool:
        marks = ",".join("?" for _ in old)
        with self.connect(True) as db:
            cursor = db.execute(
                f"UPDATE jobs SET status=?,error=?,updated=? WHERE id=? AND status IN ({marks})",
                [new, error[:500], now(), job_id, *old],
            )
            return bool(cursor.rowcount)

    def claim_job(self, job_id: str, descriptor: dict):
        with self.connect(True) as db:
            job = self.require(db, "jobs", job_id)
            version = self.require(db, "versions", job["version_id"])
            project = self.require(db, "projects", version["project_id"])
            if project["current_version"] != version["id"]:
                raise IntentError("类别配置已产生新版本，旧生成任务不能继续；请在当前版本新建任务。")
            expected = json.loads(job["provider"])
            if {k: descriptor.get(k) for k in expected} != expected:
                raise IntentError("生成配置与原任务不一致。请恢复原配置后继续，或新建任务。")
            if job["status"] not in {"queued", "paused", "interrupted", "failed"}:
                raise IntentError("当前任务不能继续，请选择暂停、中断或失败的任务。")
            busy = db.execute(
                "SELECT 1 FROM jobs j JOIN versions v ON j.version_id=v.id WHERE v.project_id=? "
                "AND j.id<>? AND j.status IN ('queued','running','pausing','cancelling')",
                (project["id"], job_id),
            ).fetchone()
            if busy:
                raise IntentError("此项目已有其他正在执行的生成任务。")
            db.execute(
                "UPDATE jobs SET status='running',error='',empty_batches=0,"
                "max_attempts=max(max_attempts,attempts+10),updated=? WHERE id=?",
                (now(), job_id),
            )

    def recover_jobs(self):
        # Called once by the process-wide service while holding the workspace lock.
        with self.connect(True) as db:
            db.execute(
                "UPDATE jobs SET status='interrupted',error='服务中断，已保留完成的批次，可继续。',updated=? "
                "WHERE status IN ('queued','running','pausing','cancelling')",
                (now(),),
            )

    def commit_batch(
        self,
        job_id: str,
        batch_id: str,
        label: str,
        texts: list[str],
        seed_ids: list[str],
        produced: int,
        invalid: int,
        error: str = "",
    ) -> int:
        with self.connect(True) as db:
            existing = db.execute(
                "SELECT accepted FROM batches WHERE id=? AND job_id=?", (batch_id, job_id)
            ).fetchone()
            if existing:
                return existing[0]
            job = self.require(db, "jobs", job_id)
            if job["status"] not in {"running", "pausing", "cancelling"}:
                raise IntentError("任务当前状态不允许写入批次。")
            targets = json.loads(job["targets"])
            if label not in targets:
                raise IntentError("批次类别不在任务中。")
            done = db.execute(
                "SELECT COALESCE(SUM(accepted),0) FROM batches WHERE job_id=? AND label=?", (job_id, label)
            ).fetchone()[0]
            remaining, accepted, duplicates = targets[label] - done, 0, 0
            config = json.loads(self.require(db, "versions", job["version_id"])["config"])
            scenario = next(x for x in config if x["label"] == label)
            reference_keys = (
                {record_fingerprint(x) for x in scenario["examples"]} if is_json_scenario(scenario) else set()
            )
            for text in texts:
                if accepted >= remaining:
                    break
                if is_json_scenario(scenario):
                    text = validate_record(text, scenario, generated=True)
                    if record_fingerprint(text) in reference_keys:
                        duplicates += 1
                        continue
                else:
                    text = clean_text(text, "生成文本")
                sample = self._insert_sample(
                    db, job["version_id"], label, text, "generated", job_id, batch_id, seed_ids
                )
                if sample:
                    accepted += 1
                else:
                    duplicates += 1
            db.execute(
                "INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?,?)",
                (batch_id, job_id, label, accepted, produced, duplicates, invalid, dump(seed_ids), error[:500], now()),
            )
            db.execute(
                "UPDATE jobs SET accepted=accepted+?,produced=produced+?,duplicates=duplicates+?,invalid=invalid+?,"
                "attempts=attempts+1,empty_batches=?,error=?,updated=? WHERE id=?",
                (
                    accepted,
                    produced,
                    duplicates,
                    invalid,
                    0 if accepted else job["empty_batches"] + 1,
                    error[:500],
                    now(),
                    job_id,
                ),
            )
            return accepted

    def exports(self, version_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(x)
                for x in db.execute(
                    "SELECT * FROM exports WHERE version_id=? ORDER BY created DESC,rowid DESC",
                    (version_id,),
                )
            ]
