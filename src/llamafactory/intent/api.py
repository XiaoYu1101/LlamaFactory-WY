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
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Mount

from .dataset_mapping import text_mapping
from .export import freeze_dataset, get_export
from .label_inference import verify_label_task
from .models import IntentError, validate_intents
from .providers import APIProvider, configured_key, private_endpoint, read_config
from .records import json_schema, parse_examples
from .service import get_service
from .training_config import LoraConfig, get_plan, list_plans, plan_bundle, save_plan, templates


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectInput(Payload):
    name: str = Field(max_length=100)
    intents: list[dict] = Field(min_length=1, max_length=100)
    project_id: str | None = None
    base_version_id: str | None = None


class SampleInput(Payload):
    label: str
    text: str = Field(default="", max_length=16000)
    record: dict | None = None
    instruction: str | None = Field(default=None, max_length=12000)
    output: str | None = Field(default=None, max_length=4000)


class Selection(Payload):
    id: str
    revision: int = Field(ge=0, strict=True)


class ReviewInput(Payload):
    selected: list[Selection] = Field(min_length=1, max_length=100)
    action: Literal["approved", "rejected", "delete", "restore", "edit"]
    text: str | None = Field(default=None, max_length=16000)
    record: dict | None = None
    instruction: str | None = Field(default=None, max_length=12000)
    output: str | None = Field(default=None, max_length=4000)
    label: str | None = None


class ScenarioDelete(Payload):
    label: str = Field(min_length=1, max_length=100)


class ExamplesInput(Payload):
    examples: str = Field(max_length=256000)


class LabelTaskInput(Payload):
    examples: list[dict] = Field(min_length=4, max_length=100)
    description: str = Field(default="", max_length=1000)
    snapshot: dict


class JobInput(Payload):
    label: str | None = None
    amount: int | None = Field(default=None, ge=1, le=100000, strict=True)


def create_api(service, engine=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    store = service.store
    chatter = engine.chatter if engine else None

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "不允许跨站修改工作区。"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "不允许跨站修改工作区。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(IntentError)
    async def business_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/training/options")
    def training_options():
        with store.connect() as db:
            rows = db.execute(
                "SELECT e.id,e.manifest,e.created,p.name FROM exports e JOIN versions v ON e.version_id=v.id JOIN projects p ON v.project_id=p.id ORDER BY e.created DESC,e.rowid DESC"
            ).fetchall()
        return {
            "templates": templates(),
            "exports": [{**dict(row), "manifest": json.loads(row["manifest"])} for row in rows],
        }

    @app.get("/training/configs")
    def training_configs():
        return list_plans(store)

    @app.get("/training/configs/{plan_id}")
    def training_config(plan_id: str):
        return get_plan(store, plan_id)

    @app.post("/exports/{export_id}/training-config")
    def save_training_config(export_id: str, body: LoraConfig):
        return save_plan(store, export_id, body)

    @app.get("/training/configs/{plan_id}/download")
    def download_training_config(plan_id: str):
        return Response(
            plan_bundle(store, plan_id),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="lora-training.zip"'},
        )

    @app.get("/settings")
    def settings():
        config = read_config(service.config_path)
        return {
            **config,
            "integrated": engine is not None,
            "api_key_required": not private_endpoint(config["base_url"]),
            "api_key_configured": bool(configured_key(config)),
            "config_path": str(service.config_path or os.getenv("WY_INTENT_CONFIG", "config/intent_generation.json")),
        }

    @app.post("/settings/discover")
    def discover_models():
        config = read_config(service.config_path)
        if config["provider"] == "local":
            raise IntentError("进程内模型请在 Chat 页面加载；自动检测用于已部署的兼容 API 服务。")
        return {"models": APIProvider(config, resolve_model=False).models()}

    @app.get("/label-tasks")
    def label_tasks(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
        return service.label_tasks.list(page, page_size)

    @app.get("/label-tasks/{task_id}")
    def label_task(task_id: str):
        return service.label_tasks.get(task_id)

    @app.post("/label-tasks")
    def infer_labels(body: LabelTaskInput):
        return service.label_tasks.create(
            body.examples,
            body.description,
            body.snapshot,
            read_config(service.config_path),
            lambda config: service.provider(chatter, config),
        )

    @app.post("/label-tasks/{task_id}/retry")
    def retry_labels(task_id: str):
        return service.label_tasks.retry(
            task_id, read_config(service.config_path), lambda config: service.provider(chatter, config)
        )

    @app.post("/scenarios/validate")
    def validate_scenario(body: dict):
        scenario = validate_intents([body])[0]
        with store.connect() as db:
            verify_label_task(db, scenario)
        return scenario

    @app.post("/examples/parse")
    def inspect_examples(body: ExamplesInput):
        examples = parse_examples(body.examples)
        return {
            "examples": examples,
            "count": len(examples),
            "output_labels": [],
            "schema": json_schema(examples[0]),
            "training_mapping": text_mapping(examples[0]),
        }

    @app.get("/projects")
    def projects():
        return store.projects()

    @app.post("/projects")
    def save_project(body: ProjectInput):
        return store.save_project(body.name, body.intents, body.project_id, body.base_version_id)

    @app.get("/projects/{project_id}")
    def project(project_id: str):
        result = store.project(project_id)
        with store.connect() as db:
            result["versions"] = [
                dict(row)
                for row in db.execute(
                    "SELECT id,created FROM versions WHERE project_id=? ORDER BY created DESC,rowid DESC",
                    (project_id,),
                )
            ]
        return result

    @app.get("/versions/{version_id}")
    def version(version_id: str):
        # Listing stays available if an exported file was moved or damaged.
        # Validate immutable files when downloading or launching training.
        with store.connect() as db:
            exports = [
                {**dict(row), "manifest": json.loads(row["manifest"])}
                for row in db.execute(
                    "SELECT * FROM exports WHERE version_id=? ORDER BY created DESC,rowid DESC", (version_id,)
                )
            ]
        return {
            **store.version(version_id),
            "counts": store.counts(version_id),
            "jobs": [store.job(row["id"]) for row in store.jobs(version_id)],
            "exports": exports,
        }

    @app.get("/versions/{version_id}/samples")
    def samples(
        version_id: str,
        label: str = "",
        status: str = "",
        query: str = "",
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        deleted: bool = False,
    ):
        store.version(version_id)
        rows, total, page = store.samples(version_id, label, status, query, page, page_size, deleted)
        return {"rows": rows, "total": total, "page": page}

    @app.post("/versions/{version_id}/samples")
    def add_sample(version_id: str, body: SampleInput):
        return {"id": store.add_sample(version_id, body.label, body.text, body.instruction, body.output, body.record)}

    @app.post("/versions/{version_id}/clear-samples")
    def clear_samples(version_id: str):
        return store.clear_samples(version_id)

    @app.post("/versions/{version_id}/delete-scenario")
    def delete_scenario(version_id: str, body: ScenarioDelete):
        return store.delete_scenario(version_id, body.label)

    @app.post("/versions/{version_id}/review")
    def review(version_id: str, body: ReviewInput):
        store.review(
            version_id,
            [x.model_dump() for x in body.selected],
            body.action,
            body.text,
            body.label,
            body.instruction,
            body.output,
            body.record,
        )
        return {"ok": True}

    @app.post("/versions/{version_id}/jobs")
    def start_job(version_id: str, body: JobInput):
        if (body.label is None) != (body.amount is None):
            raise IntentError("补生成需同时指定类别和数量。")
        return {"id": service.start(version_id, chatter, body.label, body.amount)}

    @app.post("/jobs/{job_id}/{action}")
    def job_action(job_id: str, action: Literal["pause", "resume", "cancel"]):
        store.job(job_id)
        if action == "resume":
            service.jobs.start(job_id, service.provider(chatter))
        elif action == "pause":
            service.jobs.pause(job_id)
        else:
            service.jobs.cancel(job_id)
        return store.job(job_id)

    @app.post("/versions/{version_id}/approve-all")
    def approve_all(version_id: str):
        return {"approved": store.approve_all(version_id)}

    @app.post("/versions/{version_id}/approve-export")
    def approve_export(version_id: str):
        return freeze_dataset(store, version_id, approve_pending=True)

    @app.post("/versions/{version_id}/exports")
    def freeze(version_id: str):
        return freeze_dataset(store, version_id)

    @app.get("/exports/{export_id}/json")
    def download_json(export_id: str):
        export = get_export(store, export_id)
        return FileResponse(Path(export["path"]) / "train.json", filename=f"intent-{export_id[:8]}.json")

    @app.get("/exports/{export_id}/download")
    def download(export_id: str):
        export = get_export(store, export_id)
        return FileResponse(Path(export["path"]) / "dataset.zip", filename=f"intent-{export_id[:8]}.zip")

    return app


def install_routes(app, service, engine=None):
    """Mount before Gradio's catch-all without changing its training/queue routes."""
    static = Path(__file__).with_name("static")
    if not (static / "index.html").is_file():
        raise IntentError("缺少前端构建文件，请在 web/intent 执行 npm ci 和 npm run build。")
    app.router.routes[0:0] = [
        Mount("/intent-api", app=create_api(service, engine)),
        Mount("/intent", app=StaticFiles(directory=static, html=True)),
    ]


def create_app(root=None, config_path=None, service=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    install_routes(app, service or get_service(root, config_path))

    @app.get("/")
    def home():
        return RedirectResponse("intent/")

    return app
