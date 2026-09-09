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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Mount

from .export import freeze_dataset, get_export
from .models import IntentError
from .providers import read_config
from .service import get_service


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectInput(Payload):
    name: str = Field(max_length=100)
    intents: list[dict] = Field(min_length=1, max_length=100)
    project_id: str | None = None


class SampleInput(Payload):
    label: str
    text: str = Field(max_length=1000)


class Selection(Payload):
    id: str
    revision: int = Field(ge=0, strict=True)


class ReviewInput(Payload):
    selected: list[Selection] = Field(min_length=1, max_length=100)
    action: Literal["approved", "rejected", "delete", "restore", "edit"]
    text: str | None = Field(default=None, max_length=1000)
    label: str | None = None


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

    @app.get("/settings")
    def settings():
        config = read_config(service.config_path)
        return {
            **config,
            "integrated": engine is not None,
            "config_path": str(service.config_path or os.getenv("WY_INTENT_CONFIG", "config/intent_generation.json")),
        }

    @app.get("/projects")
    def projects():
        return store.projects()

    @app.post("/projects")
    def save_project(body: ProjectInput):
        return store.save_project(body.name, body.intents, body.project_id)

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
        return {"id": store.add_sample(version_id, body.label, body.text)}

    @app.post("/versions/{version_id}/review")
    def review(version_id: str, body: ReviewInput):
        store.review(version_id, [x.model_dump() for x in body.selected], body.action, body.text, body.label)
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

    @app.post("/versions/{version_id}/exports")
    def freeze(version_id: str):
        return freeze_dataset(store, version_id)

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
