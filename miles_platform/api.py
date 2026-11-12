"""Single-workspace administration API and static console."""

import csv
import hashlib
import hmac
import io
import json
import secrets
import subprocess
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from miles_platform.bootstrap import bootstrap
from miles_platform.gateway import Chat, proxy
from miles_platform.store import configured_store, digest, now, uid


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    password: str = Field(min_length=1, max_length=200)


class Train(Input):
    name: str = Field(min_length=1, max_length=80)
    dataset_id: str
    updates: int = Field(default=2, ge=1, le=10)
    learning_rate: float = Field(default=1e-6, ge=1e-7, le=1e-5)


class Key(Input):
    name: str = Field(min_length=1, max_length=60)
    rpm: int = Field(default=30, ge=1, le=120)


class Rates(Input):
    gpu_hour: float | None = Field(default=None, ge=0, le=1000)
    input_million: float | None = Field(default=None, ge=0, le=10000)
    output_million: float | None = Field(default=None, ge=0, le=10000)


class Dataset(Input):
    name: str = Field(min_length=1, max_length=80)
    train_jsonl: str = Field(max_length=5_000_000)
    eval_jsonl: str = Field(max_length=1_000_000)


def clean_rows(text, minimum, maximum):
    rows = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            Chat(messages=row["prompt"])
            if not isinstance(row["label"], (str, int, float)) or not str(row["label"]).strip():
                raise ValueError("label must be nonempty")
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"Invalid JSONL row {len(rows) + 1}: {exc}") from None
        signature = digest(json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False))
        if signature in seen:
            raise HTTPException(422, "Duplicate prompts are not allowed")
        seen.add(signature)
        rows.append({"prompt": row["prompt"], "label": str(row["label"])})
    if not minimum <= len(rows) <= maximum:
        raise HTTPException(422, f"Expected {minimum}–{maximum} rows")
    return rows, seen


def create_app(store=None, client_factory=httpx.AsyncClient):
    store = store or configured_store()
    bootstrap(store)
    # The supported deployment has one API process. Its old in-flight requests
    # cannot survive an API restart; mark them incomplete, never invent usage.
    store.execute("UPDATE usage SET status=599,ended=? WHERE status=102 AND ended IS NULL", (now(),))
    app = FastAPI(title="Miles Platform", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    attempts = []

    @app.middleware("http")
    async def security(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            if urlparse(origin).netloc != request.headers.get("host"):
                return PlainTextResponse("Cross-origin requests are not allowed", status_code=403)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return PlainTextResponse("Invalid content length", status_code=400)
        if length > 6_100_000:
            return PlainTextResponse("Request too large", status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        )
        if request.url.path == "/api/docs":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
            )
        if request.url.path.startswith(("/api/", "/v1/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    def admin(request: Request):
        token = request.cookies.get("miles_session", "")
        session = store.one("SELECT expires FROM sessions WHERE hash=?", (digest(token),))
        if not session or session["expires"] <= now():
            raise HTTPException(401, "Please sign in")
        return "admin"

    def api_key(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "A Bearer API key is required")
        key = store.one("SELECT * FROM api_keys WHERE hash=? AND revoked IS NULL", (digest(authorization[7:]),))
        if not key:
            raise HTTPException(401, "Invalid or revoked API key")
        return key

    def model(model_id):
        row = store.one("SELECT * FROM models WHERE id=?", (model_id,))
        if not row:
            raise HTTPException(404, "Model not found")
        return row

    def enqueue(kind, name, params, model_id=None, idem=None):
        try:
            return store.enqueue(kind, name, params, model_id, idem)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/login")
    def login(body: Login, response: Response):
        attempts[:] = [t for t in attempts if t > now() - 60]
        if len(attempts) >= 8:
            raise HTTPException(429, "Too many login attempts; retry in one minute")
        attempts.append(now())
        stored = store.setting("admin_hash")
        actual = hashlib.scrypt(body.password.encode(), salt=stored["salt"].encode(), n=16384, r=8, p=1).hex()
        if not hmac.compare_digest(actual, stored["hash"]):
            raise HTTPException(401, "Incorrect password")
        token = secrets.token_urlsafe(32)
        store.execute("DELETE FROM sessions WHERE expires<?", (now(),))
        store.execute("INSERT INTO sessions VALUES (?,?)", (digest(token), now() + 8 * 3600))
        response.set_cookie("miles_session", token, httponly=True, samesite="strict", max_age=8 * 3600)
        store.audit("login", "Operator signed in")
        return {"ok": True}

    @app.post("/api/logout", dependencies=[Depends(admin)])
    def logout(request: Request, response: Response):
        store.execute("DELETE FROM sessions WHERE hash=?", (digest(request.cookies["miles_session"]),))
        response.delete_cookie("miles_session")
        return {"ok": True}

    @app.get("/api/state", dependencies=[Depends(admin)])
    def state():
        jobs = store.rows("SELECT * FROM jobs ORDER BY created DESC LIMIT 100")
        models = store.rows("SELECT * FROM models ORDER BY created DESC")
        for row in jobs:
            row["params"] = json.loads(row["params"])
        for row in models:
            row.pop("path")
            row["metadata"] = json.loads(row["metadata"])
        totals = store.one(
            """SELECT COUNT(*) requests, SUM(prompt_tokens) prompt_tokens,
                  SUM(completion_tokens) completion_tokens,SUM(cost) cost,
                  SUM(CASE WHEN status=200 THEN 1 ELSE 0 END) successful,
                  SUM(CASE WHEN complete=0 THEN 1 ELSE 0 END) unmetered FROM usage"""
        )
        return {
            "jobs": jobs,
            "job_totals": store.one(
                """SELECT COUNT(*) jobs, SUM(gpu_seconds) training_gpu_seconds,
                SUM(cost) training_cost, SUM(CASE WHEN status IN ('queued','running','cancel_requested') THEN 1 ELSE 0 END) active FROM jobs"""
            ),
            "models": models,
            "datasets": store.rows("SELECT id,name,train_rows,eval_rows,sha256 FROM datasets"),
            "evaluations": store.rows(
                "SELECT id,model_id,dataset_id,created,correct,total,threshold,passed FROM evaluations ORDER BY created DESC"
            ),
            "deployments": store.rows("SELECT * FROM deployments ORDER BY created DESC LIMIT 50"),
            "active": store.active(),
            "keys": store.rows("SELECT id,name,prefix,created,revoked,rpm FROM api_keys ORDER BY created DESC"),
            "usage": store.rows("SELECT * FROM usage ORDER BY created DESC LIMIT 50"),
            "totals": totals,
            "rates": store.setting("rates"),
            "quality_threshold": store.setting("quality_threshold"),
            "worker_heartbeat": store.setting("worker_heartbeat"),
            "audit": store.rows("SELECT * FROM audit ORDER BY id DESC LIMIT 15"),
        }

    @app.post("/api/jobs", dependencies=[Depends(admin)], status_code=202)
    def train(body: Train, idempotency_key: str | None = Header(default=None, max_length=128)):
        if not store.one("SELECT id FROM datasets WHERE id=?", (body.dataset_id,)):
            raise HTTPException(404, "Dataset not found")
        return enqueue("train", body.name, body.model_dump(), idem=idempotency_key)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(admin)])
    def cancel(job_id: str):
        with store.connect(write=True) as db:
            job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job:
                raise HTTPException(404, "Job not found")
            if job["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',phase='已取消',ended=? WHERE id=?", (now(), job_id))
            elif job["status"] == "running":
                db.execute("UPDATE jobs SET status='cancel_requested' WHERE id=?", (job_id,))
        store.audit("cancel", job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(admin)], response_class=PlainTextResponse)
    def logs(job_id: str):
        if not store.one("SELECT id FROM jobs WHERE id=?", (job_id,)):
            raise HTTPException(404, "Job not found")
        file = store.root / "logs" / f"{job_id}.log"
        if not file.exists():
            return "等待执行日志。历史导入任务的日志保留在服务器 /var/log/miles-platform-smoke.log。"
        with file.open("rb") as stream:
            stream.seek(max(0, file.stat().st_size - 64000))
            return stream.read().decode(errors="replace")

    @app.post("/api/models/{model_id}/evaluate", dependencies=[Depends(admin)], status_code=202)
    def evaluate(model_id: str, idempotency_key: str | None = Header(default=None, max_length=128)):
        item = model(model_id)
        return enqueue("evaluate", "评测 · " + item["name"], {}, model_id, idempotency_key)

    @app.post("/api/models/{model_id}/deploy", dependencies=[Depends(admin)], status_code=202)
    def deploy(model_id: str, idempotency_key: str | None = Header(default=None, max_length=128)):
        item = model(model_id)
        if not item["validated"]:
            raise HTTPException(409, "Artifact validation has not passed")
        evaluation = store.one("SELECT * FROM evaluations WHERE model_id=? ORDER BY created DESC LIMIT 1", (model_id,))
        if not evaluation or not evaluation["passed"]:
            raise HTTPException(409, "Run and pass evaluation before deployment")
        return enqueue("deploy", "部署 · " + item["name"], {}, model_id, idempotency_key)

    @app.post("/api/deployments/{deployment_id}/rollback", dependencies=[Depends(admin)], status_code=202)
    def rollback(deployment_id: str, idempotency_key: str | None = Header(default=None, max_length=128)):
        target = store.one("SELECT * FROM deployments WHERE id=?", (deployment_id,))
        if not target:
            raise HTTPException(404, "Deployment not found")
        return enqueue("deploy", "回滚模型版本", {"rollback": deployment_id}, target["model_id"], idempotency_key)

    @app.post("/api/datasets", dependencies=[Depends(admin)])
    def dataset(body: Dataset):
        training, train_hashes = clean_rows(body.train_jsonl, 4, 5000)
        evaluation, eval_hashes = clean_rows(body.eval_jsonl, 1, 64)
        if train_hashes & eval_hashes:
            raise HTTPException(422, "Training and evaluation prompts must be disjoint")
        dataset_id = uid("ds")
        directory = store.root / "datasets" / dataset_id
        directory.mkdir(parents=True, mode=0o700)
        content = []
        for filename, rows in [("train", training), ("eval", evaluation)]:
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (directory / f"{filename}.jsonl").write_text(text)
            content.append(text)
        store.execute(
            "INSERT INTO datasets VALUES (?,?,?,?,?,?,?)",
            (dataset_id, body.name, str(directory), len(training), len(evaluation), now(), digest("".join(content))),
        )
        store.audit("dataset", dataset_id)
        return {"id": dataset_id}

    @app.post("/api/keys", dependencies=[Depends(admin)])
    def create_key(body: Key):
        token = "mp_" + secrets.token_urlsafe(32)
        key_id = uid("key")
        store.execute(
            "INSERT INTO api_keys VALUES (?,?,?,?,?,NULL,?)",
            (key_id, body.name, digest(token), token[:10] + "…" + token[-4:], now(), body.rpm),
        )
        store.audit("key_created", key_id)
        return {"id": key_id, "secret": token}

    @app.delete("/api/keys/{key_id}", dependencies=[Depends(admin)])
    def revoke(key_id: str):
        store.execute("UPDATE api_keys SET revoked=? WHERE id=? AND revoked IS NULL", (now(), key_id))
        store.audit("key_revoked", key_id)
        return {"ok": True}

    @app.put("/api/rates", dependencies=[Depends(admin)])
    def rates(body: Rates):
        store.set("rates", body.model_dump())
        store.audit("rates", json.dumps(body.model_dump()))
        return {"ok": True}

    @app.get("/api/usage.csv", dependencies=[Depends(admin)])
    def export_usage():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "kind",
                "id",
                "created_unix",
                "status",
                "input_tokens",
                "output_tokens",
                "gpu_seconds",
                "estimated_usd",
                "metering_complete",
            ]
        )
        for row in store.rows("SELECT * FROM usage ORDER BY created"):
            writer.writerow(
                [
                    "inference",
                    row["id"],
                    row["created"],
                    row["status"],
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    "",
                    row["cost"],
                    row["complete"],
                ]
            )
        for row in store.rows("SELECT * FROM jobs WHERE kind='train' AND gpu_seconds>0 ORDER BY created"):
            writer.writerow(
                ["training", row["id"], row["created"], row["status"], "", "", row["gpu_seconds"], row["cost"], 1]
            )
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="miles-usage.csv"'},
        )

    @app.post("/api/playground", dependencies=[Depends(admin)])
    async def playground(body: Chat):
        return await proxy(store, body, {"id": "console", "rpm": 30}, client_factory)

    @app.get("/api/runtime", dependencies=[Depends(admin)])
    async def runtime():
        deployment = store.active()
        serving = False
        if deployment:
            try:
                async with client_factory(timeout=3, trust_env=False) as client:
                    serving = (await client.get(deployment["endpoint"] + "/health")).status_code == 200
            except httpx.HTTPError:
                pass
        gpus = []
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            for line in process.stdout.splitlines():
                index, name, used, total, utilization = [field.strip() for field in line.split(",")]
                gpus.append(
                    {
                        "index": int(index),
                        "name": name,
                        "used_mb": int(used),
                        "total_mb": int(total),
                        "utilization": int(utilization),
                    }
                )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return {"serving_healthy": serving, "gpus": gpus, "observed": now()}

    @app.post("/v1/chat/completions")
    async def chat(body: Chat, key: Annotated[dict, Depends(api_key)]):
        return await proxy(store, body, key, client_factory)

    @app.get("/v1/models")
    def models(key: Annotated[dict, Depends(api_key)]):
        return {
            "object": "list",
            "data": (
                [{"id": "miles-default", "object": "model", "owned_by": "miles-platform"}] if store.active() else []
            ),
        }

    static = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static), name="assets")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
