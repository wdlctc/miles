import json

import httpx
import pytest
from fastapi.testclient import TestClient

from miles_platform.api import create_app
from miles_platform.gateway import record
from miles_platform.runtime import deploy
from miles_platform.store import Store, digest, now
from miles_platform.worker import tick


@pytest.fixture
def system(tmp_path):
    store = Store(tmp_path / "control")
    calls = []

    def upstream(request):
        calls.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "42"}}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            ]
            content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=content, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "42"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        )

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(upstream), **kwargs)

    client = TestClient(create_app(store, factory))
    store.execute("INSERT INTO datasets VALUES ('ds','test','/unused',4,1,?, 'hash')", (now(),))
    store.execute("INSERT INTO models VALUES ('mdl','model','/unused',NULL,'ds',?,'hash',1,'{}')", (now(),))
    store.execute("INSERT INTO evaluations VALUES ('eval','mdl','ds',?,1,1,0.375,1,'[]')", (now(),))
    store.execute("INSERT INTO deployments VALUES ('dep','mdl','http://127.0.0.1:18081',?,'deploy')", (now(),))
    store.set("active_deployment", "dep")
    password = (store.root / "admin-password").read_text()
    assert client.post("/api/login", json={"password": password}).status_code == 200
    return client, store, calls


def create_key(client, rpm=30):
    response = client.post("/api/keys", json={"name": "test key", "rpm": rpm})
    assert response.status_code == 200
    return response.json()


def test_auth_origin_and_revocation(system):
    client, store, _ = system
    assert client.get("/api/state").status_code == 200
    assert client.post("/api/jobs", headers={"Origin": "https://attacker.example"}, json={}).status_code == 403
    assert client.get("/api/state", headers={"Host": "attacker.example"}).status_code == 400
    key = create_key(client)
    assert store.one("SELECT hash FROM api_keys WHERE id=?", (key["id"],))["hash"] == digest(key["secret"])
    headers = {"Authorization": "Bearer " + key["secret"]}
    assert client.get("/v1/models", headers=headers).status_code == 200
    client.delete("/api/keys/" + key["id"])
    assert client.get("/v1/models", headers=headers).status_code == 401
    client.post("/api/logout")
    assert client.get("/api/state").status_code == 401


def test_idempotency_cancel_and_restart_persistence(system):
    client, store, _ = system
    body = {"name": "run", "dataset_id": "ds", "updates": 1}
    headers = {"Idempotency-Key": "same-submission"}
    first = client.post("/api/jobs", json=body, headers=headers).json()
    second = client.post("/api/jobs", json=body, headers=headers).json()
    assert first["id"] == second["id"]
    assert client.post("/api/jobs", json={**body, "updates": 2}, headers=headers).status_code == 409
    client.post(f"/api/jobs/{first['id']}/cancel")
    reopened = Store(store.root)
    assert reopened.one("SELECT status FROM jobs WHERE id=?", (first["id"],))["status"] == "cancelled"


def test_deployment_gate_and_health_failure_preserve_route(system, monkeypatch):
    client, store, _ = system
    store.execute("UPDATE evaluations SET passed=0")
    assert client.post("/api/models/mdl/deploy").status_code == 409
    store.execute("UPDATE evaluations SET passed=1")
    job = client.post("/api/models/mdl/deploy").json()
    store.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))

    def fail(*args):
        raise RuntimeError("unhealthy")

    monkeypatch.setattr("miles_platform.runtime.ensure_server", fail)
    with pytest.raises(RuntimeError, match="unhealthy"):
        deploy(store, job)
    assert store.active()["id"] == "dep"
    monkeypatch.setattr("miles_platform.runtime.ensure_server", lambda *args: "http://127.0.0.1:18082")
    deploy(store, job)
    assert store.active()["endpoint"].endswith(":18082")


def test_stream_and_nonstream_usage_rate_snapshot(system):
    client, store, calls = system
    client.put("/api/rates", json={"gpu_hour": 1.5, "input_million": 2, "output_million": 4})
    key = create_key(client, rpm=2)
    headers = {"Authorization": "Bearer " + key["secret"]}
    body = {"messages": [{"role": "user", "content": "17 + 25?"}], "model": "miles-default"}
    assert client.post("/v1/chat/completions", headers=headers, json=body).json()["model"] == "miles-default"
    response = client.post("/v1/chat/completions", headers=headers, json={**body, "stream": True})
    assert "data: [DONE]" in response.text
    rows = store.rows("SELECT * FROM usage")
    assert len(rows) == 2
    assert all(r["complete"] and r["prompt_tokens"] == 10 and r["completion_tokens"] == 2 for r in rows)
    assert rows[0]["cost"] == pytest.approx(0.000028)
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 429
    assert len(calls) == 2
    client.put("/api/rates", json={"input_million": 20, "output_million": 40})
    assert store.rows("SELECT cost FROM usage")[0]["cost"] == pytest.approx(0.000028)
    csv = client.get("/api/usage.csv").text
    assert "estimated_usd" in csv and key["secret"] not in csv


def test_disconnected_usage_is_not_invented(system):
    client, store, _ = system
    client.post("/api/playground", json={"messages": [{"role": "user", "content": "x"}]})
    row = store.one("SELECT * FROM usage")
    record(store, row["id"], 499, None)
    row = store.one("SELECT * FROM usage")
    assert not row["complete"] and row["cost"] is None and row["status"] == 499


def test_usage_totals_include_jobs_outside_recent_page(system):
    client, store, _ = system
    with store.connect(write=True) as db:
        db.executemany(
            "INSERT INTO jobs(id,kind,name,status,params,created,gpu_seconds,cost) VALUES (?,'train','old','succeeded','{}',?,3600,2)",
            [(f"job_{index}", now() + index) for index in range(101)],
        )
    data = client.get("/api/state").json()
    assert len(data["jobs"]) == 100
    assert data["job_totals"]["jobs"] == 101
    assert data["job_totals"]["training_gpu_seconds"] == 101 * 3600
    assert data["job_totals"]["training_cost"] == 202


def test_dataset_validation_rejects_leakage_and_persists(system):
    client, store, _ = system

    def row(n):
        return json.dumps({"prompt": [{"role": "user", "content": f"{n} + 1?"}], "label": str(n + 1)})

    body = {"name": "custom", "train_jsonl": "\n".join(row(n) for n in range(4)), "eval_jsonl": row(0)}
    assert client.post("/api/datasets", json=body).status_code == 422
    body["eval_jsonl"] = row(5)
    result = client.post("/api/datasets", json=body)
    assert result.status_code == 200
    item = store.one("SELECT * FROM datasets WHERE id=?", (result.json()["id"],))
    assert item["train_rows"] == 4 and item["eval_rows"] == 1
    assert client.post("/api/jobs", json={"name": "bad", "dataset_id": item["id"], "updates": 100}).status_code == 422


def test_worker_adopts_live_runner_and_cleans_cancelled_training(system, monkeypatch):
    _, store, _ = system
    job = store.enqueue("train", "resume", {})
    store.execute("UPDATE jobs SET status='running',phase='训练中',started=?,rate=2 WHERE id=?", (now(), job["id"]))
    commands = []
    monkeypatch.setattr(
        "miles_platform.worker.supervise", lambda action, service, **kw: commands.append(action) or "RUNNING"
    )
    monkeypatch.setattr("miles_platform.worker.stop_training", lambda: commands.append("stop_ray"))
    tick(store)
    assert commands == ["status"]
    assert store.one("SELECT status FROM jobs WHERE id=?", (job["id"],))["status"] == "running"
    store.execute("UPDATE jobs SET status='cancel_requested' WHERE id=?", (job["id"],))
    tick(store)
    assert "stop" in commands and "stop_ray" in commands
    assert store.one("SELECT status FROM jobs WHERE id=?", (job["id"],))["status"] == "cancelled"
