"""CUDA-only, single-host integration using the existing Miles recipe and Supervisor."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from miles_platform.store import now, uid

REPO = Path(__file__).resolve().parents[1]
SLOTS = {letter: 18081 + index for index, letter in enumerate("abcd")}


def supervise(action, service, check=True):
    if action not in {"start", "stop", "status"} or service not in {"runner", *("slot-" + s for s in SLOTS)}:
        raise ValueError("Unsupported supervisor action")
    result = subprocess.run(
        ["supervisorctl", action, "miles-platform-" + service], capture_output=True, text=True, timeout=150
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip())
    return result.stdout.strip()


def phase(store, job_id, text):
    print(text, flush=True)
    if hasattr(store, "registry"):
        store.registry.active(store.tenant_id)
    if store.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "cancel_requested":
        raise RuntimeError("Cancellation requested")
    store.execute("UPDATE jobs SET phase=? WHERE id=?", (text, job_id))


def compute_env():
    # Customer-visible training logs/diagnostic reports must never inherit
    # payment, cloud or control-plane credentials from the supervisor environment.
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "HOSTNAME",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "HF_HOME",
        "VIRTUAL_ENV",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def train_env():
    # GPU 0 belongs to serving. Ray and all training descendants see only GPUs 1–3.
    return {
        **compute_env(),
        "CUDA_VISIBLE_DEVICES": "1,2,3",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": str(REPO),
        "NCCL_NVLS_ENABLE": "0",
    }


def stop_training():
    subprocess.run(
        [str(Path(sys.executable).parent / "ray"), "stop", "--grace-period", "5"],
        env=train_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=True,
    )


def validate_artifact(directory, base):
    # Heavy tensor libraries are imported only in the supervised GPU runner.
    import torch
    from safetensors import safe_open

    weights = directory / "model.safetensors"
    changed = 0
    with safe_open(weights, framework="pt") as trained, safe_open(
        base / "model.safetensors", framework="pt"
    ) as original:
        expected = set(original.keys())
        if "lm_head.weight" not in trained.keys():
            config = json.loads((directory / "config.json").read_text())
            if not config.get("tie_word_embeddings"):
                raise ValueError("Missing untied output weight")
            expected.remove("lm_head.weight")
        if set(trained.keys()) != expected:
            raise ValueError("Exported tensor keys differ from the base model")
        for key in trained.keys():
            value, old = trained.get_tensor(key), original.get_tensor(key)
            if value.shape != old.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid exported tensor: {key}")
            changed += int(not torch.equal(value, old))
        count = len(trained.keys())
    if changed == 0:
        raise ValueError("No stored weights changed during training")
    with weights.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    return {"sha256": checksum, "unique_tensors": count, "changed_tensors": changed, "all_finite": True}


def train(store, job):
    if shutil.disk_usage(store.root).free < 20 * 1024**3:
        raise RuntimeError("Training requires at least 20 GiB of available artifact storage")
    params = json.loads(job["params"])
    dataset = store.one("SELECT * FROM datasets WHERE id=?", (params["dataset_id"],))
    state = store.root.parent
    base_models = getattr(store, "base_models", state / "models")
    output = store.root / "runs" / job["id"]
    model_id = uid("mdl")
    exported = state / "models" / model_id
    phase(store, job["id"], "训练中 · 2 张训练 GPU + 1 张 rollout GPU")
    command = [
        sys.executable,
        "scripts/run_qwen3_0_6b_platform.py",
        "--model-dir",
        str(base_models),
        "--data-dir",
        dataset["path"],
        "--output-dir",
        str(output),
        "--num-gpus-per-node",
        "3",
        "--trainer-gpus",
        "2",
        "--num-rollout",
        str(params["updates"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--numeric-reward",
    ]
    rate = job["rate"]
    started = now()
    store.execute("UPDATE jobs SET training_started=? WHERE id=?", (started, job["id"]))
    try:
        subprocess.run(command, cwd=REPO, env=train_env(), timeout=1200, check=True)
    finally:
        stop_training()
        seconds = (now() - started) * 3
        store.execute(
            "UPDATE jobs SET gpu_seconds=?,cost=?,training_ended=? WHERE id=?",
            (seconds, seconds / 3600 * rate if rate is not None else None, now(), job["id"]),
        )
    phase(store, job["id"], "导出与权重完整性校验")
    checkpoint = output / "checkpoints" / f"iter_{params['updates']:07d}"
    subprocess.run(
        [
            sys.executable,
            "tools/convert_fsdp_to_hf.py",
            "--input-dir",
            str(checkpoint),
            "--output-dir",
            str(exported),
            "--origin-hf-dir",
            str(base_models / "Qwen3-0.6B"),
        ],
        cwd=REPO,
        timeout=300,
        check=True,
    )
    evidence = validate_artifact(exported, base_models / "Qwen3-0.6B")
    store.execute(
        "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
        (
            model_id,
            job["name"],
            str(exported),
            job["id"],
            dataset["id"],
            now(),
            evidence["sha256"],
            1,
            json.dumps(
                {"validation": evidence, "updates": params["updates"], "learning_rate": params["learning_rate"]}
            ),
        ),
    )
    store.execute("UPDATE jobs SET model_id=? WHERE id=?", (model_id, job["id"]))
    evaluate(store, {**job, "model_id": model_id})


def healthy(endpoint, model_id):
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            health = client.get(endpoint + "/health")
            models = client.get(endpoint + "/v1/models")
            expected = "miles-qwen3-smoke" if model_id == "mdl_smoke" and endpoint.endswith(":18080") else model_id
            return (
                health.status_code == 200
                and models.status_code == 200
                and any(m["id"] == expected for m in models.json()["data"])
            )
    except (httpx.HTTPError, ValueError, KeyError):
        return False


def ensure_server(store, model_id, job_id):
    if model_id == "mdl_smoke" and healthy("http://127.0.0.1:18080", model_id):
        return "http://127.0.0.1:18080"
    active = store.active()
    rows = store.rows("SELECT * FROM slots")
    for row in rows:
        endpoint = f"http://127.0.0.1:{row['port']}"
        if row["model_id"] == model_id and healthy(endpoint, model_id):
            return endpoint
    candidates = [
        name for name, port in SLOTS.items() if not active or active["endpoint"] != f"http://127.0.0.1:{port}"
    ]
    if hasattr(store, "registry"):
        # The global runner serializes allocation. Never evict another tenant's
        # active or standby engine, even when its local slots table is empty.
        owned = {r["id"]: r["tenant"] for r in store.registry.data.rows("SELECT * FROM engines")}
        candidates = [s for s in candidates if owned.get(s, store.tenant_id) == store.tenant_id]
        candidates.sort(key=lambda s: (s not in owned, s))
    if not candidates:
        raise RuntimeError("Serving capacity is full; the operator must add capacity before another deployment")
    slot = candidates[0]
    endpoint = f"http://127.0.0.1:{SLOTS[slot]}"
    # A retired slot may still be finishing requests from before the route switch.
    deadline = now() + 130
    while store.one(
        """SELECT COUNT(*) n FROM usage u JOIN deployments d ON d.id=u.deployment_id
                       WHERE d.endpoint=? AND u.ended IS NULL AND u.created>?""",
        (endpoint, now() - 130),
    )["n"]:
        if now() > deadline:
            raise RuntimeError("Retired deployment still has in-flight requests")
        phase(store, job_id, "等待旧版本请求完成")
        time.sleep(2)
    supervise("stop", "slot-" + slot, check=False)
    store.execute("INSERT OR REPLACE INTO slots VALUES (?,?,?)", (slot, model_id, SLOTS[slot]))
    if hasattr(store, "registry"):
        store.registry.data.execute(
            "INSERT OR REPLACE INTO engines VALUES (?,?,?,?)", (slot, store.tenant_id, model_id, SLOTS[slot])
        )
    supervise("start", "slot-" + slot)
    deadline = now() + 180
    while now() < deadline:
        phase(store, job_id, "等待候选模型通过健康检查")
        if healthy(endpoint, model_id):
            return endpoint
        time.sleep(3)
    raise RuntimeError("Candidate model did not become healthy; active deployment was preserved")


def evaluate(store, job):
    from miles_platform.rewards import grade

    model = store.one("SELECT * FROM models WHERE id=?", (job["model_id"],))
    dataset = store.one("SELECT * FROM datasets WHERE id=?", (model["dataset_id"],))
    endpoint = ensure_server(store, model["id"], job["id"])
    rows = [json.loads(line) for line in (Path(dataset["path"]) / "eval.jsonl").read_text().splitlines()]
    results = []
    with httpx.Client(timeout=120, trust_env=False) as client:
        for index, row in enumerate(rows):
            phase(store, job["id"], f"独立聊天 API 评测 · {index + 1}/{len(rows)}")
            name = "miles-qwen3-smoke" if endpoint.endswith(":18080") else model["id"]
            response = client.post(
                endpoint + "/v1/chat/completions",
                json={
                    "model": name,
                    "messages": row["prompt"],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            results.append(
                {
                    "index": index,
                    "correct": grade(choice["message"]["content"], row["label"]),
                    "finish_reason": choice["finish_reason"],
                    "usage": data.get("usage"),
                }
            )
    correct = sum(row["correct"] for row in results)
    threshold = store.setting("quality_threshold")
    passed = correct / len(rows) >= threshold and all(r["finish_reason"] == "stop" for r in results)
    store.execute(
        "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
        (
            uid("eval"),
            model["id"],
            dataset["id"],
            now(),
            correct,
            len(rows),
            threshold,
            int(passed),
            json.dumps(results),
        ),
    )
    phase(store, job["id"], f"评测完成 · {correct}/{len(rows)} · {'达到发布门槛' if passed else '未达到发布门槛'}")


def deploy(store, job):
    model = store.one("SELECT * FROM models WHERE id=?", (job["model_id"],))
    evaluation = store.one("SELECT * FROM evaluations WHERE model_id=? ORDER BY created DESC LIMIT 1", (model["id"],))
    if not model["validated"] or not evaluation or not evaluation["passed"]:
        raise RuntimeError("Model must pass artifact validation and its latest evaluation")
    endpoint = ensure_server(store, model["id"], job["id"])
    phase(store, job["id"], "健康检查通过 · 切换服务版本")
    deployment_id = uid("dep")
    action = "rollback" if json.loads(job["params"]).get("rollback") else "deploy"
    with store.connect(write=True) as db:
        if db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()[0] != "running":
            raise RuntimeError("Job cancelled before activation")
        db.execute("INSERT INTO deployments VALUES (?,?,?,?,?)", (deployment_id, model["id"], endpoint, now(), action))
        db.execute("INSERT OR REPLACE INTO settings VALUES ('active_deployment',?)", (json.dumps(deployment_id),))
        db.execute("UPDATE jobs SET status='succeeded',phase='版本切换完成',ended=? WHERE id=?", (now(), job["id"]))
    store.audit(action, model["id"])


def serve_slot(store, slot):
    row = store.one("SELECT * FROM slots WHERE id=?", (slot,))
    if not row:
        raise RuntimeError("No model assigned to this slot")
    model = store.one("SELECT * FROM models WHERE id=?", (row["model_id"],))
    command = [
        str(Path(sys.executable).parent / "sglang"),
        "serve",
        "--model-path",
        model["path"],
        "--served-model-name",
        model["id"],
        "--host",
        "127.0.0.1",
        "--port",
        str(row["port"]),
        "--base-gpu-id",
        "0",
        "--dtype",
        "bfloat16",
        "--context-length",
        "2048",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "4096",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--mem-fraction-static",
        "0.20",
        "--chunked-prefill-size",
        "1024",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--enable-metrics",
    ]
    os.execvpe(
        command[0],
        command,
        {**compute_env(), "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4", "NCCL_NVLS_ENABLE": "0"},
    )
