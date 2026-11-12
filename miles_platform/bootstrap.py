"""Import verified smoke artifacts without fabricating a new training run."""

import hashlib
import json
import secrets
from pathlib import Path

from miles_platform.store import now


def bootstrap(store):
    password = store.root / "admin-password"
    if not password.exists():
        with password.open("x") as output:
            password.chmod(0o600)
            output.write(secrets.token_urlsafe(24))
    if store.setting("admin_hash") is None:
        salt = secrets.token_hex(16)
        value = hashlib.scrypt(password.read_text().strip().encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
        store.set("admin_hash", {"salt": salt, "hash": value})
    for key, value in {
        "rates": {"gpu_hour": None, "input_million": None, "output_million": None},
        "quality_threshold": 0.375,
    }.items():
        if store.setting(key) is None:
            store.set(key, value)
    state = store.root.parent
    source = Path(__file__).resolve().parents[1] / "deployments/platform-smoke/validation.json"
    if not (state / "models/Qwen3-0.6B-platform-smoke/model.safetensors").exists() or not source.exists():
        return
    if store.one("SELECT id FROM models WHERE id='mdl_smoke'"):
        return
    report = json.loads(source.read_text())
    data = state / "data"
    checksum = hashlib.sha256((data / "train.jsonl").read_bytes() + (data / "eval.jsonl").read_bytes()).hexdigest()
    with store.connect(write=True) as db:
        db.execute(
            "INSERT OR IGNORE INTO datasets VALUES (?,?,?,?,?,?,?)",
            ("ds_gsm8k", "GSM8K · 工程验证集", str(data), 64, 8, now(), checksum),
        )
        db.execute(
            "INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "mdl_smoke",
                "Qwen3-0.6B · Smoke v1",
                str(state / "models/Qwen3-0.6B-platform-smoke"),
                "job_imported",
                "ds_gsm8k",
                now(),
                report["export"]["export_sha256"],
                1,
                json.dumps(
                    {"imported": True, "validation": report["export"], "quality_note": "原生 4/8；聊天接口 3/8"}
                ),
            ),
        )
        db.execute(
            """INSERT OR IGNORE INTO jobs(id,kind,name,status,phase,params,model_id,created,started,ended)
                    VALUES ('job_imported','train','首个 GPU 验证','succeeded','已导入历史记录',?, 'mdl_smoke',?,?,?)""",
            (json.dumps({"imported": True, "updates": 2, "dataset_id": "ds_gsm8k"}), now(), now(), now()),
        )
        db.execute(
            "INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "eval_imported",
                "mdl_smoke",
                "ds_gsm8k",
                now(),
                3,
                8,
                0.375,
                1,
                json.dumps(report["evaluation"]["openai_chat"]["results"]),
            ),
        )
        db.execute(
            "INSERT OR IGNORE INTO deployments VALUES (?,?,?,?,?)",
            ("dep_imported", "mdl_smoke", "http://127.0.0.1:18080", now(), "import"),
        )
        db.execute("INSERT OR IGNORE INTO settings VALUES ('active_deployment',?)", (json.dumps("dep_imported"),))
    store.audit("bootstrap", "Imported verified smoke artifacts; historical compute usage was not metered")
