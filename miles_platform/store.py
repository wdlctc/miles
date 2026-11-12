"""SQLite owns job, artifact, deployment and metering state on one host."""

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def now():
    return time.time()


def uid(prefix):
    return f"{prefix}_{secrets.token_hex(6)}"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "platform.sqlite3"
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, name TEXT, path TEXT,
                    train_rows INTEGER, eval_rows INTEGER, created REAL, sha256 TEXT);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, name TEXT, status TEXT,
                    phase TEXT, params TEXT, model_id TEXT, created REAL, started REAL, ended REAL,
                    error TEXT, gpu_seconds REAL DEFAULT 0, cost REAL, rate REAL,
                    idem TEXT UNIQUE, request_hash TEXT);
                CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, name TEXT, path TEXT UNIQUE,
                    job_id TEXT, dataset_id TEXT, created REAL, sha256 TEXT, validated INTEGER, metadata TEXT);
                CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, model_id TEXT, dataset_id TEXT,
                    created REAL, correct INTEGER, total INTEGER, threshold REAL, passed INTEGER, results TEXT);
                CREATE TABLE IF NOT EXISTS slots (id TEXT PRIMARY KEY, model_id TEXT, port INTEGER);
                CREATE TABLE IF NOT EXISTS deployments (id TEXT PRIMARY KEY, model_id TEXT, endpoint TEXT,
                    created REAL, action TEXT);
                CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, name TEXT, hash TEXT UNIQUE,
                    prefix TEXT, created REAL, revoked REAL, rpm INTEGER);
                CREATE TABLE IF NOT EXISTS usage (id TEXT PRIMARY KEY, key_id TEXT, deployment_id TEXT,
                    created REAL, ended REAL, status INTEGER, prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0, complete INTEGER DEFAULT 0, cost REAL,
                    input_rate REAL, output_rate REAL);
                CREATE INDEX IF NOT EXISTS usage_rate ON usage(key_id, created);
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, created REAL, action TEXT, detail TEXT);
            """
            )
        os.chmod(self.path, 0o600)
        with self.connect(write=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("training_started", "training_ended"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} REAL")

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args)]

    def one(self, sql, args=()):
        rows = self.rows(sql, args)
        return rows[0] if rows else None

    def execute(self, sql, args=()):
        with self.connect(write=True) as db:
            db.execute(sql, args)

    def setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))

    def audit(self, action, detail):
        self.execute("INSERT INTO audit(created,action,detail) VALUES (?,?,?)", (now(), action, detail))

    def active(self):
        return self.one(
            "SELECT d.*,m.name FROM deployments d JOIN models m ON m.id=d.model_id WHERE d.id=?",
            (self.setting("active_deployment"),),
        )

    def enqueue(self, kind, name, params, model_id=None, idem=None):
        # Import here to keep storage construction independent of API/billing.
        from miles_platform import billing

        billing.prepare(self)
        rates = billing.pricing(self)
        request_hash = digest(json.dumps([kind, name, params, model_id], sort_keys=True))
        with self.connect(write=True) as db:
            if idem:
                old = db.execute("SELECT * FROM jobs WHERE idem=?", (idem,)).fetchone()
                if old:
                    if old["request_hash"] != request_hash:
                        raise ValueError("Idempotency key was already used with different parameters")
                    return dict(old)
            job_id = uid("job")
            if (
                db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancel_requested')"
                ).fetchone()[0]
                >= 3
            ):
                raise ValueError("Workspace queue quota reached (3 active jobs)")
            models = db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            pending = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind='train' AND status IN ('queued','running','cancel_requested')"
            ).fetchone()[0]
            if models + pending >= 10 and kind == "train":
                raise ValueError("Workspace model quota reached (10); contact the operator")
            if not getattr(self, "exempt", True) and kind != "train":
                if (
                    billing.balance(db)["available_microusd"] <= 0
                    or db.execute("SELECT 1 FROM orders WHERE disputed=1 LIMIT 1").fetchone()
                ):
                    from fastapi import HTTPException

                    raise HTTPException(402, "Add prepaid credits before evaluating or deploying")
            if kind == "train":
                billing.reserve(db, self, job_id, "training", billing.training_quote(self, rates))
            db.execute(
                """INSERT INTO jobs(id,kind,name,status,phase,params,model_id,created,idem,request_hash)
                          VALUES (?,?,?,'queued','等待执行',?,?,?,?,?)""",
                (job_id, kind, name, json.dumps(params), model_id, now(), idem, request_hash),
            )
            if kind == "train":
                db.execute("UPDATE jobs SET rate=? WHERE id=?", (rates.get("gpu_hour"), job_id))
            return dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def configured_store():
    return Store(Path(os.environ.get("MILES_PLATFORM_STATE", "/workspace/miles-platform-state")) / "control")
