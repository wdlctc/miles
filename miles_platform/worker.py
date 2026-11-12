"""Durable single-runner queue; a worker restart adopts the supervised runner."""

import fcntl
import os
import time
import traceback

from miles_platform.runtime import SLOTS, deploy, evaluate, stop_training, supervise, train
from miles_platform.store import now


def run_job(store):
    job = store.one("SELECT * FROM jobs WHERE status='running' ORDER BY created LIMIT 1")
    if not job:
        return
    directory = store.root / "logs"
    directory.mkdir(exist_ok=True)
    file = directory / f"{job['id']}.log"
    with file.open("a", buffering=1) as log:
        # Child processes inherit these descriptors, so all job output is retained.
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            {"train": train, "evaluate": evaluate, "deploy": deploy}[job["kind"]](store, job)
            store.execute(
                "UPDATE jobs SET status='succeeded',ended=? WHERE id=? AND status='running'", (now(), job["id"])
            )
        except Exception as exc:
            traceback.print_exc()
            store.execute(
                "UPDATE jobs SET status='failed',error=?,ended=? WHERE id=? AND status='running'",
                (str(exc)[:500], now(), job["id"]),
            )


def tick(store):
    store.set("worker_heartbeat", now())
    active = store.one("SELECT * FROM jobs WHERE status IN ('running','cancel_requested') ORDER BY created LIMIT 1")
    status = supervise("status", "runner", check=False)
    live = "RUNNING" in status or "STARTING" in status or "STOPPING" in status
    if active:
        if active["status"] == "cancel_requested":
            supervise("stop", "runner", check=False)
            if active["kind"] == "train":
                stop_training()
            # A completed phase may have spawned a standby server; it is reusable,
            # supervised serving capacity and never a surviving Ray training worker.
            seconds = active["gpu_seconds"]
            if active["kind"] == "train" and not seconds and active["training_started"]:
                seconds = max(0, now() - active["training_started"]) * 3
            cost = seconds / 3600 * active["rate"] if active["rate"] is not None else None
            store.execute(
                "UPDATE jobs SET status='cancelled',phase='已取消并释放训练资源',ended=?,gpu_seconds=?,cost=?,training_ended=CASE WHEN training_started IS NOT NULL THEN COALESCE(training_ended,?) ELSE NULL END WHERE id=?",
                (now(), seconds, cost, now(), active["id"]),
            )
        elif not live:
            if active["kind"] == "train":
                stop_training()
            if active["training_started"] and not active["training_ended"]:
                seconds = max(0, now() - active["training_started"]) * 3
                cost = seconds / 3600 * active["rate"] if active["rate"] is not None else None
                store.execute(
                    "UPDATE jobs SET gpu_seconds=?,cost=?,training_ended=? WHERE id=?",
                    (seconds, cost, now(), active["id"]),
                )
            store.execute(
                "UPDATE jobs SET status='failed',error='Runner exited unexpectedly; inspect retained logs',ended=? WHERE id=?",
                (now(), active["id"]),
            )
        return
    if live:
        return
    with store.connect(write=True) as db:
        job = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not job:
            return
        db.execute(
            "UPDATE jobs SET status='running',phase='正在启动',started=? WHERE id=? AND status='queued'",
            (now(), job["id"]),
        )
    try:
        supervise("start", "runner")
    except Exception as exc:
        store.execute(
            "UPDATE jobs SET status='failed',error=?,ended=? WHERE id=? AND status='running'",
            (str(exc)[:500], now(), job["id"]),
        )


def work(store):
    with (store.root / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = store.active()
        for slot, port in SLOTS.items():
            if active and active["endpoint"] == f"http://127.0.0.1:{port}":
                status = supervise("status", "slot-" + slot, check=False)
                if "RUNNING" not in status and "STARTING" not in status:
                    supervise("start", "slot-" + slot)
        while True:
            try:
                tick(store)
            except Exception:
                traceback.print_exc()
            time.sleep(2)
