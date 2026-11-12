"""Read measured scalar events from Miles' retained rank-zero training logs.

The training logger is the source of truth. No estimates or missing values are
turned into measurements, and reading a snapshot never changes a running job.
"""

import json
import math
import re

from miles_platform.store import now

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EVENT = re.compile(
    r"^\(FSDPTrainRayActor pid=\d+\) \[[^\]\n]+ actor_cell\d+_rank00000\] "
    r"log_utils\.py:\d+ - (step|rollout) (\d+): (\{.*\})\s*$"
)
SCALAR = re.compile(r"['\"]([^'\"]+)['\"]:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)(?=\s*[,}])")
FIELDS = {
    "train/loss": "loss",
    "train/pg_loss": "policy_loss",
    "train/grad_norm": "grad_norm",
    "train/lr-pg_0": "learning_rate",
    "rollout/raw_reward_filtered": "reward",
    "rollout/response_lengths": "response_tokens",
    "rollout/truncated": "truncated_fraction",
}
MAX_SCAN = 16 * 1024 * 1024
LOG_TAIL = 64000


def parse_metrics(text, total):
    points, invalid = {}, False
    for line in text.splitlines(keepends=True):
        # A concurrent writer can leave its final line incomplete.
        if not line.endswith("\n"):
            continue
        match = EVENT.search(ANSI.sub("", line))
        if not match:
            continue
        kind, index, payload = match.groups()
        index = int(index)
        if not 0 <= index < total:
            continue
        values = dict(SCALAR.findall(payload))
        if kind == "step" and values.get("train/step") != str(index):
            continue
        sample = points.setdefault(index, {"step": index + 1, "completed": False})
        for key, field in FIELDS.items():
            if key in values:
                value = float(values[key])
                sample[field] = value if math.isfinite(value) else None
                invalid |= not math.isfinite(value)
        if kind == "step":
            sample["completed"] = True
    return [points[key] for key in sorted(points)], invalid


def snapshot(store, job):
    params = json.loads(job["params"])
    total = min(10, max(0, int(params.get("updates", 0))))
    path = store.root / "logs" / f"{job['id']}.log"
    text, logs, modified, truncated = "", "", None, False
    if path.exists():
        with path.open("rb") as stream:
            size = path.stat().st_size
            text = stream.read(MAX_SCAN).decode(errors="replace")
            stream.seek(max(0, size - LOG_TAIL))
            logs = ANSI.sub("", stream.read(LOG_TAIL).decode(errors="replace"))
            modified = path.stat().st_mtime
            truncated = size > MAX_SCAN
    samples, invalid = parse_metrics(text, total)
    elapsed = max(0, (job["ended"] or now()) - job["started"]) if job["started"] else 0
    evaluation = (
        store.one(
            "SELECT correct,total,passed FROM evaluations WHERE model_id=? ORDER BY created DESC LIMIT 1",
            (job["model_id"],),
        )
        if job["model_id"]
        else None
    )
    return {
        "job_id": job["id"],
        "total_updates": total,
        "completed_updates": sum(point["completed"] for point in samples),
        "samples": samples,
        "elapsed_seconds": elapsed,
        "log_updated_at": modified,
        "observed_at": now(),
        "scan_truncated": truncated,
        "invalid_metrics": invalid,
        "evaluation": evaluation,
        "logs": logs,
    }
