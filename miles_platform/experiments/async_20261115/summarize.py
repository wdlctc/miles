"""Summarise synchronous and fully-async recipe runs from their retained logs.

Every number is read from a Miles log line; nothing is estimated. Seconds per
update is the median gap between consecutive optimizer-step lines, so it
excludes start-up, export and the first update.

Example:
    python summarize.py logs/sync.log logs/async.log > results.json
"""

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from miles_platform.telemetry import ANSI, EVENT, PERF, SCALAR

EVAL = re.compile(r"^\(RolloutExecutor pid=\d+\) \[[^\]\n]+ rollout_executor\] metrics\.py:\d+ - (eval) (\d+): (\{.*\})\s*$")
TIME = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) ")
KEYS = {
    "step": {
        "train/train_rollout_kl": "sampler_trainer_kl",
        "train/train_rollout_logprob_abs_diff": "sampler_trainer_logprob_abs_diff",
        "train/grad_norm": "grad_norm",
        "train/tis_clipfrac": "tis_clip_fraction",
    },
    "rollout": {"rollout/raw_reward_filtered": "train_reward", "rollout/response_lengths": "response_tokens"},
    "perf": {
        "rollout/fully_async/avg_staleness": "staleness",
        "rollout/fully_async/token_weighted_staleness": "token_weighted_staleness",
        "rollout/fully_async/avg_generation_version_span": "versions_spanned_while_generating",
        "rollout/weight_version/mixed_version_ratio": "mixed_version_fraction",
    },
}


def summarize(path):
    series, step_times, evaluations, extremes = {}, {}, {}, {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        match = EVENT.search(line) or PERF.search(line) or EVAL.search(line)
        if not match:
            continue
        kind, index, payload = match.groups()
        values = {key: float(value) for key, value in SCALAR.findall(payload)}
        if kind == "eval":
            # The last evaluation logged for an index is the one after that update.
            evaluations[int(index)] = next(v for k, v in values.items() if k.count("/") == 1 and "-" not in k)
            continue
        if kind == "step":
            step_times[int(index)] = datetime.strptime(TIME.search(line).group(1), "%Y-%m-%d %H:%M:%S.%f")
        for key, name in KEYS[kind].items():
            if key in values:
                series.setdefault(name, {})[int(index)] = values[key]
        for key, name in (
            ("rollout/fully_async/max_staleness", "max_staleness"),
            ("rollout/fully_async/stale_groups_filtered", "stale_groups_dropped"),
            ("rollout/fully_async/aborted_groups_filtered", "aborted_groups_dropped"),
        ):
            if key in values:
                extremes[name] = max(extremes.get(name, 0), values[key])
    ordered = [step_times[index] for index in sorted(step_times)]
    gaps = [(later - earlier).total_seconds() for earlier, later in zip(ordered, ordered[1:], strict=False)]
    exit_file = Path(path).with_suffix(".exit")
    return {
        "run": Path(path).stem,
        "updates": len(step_times),
        "seconds_per_update_median": statistics.median(gaps) if gaps else None,
        "seconds_per_update_min_max": [min(gaps), max(gaps)] if gaps else None,
        "launcher": exit_file.read_text().strip() if exit_file.exists() else None,
        "mean": {name: statistics.fmean(points.values()) for name, points in sorted(series.items())},
        **extremes,
        "eval_accuracy": [evaluations[index] for index in sorted(evaluations)],
        "per_update": {name: [points[index] for index in sorted(points)] for name, points in sorted(series.items())},
    }


if __name__ == "__main__":
    print(json.dumps([summarize(path) for path in sys.argv[1:]], indent=2))
