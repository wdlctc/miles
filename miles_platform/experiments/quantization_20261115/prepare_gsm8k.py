"""Write disjoint GSM8K calibration and held-out JSONL for miles_platform.quantize.

Calibration rows come from the train split and held-out rows from the test
split, each rendered later as a user question plus the reference solution.

Example:
    python prepare_gsm8k.py --output-dir /work/quant --calibration 512 --heldout 128
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REVISION = "740312add88f781978c0658806c59bc2815b9866"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--heldout", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    questions = {}
    for name, split, count in (("calibration", "train", args.calibration), ("heldout", "test", args.heldout)):
        rows = load_dataset("openai/gsm8k", "main", split=split, revision=REVISION).select(range(count))
        questions[name] = {row["question"] for row in rows}
        with (output / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                prompt = [{"role": "user", "content": row["question"]}]
                stream.write(json.dumps({"prompt": prompt, "response": row["answer"]}) + "\n")
    overlap = questions["calibration"] & questions["heldout"]
    if overlap:
        raise SystemExit(f"{len(overlap)} questions appear in both files")
    print(json.dumps({"revision": REVISION, **{name: len(items) for name, items in questions.items()}}))


if __name__ == "__main__":
    main()
