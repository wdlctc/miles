# GPU training-to-serving prototype

This fork's first engineering milestone ran on 2026-11-12 UTC. It connects
Miles FSDP GRPO, checkpoint export, and a separate SGLang inference service.
It is a small integration experiment, not a model-quality benchmark or a
customer-ready platform. Product scope remains in [PLATFORM.md](../../PLATFORM.md).

## Observed result

- Four A100-SXM4-40GB GPUs: two FSDP trainers and two rollout engines. Two
  updates completed successfully, and Ray released its GPUs after training.
- Qwen3-0.6B, a pool of 64 GSM8K training rows, and eight disjoint test rows.
  Each update generated four responses for each of four prompts: 32 training
  responses in total. This did not train over all 64 rows.
- Final checkpoint exported successfully. All 311 parameter entries match
  after FP32-to-BF16 conversion. The export stores 310 unique tensors because
  `lm_head.weight` is tied to `model.embed_tokens.weight`. All weights are finite;
  247 stored tensors differ from the base model.
- Independent serving passed health, model listing, ordinary chat and streaming
  chat requests through an SSH tunnel. It uses GPU 0, approximately 3 GiB VRAM
  in this configuration, and listens only on loopback.
- Training evaluation was 3/8 before training, 3/8 after update one, and 4/8
  after update two. Exported-model native `/generate` evaluation was 4/8;
  `/v1/chat/completions` evaluation was 3/8. All requests stopped normally.
  Different request paths and concurrency were used; the cause is unresolved.
  Do not claim quality improvement or interface parity from these numbers.

Machine-readable evidence is in [validation.json](validation.json), source
revisions in [sources.json](sources.json), and the observed package inventory
in [environment.json](environment.json). The latter records the working
environment; it is not a portable lockfile or a complete image specification.
No customer data, server credentials, or model weights are included in Git.

## Environment

The tested host is an existing unprivileged GPU container. Its driver supports
CUDA 13.2; PyTorch uses CUDA 13.0. Installation used a separate Python 3.12 venv,
without changing the host driver, its main Python environment, or its Jupyter
service. Read the provider's operating guide before adapting these steps.

The installation sequence used was:

1. Create `/workspace/miles-platform-venv` and install `sglang==0.5.23` plus
   this checkout's `requirements.txt`, with [constraints.txt](constraints.txt).
   The recorded versions include Torch 2.14.1, Transformers 5.19.0 and Ray 2.58.1.
2. Check out `https://github.com/sgl-project/sglang.git` at
   `18a44326f6de5134fe921927d760d81f8ee48fc9` from the Miles SGLang branch.
   Before replacing the wheel install, copy its `sglang/**/*.so` files to the
   same relative paths under the checkout's `python/sglang` directory. This
   preserves the wheel's Rust extensions, including the cache-tree extension.
3. Install the SGLang checkout's `python` and `radix_transfer_plugins` packages,
   plus this Miles checkout, editable with `--no-deps`. Set
   `SGLANG_BUILD_RUST_EXTS=none` for this install. This source/wheel pairing was
   verified by the run; changing either version requires fresh validation.
4. Install these pinned wheels with `--no-deps` from the
   [Miles wheel release](https://github.com/radixark/miles-wheels/releases/tag/cu130-torch214-x86_64):
   `flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl` and
   `sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl`.

The host was provisioned by these steps, not by a reproducible container build.
Before expanding this into a platform, replace the installation procedure with
a tested, digest-pinned image. Optional Megatron dependencies were not installed
for this FSDP experiment.

## Prepare and train

On the provisioned host, use the isolated environment and a fresh output path:

```bash
cd /workspace/miles-platform
source /workspace/miles-platform-venv/bin/activate
export HF_HOME=/workspace/miles-platform-state/hf-cache
export PLATFORM_STATE=/workspace/miles-platform-state
```

The following preparation uses the revisions recorded in `sources.json` and
retains the dataset's own chat prompts and labels:

```python
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

state = Path(os.environ["PLATFORM_STATE"])
sources = json.loads(Path("deployments/platform-smoke/sources.json").read_text())
snapshot_download(
    sources["model"], revision=sources["model_revision"],
    local_dir=state / "models" / "Qwen3-0.6B",
)
(state / "data").mkdir(parents=True, exist_ok=True)
for split, target, count in [
    ("train", "train", sources["training_rows"]),
    ("test", "eval", sources["evaluation_rows"]),
]:
    file = hf_hub_download(
        sources["dataset"], f"{split}.parquet", repo_type="dataset",
        revision=sources["dataset_revision"],
    )
    rows = pq.read_table(file).slice(0, count).to_pylist()
    with (state / "data" / f"{target}.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps({"prompt": row["messages"], "label": str(row["label"])}) + "\n")
```

The launcher starts and stops a local Ray cluster through Miles' existing
command utilities. Use an idle node: rerunning the four-GPU recipe requires
stopping this prototype's inference service first. Check for other workloads
before starting; the launcher manages the node's Ray processes.

```bash
supervisorctl stop miles-platform-inference
timeout --signal=TERM --kill-after=90s 20m \
  python scripts/run_qwen3_0_6b_platform.py \
    --model-dir "$PLATFORM_STATE/models" \
    --data-dir "$PLATFORM_STATE/data" \
    --output-dir "$PLATFORM_STATE/runs/qwen3-next"
```

Defaults and supported overrides belong to the
[launcher](../../scripts/run_qwen3_0_6b_platform.py). It rejects an output
directory that already contains checkpoints. The initial observed run used a
one-off command assembled with the same Miles command utilities; the committed
launcher generalizes its paths and knobs and explicitly passes the node GPU
count. Its generated command is covered by the committed snapshot.

## Export and serve

Select the completed checkpoint deliberately. The following example is for a
new two-update run; do not replace the model directory of a running server:

```bash
python tools/convert_fsdp_to_hf.py \
  --input-dir "$PLATFORM_STATE/runs/qwen3-next/checkpoints/iter_0000002" \
  --output-dir "$PLATFORM_STATE/models/Qwen3-0.6B-next" \
  --origin-hf-dir "$PLATFORM_STATE/models/Qwen3-0.6B"
```

Check the converter's matched, missing and unexpected parameter keys; validate
shapes, finite values and weights against the selected checkpoint after dtype
conversion. Preserve the shared output/input embedding alias when comparing
keys. Record the exported weight digest and evaluate before treating it as a
publishable model. The initial smoke model is served with this command:

```bash
OMP_NUM_THREADS=4 NCCL_NVLS_ENABLE=0 sglang serve \
  --model-path "$PLATFORM_STATE/models/Qwen3-0.6B-platform-smoke" \
  --served-model-name miles-qwen3-smoke \
  --host 127.0.0.1 --port 18080 --base-gpu-id 0 \
  --dtype bfloat16 --context-length 2048 \
  --max-running-requests 8 --max-total-tokens 4096 \
  --attention-backend triton --sampling-backend pytorch \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --mem-fraction-static 0.30 --chunked-prefill-size 1024 \
  --default-chat-template-kwargs '{"enable_thinking":false}' --enable-metrics
```

On the provisioned host this process is already managed by Supervisor as
`miles-platform-inference`; do not start a second copy. Its wrapper is
`/opt/supervisor-scripts/miles-platform-inference.sh` and its configuration is
`/etc/supervisor/conf.d/miles-platform-inference.conf`. The wrapper sources the
provider logging/environment helpers before enabling strict shell options.
Use `supervisorctl status|start|stop|restart miles-platform-inference` to manage
it. Logs are at `/var/log/miles-platform-inference.log`. The one-time training
job `miles-platform-smoke` has autostart disabled; its log is at
`/var/log/miles-platform-smoke.log` and its artifacts remain under
`$PLATFORM_STATE/runs/qwen3-smoke`.

## Access

Remote port 8080 belongs to the existing Jupyter service. Forward a local port
to the inference service's remote loopback port instead:

```bash
ssh -N -L 127.0.0.1:8080:127.0.0.1:18080 -p SSH_PORT SSH_USER@SSH_HOST
```

Open `http://localhost:8080/docs`, or call:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"miles-qwen3-smoke","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":64}'
```

The OpenAI-compatible base URL is `http://localhost:8080/v1`; model name is
`miles-qwen3-smoke`. This private prototype has no platform authentication,
tenant isolation, billing or deployment controller. Keep it behind SSH until
those services exist. These weights remain on the server's container disk,
which is lost on instance recycle or destruction; move selected model versions
to durable object storage before using them for customer delivery.

## Code validation and next work

The new launcher's two snapshot tests pass, as do Black and Ruff checks.
The required full launcher suite produced 661 passes, 35 failures and four
errors. A clean archive of the baseline produced 659 passes and the identical
35 failure/four error node IDs in the same environment. Existing failures
include optional `megatron`/`tap` imports and upstream snapshot discrepancies;
the full suite is therefore not green. See [validation.json](validation.json).

Next engineering work is persistent job/model metadata, job creation and
cancellation, independent evaluation with matched inference settings, durable
artifact storage, and a versioned deployment gateway. The running endpoint is
an engineering demonstration until those acceptance criteria are met.
