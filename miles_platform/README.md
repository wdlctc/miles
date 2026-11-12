# Miles Platform console

A working single-workspace control plane for this fork, with a Chinese console,
durable job queue, model registry, evaluation, deployment switching, API keys and
metering. It runs on the authorized GPU host alongside the first
[training-to-serving experiment](../deployments/platform-smoke/README.md).
Baseten is a product reference; this app does not call Baseten cloud services
and is not affiliated with Baseten or RadixArk.

## What works

| Page | Behavior |
| --- | --- |
| Overview | Actual jobs, inference usage, active deployment and live NVIDIA GPU metrics |
| Training | Create bounded Qwen3-0.6B FSDP GRPO jobs, follow logs, cancel queued/running jobs |
| Models | Immutable export directories, SHA-256, shape/key/finite-weight checks and independent evaluation history |
| Deployments | Start a candidate, verify its model identity and health, atomically switch the stable alias, roll back |
| Playground | Real streaming responses through the metered inference gateway |
| API keys | Show the secret once, store only a hash, enforce per-key requests/minute, revoke keys |
| Usage | Input/output token counts, allocated training GPU time, optional price snapshots and CSV export |
| Datasets | Upload math-task JSONL, validate messages/labels and reject exact duplicate or overlapping prompts |

The tested training recipe uses GPU 1–3 (two FSDP trainers, one rollout engine).
GPU 0 serves the stable model and candidate slots. Training automatically exports
and evaluates its model; deployment is a separate operator action. The first
console-submitted run completed training, export and chat evaluation with 4/8
correct answers. This tiny evaluation establishes integration behavior, not
model quality. The machine-readable acceptance record is
[acceptance.json](acceptance.json).

## Start on the provisioned server

Requirements: Linux, the existing working Miles/SGLang GPU environment,
four A100 GPUs, Supervisor and the Vast logging helpers. This is a single-host
adapter, not a Kubernetes or multi-tenant scheduler. It owns the local Ray
cluster; do not run unrelated Ray workloads or GPU jobs on GPUs 1–3 concurrently.

```bash
cd /workspace/miles-platform
source /workspace/miles-platform-venv/bin/activate
export MILES_PLATFORM_STATE=/workspace/miles-platform-state
python -m miles_platform.provision
```

The provisioner imports the previously verified smoke model when present,
creates a random admin password, and installs only the platform's own services:

| Supervisor name | Role | Port |
| --- | --- | --- |
| `miles-platform-console` | Console, admin API and inference gateway | 127.0.0.1:18090 |
| `miles-platform-worker` | Persistent queue coordination | None |
| `miles-platform-runner` | One supervised job at a time | None |
| `miles-platform-slot-a` | Candidate/active inference server | 127.0.0.1:18081 |
| `miles-platform-slot-b` | Alternate inference server | 127.0.0.1:18082 |

The original `miles-platform-inference` on port 18080 remains available as the
imported version. The existing Jupyter and provider management services are
untouched. The new service ports bind to loopback; reach them privately:

```bash
ssh -N -L 127.0.0.1:8081:127.0.0.1:18090 -p SSH_PORT SSH_USER@SSH_HOST
```

Open `http://localhost:8081/`. The admin password is stored in
`$MILES_PLATFORM_STATE/control/admin-password` with mode 0600. Do not commit it
or the runtime database. Admin sessions use HttpOnly, SameSite=Strict cookies
and expire after eight hours; the supported access is HTTP over a private SSH
tunnel. Use TLS and Secure cookies before introducing a public gateway.

The standalone admin dependencies are pinned in `requirements.txt`; the GPU
runner also needs the environment documented for the smoke experiment. No
frontend package installation or build step is required.

## Operator workflow

1. Optionally upload disjoint training/evaluation JSONL in Datasets. Each row
   contains `prompt` (text chat messages) and `label` (the math answer). Request
   a final `\\boxed{}` answer in the prompt. The built-in GSM8K subset is ready.
2. Create a training task with 1–10 updates and a learning rate from 1e-7 to
   1e-5. The process has a 20-minute training timeout; there is one job runner.
3. Inspect the task log, registered model and independent evaluation. The
   initial demonstration gate is accuracy >= 37.5% and no truncated responses.
   This threshold is a prototype setting, not a business acceptance standard.
4. Deploy the model. The worker verifies candidate `/health` and `/v1/models`
   before changing the route. In-flight gateway requests keep their selected
   deployment. A retired slot waits for tracked requests before it is reused.
5. Create an API key and call the stable model alias `miles-default`. Revoke a
   key when no longer needed. In-flight requests are allowed to finish.
6. Configure your own USD unit rates under Usage, if desired. Export the ledger
   as CSV. No payment collection, credit purchase or automatic charging occurs.

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer $MILES_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"miles-default","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":64,"stream":true}'
```

Supported request fields are text `messages`, `model`, `temperature`, `top_p`,
`max_tokens` (1–512), `stream` and `stream_options`. The underlying context
window is 2,048 tokens; unsupported fields and oversized message payloads are
rejected. Only the public inference routes accept API keys; an API key cannot
administer jobs, deployments or other keys. Playground requires an admin session.

## APIs and persistence

The interactive API schema is at `/api/docs`. Admin routes require the login
cookie and reject cross-origin mutations. Examples of the implemented resources:

- `/api/jobs`, `/api/jobs/{id}/cancel`, `/api/jobs/{id}/logs`
- `/api/models/{id}/evaluate`, `/api/models/{id}/deploy`
- `/api/deployments/{id}/rollback`
- `/api/datasets`, `/api/keys`, `/api/rates`, `/api/usage.csv`
- `/api/state`, `/api/runtime`
- `/v1/models`, `/v1/chat/completions`

Job creation/evaluation/deployment accept `Idempotency-Key`: repeating the same
key and parameters returns the original job; a conflicting payload returns 409.
SQLite WAL transactions own job claims, key-rate reservations and route changes.
The worker adopts a still-running Supervisor runner after restart. If the runner
exits without recording completion, the worker marks the job failed and stops
its training Ray processes. Cancellation is recorded only after stopping the
runner and cleaning up training. It does not discard retained artifacts/logs.

State is under `$MILES_PLATFORM_STATE/control`: `platform.sqlite3`, logs,
datasets, run directories and the admin bootstrap password. Weight directories
are under `$MILES_PLATFORM_STATE/models`. Service logs are at
`/var/log/miles-platform-*.log`.

The metadata database survives service restarts. The host currently has no
persistent volume, so instance destruction/recycle removes both metadata and
weights. For backups, use SQLite's backup API rather than copying a live WAL
file alone, and copy selected immutable weight directories off-host. Treat the
database backup as sensitive: it includes authentication hashes and sessions.

## Metering semantics

Inference usage comes from the upstream response, including the final usage
chunk for streaming. On disconnect or missing usage, mark the request incomplete
instead of inventing counts or costs. An API-process restart marks its old
in-flight requests as interrupted (599). The request ledger includes failed
requests and records the deployment chosen when each request began.

Training GPU seconds are three allocated GPUs multiplied by training execution
time, including Ray initialization/cleanup. They are not hardware-utilization
samples. Rate settings are copied onto each request/job when it starts; editing
rates never changes old records. Unconfigured rates produce null costs, not zero
charges. Estimates exclude idle inference GPU rental, storage, network and taxes;
they are not invoices or a profitability calculation. The imported historical
smoke job has no metered compute duration and is excluded from compute totals.

## Validation and limits

```bash
python -m pytest tests/fast/platform -q
node --check miles_platform/static/app.js
```

CPU tests cover login/origin protections, secret hashing/revocation, transactional
rate limits, streaming/non-streaming usage, price snapshots, dataset leakage,
idempotency, restart adoption, cancellation and preserving the active route when
a candidate is unhealthy. GPU acceptance separately exercises real training,
export/evaluation, deployment/rollback, cancellation and inference continuity.

This release supports one trusted operator workspace and one tested recipe.
It does not provide customer multi-tenancy, arbitrary customer code execution,
durable off-host artifact storage, an HA database, payment processing, or an
autoscaling service. Evaluation is sequential and small; no accuracy or latency
SLA is implied. Next production work is isolation/RBAC, object storage,
project-aware accounting, a durable distributed queue and broader quality gates.
