# Miles Platform customer beta

This fork combines Miles post-training with a self-hosted console and SGLang
inference gateway. Baseten is a product reference, not a connected hosting
service. The platform is independently operated.

The customer release adds invite-only accounts, separate workspaces, roles,
prepaid USD accounting and Stripe hosted Checkout. **Live payments and public
customer launch are not activated.** The owner has not yet supplied a domain,
a Stripe merchant account or durable storage. Payment contract tests mock
Stripe HTTP responses; they are not evidence of payment on Stripe's network.

## Implemented behavior

| Area | Behavior |
| --- | --- |
| Accounts | Email/password, one-time invitations, password changes/recovery, eight-hour sessions |
| Workspaces | Separate database, datasets, models, API keys, usage, ledger and deployment route |
| Roles | Owner: billing/team; Admin: deployment/keys/invites; Member: tasks/data/playground; Viewer: read-only |
| Operator | Create/suspend workspaces, set platform prices, inspect host GPU metrics |
| Training | Bounded Qwen3-0.6B FSDP GRPO, queue, logs, cancellation and restart adoption |
| Models | Export, tensor checks, SHA-256, numeric evaluation and a publication gate |
| Serving | Workspace-owned slots, health checks, stable alias, rollback, streaming and rate limits |
| Billing | Integer-microUSD credit ledger, atomic reservations/settlement, orders and usage CSV |
| Stripe | Hosted Checkout, signed webhooks, canonical payment validation, replay protection, refunds/disputes |

This is application-level isolation on **one trusted GPU host**, not a VM or
container boundary for arbitrary customer code. Customers submit text datasets
and bounded parameters only, never executable code, reward functions, model
binaries or arbitrary commands. Numeric grading does not use `eval` or symbolic
parsing. The console is primarily Chinese with bilingual account/billing labels;
a fully English UI remains future work.

## Deploy and migrate

The supported target is the existing Linux/Vast environment with four A100 GPUs,
Miles/SGLang, Supervisor and Vast logging helpers. Run **one API process and one
worker**. The runner owns Ray and GPUs 1–3; avoid other workloads there. GPU 0
serves models.

```bash
cd /workspace/miles-platform
source /workspace/miles-platform-venv/bin/activate
export MILES_PLATFORM_STATE=/workspace/miles-platform-state
python -m miles_platform.provision
```

Only platform services are managed: `miles-platform-console` on 127.0.0.1:18090,
`miles-platform-worker`, `miles-platform-runner`, and slots a–d on 18081–18084.
The imported server on 18080 remains owned by the default workspace. Four slots
bound resident model capacity; a full pool rejects a candidate without evicting
another tenant's model. Jupyter and provider management services stay unchanged.

Before upgrading, wait for jobs, stop the platform API/worker, back up metadata
and retain the previous Git revision. Migration is additive: the original
workspace becomes `default`; login is **owner** plus the existing password from
`control/admin-password` (0600). Existing resource IDs and API keys remain valid;
browser sessions require login again. The default workspace is explicitly
exempt from credit deductions; new customer workspaces are prepaid.

```bash
ssh -N -L 127.0.0.1:8081:127.0.0.1:18090 -p SSH_PORT SSH_USER@SSH_HOST
```

Open `http://localhost:8081/`. Retain private SSH access until public launch
requirements are met. Never expose raw Ray, SGLang or root access to customers.
The API schema is available after login at `/api/docs`. Standalone dependencies
are pinned in `requirements.txt`; the GPU runner needs the existing smoke-run
environment. The frontend has no build step.

## Onboard customers

1. In Customers, create a workspace using the customer's email.
2. Deliver the one-time invitation through a trusted channel. It expires after
   24 hours and is stored only as a hash. This app does not send email.
3. A new user sets a password of at least 12 characters; an existing account must
   enter its existing password to join another workspace.
4. Set platform prices in Usage before paid compute: USD/GPU-hour and
   USD/million input/output tokens. Customers cannot edit prices. Requests/jobs
   keep the accepted price snapshot.
5. The customer uploads disjoint JSONL, buys credits once Stripe is configured,
   trains/evaluates/deploys a model, then creates an API key.

Each JSONL row has `prompt` (text chat messages) and `label` (number, decimal or
fraction). Ask for a final `\\boxed{}` answer. All other fields are discarded.
The fixed recipe accepts 1–10 updates and has a 20-minute training timeout.
The 37.5% demonstration evaluation gate is not a business-quality guarantee.

Current workspace quotas: 3 active jobs, 10 registered trained models,
20 datasets (5 MB train + 1 MB eval each), 20 active keys, 50 members/invitations,
2 concurrent inference requests, at most 120 requests/minute/key,
2,048-token context and 512 output tokens. Retention/deletion and capacity
increases are operator-managed. The runner alternates between queued tenants.
Suspension blocks new requests and cancels queued/running jobs. Accepted
inference calls may finish. Keys are workspace credentials shared by the team;
removing a member does not automatically revoke those shared keys.

After verifying a customer's identity, an operator can issue a recovery link:

```bash
python -m miles_platform.operations reset-link --email customer@example.com --output /private/reset-link.txt
```

The private output file is 0600; deliver it securely and remove it after use.
The link expires in one hour, works once, and revokes existing login sessions.

## Stripe setup

Keep credentials in the server environment, never in Git/chat. On Vast, the
existing Supervisor helper loads `/workspace/.env`; protect that file and
restart the platform API/worker after changes. Use
[production.env.example](production.env.example) as a configuration reference.
`python -m miles_platform.operations doctor` reports missing settings without keys.

1. Begin with a Stripe sandbox/test account, `STRIPE_MODE=test`, a `sk_test_…`
   secret and its webhook signing secret. For private testing, Stripe CLI can
   forward events to `http://127.0.0.1:18090/api/stripe/webhook` on the server.
2. Configure snapshot webhook API version **2026-10-28.endive** and subscribe to
   `checkout.session.completed`, `checkout.session.async_payment_succeeded`,
   `charge.refunded`, `charge.dispute.created`, `charge.dispute.updated`,
   `charge.dispute.closed`.
3. Exercise real sandbox Checkout, declines, duplicate/delayed events, refunds
   and recovery. **This has not yet been possible without merchant credentials.**
4. Use new workspaces for live customers: a workspace with test orders cannot
   switch to live credit. Test money must never be promoted to real balances.

Server-owned packs provide $25, $100 or $500 USD compute credit. Clients cannot
set the amount, currency, tenant or redirect destination. Checkout uses a stored
order and a Stripe idempotency key. Card details stay on Stripe; invoices are
requested from Stripe. Tax/legal setup remains the merchant's responsibility.
Automatic Tax is configurable and defaults to disabled.

Returning to the success page **never credits a wallet**. Webhooks verify raw
body signatures and a five-minute timestamp tolerance, then retrieve canonical
session/payment/charge data. Customer, tenant/order metadata, subtotal, currency,
mode and captured total must match. Event deduplication, net credited amount and
ledger entry commit atomically. Reconciliation failures return non-2xx for retry.
Monitor failed Stripe deliveries and replay them after fixing the cause.

Refunds are issued by the operator in Stripe Dashboard, then reconciled by
webhooks. Partial refunds reverse the credit component, excluding tax.
Duplicate/out-of-order events cannot double-credit. Open disputes pause paid
compute; lost disputes deduct disputed credit once, won disputes lift the hold.
Refunds after consumption can create a negative balance, which blocks new use.

Training reserves at most 3 GPUs × 22 minutes at the accepted rate, including
cleanup allowance. Inference reserves the engine context/output upper bound.
Settlement uses observed usage, rounds up to microUSD and never exceeds the
reservation; unused funds are released. Unknown usage is not invented or billed.
Restart reconciliation settles terminal records once. Evaluation and resident
hosting are included within beta quotas. This is prepaid compute, not recurring
subscriptions, automatic recharge or unlimited hosting.

Official references: [Checkout Sessions](https://docs.stripe.com/api/checkout/sessions/create),
[webhooks](https://docs.stripe.com/webhooks),
[fulfillment](https://docs.stripe.com/checkout/fulfillment.md?ui=full-page).

## State and recovery

The original workspace database is `control/platform.sqlite3`. Identity,
memberships, invitations, sessions and engine ownership are in
`control/identity/platform.sqlite3`. New tenant state lives under
`control/workspaces/<server-generated-id>/`; each has its own control database,
models, datasets and logs. Authenticated membership or a valid bearer key selects
the workspace before resource queries. Tenant headers never grant access.

SQLite WAL supports this single-host topology only: do not share databases over
NFS or start multiple API/worker replicas. A durable local block volume can
support the beta; PostgreSQL/object-storage/HA adapters remain future work.
**The current host has no persistent workspace volume: recycle/destroy can erase
its state.** A same-host backup is not durable protection.

After jobs finish, stop the platform API/worker for a maintenance backup:

```bash
python -m miles_platform.operations backup-metadata --destination /private/miles-metadata.tar.gz
```

This uses SQLite backup/integrity checks and includes private credentials,
identities, ledgers and datasets. It **excludes weights, checkpoints and caches**.
Copy the 0600 archive off-host and separately preserve exported model directories,
tokenizers/configs and the pinned base model. Recover to the same absolute state
path. Verify databases and artifact hashes on a fresh host before claiming a
recoverable production backup. Roll back code and its matching state snapshot
together in maintenance; old single-workspace code must not serve new customer data.

## Public launch gate

Live collection requires an HTTPS customer domain; a verified Stripe merchant;
configured tax treatment and published terms/privacy/refund policies; support
contact; durable state/model storage with off-host restore tested; actual Stripe
sandbox and controlled merchant-authorized live payment/refund acceptance;
reconciliation monitoring; and workload-specific quality/capacity validation.

`MILES_LIVE_ENABLED=true` and `MILES_DURABLE_STORAGE_ACCEPTED=true` are explicit
operator attestations, **not automated proof** of those requirements. Do not set
them just to suppress a warning. This release claims no uptime SLA, autoscaling,
arbitrary-code sandbox, recurring subscription engine or complete localization.

## Validation

```bash
python -m pytest --confcutdir=tests/fast/platform tests/fast/platform -q
python -m pytest tests/fast/launch_scripts tests/manual/launch_scripts -q
```

Application tests cover tenant/role boundaries, invitation/recovery abuse,
concurrent balances, cancellation/restart settlement, payment signatures and
mismatches, refunds/disputes, engine ownership, scheduling and metadata restoration.
They do not contact Stripe. Historical console GPU evidence is in
[acceptance.json](acceptance.json). [Customer acceptance](customer-acceptance.json)
records the new tenant's real training/export/deployment, ordinary and streaming
inference, cross-workspace denial and preservation of the default model route.
The two simple evaluation prompts establish integration only. The internal QA
workspace was suspended afterward, its key revoked and serving slot returned.

On the acceptance host, 29 platform tests passed in 10.48 seconds and the two
platform recipe snapshots passed. The full upstream launcher suite reported
661 passed, 35 failed and 4 errors, matching the failures already observed on
the preceding baseline. This release does not claim the entire upstream suite
is green. Formatting, lint and frontend syntax checks passed.
