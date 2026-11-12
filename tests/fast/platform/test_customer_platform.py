"""Tenant boundaries and financial failure cases; no network or real card charges."""

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from miles_platform import billing
from miles_platform.api import create_app
from miles_platform.gateway import reserve, record
from miles_platform.payments import PaymentConfig
from miles_platform.rewards import grade, numeric
from miles_platform.store import Store, digest, now

PASSWORD = "Customer-password-123!"


class FakeStripe:
    def __init__(self):
        self.sessions, self.intents, self.disputes, self.requests = {}, {}, {}, []

    def respond(self, request):
        self.requests.append(request)
        path = request.url.path.removeprefix("/v1/")
        data = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if path == "customers":
            return httpx.Response(200, json={"id": "cus_" + str(len(self.requests))})
        if path == "checkout/sessions":
            name = "cs_test_" + str(len(self.sessions))
            pi = "pi_" + name
            amount = int(data["line_items[0][price_data][unit_amount]"])
            metadata = {k: data[f"metadata[{k}]"] for k in ("platform", "tenant", "order")}
            intent = {
                "id": pi,
                "status": "succeeded",
                "metadata": metadata,
                "latest_charge": {
                    "id": "ch_" + name,
                    "amount": amount,
                    "amount_refunded": 0,
                    "currency": "usd",
                    "paid": True,
                    "disputed": False,
                },
            }
            session = {
                "id": name,
                "url": "https://checkout.stripe.com/c/pay/" + name,
                "livemode": False,
                "client_reference_id": data["client_reference_id"],
                "metadata": metadata,
                "customer": data["customer"],
                "currency": "usd",
                "amount_subtotal": amount,
                "amount_total": amount,
                "payment_status": "paid",
                "payment_intent": intent,
                "mode": "payment",
            }
            self.sessions[name], self.intents[pi] = session, intent
            return httpx.Response(200, json=session)
        if path.startswith("checkout/sessions/"):
            return httpx.Response(200, json=self.sessions[path.split("/")[-1]])
        if path.startswith("payment_intents/"):
            return httpx.Response(200, json=self.intents[path.split("/")[-1]])
        if path.startswith("disputes/"):
            return httpx.Response(200, json=self.disputes[path.split("/")[-1]])
        raise AssertionError(path)

    def client(self, **kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.respond), **kwargs)


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.setenv("STRIPE_MODE", "test")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fixture")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_fixture")
    monkeypatch.setenv("MILES_PUBLIC_URL", "http://testserver")
    provider = FakeStripe()
    store = Store(tmp_path / "control")
    app = create_app(store, payment_config=PaymentConfig(), payment_client_factory=provider.client)
    operator = TestClient(app)
    assert (
        operator.post("/api/login", json={"password": (store.root / "admin-password").read_text()}).status_code == 200
    )
    operator.put("/api/rates", json={"gpu_hour": 1, "input_million": 2, "output_million": 4})
    return app, operator, provider


def customer(platform, email="alice@example.com"):
    app, operator, _ = platform
    created = operator.post("/api/operator/tenants", json={"name": email.split("@")[0], "email": email})
    assert created.status_code == 201, created.text
    result = created.json()
    client = TestClient(app)
    accepted = client.post("/api/invitations/accept", json={"token": result["token"], "password": PASSWORD})
    assert accepted.status_code == 200, accepted.text
    store = app.state.registry.store(result["id"])
    billing.prepare(store)
    return client, store, result


def fund(store, amount=100_000_000):
    store.execute("INSERT INTO ledger VALUES ('test-seed',?,?,'test_fixture','fixture')", (now(), amount))


def payload(n):
    return json.dumps({"prompt": [{"role": "user", "content": f"{n}+1?"}], "label": str(n + 1)})


def upload(client):
    response = client.post(
        "/api/datasets",
        json={"name": "Private", "train_jsonl": "\n".join(payload(n) for n in range(4)), "eval_jsonl": payload(5)},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def checkout(client):
    result = client.post(
        "/api/billing/checkout", json={"pack": "starter"}, headers={"Idempotency-Key": "first-checkout"}
    )
    assert result.status_code == 200, result.text
    return result.json()


def event(provider, name="evt_paid", kind="checkout.session.completed", obj=None):
    return {
        "id": name,
        "type": kind,
        "livemode": False,
        "data": {"object": obj or next(iter(provider.sessions.values()))},
    }


def deliver(client, data, secret="whsec_fixture", timestamp=None):
    raw = json.dumps(data).encode()
    timestamp = int(now()) if timestamp is None else timestamp
    signature = hmac.new(secret.encode(), str(timestamp).encode() + b"." + raw, hashlib.sha256).hexdigest()
    return client.post(
        "/api/stripe/webhook",
        content=raw,
        headers={"Stripe-Signature": f"t={timestamp},v1={signature}", "Content-Type": "application/json"},
    )


def test_cross_tenant_read_write_logs_and_keys(platform):
    a, sa, _ = customer(platform)
    b, sb, _ = customer(platform, "bob@example.com")
    fund(sa)
    ds = upload(a)
    job = a.post("/api/jobs", json={"name": "Secret", "dataset_id": ds}).json()
    assert b.get("/api/state").json()["jobs"] == []
    assert b.get("/api/state").json()["datasets"] == []
    assert b.post("/api/jobs", json={"name": "Theft", "dataset_id": ds}).status_code == 404
    assert b.get("/api/jobs/" + job["id"] + "/logs").status_code == 404
    assert b.post("/api/jobs/" + job["id"] + "/cancel").status_code == 404
    assert b.post("/api/account/switch", json={"tenant_id": sa.tenant_id}).status_code == 403
    key = a.post("/api/keys", json={"name": "Private"}).json()
    tampered = key["secret"].replace(sa.tenant_id, sb.tenant_id)
    assert b.get("/v1/models", headers={"Authorization": "Bearer " + tampered}).status_code == 401
    assert sa.root != sb.root and str(sa.root) not in b.get("/api/state").text
    assert a.put("/api/rates", json={"gpu_hour": 0}).status_code == 403
    assert a.get("/api/operator/tenants").status_code == 403


def test_invite_replay_existing_account_roles_and_removal(platform):
    client, store, invite = customer(platform)
    assert (
        client.post("/api/invitations/accept", json={"token": invite["token"], "password": PASSWORD}).status_code
        == 400
    )
    invitation = client.post("/api/team/invite", json={"email": "view@example.com", "role": "viewer"}).json()
    viewer = TestClient(platform[0])
    assert (
        viewer.post("/api/invitations/accept", json={"token": invitation["token"], "password": PASSWORD}).status_code
        == 200
    )
    assert viewer.get("/api/state").status_code == 200
    assert viewer.post("/api/keys", json={"name": "bad"}).status_code == 403
    assert viewer.post("/api/datasets", json={"name": "x", "train_jsonl": "", "eval_jsonl": ""}).status_code == 403
    assert viewer.get("/api/billing").status_code == 403
    viewer_id = viewer.get("/api/account").json()["user"]["id"]
    assert client.patch("/api/team/" + viewer_id, json={"role": "admin"}).status_code == 200
    assert viewer.post("/api/keys", json={"name": "allowed"}).status_code == 200
    assert client.delete("/api/team/" + viewer_id).status_code == 200
    assert viewer.get("/api/state").status_code == 401
    owner_id = client.get("/api/account").json()["user"]["id"]
    assert client.delete("/api/team/" + owner_id).status_code == 409


def test_existing_account_invite_cannot_reset_password(platform):
    client, _, _ = customer(platform)
    second = platform[1].post("/api/operator/tenants", json={"name": "Second", "email": "alice@example.com"}).json()
    assert (
        client.post(
            "/api/invitations/accept", json={"token": second["token"], "password": "Attack-password-123"}
        ).status_code
        == 401
    )
    assert (
        client.post("/api/invitations/accept", json={"token": second["token"], "password": PASSWORD}).status_code
        == 200
    )
    assert len(client.get("/api/account").json()["workspaces"]) == 2


def test_password_rotation_and_suspension(platform):
    client, store, _ = customer(platform)
    other = TestClient(platform[0])
    assert other.post("/api/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 200
    assert (
        client.post("/api/account/password", json={"current": PASSWORD, "password": "New-password-12345"}).status_code
        == 200
    )
    assert other.get("/api/state").status_code == 401
    key = client.post("/api/keys", json={"name": "key"}).json()["secret"]
    platform[1].patch("/api/operator/tenants/" + store.tenant_id, json={"status": "suspended"})
    assert client.get("/api/state").status_code == 403
    assert client.get("/v1/models", headers={"Authorization": "Bearer " + key}).status_code == 403


def test_training_hold_cancel_and_restart_settlement(platform):
    client, store, _ = customer(platform)
    ds = upload(client)
    body = {"name": "train", "dataset_id": ds}
    assert client.post("/api/jobs", json=body).status_code == 402
    fund(store, 2_000_000)
    headers = {"Idempotency-Key": "train-once"}
    response = client.post("/api/jobs", json=body, headers=headers)
    assert response.status_code == 202, response.text
    job = response.json()
    assert client.post("/api/jobs", json=body, headers=headers).json()["id"] == job["id"]
    assert client.post("/api/jobs", json=body).status_code == 402
    assert client.post("/api/jobs/" + job["id"] + "/cancel").status_code == 200
    assert client.get("/api/billing").json()["held_microusd"] == 0
    next_job = client.post("/api/jobs", json=body).json()
    store.execute("UPDATE jobs SET status='succeeded',cost=0.25,ended=? WHERE id=?", (now(), next_job["id"]))
    billing.reconcile(store)
    billing.reconcile(store)
    wallet = client.get("/api/billing").json()
    assert wallet["balance_microusd"] == 1_750_000 and wallet["held_microusd"] == 0


def test_concurrent_balance_reservations_never_overspend(platform):
    _, store, _ = customer(platform)
    amount = billing.inference_quote(store)
    fund(store, amount)

    def attempt(_):
        try:
            return reserve(store, {"id": "console", "rpm": 120}, {"id": "dep"})
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    ids = [r for r in results if isinstance(r, str)]
    assert len(ids) == 1 and all(r == 402 for r in results if isinstance(r, int))
    record(store, ids[0], 200, {"prompt_tokens": 10, "completion_tokens": 2})
    record(store, ids[0], 200, {"prompt_tokens": 10, "completion_tokens": 2})
    with store.connect() as db:
        wallet = billing.balance(db)
    assert wallet["balance_microusd"] == amount - 28 and wallet["held_microusd"] == 0


def test_checkout_is_bound_idempotent_and_never_credits_on_return(platform):
    client, store, _ = customer(platform)
    first = checkout(client)
    second = checkout(client)
    assert first == second
    assert client.get("/?checkout=returned").status_code == 200
    assert client.get("/api/billing").json()["balance_microusd"] == 0
    assert (
        client.post(
            "/api/billing/checkout", json={"pack": "scale"}, headers={"Idempotency-Key": "first-checkout"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/billing/checkout", json={"pack": "starter", "cents": 1}, headers={"Idempotency-Key": "tampered"}
        ).status_code
        == 422
    )
    assert len(store.rows("SELECT * FROM orders")) == 1


def test_webhook_signature_paid_validation_and_duplicate_events(platform):
    client, _, _ = customer(platform)
    checkout(client)
    data = event(platform[2])
    assert deliver(client, data, secret="wrong").status_code == 400
    assert deliver(client, data, timestamp=int(now()) - 301).status_code == 400
    assert deliver(client, {**data, "livemode": True}).status_code == 400
    assert deliver(client, data).status_code == 200
    assert deliver(client, data).status_code == 200
    assert deliver(client, event(platform[2], "evt_duplicate_different_id")).status_code == 200
    wallet = client.get("/api/billing").json()
    assert wallet["balance_microusd"] == 25_000_000
    assert len(wallet["ledger"]) == 1


@pytest.mark.parametrize(
    "field,value",
    [("amount_subtotal", 1), ("customer", "cus_other"), ("currency", "eur"), ("client_reference_id", "other")],
)
def test_paid_order_mismatch_cannot_credit(platform, field, value):
    client, _, _ = customer(platform)
    checkout(client)
    session = next(iter(platform[2].sessions.values()))
    session[field] = value
    assert deliver(client, event(platform[2])).status_code == 409
    assert client.get("/api/billing").json()["balance_microusd"] == 0


def test_unpaid_then_paid_and_refund_before_completion(platform):
    client, _, _ = customer(platform)
    checkout(client)
    provider = platform[2]
    session = next(iter(provider.sessions.values()))
    session["payment_status"] = "unpaid"
    assert deliver(client, event(provider)).status_code == 200
    assert client.get("/api/billing").json()["balance_microusd"] == 0
    session["payment_status"] = "paid"
    charge = session["payment_intent"]["latest_charge"]
    charge["amount_refunded"] = 1000
    refund = event(
        provider, "evt_refund", "charge.refunded", {**charge, "payment_intent": session["payment_intent"]["id"]}
    )
    assert deliver(client, refund).status_code == 200
    assert deliver(client, event(provider, "evt_late_paid")).status_code == 200
    assert client.get("/api/billing").json()["balance_microusd"] == 15_000_000
    charge["amount_refunded"] = 2500
    assert deliver(client, {**refund, "id": "evt_refund_rest"}).status_code == 200
    assert client.get("/api/billing").json()["balance_microusd"] == 0


def test_dispute_pauses_usage_then_lost_debits_once(platform):
    client, store, _ = customer(platform)
    checkout(client)
    provider = platform[2]
    assert deliver(client, event(provider)).status_code == 200
    session = next(iter(provider.sessions.values()))
    session["payment_intent"]["latest_charge"]["disputed"] = True
    dispute = {
        "id": "dp_1",
        "payment_intent": session["payment_intent"]["id"],
        "status": "needs_response",
        "amount": 2500,
    }
    provider.disputes["dp_1"] = dispute
    assert deliver(client, event(provider, "evt_dispute", "charge.dispute.created", dispute)).status_code == 200
    with pytest.raises(HTTPException, match="402"):
        reserve(store, {"id": "console", "rpm": 10}, {"id": "dep"})
    dispute["status"] = "lost"
    assert deliver(client, event(provider, "evt_closed", "charge.dispute.closed", dispute)).status_code == 200
    assert deliver(client, event(provider, "evt_created_late", "charge.dispute.created", dispute)).status_code == 200
    assert client.get("/api/billing").json()["balance_microusd"] == 0


def test_live_requires_launch_acceptance_and_test_wallet_cannot_switch(platform, monkeypatch):
    client, _, _ = customer(platform)
    checkout(client)
    monkeypatch.setenv("STRIPE_MODE", "live")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_fixture")
    blockers = PaymentConfig().blockers()
    assert any("HTTPS" in b for b in blockers)
    assert any("restore" in b for b in blockers)


def test_numeric_grader_rejects_executable_and_expensive_expressions():
    assert grade(r"Answer: \boxed{42}", "42")
    assert grade(r"\boxed{\frac{1}{2}}", ".5")
    assert not grade(r"\boxed{__import__('os').system('echo bad')}", "0")
    assert not grade(r"\boxed{10**100000000}", "1")
    for value in ("1/0", "1e1000000", "float('nan')", "9" * 129):
        with pytest.raises(ValueError):
            numeric(value)


def test_recovery_link_is_one_time_and_revokes_sessions(platform):
    client, _, _ = customer(platform)
    identity = client.get("/api/account").json()["user"]["id"]
    token = "r" * 43
    registry = platform[0].state.registry
    registry.data.execute("INSERT INTO resets VALUES (?,?,?,NULL)", (digest(token), identity, now() + 60))
    body = {"token": token, "password": "Recovered-password-123"}
    assert client.post("/api/account/reset", json=body).status_code == 200
    assert client.get("/api/state").status_code == 401
    assert client.post("/api/account/reset", json=body).status_code == 400
    assert (
        client.post("/api/login", json={"email": "alice@example.com", "password": body["password"]}).status_code == 200
    )


def test_full_engine_pool_never_evicts_another_tenant(platform, monkeypatch):
    from miles_platform.runtime import SLOTS, ensure_server

    _, store, _ = customer(platform)
    registry = platform[0].state.registry
    for slot, port in SLOTS.items():
        registry.data.execute("INSERT INTO engines VALUES (?,'default','private_model',?)", (slot, port))
    commands = []
    monkeypatch.setattr("miles_platform.runtime.supervise", lambda *a, **kw: commands.append(a))
    with pytest.raises(RuntimeError, match="capacity is full"):
        ensure_server(store, "my_model", "my_job")
    assert commands == []


def test_scheduler_routes_runner_to_correct_workspace(platform, monkeypatch):
    from miles_platform.worker import tick_workspaces

    _, a, _ = customer(platform)
    _, b, _ = customer(platform, "bob@example.com")
    fund(a)
    fund(b)
    a.enqueue("train", "A", {})
    b.enqueue("train", "B", {})
    registry = platform[0].state.registry
    registry.default.set("runner_tenant", a.tenant_id)
    commands = []
    monkeypatch.setattr("miles_platform.worker.supervise", lambda *args, **kw: commands.append(args) or "EXITED")
    tick_workspaces(registry)
    assert registry.default.setting("runner_tenant") == b.tenant_id
    assert b.one("SELECT status FROM jobs")["status"] == "running"
    assert a.one("SELECT status FROM jobs")["status"] == "queued"
    assert commands.count(("start", "runner")) == 1


def test_metadata_backup_restores_identity_and_ledger(platform, tmp_path):
    import sqlite3
    import tarfile

    from miles_platform.operations import backup_metadata

    _, store, _ = customer(platform)
    fund(store)
    root = platform[0].state.store.root.parent
    # Store's root is under tmp_path in this fixture; use a sibling destination.
    destination = tmp_path.parent / (tmp_path.name + "-metadata.tar.gz")
    backup_metadata(root, destination)
    assert destination.stat().st_mode & 0o777 == 0o600
    restored = tmp_path / "restored"
    with tarfile.open(destination) as archive:
        archive.extractall(restored, filter="data")
    database = restored / store.path.relative_to(root)
    with sqlite3.connect(database) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT SUM(amount) FROM ledger").fetchone()[0] == 100_000_000
