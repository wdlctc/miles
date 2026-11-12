"""Stripe hosted Checkout and verified, idempotent payment reconciliation."""

import hashlib
import hmac
import json
import os
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from miles_platform import billing
from miles_platform.store import now, uid
from miles_platform.tenancy import authorize

PACKS = {"starter": 2500, "growth": 10000, "scale": 50000}
EVENTS = {
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "charge.refunded",
    "charge.dispute.created",
    "charge.dispute.updated",
    "charge.dispute.closed",
}


class PaymentConfig:
    def __init__(self):
        self.secret = os.environ.get("STRIPE_SECRET_KEY", "")
        self.webhook = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self.origin = os.environ.get("MILES_PUBLIC_URL", "http://localhost:8081").rstrip("/")
        self.mode = os.environ.get("STRIPE_MODE", "test")
        self.live = self.mode == "live"
        self.terms = os.environ.get("MILES_TERMS_URL", "")
        self.privacy = os.environ.get("MILES_PRIVACY_URL", "")
        self.support = os.environ.get("MILES_SUPPORT_EMAIL", "")
        self.tax = os.environ.get("STRIPE_AUTOMATIC_TAX", "false") == "true"

    def blockers(self):
        reasons = []
        if self.mode not in {"test", "live"}:
            reasons.append("STRIPE_MODE must be test or live")
        if not self.secret.startswith("sk_live_" if self.live else "sk_test_"):
            reasons.append("Stripe secret key is missing or does not match the selected mode")
        if not self.webhook.startswith("whsec_"):
            reasons.append("Stripe webhook signing secret is missing")
        parsed = urlparse(self.origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            reasons.append("MILES_PUBLIC_URL must be an origin without a path")
        if self.live:
            if parsed.scheme != "https" or parsed.hostname in {"localhost", "127.0.0.1"}:
                reasons.append("Live payments require the customer's HTTPS domain")
            for label, value in (("terms", self.terms), ("privacy", self.privacy)):
                if not value.startswith("https://"):
                    reasons.append(f"Published {label} URL is missing")
            if "@" not in self.support:
                reasons.append("Customer support email is missing")
            if os.environ.get("MILES_LIVE_ENABLED") != "true":
                reasons.append("Live launch is disabled pending merchant, tax and backup acceptance")
            if os.environ.get("MILES_DURABLE_STORAGE_ACCEPTED") != "true":
                reasons.append("Durable storage and off-host restore have not been accepted")
        return reasons


class Stripe:
    def __init__(self, config, client_factory=httpx.AsyncClient):
        self.config, self.client_factory = config, client_factory

    async def call(self, method, path, data=None, idem=None, params=None):
        headers = {"Authorization": "Bearer " + self.config.secret, "Stripe-Version": "2026-10-28.endive"}
        if idem:
            headers["Idempotency-Key"] = idem
        try:
            async with self.client_factory(timeout=20, trust_env=False) as client:
                response = await client.request(
                    method, "https://api.stripe.com/v1/" + path, headers=headers, data=data, params=params
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError):
            # Provider bodies may contain private customer details; never expose them.
            raise HTTPException(503, "Stripe is temporarily unavailable; retry with the same request key") from None


def verify_event(raw, signature, config):
    if not config.webhook:
        raise HTTPException(503, "Stripe webhook is not configured")
    try:
        fields = [item.split("=", 1) for item in signature.split(",")]
        timestamp = next(value for key, value in fields if key == "t")
        expected = hmac.new(config.webhook.encode(), timestamp.encode() + b"." + raw, hashlib.sha256).hexdigest()
        valid = any(key == "v1" and hmac.compare_digest(expected, value) for key, value in fields)
        if not valid or abs(now() - int(timestamp)) > 300:
            raise ValueError()
        event = json.loads(raw)
        if event["livemode"] is not config.live or not isinstance(event["id"], str):
            raise ValueError()
        return event
    except (ValueError, KeyError, TypeError, StopIteration):
        raise HTTPException(400, "Invalid Stripe signature, payload or payment mode") from None


def payment_schema(store):
    billing.prepare(store)
    with store.connect(write=True) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(orders)")}
        for name in ("credited", "lost"):
            if name not in columns:
                db.execute(f"ALTER TABLE orders ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0")
        db.execute(
            "CREATE TABLE IF NOT EXISTS disputes (id TEXT PRIMARY KEY, order_id TEXT, status TEXT, cents INTEGER)"
        )


async def reconcile_event(registry, stripe, event):
    if event["type"] not in EVENTS:
        return {"received": True, "ignored": True}
    obj = event["data"]["object"]
    dispute = None
    if event["type"].startswith("checkout.session."):
        metadata = obj.get("metadata", {})
    else:
        if event["type"].startswith("charge.dispute."):
            dispute = await stripe.call("GET", "disputes/" + obj["id"])
            intent_id = dispute.get("payment_intent")
        else:
            intent_id = obj.get("payment_intent")
        if not intent_id:
            return {"received": True, "ignored": True}
        intent = await stripe.call("GET", "payment_intents/" + intent_id)
        metadata = intent.get("metadata", {})
    if metadata.get("platform") != "miles":
        return {"received": True, "ignored": True}
    try:
        store = registry.store(metadata["tenant"])
    except (KeyError, HTTPException):
        raise HTTPException(503, "Payment workspace is unavailable; reconciliation must be retried") from None
    payment_schema(store)
    order = store.one("SELECT * FROM orders WHERE id=?", (metadata.get("order"),))
    if not order or not order["session"]:
        raise HTTPException(503, "Payment order is not yet available; retry delivery")
    if store.one("SELECT 1 FROM payment_events WHERE id=?", (event["id"],)):
        return {"received": True, "duplicate": True}
    session = await stripe.call(
        "GET", "checkout/sessions/" + order["session"], params={"expand[]": "payment_intent.latest_charge"}
    )
    expected = {"platform": "miles", "tenant": store.tenant_id, "order": order["id"]}
    valid = (
        session.get("id") == order["session"]
        and session.get("client_reference_id") == order["id"]
        and session.get("metadata") == expected
        and session.get("customer") == order["customer"]
        and session.get("currency") == "usd"
        and session.get("amount_subtotal") == order["cents"]
        and session.get("livemode") is stripe.config.live
        and order["mode"] == stripe.config.mode
        and session.get("mode") == "payment"
    )
    if not valid:
        raise HTTPException(409, "Payment does not match the stored order")
    if session.get("payment_status") != "paid":
        return {"received": True, "pending": True}
    intent = session.get("payment_intent")
    if not isinstance(intent, dict) or intent.get("status") != "succeeded":
        raise HTTPException(503, "Payment confirmation is incomplete")
    charge = intent.get("latest_charge")
    if not isinstance(charge, dict) or charge.get("currency") != "usd" or not charge.get("paid"):
        raise HTTPException(503, "Payment charge is not yet available")
    # Taxes are collected by Stripe but never become spendable platform credits.
    # Refund the credit component proportionally if the gross charge included tax.
    gross = session.get("amount_total")
    if not isinstance(gross, int) or gross < order["cents"] or charge.get("amount") != gross:
        raise HTTPException(409, "Payment total does not match the captured charge")
    refunded = min(order["cents"], (charge.get("amount_refunded", 0) * order["cents"] + gross - 1) // gross)
    with store.connect(write=True) as db:
        if db.execute("SELECT 1 FROM payment_events WHERE id=?", (event["id"],)).fetchone():
            return {"received": True, "duplicate": True}
        current = db.execute("SELECT * FROM orders WHERE id=?", (order["id"],)).fetchone()
        if dispute:
            previous = db.execute("SELECT status FROM disputes WHERE id=?", (dispute["id"],)).fetchone()
            if not previous or previous["status"] not in {"won", "lost", "warning_closed"}:
                cents = min(order["cents"], (dispute["amount"] * order["cents"] + gross - 1) // gross)
                db.execute(
                    "INSERT OR REPLACE INTO disputes VALUES (?,?,?,?)",
                    (dispute["id"], order["id"], dispute["status"], cents),
                )
        known = db.execute("SELECT * FROM disputes WHERE order_id=?", (order["id"],)).fetchall()
        pending = any(d["status"] not in {"won", "lost", "warning_closed"} for d in known)
        # A refund/session notification can arrive before the dispute notification.
        pending = pending or (bool(charge.get("disputed")) and not known)
        lost = max(current["lost"], sum(d["cents"] for d in known if d["status"] == "lost"))
        refunded = max(current["refunded"], refunded)
        credit = max(0, order["cents"] - refunded - lost) * 10000
        delta = credit - current["credited"]
        if delta:
            db.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?)",
                ("stripe:" + event["id"], now(), delta, "payment" if delta > 0 else "refund_or_dispute", order["id"]),
            )
        db.execute(
            "UPDATE orders SET paid=1,payment_intent=?,refunded=?,lost=?,disputed=?,credited=? WHERE id=?",
            (intent["id"], refunded, lost, int(pending), credit, order["id"]),
        )
        db.execute("INSERT INTO payment_events VALUES (?,?,?,?)", (event["id"], now(), event["type"], order["id"]))
    return {"received": True}


class Checkout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pack: str = Field(pattern="^(starter|growth|scale)$")


def install_payments(app, registry, config=None, client_factory=httpx.AsyncClient):
    config = config or PaymentConfig()
    stripe = Stripe(config, client_factory)
    router = APIRouter()

    @router.get("/api/billing")
    def state(request: Request):
        principal = request.state.principal
        authorize(principal, "owner")
        store = registry.store(principal["tenant"])
        payment_schema(store)
        with store.connect() as db:
            wallet = billing.balance(db)
        return {
            **wallet,
            "mode": config.mode,
            "exempt": store.exempt,
            "blockers": config.blockers(),
            "packs": PACKS,
            "rates": billing.pricing(store),
            "training_reservation_microusd": (
                billing.training_quote(store) if store.exempt or billing.pricing(store).get("gpu_hour") else None
            ),
            "terms": config.terms,
            "privacy": config.privacy,
            "support": config.support,
            "ledger": store.rows("SELECT * FROM ledger ORDER BY created DESC LIMIT 100"),
            "orders": store.rows(
                "SELECT id,created,cents,currency,mode,paid,refunded,lost,disputed FROM orders ORDER BY created DESC LIMIT 100"
            ),
        }

    @router.post("/api/billing/checkout")
    async def checkout(body: Checkout, request: Request, idempotency_key: str = Header(min_length=8, max_length=128)):
        principal = request.state.principal
        authorize(principal, "owner")
        if config.blockers():
            raise HTTPException(503, {"message": "Payments are not configured", "blockers": config.blockers()})
        store = registry.store(principal["tenant"])
        payment_schema(store)
        with store.connect(write=True) as db:
            modes = {r[0] for r in db.execute("SELECT DISTINCT mode FROM orders")}
            if modes and modes != {config.mode}:
                raise HTTPException(409, "Test balances cannot become live credit; use a new live workspace")
            order = db.execute("SELECT * FROM orders WHERE idem=?", (idempotency_key,)).fetchone()
            if order and order["cents"] != PACKS[body.pack]:
                raise HTTPException(409, "Request key already used for another pack")
            if not order:
                count = db.execute("SELECT COUNT(*) FROM orders WHERE created>?", (now() - 3600,)).fetchone()[0]
                if count >= 20:
                    raise HTTPException(429, "Checkout creation limit reached")
                order_id = uid("ord")
                db.execute(
                    "INSERT INTO orders(id,created,cents,currency,mode,idem) VALUES (?,?,?,'usd',?,?)",
                    (order_id, now(), PACKS[body.pack], config.mode, idempotency_key),
                )
                order = db.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            order = dict(order)
        if order["url"]:
            return {"id": order["id"], "url": order["url"]}
        # Stripe retains idempotency keys for at least 24h. Never retry an
        # unresolved creation after that window with a possibly reused key.
        if now() - order["created"] > 23 * 3600:
            raise HTTPException(409, "Checkout attempt expired; start a new checkout")
        if not order["customer"]:
            customer = await stripe.call(
                "POST",
                "customers",
                {"email": principal["email"], "metadata[tenant]": store.tenant_id},
                idem="customer:" + store.tenant_id + ":" + order["id"],
            )
            store.execute("UPDATE orders SET customer=? WHERE id=?", (customer["id"], order["id"]))
            order["customer"] = customer["id"]
        data = {
            "mode": "payment",
            "customer": order["customer"],
            "client_reference_id": order["id"],
            "success_url": config.origin + "/?checkout=returned#billing",
            "cancel_url": config.origin + "/#billing",
            "line_items[0][price_data][currency]": "usd",
            "line_items[0][price_data][unit_amount]": str(order["cents"]),
            "line_items[0][price_data][product_data][name]": "Miles Platform prepaid compute credits",
            "line_items[0][price_data][tax_behavior]": "exclusive",
            "line_items[0][quantity]": "1",
            "payment_method_types[0]": "card",
            "billing_address_collection": "required",
            "customer_update[address]": "auto",
            "automatic_tax[enabled]": str(config.tax).lower(),
            "invoice_creation[enabled]": "true",
        }
        for key, value in {"platform": "miles", "tenant": store.tenant_id, "order": order["id"]}.items():
            data[f"metadata[{key}]"] = value
            data[f"payment_intent_data[metadata][{key}]"] = value
        session = await stripe.call("POST", "checkout/sessions", data, idem="checkout:" + order["id"])
        parsed = urlparse(session.get("url", ""))
        if (
            parsed.scheme != "https"
            or parsed.hostname != "checkout.stripe.com"
            or session.get("livemode") is not config.live
        ):
            raise HTTPException(502, "Stripe returned an unexpected checkout destination")
        store.execute("UPDATE orders SET session=?,url=? WHERE id=?", (session["id"], session["url"], order["id"]))
        return {"id": order["id"], "url": session["url"]}

    @router.post("/api/stripe/webhook")
    async def webhook(request: Request, stripe_signature: str = Header(default="")):
        event = verify_event(await request.body(), stripe_signature, config)
        return await reconcile_event(registry, stripe, event)

    app.include_router(router)
