"""Integer USD credit ledger. Holds, metering and settlement share the tenant transaction."""

from decimal import Decimal, ROUND_CEILING

from fastapi import HTTPException

from miles_platform.store import now

MICRO = 1_000_000
TRAIN_SECONDS_LIMIT = 3 * 1320


def micros(dollars):
    return int((Decimal(str(dollars)) * MICRO).to_integral_value(rounding=ROUND_CEILING))


def prepare(store):
    with store.connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger (id TEXT PRIMARY KEY, created REAL NOT NULL,
                amount INTEGER NOT NULL, kind TEXT NOT NULL, reference TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS holds (id TEXT PRIMARY KEY, amount INTEGER NOT NULL,
                created REAL NOT NULL, settled REAL, charged INTEGER, kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, created REAL NOT NULL,
                cents INTEGER NOT NULL, currency TEXT NOT NULL, mode TEXT NOT NULL,
                customer TEXT, session TEXT UNIQUE, url TEXT, payment_intent TEXT UNIQUE,
                paid INTEGER NOT NULL DEFAULT 0, refunded INTEGER NOT NULL DEFAULT 0,
                disputed INTEGER NOT NULL DEFAULT 0, idem TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS payment_events (id TEXT PRIMARY KEY, created REAL,
                type TEXT, order_id TEXT);
        """
        )


def balance(db):
    total = db.execute("SELECT COALESCE(SUM(amount),0) FROM ledger").fetchone()[0]
    held = db.execute("SELECT COALESCE(SUM(amount),0) FROM holds WHERE settled IS NULL").fetchone()[0]
    return {"balance_microusd": total, "held_microusd": held, "available_microusd": total - held, "currency": "usd"}


def reserve(db, store, reference, kind, amount):
    if getattr(store, "exempt", True):
        return
    if db.execute("SELECT 1 FROM orders WHERE disputed=1 LIMIT 1").fetchone():
        raise HTTPException(402, "Billing is paused while a payment dispute is reviewed")
    if amount <= 0:
        raise HTTPException(503, "Pricing is not configured")
    if db.execute("SELECT 1 FROM holds WHERE id=?", (reference,)).fetchone():
        return
    if balance(db)["available_microusd"] < amount:
        raise HTTPException(402, "Insufficient prepaid balance; add credits in Billing")
    db.execute("INSERT INTO holds VALUES (?,?,?,NULL,NULL,?)", (reference, amount, now(), kind))


def settle(db, reference, amount):
    hold = db.execute("SELECT * FROM holds WHERE id=? AND settled IS NULL", (reference,)).fetchone()
    if not hold:
        return
    # Charge no more than the disclosed reservation. Unobserved usage is absorbed
    # by the operator rather than represented as an invented customer charge.
    charge = min(hold["amount"], max(0, amount))
    db.execute(
        "INSERT INTO ledger VALUES (?,?,?,?,?)", ("usage:" + reference, now(), -charge, hold["kind"], reference)
    )
    db.execute("UPDATE holds SET settled=?,charged=? WHERE id=?", (now(), charge, reference))


def pricing(store):
    if hasattr(store, "registry"):
        return store.registry.default.setting("rates", {})
    return store.setting("rates", {})


def training_quote(store, rates=None):
    rate = (pricing(store) if rates is None else rates).get("gpu_hour")
    if rate is None or rate <= 0:
        if getattr(store, "exempt", True):
            return 0
        raise HTTPException(503, "Training pricing is not configured")
    return micros(Decimal(str(rate)) * TRAIN_SECONDS_LIMIT / 3600)


def inference_quote(store, rates=None):
    rates = pricing(store) if rates is None else rates
    if any(rates.get(k) is None or rates[k] <= 0 for k in ("input_million", "output_million")):
        if getattr(store, "exempt", True):
            return 0
        raise HTTPException(503, "Inference pricing is not configured")
    # The engine has a hard 2048-token context and the API caps output at 512.
    return micros((Decimal(str(rates["input_million"])) * 2048 + Decimal(str(rates["output_million"])) * 512) / MICRO)


def reconcile(store):
    """Recover settlement after runner/API failure without charging twice."""
    prepare(store)
    with store.connect(write=True) as db:
        for hold in db.execute("SELECT * FROM holds WHERE settled IS NULL").fetchall():
            table = "jobs" if hold["kind"] == "training" else "usage"
            row = db.execute(f"SELECT * FROM {table} WHERE id=?", (hold["id"],)).fetchone()
            if row and row["ended"] is not None:
                settle(db, hold["id"], micros(row["cost"]) if row["cost"] is not None else 0)
