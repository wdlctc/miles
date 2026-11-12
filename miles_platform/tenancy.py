"""Identity registry and independently stored workspaces on a single trusted host."""

import contextvars
import hashlib
import hmac
import json
import secrets

from fastapi import HTTPException

from miles_platform.store import Store, digest, now

CURRENT = contextvars.ContextVar("workspace_store")
ROLES = {"viewer": 0, "member": 1, "admin": 2, "owner": 3}


def password_hash(password):
    salt = secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return json.dumps({"salt": salt, "hash": value})


def password_matches(password, encoded):
    saved = json.loads(encoded)
    value = hashlib.scrypt(password.encode(), salt=saved["salt"].encode(), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(value, saved["hash"])


class WorkspaceStore:
    """A request-local facade; never a mutable process-global tenant selection."""

    def __getattr__(self, name):
        return getattr(CURRENT.get(), name)


class Registry:
    def __init__(self, default):
        self.default = default
        self.data = Store(default.root / "identity")
        with self.data.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE,
                    password TEXT NOT NULL, operator INTEGER NOT NULL DEFAULT 0, created REAL);
                CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active', exempt INTEGER NOT NULL DEFAULT 0, created REAL);
                CREATE TABLE IF NOT EXISTS members (tenant TEXT, user TEXT, role TEXT,
                    PRIMARY KEY(tenant,user));
                CREATE TABLE IF NOT EXISTS identities (hash TEXT PRIMARY KEY, user TEXT, tenant TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS invites (hash TEXT PRIMARY KEY, tenant TEXT, email TEXT,
                    role TEXT, expires REAL, used REAL, created_by TEXT);
                CREATE TABLE IF NOT EXISTS attempts (bucket TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS resets (hash TEXT PRIMARY KEY, user TEXT, expires REAL, used REAL);
                CREATE INDEX IF NOT EXISTS attempts_bucket ON attempts(bucket,created);
                CREATE TABLE IF NOT EXISTS engines (id TEXT PRIMARY KEY, tenant TEXT, model_id TEXT, port INTEGER);
            """
            )
        with self.data.connect(write=True) as db:
            if not db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                db.execute(
                    "INSERT INTO users VALUES ('operator','owner',?,1,?)",
                    (json.dumps(default.setting("admin_hash")), now()),
                )
                db.execute("INSERT INTO tenants VALUES ('default','My workspace','active',1,?)", (now(),))
                db.execute("INSERT INTO members VALUES ('default','operator','owner')")
                for row in default.rows("SELECT * FROM slots"):
                    db.execute(
                        "INSERT OR IGNORE INTO engines VALUES (?,'default',?,?)",
                        (row["id"], row["model_id"], row["port"]),
                    )
        self.store("default")

    def store(self, tenant):
        row = self.data.one("SELECT * FROM tenants WHERE id=?", (tenant,))
        if not row:
            raise HTTPException(404, "Workspace not found")
        store = self.default if tenant == "default" else Store(self.default.root / "workspaces" / tenant / "control")
        store.registry, store.tenant_id = self, tenant
        store.exempt = bool(row["exempt"])
        store.base_models = self.default.root.parent / "models"
        if store.setting("quality_threshold") is None:
            store.set("quality_threshold", 0.375)
        return store

    def stores(self):
        return [self.store(r["id"]) for r in self.data.rows("SELECT id FROM tenants ORDER BY created")]

    def active(self, tenant):
        row = self.data.one("SELECT status FROM tenants WHERE id=?", (tenant,))
        if not row or row["status"] != "active":
            raise HTTPException(403, "Workspace is suspended; contact the platform operator")

    def limit_login(self, bucket):
        with self.data.connect(write=True) as db:
            db.execute("DELETE FROM attempts WHERE created<?", (now() - 900,))
            count = db.execute(
                "SELECT COUNT(*) FROM attempts WHERE bucket=? AND created>?", (bucket, now() - 900)
            ).fetchone()[0]
            if count >= 15:
                raise HTTPException(429, "Too many attempts; retry in 15 minutes")
            db.execute("INSERT INTO attempts VALUES (?,?)", (bucket, now()))

    def principal(self, token):
        row = self.data.one(
            """SELECT u.id,u.email,u.operator,s.tenant,m.role FROM identities s
            JOIN users u ON u.id=s.user JOIN members m ON m.user=u.id AND m.tenant=s.tenant
            WHERE s.hash=? AND s.expires>?""",
            (digest(token), now()),
        )
        if not row:
            raise HTTPException(401, "Please sign in")
        self.active(row["tenant"])
        return row

    def invitation(self, tenant, email, role, creator):
        token = secrets.token_urlsafe(32)
        with self.data.connect(write=True) as db:
            members = db.execute("SELECT COUNT(*) FROM members WHERE tenant=?", (tenant,)).fetchone()[0]
            pending = db.execute(
                "SELECT COUNT(*) FROM invites WHERE tenant=? AND used IS NULL AND expires>?", (tenant, now())
            ).fetchone()[0]
            if members + pending >= 50:
                raise HTTPException(409, "Workspace member quota reached (50)")
            db.execute(
                "INSERT INTO invites VALUES (?,?,?,?,?,NULL,?)",
                (digest(token), tenant, email, role, now() + 86400, creator),
            )
        self.store(tenant).audit("invite_created", json.dumps({"by": creator, "email": email, "role": role}))
        return {"token": token, "expires_in": 86400}


def authorize(principal, minimum="viewer", operator=False):
    if operator and not principal["operator"]:
        raise HTTPException(403, "Platform operator access required")
    if ROLES[principal["role"]] < ROLES[minimum]:
        raise HTTPException(403, f"{minimum.title()} access required")


def resource_permission(principal, method, path):
    required = "viewer"
    if method not in {"GET", "HEAD", "OPTIONS"}:
        required = "member"
        if path.startswith(("/api/keys", "/api/deployments")) or path.endswith("/deploy"):
            required = "admin"
    authorize(principal, required, operator=path == "/api/rates")
