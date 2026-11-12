"""Invite-only customer accounts, membership, session rotation and operator onboarding."""

import re
import secrets

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from miles_platform.store import digest, now, uid
from miles_platform.tenancy import authorize, password_hash, password_matches


class Form(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignIn(Form):
    email: str = Field(default="owner", max_length=254)
    password: str = Field(min_length=1, max_length=200)


class Email(Form):
    email: str = Field(max_length=254)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("A valid email address is required")
        return value


class NewTenant(Email):
    name: str = Field(min_length=1, max_length=80)


class Invite(Email):
    role: str = Field(pattern="^(admin|member|viewer)$")


class Accept(Form):
    token: str = Field(min_length=32, max_length=128)
    password: str = Field(min_length=12, max_length=200)


class Password(Form):
    current: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=12, max_length=200)


class Switch(Form):
    tenant_id: str = Field(max_length=80)


class Role(Form):
    role: str = Field(pattern="^(admin|member|viewer)$")


class Status(Form):
    status: str = Field(pattern="^(active|suspended)$")


def install_accounts(app, registry, secure):
    router = APIRouter()

    def issue(response, user, tenant):
        token = secrets.token_urlsafe(32)
        registry.data.execute("DELETE FROM identities WHERE expires<?", (now(),))
        registry.data.execute("INSERT INTO identities VALUES (?,?,?,?)", (digest(token), user, tenant, now() + 28800))
        response.set_cookie("miles_session", token, httponly=True, secure=secure, samesite="strict", max_age=28800)

    @router.post("/api/login")
    def login(body: SignIn, request: Request, response: Response):
        email = body.email.strip().lower()
        registry.limit_login("login:" + digest(email))
        registry.limit_login("ip:" + (request.client.host if request.client else "unknown"))
        user = registry.data.one("SELECT * FROM users WHERE email=?", (email,))
        encoded = (
            user["password"]
            if user
            else registry.data.one("SELECT password FROM users WHERE id='operator'")["password"]
        )
        if not password_matches(body.password, encoded) or not user:
            raise HTTPException(401, "Incorrect email or password")
        member = registry.data.one(
            "SELECT m.tenant FROM members m JOIN tenants t ON t.id=m.tenant WHERE user=? AND t.status='active' ORDER BY t.created LIMIT 1",
            (user["id"],),
        )
        if not member:
            raise HTTPException(403, "No active workspace membership")
        issue(response, user["id"], member["tenant"])
        registry.store(member["tenant"]).audit("login", user["id"])
        return {"ok": True}

    @router.post("/api/logout")
    def logout(request: Request, response: Response):
        registry.data.execute(
            "DELETE FROM identities WHERE hash=?", (digest(request.cookies.get("miles_session", "")),)
        )
        response.delete_cookie("miles_session", secure=secure, httponly=True, samesite="strict")
        return {"ok": True}

    @router.get("/api/account")
    def account(request: Request):
        principal = request.state.principal
        return {
            "user": principal,
            "workspaces": registry.data.rows(
                "SELECT t.id,t.name,m.role FROM tenants t JOIN members m ON m.tenant=t.id WHERE m.user=? AND t.status='active' ORDER BY t.created",
                (principal["id"],),
            ),
        }

    @router.post("/api/account/switch")
    def switch(body: Switch, request: Request, response: Response):
        user = request.state.principal["id"]
        if not registry.data.one("SELECT 1 FROM members WHERE user=? AND tenant=?", (user, body.tenant_id)):
            raise HTTPException(403, "Workspace membership required")
        registry.active(body.tenant_id)
        registry.data.execute(
            "DELETE FROM identities WHERE hash=?", (digest(request.cookies.get("miles_session", "")),)
        )
        issue(response, user, body.tenant_id)
        return {"ok": True}

    @router.post("/api/account/password")
    def change_password(body: Password, request: Request, response: Response):
        principal = request.state.principal
        user = registry.data.one("SELECT password FROM users WHERE id=?", (principal["id"],))
        registry.limit_login("password:" + principal["id"])
        if not password_matches(body.current, user["password"]):
            raise HTTPException(401, "Incorrect current password")
        with registry.data.connect(write=True) as db:
            db.execute("UPDATE users SET password=? WHERE id=?", (password_hash(body.password), principal["id"]))
            db.execute("DELETE FROM identities WHERE user=?", (principal["id"],))
        issue(response, principal["id"], principal["tenant"])
        return {"ok": True}

    @router.post("/api/invitations/accept")
    def accept(body: Accept, request: Request, response: Response):
        registry.limit_login("invite-ip:" + (request.client.host if request.client else "unknown"))
        with registry.data.connect(write=True) as db:
            invite = db.execute(
                "SELECT * FROM invites WHERE hash=? AND used IS NULL AND expires>?", (digest(body.token), now())
            ).fetchone()
            if not invite:
                raise HTTPException(400, "Invitation is invalid or expired")
            registry.active(invite["tenant"])
            user = db.execute("SELECT * FROM users WHERE email=?", (invite["email"],)).fetchone()
            if user and not password_matches(body.password, user["password"]):
                raise HTTPException(401, "This account already exists; enter its current password")
            user_id = user["id"] if user else uid("usr")
            if not user:
                db.execute(
                    "INSERT INTO users VALUES (?,?,?,0,?)",
                    (user_id, invite["email"], password_hash(body.password), now()),
                )
            # An invitation must never demote an existing owner or elevate an existing member.
            db.execute("INSERT OR IGNORE INTO members VALUES (?,?,?)", (invite["tenant"], user_id, invite["role"]))
            db.execute("UPDATE invites SET used=? WHERE hash=?", (now(), digest(body.token)))
        issue(response, user_id, invite["tenant"])
        registry.store(invite["tenant"]).audit("invite_accepted", user_id)
        return {"ok": True}

    @router.post("/api/account/reset")
    def reset_password(body: Accept, request: Request):
        registry.limit_login("reset-ip:" + (request.client.host if request.client else "unknown"))
        with registry.data.connect(write=True) as db:
            token = db.execute(
                "SELECT * FROM resets WHERE hash=? AND used IS NULL AND expires>?", (digest(body.token), now())
            ).fetchone()
            if not token:
                raise HTTPException(400, "Recovery link is invalid or expired")
            db.execute("UPDATE users SET password=? WHERE id=?", (password_hash(body.password), token["user"]))
            db.execute("UPDATE resets SET used=? WHERE hash=?", (now(), digest(body.token)))
            db.execute("DELETE FROM identities WHERE user=?", (token["user"],))
        registry.data.audit("password_recovered", token["user"])
        return {"ok": True}

    @router.get("/api/team")
    def team(request: Request):
        principal = request.state.principal
        authorize(principal, "admin")
        return registry.data.rows(
            "SELECT u.id,u.email,m.role FROM users u JOIN members m ON m.user=u.id WHERE m.tenant=? ORDER BY u.created",
            (principal["tenant"],),
        )

    @router.post("/api/team/invite")
    def invite(body: Invite, request: Request):
        principal = request.state.principal
        authorize(principal, "owner" if body.role == "admin" else "admin")
        count = registry.data.one("SELECT COUNT(*) n FROM members WHERE tenant=?", (principal["tenant"],))["n"]
        pending = registry.data.one(
            "SELECT COUNT(*) n FROM invites WHERE tenant=? AND used IS NULL AND expires>?",
            (principal["tenant"], now()),
        )["n"]
        if count + pending >= 50:
            raise HTTPException(409, "Workspace member limit reached")
        return registry.invitation(principal["tenant"], body.email, body.role, principal["id"])

    @router.patch("/api/team/{user_id}")
    def role(user_id: str, body: Role, request: Request):
        principal = request.state.principal
        authorize(principal, "owner")
        with registry.data.connect(write=True) as db:
            target = db.execute(
                "SELECT role FROM members WHERE tenant=? AND user=?", (principal["tenant"], user_id)
            ).fetchone()
            if not target:
                raise HTTPException(404, "Member not found")
            if target["role"] == "owner":
                raise HTTPException(409, "Workspace owner cannot be demoted")
            db.execute(
                "UPDATE members SET role=? WHERE tenant=? AND user=?", (body.role, principal["tenant"], user_id)
            )
        registry.store(principal["tenant"]).audit("member_role", f"{principal['id']}:{user_id}:{body.role}")
        return {"ok": True}

    @router.delete("/api/team/{user_id}")
    def remove(user_id: str, request: Request):
        principal = request.state.principal
        authorize(principal, "owner")
        with registry.data.connect(write=True) as db:
            target = db.execute(
                "SELECT role FROM members WHERE tenant=? AND user=?", (principal["tenant"], user_id)
            ).fetchone()
            if not target:
                raise HTTPException(404, "Member not found")
            if target["role"] == "owner":
                raise HTTPException(409, "Workspace owner cannot be removed")
            db.execute("DELETE FROM members WHERE tenant=? AND user=?", (principal["tenant"], user_id))
            db.execute("DELETE FROM identities WHERE tenant=? AND user=?", (principal["tenant"], user_id))
        registry.store(principal["tenant"]).audit("member_removed", f"{principal['id']}:{user_id}")
        return {"ok": True}

    @router.get("/api/operator/tenants")
    def tenants(request: Request):
        authorize(request.state.principal, operator=True)
        return registry.data.rows("SELECT * FROM tenants ORDER BY created DESC")

    @router.post("/api/operator/tenants", status_code=201)
    def create_tenant(body: NewTenant, request: Request):
        principal = request.state.principal
        authorize(principal, operator=True)
        tenant_id = uid("ws")
        registry.data.execute("INSERT INTO tenants VALUES (?,?,'active',0,?)", (tenant_id, body.name, now()))
        return {"id": tenant_id, **registry.invitation(tenant_id, body.email, "owner", principal["id"])}

    @router.patch("/api/operator/tenants/{tenant_id}")
    def status(tenant_id: str, body: Status, request: Request):
        authorize(request.state.principal, operator=True)
        store = registry.store(tenant_id)
        if tenant_id == "default":
            raise HTTPException(409, "The operator workspace cannot be suspended")
        registry.data.execute("UPDATE tenants SET status=? WHERE id=?", (body.status, tenant_id))
        if body.status == "suspended":
            store.execute("UPDATE jobs SET status='cancelled',ended=? WHERE status='queued'", (now(),))
            store.execute("UPDATE jobs SET status='cancel_requested' WHERE status='running'")
        store.audit("workspace_status", body.status)
        return {"ok": True}

    app.include_router(router)
