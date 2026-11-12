"""Private operator utilities for recovery and consistent SQLite metadata backups."""

import argparse
import json
import os
import secrets
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from miles_platform.bootstrap import bootstrap
from miles_platform.payments import PaymentConfig
from miles_platform.store import configured_store, digest, now
from miles_platform.tenancy import Registry


def backup_metadata(root, destination):
    """Caller quiesces API/worker first; exported weights require a separate copy."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents:
        raise ValueError("Backup destination must be outside the runtime state directory")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="miles-backup-") as temporary:
        temporary = Path(temporary)
        databases = list((root / "control").rglob("*.sqlite3"))
        manifest = {
            "created": now(),
            "scope": "metadata, credentials and datasets; model weights excluded",
            "databases": [],
        }
        for source in databases:
            target = temporary / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
                if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Database backup failed integrity validation")
            manifest["databases"].append(str(source.relative_to(root)))
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            archive.add(temporary, arcname=".")
            for source in root.rglob("*"):
                if source.is_file() and (
                    source.name == "admin-password"
                    or "datasets" in source.relative_to(root).parts
                    or source.parent == root / "data"
                ):
                    archive.add(source, arcname=str(source.relative_to(root)))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    backup = commands.add_parser("backup-metadata")
    backup.add_argument("--destination", required=True)
    reset = commands.add_parser("reset-link")
    reset.add_argument("--email", required=True)
    reset.add_argument("--output", required=True)
    args = parser.parse_args()
    store = configured_store()
    bootstrap(store)
    registry = Registry(store)
    if args.command == "doctor":
        config = PaymentConfig()
        result = {
            "mode": config.mode,
            "origin": config.origin,
            "blockers": config.blockers(),
            "topology": "one API process, one worker, local tenant databases",
            "state": str(store.root.parent),
        }
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if result["blockers"] else 0)
    if args.command == "backup-metadata":
        for workspace in registry.stores():
            if workspace.one("SELECT id FROM jobs WHERE status IN ('running','cancel_requested')"):
                raise SystemExit("Wait for active jobs before taking a maintenance backup")
        manifest = backup_metadata(store.root.parent, args.destination)
        print(json.dumps(manifest, indent=2))
    if args.command == "reset-link":
        email = args.email.strip().lower()
        user = registry.data.one("SELECT id FROM users WHERE email=?", (email,))
        if not user:
            raise SystemExit("Account not found")
        token = secrets.token_urlsafe(32)
        with registry.data.connect(write=True) as db:
            db.execute("DELETE FROM resets WHERE user=?", (user["id"],))
            db.execute("INSERT INTO resets VALUES (?,?,?,NULL)", (digest(token), user["id"], now() + 3600))
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(PaymentConfig().origin + "/#reset=" + token + "\n")
        registry.data.audit("reset_issued", user["id"])
        print("One-time recovery link written to the requested private file; expires in one hour.")


if __name__ == "__main__":
    main()
