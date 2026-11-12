import argparse

from miles_platform.bootstrap import bootstrap
from miles_platform.store import configured_store
from miles_platform.tenancy import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "api", "worker", "runner", "serve"])
    parser.add_argument("--slot", choices=list("abcd"))
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    store = configured_store()
    bootstrap(store)
    registry = Registry(store)
    if args.command == "api":
        import uvicorn
        from miles_platform.api import create_app

        uvicorn.run(create_app(store), host="127.0.0.1", port=args.port, access_log=False)
    elif args.command == "worker":
        from miles_platform.worker import work

        work(store)
    elif args.command == "runner":
        from miles_platform.worker import run_job

        run_job(registry.store(store.setting("runner_tenant", "default")))
    elif args.command == "serve":
        from miles_platform.runtime import serve_slot

        if args.slot is None:
            parser.error("--slot is required for serve")
        owner = registry.data.one("SELECT tenant FROM engines WHERE id=?", (args.slot,))
        if not owner:
            raise RuntimeError("No workspace owns this serving slot")
        serve_slot(registry.store(owner["tenant"]), args.slot)


if __name__ == "__main__":
    main()
