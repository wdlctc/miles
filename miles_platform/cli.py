import argparse

from miles_platform.bootstrap import bootstrap
from miles_platform.store import configured_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "api", "worker", "runner", "serve"])
    parser.add_argument("--slot", choices=["a", "b"])
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    store = configured_store()
    bootstrap(store)
    if args.command == "api":
        import uvicorn
        from miles_platform.api import create_app

        uvicorn.run(create_app(store), host="127.0.0.1", port=args.port, access_log=False)
    elif args.command == "worker":
        from miles_platform.worker import work

        work(store)
    elif args.command == "runner":
        from miles_platform.worker import run_job

        run_job(store)
    elif args.command == "serve":
        from miles_platform.runtime import serve_slot

        if args.slot is None:
            parser.error("--slot is required for serve")
        serve_slot(store, args.slot)


if __name__ == "__main__":
    main()
