"""Install this single-host prototype as private Supervisor services on Vast."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from miles_platform.bootstrap import bootstrap
from miles_platform.store import configured_store


def main():
    if os.geteuid() != 0:
        raise SystemExit("Supervisor provisioning requires root on the authorized host")
    store = configured_store()
    bootstrap(store)
    repo = Path(__file__).resolve().parents[1]
    services = {
        "console": ("api", True),
        "worker": ("worker", True),
        "runner": ("runner", False),
        "slot-a": ("serve --slot a", False),
        "slot-b": ("serve --slot b", False),
    }
    for suffix, (command, autostart) in services.items():
        name = "miles-platform-" + suffix
        wrapper = Path("/opt/supervisor-scripts") / (name + ".sh")
        script = "\n".join(
            [
                "#!/bin/bash",
                ". /opt/supervisor-scripts/utils/logging.sh",
                ". /opt/supervisor-scripts/utils/environment.sh",
                "set -eu",
                "export MILES_PLATFORM_STATE=" + shlex.quote(str(store.root.parent)),
                "export HF_HOME=" + shlex.quote(str(store.root.parent / "hf-cache")),
                "export PATH=" + shlex.quote(str(Path(sys.executable).parent)) + ':"$PATH"',
                "cd " + shlex.quote(str(repo)),
                "pty " + shlex.quote(sys.executable) + " -m miles_platform.cli " + command + " 2>&1",
                "",
            ]
        )
        wrapper.write_text(script)
        wrapper.chmod(0o755)
        Path("/etc/supervisor/conf.d", name + ".conf").write_text(
            f"""[program:{name}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
autostart={str(autostart).lower()}
autorestart={"false" if suffix == "runner" else "unexpected"}
startsecs={0 if suffix == "runner" else 2}
startretries=3
stopasgroup=true
killasgroup=true
stopwaitsecs=15
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
"""
        )
    subprocess.run(["supervisorctl", "reread"], check=True)
    subprocess.run(["supervisorctl", "update"], check=True)
    print("Console listens on 127.0.0.1:18090; bootstrap password is in control/admin-password.")


if __name__ == "__main__":
    main()
