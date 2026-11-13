"""Build the public, read-only showcase without copying private runtime state.

Run from the repository root: python3 miles_platform/showcase/build.py
The generated dist/ directory can be published as the root of GitHub Pages.
"""

import shutil
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent
    destination = source / "dist"
    destination.mkdir(exist_ok=True)
    for name in ("index.html", "showcase.css", "showcase.js", "data.json"):
        shutil.copy2(source / name, destination / name)
    shutil.copy2(source.parent / "static" / "style.css", destination / "console.css")
    (destination / ".nojekyll").touch()
    print(destination)


if __name__ == "__main__":
    main()
