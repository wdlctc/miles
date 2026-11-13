# Public platform showcase

This read-only website presents the platform UI and a real, completed training
and deployment acceptance run. It never connects to the private GPU server and
has no sign-in, new inference, training or payment endpoints.

`data.json` is the published historical record from the 2026-11-12 acceptance.
It includes measured per-step metrics, model validation, API response checks and
a failed quality example. It excludes credentials, runtime addresses, account
data, private logs and model weights. It is not a live status feed.

Build from the repository root:

```bash
python3 miles_platform/showcase/build.py
python3 -m http.server 8093 --bind 127.0.0.1 --directory miles_platform/showcase/dist
```

The build copies the canonical console stylesheet from `../static/style.css`.
Only showcase-specific differences belong in `showcase.css`. Publish the
contents of the generated, ignored `dist/` directory to the root of the
`gh-pages` branch. GitHub Pages serves that branch with `.nojekyll`.

For the actual server application, authentication, GPU requirements and launch
limitations, use the [platform deployment guide](../README.md).
