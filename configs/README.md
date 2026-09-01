# Configurations

Human-authored YAML experiment settings. Files in `configs/steering/` are
reusable inputs; resolved settings and generated responses are written beneath
`outputs/steering/` and are not committed.

Run a sweep with:

```bash
uv run python -m steering.cli.sweep --config configs/steering/role_direction.yaml
```

Resume the same run after interruption with:

```bash
uv run python -m steering.cli.sweep --resume outputs/steering/<study>/<experiment>/runs/<run>
```
