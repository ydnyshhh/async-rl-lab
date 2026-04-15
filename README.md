# async-rl-lab

Small educational and research-oriented async RL library scaffold for LLM and agentic systems, centered on GRPO-family objectives, policy lag, grouped rollouts, inference serving, verifier latency, and tool-use environments.

Quickstart:

```powershell
uv sync
uv run async-rl-demo
uv run python -m unittest discover -s tests -v
```

Optional Hugging Face backend:

```powershell
uv sync --extra hf
```

Dev checks:

```powershell
uv sync --group dev
uv run --group dev ruff check .
uv run --group dev mypy src
```

Implementation guide:
- `docs/implementation_spec.md`

Core package:
- `src/async_rl_lab`
