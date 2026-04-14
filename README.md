# async-rl-lab

Small educational and research-oriented async RL library scaffold for LLM and agentic systems, centered on GRPO-family objectives, policy lag, grouped rollouts, inference serving, verifier latency, and tool-use environments.

Quickstart:

```powershell
uv sync
uv run async-rl-demo
uv run python -m unittest discover -s tests -v
```

Implementation guide:
- `docs/implementation_spec.md`

Core package:
- `src/async_rl_lab`
