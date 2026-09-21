"""Verify same-instance/config/seed deterministic search replay."""

from geo_llm_scheduler.config import load_config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.runner import deterministic_trace, digest
from geo_llm_scheduler.io.loaders import load_instance

config = load_config("configs/smoke.yaml")
problem = load_instance(config.instance)
a, b = run(problem, config), run(problem, config)
assert deterministic_trace(a.trace) == deterministic_trace(b.trace)
print("Deterministic replay:", digest(deterministic_trace(a.trace)))
