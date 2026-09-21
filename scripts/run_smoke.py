"""Run the documented deterministic smoke experiment."""

from geo_llm_scheduler.experiments.runner import run_config

print(run_config("configs/smoke.yaml"))
