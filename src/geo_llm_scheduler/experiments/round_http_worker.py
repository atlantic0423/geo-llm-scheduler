"""One bounded Flash transport; authentication stays in the child environment."""

import json
import sys

from geo_llm_scheduler.experiments.llm_protocol import DeepSeekClient
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND


def main() -> None:
    """Send only the frozen English contract and sanitized dynamic packet."""
    value = json.load(sys.stdin)
    try:
        response = DeepSeekClient(max_tokens=8192, system_prompt=BACKGROUND).request(
            value["payload"], value["timeout"]
        )
        result = dict(ok=True, response=response)
    except (ValueError, RuntimeError):
        result = dict(ok=False)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
