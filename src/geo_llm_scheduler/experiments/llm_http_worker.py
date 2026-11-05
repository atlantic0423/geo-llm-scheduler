"""One-shot HTTP process: parent can enforce an absolute wall deadline safely."""

import json
import sys

from geo_llm_scheduler.experiments.llm_protocol import DeepSeekClient


def main() -> None:
    """Read data only on stdin; never pass authentication in command arguments."""
    request = json.loads(sys.stdin.read())
    try:
        response = DeepSeekClient().request(request["payload"], request["timeout"])
        print(json.dumps(dict(ok=True, response=response), ensure_ascii=False, allow_nan=False))
    except (ValueError, RuntimeError):
        print(json.dumps(dict(ok=False, error="provider_request_failed")))


if __name__ == "__main__":
    main()
