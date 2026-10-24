"""Owner-process HTTPS worker; environment credential never travels through stdin."""

import json
import os
import sys
from urllib.request import Request, urlopen


def main() -> None:
    """Perform one fixed-endpoint request and emit only sanitized success/failure."""
    try:
        value = json.load(sys.stdin)
        key = os.environ["TYPESAFE_API_KEY"]
        body = json.dumps(value["payload"], allow_nan=False).encode()
        if not key or key.encode() in body:
            raise ValueError("Invalid credential boundary")
        request = Request(
            "https://api.typesafe.ai/v1/systemone",
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=value["timeout"]) as response:
            raw = response.read()
        if key.encode() in raw:
            raise ValueError("Credential in response")
        result = dict(ok=True, response=json.loads(raw))
    except Exception:
        result = dict(ok=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
