"""Publish and verify the completed preflight evidence using the local Git credential manager."""

import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

folder = Path(sys.argv[1])
c = subprocess.run(
    ["git", "credential", "fill"],
    input="protocol=https\nhost=github.com\n\n",
    capture_output=True,
    text=True,
    check=True,
)
credential = dict(line.split("=", 1) for line in c.stdout.splitlines() if "=" in line)
headers = {
    "Authorization": "Bearer " + credential["password"],
    "Accept": "application/vnd.github+json",
    "User-Agent": "geo-llm-p1-preflight",
}
api = "https://api.github.com/repos/atlantic0423/geo-llm-scheduler"
tag = "p1-preflight-20261002"


def call(url, data=None, content_type="application/json"):
    h = dict(headers)
    if data is not None:
        h["Content-Type"] = content_type
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=60) as r:
        return json.load(r)


try:
    release = call(api + "/releases/tags/" + tag)
except urllib.error.HTTPError as error:
    if error.code != 404:
        raise
    release = call(
        api + "/releases",
        json.dumps(
            {
                "tag_name": tag,
                "target_commitish": "3ec2355fbc43993375daab06ec0ea39fb9d86991",
                "name": "MOEA/D P1 preflight and launch evidence (2026-10-02)",
                "prerelease": True,
                "body": "Completed 48 deterministic P0 runs, three legacy 200-generation replays, four 1200-second RSS pilots, throughput pilots, frozen inputs and manifest-bound launch gates. Source: 3ec2355fbc43993375daab06ec0ea39fb9d86991; PR #19; Python 3.11/3.13 CI passed. The 480 formal P1 runs are active and are not included in this preflight release. Verify the archive with SHA256SUMS and every member with preflight_manifest.json. Reproduce using scripts/run_p1_campaign.py and the operational helpers in the archive.",
            }
        ).encode(),
    )
assets = {asset["name"]: asset for asset in call(api + f"/releases/{release['id']}/assets")}
verification = []
for name in ("p1_preflight_20261002.tar.gz", "preflight_manifest.json", "SHA256SUMS"):
    data = (folder / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if name not in assets:
        url = release["upload_url"].split("{", 1)[0] + "?name=" + urllib.parse.quote(name)
        asset = call(url, data, "application/octet-stream")
    else:
        asset = assets[name]
    expected = "sha256:" + digest
    if asset.get("digest") != expected:
        h = {**headers, "Accept": "application/octet-stream"}
        with urllib.request.urlopen(
            urllib.request.Request(asset["url"], headers=h), timeout=60
        ) as r:
            assert hashlib.sha256(r.read()).hexdigest() == digest
    verification.append(
        {
            "name": name,
            "size": asset["size"],
            "sha256": digest,
            "url": asset["browser_download_url"],
            "verified": True,
        }
    )
refreshed = call(api + f"/releases/{release['id']}/assets")
assert all(
    any(x["name"] == item["name"] and x["size"] == item["size"] for x in refreshed)
    for item in verification
)
result = {"release": release["html_url"], "assets": verification}
(folder / "release_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
