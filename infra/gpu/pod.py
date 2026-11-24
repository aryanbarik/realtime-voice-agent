"""Runpod pod lifecycle CLI for the self-hosted GPU stack.

Usage: uv run python infra/gpu/pod.py {create,start,stop,status,env,wait,rtt,terminate}
Secrets are read from .env and never printed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
POD_ID_FILE = Path(__file__).parent / ".pod_id"
ENV_SELFHOST = ROOT / ".env.selfhost"
API = "https://api.runpod.io/v2"
REPO_URL = "https://github.com/aryanbarik/realtime-voice-agent.git"
CMD = (
    "apt-get update && apt-get install -y git espeak-ng && "
    "(test -d /workspace/repo || git clone $REPO_URL /workspace/repo) && "
    "git -C /workspace/repo pull && bash /workspace/repo/infra/gpu/bootstrap.sh"
)


def env_vars() -> dict[str, str]:
    return {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}


def require(name: str) -> str:
    v = env_vars().get(name)
    if not v:
        sys.exit(f"{name} is not set in .env")
    return v


def client() -> httpx.Client:
    return httpx.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {require('RUNPOD_API_KEY')}"},
        timeout=60,
    )


def pod_id() -> str:
    if not POD_ID_FILE.exists():
        sys.exit("No pod id; run `create` first")
    return POD_ID_FILE.read_text().strip()


def check(resp: httpx.Response) -> dict:
    if resp.status_code >= 400:
        sys.exit(f"Runpod API error {resp.status_code}: {resp.text[:500]}")
    return resp.json() if resp.content else {}


def cmd_create(args) -> None:
    body = {
        "name": "voice-agent-gpu",
        "image": "vllm/vllm-openai:v0.34.0",
        "cloud": "SECURE",
        "gpu": {"id": args.gpu, "count": 1},
        "disk": 40,
        "mounts": {"persistent": {"size": 100, "path": "/workspace"}},
        "ports": ["8000/tcp", "8001/tcp"],
        "env": {
            "HF_HOME": "/workspace/hf",
            "SELFHOST_API_KEY": require("SELFHOST_API_KEY"),
            "REPO_URL": REPO_URL,
        },
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [CMD],
    }
    with client() as c:
        data = check(c.post("/pods", json=body))
    pid = data.get("id")
    if not pid:
        sys.exit(f"Unexpected response (no id): keys={sorted(data)}")
    POD_ID_FILE.write_text(pid + "\n")
    print(f"Created pod {pid}")


def cmd_action(action: str):
    def run(_args) -> None:
        pid = pod_id()
        with client() as c:
            r = c.post(f"/pods/{pid}/action", json={"action": action})
        if action == "start" and r.status_code == 400:
            sys.exit("GPU no longer free on host; run terminate then create")
        check(r)
        print(f"{action} requested for pod {pid}")

    return run


def get_pod() -> dict:
    with client() as c:
        return check(c.get(f"/pods/{pod_id()}"))


def ports(pod: dict) -> list[dict]:
    return (pod.get("runtime") or {}).get("ports") or []


def cmd_status(_args) -> None:
    pod = get_pod()
    print("status:", pod.get("status"))
    for p in ports(pod):
        print(f"  private={p.get('private')} public={p.get('public')} ip={p.get('ip')}")


def endpoint(pod: dict, private: int) -> tuple[str, int]:
    for p in ports(pod):
        if int(p.get("private", -1)) == private and p.get("ip") and p.get("public"):
            return p["ip"], int(p["public"])
    sys.exit(
        f"No public endpoint for port {private}. If the pod has no public IP, recreate it "
        'with ports ["8000/http","8001/http"] and use the proxy URLs '
        "https://<pod_id>-8000.proxy.runpod.net / wss://<pod_id>-8001.proxy.runpod.net"
    )


def cmd_env(_args) -> None:
    pod = get_pod()
    llm_ip, llm_port = endpoint(pod, 8000)
    sp_ip, sp_port = endpoint(pod, 8001)
    ENV_SELFHOST.write_text(
        f"VA_SELFHOST_LLM_URL=http://{llm_ip}:{llm_port}\n"
        f"VA_SELFHOST_SPEECH_URL=ws://{sp_ip}:{sp_port}\n"
    )
    print(f"Wrote {ENV_SELFHOST.name}")


def urls() -> tuple[str, str]:
    vals = {}
    if ENV_SELFHOST.exists():
        vals = dotenv_values(ENV_SELFHOST)
    llm = vals.get("VA_SELFHOST_LLM_URL")
    sp = vals.get("VA_SELFHOST_SPEECH_URL")
    if not llm or not sp:
        sys.exit("Run `env` first")
    return llm, sp.replace("wss://", "https://").replace("ws://", "http://")


def cmd_wait(_args) -> None:
    key = require("SELFHOST_API_KEY")
    headers = {"Authorization": f"Bearer {key}"}
    llm, sp = urls()
    deadline = time.monotonic() + 40 * 60
    speech_ok = llm_ok = False
    while time.monotonic() < deadline:
        if not speech_ok:
            try:
                speech_ok = httpx.get(f"{sp}/health", headers=headers, timeout=5).status_code == 200
            except httpx.HTTPError:
                pass
        if not llm_ok:
            try:
                llm_ok = httpx.get(f"{llm}/v1/models", headers=headers, timeout=5).status_code == 200
            except httpx.HTTPError:
                pass
        print(f"speech={'ready' if speech_ok else 'waiting'} llm={'ready' if llm_ok else 'waiting'}")
        if speech_ok and llm_ok:
            print("ready")
            return
        time.sleep(10)
    sys.exit("Timed out after 40 minutes")


def cmd_rtt(_args) -> None:
    headers = {"Authorization": f"Bearer {require('SELFHOST_API_KEY')}"}
    _, sp = urls()
    times = []
    with httpx.Client(headers=headers, timeout=10) as c:
        for _ in range(10):
            t = time.perf_counter()
            c.get(f"{sp}/health").raise_for_status()
            times.append((time.perf_counter() - t) * 1000)
    print(f"median RTT: {statistics.median(times):.1f} ms")


def cmd_terminate(_args) -> None:
    pid = pod_id()
    with client() as c:
        check(c.delete(f"/pods/{pid}"))
    POD_ID_FILE.unlink(missing_ok=True)
    print(f"Terminated pod {pid}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--gpu", default="NVIDIA A40")
    c.set_defaults(fn=cmd_create)
    sub.add_parser("start").set_defaults(fn=cmd_action("start"))
    sub.add_parser("stop").set_defaults(fn=cmd_action("stop"))
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    sub.add_parser("wait").set_defaults(fn=cmd_wait)
    sub.add_parser("rtt").set_defaults(fn=cmd_rtt)
    sub.add_parser("terminate").set_defaults(fn=cmd_terminate)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
