"""RunPod pod management for the GPT-2 capture: create / status / ssh-command / terminate.

The API key is read from ``~/.runpod/config.toml`` and never printed; the SSH
key pair is ``~/.runpod/ssh/runpodctl-ssh-key``.  Usage::

    python gpu/gpt2/pod.py create            # one on-demand RTX 4090, prints the pod id
    python gpu/gpt2/pod.py status            # every pod on the account, with ssh endpoint
    python gpu/gpt2/pod.py ssh <pod-id>      # prints "host port" once port 22 is mapped
    python gpu/gpt2/pod.py terminate <pod-id>
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.request

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA GeForce RTX 4090"
NAME = "veritor-gpt2-silicon"
KEY = os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")


def _config() -> tuple[str, str]:
    with open(os.path.expanduser("~/.runpod/config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    return cfg["apikey"], cfg.get("apiurl", "https://api.runpod.io/graphql")


def graphql(query: str, variables: dict | None = None) -> dict:
    key, url = _config()
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "curl/8.7.1",  # the default urllib agent is refused
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"]))
    return payload["data"]


POD_FIELDS = """
    id name desiredStatus costPerHr imageName
    runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
    machine { gpuDisplayName cpuCount memoryTotal podHostId }
"""


def pods() -> list[dict]:
    return graphql(f"query {{ myself {{ pods {{ {POD_FIELDS} }} }} }}")["myself"][
        "pods"
    ]


def create() -> dict:
    with open(KEY + ".pub") as f:
        public_key = f.read().strip()
    query = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id imageName machineId costPerHr machine { podHostId gpuDisplayName }
      }
    }"""
    variables = {
        "input": {
            "cloudType": "ALL",
            "gpuCount": 1,
            "volumeInGb": 0,
            "containerDiskInGb": 40,
            "minVcpuCount": 8,
            "minMemoryInGb": 30,
            "gpuTypeId": GPU,
            "name": NAME,
            "imageName": IMAGE,
            "dockerArgs": "",
            "ports": "22/tcp",
            "volumeMountPath": "/workspace",
            "env": [{"key": "PUBLIC_KEY", "value": public_key}],
        }
    }
    return graphql(query, variables)["podFindAndDeployOnDemand"]


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], int(port["publicPort"])
    return None


def terminate(pod_id: str) -> None:
    graphql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "create":
        pod = create()
        print(json.dumps(pod, indent=2))
    elif command == "status":
        for pod in pods():
            endpoint = ssh_endpoint(pod)
            uptime = (pod.get("runtime") or {}).get("uptimeInSeconds")
            print(
                f"{pod['id']} {pod['name']} {pod['desiredStatus']} ${pod['costPerHr']}/h "
                f"{pod['machine']['gpuDisplayName']} uptime={uptime}s ssh={endpoint}"
            )
        if not pods():
            print("no pods")
    elif command == "ssh":
        for pod in pods():
            if pod["id"] == argv[2]:
                endpoint = ssh_endpoint(pod)
                if endpoint is None:
                    return 1
                print(endpoint[0], endpoint[1])
                return 0
        return 1
    elif command == "terminate":
        terminate(argv[2])
        print("terminated", argv[2])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
