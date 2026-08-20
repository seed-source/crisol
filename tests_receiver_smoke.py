"""Smoke test for Crisol Receiver and Remote Client Dispatch."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Ensure E:\code\crisol is in sys.path
CRISOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CRISOL_DIR))

from crisol import core, receiver, cli

def main() -> None:
    print("=== CRISOL RECEIVER & REMOTE CLIENT SMOKE TEST ===")
    
    # 1. Start receiver daemon on port 28435
    port = 28435
    token = "iris-crisol-secret-2026"
    t = threading.Thread(
        target=receiver.run_receiver,
        kwargs={"host": "127.0.0.1", "port": port, "token": token},
        daemon=True,
    )
    t.start()
    time.sleep(1)

    node_url = f"http://127.0.0.1:{port}"

    # 2. Probe health / status
    print("\n--- 1. Probing Remote Node Health ---")
    status = core.probe_remote_node(node_url, token=token)
    print(f"Health Status: {status.get('status')}")
    print(f"Python: {status.get('python')} on {status.get('platform')}")
    print(f"Nodes: {[n['name'] for n in status.get('nodes', [])]}")
    print(f"Available Jobs: {status.get('jobs')}")

    # 3. Test Unauthorized access
    print("\n--- 2. Testing Unauthorized Access Guard ---")
    try:
        core.probe_remote_node(node_url, token="wrong-token")
        print("FAIL: Unauthorized probe did not raise error!")
    except Exception as exc:
        print(f"OK: Caught expected auth error: {exc}")

    # 4. Dispatch cpu-smoke over HTTP with live event stream
    print("\n--- 3. Dispatching 'cpu-smoke' remotely with live event stream ---")
    def on_event(ev: dict) -> None:
        etype = ev.get("type")
        detail = ev.get("phase") or ev.get("event") or (f"rate={ev.get('rate')}" if "rate" in ev else "")
        print(f"  [STREAM] type={etype:<10} detail={detail}")

    rec_cpu = core.dispatch_remote(
        url=node_url,
        job_name="cpu-smoke",
        seed=1729,
        token=token,
        event_callback=on_event,
    )
    print(f"cpu-smoke Result: status={rec_cpu['status']}, exit={rec_cpu['exit_code']}, elapsed={rec_cpu['elapsed_s']}s")
    assert rec_cpu["status"] == "completed", f"Expected completed, got {rec_cpu['status']}"
    assert rec_cpu["exit_code"] == 0, f"Expected 0, got {rec_cpu['exit_code']}"

    # 5. Dispatch torch-smoke if torch is available
    if core._torch_available():
        print("\n--- 4. Dispatching 'torch-smoke' remotely (CUDA/MPS acceleration) ---")
        rec_torch = core.dispatch_remote(
            url=node_url,
            job_name="torch-smoke",
            seed=1729,
            token=token,
            event_callback=on_event,
        )
        print(f"torch-smoke Result: status={rec_torch['status']}, exit={rec_torch['exit_code']}, elapsed={rec_torch['elapsed_s']}s")
        assert rec_torch["status"] == "completed", f"Expected completed, got {rec_torch['status']}"
        assert rec_torch["exit_code"] == 0, f"Expected 0, got {rec_torch['exit_code']}"

    # 6. Test CLI subparser execution via cli.main()
    print("\n--- 5. Testing CLI entrypoint parity ---")
    exit_code = cli.main(["status", "--node", node_url, "--token", token, "--json"])
    print(f"cli status --node exit code: {exit_code}")
    assert exit_code == 0

    print("\n=== ALL RECEIVER & REMOTE DISPATCH SMOKE TESTS PASSED! ===")

if __name__ == "__main__":
    main()
