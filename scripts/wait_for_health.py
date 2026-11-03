"""
Health check polling utility for the API gateway.

Polls the /api/v1/health endpoint until the server is responsive,
then automatically launches the web UI in the default browser.
"""

import sys
import time
import urllib.request
import webbrowser
from typing import Optional


def wait_and_open(
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_seconds: int = 40,
    open_browser: bool = True,
) -> int:
    health_url = f"http://{host}:{port}/api/v1/health"
    ui_url = f"http://{host}:{port}/"

    print(f"[*] Waiting for API Gateway to become ready at {health_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "HealthChecker/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"[+] API Gateway is healthy! (Latency: {round(time.time() - start_time, 2)}s)")
                    if open_browser:
                        print(f"[*] Launching Web UI at {ui_url} in default browser...")
                        webbrowser.open(ui_url)
                    return 0
        except Exception:
            time.sleep(1.0)

    print(f"[-] Timed out after {timeout_seconds}s waiting for {health_url}.")
    return 1


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    sys.exit(wait_and_open(host, port))
