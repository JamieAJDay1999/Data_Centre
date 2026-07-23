"""
Launcher for the Data Centre Flexibility dashboard.

    python run_dashboard.py            # http://127.0.0.1:5000
    python run_dashboard.py --port 8080 --host 0.0.0.0

This is a convenience wrapper that ensures the app runs from the repository
root (so the modelling scripts' relative data paths resolve) and opens a
browser tab. It does not modify any modelling code.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import threading
import webbrowser

# Windows consoles default to cp1252; make our banner output UTF-8-safe.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)  # existing scripts use paths relative to the repo root

from webapp.app import app  # noqa: E402  (import after chdir on purpose)


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Data Centre dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    banner = (
        "\n"
        "  ==============================================================\n"
        "    Data Centre Flexibility  -  Interactive Dashboard\n"
        "  --------------------------------------------------------------\n"
        f"    Open:  {url}\n"
        "    Stop:  Ctrl + C\n"
        "  ==============================================================\n"
    )
    try:
        print(banner)
    except Exception:
        pass

    if not args.no_browser:
        threading.Timer(1.2, _open_browser, args=(url,)).start()

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
