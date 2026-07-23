"""
Flask application for the Data Centre Flexibility dashboard.

Run it from the repository root with::

    python run_dashboard.py

The app is a *presentation layer only*: it reads the artefacts produced by the
existing modelling scripts and can trigger fresh solves via ``webapp.runners``,
which reuse those scripts unchanged.
"""

from __future__ import annotations

import pathlib
import traceback

from flask import Flask, jsonify, render_template, request

from webapp import data_access, runners

ROOT = pathlib.Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(pathlib.Path(__file__).parent / "templates"),
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)
# Always pick up template edits from disk (handy during development / demos).
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Read-only data API
# ---------------------------------------------------------------------------
@app.route("/api/overview")
def api_overview():
    return jsonify(data_access.overview())


@app.route("/api/nominal")
def api_nominal():
    return jsonify(data_access.nominal_timeseries())


@app.route("/api/optimisation")
def api_optimisation():
    return jsonify(data_access.optimisation_timeseries())


@app.route("/api/flexibility")
def api_flexibility():
    return jsonify(data_access.flex_heatmap())


@app.route("/api/flexibility/detail")
def api_flex_detail():
    try:
        ts = int(request.args["ts"])
        mag = int(request.args["mag"])
    except (KeyError, ValueError):
        return jsonify({"available": False, "error": "ts and mag are required integers"}), 400
    return jsonify(data_access.flex_detail(ts, mag))


@app.route("/api/inputs")
def api_inputs():
    return jsonify(data_access.inputs_bundle())


# ---------------------------------------------------------------------------
# Live scenario runs (reuse the modelling scripts)
# ---------------------------------------------------------------------------
@app.route("/api/run/nominal", methods=["POST"])
def api_run_nominal():
    overrides = request.get_json(silent=True) or {}
    try:
        return jsonify(runners.run_nominal(overrides))
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        traceback.print_exc()
        return jsonify({"available": False, "error": str(exc)}), 500


@app.route("/api/run/optimisation", methods=["POST"])
def api_run_optimisation():
    overrides = request.get_json(silent=True) or {}
    try:
        return jsonify(runners.run_optimisation(overrides))
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        traceback.print_exc()
        return jsonify({"available": False, "error": str(exc)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "availability": data_access.availability()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
