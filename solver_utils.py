import os

import pyomo.environ as pyo


DEFAULT_SOLVER_CANDIDATES = ("scip", "highs", "appsi_highs", "cbc", "glpk")


def create_solver():
    """Create the first available Pyomo solver, honoring PYOMO_SOLVER first."""
    requested_solver = os.environ.get("PYOMO_SOLVER")
    if requested_solver:
        solver = pyo.SolverFactory(requested_solver)
        if solver.available(False):
            return solver, requested_solver
        raise RuntimeError(
            f"PYOMO_SOLVER is set to '{requested_solver}', but that solver is not available. "
            "Install it, update PATH, or unset PYOMO_SOLVER to use an automatic fallback."
        )

    for solver_name in DEFAULT_SOLVER_CANDIDATES:
        solver = pyo.SolverFactory(solver_name)
        if solver.available(False):
            return solver, solver_name

    tried = ", ".join(DEFAULT_SOLVER_CANDIDATES)
    raise RuntimeError(
        f"No available Pyomo solver found. Tried: {tried}. "
        "Install HiGHS with 'python -m pip install highspy', install SCIP/CBC/GLPK, "
        "or set PYOMO_SOLVER to a solver available in your environment."
    )


def set_time_limit(solver, solver_name, seconds):
    """Set a solver time limit when the selected solver supports one."""
    if seconds is None:
        return

    if solver_name == "scip":
        solver.options["limits/time"] = seconds
    elif solver_name in {"highs", "appsi_highs"}:
        solver.options["time_limit"] = seconds
    elif solver_name == "cbc":
        solver.options["seconds"] = seconds
    elif solver_name == "glpk":
        solver.options["tmlim"] = seconds
