"""Central configuration for the revenue-stacking analysis.

All paths are anchored on the repository root so every script can be run
from anywhere. The repo root is two levels above this file
(rev_stack/model/config.py -> repo root).
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REV_STACK_DIR = REPO_ROOT / "rev_stack"
DATA_DIR = REV_STACK_DIR / "data"            # synthetic (later: real) market data
RESULTS_DIR = REV_STACK_DIR / "results"      # per-day and annual outputs
FIGURE_DIR = RESULTS_DIR / "figures"

# The facility model (constraints.py, inputs/) lives at the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Facility inputs reused from Paper 1 (CPU-fraction profiles are scale-free).
FACILITY_INPUT_DIR = REPO_ROOT / "static" / "data" / "inputs"

# ---------------------------------------------------------------------------
# Facility scaling (Paper 2 case study: 10 MW IT capacity, doc 04 section 3)
# ---------------------------------------------------------------------------
SCALE = 10.0                  # replication scaling factor from the 1 MW case
P_CONN_KW = 12_000.0          # firm import capacity of the grid connection (kW)

# ---------------------------------------------------------------------------
# Market-layer options (defaults; scripts expose CLI overrides)
# ---------------------------------------------------------------------------
BM_ACCEPTANCE = 0.08          # kappa: BM acceptance haircut. Deliberately
                              # conservative for the synthetic pack; calibrate
                              # against comparable units' dispatch rates and
                              # the Modo BM revenue benchmark with real data
                              # (doc 04 section 5).
CM_PRICE_GBP_KW_YR = 60.0     # recent T-4 clearing (Table III of the paper)
CM_DERATING = 0.55            # indicative DSR de-rating factor (verify vs EMR DB)
CM_DEFAULT_KW = 0.0           # CM committed capacity; swept in run_year
CM_WINDOW_SLOTS = range(69, 77)   # 17:00-19:00, standing deliverability window
DFS_PRIMACY_EXCLUSIVE = True  # DFS delivery excluded from other services' headroom

# Duration certification (Section 4.5 of the paper / doc 03)
CERT_MAX_OUTER_ITERS = 3      # certification loop iterations
CERT_BISECTION_STEPS = 4      # binary-search depth on the deliverable scale
CERT_START_OFFSETS = (0,)     # call-time samples within each EFA block (slots)
CERT_TOL_KW = 1.0             # feasibility tolerance on the delivery constraint
SOLVER_TIME_LIMIT_SECONDS = 600

# Solver preference order. SCIP (Paper 1's solver) supports SOS2 natively;
# HiGHS does not, but the CPU->power curve u^1.32 is convex and always under
# downward cost pressure in this formulation, so the SOS2 adjacency
# restriction is redundant: minimising the weighted sum over the simplex
# already selects adjacent breakpoints. When falling back to HiGHS the SOS2
# components are therefore deleted after model construction (see
# stack_model.prepare_for_solver).
SOLVER_PREFERENCE = ("scip", "appsi_highs")

# Synthetic-data horizon
SYNTH_START = "2025-01-01"
SYNTH_DAYS = 366              # one year + 1 day so the 3 h extension can look ahead
SYNTH_SEED = 42

# Model horizon bookkeeping (must match inputs/parameters_optimisation.py)
N_SLOTS_DAY = 96              # 15-min slots in the trading day
N_SLOTS_EXT = 108             # incl. 3 h extension
SLOTS_PER_SP = 2              # settlement period = 30 min
SLOTS_PER_EFA = 16            # EFA block = 4 h
N_SP = 48
N_EFA = 6


def slot_to_sp(t: int) -> int:
    """Settlement period (1-48) containing 15-min slot t (1-96)."""
    return (t - 1) // SLOTS_PER_SP + 1


def slot_to_efa(t: int) -> int:
    """EFA block (1-6) containing 15-min slot t (1-96)."""
    return (t - 1) // SLOTS_PER_EFA + 1


def efa_slots(b: int):
    """15-min slots (1-96) belonging to EFA block b (1-6)."""
    return range((b - 1) * SLOTS_PER_EFA + 1, b * SLOTS_PER_EFA + 1)


def sp_slots(h: int):
    """15-min slots (1-96) belonging to settlement period h (1-48)."""
    return range((h - 1) * SLOTS_PER_SP + 1, h * SLOTS_PER_SP + 1)


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
