# Runtime Investigation

- [x] Inspect `run_day`, certification, and solver wrapper for silent long calls.
- [x] Confirm the active conda run is waiting on a SCIP subprocess.
- [x] Add solver timing/progress controls without changing the optimisation model.
- [x] Verify diagnostics with compile, CLI help, and capped SCIP/HiGHS smoke solves.

## Results

- Active user run is waiting on `scip.exe`, not burning CPU in `python.exe`.
- `python -m py_compile` passed for the edited modules.
- `run_day --help` shows `--time-limit` and `--tee`.
- Tiny Pyomo smoke solves printed start/done timing for both `scip` and `appsi_highs`.
