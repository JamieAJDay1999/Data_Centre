# Lessons

## Keep requested outputs in the project

- Mistake pattern: A user-requested plot was delivered only through Codex's hidden visualization cache, so it was not discoverable in the repository.
- Prevention rule: For repository data-plotting requests, save a conventional output file in a clearly named project directory and provide a clickable absolute path; include a reproducible plotting script when useful.

## Keep annual wrappers focused on the required model stages

- Mistake pattern: The first annual wrapper included flexibility-duration probes whose repeated optimisation cost made the multi-day trial impractical.
- Prevention rule: Before scaling a workflow across many dates, separate required outputs from expensive optional analyses; default to the minimum stages needed for the paper and add costly probes only when explicitly requested.

## Apply recovery transforms at the smallest data scope

- Mistake pattern: A proposed negative-price recovery first considered rewriting the complete historical CSV even though only a small set of daily horizons needed different treatment.
- Prevention rule: Preserve raw research data and apply documented cleaning or sensitivity rules after extracting the smallest model input unit, with explicit metadata identifying which recovered cases used the rule.
