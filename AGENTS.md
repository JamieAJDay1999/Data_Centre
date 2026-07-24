# Execution and runtime visibility

For tasks with uncertain duration, first inspect the relevant code, inputs, and recent run evidence. Then report a practical runtime estimate, with the basis for it and the main uncertainty, before launching a long run.

For any command expected to take more than five minutes, state the objective, time limit, expected duration, and the first checkpoint before starting. Use bounded benchmark runs before a full-year solver run whenever they can provide a meaningful runtime estimate.

For long solver, simulation, or test runs, provide concise milestones after setup, the first benchmark or checkpoint, and final verification. If a run exceeds its estimate, reaches a time limit, or requires a user decision, report promptly with the completed evidence and proposed next step rather than waiting silently.
