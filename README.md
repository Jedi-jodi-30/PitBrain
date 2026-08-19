# PitBrain

A locally-run agentic system for analyst-grade F1 performance reporting, with built-in reliability safeguards.

## Architecture

![PitBrain high-level architecture](pitbrain_high_level_architecture.png)

A user query is parsed for intent (session, drivers, scope), which drives a data fetch and
slice (race and/or qualifying, the full field or a subset). From there, a deterministic
analysis stage computes the catalog's metrics and attaches a chart hint to each one — this is
the only stage allowed to touch the numbers. Its output fans out to two independent paths that
never talk to each other:

- **LLM narration** — turns each section's metrics into editorial copy, then passes through a
  **reliability layer** that verifies every number and driver attribution in that copy against
  the facts it was given, before anything is published.
- **Chart rendering** — turns each metric's `chart_hint` directly into a chart, with no LLM in
  the loop at all.

Both paths converge into the final report package.

| Stage | Color |
|---|---|
| Intent parser, data fetch and slice | Blue |
| Deterministic analysis, chart rendering | Teal |
| LLM narration | Coral |
| Reliability layer | Purple |