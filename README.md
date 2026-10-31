# PitBrain

A locally-run agentic system for analyst-grade F1 performance reporting, with built-in
reliability safeguards. Ask it a plain-English question about a race weekend; it routes the
request, runs deterministic analysis, mines the results for actual storylines, narrates them
with a local model, fact-checks every number before publishing, and writes a self-contained
HTML report.

```bash
python pitbrain.py "give me a rundown of the monza race 2023"
```

## Original design

![PitBrain high-level architecture](pitbrain_high_level_architecture.png)
![PitBrain agent pipeline](pitbrain_agent_pipeline.svg)

The two diagrams above are the original design docs this project started from: intent → fetch
→ deterministic analysis → a narration path (with a reliability layer) and a chart-rendering
path, converging into a report; and a five-agent pipeline (Supervision → Intent Parser →
Analysis → Reporting → Reliability) implementing that flow. They're kept here as the reference
design. The sections below describe what's actually built today, which has evolved past them
in a few ways — most notably, the Supervisor is now the thing that routes by *reasoning* about
the query rather than a separate rigid intent parser gating everything, and the Reliability
Layer is one agent with two distinct jobs rather than a single narration check.

## Current implementation

PitBrain's orchestration is a real [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` (`supervisor/graph.py`), not a conceptual pipeline:

```
query ──▶ supervisor ──▶ analysis ──▶ narration ⇄ reliability ──▶ report
              │                                        │
              ▼                                        ▼
           blocked ◀───────────────────────────────────┘
```

**Supervisor** — an Ollama tool-calling call (`llama3.2:3b`) with three tools:
`run_weekend_rundown`, `run_pace_analysis`, `run_driver_comparison`. It reasons about the raw
query rather than matching a fixed `analysis_type` string, so "give me a complete race
analysis", "what happened this weekend", and "give me a rundown" all correctly route to the
same capability even though none of them share exact phrasing. `intent_paser_agent.py` still
exists as a narrower structured-field extractor but is no longer the hard router.

**Analysis** — runs the chosen capability's deterministic half. For a weekend rundown this
means: fetch the session (`analysis_suite/basic_imports.py`), run the full metrics catalog
(`analysis_suite/new_analysis.py`, ~40 functions per `docs/analysis_catalog.md`, cached per
session under `analysis_output/sessions/<year>_<event>_<session_type>/` — **never a shared
path**, since that was a real bug: the first version cached globally, so every race after the
first silently served back the first race's analysis), then mine those facts for actual
storylines (`analysis_suite/storylines.py`, 18 independent detectors spanning pace, tyre
degradation, consistency, sector breakdown, speed trap, position, and team-level play).
Two detectors reach beyond the catalog entirely: **pit-stop strategy efficiency** (real pit
counts pulled from the raw, unfiltered laps — `clean_laps` has every pit lap filtered out by
definition — cross-referenced against each team's net position change) and **qualifying pace
vs. top speed by team** (an optional second session fetch). Any fetch or compute failure comes
back as `{"error": ...}` instead of an exception — the graph's contract for "this run can't
proceed."

**Narration** — drafts prose per storyline with `llama3.2:3b`, one call each.

**Reliability — the agent the user asked to have highlighted.** It's one agent with two
distinct jobs, not two separate checks bolted together:

1. *Analysis-level gate.* If the analysis stage errored or came back empty, the reliability
   node hard-fails the whole run — routes straight to `blocked`, no report is written. This is
   the thing that was actually missing before: nothing used to check whether the Analysis
   Agent's output was even usable before narration ran on it.
2. *Narration-level fact-check.* For every storyline's drafted paragraph, it verifies every
   cited number traces back to that storyline's own supporting facts (not the whole catalog),
   and that the subject is actually named in the text (catches the model substituting a
   different real driver while still citing genuine numbers — this happened in testing:
   a storyline about VER came back narrated entirely about "Charles Leclerc"). A failed check
   doesn't just get flagged — the graph loops back to **narration** with targeted correction
   feedback built from exactly what was wrong ("these numbers don't appear in the facts",
   "name the subject explicitly"), and redrafts, for up to 2 rounds total.

**The publishing rule this produces: only verified content is ever reported.** A storyline
that still fails after its redraft round is excluded from the published report entirely — not
shown with a warning, not published with a caveat. Nothing a reader sees has skipped the
fact-check. Every draft, every correction prompt, and every verdict is still logged in full
(`report_suite/weekend_story_history.json`, `supervisor/run_log.json`), so a run that excluded
storylines is fully auditable even though the report itself stays clean.

**Report** — a self-contained HTML file. The weekend-rundown path writes a magazine-style
article: headline + dek drafted from only the published storylines, one card per storyline
with its own matched chart (an exact-match guard prevents two different storylines from
illustrating themselves with the same field-wide chart; a chart that can't be guaranteed to
show the right subject is swapped for a one-off chart built from that storyline's own facts).
The pace/comparison routes publish a smaller stats report from the same analysis output,
not a raw JSON dump.

### Known gaps
- No interactive ask-the-user loop or multi-turn checkpointing — each query is one fresh graph
  invocation.
- Track-to-track comparison (mentioned in the original design) isn't built.
- FastF1 live fetches have been unreliable from this environment during development (see
  `supervisor/run_log.json` / git history for the debugging trail) — cached sessions under
  `fastf1_cache/` work regardless; a session that was never fetched may currently fail even
  for long-settled historical races.

## How to run

**Prerequisites** (one-time):
```bash
ollama pull llama3.2:3b
pip install -r requirements.txt   # or: .venv\Scripts\pip install -r requirements.txt
```

**Ask it anything:**
```bash
python pitbrain.py "give me a rundown of monza race 2023"
python pitbrain.py "what happened during the 2023 monza grand prix weekend"
python pitbrain.py "give me pace stats for the 2023 monza race"
python pitbrain.py "compare Verstappen and Piastri pace at 2023 monza"
```

Each run prints the recognized route and result, and — on success — a `report_path` to the
published HTML in `reports/`. A request that can't be fulfilled (bad fetch, no data) comes
back as a clear `error`, not a broken report.

**Inspect what happened:**
- `supervisor/run_log.json` — every query: route taken, blocked or not, and why
- `report_suite/weekend_story_history.json` — full per-storyline narration detail, including
  every flag → correction → redraft cycle

**Run one stage directly** (useful for debugging without the full graph):
```bash
python report_suite/generate_weekend_story.py --year 2023 --event Monza --session R
```

`gemma2:2b` was dropped in favor of `llama3.2:3b` throughout — Gemma's Ollama template doesn't
support tool-calling, which the Supervisor, Intent Parser, and Analysis Agent all rely on.
