# Narrator reliability — knobs, metric, and experiment ideas

PitBrain's narration stage (`report_suite/narrative.py`) is internally named **Roger Kint** —
after "Verbal" Kint, *The Usual Suspects*' narrator, who tells a detailed, confident story that
turns out to be mostly fabricated. That's the literal failure mode this pipeline has to defend
against: during testing, a storyline about VER came back narrated entirely about "Charles
Leclerc" — genuine numbers, wrong driver. The name is only ever used internally (logs, this
doc) — it's never injected into the published prose, which stays a plain motorsport-analyst
voice.

Because Roger *will* be wrong sometimes, every draft is scored, not just judged pass/fail.

## The metric: `correctness_pct`

`verify_storyline()` scores every draft 0–100: the fraction of its checkable **claims** that
held up against the storyline's own facts. A claim is either a cited number or the implicit
"this paragraph is about the right subject" claim — so a draft with zero numbers but the wrong
driver still scores less than 100, not a vacuous pass.

```
total_claims   = cited_numbers + 1                                    # +1 = subject attribution
correct_claims = (cited_numbers - unverified_numbers) + (1 if subject_mentioned else 0)
correctness_pct = 100 * correct_claims / total_claims
```

Every run aggregates this two ways in `report_suite/weekend_story_history.json`'s
`correctness_summary`:

- **`first_draft_correctness_pct`** — Roger's own, unassisted quality. This is the number a
  prompt or model change should actually move.
- **`final_correctness_pct`** — quality after the reliability loop's correction rounds.

The gap between the two tells you how much the redraft loop is doing the work vs. how good the
narrator already was — a run where the loop is propping up a weak first draft looks different
from one that barely needed it, even if both end at 100% final.

## Control knobs (what exists today)

| Knob | Where | Effect |
|---|---|---|
| `MODEL` | `narrative.py` | Swap `llama3.2:3b` for any other Ollama tool-calling-capable model |
| `STORY_SYSTEM_PROMPT` | `narrative.py` | The narration instructions — wording changes here are the cheapest lever |
| `max_attempts` | `narrate_and_verify()` / `supervisor/graph.py`'s `MAX_NARRATION_ROUNDS` | How many redraft rounds before a storyline is excluded rather than corrected |
| `build_correction_feedback()` | `narrative.py` | How specific/targeted the retry instruction is — currently lists exactly which numbers were unverified and whether the subject was missing |
| temperature / sampling | *not currently set* | Ollama default — an easy first knob to add and measure |

## How to run an experiment

1. Pick one knob, change it.
2. Run the same query a handful of times (`python pitbrain.py "give me a rundown of monza race 2023"`).
3. Read `correctness_summary` out of each new entry in `weekend_story_history.json`.
4. Compare `first_draft_correctness_pct` across runs — that's the signal, not `final`
   (the loop can mask a worse first draft).

This is deliberately file-based rather than a dashboard right now — enough to A/B a prompt
change by eye across a handful of runs. See "future enhancements" below for where this goes
next.

## Future enhancement ideas

Roughly in order of effort:

- **Set and sweep temperature.** Lower temperature should raise `first_draft_correctness_pct`
  at some cost to prose variety — worth confirming rather than assuming.
- **Few-shot the system prompt.** One or two example storyline → correct-paragraph pairs,
  including a *bad* example showing the exact failure mode (wrong subject, invented number) and
  why it's wrong, the way the correction-feedback prompt already does per-retry.
- **Structured/constrained generation.** Have the model cite facts by key (e.g. `{{supporting_facts[0].value}}`
  substitution) instead of writing numbers freely — trades prose naturalness for not needing a
  verifier at all for the numeric half of the check.
- **Self-consistency / best-of-N.** Draft 2–3 candidates per storyline in parallel, verify all,
  keep the highest `correctness_pct` — costs more tokens, avoids the serial redraft latency.
  Compare against the current retry-on-failure loop on cost and on final correctness.
- **LLM-as-judge, as a second check alongside the regex verifier.** Have a separate model (or
  the same model, in a separate call) judge whether the draft is faithful to the facts, and
  compare its agreement rate against `verify_storyline`'s word-boundary number matcher — the
  current checker still can't catch an invented *non-numeric* claim, like a fabricated grid
  position spelled out as "second" instead of "2".
- **Track correctness_pct per detector, not just per run.** Some storyline *types* (tyre-cliff,
  team-strategy) may be harder for the narrator to stay grounded on than others (position
  swings) — aggregating by `category`/detector name in `_run_correctness_summary` would surface
  that instead of hiding it in one run-wide average.
- **A small eval set.** A fixed list of ~10 storylines (one of each detector type, frozen as
  JSON fixtures) run against every prompt/model/knob change, so experiments compare against the
  same inputs instead of whatever a live fetch happens to return that day.
- **Lightweight dashboard over `weekend_story_history.json`.** Even a simple "correctness_pct
  over time" line chart (one point per run) would turn the current "read the JSON" workflow
  into something glanceable — the dataviz skill's conventions already used in the report charts
  would carry over directly.
