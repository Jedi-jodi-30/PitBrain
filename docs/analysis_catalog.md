# Race Analysis Catalog — Numeric Metrics for Visualization Pipeline

Source: `comparison_laps.csv` — 2023 Monza Race, clean laps only (TrackStatus==1, no pit in/out), 19 drivers, MEDIUM/HARD compounds.

Each metric below lists: **grain** (what one row of output represents), **formula** (columns used), and **suggested chart** for the downstream visualization stage. Group into a "report package" with sections mirroring a race-weekend digital feature.

---

## 1. Headline Pace
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Median race pace | per driver | `median(LapTimeSeconds)` on clean laps | Ranked horizontal bar |
| Fastest lap of race | per driver + overall | `min(LapTimeSeconds)`, note `LapNumber`, `Compound` | Headline stat card / table |
| Mean pace | per driver | `mean(LapTimeSeconds)` | Bar (compare to median for skew) |
| Pace delta to race-fastest | per driver | `driver_median - field_min_median` | Diverging bar (gap to P1 pace) |
| Rolling pace (smoothed) | per driver, per lap | rolling mean window=3–5 over `LapTimeSeconds` ordered by `LapNumber` | Multi-line time series |

## 2. Consistency / Variability
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Lap time std dev | per driver (per stint) | `std(LapTimeSeconds)` | Bar, ascending = most consistent |
| Coefficient of variation | per driver | `std/mean` | Bar (normalizes for pace level) |
| Interquartile range | per driver | `Q3 - Q1` of `LapTimeSeconds` | Box plot per driver |
| Consistency band % | per driver | % of laps within 102% of driver's own best lap | Bar / gauge |
| Lap time range | per stint | `max - min` of `LapTimeSeconds` within `Stint` | Grouped bar by stint |

## 3. Tyre Strategy & Degradation
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Stint composition | per driver, per stint | `Stint`, `Compound`, count of laps, `min/max LapNumber` | Strategy Gantt / stacked horizontal bar |
| Pit stop count & laps | per driver | count of `PitInTime` non-null; lap numbers where it occurs | Timeline markers |
| Compound usage split | field-wide | % laps on MEDIUM vs HARD | Stacked bar or donut |
| Degradation slope | per driver, per stint | linear regression slope of `LapTimeSeconds` vs `TyreLife` | Small-multiple scatter+trendline per stint |
| Early vs late stint pace | per driver, per stint | `mean(first 5 laps)` vs `mean(last 5 laps)` of stint | Slope chart (before/after) |
| Fresh vs used tyre delta | per driver | `mean(LapTimeSeconds where FreshTyre==True)` vs `False` | Grouped bar |
| Cliff detection | per driver, per stint | lap-over-lap delta (`LapTimeSeconds[n] - LapTimeSeconds[n-1]`); flag jumps > 1.5s | Annotated line chart |

## 4. Sector Breakdown
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Median sector times | per driver × sector | `median(Sector1/2/3Time)` (convert to seconds) | Grouped/stacked bar, 3 sectors per driver |
| Theoretical best lap | per driver | `min(Sector1) + min(Sector2) + min(Sector3)` vs actual best lap | Gap bar (real vs theoretical) |
| Sector delta to fastest | per driver × sector | `driver_sector - field_min_sector` | Heatmap (drivers × 3 sectors) |
| Sector trend over race | per driver × sector, per lap | raw sector time by `LapNumber` | 3 small-multiple line charts |

## 5. Speed Trap / Top Speed
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Max speed by trap | per driver | `max(SpeedI1)`, `max(SpeedI2)`, `max(SpeedFL)`, `max(SpeedST)` | Grouped bar, 4 traps per driver |
| Avg speed trap (straight-line proxy) | per driver | `mean(SpeedST)` | Ranked bar |
| Speed trap trend (fuel burn effect) | per driver, per lap | `SpeedST` vs `LapNumber` | Line chart, expect gradual rise |

## 6. Race Position & Story
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Position per lap | per driver, per lap | `Position` vs `LapNumber` | Classic F1 position line chart (inverted y-axis) |
| Net position change | per driver | `Position(last lap) - Position(first lap)` | Diverging bar (gained/lost) |
| Laps led | per driver | count where `Position == 1` | Bar |
| Position volatility | per driver | `std(Position)` | Bar |

## 7. Head-to-Head Duels
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Lap time delta (driver A − B) | per lap | matched by `LapNumber` | Line chart around zero line |
| Cumulative gap | per lap | running sum of lap-time delta | Race gap chart (broadcast-style) |
| Lap-win count | per pair | count of laps where A faster vs B faster | Stacked bar / scoreboard |

## 8. Team-Level Aggregates
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Team median pace | per team | `median(LapTimeSeconds)` grouped by `Team` (both cars) | Ranked bar (constructor pace order) |
| Intra-team teammate gap | per team | `abs(driverA_median - driverB_median)` | Bar |
| Team pit stop timing | per team | pit laps per car | Timeline |

## 9. Personal Bests & Highlights
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Personal-best count/timing | per driver | rows where `IsPersonalBest==True`, their `LapNumber` | Timeline dots over position/pace chart |
| Fastest lap headline | overall | single row: driver, time, lap, compound | Stat card |
| Session-best sector ("purple") moments | per lap | row where sector time == field min for that sector | Annotated markers |

## 10. Data Quality / Context Flags
| Metric | Grain | Formula | Chart |
|---|---|---|---|
| Excluded laps note | field-wide | count of raw laps removed via `TrackStatus != 1` or pit in/out (already filtered here) | Footnote stat |
| Deleted lap count/reasons | per driver | `Deleted==True`, `DeletedReason` | Small table |
| Accuracy ratio | per driver | % laps with `IsAccurate==True` | Footnote / data-quality badge |

---

## Suggested handoff shape

Emit each metric as flat records so the visualization stage doesn't need to re-derive anything:

```json
{
  "category": "tyre_degradation",
  "metric": "degradation_slope_sec_per_lap",
  "driver": "VER",
  "stint": 2,
  "compound": "HARD",
  "value": 0.012,
  "n_laps": 30,
  "chart_hint": "scatter_trendline"
}
```

## Report package structure (newspaper-style)
1. **Headline strip** — fastest lap, race-pace leader, biggest mover, tyre gamble of the day (§1, §6, §9)
2. **The Race Story** — position chart + cumulative gap for top storylines (§6, §7)
3. **Pace & Consistency** — ranked pace bar + consistency bar side by side (§1, §2)
4. **Strategy Room** — stint Gantt + degradation slopes (§3)
5. **Under the Microscope** — sector heatmap + speed trap bars (§4, §5)
6. **Team Battle** — constructor pace order + teammate gaps (§8)
7. **Fine Print** — data-quality footnotes (§10)
