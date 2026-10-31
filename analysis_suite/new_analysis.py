"""Every metric in docs/analysis_catalog.md, each as its own function.

Each function takes `clean_laps` (the DataFrame from basic_imports.get_clean_laps) and returns
a dict: {"facts": [...flat records, matching the catalog's handoff shape...], "chart": a
matplotlib Figure, or None where the metric is a footnote with nothing to plot}.

Two metrics in the catalog (pit stop count/laps, team pit stop timing) need the *raw*, unfiltered
laps — clean_laps has already dropped every pit in/out lap by definition (PitInTime is always
null here), so computing them from clean_laps would silently report zero pit stops for a race
where cars obviously pitted. Those two return a caveat fact instead of a wrong number.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared chart chrome — same validated palette as visualization_suite/visualization.py,
# so every chart in the project reads as one system.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQ_BLUE = "#2a78d6"
SEQ_BLUE_DARK = "#256abf"
SEQ_BLUE_LIGHT = "#86b6ef"
ACCENT_ORANGE = "#eb6834"
CONTEXT_GRAY = "#c3c2b7"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Small shared helpers — pandas plumbing and chart builders reused across metrics
# ---------------------------------------------------------------------------
def _seconds(series: pd.Series) -> pd.Series:
    """Coerce a Timedelta-like column (LapTime, SectorNTime, ...) to float seconds."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_timedelta(series).dt.total_seconds()


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)


def _style_axes(ax, xlabel=None, ylabel=None):
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _ranked_bar(data: pd.DataFrame, category_col: str, value_col: str, title: str, xlabel: str,
                highlight: str = "best", value_fmt: str = "{:.3f}") -> plt.Figure:
    """Horizontal ranked bar. highlight='best'|'worst'|None picks which end gets the accent."""
    data = data.sort_values(value_col, ascending=True).reset_index(drop=True)
    n = len(data)
    colors = [SEQ_BLUE] * n
    if highlight == "best" and n:
        colors[0] = ACCENT_ORANGE
    elif highlight == "worst" and n:
        colors[-1] = ACCENT_ORANGE

    fig, ax = plt.subplots(figsize=(8, 0.32 * max(n, 1) + 1))
    bars = ax.barh(data[category_col], data[value_col], color=colors)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel=xlabel)
    _title(ax, title)
    span = data[value_col].max() - data[value_col].min() if n else 1
    for bar, value in zip(bars, data[value_col]):
        ax.text(value + span * 0.015, bar.get_y() + bar.get_height() / 2,
                 value_fmt.format(value), va="center", ha="left", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def _diverging_bar(data: pd.DataFrame, category_col: str, value_col: str, title: str, xlabel: str) -> plt.Figure:
    data = data.sort_values(value_col, ascending=True).reset_index(drop=True)
    colors = [STATUS_CRITICAL if v < 0 else STATUS_GOOD for v in data[value_col]]
    fig, ax = plt.subplots(figsize=(8, 0.32 * max(len(data), 1) + 1))
    ax.barh(data[category_col], data[value_col], color=colors)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel=xlabel)
    _title(ax, title)
    fig.tight_layout()
    return fig


def _emphasis_line(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, leaders: list,
                    title: str, xlabel: str, ylabel: str, invert_y: bool = False) -> plt.Figure:
    """Line chart with `leaders` picked out in fixed categorical colors, everyone else in gray —
    the standard treatment here for "one line per driver" charts, since >6-8 distinct hues stop
    being tell-apart-able (see the dataviz skill's series-count ladder)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, g in df.groupby(group_col):
        if key in leaders:
            continue
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], color=CONTEXT_GRAY, linewidth=1, alpha=0.5, zorder=1)
    for i, key in enumerate(leaders):
        g = df[df[group_col] == key].sort_values(x_col)
        ax.plot(g[x_col], g[y_col], color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2, zorder=2, label=key)
    if invert_y:
        ax.invert_yaxis()
    _style_axes(ax, xlabel=xlabel, ylabel=ylabel)
    _title(ax, title)
    ax.legend(frameon=False, ncol=min(len(leaders), 8), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def _top_leaders(clean_laps: pd.DataFrame, n: int = 6) -> list:
    return clean_laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().head(n).index.tolist()


def _stat_card(lines: list, title: str) -> plt.Figure:
    """A big-number 'stat card' for headline facts that aren't really a data chart."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.axis("off")
    _title(ax, title)
    for i, (value, label) in enumerate(lines):
        y = 0.62 - i * 0.42
        ax.text(0.02, y, value, fontsize=22, fontweight="bold", color=INK_PRIMARY, transform=ax.transAxes)
        ax.text(0.02, y - 0.16, label, fontsize=9, color=INK_MUTED, transform=ax.transAxes)
    fig.tight_layout()
    return fig


# ===========================================================================
# 1. Headline Pace
# ===========================================================================
def median_pace(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().reset_index()
    facts = [
        {"category": "headline_pace", "metric": "median_pace_seconds", "driver": r.Driver,
         "value": round(r.LapTimeSeconds, 3), "chart_hint": "ranked_bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df, "Driver", "LapTimeSeconds", "Median Race Pace by Driver", "Median lap time (s)")
    return {"facts": facts, "chart": chart}


def fastest_lap(clean_laps: pd.DataFrame) -> dict:
    row = clean_laps.loc[clean_laps["LapTimeSeconds"].idxmin()]
    per_driver = clean_laps.loc[clean_laps.groupby("Driver")["LapTimeSeconds"].idxmin()]
    facts = [{"category": "headline_pace", "metric": "fastest_lap_overall", "driver": row["Driver"],
              "value": round(float(row["LapTimeSeconds"]), 3), "lap_number": int(row["LapNumber"]),
              "compound": row["Compound"], "chart_hint": "stat_card"}]
    facts += [
        {"category": "headline_pace", "metric": "fastest_lap_per_driver", "driver": r.Driver,
         "value": round(r.LapTimeSeconds, 3), "lap_number": int(r.LapNumber), "chart_hint": "ranked_bar"}
        for r in per_driver.itertuples()
    ]
    chart = _ranked_bar(per_driver, "Driver", "LapTimeSeconds", "Fastest Lap by Driver", "Fastest lap time (s)")
    return {"facts": facts, "chart": chart}


def mean_pace(clean_laps: pd.DataFrame) -> dict:
    mean_df = clean_laps.groupby("Driver")["LapTimeSeconds"].mean().reset_index()
    median_df = clean_laps.groupby("Driver")["LapTimeSeconds"].median().reset_index()
    merged = mean_df.merge(median_df, on="Driver", suffixes=("_mean", "_median")).sort_values("LapTimeSeconds_mean")
    facts = [
        {"category": "headline_pace", "metric": "mean_pace_seconds", "driver": r.Driver,
         "value": round(r.LapTimeSeconds_mean, 3), "median_seconds": round(r.LapTimeSeconds_median, 3),
         "skew_seconds": round(r.LapTimeSeconds_mean - r.LapTimeSeconds_median, 3), "chart_hint": "dumbbell"}
        for r in merged.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(merged) + 1))
    y = range(len(merged))
    ax.hlines(y, merged["LapTimeSeconds_median"], merged["LapTimeSeconds_mean"], color=BASELINE, linewidth=1.5)
    ax.scatter(merged["LapTimeSeconds_median"], y, color=SEQ_BLUE_LIGHT, s=40, zorder=2, label="Median")
    ax.scatter(merged["LapTimeSeconds_mean"], y, color=SEQ_BLUE_DARK, s=40, zorder=3, label="Mean")
    ax.set_yticks(list(y)); ax.set_yticklabels(merged["Driver"]); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Lap time (s)")
    _title(ax, "Mean vs Median Pace (compare for skew)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def pace_delta_to_fastest(clean_laps: pd.DataFrame) -> dict:
    median_df = clean_laps.groupby("Driver")["LapTimeSeconds"].median().reset_index()
    field_min = median_df["LapTimeSeconds"].min()
    median_df["PaceDelta"] = median_df["LapTimeSeconds"] - field_min
    median_df = median_df.sort_values("PaceDelta")
    facts = [
        {"category": "headline_pace", "metric": "pace_delta_to_fastest_seconds", "driver": r.Driver,
         "value": round(r.PaceDelta, 3), "chart_hint": "ranked_bar"}
        for r in median_df.itertuples()
    ]
    chart = _ranked_bar(median_df, "Driver", "PaceDelta", "Pace Delta to Fastest Median Pace",
                         "Delta to field-fastest median (s)", highlight="best", value_fmt="+{:.3f}")
    return {"facts": facts, "chart": chart}


def rolling_pace(clean_laps: pd.DataFrame, window: int = 3) -> dict:
    df = clean_laps.sort_values(["Driver", "LapNumber"]).copy()
    df["RollingPace"] = df.groupby("Driver")["LapTimeSeconds"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    facts = [
        {"category": "headline_pace", "metric": "rolling_pace_seconds", "driver": r.Driver,
         "lap_number": int(r.LapNumber), "value": round(r.RollingPace, 3), "chart_hint": "line"}
        for r in df.itertuples()
    ]
    leaders = _top_leaders(clean_laps)
    chart = _emphasis_line(df, "LapNumber", "RollingPace", "Driver", leaders,
                            f"Rolling Pace ({window}-lap avg) — Top {len(leaders)} Highlighted",
                            "Lap number", "Rolling pace (s)")
    return {"facts": facts, "chart": chart}


# ===========================================================================
# 2. Consistency / Variability
# ===========================================================================
def lap_time_std_dev(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby(["Driver", "Stint"])["LapTimeSeconds"].std().reset_index(name="StdDev").dropna()
    overall = clean_laps.groupby("Driver")["LapTimeSeconds"].std().sort_values().reset_index(name="StdDev")
    facts = [
        {"category": "consistency", "metric": "lap_time_std_dev", "driver": r.Driver, "stint": r.Stint,
         "value": round(r.StdDev, 3), "chart_hint": "bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(overall, "Driver", "StdDev", "Lap Time Std Dev by Driver (lower = more consistent)",
                         "Std dev (s)", highlight="best")
    return {"facts": facts, "chart": chart}


def coefficient_of_variation(clean_laps: pd.DataFrame) -> dict:
    grouped = clean_laps.groupby("Driver")["LapTimeSeconds"].agg(["std", "mean"])
    grouped["cv"] = grouped["std"] / grouped["mean"]
    grouped = grouped.reset_index().sort_values("cv")
    facts = [
        {"category": "consistency", "metric": "coefficient_of_variation", "driver": r.Driver,
         "value": round(r.cv, 5), "chart_hint": "bar"}
        for r in grouped.itertuples()
    ]
    chart = _ranked_bar(grouped, "Driver", "cv", "Coefficient of Variation by Driver", "Std dev / mean",
                         highlight="best", value_fmt="{:.4f}")
    return {"facts": facts, "chart": chart}


def interquartile_range(clean_laps: pd.DataFrame) -> dict:
    order = clean_laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().index.tolist()
    grouped = [clean_laps.loc[clean_laps["Driver"] == d, "LapTimeSeconds"].values for d in order]
    iqr_facts = []
    for d, values in zip(order, grouped):
        q1, q3 = np.percentile(values, [25, 75])
        iqr_facts.append({"category": "consistency", "metric": "interquartile_range", "driver": d,
                           "value": round(float(q3 - q1), 3), "chart_hint": "box_plot"})

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(order)), 5))
    box = ax.boxplot(grouped, tick_labels=order, patch_artist=True, widths=0.55,
                       medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
                       whiskerprops=dict(color=BASELINE), capprops=dict(color=BASELINE),
                       flierprops=dict(markeredgecolor=INK_MUTED, markersize=3))
    for patch in box["boxes"]:
        patch.set_facecolor(SEQ_BLUE); patch.set_alpha(0.75); patch.set_edgecolor(SEQ_BLUE)
    ax.grid(axis="x", visible=False)
    _style_axes(ax, ylabel="Lap time (s)")
    _title(ax, "Lap Time Distribution by Driver (IQR)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return {"facts": iqr_facts, "chart": fig}


def consistency_band_pct(clean_laps: pd.DataFrame, band: float = 1.02) -> dict:
    best = clean_laps.groupby("Driver")["LapTimeSeconds"].transform("min")
    within_band = clean_laps["LapTimeSeconds"] <= best * band
    df = within_band.groupby(clean_laps["Driver"]).mean().mul(100).sort_values(ascending=False).reset_index(name="PctWithinBand")
    facts = [
        {"category": "consistency", "metric": "consistency_band_pct", "driver": r.Driver,
         "value": round(r.PctWithinBand, 1), "band": band, "chart_hint": "bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df.sort_values("PctWithinBand"), "Driver", "PctWithinBand",
                         f"% of Laps Within {int((band - 1) * 100)}% of Own Best Lap", "% of laps",
                         highlight="worst", value_fmt="{:.1f}%")
    return {"facts": facts, "chart": chart}


def lap_time_range(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby(["Driver", "Stint"])["LapTimeSeconds"].agg(lambda s: s.max() - s.min())
    df = df.reset_index(name="Range").dropna()
    facts = [
        {"category": "consistency", "metric": "lap_time_range", "driver": r.Driver, "stint": r.Stint,
         "value": round(r.Range, 3), "chart_hint": "grouped_bar"}
        for r in df.itertuples()
    ]
    pivot = df.pivot(index="Driver", columns="Stint", values="Range")
    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(pivot)), 5))
    pivot.plot(kind="bar", ax=ax, color=CATEGORICAL[: pivot.shape[1]], width=0.8)
    ax.grid(axis="x", visible=False)
    _style_axes(ax, ylabel="Lap time range (s)")
    _title(ax, "Lap Time Range by Driver and Stint")
    ax.legend(title="Stint", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


# ===========================================================================
# 3. Tyre Strategy & Degradation
# ===========================================================================
def stint_composition(clean_laps: pd.DataFrame) -> dict:
    grouped = clean_laps.groupby(["Driver", "Stint"]).agg(
        Compound=("Compound", "first"), Laps=("LapNumber", "count"),
        StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"),
    ).reset_index()
    facts = [
        {"category": "tyre_strategy", "metric": "stint_composition", "driver": r.Driver, "stint": r.Stint,
         "compound": r.Compound, "laps": int(r.Laps), "start_lap": int(r.StartLap), "end_lap": int(r.EndLap),
         "chart_hint": "gantt"}
        for r in grouped.itertuples()
    ]
    compound_color = {"MEDIUM": "#eda100", "HARD": "#c3c2b7", "SOFT": "#d03b3b"}
    drivers = sorted(grouped["Driver"].unique(), key=lambda d: grouped[grouped.Driver == d]["StartLap"].min())
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(drivers) + 1))
    for i, d in enumerate(drivers):
        rows = grouped[grouped["Driver"] == d]
        for r in rows.itertuples():
            ax.barh(i, r.EndLap - r.StartLap + 1, left=r.StartLap,
                     color=compound_color.get(r.Compound, SEQ_BLUE), edgecolor=INK_MUTED, linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(drivers))); ax.set_yticklabels(drivers); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Lap number")
    _title(ax, "Stint Composition (tyre strategy)")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def pit_stop_count(clean_laps: pd.DataFrame) -> dict:
    return {"facts": [{"category": "tyre_strategy", "metric": "pit_stop_count", "driver": None, "value": None,
                        "note": "Not computable from clean_laps — pit in/out laps are excluded by the "
                                "clean-lap filter itself (PitInTime is null for every row here). "
                                "Needs the raw, unfiltered session.laps.", "chart_hint": None}],
            "chart": None}


def compound_usage_split(clean_laps: pd.DataFrame) -> dict:
    counts = clean_laps["Compound"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    facts = [
        {"category": "tyre_strategy", "metric": "compound_usage_pct", "compound": compound,
         "value": float(p), "chart_hint": "stacked_bar"}
        for compound, p in pct.items()
    ]
    fig, ax = plt.subplots(figsize=(7, 1.6))
    left = 0
    colors = {"MEDIUM": "#eda100", "HARD": "#c3c2b7", "SOFT": "#d03b3b"}
    for compound, p in pct.items():
        ax.barh(0, p, left=left, color=colors.get(compound, SEQ_BLUE), label=f"{compound} ({p}%)")
        left += p
    ax.set_yticks([]); ax.set_xlim(0, 100)
    _style_axes(ax, xlabel="% of laps")
    _title(ax, "Compound Usage Split (field-wide)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.6), ncol=len(pct))
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def degradation_slope(clean_laps: pd.DataFrame) -> dict:
    facts = []
    longest_stint_rows = []
    for (driver, stint), g in clean_laps.groupby(["Driver", "Stint"]):
        g = g.dropna(subset=["TyreLife", "LapTimeSeconds"])
        if len(g) < 3:
            continue
        slope = float(np.polyfit(g["TyreLife"], g["LapTimeSeconds"], 1)[0])
        facts.append({"category": "tyre_degradation", "metric": "degradation_slope_sec_per_lap",
                       "driver": driver, "stint": stint, "compound": g["Compound"].iloc[0],
                       "value": round(slope, 4), "n_laps": len(g), "chart_hint": "scatter_trendline"})
        longest_stint_rows.append((driver, stint, len(g), slope))

    # chart: each driver's longest stint (most laps = most reliable slope estimate)
    best_per_driver = {}
    for driver, stint, n, slope in longest_stint_rows:
        if driver not in best_per_driver or n > best_per_driver[driver][1]:
            best_per_driver[driver] = (stint, n, slope)
    df = pd.DataFrame(
        [{"Driver": d, "Slope": v[2]} for d, v in best_per_driver.items()]
    ).sort_values("Slope")
    chart = _ranked_bar(df, "Driver", "Slope", "Tyre Degradation Slope (longest stint per driver)",
                         "Seconds / lap of tyre life", highlight="best", value_fmt="{:+.4f}")
    return {"facts": facts, "chart": chart}


def early_vs_late_stint_pace(clean_laps: pd.DataFrame, n_laps: int = 5) -> dict:
    facts = []
    rows = []
    for (driver, stint), g in clean_laps.sort_values("LapNumber").groupby(["Driver", "Stint"]):
        if len(g) < n_laps * 2:
            continue
        early = g["LapTimeSeconds"].head(n_laps).mean()
        late = g["LapTimeSeconds"].tail(n_laps).mean()
        facts.append({"category": "tyre_degradation", "metric": "early_vs_late_stint_pace",
                       "driver": driver, "stint": stint, "early_mean": round(early, 3),
                       "late_mean": round(late, 3), "value": round(late - early, 3), "chart_hint": "slope_chart"})
        rows.append((driver, stint, len(g), early, late))

    best_per_driver = {}
    for driver, stint, n, early, late in rows:
        if driver not in best_per_driver or n > best_per_driver[driver][0]:
            best_per_driver[driver] = (n, early, late)
    df = pd.DataFrame([{"Driver": d, "Early": v[1], "Late": v[2]} for d, v in best_per_driver.items()])
    df = df.sort_values("Early")

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1))
    y = range(len(df))
    ax.hlines(y, df["Early"], df["Late"], color=BASELINE, linewidth=1.5)
    ax.scatter(df["Early"], y, color=SEQ_BLUE_LIGHT, s=40, zorder=2, label=f"First {n_laps} laps")
    ax.scatter(df["Late"], y, color=ACCENT_ORANGE, s=40, zorder=3, label=f"Last {n_laps} laps")
    ax.set_yticks(list(y)); ax.set_yticklabels(df["Driver"]); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Lap time (s)")
    _title(ax, f"Early vs Late Stint Pace (longest stint per driver, first/last {n_laps} laps)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def fresh_vs_used_tyre_delta(clean_laps: pd.DataFrame) -> dict:
    grouped = clean_laps.groupby(["Driver", "FreshTyre"])["LapTimeSeconds"].mean().unstack()
    grouped = grouped.rename(columns={True: "Fresh", False: "Used"}).dropna()
    facts = []
    for driver, r in grouped.iterrows():
        facts.append({"category": "tyre_degradation", "metric": "fresh_vs_used_tyre_delta", "driver": driver,
                       "fresh_mean": round(r["Fresh"], 3), "used_mean": round(r["Used"], 3),
                       "value": round(r["Used"] - r["Fresh"], 3), "chart_hint": "grouped_bar"})
    df = grouped.reset_index().sort_values("Fresh")
    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(df)), 5))
    x = np.arange(len(df)); width = 0.38
    ax.bar(x - width / 2, df["Fresh"], width, color=SEQ_BLUE_LIGHT, label="Fresh tyre laps")
    ax.bar(x + width / 2, df["Used"], width, color=ACCENT_ORANGE, label="Used tyre laps")
    ax.set_xticks(x); ax.set_xticklabels(df["Driver"], rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    _style_axes(ax, ylabel="Mean lap time (s)")
    _title(ax, "Fresh vs Used Tyre Pace by Driver")
    ax.legend(frameon=False)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def cliff_detection(clean_laps: pd.DataFrame, threshold: float = 1.5) -> dict:
    df = clean_laps.sort_values(["Driver", "Stint", "LapNumber"]).copy()
    df["Delta"] = df.groupby(["Driver", "Stint"])["LapTimeSeconds"].diff()
    cliffs = df[df["Delta"] > threshold]
    facts = [
        {"category": "tyre_degradation", "metric": "cliff_detection", "driver": r.Driver, "stint": r.Stint,
         "lap_number": int(r.LapNumber), "value": round(r.Delta, 3), "threshold": threshold,
         "chart_hint": "annotated_line"}
        for r in cliffs.itertuples()
    ]

    # illustrative chart: the driver/stint with the most flagged cliffs
    if len(cliffs):
        top_key = cliffs.groupby(["Driver", "Stint"]).size().idxmax()
        driver, stint = top_key
    else:
        driver, stint = df.iloc[0][["Driver", "Stint"]]
    trace = df[(df["Driver"] == driver) & (df["Stint"] == stint)].sort_values("LapNumber")
    flagged = trace[trace["Delta"] > threshold]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trace["LapNumber"], trace["LapTimeSeconds"], color=SEQ_BLUE, linewidth=2)
    ax.scatter(flagged["LapNumber"], flagged["LapTimeSeconds"], color=STATUS_CRITICAL, s=50, zorder=3,
               label=f"Cliff (> {threshold}s jump)")
    _style_axes(ax, xlabel="Lap number", ylabel="Lap time (s)")
    _title(ax, f"Cliff Detection — {driver}, Stint {int(stint)}")
    if len(flagged):
        ax.legend(frameon=False)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


# ===========================================================================
# 4. Sector Breakdown
# ===========================================================================
def median_sector_times(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.copy()
    for s in (1, 2, 3):
        df[f"Sector{s}Seconds"] = _seconds(df[f"Sector{s}Time"])
    medians = df.groupby("Driver")[["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]].median()
    medians = medians.sort_values("Sector1Seconds")

    facts = []
    for driver, r in medians.iterrows():
        for s in (1, 2, 3):
            facts.append({"category": "sector_breakdown", "metric": "median_sector_time", "driver": driver,
                           "sector": s, "value": round(r[f"Sector{s}Seconds"], 3), "chart_hint": "stacked_bar"})

    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(medians)), 5))
    bottom = np.zeros(len(medians))
    for i, s in enumerate((1, 2, 3)):
        vals = medians[f"Sector{s}Seconds"].values
        ax.bar(medians.index, vals, bottom=bottom, color=CATEGORICAL[i], label=f"Sector {s}")
        bottom += vals
    ax.grid(axis="x", visible=False)
    _style_axes(ax, ylabel="Median sector time, stacked (s)")
    _title(ax, "Median Sector Times by Driver")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def theoretical_best_lap(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.copy()
    for s in (1, 2, 3):
        df[f"Sector{s}Seconds"] = _seconds(df[f"Sector{s}Time"])
    best_sectors = df.groupby("Driver")[["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]].min()
    theoretical = best_sectors.sum(axis=1)
    actual = df.groupby("Driver")["LapTimeSeconds"].min()
    merged = pd.DataFrame({"Theoretical": theoretical, "Actual": actual}).dropna()
    merged["Gap"] = merged["Actual"] - merged["Theoretical"]
    merged = merged.sort_values("Gap").reset_index()

    facts = [
        {"category": "sector_breakdown", "metric": "theoretical_best_lap_gap", "driver": r.Driver,
         "theoretical_seconds": round(r.Theoretical, 3), "actual_seconds": round(r.Actual, 3),
         "value": round(r.Gap, 3), "chart_hint": "gap_bar"}
        for r in merged.itertuples()
    ]
    chart = _ranked_bar(merged, "Driver", "Gap", "Gap Between Actual and Theoretical Best Lap",
                         "Actual − theoretical (s)", highlight="best")
    return {"facts": facts, "chart": chart}


def sector_delta_to_fastest(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.copy()
    for s in (1, 2, 3):
        df[f"Sector{s}Seconds"] = _seconds(df[f"Sector{s}Time"])
    medians = df.groupby("Driver")[["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]].median()
    deltas = medians - medians.min()
    deltas = deltas.sort_values("Sector1Seconds")

    facts = []
    for driver, r in deltas.iterrows():
        for s in (1, 2, 3):
            facts.append({"category": "sector_breakdown", "metric": "sector_delta_to_fastest", "driver": driver,
                           "sector": s, "value": round(r[f"Sector{s}Seconds"], 3), "chart_hint": "heatmap"})

    fig, ax = plt.subplots(figsize=(5, 0.32 * len(deltas) + 1))
    matrix = deltas[["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]].values
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["S1", "S2", "S3"])
    ax.set_yticks(range(len(deltas))); ax.set_yticklabels(deltas.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if matrix[i, j] > matrix.max() * 0.6 else INK_PRIMARY)
    ax.grid(False)
    _title(ax, "Sector Delta to Field-Fastest (s)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def sector_trend_over_race(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.copy()
    for s in (1, 2, 3):
        df[f"Sector{s}Seconds"] = _seconds(df[f"Sector{s}Time"])
    leaders = _top_leaders(clean_laps)

    facts = []
    for s in (1, 2, 3):
        for r in df.itertuples():
            val = getattr(r, f"Sector{s}Seconds")
            if pd.notna(val):
                facts.append({"category": "sector_breakdown", "metric": "sector_trend", "driver": r.Driver,
                               "sector": s, "lap_number": int(r.LapNumber), "value": round(val, 3),
                               "chart_hint": "small_multiple_line"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for i, s in enumerate((1, 2, 3)):
        ax = axes[i]
        for key, g in df.groupby("Driver"):
            g = g.sort_values("LapNumber")
            if key in leaders:
                continue
            ax.plot(g["LapNumber"], g[f"Sector{s}Seconds"], color=CONTEXT_GRAY, linewidth=0.8, alpha=0.4)
        for j, key in enumerate(leaders):
            g = df[df["Driver"] == key].sort_values("LapNumber")
            ax.plot(g["LapNumber"], g[f"Sector{s}Seconds"], color=CATEGORICAL[j % len(CATEGORICAL)], linewidth=1.6)
        _style_axes(ax, xlabel="Lap number", ylabel="Sector time (s)" if i == 0 else None)
        _title(ax, f"Sector {s}")
    fig.suptitle(f"Sector Time Trend Over the Race — Top {len(leaders)} Highlighted", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return {"facts": facts, "chart": fig}


# ===========================================================================
# 5. Speed Trap / Top Speed
# ===========================================================================
def max_speed_by_trap(clean_laps: pd.DataFrame) -> dict:
    traps = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    maxes = clean_laps.groupby("Driver")[traps].max().sort_values("SpeedST", ascending=False)
    facts = []
    for driver, r in maxes.iterrows():
        for trap in traps:
            facts.append({"category": "speed_trap", "metric": "max_speed", "driver": driver, "trap": trap,
                           "value": float(r[trap]), "chart_hint": "grouped_bar"})

    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(maxes)), 5))
    x = np.arange(len(maxes)); width = 0.2
    for i, trap in enumerate(traps):
        ax.bar(x + (i - 1.5) * width, maxes[trap], width, color=CATEGORICAL[i], label=trap)
    ax.set_xticks(x); ax.set_xticklabels(maxes.index, rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    _style_axes(ax, ylabel="Max speed (km/h)")
    _title(ax, "Max Speed by Trap, per Driver")
    ax.legend(frameon=False)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def avg_speed_trap(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby("Driver")["SpeedST"].mean().sort_values(ascending=False).reset_index()
    facts = [
        {"category": "speed_trap", "metric": "avg_speed_trap", "driver": r.Driver,
         "value": round(r.SpeedST, 1), "chart_hint": "ranked_bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df, "Driver", "SpeedST", "Average Speed Trap (straight-line proxy)",
                         "Mean SpeedST (km/h)", highlight="best", value_fmt="{:.1f}")
    return {"facts": facts, "chart": chart}


def speed_trap_trend(clean_laps: pd.DataFrame) -> dict:
    leaders = _top_leaders(clean_laps)
    facts = [
        {"category": "speed_trap", "metric": "speed_trap_trend", "driver": r.Driver,
         "lap_number": int(r.LapNumber), "value": float(r.SpeedST), "chart_hint": "line"}
        for r in clean_laps.dropna(subset=["SpeedST"]).itertuples()
    ]
    chart = _emphasis_line(clean_laps.dropna(subset=["SpeedST"]), "LapNumber", "SpeedST", "Driver", leaders,
                            "Speed Trap Trend Over the Race (fuel-burn effect)", "Lap number", "SpeedST (km/h)")
    return {"facts": facts, "chart": chart}


# ===========================================================================
# 6. Race Position & Story
# ===========================================================================
def position_per_lap(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.dropna(subset=["Position"])
    leaders = _top_leaders(clean_laps)
    facts = [
        {"category": "race_story", "metric": "position_per_lap", "driver": r.Driver,
         "lap_number": int(r.LapNumber), "value": int(r.Position), "chart_hint": "position_line"}
        for r in df.itertuples()
    ]
    chart = _emphasis_line(df, "LapNumber", "Position", "Driver", leaders,
                            f"Race Position by Lap — Top {len(leaders)} Highlighted", "Lap number",
                            "Position", invert_y=True)
    return {"facts": facts, "chart": chart}


def net_position_change(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.dropna(subset=["Position"]).sort_values("LapNumber")
    first = df.groupby("Driver").first()["Position"]
    last = df.groupby("Driver").last()["Position"]
    net = (first - last).sort_values(ascending=False).reset_index(name="NetChange")
    facts = [
        {"category": "race_story", "metric": "net_position_change", "driver": r.Driver,
         "value": int(r.NetChange), "chart_hint": "diverging_bar"}
        for r in net.itertuples()
    ]
    chart = _diverging_bar(net, "Driver", "NetChange", "Net Position Change (positive = gained places)",
                            "Positions gained (+) / lost (−)")
    return {"facts": facts, "chart": chart}


def laps_led(clean_laps: pd.DataFrame) -> dict:
    df = (clean_laps["Position"] == 1).groupby(clean_laps["Driver"]).sum().sort_values(ascending=False)
    df = df[df > 0].reset_index(name="LapsLed")
    facts = [
        {"category": "race_story", "metric": "laps_led", "driver": r.Driver, "value": int(r.LapsLed),
         "chart_hint": "bar"}
        for r in df.itertuples()
    ]
    chart = None
    if len(df):
        chart = _ranked_bar(df.sort_values("LapsLed"), "Driver", "LapsLed", "Laps Led", "Laps led",
                             highlight="worst", value_fmt="{:.0f}")
    return {"facts": facts, "chart": chart}


def position_volatility(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby("Driver")["Position"].std().sort_values().reset_index(name="Volatility").dropna()
    facts = [
        {"category": "race_story", "metric": "position_volatility", "driver": r.Driver,
         "value": round(r.Volatility, 3), "chart_hint": "bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df, "Driver", "Volatility", "Position Volatility by Driver (lower = more stable)",
                         "Std dev of position", highlight="best")
    return {"facts": facts, "chart": chart}


# ===========================================================================
# 7. Head-to-Head Duels
# ===========================================================================
def _h2h_frame(clean_laps: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    a = clean_laps[clean_laps["Driver"] == driver_a].set_index("LapNumber")["LapTimeSeconds"].dropna()
    b = clean_laps[clean_laps["Driver"] == driver_b].set_index("LapNumber")["LapTimeSeconds"].dropna()
    common = a.index.intersection(b.index).sort_values()
    df = pd.DataFrame({"LapNumber": common, "A": a.loc[common].values, "B": b.loc[common].values})
    df["LapDelta"] = df["A"] - df["B"]
    df["CumulativeGap"] = df["LapDelta"].cumsum()
    return df


def head_to_head_lap_delta(clean_laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    df = _h2h_frame(clean_laps, driver_a, driver_b)
    facts = [
        {"category": "head_to_head", "metric": "lap_time_delta", "driver_a": driver_a, "driver_b": driver_b,
         "lap_number": int(r.LapNumber), "value": round(r.LapDelta, 3), "chart_hint": "line_zero"}
        for r in df.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [STATUS_GOOD if v < 0 else STATUS_CRITICAL for v in df["LapDelta"]]
    ax.bar(df["LapNumber"], df["LapDelta"], color=colors, width=0.8)
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axes(ax, xlabel="Lap number", ylabel=f"{driver_a} − {driver_b} (s)")
    _title(ax, f"Lap Time Delta: {driver_a} vs {driver_b} (below zero = {driver_a} faster)")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def cumulative_gap(clean_laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    df = _h2h_frame(clean_laps, driver_a, driver_b)
    facts = [
        {"category": "head_to_head", "metric": "cumulative_gap", "driver_a": driver_a, "driver_b": driver_b,
         "lap_number": int(r.LapNumber), "value": round(r.CumulativeGap, 3), "chart_hint": "race_gap_chart"}
        for r in df.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["LapNumber"], df["CumulativeGap"], color=SEQ_BLUE, linewidth=2)
    ax.fill_between(df["LapNumber"], df["CumulativeGap"], 0,
                     color=STATUS_GOOD if df["CumulativeGap"].iloc[-1] < 0 else STATUS_CRITICAL, alpha=0.12)
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axes(ax, xlabel="Lap number", ylabel=f"Cumulative gap, {driver_a} − {driver_b} (s)")
    _title(ax, f"Race Gap: {driver_a} vs {driver_b} (below zero = {driver_a} ahead)")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def lap_win_count(clean_laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    df = _h2h_frame(clean_laps, driver_a, driver_b)
    a_wins = int((df["LapDelta"] < 0).sum())
    b_wins = int((df["LapDelta"] > 0).sum())
    ties = int((df["LapDelta"] == 0).sum())
    facts = [{"category": "head_to_head", "metric": "lap_win_count", "driver_a": driver_a, "driver_b": driver_b,
              "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "laps_compared": len(df),
              "chart_hint": "scoreboard"}]
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.barh(0, a_wins, color=SEQ_BLUE, label=f"{driver_a} ({a_wins})")
    ax.barh(0, b_wins, left=a_wins, color=ACCENT_ORANGE, label=f"{driver_b} ({b_wins})")
    ax.set_yticks([]); ax.set_xlim(0, max(len(df), 1))
    _style_axes(ax, xlabel="Laps won")
    _title(ax, f"Lap-Win Count: {driver_a} vs {driver_b}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2)
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


# ===========================================================================
# 8. Team-Level Aggregates
# ===========================================================================
def team_median_pace(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby("Team")["LapTimeSeconds"].median().sort_values().reset_index()
    facts = [
        {"category": "team", "metric": "team_median_pace", "team": r.Team, "value": round(r.LapTimeSeconds, 3),
         "chart_hint": "ranked_bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df, "Team", "LapTimeSeconds", "Team Median Pace (constructor order)",
                         "Median lap time (s)", highlight="best")
    return {"facts": facts, "chart": chart}


def intra_team_teammate_gap(clean_laps: pd.DataFrame) -> dict:
    per_driver = clean_laps.groupby(["Team", "Driver"])["LapTimeSeconds"].median().reset_index()
    rows = []
    for team, g in per_driver.groupby("Team"):
        if len(g) < 2:
            continue
        vals = g["LapTimeSeconds"].values
        rows.append({"Team": team, "Gap": abs(vals[0] - vals[1]),
                      "Drivers": ", ".join(g["Driver"])})
    df = pd.DataFrame(rows).sort_values("Gap")
    facts = [
        {"category": "team", "metric": "intra_team_teammate_gap", "team": r.Team, "drivers": r.Drivers,
         "value": round(r.Gap, 3), "chart_hint": "bar"}
        for r in df.itertuples()
    ]
    chart = None
    if len(df):
        chart = _ranked_bar(df, "Team", "Gap", "Intra-Team Teammate Pace Gap", "Abs. median pace gap (s)",
                             highlight="best")
    return {"facts": facts, "chart": chart}


def team_pit_stop_timing(clean_laps: pd.DataFrame) -> dict:
    return {"facts": [{"category": "team", "metric": "team_pit_stop_timing", "team": None, "value": None,
                        "note": "Not computable from clean_laps — pit in/out laps are excluded by the "
                                "clean-lap filter itself. Needs the raw, unfiltered session.laps.",
                        "chart_hint": None}],
            "chart": None}


# ===========================================================================
# 9. Personal Bests & Highlights
# ===========================================================================
def personal_best_timing(clean_laps: pd.DataFrame) -> dict:
    pbs = clean_laps[clean_laps["IsPersonalBest"] == True].sort_values("LapNumber")
    facts = [
        {"category": "highlights", "metric": "personal_best", "driver": r.Driver,
         "lap_number": int(r.LapNumber), "value": round(r.LapTimeSeconds, 3), "chart_hint": "timeline_dots"}
        for r in pbs.itertuples()
    ]
    order = clean_laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().index.tolist()
    y_pos = {d: i for i, d in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(order) + 1))
    ax.scatter(pbs["LapNumber"], pbs["Driver"].map(y_pos), color=ACCENT_ORANGE, s=35, zorder=2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Lap number")
    _title(ax, "Personal-Best Lap Timing")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def fastest_lap_headline(clean_laps: pd.DataFrame) -> dict:
    row = clean_laps.loc[clean_laps["LapTimeSeconds"].idxmin()]
    facts = [{"category": "highlights", "metric": "fastest_lap_headline", "driver": row["Driver"],
              "value": round(float(row["LapTimeSeconds"]), 3), "lap_number": int(row["LapNumber"]),
              "compound": row["Compound"], "chart_hint": "stat_card"}]
    chart = _stat_card(
        [(f"{row['LapTimeSeconds']:.3f}s", f"Fastest lap — {row['Driver']}"),
         (f"Lap {int(row['LapNumber'])}", f"{row['Compound'].title()} tyre")],
        "Fastest Lap of the Race",
    )
    return {"facts": facts, "chart": chart}


def session_best_sector_moments(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.copy()
    for s in (1, 2, 3):
        df[f"Sector{s}Seconds"] = _seconds(df[f"Sector{s}Time"])
    facts = []
    moments = []
    for s in (1, 2, 3):
        col = f"Sector{s}Seconds"
        best_val = df[col].min()
        best_rows = df[df[col] == best_val]
        for r in best_rows.itertuples():
            facts.append({"category": "highlights", "metric": "session_best_sector", "driver": r.Driver,
                           "sector": s, "lap_number": int(r.LapNumber), "value": round(best_val, 3),
                           "chart_hint": "annotated_marker"})
            moments.append((r.LapNumber, s, r.Driver))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for lap, sector, driver in moments:
        ax.scatter(lap, sector, color=CATEGORICAL[(sector - 1) % len(CATEGORICAL)], s=80, zorder=2)
        ax.annotate(driver, (lap, sector), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["Sector 1", "Sector 2", "Sector 3"])
    ax.set_ylim(0.5, 3.5)
    _style_axes(ax, xlabel="Lap number")
    _title(ax, "Session-Best (\"Purple\") Sector Moments")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


# ===========================================================================
# 10. Data Quality / Context Flags
# ===========================================================================
def excluded_laps_note(clean_laps: pd.DataFrame) -> dict:
    facts = [{"category": "data_quality", "metric": "excluded_laps_note", "value": None,
              "clean_lap_count": int(len(clean_laps)),
              "note": "clean_laps is already filtered to TrackStatus == '1' with no pit in/out lap — "
                      "the raw excluded-lap count needs the unfiltered session.laps for comparison.",
              "chart_hint": None}]
    return {"facts": facts, "chart": None}


def deleted_lap_count(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps[clean_laps["Deleted"] == True]
    counts = df.groupby("Driver").size().sort_values(ascending=False)
    facts = [
        {"category": "data_quality", "metric": "deleted_lap_count", "driver": driver, "value": int(count),
         "chart_hint": "table"}
        for driver, count in counts.items()
    ]
    if not len(counts):
        return {"facts": [{"category": "data_quality", "metric": "deleted_lap_count", "value": 0,
                            "note": "No deleted laps in clean_laps.", "chart_hint": None}], "chart": None}

    reasons = df.groupby("Driver")["DeletedReason"].agg(lambda s: "; ".join(sorted(set(s.dropna()))))
    table_rows = [[driver, str(count), reasons.get(driver, "")] for driver, count in counts.items()]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(table_rows) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=["Driver", "Deleted laps", "Reason"],
                    loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    _title(ax, "Deleted Lap Count & Reasons")
    fig.tight_layout()
    return {"facts": facts, "chart": fig}


def accuracy_ratio(clean_laps: pd.DataFrame) -> dict:
    df = clean_laps.groupby("Driver")["IsAccurate"].mean().mul(100).sort_values().reset_index(name="AccuracyPct")
    facts = [
        {"category": "data_quality", "metric": "accuracy_ratio_pct", "driver": r.Driver,
         "value": round(r.AccuracyPct, 1), "chart_hint": "badge_bar"}
        for r in df.itertuples()
    ]
    chart = _ranked_bar(df, "Driver", "AccuracyPct", "Lap-Accuracy Ratio by Driver", "% laps marked accurate",
                         highlight="worst", value_fmt="{:.1f}%")
    return {"facts": facts, "chart": chart}
