"""Visualizations for analysis_suite/pace_analysis.py — one chart per analysis.

Color rules follow the dataviz skill's validated default palette:
magnitude comparisons (rankings, distributions) use a single sequential hue;
identity/emphasis (the race-fastest driver, the top pace-setters) is called
out with the accent hue while everything else recedes to gray.
"""
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis_suite"))  # pace_analysis's own `from basic_imports import *`

import matplotlib.pyplot as plt

import analysis_suite.pace_analysis as pa

# ---------------------------------------------------------------------------
# Chart chrome (validated light-mode palette — see dataviz skill reference)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQ_BLUE = "#2a78d6"       # sequential default hue, step 450 — magnitude comparisons
SEQ_BLUE_DARK = "#256abf"  # step 500 — dumbbell "shade 2"
SEQ_BLUE_LIGHT = "#86b6ef" # step 250 — dumbbell "shade 1"
ACCENT_ORANGE = "#eb6834"  # emphasis highlight — the one bar/line that's "the point"
CONTEXT_GRAY = "#c3c2b7"   # de-emphasised context in emphasis charts

CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
]  # fixed order, first six slots of the validated 8-hue set (adjacent-safe)

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


def _style_axes(ax, xlabel=None, ylabel=None):
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=13, fontweight="bold", color=INK_PRIMARY)


# ---------------------------------------------------------------------------
# 1. Median race pace per driver -> ranked horizontal bar (magnitude, sequential)
# ---------------------------------------------------------------------------
def plot_median_pace(df=pa.median_pace_drivers_df):
    data = df.sort_values("LapTimeSeconds", ascending=True)
    colors = [ACCENT_ORANGE if i == 0 else SEQ_BLUE for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1))
    bars = ax.barh(data["Driver"], data["LapTimeSeconds"], color=colors)
    ax.invert_yaxis()  # fastest driver on top
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Median lap time (s)")
    _title(ax, "Median Race Pace by Driver")

    for bar, value in zip(bars, data["LapTimeSeconds"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.3f}s",
                 va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Fastest lap per driver -> ranked horizontal bar, race-fastest lap called out
# ---------------------------------------------------------------------------
def plot_fastest_lap_per_driver(df=pa.fastest_laps_per_driver_df):
    data = df.sort_values("LapTimeSeconds", ascending=True)
    colors = [ACCENT_ORANGE if drv == pa.fastest_lap_driver else SEQ_BLUE for drv in data["Driver"]]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1))
    bars = ax.barh(data["Driver"], data["LapTimeSeconds"], color=colors)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Fastest lap time (s)")
    _title(ax, "Fastest Lap by Driver")
    ax.text(0.99, 1.06,
             f"Race fastest: {pa.fastest_lap_driver} — {pa.fastest_lap_time:.3f}s "
             f"(lap {int(pa.fastest_lap_lap_number)}, {pa.fastest_lap_compound})",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)

    for bar, value in zip(bars, data["LapTimeSeconds"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.3f}s",
                 va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Mean vs median pace per driver -> dumbbell (1 hue, 2 shades; shows skew)
# ---------------------------------------------------------------------------
def plot_mean_pace(mean_df=pa.mean_lap_time_per_driver_df, median_df=pa.median_pace_drivers_df):
    data = (
        mean_df.rename(columns={"LapTimeSeconds": "Mean"})
        .merge(median_df.rename(columns={"LapTimeSeconds": "Median"}), on="Driver")
        .sort_values("Mean", ascending=True)
        .reset_index(drop=True)
    )
    y = range(len(data))

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1))
    ax.hlines(y, data["Median"], data["Mean"], color=BASELINE, linewidth=1.5, zorder=1)
    ax.scatter(data["Median"], y, color=SEQ_BLUE_LIGHT, s=45, zorder=2, label="Median")
    ax.scatter(data["Mean"], y, color=SEQ_BLUE_DARK, s=45, zorder=3, label="Mean")

    ax.set_yticks(list(y))
    ax.set_yticklabels(data["Driver"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _style_axes(ax, xlabel="Lap time (s)")
    _title(ax, "Mean vs Median Pace by Driver")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.02, 0.01, "Wider gap = more skew from slow outlier laps (traffic, lockups, etc.)",
              fontsize=8, color=INK_SECONDARY, ha="left")
    return fig


# ---------------------------------------------------------------------------
# 4. Pace delta to the field's fastest median pace -> gap-to-leader bar
#    (always >= 0 by construction, since field_min_median is the field's own
#    minimum — a true diverging/bipolar chart would need a negative pole that
#    doesn't exist here, so this renders as a bar from a zero baseline)
# ---------------------------------------------------------------------------
def plot_pace_delta(df=pa.pace_delta_to_fastest_lap_df):
    data = df.sort_values("PaceDelta", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1))
    colors = [ACCENT_ORANGE if i == 0 else SEQ_BLUE for i in range(len(data))]
    bars = ax.barh(data["Driver"], data["PaceDelta"], color=colors)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASELINE, linewidth=1)
    _style_axes(ax, xlabel="Median pace delta to field-fastest median (s)")
    _title(ax, "Pace Delta to Fastest Median Pace, by Driver")

    for bar, value in zip(bars, data["PaceDelta"]):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}s",
                 va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Rolling pace over the race -> emphasis line chart (top-N highlighted, rest gray)
# ---------------------------------------------------------------------------
def plot_rolling_pace(df=pa.rolling_pace_df, top_n=6):
    leaders = pa.median_pace_drivers_df.sort_values("LapTimeSeconds")["Driver"].head(top_n).tolist()

    fig, ax = plt.subplots(figsize=(11, 6))

    for drv, drv_laps in df.groupby("Driver"):
        if drv in leaders:
            continue
        drv_laps = drv_laps.sort_values("LapNumber")
        ax.plot(drv_laps["LapNumber"], drv_laps["RollingPace"], color=CONTEXT_GRAY,
                 linewidth=1, alpha=0.5, zorder=1, solid_capstyle="round")

    for i, drv in enumerate(leaders):
        drv_laps = df[df["Driver"] == drv].sort_values("LapNumber")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(drv_laps["LapNumber"], drv_laps["RollingPace"], color=color,
                 linewidth=2, zorder=2, solid_capstyle="round", label=drv)

    _style_axes(ax, xlabel="Lap number", ylabel="Rolling pace, 3-lap avg (s)")
    _title(ax, f"Rolling Race Pace — Top {top_n} Drivers Highlighted")
    ax.legend(frameon=False, ncol=top_n, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plot_median_pace()
    plot_fastest_lap_per_driver()
    plot_mean_pace()
    plot_pace_delta()
    plot_rolling_pace()
    plt.show()
