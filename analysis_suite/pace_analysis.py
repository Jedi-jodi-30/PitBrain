from analysis_suite.basic_imports import *
import pandas as pd


class GlobalPaceAnalysis:
    """Pace analysis computed once, across the whole field."""

    def __init__(self, clean_laps: pd.DataFrame):
        self.clean_laps = clean_laps

    def median_pace(self) -> pd.DataFrame:
        median_pace_drivers = self.clean_laps.groupby('Driver')['LapTimeSeconds'].median().sort_values()
        return pd.DataFrame(median_pace_drivers).reset_index()

    def fastest_lap(self) -> pd.Series:
        return self.clean_laps.loc[self.clean_laps['LapTimeSeconds'].idxmin()]

    def fastest_lap_per_driver(self) -> pd.DataFrame:
        fastest_laps = self.clean_laps.loc[self.clean_laps.groupby('Driver')['LapTimeSeconds'].idxmin()]
        return pd.DataFrame(fastest_laps).reset_index()

    def mean_pace(self) -> pd.DataFrame:
        mean_pace_drivers = self.clean_laps.groupby('Driver')['LapTimeSeconds'].mean().sort_values()
        return pd.DataFrame(mean_pace_drivers).reset_index()

    def pace_delta(self) -> pd.DataFrame:
        """Each driver's median pace minus the field's fastest median pace."""
        median_df = self.median_pace()
        field_min_median = median_df['LapTimeSeconds'].min()
        delta_df = median_df.copy()
        delta_df['PaceDelta'] = delta_df['LapTimeSeconds'] - field_min_median
        return delta_df

    def rolling_pace(self, window: int = 3) -> pd.DataFrame:
        rolling = self.clean_laps.sort_values(['Driver', 'LapNumber']).copy()
        rolling['RollingPace'] = rolling.groupby('Driver')['LapTimeSeconds'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        return rolling.reset_index(drop=True)


class DriverComparisonAnalysis:
    """Pace analysis narrowed to a specific set of drivers."""

    def __init__(self, clean_laps: pd.DataFrame, drivers: list[str] = None):
        self.clean_laps = clean_laps
        self.drivers = drivers

    def _laps_for(self) -> pd.DataFrame:
        return self.clean_laps[self.clean_laps['Driver'].isin(self.drivers)]

    def median_pace(self) -> pd.DataFrame:
        subset = self._laps_for()
        median = subset.groupby('Driver')['LapTimeSeconds'].median().sort_values()
        return pd.DataFrame(median).reset_index()

    def mean_pace(self) -> pd.DataFrame:
        subset = self._laps_for()
        mean = subset.groupby('Driver')['LapTimeSeconds'].mean().sort_values()
        return pd.DataFrame(mean).reset_index()

    def rolling_pace(self, window: int = 3) -> pd.DataFrame:
        subset = self._laps_for().sort_values(['Driver', 'LapNumber']).copy()
        subset['RollingPace'] = subset.groupby('Driver')['LapTimeSeconds'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        return subset.reset_index(drop=True)

    def head_to_head(self, driver_a: str, driver_b: str) -> pd.DataFrame:
        """Lap-by-lap gap between two drivers on laps they both completed.

        LapDelta = A's lap time - B's lap time (negative means A was faster
        that lap); CumulativeGap is the running sum, i.e. the race-time gap
        between them as of that lap.
        """
        a = self.clean_laps[self.clean_laps['Driver'] == driver_a].set_index('LapNumber')['LapTimeSeconds'].dropna()
        b = self.clean_laps[self.clean_laps['Driver'] == driver_b].set_index('LapNumber')['LapTimeSeconds'].dropna()
        common_laps = a.index.intersection(b.index).sort_values()

        result = pd.DataFrame({
            'LapNumber': common_laps,
            f'{driver_a}_LapTime': a.loc[common_laps].values,
            f'{driver_b}_LapTime': b.loc[common_laps].values,
            'LapDelta': (a.loc[common_laps] - b.loc[common_laps]).values,
        })
        result['CumulativeGap'] = result['LapDelta'].cumsum()
        return result

    def head_to_head_summary(self, driver_a: str, driver_b: str) -> dict:
        h2h = self.head_to_head(driver_a, driver_b)
        return {
            'driver_a': driver_a,
            'driver_b': driver_b,
            'laps_compared': len(h2h),
            'laps_a_faster': int((h2h['LapDelta'] < 0).sum()),
            'laps_b_faster': int((h2h['LapDelta'] > 0).sum()),
            'final_gap_seconds': float(h2h['CumulativeGap'].iloc[-1]) if len(h2h) else None,
            'lap_by_lap': h2h,
        }


class PaceAnalysis:
    """Loads a session's clean laps once and exposes both analysis types."""

    def __init__(self, year=2023, event='Monza', session_type='R', check_cache=True, session=None):
        self.session = session or get_session(year, event, session_type, check_cache=check_cache)
        self.clean_laps = get_clean_laps(self.session)
        self.global_analysis = GlobalPaceAnalysis(self.clean_laps)
        self.comparison = DriverComparisonAnalysis(self.clean_laps)


# analysis = PaceAnalysis()
# print(analysis.global_analysis.median_pace())
