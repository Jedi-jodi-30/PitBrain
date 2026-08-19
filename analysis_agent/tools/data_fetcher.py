import fastf1
import pandas as pd
from ollama import ChatResponse, chat
import os

def get_clean_laps(year: int, event: str, session_type: str):
    """
    Get clean laps from a FastF1 session.

    Parameters:
    year (int): The year of the session.
    event (str): The event name (e.g., 'Monza').
    session_type (str): The session type (e.g., 'R' for race).

    Returns:
    pd.DataFrame: A DataFrame containing clean laps with LapTime in seconds.
    """
    # Tool-call arguments come back as whatever the model actually sent, not
    # necessarily what the schema asked for (e.g. year as "2023" instead of
    # 2023) — coerce defensively rather than trust it.
    year = int(year)
    check_cache = True  # Default to True

    if check_cache:
       #create a cache directory for fastf1 if it doesn't exist
       os.makedirs('fastf1_cache', exist_ok=True)
       fastf1.Cache.enable_cache('fastf1_cache')
    session = fastf1.get_session(year, event, session_type)
    session.load()

    laps = session.laps
    clean_laps = laps.pick_track_status('1', how='equals')
    clean_laps = clean_laps[clean_laps['PitOutTime'].isna() & clean_laps['PitInTime'].isna()]
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()

    return clean_laps, session

def resolve_driver_code(session, name: str) -> str | None:
    """
    Resolve a driver name to this session's 3-letter FastF1 code.

    Accepts a code already ("VER"), a last name ("Verstappen"), a first name,
    or a full name — all case-insensitively. The intent parser only ever
    extracts free-text names from the user's query, and DriverComparisonAnalysis
    filters on the `Driver` column, which holds codes — so this is what bridges
    the two.

    Parameters:
    session (fastf1.core.Session): The loaded FastF1 session.
    name (str): The driver name or code to resolve.

    Returns:
    str | None: The 3-letter code, or None if no driver matched.
    """
    if not name:
        return None

    infos = [session.get_driver(num) for num in session.drivers]

    name_stripped = name.strip()
    for info in infos:
        if info['Abbreviation'].lower() == name_stripped.lower():
            return info['Abbreviation']

    name_lower = name_stripped.lower()
    for info in infos:
        if name_lower in (info['LastName'].lower(), info['FirstName'].lower(), info['FullName'].lower()):
            return info['Abbreviation']

    return None

def get_drivers_data_only(global_data: pd.DataFrame, driver_code: str):
    """
    Get driver data from clean laps.

    Parameters:
    global_data (pd.DataFrame): A DataFrame containing clean laps.
    driver_code (str): The driver's 3-letter FastF1 code (resolve names via
    resolve_driver_code() first — the `Driver` column holds codes, not names).

    Returns:
    pd.DataFrame: A DataFrame containing driver data.
    """
    drivers_data = global_data[global_data['Driver'] == driver_code].copy()
    return drivers_data

