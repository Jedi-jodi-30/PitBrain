import fastf1 as ff1
import pandas as pd

def get_session(year, event, session_type,check_cache=True):
    """
    Get a FastF1 session.

    Parameters:
    year (int): The year of the session.
    event (str): The event name (e.g., 'Monza').
    session_type (str): The session type (e.g., 'R' for race).  
    """
    if check_cache:
        ff1.Cache.enable_cache('fastf1_cache')
    session = ff1.get_session(year, event, session_type)
    return session

def get_clean_laps(session):
    """
    Get clean laps from a FastF1 session.

    Parameters:
    session (fastf1.core.Session): The FastF1 session object.

    Returns:
    pd.DataFrame: A DataFrame containing clean laps with LapTime in seconds.
    """
    session.load()
    laps = session.laps
    clean_laps = laps.pick_track_status('1', how='equals')
    clean_laps = clean_laps[clean_laps['PitOutTime'].isna() & clean_laps['PitInTime'].isna()]
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
    
    return clean_laps