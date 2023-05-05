import pandas as pd
import datetime as dt

def arg_nearest_after(date: dt.datetime, dates: pd.Series):
    """Gets the index of the nearest date in the dates series to the given date.
    That is that date itself if it exists in the series, otherwise the first one that comes after it.
    If no such date exists (dates.max() < date) it returns None.

    Args:
        date (dt.datetime): the minimum date to look for.
        dates (pd.Series): the series of dates to look into.

    Returns:
        The index or None.
    """
    date_diffs = dates - date
    for index in date_diffs.argsort():
        if date_diffs.iloc[index].days >= 0:
            return index
    return None