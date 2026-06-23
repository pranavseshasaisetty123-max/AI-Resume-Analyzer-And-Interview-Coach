import datetime

def format_date(value):
    """
    Formats a datetime object or string representing a date/datetime to YYYY-MM-DD format.
    Handles datetime, date, and string types safely.
    """
    if value is None:
        return "N/A"
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        # Cleanly return YYYY-MM-DD from start of string
        return value[:10]
    return str(value)
