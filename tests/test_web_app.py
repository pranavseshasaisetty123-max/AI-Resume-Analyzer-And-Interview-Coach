import pytest
import datetime
from src.utils.date_utils import format_date

def test_format_date_with_datetime():
    """Verifies that a datetime.datetime object is formatted correctly to YYYY-MM-DD."""
    dt = datetime.datetime(2026, 6, 23, 21, 47, 18)
    formatted = format_date(dt)
    assert formatted == "2026-06-23"

def test_format_date_with_date():
    """Verifies that a datetime.date object is formatted correctly to YYYY-MM-DD."""
    d = datetime.date(2026, 6, 23)
    formatted = format_date(d)
    assert formatted == "2026-06-23"

def test_format_date_with_string_iso():
    """Verifies that an ISO-formatted string is sliced/formatted correctly to YYYY-MM-DD."""
    date_str = "2026-06-23T21:47:18"
    formatted = format_date(date_str)
    assert formatted == "2026-06-23"

def test_format_date_with_string_short():
    """Verifies that a YYYY-MM-DD string remains intact."""
    date_str = "2026-06-23"
    formatted = format_date(date_str)
    assert formatted == "2026-06-23"

def test_format_date_with_none():
    """Verifies that None input is handled safely."""
    formatted = format_date(None)
    assert formatted == "N/A"
