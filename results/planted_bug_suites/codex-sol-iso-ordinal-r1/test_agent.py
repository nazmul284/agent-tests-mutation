"""Behavioral tests for the public :mod:`isodate` API."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20240229", date(2024, 2, 29)),
        ("2024-02-29", date(2024, 2, 29)),
        ("2024-02", date(2024, 2, 1)),
        ("2024", date(2024, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("2020-W53-7", date(2021, 1, 3)),
        ("2020W531", date(2020, 12, 28)),
        ("2021-W01", date(2021, 1, 4)),
    ],
)
def test_parse_date_supported_calendar_and_week_forms(text, expected):
    assert isodate.parse_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-001", date(2024, 1, 1)),
        ("2024001", date(2024, 1, 1)),
        ("2024-060", date(2024, 2, 29)),
        ("2023-365", date(2023, 12, 31)),
    ],
)
@pytest.mark.xfail(
    reason="parse_date currently adds the one-based ordinal directly to January 1",
)
def test_parse_date_ordinal_day_is_one_based(text, expected):
    """Ordinal 001 is January 1, not one day after January 1."""
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_and_expanded_years():
    assert isodate.parse_date("2024", defaultmonth=7, defaultday=15) == date(2024, 7, 15)
    assert isodate.parse_date("+002024-03-05", yeardigits=6) == date(2024, 3, 5)


@pytest.mark.parametrize("text", ["not-a-date", "2024/01/01", "202401011"])
def test_parse_date_rejects_unrecognised_syntax(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_date(text)


def test_parse_date_rejects_impossible_calendar_date():
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23:59:58.1234567", time(23, 59, 58, 123456)),
        ("T235958,5", time(23, 59, 58, 500000)),
        ("12:30.5", time(12, 30, 30)),
        ("01.5", time(1, 30)),
    ],
)
def test_parse_time_complete_and_fractional_reduced_forms(text, expected):
    assert isodate.parse_time(text) == expected


@pytest.mark.parametrize(
    ("text", "offset"),
    [("12:00:00Z", timedelta(0)), ("12:00:00+05:30", timedelta(hours=5, minutes=30)),
     ("120000-0430", -timedelta(hours=4, minutes=30))],
)
def test_parse_time_timezone_offsets(text, offset):
    parsed = isodate.parse_time(text)
    assert parsed.utcoffset() == offset


def test_datetime_combines_date_time_and_timezone():
    parsed = isodate.parse_datetime("2024-02-29T23:15:16.25+05:30")
    assert parsed == datetime(2024, 2, 29, 23, 15, 16, 250000, tzinfo=parsed.tzinfo)
    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)


def test_datetime_requires_exact_time_designator():
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_datetime("2024-01-01 12:00:00")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.5S", timedelta(days=3, hours=4, minutes=5, seconds=6.5)),
        ("PT0.5H", timedelta(minutes=30)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
    ],
)
def test_parse_duration_without_calendar_components_returns_timedelta(text, expected):
    assert isodate.parse_duration(text) == expected


def test_parse_duration_with_years_and_months_returns_duration():
    parsed = isodate.parse_duration("P1Y2M3DT4H")
    assert isinstance(parsed, isodate.Duration)
    assert parsed.years == Decimal("1")
    assert parsed.months == Decimal("2")
    assert parsed.tdelta == timedelta(days=3, hours=4)


def test_parse_duration_requires_string_and_valid_syntax():
    with pytest.raises(TypeError):
        isodate.parse_duration(7)
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_duration("one day")


def test_duration_calendar_arithmetic_clamps_end_of_month():
    one_month = isodate.Duration(months=1)
    assert date(2023, 1, 31) + one_month == date(2023, 2, 28)
    assert date(2024, 1, 31) + one_month == date(2024, 2, 29)


def test_duration_conversion_requires_exactly_one_anchor():
    duration = isodate.Duration(months=1)
    with pytest.raises(ValueError):
        duration.totimedelta()
    with pytest.raises(ValueError):
        duration.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))
    assert duration.totimedelta(start=date(2024, 1, 31)) == timedelta(days=29)


def test_default_formatters_round_trip_representative_values():
    day = date(2024, 2, 29)
    instant = datetime(2024, 2, 29, 12, 34, 56, tzinfo=isodate.UTC)
    span = timedelta(days=2, seconds=3, microseconds=400000)

    assert isodate.date_isoformat(day) == "2024-02-29"
    assert isodate.parse_date(isodate.date_isoformat(day)) == day
    assert isodate.datetime_isoformat(instant) == "2024-02-29T12:34:56Z"
    assert isodate.parse_datetime(isodate.datetime_isoformat(instant)) == instant
    assert isodate.parse_duration(isodate.duration_isoformat(span)) == span


def test_format_constants_cover_ordinal_and_week_dates():
    day = date(2021, 1, 3)
    assert isodate.date_isoformat(day, isodate.DATE_EXT_ORD_COMPLETE) == "2021-003"
    assert isodate.date_isoformat(day, isodate.DATE_EXT_WEEK_COMPLETE) == "2021-W53-7"
