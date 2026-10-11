"""Behavioral tests for the public :mod:`isodate` API.

These tests intentionally use concrete boundary values (leap years, ISO week
year crossings, negative offsets, and month-end arithmetic), since those are
the places where plausible parser implementations most often go wrong.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest

import isodate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20240229", date(2024, 2, 29)),
        ("2024-02-29", date(2024, 2, 29)),
        ("2024-060", date(2024, 2, 29)),
        ("2024060", date(2024, 2, 29)),
        ("2024-W01-1", date(2024, 1, 1)),
        ("2020W537", date(2021, 1, 3)),
        ("2024-W09", date(2024, 2, 26)),
        ("2024-02", date(2024, 2, 1)),
        ("2024", date(2024, 1, 1)),
        ("20", date(2001, 1, 1)),
    ],
)
def test_parse_date_supported_representations(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_custom_defaults_and_expanded_year():
    assert isodate.parse_date("2024", defaultmonth=7, defaultday=15) == date(2024, 7, 15)
    assert isodate.parse_date("+002024-03-01", yeardigits=6) == date(2024, 3, 1)


@pytest.mark.parametrize("text", ["", "2024/01/02", "2024-1-02", "not-a-date"])
def test_parse_date_rejects_unrecognised_syntax(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_date(text)


def test_parse_date_propagates_invalid_calendar_values():
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")


@pytest.mark.parametrize(
    ("text", "expected", "offset"),
    [
        ("23:59:58", time(23, 59, 58), None),
        ("T235958Z", time(23, 59, 58, tzinfo=isodate.UTC), timedelta(0)),
        ("12:30.5+05:30", time(12, 30, 30), timedelta(hours=5, minutes=30)),
        ("01.5", time(1, 30), None),
        ("12:34:56,25-0230", time(12, 34, 56, 250000), -timedelta(hours=2, minutes=30)),
    ],
)
def test_parse_time_forms_fractions_and_offsets(text, expected, offset):
    result = isodate.parse_time(text)
    assert result.replace(tzinfo=None) == expected.replace(tzinfo=None)
    assert result.utcoffset() == offset


def test_fractional_seconds_are_truncated_to_microseconds():
    assert isodate.parse_time("00:00:00.1234569").microsecond == 123456


@pytest.mark.parametrize("text", ["", "1:02:03", "12-30-00", "12:00:00junk"])
def test_parse_time_rejects_bad_syntax(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_time(text)


def test_out_of_range_time_is_a_value_error():
    with pytest.raises(ValueError):
        isodate.parse_time("24:00:00")


def test_datetime_combines_date_time_and_timezone():
    parsed = isodate.parse_datetime("2024-02-29T23:15:01.5-04:30")
    assert parsed.replace(tzinfo=None) == datetime(2024, 2, 29, 23, 15, 1, 500000)
    assert parsed.utcoffset() == -timedelta(hours=4, minutes=30)


def test_datetime_requires_uppercase_time_designator():
    with pytest.raises(isodate.ISO8601Error, match="designator 'T' missing"):
        isodate.parse_datetime("2024-01-01 12:00:00")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.25S", timedelta(days=3, hours=4, minutes=5, seconds=6.25)),
        ("PT0.000001S", timedelta(microseconds=1)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
        ("P1,5D", timedelta(days=1, hours=12)),
    ],
)
def test_parse_duration_without_calendar_components(text, expected):
    result = isodate.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


def test_parse_duration_preserves_years_months_and_time():
    result = isodate.parse_duration("P1Y2M3DT4H5M6.25S")
    assert isinstance(result, isodate.Duration)
    assert result.years == Decimal("1")
    assert result.months == Decimal("2")
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.25)


def test_duration_can_be_forced_even_without_years_or_months():
    result = isodate.parse_duration("P2D", as_timedelta_if_possible=False)
    assert isinstance(result, isodate.Duration)
    assert result == isodate.Duration(days=2)


@pytest.mark.parametrize("text", ["", "P", "1D", "P1Q"])
def test_parse_duration_rejects_empty_or_unknown_components(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_duration(text)


def test_parse_duration_requires_a_string():
    with pytest.raises(TypeError, match="Expecting a string"):
        isodate.parse_duration(3)  # type: ignore[arg-type]


def test_duration_month_end_and_leap_year_arithmetic():
    one_month = isodate.Duration(months=1)
    one_year = isodate.Duration(years=1)
    assert date(2023, 1, 31) + one_month == date(2023, 2, 28)
    assert date(2024, 1, 31) + one_month == date(2024, 2, 29)
    assert date(2024, 2, 29) + one_year == date(2025, 2, 28)
    assert date(2024, 3, 31) - one_month == date(2024, 2, 29)


def test_duration_arithmetic_equality_and_conversion():
    duration = isodate.Duration(years=1, months=2, days=3)
    assert duration + isodate.Duration(months=10) == isodate.Duration(years=2, days=3)
    assert 2 * isodate.Duration(months=2, days=1) == isodate.Duration(months=4, days=2)
    assert isodate.Duration(months=12) == isodate.Duration(years=1)
    assert isodate.Duration(days=2) == timedelta(days=2)
    assert isodate.Duration(months=1).totimedelta(start=date(2024, 1, 31)) == timedelta(days=29)


def test_duration_rejects_ambiguous_or_fractional_date_conversion():
    with pytest.raises(ValueError, match="start or end required"):
        isodate.Duration(months=1).totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        isodate.Duration(days=1).totimedelta(start=date(2024, 1, 1), end=date(2024, 1, 2))
    with pytest.raises(ValueError, match="fractional years or months"):
        date(2024, 1, 1) + isodate.Duration(months=Decimal("0.5"))


def test_timezone_parsing_and_formatting():
    assert isodate.parse_tzinfo("") is None
    assert isodate.parse_tzinfo("Z") is isodate.UTC
    west = isodate.parse_tzinfo("-03:45")
    east = isodate.parse_tzinfo("+0530")
    assert west.utcoffset(None) == -timedelta(hours=3, minutes=45)
    assert east.utcoffset(None) == timedelta(hours=5, minutes=30)
    aware = datetime(2024, 1, 1, tzinfo=east)
    assert isodate.tz_isoformat(aware) == "+05:30"
    assert isodate.tz_isoformat(aware, "%z") == "+0530"
    assert isodate.tz_isoformat(aware, "%h") == "+05"
    assert isodate.tz_isoformat(datetime(2024, 1, 1)) == ""


def test_utc_singleton_survives_pickling():
    assert pickle.loads(pickle.dumps(isodate.UTC)) is isodate.UTC


def test_isoformat_defaults_and_round_trips():
    dt = datetime(2024, 2, 29, 12, 34, 56, tzinfo=isodate.parse_tzinfo("+05:30"))
    assert isodate.date_isoformat(dt.date()) == "2024-02-29"
    assert isodate.time_isoformat(dt.timetz()) == "12:34:56+05:30"
    assert isodate.datetime_isoformat(dt) == "2024-02-29T12:34:56+05:30"
    assert isodate.parse_datetime(isodate.datetime_isoformat(dt)) == dt


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(0), "P0D"),
        (timedelta(days=2, seconds=3661, microseconds=500000), "P2DT1H1M1.5S"),
        (-timedelta(days=1, seconds=2), "-P1DT2S"),
        (isodate.Duration(years=1, months=2, days=3), "P1Y2M3D"),
    ],
)
def test_duration_isoformat(value, expected):
    assert isodate.duration_isoformat(value) == expected


def test_custom_strftime_directives():
    value = datetime(2020, 12, 31, 2, 3, 4, 5006, tzinfo=isodate.UTC)
    assert isodate.strftime(value, "%Y-%j W%W-%w %H:%M:%S.%f %Z %%") == (
        "2020-366 W53-4 02:03:04.005006 Z %"
    )


def test_unknown_timezone_format_is_rejected():
    offset = isodate.parse_tzinfo("+01:00")
    with pytest.raises(ValueError, match="unknown format string"):
        isodate.tz_isoformat(datetime(2024, 1, 1, tzinfo=offset), "%q")
