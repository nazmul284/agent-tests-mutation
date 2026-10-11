"""Behavioral and regression tests for isodate's public API."""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import isodate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-02-29", date(2024, 2, 29)),
        ("20240229", date(2024, 2, 29)),
        ("2024-060", date(2024, 2, 29)),
        ("2024060", date(2024, 2, 29)),
        ("2020-W53-7", date(2021, 1, 3)),
        ("2020W537", date(2021, 1, 3)),
        ("2024-02", date(2024, 2, 1)),
        ("2024", date(2024, 1, 1)),
    ],
)
def test_parse_date_supported_representations(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_reduced_precision_honors_defaults():
    assert isodate.parse_date("2024", defaultmonth=7, defaultday=15) == date(2024, 7, 15)
    assert isodate.parse_date("2024-02", defaultday=20) == date(2024, 2, 20)


@pytest.mark.parametrize("text", ["", "2024/01/01", "not-a-date", "2024-1-01"])
def test_parse_date_rejects_unrecognized_syntax(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_date(text)


def test_invalid_calendar_date_is_not_silently_normalized():
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23:59:58.123456", time(23, 59, 58, 123456)),
        ("T235958,5", time(23, 59, 58, 500000)),
        ("12:30.5", time(12, 30, 30)),
        ("12.5", time(12, 30)),
    ],
)
def test_parse_time_complete_and_fractional_representations(text, expected):
    assert isodate.parse_time(text) == expected


@pytest.mark.parametrize(
    ("text", "offset"),
    [
        ("+05:30", timedelta(hours=5, minutes=30)),
        ("+0530", timedelta(hours=5, minutes=30)),
        ("+05", timedelta(hours=5)),
        ("-03:30", -timedelta(hours=3, minutes=30)),
        ("-0330", -timedelta(hours=3, minutes=30)),
        ("-03", -timedelta(hours=3)),
    ],
)
def test_parse_tzinfo_preserves_iso_offset_direction(text, offset):
    tz = isodate.parse_tzinfo(text)
    assert tz is not None
    assert tz.utcoffset(None) == offset


def test_parse_timezone_special_cases():
    assert isodate.parse_tzinfo("Z") is isodate.UTC
    assert isodate.parse_tzinfo("") is None


def test_timezone_direction_is_retained_in_time_and_datetime_parsing():
    parsed_time = isodate.parse_time("09:10:11+05:45")
    parsed_datetime = isodate.parse_datetime("2024-01-02T09:10:11-04:30")

    assert parsed_time.utcoffset() == timedelta(hours=5, minutes=45)
    assert parsed_datetime.utcoffset() == -timedelta(hours=4, minutes=30)


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 6, 7, 8, 9, 10, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        datetime(2024, 6, 7, 8, 9, 10, tzinfo=timezone(-timedelta(hours=3, minutes=30))),
    ],
)
def test_aware_datetime_default_format_round_trips(value):
    encoded = isodate.datetime_isoformat(value)
    decoded = isodate.parse_datetime(encoded)
    assert decoded == value
    assert decoded.utcoffset() == value.utcoffset()


def test_datetime_requires_the_iso_time_designator():
    with pytest.raises(isodate.ISO8601Error, match="designator 'T' missing"):
        isodate.parse_datetime("2024-01-02 03:04:05")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.25S", timedelta(days=3, hours=4, minutes=5, seconds=6.25)),
        ("-PT90M", -timedelta(minutes=90)),
        ("PT0.000001S", timedelta(microseconds=1)),
    ],
)
def test_parse_duration_without_calendar_components(text, expected):
    result = isodate.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


def test_calendar_duration_preserves_decimal_years_and_months():
    result = isodate.parse_duration("P1.5Y2.25M3D")
    assert isinstance(result, isodate.Duration)
    assert result.years == Decimal("1.5")
    assert result.months == Decimal("2.25")
    assert result.tdelta == timedelta(days=3)


def test_duration_month_arithmetic_clamps_to_last_day_of_month():
    one_month = isodate.Duration(months=1)
    assert date(2023, 1, 31) + one_month == date(2023, 2, 28)
    assert date(2024, 1, 31) + one_month == date(2024, 2, 29)


@pytest.mark.parametrize("text", ["", "P", "P1Q", "1D"])
def test_parse_duration_rejects_incomplete_or_unknown_forms(text):
    with pytest.raises(isodate.ISO8601Error):
        isodate.parse_duration(text)


def test_duration_default_format_round_trip():
    value = isodate.Duration(years=2, months=3, days=4, hours=5, minutes=6, seconds=7.5)
    assert isodate.parse_duration(isodate.duration_isoformat(value)) == value


def test_date_and_time_default_formatting():
    assert isodate.date_isoformat(date(2024, 2, 29)) == "2024-02-29"
    assert isodate.time_isoformat(time(1, 2, 3, tzinfo=isodate.UTC)) == "01:02:03Z"


def test_fixed_offset_format_variants_and_naive_datetime():
    value = datetime(2024, 1, 1, tzinfo=timezone(-timedelta(hours=4, minutes=30)))
    assert isodate.tz_isoformat(value, "%Z") == "-04:30"
    assert isodate.tz_isoformat(value, "%z") == "-0430"
    assert isodate.tz_isoformat(value, "%h") == "-04"
    assert isodate.tz_isoformat(datetime(2024, 1, 1)) == ""
    with pytest.raises(ValueError):
        isodate.tz_isoformat(value, "%q")
