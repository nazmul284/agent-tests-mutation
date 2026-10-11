"""Behavioral tests for isodate's public parsing and formatting API."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest

import isodate
from isodate import Duration, ISO8601Error
from isodate.duration import fquotmod, max_days_in_month


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20240229", date(2024, 2, 29)),
        ("2024-02-29", date(2024, 2, 29)),
        ("2024-060", date(2024, 2, 29)),
        ("2024060", date(2024, 2, 29)),
        ("2020-W53-7", date(2021, 1, 3)),
        ("2020W537", date(2021, 1, 3)),
        ("2024-W01", date(2024, 1, 1)),
        ("202402", date(2024, 2, 1)),
        ("2024", date(2024, 1, 1)),
        ("19", date(1901, 1, 1)),
    ],
)
def test_parse_date_supported_representations(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_and_expanded_years():
    assert isodate.parse_date("2024", defaultmonth=7, defaultday=4) == date(2024, 7, 4)
    assert isodate.parse_date("+002024-02-29", yeardigits=6) == date(2024, 2, 29)
    assert isodate.parse_date("+2024-12-31", expanded=True) == date(2024, 12, 31)


@pytest.mark.parametrize("text", ["", "2024/01/01", "2024-1-01", "not-a-date"])
def test_parse_date_rejects_unrecognized_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_date(text)


def test_parse_date_propagates_invalid_calendar_values():
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")
    with pytest.raises(ValueError):
        isodate.parse_date("2024-13-01")


def test_date_format_variants_and_round_trip():
    value = date(2000, 2, 29)
    assert isodate.date_isoformat(value) == "2000-02-29"
    assert isodate.date_isoformat(value, isodate.DATE_BAS_COMPLETE) == "20000229"
    assert isodate.date_isoformat(value, isodate.DATE_EXT_ORD_COMPLETE) == "2000-060"
    assert isodate.parse_date(isodate.date_isoformat(value)) == value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23:59:58", time(23, 59, 58)),
        ("235958", time(23, 59, 58)),
        ("T12.5", time(12, 30)),
        ("12:30,5", time(12, 30, 30)),
        ("00:00:00.000001", time(0, 0, 0, 1)),
        ("23:59:59.1234569", time(23, 59, 59, 123456)),
    ],
)
def test_parse_time_complete_reduced_and_fractional(text, expected):
    assert isodate.parse_time(text) == expected


def test_timezones_are_parsed_with_correct_offsets():
    utc = isodate.parse_time("12:00:00Z")
    east = isodate.parse_time("12:00:00+05:30")
    west = isodate.parse_time("12:00:00-0230")
    assert utc.tzinfo is isodate.UTC
    assert utc.utcoffset() == timedelta(0)
    assert east.utcoffset() == timedelta(hours=5, minutes=30)
    assert west.utcoffset() == -timedelta(hours=2, minutes=30)
    assert east.tzname() == "+05:30"


@pytest.mark.parametrize("text", ["", "1:02:03", "24:00:01", "12:60:00", "noon"])
def test_parse_time_rejects_malformed_or_out_of_range_values(text):
    exception = ValueError if text in {"24:00:01", "12:60:00"} else ISO8601Error
    with pytest.raises(exception):
        isodate.parse_time(text)


def test_time_format_timezone_directives():
    value = time(7, 8, 9, tzinfo=isodate.FixedOffset(-4, -30, "custom"))
    assert isodate.time_isoformat(value) == "07:08:09-04:30"
    assert isodate.time_isoformat(value, isodate.TIME_BAS_COMPLETE + isodate.TZ_BAS) == "070809-0430"
    assert isodate.time_isoformat(time(7, 8, 9)) == "07:08:09"


def test_datetime_parsing_and_formatting_preserve_date_time_and_zone():
    value = isodate.parse_datetime("2024-02-29T23:15:07.25+05:45")
    assert value == datetime(2024, 2, 29, 23, 15, 7, 250000, value.tzinfo)
    assert value.utcoffset() == timedelta(hours=5, minutes=45)
    assert isodate.datetime_isoformat(value) == "2024-02-29T23:15:07+05:45"
    assert isodate.parse_datetime("20240229T231507Z").tzinfo is isodate.UTC


@pytest.mark.parametrize("text", ["2024-01-01 12:00:00", "2024-01-01", "T12:00:00"])
def test_datetime_requires_valid_date_and_t_separator(text):
    with pytest.raises((ISO8601Error, ValueError)):
        isodate.parse_datetime(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P0D", timedelta(0)),
        ("P2W", timedelta(days=14)),
        ("-P2W", timedelta(days=-14)),
        ("P3DT4H5M6S", timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("PT0.000001S", timedelta(microseconds=1)),
        ("PT1,5H", timedelta(hours=1, minutes=30)),
        ("P0.5D", timedelta(hours=12)),
    ],
)
def test_parse_duration_without_calendar_units_returns_timedelta(text, expected):
    result = isodate.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


def test_parse_duration_with_calendar_units_returns_duration():
    result = isodate.parse_duration("P1Y2M3DT4H5M6.5S")
    assert isinstance(result, Duration)
    assert result.years == Decimal("1")
    assert result.months == Decimal("2")
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.5)


def test_parse_duration_can_force_duration_and_handle_negative_sign():
    forced = isodate.parse_duration("-P1DT2H", as_timedelta_if_possible=False)
    assert isinstance(forced, Duration)
    assert forced.years == 0 and forced.months == 0
    assert forced.tdelta == -timedelta(days=1, hours=2)


@pytest.mark.parametrize("value", [None, 42, timedelta(1)])
def test_parse_duration_requires_a_string(value):
    with pytest.raises(TypeError):
        isodate.parse_duration(value)


@pytest.mark.parametrize("text", ["", "P", "1D", "PX"])
def test_parse_duration_rejects_invalid_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_duration(text)


def test_duration_formatting_zero_fractional_and_negative_values():
    assert isodate.duration_isoformat(timedelta(0)) == "P0D"
    assert isodate.duration_isoformat(timedelta(seconds=1, microseconds=250000)) == "PT1.25S"
    assert isodate.duration_isoformat(timedelta(days=-2, hours=-3)) == "-P2DT3H"
    value = Duration(years=2, months=3, days=4, minutes=5)
    assert isodate.duration_isoformat(value) == "P2Y3M4DT5M"


def test_duration_calendar_arithmetic_clamps_end_of_month():
    one_month = Duration(months=1)
    assert date(2024, 1, 31) + one_month == date(2024, 2, 29)
    assert date(2023, 1, 31) + one_month == date(2023, 2, 28)
    assert date(2024, 3, 31) - one_month == date(2024, 2, 29)
    assert datetime(2024, 1, 31, 12, 30) + one_month == datetime(2024, 2, 29, 12, 30)


def test_duration_arithmetic_equality_hash_and_conversion():
    first = Duration(years=1, days=2)
    equivalent = Duration(months=12, days=2)
    assert first == equivalent
    assert first != Duration(years=1, days=3)
    assert Duration(days=2) == timedelta(days=2)
    assert first + Duration(months=2, hours=3) == Duration(years=1, months=2, days=2, hours=3)
    assert 2 * Duration(months=1, days=2) == Duration(months=2, days=4)
    assert -Duration(months=1, days=2) == Duration(months=-1, days=-2)
    assert len({first, Duration(years=1, days=2)}) == 1
    assert Duration(months=1).totimedelta(start=date(2024, 1, 31)) == timedelta(days=29)
    assert Duration(months=1).totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)


def test_duration_rejects_ambiguous_or_fractional_calendar_conversion():
    with pytest.raises(ValueError, match="start or end required"):
        Duration(months=1).totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        Duration(months=1).totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))
    with pytest.raises(ValueError, match="fractional"):
        date(2024, 1, 1) + Duration(months=Decimal("0.5"))


def test_timezone_public_api_and_pickle_identity():
    assert isodate.parse_tzinfo("") is None
    assert isodate.parse_tzinfo("Z") is isodate.UTC
    assert isodate.parse_tzinfo("-07").utcoffset(None) == timedelta(hours=-7)
    assert isodate.parse_tzinfo("+1245").utcoffset(None) == timedelta(hours=12, minutes=45)
    assert pickle.loads(pickle.dumps(isodate.UTC)) is isodate.UTC


def test_tz_isoformat_and_invalid_directive():
    naive = datetime(2024, 1, 1)
    utc = datetime(2024, 1, 1, tzinfo=isodate.UTC)
    offset = datetime(2024, 1, 1, tzinfo=isodate.FixedOffset(5, 45, "NPT"))
    assert isodate.tz_isoformat(naive) == ""
    assert isodate.tz_isoformat(utc) == "Z"
    assert isodate.tz_isoformat(offset) == "+05:45"
    assert isodate.tz_isoformat(offset, "%z") == "+0545"
    assert isodate.tz_isoformat(offset, "%h") == "+05"
    with pytest.raises(ValueError, match="unknown format"):
        isodate.tz_isoformat(offset, "%q")


def test_internal_calendar_helpers_at_boundaries():
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2024, 4) == 30
    assert max_days_in_month(2024, 12) == 31
    assert fquotmod(Decimal("0"), 1, 13) == (-1, Decimal("12"))
    assert fquotmod(Decimal("13"), 1, 13) == (1, Decimal("1"))


def test_strftime_literal_percent_microseconds_and_unknown_directive():
    value = datetime(2024, 2, 3, 4, 5, 6, 7008)
    assert isodate.strftime(value, "%Y-%m-%d %H:%M:%S.%f %%") == "2024-02-03 04:05:06.007008 %"
    assert isodate.strftime(value, "%q") == "%q"
