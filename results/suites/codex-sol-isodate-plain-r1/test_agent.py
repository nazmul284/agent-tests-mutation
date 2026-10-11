"""Behavioral tests for the public isodate API.

The cases here are intentionally derived from the implementation and ISO 8601
semantics, rather than from an external test suite.
"""

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
        ("2020W531", date(2020, 12, 28)),
        ("2024-W01", date(2024, 1, 1)),
        ("2024-07", date(2024, 7, 1)),
        ("2024", date(2024, 1, 1)),
        ("19", date(1901, 1, 1)),
    ],
)
def test_parse_date_supported_representations(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_and_expanded_years():
    assert isodate.parse_date("2024", defaultmonth=6, defaultday=15) == date(2024, 6, 15)
    assert isodate.parse_date("+001999-12-31", yeardigits=6) == date(1999, 12, 31)
    assert isodate.parse_date("+2024-03-02", expanded=True) == date(2024, 3, 2)


@pytest.mark.parametrize("text", ["", "2024/01/02", "2024-1-02", "not-a-date"])
def test_parse_date_rejects_unrecognised_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_date(text)


def test_parse_date_rejects_impossible_calendar_date():
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12:34:56", time(12, 34, 56)),
        ("T123456", time(12, 34, 56)),
        ("12:34:56,25", time(12, 34, 56, 250000)),
        ("12:30.5", time(12, 30, 30)),
        ("23.5", time(23, 30)),
        ("01.25", time(1, 15)),
    ],
)
def test_parse_time_accuracy_and_fractions(text, expected):
    assert isodate.parse_time(text) == expected


def test_parse_time_truncates_beyond_microsecond_precision():
    assert isodate.parse_time("12:34:56.1234569") == time(12, 34, 56, 123456)


@pytest.mark.parametrize(
    ("text", "offset"),
    [
        ("12:00Z", timedelta(0)),
        ("12:00+05:30", timedelta(hours=5, minutes=30)),
        ("1200-0230", -timedelta(hours=2, minutes=30)),
        ("12+04", timedelta(hours=4)),
    ],
)
def test_parse_time_zones(text, offset):
    parsed = isodate.parse_time(text)
    assert parsed.utcoffset() == offset


@pytest.mark.parametrize("text", ["", "1:00", "24:00", "12:60", "12:00:60", "noon"])
def test_parse_time_rejects_invalid_values(text):
    with pytest.raises((ISO8601Error, ValueError)):
        isodate.parse_time(text)


def test_datetime_parse_and_format_round_trip_with_offset():
    text = "2024-02-29T23:45:06+05:30"
    parsed = isodate.parse_datetime(text)
    assert parsed == datetime(2024, 2, 29, 23, 45, 6, tzinfo=parsed.tzinfo)
    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)
    assert isodate.datetime_isoformat(parsed) == text


@pytest.mark.parametrize("text", ["2024-01-01 12:00", "2024-01-01", "2024T01:00Textra"])
def test_datetime_requires_exactly_one_time_designator(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_datetime(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.5S", timedelta(days=3, hours=4, minutes=5, seconds=6.5)),
        ("PT1.5H", timedelta(hours=1, minutes=30)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
        ("P0D", timedelta(0)),
    ],
)
def test_parse_duration_without_calendar_units(text, expected):
    parsed = isodate.parse_duration(text)
    assert isinstance(parsed, timedelta)
    assert parsed == expected


def test_parse_duration_calendar_units_and_alternative_form():
    parsed = isodate.parse_duration("P1Y2M3DT4H5M6.25S")
    assert isinstance(parsed, Duration)
    assert parsed.years == Decimal("1")
    assert parsed.months == Decimal("2")
    assert parsed.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.25)

    alternative = isodate.parse_duration("P0001-02-03T04:05:06")
    assert alternative == Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)


def test_parse_duration_can_force_duration_result():
    parsed = isodate.parse_duration("P2D", as_timedelta_if_possible=False)
    assert isinstance(parsed, Duration)
    assert parsed == Duration(days=2)


@pytest.mark.parametrize("value", [None, 12, timedelta(1)])
def test_parse_duration_requires_string(value):
    with pytest.raises(TypeError):
        isodate.parse_duration(value)


@pytest.mark.parametrize("text", ["", "P", "1D", "P1Q"])
def test_parse_duration_rejects_incomplete_or_unknown_forms(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_duration(text)


def test_duration_formatting_normalises_components_and_sign():
    value = timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=120000)
    assert isodate.duration_isoformat(value) == "P2DT3H4M5.12S"
    assert isodate.duration_isoformat(-value) == "-P2DT3H4M5.12S"
    assert isodate.duration_isoformat(timedelta(0)) == "P0D"
    assert isodate.duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"


def test_duration_arithmetic_clamps_month_ends_and_preserves_datetime_time():
    one_month = Duration(months=1)
    assert date(2023, 1, 31) + one_month == date(2023, 2, 28)
    assert date(2024, 1, 31) + one_month == date(2024, 2, 29)
    assert datetime(2024, 1, 31, 8, 30) + one_month == datetime(2024, 2, 29, 8, 30)
    assert date(2024, 3, 31) - one_month == date(2024, 2, 29)


def test_duration_arithmetic_and_equality_rules():
    a = Duration(years=1, days=2)
    b = Duration(months=12, days=2)
    assert a == b
    assert a + Duration(months=1, days=3) == Duration(years=1, months=1, days=5)
    assert a - timedelta(days=1) == Duration(years=1, days=1)
    assert -a == Duration(years=-1, days=-2)
    assert 2 * Duration(months=2, days=1) == Duration(months=4, days=2)
    assert Duration(days=2) == timedelta(days=2)
    assert Duration(months=1) != timedelta(days=31)


def test_duration_date_conversion_requires_one_endpoint():
    duration = Duration(months=1, days=2)
    assert duration.totimedelta(start=date(2024, 1, 30)) == timedelta(days=32)
    assert duration.totimedelta(end=date(2024, 3, 2)) == timedelta(days=31)
    with pytest.raises(ValueError, match="start or end required"):
        duration.totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        duration.totimedelta(start=date.today(), end=date.today())


def test_fractional_calendar_duration_cannot_be_applied_to_date():
    with pytest.raises(ValueError, match="fractional years or months"):
        date(2024, 1, 1) + Duration(months=Decimal("1.5"))


def test_duration_helpers_cover_leap_rules_and_negative_wrap():
    assert [max_days_in_month(y, 2) for y in (2000, 1900, 2024, 2023)] == [29, 28, 29, 28]
    assert max_days_in_month(2024, 4) == 30
    assert max_days_in_month(2024, 1) == 31
    assert fquotmod(Decimal("0"), 1, 13) == (-1, Decimal("12"))


def test_timezone_parsing_and_format_variants():
    assert isodate.parse_tzinfo("") is None
    assert isodate.parse_tzinfo("Z") is isodate.UTC
    zone = isodate.parse_tzinfo("-03:30")
    dt = datetime(2024, 1, 1, tzinfo=zone)
    assert dt.utcoffset() == -timedelta(hours=3, minutes=30)
    assert isodate.tz_isoformat(dt) == "-03:30"
    assert isodate.tz_isoformat(dt, "%z") == "-0330"
    assert isodate.tz_isoformat(dt, "%h") == "-03"
    assert isodate.tz_isoformat(datetime(2024, 1, 1)) == ""
    assert isodate.tz_isoformat(datetime(2024, 1, 1, tzinfo=isodate.UTC)) == "Z"


def test_timezone_invalid_format_directive_and_pickle_identity():
    nonzero_zone = isodate.FixedOffset(1, 0, "+01:00")
    with pytest.raises(ValueError, match="unknown format"):
        isodate.tz_isoformat(datetime.now(nonzero_zone), "%q")
    assert pickle.loads(pickle.dumps(isodate.UTC)) is isodate.UTC


def test_format_constants_and_custom_strftime():
    d = date(2020, 12, 31)
    assert isodate.date_isoformat(d) == "2020-12-31"
    assert isodate.date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "2020-366"
    assert isodate.date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "2020-W53-4"
    assert isodate.strftime(d, "%Y/%m/%d %%") == "2020/12/31 %"
    assert isodate.time_isoformat(time(1, 2, 3, tzinfo=isodate.UTC)) == "01:02:03Z"


def test_duration_repr_hash_and_attribute_delegation():
    value = Duration(years=1, months=2, days=3, seconds=4)
    assert value.days == 3
    assert "Duration(3, 4, 0, years=1, months=2)" in repr(value)
    assert str(value) == "1 years, 2 months, 3 days, 0:00:04"
    assert hash(value) == hash(Duration(years=1, months=2, days=3, seconds=4))
