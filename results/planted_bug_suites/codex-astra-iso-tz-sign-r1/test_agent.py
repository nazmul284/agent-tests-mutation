"""Behavioral tests using independently calculated values and stdlib oracles.

Strict xfails document the existing timezone-sign defect without treating the
incorrect result as the API contract. Remove the marker when fixing the defect.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest
import isodate as iso


@pytest.mark.parametrize("text, expected", [
    ("2024-02-29", date(2024, 2, 29)),
    ("20240229", date(2024, 2, 29)),
    ("2024-060", date(2024, 2, 29)),
    ("2024060", date(2024, 2, 29)),
    ("2023-365", date(2023, 12, 31)),
    ("2020-W53-7", date(2021, 1, 3)),
    ("2020W537", date(2021, 1, 3)),
    ("2019-W01", date(2018, 12, 31)),
    ("2019W01", date(2018, 12, 31)),
    ("2024-02", date(2024, 2, 1)),
    ("202402", date(2024, 2, 1)),
    ("2024", date(2024, 1, 1)),
    ("20", date(2001, 1, 1)),
    ("0001-01-01", date.min),
    ("9999-12-31", date.max),
])
def test_parse_date(text, expected):
    assert iso.parse_date(text) == expected


def test_date_defaults_and_expanded_years():
    assert iso.parse_date("2024", defaultmonth=2, defaultday=29) == date(2024, 2, 29)
    assert iso.parse_date("2024-03", defaultday=15) == date(2024, 3, 15)
    assert iso.parse_date("+002024-02-29", yeardigits=6) == date(2024, 2, 29)
    assert iso.parse_date("+2024-02-29", expanded=True) == date(2024, 2, 29)
    # Changing parsing options must not poison cached regular expressions.
    assert iso.parse_date("2024-02-29") == date(2024, 2, 29)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date("+2024-02-29")


@pytest.mark.parametrize("text", ["", "2024/02/29", "2024-2-29", "2024-02-29junk", " 2024-02-29", "2024-02-29\n"])
def test_reject_date_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize("text", ["2023-02-29", "1900-02-29", "2024-13-01", "2024-04-31", "2024-01-00", "0000-01-01"])
def test_reject_impossible_calendar_dates(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


def test_week_and_ordinal_dates_against_stdlib():
    for year in (1900, 2000, 2015, 2016, 2019, 2020, 2021, 2024):
        for offset in range(-7, 370):
            day = date(year, 1, 1) + timedelta(days=offset)
            weekyear, week, weekday = day.isocalendar()
            assert iso.parse_date(f"{weekyear:04}-W{week:02}-{weekday}") == day
            ordinal = (day - date(day.year, 1, 1)).days + 1
            assert iso.parse_date(f"{day.year:04}-{ordinal:03}") == day


@pytest.mark.parametrize("text, expected", [
    ("12:34:56", time(12, 34, 56)), ("123456", time(12, 34, 56)),
    ("T12:34:56", time(12, 34, 56)), ("12:34", time(12, 34)),
    ("1234", time(12, 34)), ("12", time(12)),
    ("12.5", time(12, 30)), ("12,25", time(12, 15)),
    ("12:34.5", time(12, 34, 30)), ("1234,25", time(12, 34, 15)),
    ("12:34:56.1234569", time(12, 34, 56, 123456)),
    ("12:34:59.9999999", time(12, 34, 59, 999999)),
    ("00:00:00,000001", time(0, 0, 0, 1)),
    ("23:59:59Z", time(23, 59, 59, tzinfo=iso.UTC)),
])
def test_parse_time(text, expected):
    actual = iso.parse_time(text)
    assert actual == expected
    assert actual.tzinfo is expected.tzinfo


@pytest.mark.parametrize("text", ["", "1:02:03", "12:3456", "12:34:56garbage", "12:34:56\n", "12.5:30", "12:34:56+5:00"])
def test_reject_time_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_time(text)


@pytest.mark.parametrize("text", ["24:00:00", "12:60:00", "12:34:60"])
def test_reject_unrepresentable_times(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize("text, expected", [
    ("2024-02-29T12:34:56.123456", datetime(2024, 2, 29, 12, 34, 56, 123456)),
    ("2024060T123456Z", datetime(2024, 2, 29, 12, 34, 56, tzinfo=iso.UTC)),
    ("2020-W53-7T00:00:00", datetime(2021, 1, 3)),
])
def test_parse_datetime(text, expected):
    assert iso.parse_datetime(text) == expected


@pytest.mark.parametrize("text", ["2024-02-29 12:00:00", "2024-02-29", "2024-02-29T12:00:00T", "T12:00:00"])
def test_reject_datetime_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="build_tzinfo reverses the sign of nonzero ISO offsets")
@pytest.mark.parametrize("suffix, minutes", [
    ("+05:30", 330), ("-03:30", -210), ("+0545", 345),
    ("-0045", -45), ("+02", 120), ("-08", -480),
])
@pytest.mark.parametrize("parser", ["zone", "time", "datetime"])
def test_parsed_timezone_sign(parser, suffix, minutes):
    if parser == "zone":
        offset = iso.parse_tzinfo(suffix).utcoffset(None)
    elif parser == "time":
        offset = iso.parse_time("12:34:56" + suffix).utcoffset()
    else:
        offset = iso.parse_datetime("2024-02-29T12:34:56" + suffix).utcoffset()
    assert offset == timedelta(minutes=minutes)


def test_utc_and_absent_timezone():
    assert iso.parse_tzinfo("") is None
    assert iso.parse_tzinfo("Z") is iso.UTC
    for suffix in ("+00:00", "-0000", "+00"):
        assert iso.parse_tzinfo(suffix).utcoffset(None) == timedelta(0)
    assert iso.UTC.dst(None) == timedelta(0)
    assert iso.UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ""
    assert iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=iso.UTC)) == "Z"


@pytest.mark.parametrize("hours, minutes, extended, basic, short", [
    (5, 30, "+05:30", "+0530", "+05"),
    (-3, -30, "-03:30", "-0330", "-03"),
    (0, -45, "-00:45", "-0045", "-00"),
])
def test_fixed_offset_and_timezone_format(hours, minutes, extended, basic, short):
    zone = iso.FixedOffset(hours, minutes, "test-zone")
    assert zone.utcoffset(None) == timedelta(hours=hours, minutes=minutes)
    assert zone.dst(None) == timedelta(0)
    assert zone.tzname(None) == "test-zone"
    value = datetime(2024, 2, 29, 12, 34, 56, tzinfo=zone)
    assert iso.tz_isoformat(value) == extended
    assert iso.tz_isoformat(value, "%z") == basic
    assert iso.tz_isoformat(value, "%h") == short
    assert iso.datetime_isoformat(value) == "2024-02-29T12:34:56" + extended
    with pytest.raises(ValueError):
        iso.tz_isoformat(value, "invalid")


@pytest.mark.parametrize("fmt, expected", [
    (iso.DATE_EXT_COMPLETE, "2024-02-29"), (iso.DATE_BAS_COMPLETE, "20240229"),
    (iso.DATE_EXT_ORD_COMPLETE, "2024-060"), (iso.DATE_BAS_ORD_COMPLETE, "2024060"),
    (iso.DATE_EXT_WEEK_COMPLETE, "2024-W09-4"), (iso.DATE_BAS_WEEK_COMPLETE, "2024W094"),
    (iso.DATE_EXT_MONTH, "2024-02"), (iso.DATE_YEAR, "2024"),
])
def test_date_format(fmt, expected):
    assert iso.date_isoformat(date(2024, 2, 29), fmt) == expected
    assert iso.parse_date(expected).year == 2024


def test_custom_formats_and_small_years():
    assert iso.date_isoformat(date.min) == "0001-01-01"
    assert iso.date_isoformat(date(2024, 2, 29), yeardigits=6) == "+002024-02-29"
    value = datetime(2024, 2, 29, 1, 2, 3, 45, tzinfo=iso.UTC)
    assert iso.strftime(value, "%Y-%m-%dT%H:%M:%S.%f%Z %%") == "2024-02-29T01:02:03.000045Z %"
    assert iso.time_isoformat(value.time(), iso.TIME_BAS_COMPLETE) == "010203"
    assert iso.datetime_isoformat(value, iso.DT_BAS_COMPLETE) == "20240229T010203Z"


@pytest.mark.parametrize("text, expected", [
    ("P0D", timedelta(0)), ("PT0S", timedelta(0)),
    ("P2W", timedelta(days=14)), ("P1.5D", timedelta(hours=36)),
    ("PT1,5H", timedelta(minutes=90)), ("PT1.5M", timedelta(seconds=90)),
    ("P2DT3H4M5.000006S", timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=6)),
    ("-PT0.000001S", timedelta(microseconds=-1)),
    ("-P2DT3H", -timedelta(days=2, hours=3)),
    ("+PT30S", timedelta(seconds=30)),
    ("P0Y0M1D", timedelta(days=1)),
])
def test_parse_timedelta(text, expected):
    result = iso.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


@pytest.mark.parametrize("sign", [1, -1])
def test_parse_calendar_duration(sign):
    result = iso.parse_duration(("-" if sign < 0 else "") + "P1.5Y2.25M3DT4H")
    assert isinstance(result, iso.Duration)
    assert result.years == sign * Decimal("1.5")
    assert result.months == sign * Decimal("2.25")
    assert result.tdelta == sign * timedelta(days=3, hours=4)


def test_duration_explicit_type_and_alternative_format():
    result = iso.parse_duration("P2DT3H", as_timedelta_if_possible=False)
    assert isinstance(result, iso.Duration)
    assert result.years == result.months == 0
    assert result.tdelta == timedelta(days=2, hours=3)
    assert iso.parse_duration("P0001-02-03T04:05:06") == iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize("text", ["", "P", "1D", "P-1D", "PT1Q", "P1S", "P1Dgarbage", "P1D2Y"])
def test_reject_duration_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize("value", [None, 123, b"P1D", timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize("value, expected", [
    (timedelta(0), "P0D"), (timedelta(days=2), "P2D"),
    (timedelta(hours=25, minutes=2, seconds=3), "P1DT1H2M3S"),
    (timedelta(microseconds=1), "PT0.000001S"),
    (timedelta(seconds=1, microseconds=120000), "PT1.12S"),
    (timedelta(microseconds=-1), "-PT0.000001S"),
    (-timedelta(days=2, seconds=3), "-P2DT3S"),
    (iso.Duration(years=1, months=2, days=3), "P1Y2M3D"),
    (iso.Duration(months=Decimal("1.25")), "P1.25M"),
    (iso.Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
])
def test_duration_format_and_roundtrip(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value


@pytest.mark.parametrize("start, delta, expected", [
    (date(2024, 1, 31), iso.Duration(months=1), date(2024, 2, 29)),
    (date(2023, 1, 31), iso.Duration(months=1), date(2023, 2, 28)),
    (date(2024, 2, 29), iso.Duration(years=1), date(2025, 2, 28)),
    (date(2000, 2, 29), iso.Duration(years=-100), date(1900, 2, 28)),
    (date(2024, 1, 31), iso.Duration(months=-2), date(2023, 11, 30)),
    (date(2023, 12, 31), iso.Duration(months=14), date(2025, 2, 28)),
    (date(2024, 1, 31), iso.Duration(months=1, days=1), date(2024, 3, 1)),
])
def test_calendar_arithmetic(start, delta, expected):
    assert start + delta == expected
    assert delta + start == expected
    assert start - (-delta) == expected
    assert delta.totimedelta(start=start) == expected - start


def test_duration_arithmetic_and_equality():
    left = iso.Duration(years=1, months=2, days=3)
    right = iso.Duration(months=4, hours=5)
    assert left + right == iso.Duration(years=1, months=6, days=3, hours=5)
    assert left - right == iso.Duration(years=1, months=-2, days=3, hours=-5)
    assert left * 2 == 2 * left == iso.Duration(years=2, months=4, days=6)
    assert left * 0 == timedelta(0)
    assert -left == iso.Duration(years=-1, months=-2, days=-3)
    assert left + timedelta(hours=2) == timedelta(hours=2) + left
    assert timedelta(days=4) - left == iso.Duration(years=-1, months=-2, days=1)
    assert iso.Duration(years=1) == iso.Duration(months=12)
    assert not (iso.Duration(years=1) != iso.Duration(months=12))
    assert iso.Duration(years=1) != timedelta(days=365)
    assert iso.Duration(days=3) == timedelta(days=3)
    assert left.years == 1 and left.months == 2 and left.days == 3
    assert pickle.loads(pickle.dumps(left)) == left


def test_duration_conversion_and_fractional_calendar_errors():
    delta = iso.Duration(months=1)
    assert delta.totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError):
        delta.totimedelta()
    with pytest.raises(ValueError):
        delta.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))
    for fraction in (iso.Duration(months=Decimal("0.5")), iso.Duration(years=Decimal("1.5"))):
        with pytest.raises(ValueError):
            date(2024, 1, 1) + fraction
        with pytest.raises(ValueError):
            date(2024, 1, 1) - fraction
    with pytest.raises(TypeError):
        delta + 1
    with pytest.raises(TypeError):
        delta * 1.5


def test_calendar_arithmetic_preserves_datetime_clock_and_zone():
    value = datetime(2024, 1, 31, 23, 30, 10, 123, tzinfo=iso.UTC)
    result = value + iso.Duration(months=1, hours=2)
    assert result == datetime(2024, 3, 1, 1, 30, 10, 123, tzinfo=iso.UTC)
    assert result.tzinfo is iso.UTC
