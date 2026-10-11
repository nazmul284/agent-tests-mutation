"""Tests for the isodate library, written from the library's source/specification."""

import copy
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    LOCAL,
    UTC,
    Duration,
    FixedOffset,
    ISO8601Error,
    date_isoformat,
    datetime_isoformat,
    duration_isoformat,
    parse_date,
    parse_datetime,
    parse_duration,
    parse_time,
    parse_tzinfo,
    strftime,
    time_isoformat,
    tz_isoformat,
)
from isodate.duration import fquotmod, max_days_in_month
from isodate.isotzinfo import build_tzinfo
from isodate.tzinfo import ZERO, LocalTimezone, Utc


# --------------------------------------------------------------------------
# parse_date
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        ("2000-02-29", date(2000, 2, 29)),
        # ordinal
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("2000-001", date(2000, 1, 1)),
        ("2000-366", date(2000, 12, 31)),
        ("1999-365", date(1999, 12, 31)),
        # week dates
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        ("2004-W01-1", date(2003, 12, 29)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2005-W01-1", date(2005, 1, 3)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W52-7", date(2011, 1, 2)),
        # reduced accuracy
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
    ],
)
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=15) == date(1985, 4, 15)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    # explicit values are never overridden by defaults
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)


def test_parse_date_expanded():
    assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+0019850412", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-102", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-W15-5", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
    assert parse_date("+0019", yeardigits=6) == date(1901, 1, 1)
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    assert parse_date("+19850412", expanded=True) == date(1985, 4, 12)
    # a sign is mandatory for expanded format
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", expanded=True)
    # and forbidden for the normal one
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")
    # negative years can not be represented
    with pytest.raises(ValueError):
        parse_date("-1985-04-12", expanded=True)


@pytest.mark.parametrize(
    "text",
    ["", "garbage", "1985-4-12", "1985/04/12", "85-04-12", "1985-04-12 ", " 1985-04-12",
     "1985-04-12\n", "1985-W1-1", "1985-W15-55", "12345", "1985-0412", "198-04-12",
     "19850", "1985-1", "1985-W", "1985-04-1"],
)
def test_parse_date_unrecognised(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1999-02-29", "1985-04-31",
                                  "1985-04-00"])
def test_parse_date_invalid_values(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_iso8601error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)


# --------------------------------------------------------------------------
# parse_time
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("20:23:12", time(20, 23, 12)),
        ("202312", time(20, 23, 12)),
        ("T20:23:12", time(20, 23, 12)),
        ("T202312", time(20, 23, 12)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
        ("20:23", time(20, 23)),
        ("2023", time(20, 23)),
        ("20", time(20)),
        ("T20", time(20)),
        ("20:23:12.5", time(20, 23, 12, 500000)),
        ("20:23:12,25", time(20, 23, 12, 250000)),
        ("202312.000001", time(20, 23, 12, 1)),
        ("20:23:12.123456", time(20, 23, 12, 123456)),
        ("00:00:00.1", time(0, 0, 0, 100000)),
        ("20:23:12.9999999", time(20, 23, 12, 999999)),
        ("20:23.5", time(20, 23, 30)),
        ("2023,25", time(20, 23, 15)),
        ("20:23.25", time(20, 23, 15)),
        ("20.5", time(20, 30)),
        ("20,25", time(20, 15)),
        ("20.0", time(20)),
    ],
)
def test_parse_time_formats(text, expected):
    assert parse_time(text) == expected


def test_parse_time_naive_has_no_tzinfo():
    assert parse_time("20:23:12").tzinfo is None


def test_parse_time_zones():
    t = parse_time("20:23:12Z")
    assert t.tzinfo is UTC
    assert t.utcoffset() == timedelta(0)

    t = parse_time("20:23:12+05:30")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    assert t.tzname() == "+05:30"

    t = parse_time("202312+0530")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    assert t.tzname() == "+0530"

    t = parse_time("20:23:12-08:00")
    assert t.utcoffset() == timedelta(hours=-8)

    t = parse_time("20:23:12-05:30")
    assert t.utcoffset() == -timedelta(hours=5, minutes=30)

    t = parse_time("20:23:12+05")
    assert t.utcoffset() == timedelta(hours=5)

    t = parse_time("2023-0330")
    assert t.utcoffset() == -timedelta(hours=3, minutes=30)

    t = parse_time("20+0100")
    assert t == time(20, tzinfo=FixedOffset(1, 0))
    assert t.utcoffset() == timedelta(hours=1)

    t = parse_time("20:23:12.5Z")
    assert t.microsecond == 500000
    assert t.utcoffset() == ZERO


@pytest.mark.parametrize(
    "text",
    ["", "garbage", "20:", "2:23:12", "20:2:12", "20-23-12", "20:23:12Q", " 20:23:12",
     "20:23:12 ", "20:23:12\n", "20:23:12.", "20:23:12+5", "202", "20:23:12+05:3"],
)
def test_parse_time_unrecognised(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["24:00:00", "25:00:00", "20:60:00", "20:23:60", "20:23:99",
                                  "99"])
def test_parse_time_invalid_values(text):
    with pytest.raises(ValueError):
        parse_time(text)


# --------------------------------------------------------------------------
# parse_datetime
# --------------------------------------------------------------------------
def test_parse_datetime():
    assert parse_datetime("1985-04-12T10:15:30") == datetime(1985, 4, 12, 10, 15, 30)
    assert parse_datetime("19850412T101530") == datetime(1985, 4, 12, 10, 15, 30)
    assert parse_datetime("1985-04-12T10:15:30.5") == datetime(1985, 4, 12, 10, 15, 30, 500000)
    assert parse_datetime("1985-102T10:15") == datetime(1985, 4, 12, 10, 15)
    assert parse_datetime("1985-W15-5T10") == datetime(1985, 4, 12, 10)
    assert parse_datetime("1985-04-12T00:00:00") == datetime(1985, 4, 12)


def test_parse_datetime_timezone():
    dt = parse_datetime("1985-04-12T10:15:30Z")
    assert dt.tzinfo is UTC
    assert dt.utcoffset() == timedelta(0)
    dt = parse_datetime("1985-04-12T10:15:30+05:30")
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert dt == datetime(1985, 4, 12, 4, 45, 30, tzinfo=UTC)
    dt = parse_datetime("1985-04-12T10:15:30-0800")
    assert dt.utcoffset() == timedelta(hours=-8)
    assert dt == datetime(1985, 4, 12, 18, 15, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    "text",
    ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10:15:30T", "T10:15:30", "", "10:15:30",
     "1985-04-12t10:15:30"],
)
def test_parse_datetime_errors(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


def test_parse_datetime_bad_parts():
    with pytest.raises(ISO8601Error):
        parse_datetime("garbage T10:15:30")
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12Tgarbage")
    with pytest.raises(ValueError):
        parse_datetime("1985-13-12T10:15:30")


# --------------------------------------------------------------------------
# parse_duration
# --------------------------------------------------------------------------
def test_parse_duration_timedelta():
    r = parse_duration("P1D")
    assert type(r) is timedelta
    assert r == timedelta(days=1)
    assert parse_duration("PT36H") == timedelta(hours=36)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    assert parse_duration("PT1S") == timedelta(seconds=1)
    assert parse_duration("P1W") == timedelta(weeks=1)
    assert parse_duration("P2W") == timedelta(days=14)
    assert parse_duration("P1W2D") == timedelta(days=9)
    assert parse_duration("P3DT4H5M6S") == timedelta(days=3, hours=4, minutes=5, seconds=6)
    assert parse_duration("PT0.5S") == timedelta(milliseconds=500)
    assert parse_duration("PT0,5S") == timedelta(milliseconds=500)
    assert parse_duration("P1.5D") == timedelta(hours=36)
    assert parse_duration("PT1.5H") == timedelta(minutes=90)
    assert parse_duration("PT1.5M") == timedelta(seconds=90)
    assert parse_duration("PT0S") == timedelta(0)
    assert parse_duration("P0D") == timedelta(0)
    assert parse_duration("PT100H") == timedelta(hours=100)
    assert parse_duration("PT0.000001S") == timedelta(microseconds=1)


def test_parse_duration_minutes_vs_months():
    d = parse_duration("P1M")
    assert isinstance(d, Duration)
    assert d.months == 1
    assert d.tdelta == timedelta(0)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    d = parse_duration("P1MT1M")
    assert d.months == 1
    assert d.tdelta == timedelta(minutes=1)


def test_parse_duration_full():
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    assert d == Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)

    d = parse_duration("P1Y")
    assert isinstance(d, Duration)
    assert (d.years, d.months, d.tdelta) == (1, 0, timedelta(0))

    d = parse_duration("P10Y")
    assert d.years == 10

    d = parse_duration("P1Y1W")
    assert d.tdelta == timedelta(weeks=1)


def test_parse_duration_fractional_years_months():
    d = parse_duration("P1.5Y")
    assert d.years == Decimal("1.5")
    d = parse_duration("P0,5M")
    assert d.months == Decimal("0.5")
    assert isinstance(d.months, Decimal)


def test_parse_duration_negative():
    assert parse_duration("-P1D") == timedelta(days=-1)
    assert parse_duration("-PT1H30M") == -timedelta(hours=1, minutes=30)
    assert parse_duration("-P1W") == timedelta(weeks=-1)
    d = parse_duration("-P1Y2M")
    assert d.years == -1
    assert d.months == -2
    assert d == Duration(years=-1, months=-2)
    d = parse_duration("-P1Y2M3D")
    assert d.tdelta == timedelta(days=-3)
    # an explicit plus sign is accepted by the grammar
    assert parse_duration("+P1D") == timedelta(days=1)
    d = parse_duration("+P1Y")
    assert d.years == 1


def test_parse_duration_as_timedelta_if_possible_false():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=1)
    assert d.years == 0 and d.months == 0
    d = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=-1)


def test_parse_duration_alternative_format():
    d = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(d, Duration)
    assert d == Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert d == Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P0001-02-03T04:05:06.25")
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=250000)


@pytest.mark.parametrize(
    "text",
    ["", "1D", "P", "PD", "P1", "P1Y2", "PT1X", "P1H", "P-1D", "P1D1Y", "P1Y2D3M", "PT1S1M",
     "P1.D", "P.5D", "garbage", "P1D ", " P1D", "p1d", "--P1D"],
)
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


@pytest.mark.parametrize("value", [None, 5, b"P1D", 1.5, ["P1D"]])
def test_parse_duration_type_error(value):
    with pytest.raises(TypeError):
        parse_duration(value)


# --------------------------------------------------------------------------
# strftime / formatting
# --------------------------------------------------------------------------
def test_date_isoformat():
    d = date(1985, 4, 12)
    assert date_isoformat(d) == "1985-04-12"
    assert date_isoformat(d, isodate.DATE_BAS_COMPLETE) == "19850412"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "1985-W15-5"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK_COMPLETE) == "1985W155"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK) == "1985-W15"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK) == "1985W15"
    assert date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "1985-102"
    assert date_isoformat(d, isodate.DATE_BAS_ORD_COMPLETE) == "1985102"
    assert date_isoformat(d, isodate.DATE_EXT_MONTH) == "1985-04"
    assert date_isoformat(d, isodate.DATE_BAS_MONTH) == "198504"
    assert date_isoformat(d, isodate.DATE_YEAR) == "1985"
    assert date_isoformat(d, isodate.DATE_CENTURY) == "19"


def test_date_isoformat_padding_and_edges():
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
    assert date_isoformat(date(9999, 12, 31)) == "9999-12-31"
    assert date_isoformat(date(2000, 12, 31), "%j") == "366"
    assert date_isoformat(date(1999, 12, 31), "%j") == "365"
    assert date_isoformat(date(2000, 1, 1), "%j") == "001"
    assert date_isoformat(date(1985, 4, 14), "%w") == "7"  # Sunday
    assert date_isoformat(date(1985, 4, 8), "%w") == "1"  # Monday
    assert date_isoformat(date(1985, 4, 12), "%d/%m") == "12/04"
    assert date_isoformat(date(2000, 1, 1), "%C") == "20"
    assert date_isoformat(date(1999, 1, 1), "%C") == "19"


def test_date_isoformat_yeardigits():
    d = date(1985, 4, 12)
    assert date_isoformat(d, "%Y-%m-%d", 6) == "+001985-04-12"
    assert date_isoformat(d, "%C", 6) == "+0019"
    assert date_isoformat(d, "%Y", 4) == "1985"


def test_strftime_literals_and_unknown():
    d = date(1985, 4, 12)
    assert strftime(d, "%%") == "%"
    assert strftime(d, "%%Y") == "%Y"
    assert strftime(d, "100%% %Y") == "100% 1985"
    assert strftime(d, "abc") == "abc"
    assert strftime(d, "%q") == "%q"
    assert strftime(d, "") == ""


def test_week_formatting_matches_parsing():
    for text in ["1985-W15-5", "2004-W01-1", "2005-W01-1", "2009-W53-7", "2010-W52-7"]:
        d = parse_date(text)
        assert date_isoformat(d, "W%W-%w") == text[text.index("W"):]


def test_time_isoformat():
    assert time_isoformat(time(20, 23, 12)) == "20:23:12"
    assert time_isoformat(time(5, 3, 2)) == "05:03:02"
    assert time_isoformat(time(20, 23, 12), isodate.TIME_BAS_COMPLETE) == "202312"
    assert time_isoformat(time(20, 23, 12), isodate.TIME_EXT_MINUTE) == "20:23"
    assert time_isoformat(time(20, 23, 12), isodate.TIME_BAS_MINUTE) == "2023"
    assert time_isoformat(time(20, 23, 12), isodate.TIME_HOUR) == "20"
    assert time_isoformat(time(20, 23, 12, 123), "%H:%M:%S.%f") == "20:23:12.000123"
    assert time_isoformat(time(0, 0, 0)) == "00:00:00"


def test_time_isoformat_timezone():
    assert time_isoformat(time(20, 23, 12, tzinfo=UTC)) == "20:23:12Z"
    assert time_isoformat(time(20, 23, 12, tzinfo=FixedOffset(5, 30, "x"))) == "20:23:12+05:30"
    assert time_isoformat(time(20, 23, 12, tzinfo=FixedOffset(-8, 0, "x"))) == "20:23:12-08:00"
    t = time(20, 23, 12, tzinfo=FixedOffset(5, 30, "x"))
    assert time_isoformat(t, "%H%M%S%z") == "202312+0530"
    assert time_isoformat(t, "%H%M%S%h") == "202312+05"


def test_datetime_isoformat():
    dt = datetime(1985, 4, 12, 10, 15, 30)
    assert datetime_isoformat(dt) == "1985-04-12T10:15:30"
    assert datetime_isoformat(dt.replace(tzinfo=UTC)) == "1985-04-12T10:15:30Z"
    tz = FixedOffset(5, 30, "x")
    assert datetime_isoformat(dt.replace(tzinfo=tz)) == "1985-04-12T10:15:30+05:30"
    assert datetime_isoformat(dt.replace(tzinfo=tz), isodate.DT_BAS_COMPLETE) == (
        "19850412T101530+0530"
    )
    assert datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE) == "1985-102T10:15:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_ORD_COMPLETE) == "1985102T101530"
    assert datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE) == "1985-W15-5T10:15:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "1985W155T101530"


def test_datetime_roundtrip():
    for text in [
        "1985-04-12T10:15:30",
        "1985-04-12T10:15:30Z",
        "1985-04-12T10:15:30+05:30",
        "1985-04-12T10:15:30-08:00",
        "2000-02-29T23:59:59Z",
        "0001-01-01T00:00:00",
    ]:
        assert datetime_isoformat(parse_datetime(text)) == text


def test_date_roundtrip_all_formats():
    d = date(2004, 12, 31)
    for fmt in [
        isodate.DATE_EXT_COMPLETE,
        isodate.DATE_BAS_COMPLETE,
        isodate.DATE_EXT_WEEK_COMPLETE,
        isodate.DATE_BAS_WEEK_COMPLETE,
        isodate.DATE_EXT_ORD_COMPLETE,
        isodate.DATE_BAS_ORD_COMPLETE,
    ]:
        assert parse_date(date_isoformat(d, fmt)) == d


# --------------------------------------------------------------------------
# duration_isoformat
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(0), "P0D"),
        (timedelta(days=1), "P1D"),
        (timedelta(days=1, hours=2), "P1DT2H"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(seconds=90), "PT1M30S"),
        (timedelta(hours=1, seconds=1), "PT1H1S"),
        (timedelta(seconds=1.5), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(microseconds=500000), "PT0.5S"),
        (timedelta(days=400), "P400D"),
        (timedelta(days=3, hours=4, minutes=5, seconds=6), "P3DT4H5M6S"),
        (timedelta(days=-1), "-P1D"),
        (-timedelta(hours=1, minutes=30), "-PT1H30M"),
        (timedelta(seconds=-1), "-PT1S"),
        (timedelta(microseconds=-1), "-PT0.000001S"),
        (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
        (Duration(years=1), "P1Y"),
        (Duration(months=3), "P3M"),
        (Duration(years=1, hours=1), "P1YT1H"),
        (Duration(years=1.5), "P1.5Y"),
        (Duration(days=1), "P1D"),
        (Duration(), "P0D"),
        (Duration(years=-1), "-P1Y"),
        (Duration(months=-2), "-P2M"),
        (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
        (Duration(days=-1), "-P1D"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


def test_duration_isoformat_weeks():
    assert duration_isoformat(timedelta(weeks=2), isodate.D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(days=6), isodate.D_WEEK) == "P0W"
    assert duration_isoformat(timedelta(weeks=-2), isodate.D_WEEK) == "-P2W"


def test_duration_isoformat_alternative():
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d, isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(d, isodate.D_ALT_BAS) == "P00010203T040506"
    d = Duration(years=12, months=11, days=30, hours=23, minutes=59, seconds=58)
    assert duration_isoformat(d, isodate.D_ALT_EXT) == "P0012-11-30T23:59:58"


def test_duration_strftime_codes():
    td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    assert strftime(td, "%d|%H|%M|%S|%f") == "03|04|05|06|000007"
    assert strftime(td, "%%") == "%"
    assert strftime(td, "x%Py") == "x3DT4H5M6.000007Sy"
    assert strftime(td, "%P") == "3DT4H5M6.000007S"
    d = Duration(years=7, months=8)
    assert strftime(d, "%Y-%m") == "0007-08"
    assert strftime(d, "%Y", 6) == "+000007"


def test_duration_roundtrip():
    for text in [
        "P1D",
        "PT1H",
        "PT1M",
        "PT1S",
        "P1DT2H3M4S",
        "PT0.5S",
        "P1Y",
        "P1M",
        "P1Y2M",
        "P1Y2M3DT4H5M6S",
        "P1Y1D",
        "-P1D",
        "-P1Y2M3DT4H5M6S",
        "-PT1.5S",
        "P0D",
    ]:
        assert duration_isoformat(parse_duration(text)) == text


def test_duration_week_normalised_to_days():
    assert duration_isoformat(parse_duration("P2W")) == "P14D"
    assert duration_isoformat(parse_duration("P2W"), isodate.D_WEEK) == "P2W"


def test_duration_alt_roundtrip():
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    for fmt in (isodate.D_ALT_EXT, isodate.D_ALT_BAS):
        assert parse_duration(duration_isoformat(d, fmt)) == d


# --------------------------------------------------------------------------
# tz handling
# --------------------------------------------------------------------------
def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "+05:30"
    assert parse_tzinfo("-0800").utcoffset(None) == timedelta(hours=-8)
    assert parse_tzinfo("+05").utcoffset(None) == timedelta(hours=5)
    assert parse_tzinfo("-05:30").utcoffset(None) == -timedelta(hours=5, minutes=30)
    assert parse_tzinfo("+00:00").utcoffset(None) == ZERO


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("+02:30", "+", 2, 30)
    assert tz.utcoffset(None) == timedelta(hours=2, minutes=30)
    tz = build_tzinfo("-02:30", "-", 2, 30)
    assert tz.utcoffset(None) == -timedelta(hours=2, minutes=30)
    assert tz.tzname(None) == "-02:30"


def test_tz_isoformat():
    naive = datetime(2000, 1, 1)
    assert tz_isoformat(naive) == ""
    assert tz_isoformat(naive, "%z") == ""
    assert tz_isoformat(naive.replace(tzinfo=UTC)) == "Z"
    assert tz_isoformat(naive.replace(tzinfo=UTC), "%h") == "Z"
    assert tz_isoformat(naive.replace(tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    dt = naive.replace(tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(dt) == "+05:30"
    assert tz_isoformat(dt, "%Z") == "+05:30"
    assert tz_isoformat(dt, "%z") == "+0530"
    assert tz_isoformat(dt, "%h") == "+05"
    dt = naive.replace(tzinfo=FixedOffset(-3, -30, "x"))
    assert tz_isoformat(dt, "%Z") == "-03:30"
    assert tz_isoformat(dt, "%z") == "-0330"
    assert tz_isoformat(dt, "%h") == "-03"
    dt = naive.replace(tzinfo=FixedOffset(-8, 0, "x"))
    assert tz_isoformat(dt, "%Z") == "-08:00"
    dt = naive.replace(tzinfo=FixedOffset(0, 45, "x"))
    assert tz_isoformat(dt, "%Z") == "+00:45"
    dt = naive.replace(tzinfo=FixedOffset(0, -45, "x"))
    assert tz_isoformat(dt, "%Z") == "-00:45"
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%x")


def test_tz_isoformat_overflow():
    class Big(Utc):
        def utcoffset(self, dt):
            return timedelta(hours=100)

        def dst(self, dt):
            return ZERO

    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=Big()), "%Z")
    class Edge(Big):
        def utcoffset(self, dt):
            return timedelta(hours=99, minutes=59)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Edge()), "%Z") == "+99:59"


def test_tz_isoformat_utcoffset_none():
    class NoOffset(Utc):
        def utcoffset(self, dt):
            return None

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=NoOffset())) == ""


def test_tz_isoformat_nonzero_dst_with_zero_offset_is_not_z():
    class Odd(Utc):
        def utcoffset(self, dt):
            return ZERO

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Odd()), "%Z") == "+00:00"


def test_utc_class():
    assert UTC.utcoffset(None) == ZERO
    assert UTC.dst(None) == ZERO
    assert UTC.tzname(None) == "UTC"
    assert isinstance(UTC, Utc)
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert copy.deepcopy(UTC) is UTC


def test_fixed_offset_class():
    tz = FixedOffset(5, 30, "name")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "name"
    assert tz.dst(None) == ZERO
    assert repr(tz) == "<FixedOffset 'name'>"
    default = FixedOffset()
    assert default.utcoffset(None) == ZERO
    assert default.tzname(None) == "UTC"
    neg = FixedOffset(-5, -30, "n")
    assert neg.utcoffset(None) == -timedelta(hours=5, minutes=30)
    dt = datetime(2000, 1, 1, 12, tzinfo=tz)
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert dt == datetime(2000, 1, 1, 6, 30, tzinfo=UTC)


def test_local_timezone():
    dt = datetime(2000, 1, 1, 12)
    assert isinstance(LOCAL, LocalTimezone)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert isinstance(LOCAL.tzname(dt), str)
    assert (LOCAL.dst(dt) != ZERO) == LOCAL._isdst(dt)
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


# --------------------------------------------------------------------------
# helper functions in duration module
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "year, month, days",
    [
        (2001, 1, 31), (2001, 2, 28), (2001, 3, 31), (2001, 4, 30), (2001, 5, 31),
        (2001, 6, 30), (2001, 7, 31), (2001, 8, 31), (2001, 9, 30), (2001, 10, 31),
        (2001, 11, 30), (2001, 12, 31),
        (2000, 2, 29), (1900, 2, 28), (2100, 2, 28), (2024, 2, 29), (2023, 2, 28),
        (1600, 2, 29), (1996, 2, 29), (1997, 2, 28), (2400, 2, 29), (1800, 2, 28),
    ],
)
def test_max_days_in_month(year, month, days):
    assert max_days_in_month(year, month) == days


@pytest.mark.parametrize(
    "val, low, high, expected",
    [
        (13, 1, 13, (1, 1)),
        (12, 1, 13, (0, 12)),
        (1, 1, 13, (0, 1)),
        (0, 1, 13, (-1, 12)),
        (-1, 1, 13, (-1, 11)),
        (-11, 1, 13, (-1, 1)),
        (-12, 1, 13, (-2, 12)),
        (25, 1, 13, (2, 1)),
        (24, 1, 13, (1, 12)),
        (7, 0, 5, (1, 2)),
        (-7, 0, 5, (-2, 3)),
    ],
)
def test_fquotmod(val, low, high, expected):
    div, mod = fquotmod(Decimal(val), low, high)
    assert (div, mod) == expected
    assert isinstance(div, int)


# --------------------------------------------------------------------------
# Duration
# --------------------------------------------------------------------------
def test_duration_init_and_attributes():
    d = Duration(days=2, seconds=5, microseconds=7, milliseconds=1, minutes=1, hours=1,
                 weeks=1, months=3, years=4)
    assert d.years == 4
    assert d.months == 3
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)
    assert d.tdelta == timedelta(days=2, seconds=5, microseconds=7, milliseconds=1,
                                 minutes=1, hours=1, weeks=1)
    # attributes are proxied to the timedelta
    assert d.days == 9
    assert d.seconds == 3665
    assert d.microseconds == 1007
    assert d.total_seconds() == d.tdelta.total_seconds()
    with pytest.raises(AttributeError):
        d.nonexistent_attribute
    assert Duration(years=Decimal("1.5")).years == Decimal("1.5")
    assert Duration(months=0.1).months == Decimal("0.1")


def test_duration_equality():
    assert Duration(years=1) == Duration(years=1)
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1) != Duration(years=2)
    assert Duration(months=1) != Duration(months=2)
    assert Duration(days=1) != Duration(days=2)
    assert Duration(years=1, days=1) != Duration(years=1)
    assert not (Duration(years=1) != Duration(months=12))
    assert not (Duration(years=1) == Duration(years=2))
    # compare with timedelta
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1) != timedelta(days=365)
    assert not (Duration(years=1) == timedelta(days=365))
    assert Duration(months=1) != timedelta(days=30)
    assert Duration() == timedelta(0)


def test_duration_hash():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2
    d = {Duration(months=1): "x"}
    assert d[Duration(months=1)] == "x"


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3, hours=4)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == -timedelta(days=3, hours=4)
    assert -(-Duration(years=1, days=1)) == Duration(years=1, days=1)
    assert -Duration() == Duration()


def test_duration_add_duration():
    r = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
    assert isinstance(r, Duration)
    assert r.years == 5
    assert r.months == 7
    assert r.tdelta == timedelta(days=9)
    r = Duration(years=1, hours=1) + Duration(years=-1, hours=-1)
    assert r == Duration()


def test_duration_add_timedelta():
    r = Duration(years=1, days=1) + timedelta(days=2)
    assert isinstance(r, Duration)
    assert r.years == 1
    assert r.tdelta == timedelta(days=3)
    r = timedelta(days=2) + Duration(years=1, days=1)
    assert isinstance(r, Duration)
    assert r.years == 1
    assert r.tdelta == timedelta(days=3)


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(years=1) + 1
    with pytest.raises(TypeError):
        Duration(years=1) + "x"


def test_duration_sub():
    r = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
    assert isinstance(r, Duration)
    assert (r.years, r.months, r.tdelta) == (4, 5, timedelta(days=6))
    r = Duration(years=1, days=5) - timedelta(days=2)
    assert isinstance(r, Duration)
    assert r.years == 1
    assert r.tdelta == timedelta(days=3)
    r = timedelta(days=5) - Duration(days=2)
    assert isinstance(r, Duration)
    assert r.tdelta == timedelta(days=3)
    r = timedelta(days=5) - Duration(years=1, days=2)
    assert r.years == -1
    assert r.tdelta == timedelta(days=3)
    assert Duration(years=1) - Duration(years=1) == Duration()
    with pytest.raises(TypeError):
        Duration(years=1) - "x"


def test_duration_mul():
    r = Duration(years=1, months=2, days=3) * 3
    assert isinstance(r, Duration)
    assert (r.years, r.months, r.tdelta) == (3, 6, timedelta(days=9))
    r = 2 * Duration(years=1, months=2, days=3)
    assert (r.years, r.months, r.tdelta) == (2, 4, timedelta(days=6))
    assert Duration(years=1) * 0 == Duration()
    assert (Duration(years=1, days=1) * -1) == -Duration(years=1, days=1)
    with pytest.raises(TypeError):
        Duration(years=1) * 1.5
    with pytest.raises(TypeError):
        Duration(years=1) * "a"


def test_duration_str_repr():
    d = Duration(years=1, months=2, days=3)
    assert str(d) == "1 years, 2 months, 3 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(days=1, hours=2)) == "1 day, 2:00:00"
    assert str(Duration(years=5)) == "5 years, 0:00:00"
    assert repr(d) == "isodate.duration.Duration(3, 0, 0, years=1, months=2)"
    assert repr(Duration(days=1, seconds=2, microseconds=3)) == (
        "isodate.duration.Duration(1, 2, 3, years=0, months=0)"
    )


def test_duration_pickle_copy():
    d = Duration(years=1, months=2, days=3, hours=4)
    assert pickle.loads(pickle.dumps(d)) == d
    assert copy.copy(d) == d
    assert copy.deepcopy(d) == d
    p = pickle.loads(pickle.dumps(d))
    assert p.years == 1 and p.months == 2 and p.tdelta == d.tdelta


# --------------------------------------------------------------------------
# Duration + date / datetime
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 15), Duration(months=1), date(2000, 2, 15)),
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(1999, 1, 31), Duration(months=1), date(1999, 2, 28)),
        (date(2000, 1, 31), Duration(months=2), date(2000, 3, 31)),
        (date(2000, 1, 31), Duration(months=3), date(2000, 4, 30)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 11, 15), Duration(months=2), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=12), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(2001, 2, 15)),
        (date(2000, 1, 15), Duration(months=24), date(2002, 1, 15)),
        (date(2000, 1, 15), Duration(months=25), date(2002, 2, 15)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 2, 29), Duration(years=100), date(2100, 2, 28)),
        (date(2000, 1, 15), Duration(years=1, months=1, days=1), date(2001, 2, 16)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 15), Duration(months=-12), date(1999, 1, 15)),
        (date(2000, 1, 15), Duration(months=-13), date(1998, 12, 15)),
        (date(2000, 1, 15), Duration(years=-1), date(1999, 1, 15)),
        (date(2000, 1, 15), Duration(days=10), date(2000, 1, 25)),
        (date(2000, 1, 15), Duration(), date(2000, 1, 15)),
        (date(2000, 5, 15), Duration(months=7), date(2000, 12, 15)),
        (date(2000, 5, 15), Duration(months=8), date(2001, 1, 15)),
    ],
)
def test_add_duration_to_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


def test_add_duration_to_datetime():
    dt = datetime(2000, 1, 31, 10, 30, 15, 5)
    assert dt + Duration(months=1) == datetime(2000, 2, 29, 10, 30, 15, 5)
    assert dt + Duration(months=1, hours=20) == datetime(2000, 3, 1, 6, 30, 15, 5)
    assert Duration(years=1, hours=1) + dt == datetime(2001, 1, 31, 11, 30, 15, 5)
    aware = datetime(2000, 1, 31, 10, tzinfo=UTC)
    r = aware + Duration(months=1)
    assert r == datetime(2000, 2, 29, 10, tzinfo=UTC)
    assert r.tzinfo is UTC
    assert type(dt + Duration(months=1)) is datetime
    assert type(date(2000, 1, 1) + Duration(months=1)) is date


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 15), Duration(months=1), date(2000, 2, 15)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(1999, 3, 31), Duration(months=1), date(1999, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2000, 1, 15), Duration(months=12), date(1999, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(1998, 12, 15)),
        (date(2000, 12, 15), Duration(months=11), date(2000, 1, 15)),
        (date(2000, 12, 15), Duration(months=12), date(1999, 12, 15)),
        (date(2000, 2, 29), Duration(years=1), date(1999, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(1996, 2, 29)),
        (date(2000, 1, 15), Duration(years=1, months=1, days=1), date(1998, 12, 14)),
        (date(2000, 3, 1), Duration(months=1, days=1), date(2000, 1, 31)),
        (date(2000, 1, 15), Duration(months=-1), date(2000, 2, 15)),
        (date(2000, 1, 15), Duration(days=15), date(1999, 12, 31)),
        (date(2000, 1, 15), Duration(), date(2000, 1, 15)),
    ],
)
def test_subtract_duration_from_date(start, dur, expected):
    assert start - dur == expected


def test_subtract_duration_from_datetime():
    dt = datetime(2000, 3, 31, 10, 30)
    assert dt - Duration(months=1) == datetime(2000, 2, 29, 10, 30)
    assert dt - Duration(months=1, hours=11) == datetime(2000, 2, 28, 23, 30)
    assert type(dt - Duration(months=1)) is datetime


def test_add_sub_duration_inverse_when_no_clipping():
    d = date(2000, 6, 15)
    dur = Duration(years=3, months=7, days=11)
    assert (d + dur) - Duration(years=3, months=7) - timedelta(days=11) == d


def test_fractional_duration_with_date_raises():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=1.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=0.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=1.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=0.5)
    # whole numbers expressed as fractions are fine
    assert date(2000, 1, 1) + Duration(months=Decimal("2.0")) == date(2000, 3, 1)
    assert date(2000, 1, 1) + Duration(years=Decimal("1.0")) == date(2001, 1, 1)


def test_subtract_duration_from_timedelta_type():
    r = timedelta(days=10) - Duration(days=3)
    assert isinstance(r, Duration)
    assert r == timedelta(days=7)


def test_date_minus_unsupported_raises():
    with pytest.raises(TypeError):
        "abc" - Duration(years=1)
    with pytest.raises(TypeError):
        5 - Duration(years=1)


# --------------------------------------------------------------------------
# totimedelta
# --------------------------------------------------------------------------
def test_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=datetime(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=datetime(1999, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=datetime(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=datetime(2000, 2, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=365)
    assert Duration(years=1, days=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=366)
    assert Duration(days=2, hours=3).totimedelta(start=datetime(2000, 1, 1)) == timedelta(
        days=2, hours=3
    )
    assert Duration(years=1).totimedelta(end=date(2001, 1, 1)) == timedelta(days=366)


def test_totimedelta_argument_validation():
    d = Duration(months=1)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=datetime(2000, 1, 1), end=datetime(2000, 2, 1))


# --------------------------------------------------------------------------
# Package surface
# --------------------------------------------------------------------------
def test_public_api_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name), name


def test_format_constants():
    assert isodate.DATE_EXT_COMPLETE == "%Y-%m-%d"
    assert isodate.DATE_BAS_COMPLETE == "%Y%m%d"
    assert isodate.TIME_EXT_COMPLETE == "%H:%M:%S"
    assert isodate.TIME_BAS_COMPLETE == "%H%M%S"
    assert isodate.TZ_BAS == "%z"
    assert isodate.TZ_EXT == "%Z"
    assert isodate.TZ_HOUR == "%h"
    assert isodate.D_DEFAULT == "P%P"
    assert isodate.D_WEEK == "P%p"
    assert isodate.DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert isodate.DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
