"""Test suite for the isodate package."""

import pickle
import time as pytime
from datetime import date, datetime, time, timedelta, tzinfo
from decimal import Decimal

import pytest

import isodate
from isodate import (
    D_ALT_BAS,
    D_ALT_BAS_ORD,
    D_ALT_EXT,
    D_ALT_EXT_ORD,
    D_DEFAULT,
    D_WEEK,
    DATE_BAS_COMPLETE,
    DATE_BAS_MONTH,
    DATE_BAS_ORD_COMPLETE,
    DATE_BAS_WEEK,
    DATE_BAS_WEEK_COMPLETE,
    DATE_CENTURY,
    DATE_EXT_COMPLETE,
    DATE_EXT_MONTH,
    DATE_EXT_ORD_COMPLETE,
    DATE_EXT_WEEK,
    DATE_EXT_WEEK_COMPLETE,
    DATE_YEAR,
    DT_BAS_COMPLETE,
    DT_BAS_ORD_COMPLETE,
    DT_BAS_WEEK_COMPLETE,
    DT_EXT_COMPLETE,
    DT_EXT_ORD_COMPLETE,
    DT_EXT_WEEK_COMPLETE,
    LOCAL,
    TIME_BAS_COMPLETE,
    TIME_BAS_MINUTE,
    TIME_EXT_COMPLETE,
    TIME_EXT_MINUTE,
    TIME_HOUR,
    TZ_BAS,
    TZ_EXT,
    TZ_HOUR,
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
from isodate import tzinfo as tzmod
from isodate.duration import fquotmod, max_days_in_month
from isodate.isodates import DATE_REGEX_CACHE, build_date_regexps
from isodate.isotime import TIME_REGEX_CACHE, build_time_regexps
from isodate.isotzinfo import build_tzinfo


@pytest.fixture(autouse=True)
def _clear_regex_caches():
    """Rebuild cached regular expressions in every test."""
    DATE_REGEX_CACHE.clear()
    del TIME_REGEX_CACHE[:]
    yield


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_format_constants():
    assert DATE_BAS_COMPLETE == "%Y%m%d"
    assert DATE_EXT_COMPLETE == "%Y-%m-%d"
    assert DATE_BAS_WEEK_COMPLETE == "%YW%W%w"
    assert DATE_EXT_WEEK_COMPLETE == "%Y-W%W-%w"
    assert DATE_BAS_ORD_COMPLETE == "%Y%j"
    assert DATE_EXT_ORD_COMPLETE == "%Y-%j"
    assert DATE_BAS_WEEK == "%YW%W"
    assert DATE_EXT_WEEK == "%Y-W%W"
    assert DATE_BAS_MONTH == "%Y%m"
    assert DATE_EXT_MONTH == "%Y-%m"
    assert DATE_YEAR == "%Y"
    assert DATE_CENTURY == "%C"
    assert TIME_BAS_COMPLETE == "%H%M%S"
    assert TIME_EXT_COMPLETE == "%H:%M:%S"
    assert TIME_BAS_MINUTE == "%H%M"
    assert TIME_EXT_MINUTE == "%H:%M"
    assert TIME_HOUR == "%H"
    assert TZ_BAS == "%z"
    assert TZ_EXT == "%Z"
    assert TZ_HOUR == "%h"
    assert DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
    assert DT_EXT_ORD_COMPLETE == "%Y-%jT%H:%M:%S%Z"
    assert DT_BAS_ORD_COMPLETE == "%Y%jT%H%M%S%z"
    assert DT_EXT_WEEK_COMPLETE == "%Y-W%W-%wT%H:%M:%S%Z"
    assert DT_BAS_WEEK_COMPLETE == "%YW%W%wT%H%M%S%z"
    assert D_DEFAULT == "P%P"
    assert D_WEEK == "P%p"
    assert D_ALT_EXT == "P%Y-%m-%dT%H:%M:%S"
    assert D_ALT_BAS == "P%Y%m%dT%H%M%S"
    assert D_ALT_EXT_ORD == "P%Y-%jT%H:%M:%S"
    assert D_ALT_BAS_ORD == "P%Y%jT%H%M%S"


def test_iso8601error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("1985", date(1985, 1, 1)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-001", date(1985, 1, 1)),
        ("2000-366", date(2000, 12, 31)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        # 2008-01-01 is a Tuesday in ISO week 1 of 2008
        ("2008-W01-1", date(2007, 12, 31)),
        ("2008-W01-2", date(2008, 1, 1)),
        ("2008-W01", date(2007, 12, 31)),
        ("2008-W52-7", date(2008, 12, 28)),
        # 2010-01-01 is a Friday in ISO week 53 of 2009
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W01", date(2010, 1, 4)),
        ("2010W017", date(2010, 1, 10)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2004-W53-6", date(2005, 1, 1)),
        # 2007-01-01 is a Monday
        ("2007-W01-1", date(2007, 1, 1)),
        ("2007-W01-0", date(2006, 12, 31)),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("+001985-04-12", date(1985, 4, 12)),
        ("+0019850412", date(1985, 4, 12)),
        ("+001985-W15-5", date(1985, 4, 12)),
        ("+001985W155", date(1985, 4, 12)),
        ("+001985-102", date(1985, 4, 12)),
        ("+001985102", date(1985, 4, 12)),
        ("+001985-W15", date(1985, 4, 8)),
        ("+001985W15", date(1985, 4, 8)),
        ("+001985-04", date(1985, 4, 1)),
        ("+00198504", date(1985, 4, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+0019", date(1901, 1, 1)),
    ],
)
def test_parse_date_expanded_6(text, expected):
    assert parse_date(text, 6) == expected
    assert parse_date(text, yeardigits=6, expanded=False) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("+1985-04-12", date(1985, 4, 12)),
        ("+19850412", date(1985, 4, 12)),
        ("+1985-W15-5", date(1985, 4, 12)),
        ("+1985W155", date(1985, 4, 12)),
        ("+1985-102", date(1985, 4, 12)),
        ("+1985102", date(1985, 4, 12)),
        ("+1985-W15", date(1985, 4, 8)),
        ("+1985W15", date(1985, 4, 8)),
        ("+1985-04", date(1985, 4, 1)),
        ("+198504", date(1985, 4, 1)),
        ("+1985", date(1985, 1, 1)),
        ("+19", date(1901, 1, 1)),
    ],
)
def test_parse_date_expanded_4(text, expected):
    assert parse_date(text, 4, True) == expected
    assert parse_date(text, expanded=True) == expected
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize(
    "text",
    [
        "-001985-04-12",
        "-0019850412",
        "-001985-W15-5",
        "-001985-102",
        "-001985-W15",
        "-001985-04",
        "-001985",
        "-0019",
    ],
)
def test_parse_date_negative_years_unsupported(text):
    with pytest.raises(ValueError):
        parse_date(text, 6)


@pytest.mark.parametrize(
    "text",
    ["", "1", "198", "1985-4-12", "1985-04-1", "85-04-12", "1985/04/12", "1985-04-12x",
     "x1985", "1985-W1", "1985-W15-55", "+001985-04-12", "001985-04-12", "1985-1234"],
)
def test_parse_date_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


def test_parse_date_invalid_message():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 date format: 'abc'"):
        parse_date("abc")


def test_parse_date_expanded_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", 6)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", 4, True)


def test_parse_date_out_of_range_values():
    with pytest.raises(ValueError):
        parse_date("1985-13-01")
    with pytest.raises(ValueError):
        parse_date("1985-02-30")


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultmonth=6, defaultday=15) == date(1985, 4, 15)
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("19", defaultmonth=6, defaultday=15) == date(1901, 6, 15)
    assert parse_date("+0019", 6, defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    # month "00" falls back to the default month
    assert parse_date("1985-00", defaultmonth=7) == date(1985, 7, 1)
    assert parse_date("1985-00-05", defaultmonth=7) == date(1985, 7, 5)
    # defaults do not influence week and ordinal dates
    assert parse_date("1985-W15", defaultmonth=6, defaultday=15) == date(1985, 4, 8)
    assert parse_date("1985-102", defaultmonth=6, defaultday=15) == date(1985, 4, 12)


def test_build_date_regexps():
    default = build_date_regexps()
    assert len(default) == 12
    assert default is build_date_regexps(4, False)
    assert (4, False) in DATE_REGEX_CACHE
    expanded6 = build_date_regexps(6)
    assert expanded6 is build_date_regexps(6, True)
    assert expanded6 is build_date_regexps(6, False)
    assert (6, True) in DATE_REGEX_CACHE
    assert (6, False) not in DATE_REGEX_CACHE
    expanded4 = build_date_regexps(4, True)
    assert expanded4 is not default
    assert default[0].match("1985-04-12")
    assert not default[0].match("+1985-04-12")
    assert expanded4[0].match("+1985-04-12")
    assert not expanded4[0].match("1985-04-12")
    assert expanded6[0].match("-001985-04-12")
    # century pattern uses two fewer digits than the year
    assert default[-1].match("19")
    assert not default[-1].match("199")
    assert expanded6[-1].match("+0019")
    assert not expanded6[-1].match("+00190")


def test_build_date_regexps_patterns_order():
    pats = build_date_regexps(4, False)
    samples = [
        "1985-04-12",
        "19850412",
        "1985-W15-5",
        "1985W155",
        "1985-102",
        "1985102",
        "1985-W15",
        "1985W15",
        "1985-04",
        "198504",
        "1985",
        "19",
    ]
    for i, (pat, sample) in enumerate(zip(pats, samples)):
        m = pat.match(sample)
        assert m, (i, sample)
        # anchored at both ends
        assert not pat.match(sample + "0")
        assert not pat.match("0" + sample)
        assert m.group("sign") is None


def test_date_isoformat():
    d = date(1985, 4, 12)
    assert date_isoformat(d) == "1985-04-12"
    assert date_isoformat(d, DATE_BAS_COMPLETE) == "19850412"
    assert date_isoformat(d, DATE_EXT_COMPLETE, 6) == "+001985-04-12"
    assert date_isoformat(d, DATE_YEAR, yeardigits=5) == "+01985"


# ---------------------------------------------------------------------------
# strftime for dates / times / datetimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,expected",
    [
        (DATE_BAS_COMPLETE, "19850412"),
        (DATE_EXT_COMPLETE, "1985-04-12"),
        (DATE_BAS_WEEK_COMPLETE, "1985W155"),
        (DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (DATE_BAS_ORD_COMPLETE, "1985102"),
        (DATE_EXT_ORD_COMPLETE, "1985-102"),
        (DATE_BAS_WEEK, "1985W15"),
        (DATE_EXT_WEEK, "1985-W15"),
        (DATE_BAS_MONTH, "198504"),
        (DATE_EXT_MONTH, "1985-04"),
        (DATE_YEAR, "1985"),
        (DATE_CENTURY, "19"),
        ("%%", "%"),
        ("%d%%%m", "12%04"),
        ("abc", "abc"),
        ("%x", "%x"),
    ],
)
def test_strftime_date(fmt, expected):
    assert strftime(date(1985, 4, 12), fmt) == expected


@pytest.mark.parametrize(
    "fmt,expected",
    [
        (DATE_BAS_COMPLETE, "+0019850412"),
        (DATE_EXT_COMPLETE, "+001985-04-12"),
        (DATE_EXT_WEEK_COMPLETE, "+001985-W15-5"),
        (DATE_EXT_ORD_COMPLETE, "+001985-102"),
        (DATE_YEAR, "+001985"),
        (DATE_CENTURY, "+0019"),
    ],
)
def test_strftime_date_expanded(fmt, expected):
    assert strftime(date(1985, 4, 12), fmt, 6) == expected


def test_strftime_date_misc():
    assert strftime(date(2000, 12, 31), "%j") == "366"
    assert strftime(date(2001, 12, 31), "%j") == "365"
    assert strftime(date(2001, 1, 1), "%j") == "001"
    assert strftime(date(2007, 1, 7), "%w") == "7"
    assert strftime(date(2007, 1, 1), "%w") == "1"
    assert strftime(date(2008, 12, 29), "%Y-W%W-%w") == "2008-W01-1"
    assert strftime(date(5, 2, 3), DATE_EXT_COMPLETE) == "0005-02-03"
    assert strftime(date(2000, 1, 1), DATE_CENTURY) == "20"
    assert strftime(date(1999, 1, 1), DATE_CENTURY) == "19"
    assert strftime(date(1985, 4, 12), DATE_YEAR, 4) == "1985"
    assert strftime(date(1985, 4, 12), DATE_YEAR) == "1985"


def test_strftime_time():
    t = time(3, 4, 5, 6)
    assert strftime(t, TIME_EXT_COMPLETE) == "03:04:05"
    assert strftime(t, TIME_BAS_COMPLETE) == "030405"
    assert strftime(t, TIME_EXT_MINUTE) == "03:04"
    assert strftime(t, TIME_BAS_MINUTE) == "0304"
    assert strftime(t, TIME_HOUR) == "03"
    assert strftime(t, "%H:%M:%S.%f") == "03:04:05.000006"
    assert strftime(t, TIME_EXT_COMPLETE + TZ_EXT) == "03:04:05"
    assert strftime(t, "%z%h") == ""


def test_time_isoformat():
    assert time_isoformat(time(23, 20, 50)) == "23:20:50"
    assert time_isoformat(time(23, 20, 50, tzinfo=UTC)) == "23:20:50Z"
    assert time_isoformat(time(23, 20, 50, tzinfo=FixedOffset(1, 30, "x"))) == "23:20:50+01:30"
    assert time_isoformat(time(23, 20, 50, tzinfo=FixedOffset(-5, 0, "x")),
                          TIME_BAS_COMPLETE + TZ_BAS) == "232050-0500"
    assert time_isoformat(time(23, 20, 50, tzinfo=FixedOffset(-5, 0, "x")),
                          TIME_HOUR + TZ_HOUR) == "23-05"


def test_datetime_isoformat():
    dt = datetime(1985, 4, 12, 23, 20, 50)
    assert datetime_isoformat(dt) == "1985-04-12T23:20:50"
    dt = datetime(1985, 4, 12, 23, 20, 50, tzinfo=UTC)
    assert datetime_isoformat(dt) == "1985-04-12T23:20:50Z"
    dt = datetime(1985, 4, 12, 23, 20, 50, tzinfo=FixedOffset(-5, -30, "x"))
    assert datetime_isoformat(dt) == "1985-04-12T23:20:50-05:30"
    assert datetime_isoformat(dt, DT_BAS_COMPLETE) == "19850412T232050-0530"
    assert datetime_isoformat(dt, DT_EXT_ORD_COMPLETE) == "1985-102T23:20:50-05:30"
    assert datetime_isoformat(dt, DT_BAS_ORD_COMPLETE) == "1985102T232050-0530"
    assert datetime_isoformat(dt, DT_EXT_WEEK_COMPLETE) == "1985-W15-5T23:20:50-05:30"
    assert datetime_isoformat(dt, DT_BAS_WEEK_COMPLETE) == "1985W155T232050-0530"
    dt = datetime(1985, 4, 12, 23, 20, 50, tzinfo=FixedOffset(2, 0, "x"))
    assert datetime_isoformat(dt, "%h") == "+02"
    assert datetime_isoformat(dt, "%z") == "+0200"
    assert datetime_isoformat(dt, "%Z") == "+02:00"


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("23:20:50", time(23, 20, 50)),
        ("232050", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("T232050", time(23, 20, 50)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50,5", time(23, 20, 50, 500000)),
        ("232050.5", time(23, 20, 50, 500000)),
        ("232050,5", time(23, 20, 50, 500000)),
        ("12:30:15.1234567", time(12, 30, 15, 123456)),
        ("12:30:15.9999999", time(12, 30, 15, 999999)),
        ("12:30:15.000001", time(12, 30, 15, 1)),
        ("23:20", time(23, 20)),
        ("2320", time(23, 20)),
        ("T23:20", time(23, 20)),
        ("T2320", time(23, 20)),
        ("23:20.5", time(23, 20, 30)),
        ("23:20,5", time(23, 20, 30)),
        ("2320.5", time(23, 20, 30)),
        ("2320,5", time(23, 20, 30)),
        ("12:30.0001", time(12, 30, 0, 6000)),
        ("12:30.1234567", time(12, 30, 7, 407402)),
        ("12:30.00000001", time(12, 30, 0, 0)),
        ("23", time(23)),
        ("T23", time(23)),
        ("23.5", time(23, 30)),
        ("23,5", time(23, 30)),
        ("12.25", time(12, 15)),
        ("12.01", time(12, 0, 36)),
        ("12.0001", time(12, 0, 0, 360000)),
        ("12.123", time(12, 7, 22, 800000)),
        ("00:00:00", time(0)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "suffix,offset",
    [
        ("+01:00", timedelta(hours=1)),
        ("+0100", timedelta(hours=1)),
        ("+01", timedelta(hours=1)),
        ("-05:30", -timedelta(hours=5, minutes=30)),
        ("-0530", -timedelta(hours=5, minutes=30)),
        ("-05", -timedelta(hours=5)),
        ("+00:00", timedelta(0)),
        ("+12:45", timedelta(hours=12, minutes=45)),
    ],
)
@pytest.mark.parametrize(
    "base,expected",
    [
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("232050", time(23, 20, 50)),
        ("23:20.5", time(23, 20, 30)),
        ("2320", time(23, 20)),
        ("23.5", time(23, 30)),
    ],
)
def test_parse_time_with_offset(base, expected, suffix, offset):
    result = parse_time(base + suffix)
    assert result.replace(tzinfo=None) == expected
    assert result.tzinfo.utcoffset(None) == offset
    assert result.tzinfo.tzname(None) == suffix


@pytest.mark.parametrize("base", ["23:20:50", "232050", "23:20", "2320", "23"])
def test_parse_time_utc(base):
    result = parse_time(base + "Z")
    assert result.tzinfo is UTC


@pytest.mark.parametrize(
    "text",
    ["", "2", "232", "23:2", "23:20:5", "24:00:00x", "23:20:50Q", "23:20:50+1",
     "23-20-50", "x23:20", "23:20:50+01:0", "23:20:50.", "TT23"],
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


def test_parse_time_invalid_message():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 time format: 'abc'"):
        parse_time("abc")


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("25:00:00")


def test_build_time_regexps():
    pats = build_time_regexps()
    assert len(pats) == 5
    assert pats is build_time_regexps()
    assert len(build_time_regexps()) == 5
    samples = ["23:20:50.5", "232050.5", "23:20.5", "2320.5", "23.5"]
    for pat, sample in zip(pats, samples):
        assert pat.match(sample)
        assert pat.match("T" + sample + "Z")
        assert not pat.match(sample + "x")
        assert not pat.match("x" + sample)


# ---------------------------------------------------------------------------
# parse_datetime
# ---------------------------------------------------------------------------


def test_parse_datetime():
    assert parse_datetime("1985-04-12T23:20:50") == datetime(1985, 4, 12, 23, 20, 50)
    assert parse_datetime("19850412T232050") == datetime(1985, 4, 12, 23, 20, 50)
    assert parse_datetime("1985-102T23:20") == datetime(1985, 4, 12, 23, 20)
    assert parse_datetime("1985-W15-5T23") == datetime(1985, 4, 12, 23)
    dt = parse_datetime("1985-04-12T23:20:50.5Z")
    assert dt == datetime(1985, 4, 12, 23, 20, 50, 500000, tzinfo=UTC)
    assert dt.tzinfo is UTC
    dt = parse_datetime("1985-04-12T23:20:50+04:00")
    assert dt.utcoffset() == timedelta(hours=4)
    assert dt.replace(tzinfo=None) == datetime(1985, 4, 12, 23, 20, 50)


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 23:20:50", "1985-04-12T23T20",
                                  "T", ""])
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


def test_parse_datetime_missing_t_message():
    with pytest.raises(ISO8601Error, match="time designator 'T' missing"):
        parse_datetime("19850412")


# ---------------------------------------------------------------------------
# time zones
# ---------------------------------------------------------------------------


def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+01:00")
    assert tz.utcoffset(None) == timedelta(hours=1)
    assert tz.tzname(None) == "+01:00"
    tz = parse_tzinfo("-0130")
    assert tz.utcoffset(None) == -timedelta(hours=1, minutes=30)
    tz = parse_tzinfo("-01")
    assert tz.utcoffset(None) == -timedelta(hours=1)
    tz = parse_tzinfo("+05")
    assert tz.utcoffset(None) == timedelta(hours=5)
    assert tz.dst(None) == timedelta(0)


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    assert build_tzinfo("Z", "-", 5, 30) is UTC
    tz = build_tzinfo("foo")
    assert tz.utcoffset(None) == timedelta(0)
    assert tz.tzname(None) == "foo"
    tz = build_tzinfo("foo", "-", 2, 30)
    assert tz.utcoffset(None) == -timedelta(hours=2, minutes=30)
    tz = build_tzinfo("foo", "+", 2, 30)
    assert tz.utcoffset(None) == timedelta(hours=2, minutes=30)
    tz = build_tzinfo("foo", tzhour=3)
    assert tz.utcoffset(None) == timedelta(hours=3)
    tz = build_tzinfo("foo", tzmin=3)
    assert tz.utcoffset(None) == timedelta(minutes=3)


class _ZeroOffsetWithDst(tzinfo):
    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return timedelta(hours=1)


class _NoneOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def test_tz_isoformat():
    assert tz_isoformat(datetime(2000, 1, 1)) == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=_NoneOffset())) == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC)) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC), "%z") == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0))) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=_ZeroOffsetWithDst())) == "+00:00"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 45, "x"))
    assert tz_isoformat(dt) == "+05:45"
    assert tz_isoformat(dt, "%Z") == "+05:45"
    assert tz_isoformat(dt, "%z") == "+0545"
    assert tz_isoformat(dt, "%h") == "+05"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-5, -45, "x"))
    assert tz_isoformat(dt) == "-05:45"
    assert tz_isoformat(dt, "%z") == "-0545"
    assert tz_isoformat(dt, "%h") == "-05"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(0, -1, "x"))
    assert tz_isoformat(dt) == "-00:01"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(0, 1, "x"))
    assert tz_isoformat(dt) == "+00:01"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(23, 59, "x"))
    assert tz_isoformat(dt) == "+23:59"


def test_tz_isoformat_large_offsets():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(99, 59, "x"))
    assert tz_isoformat(dt) == "+99:59"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-99, 0, "x"))
    assert tz_isoformat(dt, "%h") == "-99"
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(100, 0, "x")))
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(-100, 0, "x")))


def test_tz_isoformat_unknown_format():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x"))
    with pytest.raises(ValueError, match='unknown format string "%Q"'):
        tz_isoformat(dt, "%Q")
    # naive and UTC never reach the format check
    assert tz_isoformat(datetime(2000, 1, 1), "%Q") == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC), "%Q") == "Z"


def test_utc():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert tzmod.ZERO == timedelta(0)
    assert tzmod._Utc() is UTC


def test_fixed_offset():
    tz = FixedOffset()
    assert tz.utcoffset(None) == timedelta(0)
    assert tz.tzname(None) == "UTC"
    assert tz.dst(None) == timedelta(0)
    tz = FixedOffset(-3, -15, "abc")
    assert tz.utcoffset(None) == -timedelta(hours=3, minutes=15)
    assert tz.tzname(None) == "abc"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedOffset 'abc'>"
    assert FixedOffset(offset_minutes=90).utcoffset(None) == timedelta(hours=1, minutes=30)
    assert FixedOffset(2).utcoffset(None) == timedelta(hours=2)


def test_local_constants():
    assert tzmod.STDOFFSET == timedelta(seconds=-pytime.timezone)
    if pytime.daylight:
        assert tzmod.DSTOFFSET == timedelta(seconds=-pytime.altzone)
    else:
        assert tzmod.DSTOFFSET == tzmod.STDOFFSET
    assert tzmod.DSTDIFF == tzmod.DSTOFFSET - tzmod.STDOFFSET


class _FakeTime:
    def __init__(self, isdst):
        self.isdst = isdst
        self.mktime_args = None
        self.tzname = ("STDNAME", "DSTNAME")

    def mktime(self, tt):
        self.mktime_args = tt
        return 12345.0

    def localtime(self, stamp):
        assert stamp == 12345.0

        class _St:
            tm_isdst = self.isdst

        return _St()


@pytest.fixture
def fake_offsets(monkeypatch):
    monkeypatch.setattr(tzmod, "STDOFFSET", timedelta(hours=1))
    monkeypatch.setattr(tzmod, "DSTOFFSET", timedelta(hours=2))
    monkeypatch.setattr(tzmod, "DSTDIFF", timedelta(hours=1))


@pytest.mark.parametrize("isdst", [1, 2])
def test_local_timezone_dst(monkeypatch, fake_offsets, isdst):
    fake = _FakeTime(isdst)
    monkeypatch.setattr(tzmod, "time", fake)
    dt = datetime(2000, 7, 1, 12, 30, 45)
    assert LOCAL.utcoffset(dt) == timedelta(hours=2)
    assert LOCAL.dst(dt) == timedelta(hours=1)
    assert LOCAL.tzname(dt) == "DSTNAME"
    assert LOCAL._isdst(dt) is True
    assert fake.mktime_args == (2000, 7, 1, 12, 30, 45, dt.weekday(), 0, -1)


@pytest.mark.parametrize("isdst", [0, -1])
def test_local_timezone_std(monkeypatch, fake_offsets, isdst):
    fake = _FakeTime(isdst)
    monkeypatch.setattr(tzmod, "time", fake)
    dt = datetime(2000, 1, 2, 3, 4, 5)
    assert LOCAL.utcoffset(dt) == timedelta(hours=1)
    assert LOCAL.dst(dt) == timedelta(0)
    assert LOCAL.tzname(dt) == "STDNAME"
    assert LOCAL._isdst(dt) is False
    assert fake.mktime_args == (2000, 1, 2, 3, 4, 5, dt.weekday(), 0, -1)


def test_local_timezone_none():
    with pytest.raises(Exception, match="was None"):
        LOCAL.utcoffset(None)


def test_local_timezone_real():
    dt = datetime(2000, 1, 1, 12)
    off = LOCAL.utcoffset(dt)
    assert off in (tzmod.STDOFFSET, tzmod.DSTOFFSET)
    assert isinstance(LOCAL.tzname(dt), str)


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def test_fquotmod():
    assert fquotmod(Decimal(1), 1, 13) == (0, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-1), 1, 13) == (-1, Decimal(11))
    assert fquotmod(Decimal(-11), 1, 13) == (-1, Decimal(1))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(7), 0, 5) == (1, Decimal(2))
    assert fquotmod(Decimal(-7), 0, 5) == (-2, Decimal(3))
    div, mod = fquotmod(Decimal(13), 1, 13)
    assert isinstance(div, int)


def test_max_days_in_month():
    for m in (1, 3, 5, 7, 8, 10, 12):
        assert max_days_in_month(2001, m) == 31
    for m in (4, 6, 9, 11):
        assert max_days_in_month(2001, m) == 30
    assert max_days_in_month(2001, 2) == 28
    assert max_days_in_month(2004, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(2100, 2) == 28
    assert max_days_in_month(2400, 2) == 29
    assert max_days_in_month(1996, 2) == 29
    assert max_days_in_month(2002, 2) == 28


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def test_duration_init_defaults():
    d = Duration()
    assert d.years == 0
    assert d.months == 0
    assert d.tdelta == timedelta(0)
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)


def test_duration_init_args():
    d = Duration(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.months == Decimal(8)
    assert d.years == Decimal(9)
    d = Duration(months=1.5, years=2.5)
    assert d.months == Decimal("1.5")
    assert d.years == Decimal("2.5")
    assert isinstance(d.months, Decimal)
    assert isinstance(d.years, Decimal)
    d = Duration(months=Decimal("0.1"), years=Decimal("0.2"))
    assert d.months == Decimal("0.1")
    assert d.years == Decimal("0.2")


def test_duration_getattr():
    d = Duration(days=2, seconds=3, microseconds=4)
    assert d.days == 2
    assert d.seconds == 3
    assert d.microseconds == 4
    assert d.total_seconds() == 2 * 86400 + 3 + 4e-6
    with pytest.raises(AttributeError):
        d.nonexistent


def test_duration_pickle():
    d = Duration(days=1, hours=2, months=3, years=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d
    assert d2.years == 4
    assert d2.months == 3
    assert d2.tdelta == timedelta(days=1, hours=2)
    assert d.__getstate__() == {"months": Decimal(3), "years": Decimal(4),
                                "tdelta": timedelta(days=1, hours=2)}


def test_duration_str():
    assert str(Duration()) == "0:00:00"
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    assert str(Duration(years=1)) == "1 years, 0:00:00"
    assert str(Duration(years=2)) == "2 years, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(months=-3)) == "-3 month, 0:00:00"
    assert str(Duration(years=3, months=5, days=1, hours=2)) == "3 years, 5 months, 1 day, 2:00:00"


def test_duration_repr():
    d = Duration(days=1, seconds=2, microseconds=3, years=4, months=5)
    assert repr(d) == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash():
    d = Duration(days=1, months=2, years=3)
    assert hash(d) == hash((timedelta(days=1), Decimal(2), Decimal(3)))
    assert hash(d) == hash(Duration(days=1, months=2, years=3))
    assert len({d, Duration(days=1, months=2, years=3)}) == 1


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3, hours=4)
    assert isinstance(d, Duration)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == -timedelta(days=3, hours=4)


def test_duration_add_duration():
    d = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, hours=6)
    assert isinstance(d, Duration)
    assert d.years == 5
    assert d.months == 7
    assert d.tdelta == timedelta(days=3, hours=6)


def test_duration_add_timedelta():
    d = Duration(years=1, months=2, days=3) + timedelta(hours=1)
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=3, hours=1)
    d = timedelta(hours=1) + Duration(years=1, months=2, days=3)
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=3, hours=1)


@pytest.mark.parametrize(
    "start,dur,expected",
    [
        (date(2000, 1, 15), Duration(months=1), date(2000, 2, 15)),
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 30), Duration(months=1), date(2000, 2, 29)),
        (date(2000, 1, 29), Duration(months=1), date(2000, 2, 29)),
        (date(2000, 1, 28), Duration(months=1), date(2000, 2, 28)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 1, 15), Duration(months=12), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=25), date(2002, 2, 15)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 1, 15), Duration(days=20), date(2000, 2, 4)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (datetime(2000, 1, 31, 12), Duration(years=1, months=1, days=1, hours=13),
         datetime(2001, 3, 2, 1)),
        (date(2000, 1, 15), Duration(years=-1, months=-13), date(1997, 12, 15)),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


@pytest.mark.parametrize(
    "start,dur,expected",
    [
        (date(2000, 2, 15), Duration(months=1), date(2000, 1, 15)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 3, 30), Duration(months=1), date(2000, 2, 29)),
        (date(2000, 3, 29), Duration(months=1), date(2000, 2, 29)),
        (date(2000, 3, 28), Duration(months=1), date(2000, 2, 28)),
        (date(2001, 1, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=12), date(1999, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(1998, 12, 15)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2000, 3, 31), Duration(months=1, days=1), date(2000, 2, 28)),
        (datetime(2000, 3, 31, 12), Duration(years=1, months=1, hours=13),
         datetime(1999, 2, 27, 23)),
        (date(2000, 1, 15), Duration(days=20), date(1999, 12, 26)),
    ],
)
def test_duration_rsub_date(start, dur, expected):
    assert start - dur == expected


@pytest.mark.parametrize(
    "dur",
    [Duration(years=1.5), Duration(months=0.5), Duration(years=1, months=1.5),
     Duration(years=1.5, months=1)],
)
def test_duration_fractional_with_dates(dur):
    with pytest.raises(ValueError, match="fractional years or months"):
        date(2000, 1, 1) + dur
    with pytest.raises(ValueError, match="fractional years or months"):
        dur + date(2000, 1, 1)
    with pytest.raises(ValueError, match="fractional years or months"):
        date(2000, 1, 1) - dur


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) + "x"
    with pytest.raises(TypeError):
        "x" + Duration(days=1)
    assert Duration().__add__("x") is NotImplemented
    with pytest.raises(TypeError):
        Duration(days=1) + 1


def test_duration_mul():
    d = Duration(years=1, months=2, days=3) * 3
    assert isinstance(d, Duration)
    assert d.years == 3
    assert d.months == 6
    assert d.tdelta == timedelta(days=9)
    d = 2 * Duration(years=1, months=2, days=3)
    assert d.years == 2
    assert d.months == 4
    assert d.tdelta == timedelta(days=6)
    d = Duration(years=1, months=2, days=3) * -1
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == timedelta(days=-3)
    with pytest.raises(TypeError):
        Duration(days=1) * 1.5
    with pytest.raises(TypeError):
        Duration(days=1) * "x"
    assert Duration().__mul__(1.5) is NotImplemented


def test_duration_sub_duration():
    d = Duration(years=5, months=7, days=3) - Duration(years=1, months=2, hours=1)
    assert isinstance(d, Duration)
    assert d.years == 4
    assert d.months == 5
    assert d.tdelta == timedelta(days=3) - timedelta(hours=1)


def test_duration_sub_timedelta():
    d = Duration(years=1, months=2, days=3) - timedelta(days=1)
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=2)


def test_duration_rsub_timedelta():
    d = timedelta(days=3) - Duration(years=1, months=2, days=1)
    assert isinstance(d, Duration)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == timedelta(days=2)


def test_duration_sub_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) - "x"
    assert Duration().__sub__("x") is NotImplemented
    with pytest.raises(TypeError):
        "x" - Duration(days=1)
    assert Duration().__rsub__("x") is NotImplemented
    with pytest.raises(TypeError):
        1 - Duration(days=1)


def test_duration_eq():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=2) == Duration(years=1, months=12)
    assert not (Duration(years=1) == Duration(months=13))
    assert not (Duration(years=1) == Duration(years=1, days=1))
    assert Duration(days=1, hours=2) == Duration(hours=26)
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert not (Duration(days=1) == timedelta(days=2))
    assert not (Duration(years=1) == timedelta(0))
    assert not (Duration(months=1) == timedelta(0))
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert not (Duration(months=1, days=1) == timedelta(days=1))
    assert Duration(years=0, months=0) == timedelta(0)
    assert Duration().__eq__(Duration()) is True
    assert Duration().__eq__(Duration(days=1)) is False
    assert Duration(years=1).__eq__(timedelta(0)) is False


def test_duration_ne():
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1) != Duration(months=13)
    assert Duration(years=1) != Duration(years=1, days=1)
    assert not (Duration(days=1) != Duration(hours=24))
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(years=1) != timedelta(0)
    assert Duration(months=1) != timedelta(0)
    assert Duration(months=1, days=1) != timedelta(days=1)
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert Duration().__ne__(Duration()) is False
    assert Duration().__ne__(Duration(days=1)) is True
    assert Duration(years=1).__ne__(timedelta(0)) is True
    assert Duration(years=1).__ne__("x") is True


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(start=date(2000, 4, 1)) == timedelta(days=30)
    assert d.totimedelta(end=date(2000, 4, 1)) == timedelta(days=31)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(date(2000, 4, 1)) == timedelta(days=30)
    assert Duration(days=3).totimedelta(end=datetime(2000, 1, 1)) == timedelta(days=3)
    with pytest.raises(ValueError, match="start or end required"):
        d.totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 1))


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("P1D", timedelta(days=1)),
        ("P3W", timedelta(weeks=3)),
        ("P1W1D", timedelta(days=8)),
        ("PT1H", timedelta(hours=1)),
        ("PT1M", timedelta(minutes=1)),
        ("PT1S", timedelta(seconds=1)),
        ("PT36H", timedelta(hours=36)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("PT0.5S", timedelta(seconds=0.5)),
        ("PT0,5S", timedelta(seconds=0.5)),
        ("P1.5D", timedelta(days=1, hours=12)),
        ("P1,5D", timedelta(days=1, hours=12)),
        ("PT1.5H", timedelta(hours=1, minutes=30)),
        ("PT1.5M", timedelta(minutes=1, seconds=30)),
        ("P0.5W", timedelta(days=3.5)),
        ("+P1D", timedelta(days=1)),
        ("-P1D", timedelta(days=-1)),
        ("-PT1H30M", -timedelta(hours=1, minutes=30)),
        ("-P2W", timedelta(weeks=-2)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("P0Y0M1D", timedelta(days=1)),
    ],
)
def test_parse_duration_timedelta(text, expected):
    result = parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


@pytest.mark.parametrize(
    "text,years,months,tdelta",
    [
        ("P1Y", 1, 0, timedelta(0)),
        ("P1M", 0, 1, timedelta(0)),
        ("P1Y2M", 1, 2, timedelta(0)),
        ("P1Y2M3DT4H5M6S", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("P1Y2M3W", 1, 2, timedelta(weeks=3)),
        ("P1.5Y", Decimal("1.5"), 0, timedelta(0)),
        ("P1,5Y", Decimal("1.5"), 0, timedelta(0)),
        ("P2.5M", 0, Decimal("2.5"), timedelta(0)),
        ("P2,5M", 0, Decimal("2.5"), timedelta(0)),
        ("+P1Y", 1, 0, timedelta(0)),
        ("-P1Y", -1, 0, timedelta(0)),
        ("-P1Y2M3DT4H", -1, -2, -timedelta(days=3, hours=4)),
        ("P18Y9M4DT11H9M8S", 18, 9, timedelta(days=4, hours=11, minutes=9, seconds=8)),
        ("P1MT1M", 0, 1, timedelta(minutes=1)),
    ],
)
def test_parse_duration_duration(text, years, months, tdelta):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result.years == years
    assert result.months == months
    assert result.tdelta == tdelta
    assert isinstance(result.years, Decimal)
    assert isinstance(result.months, Decimal)


def test_parse_duration_not_as_timedelta():
    result = parse_duration("P1DT2H", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result.years == 0
    assert result.months == 0
    assert result.tdelta == timedelta(days=1, hours=2)
    result = parse_duration("-P1D", False)
    assert isinstance(result, Duration)
    assert result.tdelta == timedelta(days=-1)
    result = parse_duration("P3W", False)
    assert isinstance(result, Duration)
    assert result.tdelta == timedelta(weeks=3)


@pytest.mark.parametrize(
    "text,years,months,tdelta",
    [
        ("P0001-02-03T04:05:06", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("P00010203T040506", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("P0018-09-04T11:09:08", 18, 9, timedelta(days=4, hours=11, minutes=9, seconds=8)),
        ("P0001-01-01T00:00:00.5", 1, 1, timedelta(days=1, microseconds=500000)),
        ("P0001-032T01:02:03", 1, 2, timedelta(days=1, hours=1, minutes=2, seconds=3)),
    ],
)
def test_parse_duration_alternative(text, years, months, tdelta):
    for result in (parse_duration(text), parse_duration(text, False)):
        assert isinstance(result, Duration)
        assert result.years == years
        assert result.months == months
        assert result.tdelta == tdelta


@pytest.mark.parametrize(
    "text", ["", "1D", "P", "PX", "P1", "P1H", "PT1D", "P1S", "XP1D", "P1D2", "--P1D",
             "P-1D", "1Y2M", "P1.D"],
)
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_invalid_message():
    with pytest.raises(ISO8601Error, match="Unable to parse duration string '1D'"):
        parse_duration("1D")


def test_parse_duration_type_error():
    with pytest.raises(TypeError, match="Expecting a string"):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


def test_parse_duration_alternative_invalid_values():
    with pytest.raises(ValueError):
        parse_duration("P0000-00-00T00:00:00")


# ---------------------------------------------------------------------------
# duration formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (timedelta(0), "P0D"),
        (timedelta(days=1), "P1D"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
        (timedelta(days=1, seconds=1), "P1DT1S"),
        (timedelta(hours=1, seconds=1), "PT1H1S"),
        (timedelta(seconds=1.5), "PT1.5S"),
        (timedelta(seconds=0.5), "PT0.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=59, microseconds=999999), "PT59.999999S"),
        (timedelta(seconds=10, microseconds=120000), "PT10.12S"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(weeks=2), "P14D"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-1), "-PT1H"),
        (-timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5), "-P1DT2H3M4.000005S"),
        (timedelta(seconds=60), "PT1M"),
        (timedelta(seconds=3600), "PT1H"),
        (timedelta(seconds=86400), "P1D"),
        (Duration(), "P0D"),
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=1, months=2, days=3), "P1Y2M3D"),
        (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
        (Duration(years=1, minutes=5), "P1YT5M"),
        (Duration(years=-1), "-P1Y"),
        (Duration(months=-1), "-P1M"),
        (Duration(days=-1), "-P1D"),
        (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
        (Duration(months=Decimal("2.5")), "P2.5M"),
    ],
)
def test_duration_isoformat_default(value, expected):
    assert duration_isoformat(value) == expected
    assert duration_isoformat(value, D_DEFAULT) == expected


def test_duration_isoformat_roundtrip():
    for text in ["P1Y2M3DT4H5M6S", "P1D", "PT1H", "PT0.5S", "-P1D", "-P1Y2M", "P1Y",
                 "P1DT12H", "PT1M1S"]:
        assert duration_isoformat(parse_duration(text)) == text


def test_duration_isoformat_week():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=10), D_WEEK) == "P1W"
    assert duration_isoformat(timedelta(days=6), D_WEEK) == "P0W"
    assert duration_isoformat(timedelta(days=-14), D_WEEK) == "-P2W"
    assert duration_isoformat(Duration(weeks=2), D_WEEK) == "P2W"


def test_duration_isoformat_alternative():
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d, D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(d, D_ALT_BAS) == "P00010203T040506"
    td = timedelta(days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(td, "P%dT%H:%M:%S") == "P03T04:05:06"
    assert duration_isoformat(td, "%%%f%x") == "%000000%x"


@pytest.mark.parametrize(
    "value,fmt,expected",
    [
        (timedelta(hours=5, minutes=30, seconds=15), "%H", "05"),
        (timedelta(hours=5, minutes=30, seconds=15), "%M", "30"),
        (timedelta(hours=5, minutes=30, seconds=15), "%S", "15"),
        (timedelta(hours=23, minutes=59, seconds=59), "%H:%M:%S", "23:59:59"),
        (timedelta(seconds=5399), "%H:%M:%S", "01:29:59"),
        (timedelta(days=15), "%d", "15"),
        (timedelta(days=15), "%W", "02"),
        (timedelta(days=13), "%W", "01"),
        (timedelta(days=-14), "%W", "02"),
        (timedelta(microseconds=42), "%f", "000042"),
        (Duration(years=3, months=4), "%Y-%m", "0003-04"),
        (Duration(years=1985), "%C", "19"),
        (timedelta(0), "%%", "%"),
        (timedelta(0), "%j%w", "%j%w"),
    ],
)
def test_strftime_duration(value, fmt, expected):
    assert strftime(value, fmt) == expected


def test_strftime_duration_yeardigits():
    d = Duration(years=3)
    assert strftime(d, "%Y", 6) == "+000003"
    assert strftime(d, "%Y", 4) == "0003"
    assert strftime(d, "%Y") == "0003"
    assert strftime(Duration(years=1985), "%C", 6) == "+0019"
    assert strftime(Duration(years=1985), "%C") == "19"


def test_version():
    assert isinstance(isodate.__version__, str)


def test_all_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name)


# ---------------------------------------------------------------------------
# exact messages and edge cases
# ---------------------------------------------------------------------------


def _message(excinfo):
    return str(excinfo.value)


def test_error_messages_exact():
    with pytest.raises(ISO8601Error) as exc:
        parse_date("abc")
    assert _message(exc) == "Unrecognised ISO 8601 date format: 'abc'"
    with pytest.raises(ISO8601Error) as exc:
        parse_time("abc")
    assert _message(exc) == "Unrecognised ISO 8601 time format: 'abc'"
    with pytest.raises(ISO8601Error) as exc:
        parse_datetime("abc")
    assert _message(exc) == (
        "ISO 8601 time designator 'T' missing. Unable to parse datetime string 'abc'"
    )
    with pytest.raises(ISO8601Error) as exc:
        parse_duration("abc")
    assert _message(exc) == "Unable to parse duration string 'abc'"
    with pytest.raises(TypeError) as exc:
        parse_duration(5)
    assert _message(exc) == "Expecting a string 5"
    with pytest.raises(ValueError) as exc:
        Duration(months=1).totimedelta()
    assert _message(exc) == "start or end required"
    with pytest.raises(ValueError) as exc:
        Duration(months=1).totimedelta(date(2000, 1, 1), date(2000, 1, 1))
    assert _message(exc) == "only start or end allowed"
    with pytest.raises(ValueError) as exc:
        date(2000, 1, 1) + Duration(months=0.5)
    assert _message(exc) == "fractional years or months not supported for date calculations"
    with pytest.raises(ValueError) as exc:
        date(2000, 1, 1) - Duration(months=0.5)
    assert _message(exc) == "fractional years or months not supported for date calculations"
    with pytest.raises(OverflowError) as exc:
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(100, 0, "x")))
    assert _message(exc) == "can not handle differences > 99 hours"
    with pytest.raises(ValueError) as exc:
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%Q")
    assert _message(exc) == 'unknown format string "%Q"'
    with pytest.raises(Exception) as exc:
        LOCAL.dst(None)
    assert _message(exc) == "datetime object dt was None!"


def test_duration_ne_combined_years_months():
    assert not (Duration(years=1, months=1) != Duration(months=13))
    assert not (Duration(months=13) != Duration(years=1, months=1))
    assert not (Duration(years=2) != Duration(years=1, months=12))
    assert not (Duration(months=24) != Duration(years=2))
    assert Duration(years=1, months=1) != Duration(years=1, months=2)


def test_duration_rsub_into_31_day_month():
    assert date(2000, 4, 30) - Duration(months=1) == date(2000, 3, 30)
    assert date(2000, 6, 30) - Duration(months=1) == date(2000, 5, 30)
    assert date(2000, 2, 29) + Duration(months=1) == date(2000, 3, 29)


def test_parse_time_fraction_scaling_edge():
    assert parse_time("12:30.99999999") == time(12, 30, 59, 999999)
    assert parse_time("12:30:59.9999999") == time(12, 30, 59, 999999)


def test_private_strf_defaults():
    from isodate.isostrf import _strfdt, _strfduration

    assert _strfduration(Duration(years=3), "%Y") == "0003"
    assert _strfdt(date(1985, 4, 12), "%Y") == "1985"


class _DtDependentTz(tzinfo):
    """Offset depends on dt; answers None when dt is None."""

    def utcoffset(self, dt):
        if dt is None:
            return None
        return timedelta(hours=dt.month - 1)

    def dst(self, dt):
        if dt is None:
            return None
        return timedelta(0)


class _VanishingTz(tzinfo):
    """Reports an offset on the first two calls, then None."""

    def __init__(self):
        self.calls = 0

    def utcoffset(self, dt):
        self.calls += 1
        if self.calls <= 2:
            return timedelta(hours=1)
        return None

    def dst(self, dt):
        return timedelta(0)


def test_tz_isoformat_passes_dt_to_tzinfo():
    tz = _DtDependentTz()
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=tz)) == "Z"
    assert tz_isoformat(datetime(2000, 3, 1, tzinfo=tz)) == "+02:00"
    assert tz_isoformat(datetime(2000, 3, 1, tzinfo=tz), "%z") == "+0200"


def test_tz_isoformat_offset_disappears():
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=_VanishingTz())) == ""
