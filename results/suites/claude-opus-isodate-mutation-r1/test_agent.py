"""Test suite for isodate."""

import os
import pickle
import subprocess
import sys
import types
from datetime import date, datetime, time, timedelta, tzinfo
from decimal import Decimal

import pytest

import isodate
from isodate import (
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
    D_ALT_BAS,
    D_ALT_EXT,
    D_DEFAULT,
    D_WEEK,
    DT_BAS_COMPLETE,
    DT_BAS_ORD_COMPLETE,
    DT_BAS_WEEK_COMPLETE,
    DT_EXT_COMPLETE,
    DT_EXT_ORD_COMPLETE,
    DT_EXT_WEEK_COMPLETE,
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
from isodate.duration import fquotmod, max_days_in_month
from isodate.isodates import build_date_regexps
from isodate.isotime import build_time_regexps
from isodate.isotzinfo import build_tzinfo
from isodate.tzinfo import LOCAL, ZERO, LocalTimezone, Utc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- dates


@pytest.mark.parametrize(
    "text, expected",
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
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W01-4", date(2009, 1, 1)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W01", date(2010, 1, 4)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2015-W01-1", date(2014, 12, 29)),
        ("2016-W01-1", date(2016, 1, 4)),
        ("1985-00", date(1985, 1, 1)),
    ],
)
def test_parse_date(text, expected):
    result = parse_date(text)
    assert result == expected
    assert type(result) is date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+001985-04-12", date(1985, 4, 12)),
        ("+00198504", date(1985, 4, 1)),
        ("+0019850412", date(1985, 4, 12)),
        ("+001985-W15-5", date(1985, 4, 12)),
        ("+001985W155", date(1985, 4, 12)),
        ("+001985-102", date(1985, 4, 12)),
        ("+001985102", date(1985, 4, 12)),
        ("+001985-W15", date(1985, 4, 8)),
        ("+001985W15", date(1985, 4, 8)),
        ("+001985-04", date(1985, 4, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+0019", date(1901, 1, 1)),
    ],
)
def test_parse_date_expanded_6(text, expected):
    assert parse_date(text, 6) == expected
    assert parse_date(text, 6, expanded=False) == expected


def test_parse_date_expanded_4():
    assert parse_date("+1985-04-12", 4, True) == date(1985, 4, 12)
    assert parse_date("+19", 4, True) == date(1901, 1, 1)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", 4, True)


def test_parse_date_negative_sign_rejected():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)
    with pytest.raises(ValueError):
        parse_date("-0019", 6)
    with pytest.raises(ValueError):
        parse_date("-1985", 4, True)
    with pytest.raises(ValueError):
        parse_date("-1985-W01", 4, True)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=15) == date(1985, 4, 15)
    assert parse_date("1985-04", defaultmonth=7, defaultday=15) == date(1985, 4, 15)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    assert parse_date("1985-00", defaultmonth=5) == date(1985, 5, 1)
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-102", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-W15", defaultmonth=6, defaultday=15) == date(1985, 4, 8)


@pytest.mark.parametrize(
    "text", ["", "abc", "1985-4-12", "85-04-12", "1985-04-12T", "1985/04/12", "1985-W1"]
)
def test_parse_date_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


def test_parse_date_out_of_range():
    with pytest.raises(ValueError):
        parse_date("1985-13-01")
    with pytest.raises(ValueError):
        parse_date("1985-02-30")


def test_build_date_regexps_cache():
    a = build_date_regexps()
    assert a is build_date_regexps(4, False)
    assert len(a) == 12
    b = build_date_regexps(6)
    assert b is build_date_regexps(6, False)
    assert b is build_date_regexps(6, True)
    assert b is not a
    assert build_date_regexps(4, True) is not a


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_EXT_COMPLETE, "1985-04-12"),
        (DATE_BAS_COMPLETE, "19850412"),
        (DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (DATE_BAS_WEEK_COMPLETE, "1985W155"),
        (DATE_EXT_ORD_COMPLETE, "1985-102"),
        (DATE_BAS_ORD_COMPLETE, "1985102"),
        (DATE_EXT_WEEK, "1985-W15"),
        (DATE_BAS_WEEK, "1985W15"),
        (DATE_EXT_MONTH, "1985-04"),
        (DATE_BAS_MONTH, "198504"),
        (DATE_YEAR, "1985"),
        (DATE_CENTURY, "19"),
    ],
)
def test_date_isoformat(fmt, expected):
    assert date_isoformat(date(1985, 4, 12), fmt) == expected
    assert parse_date(expected) <= date(1985, 4, 12)


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_EXT_COMPLETE, "+001985-04-12"),
        (DATE_BAS_COMPLETE, "+0019850412"),
        (DATE_EXT_WEEK_COMPLETE, "+001985-W15-5"),
        (DATE_YEAR, "+001985"),
        (DATE_CENTURY, "+0019"),
    ],
)
def test_date_isoformat_expanded(fmt, expected):
    assert date_isoformat(date(1985, 4, 12), fmt, 6) == expected


def test_date_isoformat_default():
    assert date_isoformat(date(2001, 2, 3)) == "2001-02-03"
    assert date_isoformat(date(2000, 12, 31), DATE_EXT_ORD_COMPLETE) == "2000-366"
    assert date_isoformat(date(2001, 12, 31), DATE_EXT_ORD_COMPLETE) == "2001-365"
    assert date_isoformat(date(2001, 1, 1), DATE_EXT_ORD_COMPLETE) == "2001-001"
    # Sunday is weekday 7
    assert date_isoformat(date(2010, 1, 3), DATE_EXT_WEEK_COMPLETE) == "2010-W53-7"  # %Y is the calendar year
    assert date_isoformat(date(2010, 1, 4), DATE_EXT_WEEK_COMPLETE) == "2010-W01-1"
    assert date_isoformat(date(1999, 1, 1), DATE_CENTURY) == "19"
    assert date_isoformat(date(2000, 1, 1), DATE_CENTURY) == "20"
    assert date_isoformat(date(5, 1, 1), DATE_YEAR) == "0005"


# ---------------------------------------------------------------- times


@pytest.mark.parametrize(
    "text, expected",
    [
        ("232050", time(23, 20, 50)),
        ("23:20:50", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("T232050", time(23, 20, 50)),
        ("2320", time(23, 20)),
        ("23:20", time(23, 20)),
        ("23", time(23)),
        ("T23", time(23)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("232050,5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.1234567", time(23, 20, 50, 123456)),
        ("23:20:50.000001", time(23, 20, 50, 1)),
        ("23:20.5", time(23, 20, 30)),
        ("2320,8", time(23, 20, 48)),
        ("00:00.0001", time(0, 0, 0, 6000)),
        ("00:01.5025", time(0, 1, 30, 150000)),
        ("23.3", time(23, 18)),
        ("23,3", time(23, 18)),
        ("23.5", time(23, 30)),
        ("12.5025", time(12, 30, 9)),
        ("00.0001", time(0, 0, 0, 360000)),
        ("01.01", time(1, 0, 36)),
        ("0920", time(9, 20)),
        ("0920,5", time(9, 20, 30)),
        ("092005", time(9, 20, 5)),
        ("09:20", time(9, 20)),
        ("09", time(9)),
        ("00:00:00", time(0)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "text, expected, offset, name",
    [
        ("152746+0100", time(15, 27, 46), timedelta(hours=1), "+0100"),
        ("15:27:46+01:00", time(15, 27, 46), timedelta(hours=1), "+01:00"),
        ("15:27:46+01", time(15, 27, 46), timedelta(hours=1), "+01"),
        ("15:27:46-05:00", time(15, 27, 46), timedelta(hours=-5), "-05:00"),
        ("15:27:46-0530", time(15, 27, 46), timedelta(hours=-5, minutes=-30), "-0530"),
        ("15:27:46+05:30", time(15, 27, 46), timedelta(hours=5, minutes=30), "+05:30"),
        ("1527+0530", time(15, 27), timedelta(hours=5, minutes=30), "+0530"),
        ("15-0230", time(15), timedelta(hours=-2, minutes=-30), "-0230"),
        ("15:27:46.5+01:00", time(15, 27, 46, 500000), timedelta(hours=1), "+01:00"),
    ],
)
def test_parse_time_tz(text, expected, offset, name):
    result = parse_time(text)
    assert result.replace(tzinfo=None) == expected
    assert result.utcoffset() == offset
    assert result.tzinfo.tzname(None) == name
    assert result.dst() == ZERO


def test_parse_time_utc():
    result = parse_time("15:27:46Z")
    assert result.tzinfo is UTC
    assert result.replace(tzinfo=None) == time(15, 27, 46)
    assert parse_time("152746Z").tzinfo is UTC
    assert parse_time("15Z").tzinfo is UTC
    assert parse_time("15.5Z") == time(15, 30, tzinfo=UTC)


@pytest.mark.parametrize("text", ["", "abc", "1", "1:20", "23:2", "23:20:5", "23:20:50X", "2320505"])
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("25:00")
    with pytest.raises(ValueError):
        parse_time("23:60")


def test_build_time_regexps_cache():
    a = build_time_regexps()
    assert a is build_time_regexps()
    assert len(a) == 5


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (time(23, 20, 50), TIME_EXT_COMPLETE, "23:20:50"),
        (time(23, 20, 50), TIME_BAS_COMPLETE, "232050"),
        (time(23, 20, 50), TIME_EXT_MINUTE, "23:20"),
        (time(23, 20, 50), TIME_BAS_MINUTE, "2320"),
        (time(23, 20, 50), TIME_HOUR, "23"),
        (time(3, 4, 5), TIME_EXT_COMPLETE + TZ_EXT, "03:04:05"),
        (time(3, 4, 5, tzinfo=UTC), TIME_EXT_COMPLETE + TZ_EXT, "03:04:05Z"),
        (time(3, 4, 5, tzinfo=UTC), TIME_BAS_COMPLETE + TZ_BAS, "030405Z"),
        (time(3, 4, 5, tzinfo=FixedOffset(1, 30)), TIME_EXT_COMPLETE + TZ_EXT, "03:04:05+01:30"),
        (time(3, 4, 5, tzinfo=FixedOffset(1, 30)), TIME_BAS_COMPLETE + TZ_BAS, "030405+0130"),
        (time(3, 4, 5, tzinfo=FixedOffset(1, 30)), TIME_HOUR + TZ_HOUR, "03+01"),
        (time(3, 4, 5, tzinfo=FixedOffset(-5, 0)), TIME_EXT_COMPLETE + TZ_EXT, "03:04:05-05:00"),
        (time(3, 4, 5, tzinfo=FixedOffset(-1, -30)), TIME_BAS_COMPLETE + TZ_BAS, "030405-0130"),
        (time(3, 4, 5, tzinfo=FixedOffset(-1, -30)), TZ_HOUR, "-01"),
        (time(3, 4, 5, 123), "%H:%M:%S.%f", "03:04:05.000123"),
        (time(3, 4, 5), "%%H", "%H"),
        (time(3, 4, 5), "%x %H", "%x 03"),
    ],
)
def test_time_isoformat(value, fmt, expected):
    assert time_isoformat(value, fmt) == expected


def test_time_isoformat_default():
    assert time_isoformat(time(1, 2, 3)) == "01:02:03"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(2, 0))) == "01:02:03+02:00"


# ---------------------------------------------------------------- tz


def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+01:00")
    assert tz.utcoffset(None) == timedelta(hours=1)
    assert tz.tzname(None) == "+01:00"
    assert parse_tzinfo("-0530").utcoffset(None) == timedelta(hours=-5, minutes=-30)
    assert parse_tzinfo("+05").utcoffset(None) == timedelta(hours=5)
    assert parse_tzinfo("-05").utcoffset(None) == timedelta(hours=-5)
    assert parse_tzinfo("+0545").utcoffset(None) == timedelta(hours=5, minutes=45)


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("+0130", "+", 1, 30)
    assert tz.utcoffset(None) == timedelta(hours=1, minutes=30)
    tz = build_tzinfo("-0130", "-", 1, 30)
    assert tz.utcoffset(None) == timedelta(hours=-1, minutes=-30)
    assert tz.tzname(None) == "-0130"
    tz = build_tzinfo("x")
    assert tz.utcoffset(None) == ZERO


class _DstZeroOffset(tzinfo):
    def utcoffset(self, dt):
        return ZERO

    def dst(self, dt):
        return timedelta(hours=1)


class _NoneOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def test_tz_isoformat():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(2, 15))
    assert tz_isoformat(dt) == "+02:15"
    assert tz_isoformat(dt, "%Z") == "+02:15"
    assert tz_isoformat(dt, "%z") == "+0215"
    assert tz_isoformat(dt, "%h") == "+02"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-2, -15))
    assert tz_isoformat(dt, "%Z") == "-02:15"
    assert tz_isoformat(dt, "%z") == "-0215"
    assert tz_isoformat(dt, "%h") == "-02"
    assert tz_isoformat(datetime(2000, 1, 1)) == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC)) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0))) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=_DstZeroOffset())) == "+00:00"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=_NoneOffset())) == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 1)), "%z") == "+0001"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, -1)), "%Z") == "-00:01"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(23, 59)), "%Z") == "+23:59"


def test_tz_isoformat_bad_format():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(2, 0))
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%Q")


def test_tz_isoformat_large_offsets():
    fake = types.SimpleNamespace(tzinfo=FixedOffset(99, 0))
    assert tz_isoformat(fake, "%Z") == "+99:00"
    fake = types.SimpleNamespace(tzinfo=FixedOffset(100, 0))
    with pytest.raises(OverflowError, match="can not handle differences > 99 hours"):
        tz_isoformat(fake, "%Z")
    fake = types.SimpleNamespace(tzinfo=FixedOffset(-100, 0))
    with pytest.raises(OverflowError):
        tz_isoformat(fake, "%Z")
    fake = types.SimpleNamespace(tzinfo=FixedOffset(30, 0))
    assert tz_isoformat(fake, "%z") == "+3000"
    fake = types.SimpleNamespace(tzinfo=FixedOffset(-30, 0))
    assert tz_isoformat(fake, "%z") == "-3000"


def test_fixed_offset():
    tz = FixedOffset(5, 30, "IST")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "IST"
    assert tz.dst(None) == ZERO
    assert repr(tz) == "<FixedOffset 'IST'>"
    default = FixedOffset()
    assert default.utcoffset(None) == ZERO
    assert default.tzname(None) == "UTC"


def test_utc():
    assert UTC.utcoffset(None) == ZERO
    assert UTC.dst(None) == ZERO
    assert UTC.tzname(None) == "UTC"
    assert isinstance(UTC, Utc)
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert pickle.loads(pickle.dumps(Utc())) is UTC
    assert ZERO == timedelta(0)


def test_local_timezone_consistency():
    dt = datetime(2020, 1, 15, 12)
    assert LOCAL.utcoffset(dt) - LOCAL.dst(dt) == isodate.tzinfo.STDOFFSET
    assert LOCAL.tzname(dt) in __import__("time").tzname
    assert isinstance(LOCAL, LocalTimezone)
    with pytest.raises(Exception, match="dt was None"):
        LOCAL.utcoffset(None)
    with pytest.raises(Exception, match="dt was None"):
        LOCAL.dst(None)


_LOCAL_SCRIPT = r"""
import datetime, time, isodate
from isodate.tzinfo import LOCAL, STDOFFSET, DSTOFFSET, DSTDIFF
w = datetime.datetime(2020, 1, 15, 12)
s = datetime.datetime(2020, 7, 15, 12)
print(LOCAL.utcoffset(w).total_seconds())
print(LOCAL.utcoffset(s).total_seconds())
print(LOCAL.dst(w).total_seconds())
print(LOCAL.dst(s).total_seconds())
print(LOCAL.tzname(w))
print(LOCAL.tzname(s))
print(STDOFFSET.total_seconds(), DSTOFFSET.total_seconds(), DSTDIFF.total_seconds())
print(isodate.tz_isoformat(datetime.datetime(2020, 7, 15, 12, tzinfo=LOCAL)))
print(isodate.tz_isoformat(datetime.datetime(2020, 1, 15, 12, tzinfo=LOCAL)))
print(time.tzname[0])
"""


def _run_local(tz):
    env = dict(os.environ)
    env["TZ"] = tz
    env["PYTHONPATH"] = REPO_ROOT
    out = subprocess.run(
        [sys.executable, "-c", _LOCAL_SCRIPT],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split("\n")


def test_local_timezone_new_york():
    lines = _run_local("America/New_York")
    if lines[9] != "EST":
        pytest.skip("tz database not available")
    assert lines[0] == "-18000.0"
    assert lines[1] == "-14400.0"
    assert lines[2] == "0.0"
    assert lines[3] == "3600.0"
    assert lines[4] == "EST"
    assert lines[5] == "EDT"
    assert lines[6] == "-18000.0 -14400.0 3600.0"
    assert lines[7] == "-04:00"
    assert lines[8] == "-05:00"


def test_local_timezone_utc():
    lines = _run_local("UTC")
    assert lines[0] == "0.0"
    assert lines[1] == "0.0"
    assert lines[2] == "0.0"
    assert lines[3] == "0.0"
    assert lines[6] == "0.0 0.0 0.0"
    assert lines[7] == "Z"


# ---------------------------------------------------------------- datetimes


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
        ("19850412T101530", datetime(1985, 4, 12, 10, 15, 30)),
        ("1985-W15-5T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985102T23", datetime(1985, 4, 12, 23)),
        ("1985-102T23:50:30.5", datetime(1985, 4, 12, 23, 50, 30, 500000)),
    ],
)
def test_parse_datetime_naive(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.tzinfo is None


def test_parse_datetime_tz():
    result = parse_datetime("1985-04-12T10:15:30Z")
    assert result == datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
    assert result.tzinfo is UTC
    result = parse_datetime("19850412T101530+0400")
    assert result.utcoffset() == timedelta(hours=4)
    assert result.replace(tzinfo=None) == datetime(1985, 4, 12, 10, 15, 30)
    result = parse_datetime("1985-04-12T10:15:30-05:30")
    assert result.utcoffset() == timedelta(hours=-5, minutes=-30)


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10T15", ""])
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DT_EXT_COMPLETE, "1985-04-12T10:15:30+04:00"),
        (DT_BAS_COMPLETE, "19850412T101530+0400"),
        (DT_EXT_ORD_COMPLETE, "1985-102T10:15:30+04:00"),
        (DT_BAS_ORD_COMPLETE, "1985102T101530+0400"),
        (DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30+04:00"),
        (DT_BAS_WEEK_COMPLETE, "1985W155T101530+0400"),
    ],
)
def test_datetime_isoformat(fmt, expected):
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0))
    assert datetime_isoformat(dt, fmt) == expected
    assert parse_datetime(expected) == dt


def test_datetime_isoformat_default():
    assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30)) == "1985-04-12T10:15:30"
    assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)) == (
        "1985-04-12T10:15:30Z"
    )
    assert (
        datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30, 12), "%Y-%m-%dT%H:%M:%S.%f")
        == "1985-04-12T10:15:30.000012"
    )


# ---------------------------------------------------------------- durations parse


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P2W", timedelta(weeks=2)),
        ("P23DT23H", timedelta(days=23, hours=23)),
        ("PT36H", timedelta(hours=36)),
        ("PT1M", timedelta(minutes=1)),
        ("PT1S", timedelta(seconds=1)),
        ("PT1.5S", timedelta(seconds=1.5)),
        ("PT1,5S", timedelta(seconds=1.5)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P1W2D", timedelta(weeks=1, days=2)),
        ("P0.5D", timedelta(hours=12)),
        ("PT0.5H", timedelta(minutes=30)),
        ("PT0.5M", timedelta(seconds=30)),
        ("+P1D", timedelta(days=1)),
        ("-P2W", timedelta(weeks=-2)),
        ("-P1DT1H", -timedelta(days=1, hours=1)),
        ("-PT1S", timedelta(seconds=-1)),
        ("P0D", timedelta(0)),
        ("P0Y0M1D", timedelta(days=1)),
    ],
)
def test_parse_duration_timedelta(text, expected):
    result = parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P18Y9M4DT11H9M8S", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P3Y6M4DT12H30M5S", Duration(days=4, hours=12, minutes=30, seconds=5, months=6, years=3)),
        ("P4Y", Duration(years=4)),
        ("P1M", Duration(months=1)),
        ("P1Y2W", Duration(years=1, weeks=2)),
        ("P0.5Y", Duration(years=Decimal("0.5"))),
        ("P0,5Y", Duration(years=Decimal("0.5"))),
        ("P1.5M", Duration(months=Decimal("1.5"))),
        ("-P1Y", Duration(years=-1)),
        ("-P1Y2M3DT4H", Duration(years=-1, months=-2, days=-3, hours=-4)),
        ("+P1Y", Duration(years=1)),
    ],
)
def test_parse_duration_duration(text, expected):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result == expected
    assert result.years == expected.years
    assert result.months == expected.months
    assert result.tdelta == expected.tdelta


def test_parse_duration_fields():
    d = parse_duration("P1Y2M3W4DT5H6M7S")
    assert d.years == Decimal(1)
    assert d.months == Decimal(2)
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)
    assert d.tdelta == timedelta(weeks=3, days=4, hours=5, minutes=6, seconds=7)


def test_parse_duration_not_as_timedelta():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=1)
    assert d.years == 0 and d.months == 0
    d = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=-1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0003-06-04T12:30:05", Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5)),
        ("P00030604T123005", Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5)),
        (
            "P0003-06-04T12:30:05.5",
            Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5, milliseconds=500),
        ),
        ("P0001-01-01T00:00:01", Duration(years=1, months=1, days=1, seconds=1)),
    ],
)
def test_parse_duration_alternative(text, expected):
    for flag in (True, False):
        result = parse_duration(text, flag)
        assert isinstance(result, Duration)
        assert result == expected
        assert result.years == expected.years
        assert result.months == expected.months
        assert result.tdelta == expected.tdelta


@pytest.mark.parametrize("text", ["", "P", "1Y", "PX", "P1Y2", "P1S", "PT1D", "-P", "P1H", "Q1D"])
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_type_error():
    with pytest.raises(TypeError, match="Expecting a string"):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


# ---------------------------------------------------------------- durations format


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(weeks=2), "P14D"),
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(minutes=5), "PT5M"),
        (timedelta(seconds=5), "PT5S"),
        (timedelta(hours=1, seconds=5), "PT1H5S"),
        (timedelta(seconds=1.5), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(microseconds=120), "PT0.00012S"),
        (timedelta(days=1, microseconds=1), "P1DT0.000001S"),
        (timedelta(days=1), "P1D"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-1), "-PT1H"),
        (timedelta(seconds=-90), "-PT1M30S"),
        (timedelta(days=23, hours=23, minutes=59, seconds=59), "P23DT23H59M59S"),
        (Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8), "P18Y9M4DT11H9M8S"),
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=-1), "-P1Y"),
        (Duration(months=-1), "-P1M"),
        (Duration(days=-1), "-P1D"),
        (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
        (Duration(months=2, hours=3), "P2MT3H"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected
    assert duration_isoformat(value, D_DEFAULT) == expected


def test_duration_roundtrip():
    for text in ["P18Y9M4DT11H9M8S", "P1DT12H", "PT1.5S", "-P1Y2M3D", "-PT1H", "P2MT3H"]:
        assert duration_isoformat(parse_duration(text)) == text


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (timedelta(weeks=3), D_WEEK, "P3W"),
        (timedelta(days=20), D_WEEK, "P2W"),
        (timedelta(weeks=-3), D_WEEK, "-P3W"),
        (
            Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5),
            D_ALT_EXT,
            "P0003-06-04T12:30:05",
        ),
        (
            Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5),
            D_ALT_BAS,
            "P00030604T123005",
        ),
        (timedelta(days=1, hours=2), "%d %H", "01 02"),
        (timedelta(hours=23, minutes=59, seconds=58), "%H:%M:%S", "23:59:58"),
        (timedelta(microseconds=42), "%f", "000042"),
        (timedelta(days=15), "%W", "02"),
        (Duration(years=1985), "%C", "19"),
        (Duration(years=2000), "%C", "20"),
        (Duration(years=1985), "%Y", "1985"),
        (Duration(months=7), "%m", "07"),
        (timedelta(days=1), "%%d", "%d"),
        (timedelta(days=1), "%x", "%x"),
    ],
)
def test_duration_isoformat_formats(value, fmt, expected):
    assert duration_isoformat(value, fmt) == expected


def test_strftime_duration_yeardigits():
    assert strftime(Duration(years=3), "%Y", 6) == "+000003"
    assert strftime(Duration(years=1985), "%C", 6) == "+0019"
    assert strftime(Duration(years=3), "%Y") == "0003"
    assert strftime(timedelta(days=-15), "%W") == "02"


def test_strftime_date_misc():
    d = datetime(2001, 2, 3, 4, 5, 6, 7, tzinfo=FixedOffset(1, 0))
    assert strftime(d, "%d|%f|%H|%j|%m|%M|%S|%w|%W|%Y|%C|%z|%Z|%h|%%") == (
        "03|000007|04|034|02|05|06|6|05|2001|20|+0100|+01:00|+01|%"
    )
    assert strftime(date(2001, 2, 3), "%Y", 5) == "+02001"
    assert strftime(datetime(2001, 2, 3), "%z%Z%h") == ""


# ---------------------------------------------------------------- Duration class


@pytest.mark.parametrize(
    "val, low, high, expected",
    [
        (13, 1, 13, (1, 1)),
        (1, 1, 13, (0, 1)),
        (12, 1, 13, (0, 12)),
        (0, 1, 13, (-1, 12)),
        (-1, 1, 13, (-1, 11)),
        (25, 1, 13, (2, 1)),
        (-11, 1, 13, (-1, 1)),
        (-12, 1, 13, (-2, 12)),
        (7, 0, 5, (1, 2)),
        (Decimal("2.5"), 0, 1, (2, Decimal("0.5"))),
    ],
)
def test_fquotmod(val, low, high, expected):
    div, mod = fquotmod(Decimal(val), low, high)
    assert (div, mod) == expected
    assert isinstance(div, int)


@pytest.mark.parametrize(
    "year, month, expected",
    [
        (2001, 1, 31),
        (2001, 2, 28),
        (2001, 3, 31),
        (2001, 4, 30),
        (2001, 5, 31),
        (2001, 6, 30),
        (2001, 7, 31),
        (2001, 8, 31),
        (2001, 9, 30),
        (2001, 10, 31),
        (2001, 11, 30),
        (2001, 12, 31),
        (2000, 2, 29),
        (2004, 2, 29),
        (1900, 2, 28),
        (2100, 2, 28),
        (1600, 2, 29),
        (2002, 2, 28),
        (2003, 2, 28),
    ],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


def test_duration_init():
    d = Duration(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.months == Decimal(8)
    assert d.years == Decimal(9)
    assert isinstance(d.months, Decimal)
    assert isinstance(d.years, Decimal)
    d = Duration(months=1.5, years=0.25)
    assert d.months == Decimal("1.5")
    assert d.years == Decimal("0.25")
    d = Duration(months=Decimal("2.5"))
    assert d.months == Decimal("2.5")


def test_duration_getattr():
    d = Duration(days=2, seconds=5, microseconds=7)
    assert d.days == 2
    assert d.seconds == 5
    assert d.microseconds == 7
    assert d.total_seconds() == timedelta(days=2, seconds=5, microseconds=7).total_seconds()
    with pytest.raises(AttributeError):
        d.nonexistent_attribute


def test_duration_pickle():
    d = Duration(days=3, hours=4, months=5, years=6)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d
    assert d2.years == 6 and d2.months == 5 and d2.tdelta == d.tdelta
    assert d.__getstate__() == {"months": Decimal(5), "years": Decimal(6), "tdelta": d.tdelta}


def test_duration_str():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(years=2, hours=3)) == "2 years, 3:00:00"
    assert str(Duration()) == "0:00:00"
    assert str(Duration(months=-2)) == "-2 month, 0:00:00"


def test_duration_repr():
    assert repr(Duration(10, 10, years=10, months=10)) == (
        "isodate.duration.Duration(10, 10, 0, years=10, months=10)"
    )
    assert repr(Duration(1, 2, 3, years=4, months=5)) == (
        "isodate.duration.Duration(1, 2, 3, years=4, months=5)"
    )


def test_duration_hash():
    assert hash(Duration(days=1, months=2, years=3)) == hash(Duration(days=1, months=2, years=3))
    base = Duration(days=1)
    assert hash(base) != hash(Duration(days=1, months=1))
    assert hash(base) != hash(Duration(days=1, years=1))
    assert hash(base) != hash(Duration(days=2))
    assert hash(Duration(months=1)) != hash(Duration(years=1))
    s = {Duration(days=1), Duration(days=1), Duration(months=1)}
    assert len(s) == 2


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == timedelta(days=-3)
    assert isinstance(d, Duration)


def test_duration_add_duration():
    d = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
    assert isinstance(d, Duration)
    assert d.years == 5
    assert d.months == 7
    assert d.tdelta == timedelta(days=9)


def test_duration_add_timedelta():
    for d in (
        Duration(years=1, months=2, days=3) + timedelta(days=4),
        timedelta(days=4) + Duration(years=1, months=2, days=3),
    ):
        assert isinstance(d, Duration)
        assert d.years == 1
        assert d.months == 2
        assert d.tdelta == timedelta(days=7)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 3, 30), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 3, 29), Duration(months=1), date(2000, 4, 29)),
        (date(2000, 1, 1), Duration(years=1, months=13, days=1), date(2002, 2, 2)),
        (date(2000, 1, 1), Duration(months=24), date(2002, 1, 1)),
        (date(2000, 1, 1), Duration(days=40), date(2000, 2, 10)),
        (date(2000, 5, 10), Duration(years=-1, months=-5), date(1998, 12, 10)),
        (
            datetime(2000, 1, 31, 23, 0),
            Duration(months=1, hours=2),
            datetime(2000, 3, 1, 1, 0),
        ),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected
    assert type(start + dur) is type(start)


def test_duration_add_date_keeps_tz():
    start = datetime(2000, 1, 1, tzinfo=UTC)
    res = start + Duration(months=1)
    assert res == datetime(2000, 2, 1, tzinfo=UTC)
    assert res.tzinfo is UTC


def test_duration_add_fractional_errors():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("1.5"))


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) + "abc"
    with pytest.raises(TypeError):
        "abc" + Duration(days=1)
    assert Duration(days=1).__add__("abc") is NotImplemented
    assert Duration(days=1).__add__(1) is NotImplemented


def test_duration_mul():
    d = Duration(years=1, months=2, days=3)
    for r in (d * 3, 3 * d):
        assert isinstance(r, Duration)
        assert r.years == 3
        assert r.months == 6
        assert r.tdelta == timedelta(days=9)
    r = d * -1
    assert r.years == -1 and r.months == -2 and r.tdelta == timedelta(days=-3)
    with pytest.raises(TypeError):
        d * 1.5
    with pytest.raises(TypeError):
        d * d
    assert d.__mul__(1.5) is NotImplemented


def test_duration_sub_duration():
    d = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
    assert isinstance(d, Duration)
    assert d.years == 4
    assert d.months == 5
    assert d.tdelta == timedelta(days=6)


def test_duration_sub_timedelta():
    d = Duration(years=1, months=2, days=9) - timedelta(days=3)
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=6)


def test_duration_sub_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) - "abc"
    assert Duration(days=1).__sub__("abc") is NotImplemented
    with pytest.raises(TypeError):
        "abc" - Duration(days=1)
    assert Duration(days=1).__rsub__("abc") is NotImplemented
    with pytest.raises(TypeError):
        Duration(days=1) - date(2000, 1, 1)


def test_timedelta_minus_duration():
    d = timedelta(days=10) - Duration(days=3, months=1, years=2)
    assert isinstance(d, Duration)
    assert d.years == -2
    assert d.months == -1
    assert d.tdelta == timedelta(days=7)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=1), date(2000, 11, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(2000, 2, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2001, 3, 1), Duration(days=1), date(2001, 2, 28)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2004, 2, 29), Duration(years=4), date(2000, 2, 29)),
        (date(2002, 2, 2), Duration(years=1, months=13, days=1), date(2000, 1, 1)),
        (date(2000, 5, 31), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 5, 30), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 5, 29), Duration(months=1), date(2000, 4, 29)),
        (
            datetime(2000, 3, 1, 1, 0),
            Duration(months=1, hours=2),
            datetime(2000, 1, 31, 23, 0),
        ),
    ],
)
def test_date_minus_duration(start, dur, expected):
    assert start - dur == expected
    assert type(start - dur) is type(start)


def test_duration_eq():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert not (Duration(years=1) == Duration(months=11))
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert not (Duration(days=1) == timedelta(days=2))
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert not (Duration(months=1, days=1) == timedelta(days=1))
    assert not (Duration(months=1) == Duration(months=2))
    assert not (Duration(days=1) == Duration(days=2))
    assert not (Duration(days=1, months=1) == Duration(days=2, months=1))
    assert Duration(days=1, months=1) == Duration(days=1, months=1)
    assert Duration(days=7) == Duration(weeks=1)
    assert Duration(years=1).__eq__(timedelta(0)) is False
    assert Duration(months=1).__eq__(timedelta(0)) is False
    assert Duration().__eq__(timedelta(0)) is True
    assert Duration(days=1).__eq__(Duration(days=2)) is False
    assert Duration(months=1).__eq__(Duration(months=2)) is False
    assert Duration(years=2).__eq__(Duration(years=2)) is True


def test_duration_ne():
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) != Duration(days=2)
    assert Duration(days=1, months=1) != Duration(days=2, months=1)
    assert Duration(days=1, months=1) != Duration(days=1, months=2)
    assert not (Duration(days=1, months=1) != Duration(days=1, months=1))
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(years=1) != timedelta(0)
    assert Duration(months=1) != timedelta(0)
    assert Duration(years=1).__ne__(timedelta(0)) is True
    assert Duration(months=1).__ne__(timedelta(0)) is True
    assert Duration().__ne__(timedelta(0)) is False
    assert Duration().__ne__(timedelta(1)) is True
    assert Duration(years=1).__ne__(Duration(months=12)) is False
    assert Duration(years=1).__ne__(Duration(months=13)) is True
    assert Duration(years=1).__ne__(Duration(years=1)) is False
    assert Duration(months=12).__ne__(Duration(years=1)) is False
    assert Duration(years=2).__ne__(Duration(years=1, months=12)) is False
    assert Duration(years=1, days=1).__ne__(Duration(years=1, days=1)) is False


def test_duration_totimedelta():
    assert Duration(months=1).totimedelta(start=date(2000, 2, 1)) == timedelta(29)
    assert Duration(months=1).totimedelta(start=date(2000, 4, 1)) == timedelta(30)
    assert Duration(months=1).totimedelta(end=date(2000, 4, 1)) == timedelta(31)
    assert Duration(months=1).totimedelta(end=date(2000, 3, 1)) == timedelta(29)
    assert Duration(years=1, days=1).totimedelta(start=date(2000, 1, 1)) == timedelta(367)
    assert Duration(days=5).totimedelta(start=datetime(2000, 1, 1)) == timedelta(5)
    with pytest.raises(ValueError, match="start or end required"):
        Duration(months=1).totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        Duration(months=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_public_api():
    assert isinstance(isodate.__version__, str)
    for name in isodate.__all__:
        assert hasattr(isodate, name)


def test_error_messages():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 date format: 'abc'"):
        parse_date("abc")
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 time format: 'abc'"):
        parse_time("abc")
    with pytest.raises(ISO8601Error, match="time designator 'T' missing"):
        parse_datetime("1985-04-12")
    with pytest.raises(ISO8601Error, match="Unable to parse duration string '1Y'"):
        parse_duration("1Y")
    with pytest.raises(ValueError, match="fractional years or months not supported"):
        date(2000, 1, 1) + Duration(months=Decimal("1.5"))
    with pytest.raises(ValueError, match="fractional years or months not supported"):
        date(2000, 1, 1) - Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError, match='unknown format string "%Q"'):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(2, 0)), "%Q")
    assert issubclass(ISO8601Error, ValueError)
