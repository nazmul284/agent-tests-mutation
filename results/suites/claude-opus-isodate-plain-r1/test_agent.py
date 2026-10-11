"""Test suite for the isodate package."""

import pickle
from datetime import date, datetime, time, timedelta, timezone
from datetime import tzinfo as dt_tzinfo
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
from isodate.duration import fquotmod, max_days_in_month
from isodate.isotzinfo import build_tzinfo


# ---------------------------------------------------------------------------
# Format string constants
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


def test_iso8601error_is_value_error():
    assert issubclass(ISO8601Error, ValueError)


def test_all_exports_exist():
    for name in isodate.__all__:
        assert hasattr(isodate, name)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


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
        # Year where Jan 1 lies in week 1 (Thursday 2004-01-01)
        ("2004-W01-1", date(2003, 12, 29)),
        ("2004-W53-7", date(2005, 1, 2)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W53-7", date(2010, 1, 3)),
        # Year where Jan 1 lies in last week of previous year (Fri 2010-01-01)
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W01", date(2010, 1, 4)),
        ("2010-W01-7", date(2010, 1, 10)),
        ("2010W017", date(2010, 1, 10)),
        # Jan 1 on a Monday
        ("2018-W01-1", date(2018, 1, 1)),
        ("2018-W01-3", date(2018, 1, 3)),
        # Jan 1 on a Sunday (2017)
        ("2017-W01-1", date(2017, 1, 2)),
        ("2017-W52-7", date(2017, 12, 31)),
    ],
)
def test_parse_date_basic(text, expected):
    result = parse_date(text)
    assert result == expected
    assert type(result) is date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+19", date(1901, 1, 1)),
        ("+1985", date(1985, 1, 1)),
        ("+1985-04", date(1985, 4, 1)),
        ("+198504", date(1985, 4, 1)),
        ("+1985-04-12", date(1985, 4, 12)),
        ("+19850412", date(1985, 4, 12)),
        ("+1985-102", date(1985, 4, 12)),
        ("+1985102", date(1985, 4, 12)),
        ("+1985-W15-5", date(1985, 4, 12)),
        ("+1985W155", date(1985, 4, 12)),
        ("+1985-W15", date(1985, 4, 8)),
        ("+1985W15", date(1985, 4, 8)),
    ],
)
def test_parse_date_expanded_four_digits(text, expected):
    assert parse_date(text, 4, True) == expected
    # without expanded, the sign is not allowed
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+0019", date(1901, 1, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+001985-04", date(1985, 4, 1)),
        ("+00198504", date(1985, 4, 1)),
        ("+001985-04-12", date(1985, 4, 12)),
        ("+0019850412", date(1985, 4, 12)),
        ("+001985-102", date(1985, 4, 12)),
        ("+001985102", date(1985, 4, 12)),
        ("+001985-W15-5", date(1985, 4, 12)),
        ("+001985W155", date(1985, 4, 12)),
        ("+001985-W15", date(1985, 4, 8)),
        ("+001985W15", date(1985, 4, 8)),
    ],
)
def test_parse_date_six_digits(text, expected):
    assert parse_date(text, 6) == expected
    # expanded is forced for yeardigits != 4
    assert parse_date(text, 6, False) == expected


def test_parse_date_six_digits_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", 6)


def test_parse_date_negative_year_raises_value_error():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)
    with pytest.raises(ValueError):
        parse_date("-1985", 4, True)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)
    assert parse_date("1985-04", defaultmonth=7, defaultday=20) == date(1985, 4, 20)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    # default month/day do not affect complete dates
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-102", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-W15", defaultmonth=6, defaultday=15) == date(1985, 4, 8)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1",
        "198",
        "1985-4-12",
        "1985-04-1",
        "85-04-12",
        "1985/04/12",
        "1985-04-12T",
        " 1985-04-12",
        "1985-04-12 ",
        "1985-W1",
        "1985-W15-",
        "1985-W15-12",
        "abcd",
        "1985-04-12x",
        "x1985",
    ],
)
def test_parse_date_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


def test_parse_date_error_message_contains_input():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 date format"):
        parse_date("garbage")


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1985-04-00", "19850231"])
def test_parse_date_out_of_range(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_parse_date_feb_29_leap():
    assert parse_date("2000-02-29") == date(2000, 2, 29)
    with pytest.raises(ValueError):
        parse_date("1900-02-29")


# ---------------------------------------------------------------------------
# date_isoformat / strftime on dates
# ---------------------------------------------------------------------------


D = date(1985, 4, 12)


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_CENTURY, "19"),
        (DATE_YEAR, "1985"),
        (DATE_EXT_MONTH, "1985-04"),
        (DATE_BAS_MONTH, "198504"),
        (DATE_EXT_COMPLETE, "1985-04-12"),
        (DATE_BAS_COMPLETE, "19850412"),
        (DATE_EXT_ORD_COMPLETE, "1985-102"),
        (DATE_BAS_ORD_COMPLETE, "1985102"),
        (DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (DATE_BAS_WEEK_COMPLETE, "1985W155"),
        (DATE_EXT_WEEK, "1985-W15"),
        (DATE_BAS_WEEK, "1985W15"),
    ],
)
def test_date_isoformat(fmt, expected):
    assert date_isoformat(D, fmt) == expected
    # roundtrip
    if fmt not in (DATE_EXT_WEEK, DATE_BAS_WEEK, DATE_EXT_MONTH, DATE_BAS_MONTH):
        if fmt not in (DATE_YEAR, DATE_CENTURY):
            assert parse_date(expected) == D


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_CENTURY, "+0019"),
        (DATE_YEAR, "+001985"),
        (DATE_EXT_MONTH, "+001985-04"),
        (DATE_EXT_COMPLETE, "+001985-04-12"),
        (DATE_BAS_COMPLETE, "+0019850412"),
        (DATE_EXT_ORD_COMPLETE, "+001985-102"),
        (DATE_EXT_WEEK_COMPLETE, "+001985-W15-5"),
    ],
)
def test_date_isoformat_six_digits(fmt, expected):
    assert date_isoformat(D, fmt, 6) == expected


def test_date_isoformat_default():
    assert date_isoformat(D) == "1985-04-12"
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"


def test_strftime_date_codes():
    d = date(2004, 12, 31)
    assert strftime(d, "%j") == "366"
    assert strftime(d, "%w") == "5"
    assert strftime(d, "%W") == "53"
    assert strftime(d, "%d") == "31"
    assert strftime(d, "%m") == "12"
    assert strftime(date(2005, 1, 1), "%j") == "001"
    assert strftime(date(2005, 1, 2), "%w") == "7"
    assert strftime(date(2005, 1, 3), "%w") == "1"
    assert strftime(date(2005, 1, 3), "%W") == "01"
    assert strftime(d, "%%") == "%"
    assert strftime(d, "100%% %Y") == "100% 2004"
    # unknown codes are left untouched
    assert strftime(d, "%Q %y") == "%Q %y"


def test_strftime_century():
    assert strftime(date(2000, 1, 1), "%C") == "20"
    assert strftime(date(1999, 1, 1), "%C") == "19"
    assert strftime(date(1, 1, 1), "%C") == "00"


def test_strftime_week_date_roundtrip_many():
    start = date(1999, 12, 20)
    for offset in range(0, 800, 3):
        d = start + timedelta(days=offset)
        if d.isocalendar()[0] == d.year:
            # %Y is the calendar year, so week dates only roundtrip when
            # the ISO week-numbering year matches it.
            assert parse_date(date_isoformat(d, DATE_EXT_WEEK_COMPLETE)) == d
            assert parse_date(date_isoformat(d, DATE_BAS_WEEK_COMPLETE)) == d
        assert parse_date(date_isoformat(d, DATE_EXT_ORD_COMPLETE)) == d
        assert parse_date(date_isoformat(d, DATE_BAS_ORD_COMPLETE)) == d
        assert parse_date(date_isoformat(d, DATE_BAS_COMPLETE)) == d


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("232050", time(23, 20, 50)),
        ("23:20:50", time(23, 20, 50)),
        ("2320", time(23, 20)),
        ("23:20", time(23, 20)),
        ("23", time(23)),
        ("T232050", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("T2320", time(23, 20)),
        ("T23", time(23)),
        ("232050,5", time(23, 20, 50, 500000)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50,5", time(23, 20, 50, 500000)),
        ("2320,8", time(23, 20, 48)),
        ("23:20,8", time(23, 20, 48)),
        ("23:20.5", time(23, 20, 30)),
        ("23,3", time(23, 18)),
        ("23.5", time(23, 30)),
        ("00.25", time(0, 15)),
        ("12.0125", time(12, 0, 45)),
        ("00:00:00", time(0, 0, 0)),
        ("15:27:46.123456", time(15, 27, 46, 123456)),
        ("15:27:46.1234567", time(15, 27, 46, 123456)),
        ("15:27:46.1234569", time(15, 27, 46, 123456)),
        ("15:27:46.000001", time(15, 27, 46, 1)),
        ("15:27:46.9999999", time(15, 27, 46, 999999)),
        ("10:15:00.1", time(10, 15, 0, 100000)),
        ("101500.01", time(10, 15, 0, 10000)),
        ("12:00.000001", time(12, 0, 0, 60)),
        ("12:00.0001", time(12, 0, 0, 6000)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "text, expected_time, offset",
    [
        ("232030Z", time(23, 20, 30), timedelta(0)),
        ("23:20:30Z", time(23, 20, 30), timedelta(0)),
        ("2320Z", time(23, 20), timedelta(0)),
        ("23Z", time(23), timedelta(0)),
        ("152746+0100", time(15, 27, 46), timedelta(hours=1)),
        ("15:27:46+01:00", time(15, 27, 46), timedelta(hours=1)),
        ("15:27:46+01", time(15, 27, 46), timedelta(hours=1)),
        ("152746-0500", time(15, 27, 46), timedelta(hours=-5)),
        ("15:27:46-05:00", time(15, 27, 46), timedelta(hours=-5)),
        ("15:27:46-05", time(15, 27, 46), timedelta(hours=-5)),
        ("15:27:46+05:30", time(15, 27, 46), timedelta(hours=5, minutes=30)),
        ("15:27:46-05:30", time(15, 27, 46), timedelta(hours=-5, minutes=-30)),
        ("1527-0230", time(15, 27), timedelta(hours=-2, minutes=-30)),
        ("15.5+0130", time(15, 30), timedelta(hours=1, minutes=30)),
        ("15:27:46.5+01:00", time(15, 27, 46, 500000), timedelta(hours=1)),
        ("15:27:46,5Z", time(15, 27, 46, 500000), timedelta(0)),
    ],
)
def test_parse_time_with_tz(text, expected_time, offset):
    result = parse_time(text)
    assert result.replace(tzinfo=None) == expected_time
    assert result.tzinfo is not None
    assert result.utcoffset() == offset


def test_parse_time_z_is_utc_singleton():
    assert parse_time("12:00Z").tzinfo is UTC


def test_parse_time_tzname():
    assert parse_time("12:00+01:00").tzinfo.tzname(None) == "+01:00"
    assert parse_time("12:00-0130").tzinfo.tzname(None) == "-0130"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2",
        "232",
        "23:2",
        "23:20:5",
        "2320505",
        "23:20:50.",
        "23:20:50Z+01",
        "23:20:50+1",
        "23:20:50+01:0",
        "23:20:50 ",
        "abc",
        "23-20-50",
        "TT23",
        "23:20:50X",
    ],
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 time format"):
        parse_time(text)


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("25:00:00")
    with pytest.raises(ValueError):
        parse_time("23:60:00")


# ---------------------------------------------------------------------------
# time_isoformat
# ---------------------------------------------------------------------------


def test_time_isoformat_naive():
    t = time(23, 20, 50)
    assert time_isoformat(t) == "23:20:50"
    assert time_isoformat(t, TIME_BAS_COMPLETE) == "232050"
    assert time_isoformat(t, TIME_EXT_MINUTE) == "23:20"
    assert time_isoformat(t, TIME_BAS_MINUTE) == "2320"
    assert time_isoformat(t, TIME_HOUR) == "23"
    assert time_isoformat(time(1, 2, 3)) == "01:02:03"


def test_time_isoformat_with_tz():
    t = time(15, 27, 46, tzinfo=FixedOffset(1, 0, "+01:00"))
    assert time_isoformat(t) == "15:27:46+01:00"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "152746+0100"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_HOUR) == "152746+01"
    t = time(15, 27, 46, tzinfo=FixedOffset(-5, -30, "-05:30"))
    assert time_isoformat(t) == "15:27:46-05:30"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "152746-0530"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_HOUR) == "152746-05"
    t = time(15, 27, 46, tzinfo=UTC)
    assert time_isoformat(t) == "15:27:46Z"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "152746Z"
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_HOUR) == "152746Z"


def test_time_isoformat_microseconds():
    t = time(1, 2, 3, 45)
    assert time_isoformat(t, "%H:%M:%S.%f") == "01:02:03.000045"


@pytest.mark.parametrize(
    "text",
    ["23:20:50", "232050", "15:27:46+01:00", "15:27:46-05:30", "15:27:46Z"],
)
def test_time_roundtrip(text):
    t = parse_time(text)
    out = time_isoformat(t)
    assert parse_time(out) == t


# ---------------------------------------------------------------------------
# parse_datetime / datetime_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("19850412T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985102T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985W155T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-W15-5T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15:30.5", datetime(1985, 4, 12, 10, 15, 30, 500000)),
        ("2012-10-30T08:55:22.1234567", datetime(2012, 10, 30, 8, 55, 22, 123456)),
        ("2012-10-12T08:29:46.069178", datetime(2012, 10, 12, 8, 29, 46, 69178)),
    ],
)
def test_parse_datetime_naive(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.tzinfo is None
    assert type(result) is datetime


@pytest.mark.parametrize(
    "text, expected",
    [
        ("19850412T101530Z", datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone.utc)),
        (
            "1985-04-12T10:15:30+04:00",
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone(timedelta(hours=4))),
        ),
        (
            "1985-04-12T10:15:30-0430",
            datetime(
                1985, 4, 12, 10, 15, 30, tzinfo=timezone(-timedelta(hours=4, minutes=30))
            ),
        ),
        (
            "1985-04-12T10:15:30+04",
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone(timedelta(hours=4))),
        ),
    ],
)
def test_parse_datetime_with_tz(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize(
    "text",
    ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10:15T30", "", "T10:15"],
)
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


def test_parse_datetime_missing_t_message():
    with pytest.raises(ISO8601Error, match="time designator 'T' missing"):
        parse_datetime("1985-04-12 10:15")


def test_parse_datetime_bad_parts():
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-4-12T10:15")
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12T1:15")


def test_datetime_isoformat():
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "+04:00"))
    assert datetime_isoformat(dt) == "1985-04-12T10:15:30+04:00"
    assert datetime_isoformat(dt, DT_BAS_COMPLETE) == "19850412T101530+0400"
    assert datetime_isoformat(dt, DT_EXT_ORD_COMPLETE) == "1985-102T10:15:30+04:00"
    assert datetime_isoformat(dt, DT_BAS_ORD_COMPLETE) == "1985102T101530+0400"
    assert datetime_isoformat(dt, DT_EXT_WEEK_COMPLETE) == "1985-W15-5T10:15:30+04:00"
    assert datetime_isoformat(dt, DT_BAS_WEEK_COMPLETE) == "1985W155T101530+0400"
    naive = datetime(1985, 4, 12, 10, 15, 30)
    assert datetime_isoformat(naive) == "1985-04-12T10:15:30"
    utc = datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
    assert datetime_isoformat(utc) == "1985-04-12T10:15:30Z"
    assert datetime_isoformat(utc, DT_BAS_COMPLETE) == "19850412T101530Z"


def test_datetime_roundtrip():
    for text in [
        "1985-04-12T10:15:30+04:00",
        "1985-04-12T10:15:30-05:30",
        "1985-04-12T10:15:30Z",
        "2000-02-29T00:00:00",
    ]:
        assert datetime_isoformat(parse_datetime(text)) == text


# ---------------------------------------------------------------------------
# tzinfo parsing/formatting
# ---------------------------------------------------------------------------


def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+01:00")
    assert tz.utcoffset(None) == timedelta(hours=1)
    assert tz.tzname(None) == "+01:00"
    assert parse_tzinfo("-0130").utcoffset(None) == timedelta(hours=-1, minutes=-30)
    assert parse_tzinfo("+05").utcoffset(None) == timedelta(hours=5)
    assert parse_tzinfo("-05").utcoffset(None) == timedelta(hours=-5)
    assert parse_tzinfo("+00:00").utcoffset(None) == timedelta(0)


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("+0230", "+", 2, 30)
    assert tz.utcoffset(None) == timedelta(hours=2, minutes=30)
    tz = build_tzinfo("-0230", "-", 2, 30)
    assert tz.utcoffset(None) == timedelta(hours=-2, minutes=-30)
    assert tz.tzname(None) == "-0230"
    tz = build_tzinfo("+02")
    assert tz.utcoffset(None) == timedelta(0)


def test_tz_isoformat():
    naive = datetime(2000, 1, 1)
    assert tz_isoformat(naive) == ""
    utc = datetime(2000, 1, 1, tzinfo=UTC)
    assert tz_isoformat(utc) == "Z"
    assert tz_isoformat(utc, "%z") == "Z"
    assert tz_isoformat(utc, "%h") == "Z"
    pos = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 45))
    assert tz_isoformat(pos) == "+05:45"
    assert tz_isoformat(pos, "%Z") == "+05:45"
    assert tz_isoformat(pos, "%z") == "+0545"
    assert tz_isoformat(pos, "%h") == "+05"
    neg = datetime(2000, 1, 1, tzinfo=FixedOffset(-3, -15))
    assert tz_isoformat(neg) == "-03:15"
    assert tz_isoformat(neg, "%z") == "-0315"
    assert tz_isoformat(neg, "%h") == "-03"
    big = datetime(2000, 1, 1, tzinfo=FixedOffset(12, 0))
    assert tz_isoformat(big, "%Z") == "+12:00"
    assert tz_isoformat(big, "%z") == "+1200"


def test_tz_isoformat_fixedoffset_zero_is_z():
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0))) == "Z"


def test_tz_isoformat_stdlib_timezone():
    dt = datetime(2000, 1, 1, tzinfo=timezone(timedelta(hours=-8)))
    assert tz_isoformat(dt) == "-08:00"


def test_tz_isoformat_unknown_format():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0))
    with pytest.raises(ValueError, match="unknown format"):
        tz_isoformat(dt, "%X")


def test_tz_isoformat_tzinfo_returning_none():
    class NoneTz(dt_tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=NoneTz())) == ""


def test_tz_isoformat_dst_nonzero_is_not_z():
    class Weird(dt_tzinfo):
        def utcoffset(self, dt):
            return timedelta(0)

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Weird())) == "+00:00"


def test_tz_isoformat_overflow():
    class Huge:
        def utcoffset(self, dt):
            return timedelta(hours=100)

        def dst(self, dt):
            return timedelta(0)

    class FakeDT:
        tzinfo = Huge()

    with pytest.raises(OverflowError):
        tz_isoformat(FakeDT())

    class Edge:
        def utcoffset(self, dt):
            return timedelta(hours=99, minutes=59)

        def dst(self, dt):
            return timedelta(0)

    class FakeDT2:
        tzinfo = Edge()

    assert tz_isoformat(FakeDT2()) == "+99:59"


def test_tz_isoformat_negative_with_days():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-23, -59))
    assert tz_isoformat(dt) == "-23:59"


# ---------------------------------------------------------------------------
# tzinfo classes
# ---------------------------------------------------------------------------


def test_utc():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_fixed_offset():
    fo = FixedOffset()
    assert fo.utcoffset(None) == timedelta(0)
    assert fo.tzname(None) == "UTC"
    assert fo.dst(None) == timedelta(0)
    fo = FixedOffset(-2, -30, "custom")
    assert fo.utcoffset(None) == timedelta(hours=-2, minutes=-30)
    assert fo.tzname(None) == "custom"
    assert fo.dst(None) == timedelta(0)
    assert repr(fo) == "<FixedOffset 'custom'>"
    fo = FixedOffset(offset_minutes=90)
    assert fo.utcoffset(None) == timedelta(hours=1, minutes=30)


def test_local_timezone():
    import time as _time

    dt = datetime(2020, 6, 1, 12, 0)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert LOCAL.dst(dt) in (timedelta(0), off - timedelta(seconds=-_time.timezone))
    assert LOCAL.tzname(dt) in _time.tzname
    # offset must be consistent with time.localtime
    stamp = _time.mktime((2020, 6, 1, 12, 0, 0, 0, 0, -1))
    lt = _time.localtime(stamp)
    assert off == timedelta(seconds=lt.tm_gmtoff)
    assert LOCAL.dst(dt) == (off - timedelta(seconds=-_time.timezone) if lt.tm_isdst > 0 else timedelta(0))
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


def test_local_timezone_winter():
    import time as _time

    dt = datetime(2020, 1, 15, 12, 0)
    stamp = _time.mktime((2020, 1, 15, 12, 0, 0, 0, 0, -1))
    lt = _time.localtime(stamp)
    assert LOCAL.utcoffset(dt) == timedelta(seconds=lt.tm_gmtoff)
    assert LOCAL.tzname(dt) == _time.tzname[lt.tm_isdst > 0]


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def test_fquotmod():
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(1), 1, 13) == (0, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-11), 1, 13) == (-1, Decimal(1))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(7), 0, 5) == (1, Decimal(2))
    assert fquotmod(Decimal(-1), 0, 5) == (-1, Decimal(4))
    div, mod = fquotmod(Decimal(14), 1, 13)
    assert isinstance(div, int)


@pytest.mark.parametrize(
    "year, month, days",
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
        (1900, 2, 28),
        (2004, 2, 29),
        (2100, 2, 28),
        (2400, 2, 29),
        (1600, 2, 29),
        (1996, 2, 29),
    ],
)
def test_max_days_in_month(year, month, days):
    assert max_days_in_month(year, month) == days


def test_max_days_in_month_matches_calendar():
    import calendar

    for year in range(1890, 2110):
        for month in range(1, 13):
            assert max_days_in_month(year, month) == calendar.monthrange(year, month)[1]


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


def test_duration_init():
    d = Duration(
        days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=7,
        months=8, years=9,
    )
    assert d.years == Decimal(9)
    assert d.months == Decimal(8)
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    # attribute delegation
    assert d.days == 50
    assert d.seconds == 6 * 3600 + 5 * 60 + 2
    assert d.microseconds == 4003


def test_duration_defaults():
    d = Duration()
    assert d.years == 0
    assert d.months == 0
    assert d.tdelta == timedelta(0)


def test_duration_decimal_values_kept():
    d = Duration(years=Decimal("1.5"), months=Decimal("2.25"))
    assert d.years == Decimal("1.5")
    assert d.months == Decimal("2.25")
    d = Duration(years=1.5, months=0.1)
    assert d.years == Decimal("1.5")
    assert d.months == Decimal("0.1")


def test_duration_getattr_missing():
    with pytest.raises(AttributeError):
        Duration().nonexistent_attribute


def test_duration_str():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(years=2, months=2)) == "2 years, 2 months, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=-1)) == "-1 month, 0:00:00"
    assert str(Duration(months=0.5)) == "0.5 month, 0:00:00"
    assert str(Duration(months=1.5)) == "1.5 months, 0:00:00"
    assert str(Duration(hours=1)) == "1:00:00"
    assert str(Duration(years=1)) == "1 years, 0:00:00"


def test_duration_repr():
    d = Duration(10, 10, years=10, months=10)
    assert repr(d) == "isodate.duration.Duration(10, 10, 0, years=10, months=10)"
    d = Duration(1, 2, 3)
    assert repr(d) == "isodate.duration.Duration(1, 2, 3, years=0, months=0)"


def test_duration_hash():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=1, months=2, days=3)
    c = Duration(years=1, months=2, days=4)
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)
    assert len({a, b, c}) == 2
    assert hash(a) == hash((timedelta(days=3), Decimal(2), Decimal(1)))


def test_duration_neg():
    d = Duration(years=1, months=2, days=3, hours=4)
    n = -d
    assert n.years == -1
    assert n.months == -2
    assert n.tdelta == -timedelta(days=3, hours=4)
    assert -n == d
    # original unchanged
    assert d.years == 1


def test_duration_add_duration():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=4, months=5, hours=6)
    r = a + b
    assert isinstance(r, Duration)
    assert r.years == 5
    assert r.months == 7
    assert r.tdelta == timedelta(days=3, hours=6)


def test_duration_add_timedelta():
    a = Duration(years=1, months=2, days=3)
    r = a + timedelta(hours=5)
    assert isinstance(r, Duration)
    assert r.years == 1
    assert r.months == 2
    assert r.tdelta == timedelta(days=3, hours=5)
    r2 = timedelta(hours=5) + a
    assert isinstance(r2, Duration)
    assert r2 == r


def test_duration_add_unsupported():
    assert Duration().__add__("x") is NotImplemented
    with pytest.raises(TypeError):
        Duration() + "x"
    with pytest.raises(TypeError):
        Duration() + 1


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(2000, 2, 15)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 1, 15), Duration(months=12), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(2001, 2, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 1, 1), Duration(years=1, months=2, days=3), date(2001, 3, 4)),
        (date(2000, 5, 31), Duration(months=1), date(2000, 6, 30)),
        (date(2000, 5, 30), Duration(months=1), date(2000, 6, 30)),
        (date(2000, 1, 1), Duration(days=-1), date(1999, 12, 31)),
        (
            datetime(2000, 1, 31, 12, 30),
            Duration(months=1, hours=12),
            datetime(2000, 3, 1, 0, 30),
        ),
        (
            datetime(1999, 12, 31, 23, 59, 59),
            Duration(years=1, seconds=1),
            datetime(2001, 1, 1),
        ),
    ],
)
def test_duration_add_date(start, dur, expected):
    r = start + dur
    assert r == expected
    assert type(r) is type(expected)
    assert dur + start == expected


def test_duration_add_preserves_tzinfo():
    tz = FixedOffset(2, 0)
    dt = datetime(2000, 1, 1, 10, tzinfo=tz)
    r = dt + Duration(months=1)
    assert r.tzinfo is tz
    assert r == datetime(2000, 2, 1, 10, tzinfo=tz)


def test_duration_add_date_fractional_raises():
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) + Duration(years=1.5)
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) + Duration(months=0.5)
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) - Duration(years=1.5)
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) - Duration(months=0.5)
    # integral decimal values are fine
    assert date(2000, 1, 1) + Duration(years=Decimal("1.0")) == date(2001, 1, 1)
    assert date(2000, 1, 1) - Duration(months=Decimal("1.0")) == date(1999, 12, 1)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3, hours=1)
    r = d * 3
    assert r.years == 3
    assert r.months == 6
    assert r.tdelta == timedelta(days=9, hours=3)
    assert 3 * d == r
    assert d * 0 == Duration()
    r = d * -2
    assert r.years == -2
    assert r.months == -4
    assert r.tdelta == timedelta(days=-6, hours=-2)
    assert d.__mul__(1.5) is NotImplemented
    with pytest.raises(TypeError):
        d * 1.5
    with pytest.raises(TypeError):
        d * "a"


def test_duration_sub_duration():
    a = Duration(years=3, months=5, days=10)
    b = Duration(years=1, months=2, days=3, hours=1)
    r = a - b
    assert isinstance(r, Duration)
    assert r.years == 2
    assert r.months == 3
    assert r.tdelta == timedelta(days=7) - timedelta(hours=1)


def test_duration_sub_timedelta():
    a = Duration(years=3, months=5, days=10)
    r = a - timedelta(days=4)
    assert isinstance(r, Duration)
    assert r.years == 3
    assert r.months == 5
    assert r.tdelta == timedelta(days=6)


def test_duration_sub_unsupported():
    assert Duration().__sub__("x") is NotImplemented
    with pytest.raises(TypeError):
        Duration() - "x"
    with pytest.raises(TypeError):
        Duration() - date(2000, 1, 1)


def test_duration_rsub_timedelta():
    d = Duration(years=1, months=2, days=3)
    r = timedelta(days=10) - d
    assert isinstance(r, Duration)
    assert r.years == -1
    assert r.months == -2
    assert r.tdelta == timedelta(days=7)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=1), date(2000, 11, 15)),
        (date(2000, 1, 15), Duration(months=12), date(1999, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(1998, 12, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(2000, 2, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2000, 2, 29), Duration(years=1), date(1999, 2, 28)),
        (date(2001, 3, 4), Duration(years=1, months=2, days=3), date(2000, 1, 1)),
        (date(2000, 7, 31), Duration(months=1), date(2000, 6, 30)),
        (date(2000, 7, 30), Duration(months=1), date(2000, 6, 30)),
        (
            datetime(2000, 3, 1, 0, 30),
            Duration(months=1, hours=12),
            datetime(2000, 1, 31, 12, 30),
        ),
    ],
)
def test_duration_rsub_date(start, dur, expected):
    r = start - dur
    assert r == expected
    assert type(r) is type(expected)


def test_duration_rsub_unsupported():
    assert Duration().__rsub__("x") is NotImplemented
    with pytest.raises(TypeError):
        "x" - Duration()
    with pytest.raises(TypeError):
        5 - Duration()


def test_duration_eq():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1, days=1) == Duration(months=12, hours=24)
    assert not (Duration(years=1) == Duration(months=11))
    assert not (Duration(years=1, days=1) == Duration(years=1, days=2))
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert not (Duration(days=1) == timedelta(days=2))
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert not (Duration(months=1, days=1) == timedelta(days=1))
    assert Duration() == timedelta(0)
    assert not (Duration(days=1) == "P1D")


def test_duration_ne():
    assert Duration(years=1) != Duration(months=11)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1, days=1) != Duration(years=1, days=2)
    assert not (Duration(years=1, days=1) != Duration(months=12, days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert Duration(months=1, days=1) != timedelta(days=1)
    assert timedelta(days=1) != Duration(years=1, days=1)
    assert Duration(days=1) != "P1D"


def test_duration_comparison_not_supported():
    with pytest.raises(TypeError):
        Duration(years=1) < Duration(years=2)


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        r = pickle.loads(pickle.dumps(d, proto))
        assert r == d
        assert r.years == d.years
        assert r.months == d.months
        assert r.tdelta == d.tdelta
    assert d.__getstate__() is d.__dict__
    e = Duration()
    e.__setstate__({"years": Decimal(5), "months": Decimal(1), "tdelta": timedelta(1)})
    assert e.years == 5
    assert e.tdelta == timedelta(1)


def test_duration_copy():
    import copy

    d = Duration(years=1, months=2, days=3)
    assert copy.copy(d) == d
    assert copy.deepcopy(d) == d


@pytest.mark.parametrize(
    "dur, start, expected",
    [
        (Duration(months=1), date(2000, 1, 1), timedelta(days=31)),
        (Duration(months=1), date(2000, 2, 1), timedelta(days=29)),
        (Duration(months=1), date(2001, 2, 1), timedelta(days=28)),
        (Duration(years=1), date(2000, 1, 1), timedelta(days=366)),
        (Duration(years=1), date(2001, 1, 1), timedelta(days=365)),
        (Duration(years=1, days=1), date(2001, 1, 1), timedelta(days=366)),
        (Duration(days=3, hours=4), datetime(2001, 1, 1), timedelta(days=3, hours=4)),
    ],
)
def test_duration_totimedelta_start(dur, start, expected):
    assert dur.totimedelta(start=start) == expected
    assert dur.totimedelta(start) == expected


@pytest.mark.parametrize(
    "dur, end, expected",
    [
        (Duration(months=1), date(2000, 3, 1), timedelta(days=29)),
        (Duration(months=1), date(2001, 3, 1), timedelta(days=28)),
        (Duration(months=1), date(2000, 2, 1), timedelta(days=31)),
        (Duration(years=1), date(2001, 1, 1), timedelta(days=366)),
        (Duration(years=1), date(2002, 1, 1), timedelta(days=365)),
    ],
)
def test_duration_totimedelta_end(dur, end, expected):
    assert dur.totimedelta(end=end) == expected


def test_duration_totimedelta_errors():
    with pytest.raises(ValueError, match="start or end required"):
        Duration(months=1).totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        Duration(months=1).totimedelta(date(2000, 1, 1), date(2000, 2, 1))


def test_duration_add_then_sub_roundtrip():
    base = date(2003, 1, 15)
    for months in range(-30, 31):
        d = Duration(months=months)
        assert (base + d) - d == base


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P18Y9M4DT11H9M8S", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P2W", timedelta(weeks=2)),
        ("P3Y6M4DT12H30M5S", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P23DT23H", timedelta(days=23, hours=23)),
        ("P4Y", Duration(years=4)),
        ("P1M", Duration(months=1)),
        ("PT1M", timedelta(minutes=1)),
        ("P0.5Y", Duration(years=Decimal("0.5"))),
        ("PT36H", timedelta(hours=36)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("+P11D", timedelta(days=11)),
        ("-P2W", timedelta(weeks=-2)),
        ("-P2.2W", timedelta(weeks=-2.2)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P1DT2H3M", timedelta(days=1, hours=2, minutes=3)),
        ("P1DT2H", timedelta(days=1, hours=2)),
        ("PT2H", timedelta(hours=2)),
        ("PT2.3H", timedelta(hours=2.3)),
        ("PT2H3M4S", timedelta(hours=2, minutes=3, seconds=4)),
        ("PT3M4S", timedelta(minutes=3, seconds=4)),
        ("PT22S", timedelta(seconds=22)),
        ("PT22.22S", timedelta(seconds=22.22)),
        ("PT22,22S", timedelta(seconds=22.22)),
        ("-P2Y", Duration(years=-2)),
        ("-P3Y6M4DT12H30M5S", -Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("-P1DT2H3M4S", -timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("P1Y2M", Duration(years=1, months=2)),
        ("P1,5Y", Duration(years=Decimal("1.5"))),
        ("P1.5M", Duration(months=Decimal("1.5"))),
        ("P1W2D", timedelta(weeks=1, days=2)),
        ("P1.5D", timedelta(days=1.5)),
        ("PT1.5M", timedelta(seconds=90)),
        ("P0Y0M1D", timedelta(days=1)),
    ],
)
def test_parse_duration(text, expected):
    result = parse_duration(text)
    assert result == expected
    assert type(result) is type(expected)


def test_parse_duration_negative_duration_components():
    d = parse_duration("-P1Y2M3D")
    assert isinstance(d, Duration)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == timedelta(days=-3)


def test_parse_duration_decimal_types():
    d = parse_duration("P1.25Y2.5M")
    assert isinstance(d.years, Decimal)
    assert d.years == Decimal("1.25")
    assert d.months == Decimal("2.5")


def test_parse_duration_not_as_timedelta():
    r = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)
    assert r.years == 0 and r.months == 0
    assert r.tdelta == timedelta(days=1)
    r = parse_duration("-PT1H", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)
    assert r.tdelta == timedelta(hours=-1)
    r = parse_duration("P1Y", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)


def test_parse_duration_minutes_vs_months():
    assert parse_duration("P1M") == Duration(months=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    r = parse_duration("P1MT1M")
    assert r.months == 1
    assert r.tdelta == timedelta(minutes=1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0018-09-04T11:09:08", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P00180904T110908", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P0003-06-04T12:30:05", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P0001-01-01T00:00:01.5", Duration(1, 1, 500000, months=1, years=1)),
    ],
)
def test_parse_duration_alternative(text, expected):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result == expected
    assert result.years == expected.years
    assert result.months == expected.months


def test_parse_duration_alternative_not_as_timedelta():
    r = parse_duration("P0018-09-04T11:09:08", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)
    assert r.years == 18
    assert r.months == 9
    assert r.tdelta == timedelta(days=4, hours=11, minutes=9, seconds=8)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "P",
        "1Y",
        "P1",
        "P1Y1Y",
        "P1D1Y",
        "PT1D",
        "P1H",
        "P1S",
        "-P",
        "P-1Y",
        "P1.Y",
        "P.5Y",
        "Q1Y",
        " P1Y",
        "P1Y ",
        "P1Y2M3DT4H5M6S7",
        "xP1D",
    ],
)
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_invalid_message():
    with pytest.raises(ISO8601Error, match="Unable to parse duration string"):
        parse_duration("garbage")


def test_parse_duration_alternative_bad():
    with pytest.raises(ISO8601Error):
        parse_duration("P0018-09-04")
    with pytest.raises(ValueError):
        parse_duration("P0000-00-00T00:00:00")


@pytest.mark.parametrize("value", [None, 1, 1.5, b"P1D", timedelta(1)])
def test_parse_duration_type_error(value):
    with pytest.raises(TypeError, match="Expecting a string"):
        parse_duration(value)


# ---------------------------------------------------------------------------
# duration_isoformat / strftime on durations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (Duration(4, 8, 0, 0, 9, 11, 0, 9, 18), "P18Y9M4DT11H9M8S"),
        (timedelta(weeks=2), "P14D"),
        (Duration(4, 5, 0, 0, 30, 12, 0, 6, 3), "P3Y6M4DT12H30M5S"),
        (timedelta(days=23, hours=23), "P23DT23H"),
        (Duration(years=4), "P4Y"),
        (Duration(months=1), "P1M"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(hours=2), "PT2H"),
        (timedelta(seconds=22), "PT22S"),
        (timedelta(seconds=22, microseconds=220000), "PT22.22S"),
        (timedelta(seconds=1, microseconds=1), "PT1.000001S"),
        (timedelta(microseconds=500000), "PT0.5S"),
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-2, minutes=-3), "-PT2H3M"),
        (-Duration(4, 5, 0, 0, 30, 12, 0, 6, 3), "-P3Y6M4DT12H30M5S"),
        (Duration(years=-1), "-P1Y"),
        (Duration(months=-1), "-P1M"),
        (Duration(days=-1), "-P1D"),
        (Duration(years=1, days=1), "P1Y1D"),
        (Duration(months=1, minutes=1), "P1MT1M"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
        (timedelta(days=1, seconds=1), "P1DT1S"),
        (timedelta(hours=1, seconds=1), "PT1H1S"),
        (timedelta(minutes=1, seconds=1), "PT1M1S"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


@pytest.mark.parametrize(
    "text",
    [
        "P18Y9M4DT11H9M8S",
        "P3Y6M4DT12H30M5S",
        "P23DT23H",
        "P4Y",
        "P1M",
        "PT1M",
        "PT22.5S",
        "-P1Y2M",
        "-P1DT2H",
        "P1Y1D",
        "P1MT1M",
    ],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


def test_duration_isoformat_week():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=20), D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(weeks=-3), D_WEEK) == "-P3W"
    assert duration_isoformat(Duration(weeks=5), D_WEEK) == "P5W"
    assert strftime(timedelta(weeks=-3), "%p") == "3W"


def test_duration_isoformat_alternative():
    d = Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)
    assert duration_isoformat(d, D_ALT_EXT) == "P0018-09-04T11:09:08"
    assert duration_isoformat(d, D_ALT_BAS) == "P00180904T110908"


def test_duration_isoformat_time_and_date_passthrough():
    # non-duration objects are passed to strftime and never get a sign
    assert duration_isoformat(date(2000, 1, 2), "%Y-%m-%d") == "2000-01-02"
    assert duration_isoformat(time(1, 2, 3), "%H:%M:%S") == "01:02:03"


def test_duration_isoformat_negative_mixed():
    # mixed: tdelta negative only
    d = Duration(years=0, months=0, days=-2)
    assert duration_isoformat(d) == "-P2D"
    d = Duration(years=1)
    d.tdelta = timedelta(days=-1)
    assert duration_isoformat(d).startswith("-")


def test_strftime_duration_codes():
    td = timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5)
    assert strftime(td, "%d") == "01"
    assert strftime(td, "%H") == "02"
    assert strftime(td, "%M") == "03"
    assert strftime(td, "%S") == "04"
    assert strftime(td, "%f") == "000005"
    assert strftime(td, "%%") == "%"
    assert strftime(timedelta(days=15), "%W") == "02"
    assert strftime(timedelta(days=-15), "%W") == "02"
    assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H:%M:%S") == "23:59:59"
    d = Duration(years=1985, months=4, days=12)
    assert strftime(d, "%Y") == "1985"
    assert strftime(d, "%m") == "04"
    assert strftime(d, "%C") == "19"
    assert strftime(d, "%Y", 6) == "+001985"
    assert strftime(d, "%C", 6) == "+0019"
    # unknown codes untouched
    assert strftime(td, "%j %w %z") == "%j %w %z"
    assert strftime(td, "%x") == "%x"


def test_strftime_P_rounding_of_microseconds():
    assert strftime(timedelta(seconds=5, microseconds=100), "%P") == "T5.0001S"
    assert strftime(timedelta(microseconds=10), "%P") == "T0.00001S"
    assert strftime(timedelta(seconds=10), "%P") == "T10S"
    assert strftime(timedelta(days=1), "%P") == "1D"
    assert strftime(timedelta(0), "%P") == "0D"
    assert strftime(timedelta(days=-1, hours=-1), "%P") == "1DT1H"


def test_strftime_P_duration_negative_parts_abs():
    d = Duration(years=-2, months=-3)
    assert strftime(d, "%P") == "2Y3M"


def test_strftime_time_codes():
    t = time(1, 2, 3, 4)
    assert strftime(t, "%H%M%S%f") == "010203000004"
    assert strftime(t, "%z") == ""
    assert strftime(t, "%Z") == ""
    assert strftime(t, "%h") == ""


def test_strftime_datetime_all_codes():
    dt = datetime(2004, 2, 3, 4, 5, 6, 7, tzinfo=FixedOffset(-1, -30))
    assert (
        strftime(dt, "%Y|%m|%d|%H|%M|%S|%f|%j|%w|%W|%C|%z|%Z|%h|%%")
        == "2004|02|03|04|05|06|000007|034|2|06|20|-0130|-01:30|-01|%"
    )
