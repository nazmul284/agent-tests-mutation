"""Test suite for the isodate package."""

import pickle
from datetime import date, datetime, time, timedelta
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
from isodate.tzinfo import ZERO, LocalTimezone, Utc


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


def test_iso8601error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)


def test_all_exports_exist():
    for name in isodate.__all__:
        assert hasattr(isodate, name)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "string, expected",
    [
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("1985", date(1985, 1, 1)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-102", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W15", date(1985, 4, 8)),
        ("1985-W15", date(1985, 4, 8)),
        ("1989-W15", date(1989, 4, 10)),
        ("1989-W15-5", date(1989, 4, 14)),
        ("1985-001", date(1985, 1, 1)),
        ("1985-365", date(1985, 12, 31)),
        ("2000-366", date(2000, 12, 31)),
        ("2000-060", date(2000, 2, 29)),
        ("2001-060", date(2001, 3, 1)),
        # ISO week 1 starting in previous year
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2015-W01-4", date(2015, 1, 1)),
        ("2016-W01-1", date(2016, 1, 4)),
        ("2018-W01-1", date(2018, 1, 1)),
        ("2018-W01-7", date(2018, 1, 7)),
        ("2018-W02-1", date(2018, 1, 8)),
        ("2020-W01-3", date(2020, 1, 1)),
        ("2020W013", date(2020, 1, 1)),
        ("2021-W01", date(2021, 1, 4)),
        ("2021-W52-7", date(2022, 1, 2)),
    ],
)
def test_parse_date(string, expected):
    assert parse_date(string) == expected


@pytest.mark.parametrize(
    "string, expected",
    [
        ("+0019", date(1901, 1, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+001985-04", date(1985, 4, 1)),
        ("+00198504", date(1985, 4, 1)),
        ("+001985-04-12", date(1985, 4, 12)),
        ("+0019850412", date(1985, 4, 12)),
        ("+001985102", date(1985, 4, 12)),
        ("+001985-102", date(1985, 4, 12)),
        ("+001985W155", date(1985, 4, 12)),
        ("+001985-W15-5", date(1985, 4, 12)),
        ("+001985W15", date(1985, 4, 8)),
        ("+001985-W15", date(1985, 4, 8)),
    ],
)
def test_parse_date_expanded_6_digits(string, expected):
    assert parse_date(string, 6) == expected


def test_parse_date_expanded_requires_sign_for_non_4_digits():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", 6)


def test_parse_date_expanded_4_digits():
    assert parse_date("+1985-04-12", 4, True) == date(1985, 4, 12)
    assert parse_date("+19", 4, True) == date(1901, 1, 1)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", 4, True)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")


def test_parse_date_negative_year_not_representable():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)
    with pytest.raises(ValueError):
        parse_date("-1985", 4, True)


def test_parse_date_negative_sign_applied():
    # negative sign must produce a negative year → invalid for datetime.date
    with pytest.raises(ValueError):
        parse_date("-0019", 4, True)
    with pytest.raises(ValueError):
        parse_date("-1985-W15", 4, True)
    with pytest.raises(ValueError):
        parse_date("-1985-102", 4, True)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)
    assert parse_date("198504", defaultmonth=7, defaultday=20) == date(1985, 4, 20)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    # defaults do not affect complete dates
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-102", defaultmonth=6, defaultday=15) == date(1985, 4, 12)
    assert parse_date("1985-W15", defaultmonth=6, defaultday=15) == date(1985, 4, 8)


@pytest.mark.parametrize(
    "string",
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
        "abcd",
        "1985-04-12x",
        "x1985",
    ],
)
def test_parse_date_invalid(string):
    with pytest.raises(ISO8601Error):
        parse_date(string)


def test_parse_date_invalid_values():
    with pytest.raises(ValueError):
        parse_date("1985-13-01")
    with pytest.raises(ValueError):
        parse_date("1985-02-30")


def test_parse_date_month_zero_falls_back_to_default_month():
    # current behaviour: a zero month is replaced by defaultmonth
    assert parse_date("1985-00-10") == date(1985, 1, 10)
    assert parse_date("1985-00-10", defaultmonth=5) == date(1985, 5, 10)


def test_parse_date_error_message():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 date format"):
        parse_date("garbage")


def test_parse_date_returns_date_not_datetime():
    result = parse_date("1985-04-12")
    assert type(result) is date


# ---------------------------------------------------------------------------
# date_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, fmt, expected",
    [
        (date(1985, 4, 12), DATE_EXT_COMPLETE, "1985-04-12"),
        (date(1985, 4, 12), DATE_BAS_COMPLETE, "19850412"),
        (date(1985, 4, 12), DATE_BAS_ORD_COMPLETE, "1985102"),
        (date(1985, 4, 12), DATE_EXT_ORD_COMPLETE, "1985-102"),
        (date(1985, 4, 12), DATE_BAS_WEEK_COMPLETE, "1985W155"),
        (date(1985, 4, 12), DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (date(1985, 4, 12), DATE_BAS_WEEK, "1985W15"),
        (date(1985, 4, 12), DATE_EXT_WEEK, "1985-W15"),
        (date(1985, 4, 12), DATE_BAS_MONTH, "198504"),
        (date(1985, 4, 12), DATE_EXT_MONTH, "1985-04"),
        (date(1985, 4, 12), DATE_YEAR, "1985"),
        (date(1985, 4, 12), DATE_CENTURY, "19"),
        (date(2000, 1, 1), DATE_CENTURY, "20"),
        (date(1, 1, 1), DATE_EXT_COMPLETE, "0001-01-01"),
        (date(2000, 12, 31), DATE_EXT_ORD_COMPLETE, "2000-366"),
        (date(2001, 1, 1), DATE_EXT_ORD_COMPLETE, "2001-001"),
        (date(2009, 12, 31), DATE_EXT_WEEK_COMPLETE, "2009-W53-4"),
        # %Y renders the calendar year, not the ISO week-numbering year
        (date(2010, 1, 3), DATE_EXT_WEEK_COMPLETE, "2010-W53-7"),
        (date(2008, 12, 29), DATE_EXT_WEEK_COMPLETE, "2008-W01-1"),
        (date(2018, 1, 7), DATE_EXT_WEEK_COMPLETE, "2018-W01-7"),
        (date(2018, 1, 1), DATE_EXT_WEEK_COMPLETE, "2018-W01-1"),
    ],
)
def test_date_isoformat(d, fmt, expected):
    assert date_isoformat(d, fmt) == expected


def test_date_isoformat_default():
    assert date_isoformat(date(1985, 4, 12)) == "1985-04-12"


@pytest.mark.parametrize(
    "d, fmt, expected",
    [
        (date(1985, 4, 12), DATE_EXT_COMPLETE, "+001985-04-12"),
        (date(1985, 4, 12), DATE_BAS_COMPLETE, "+0019850412"),
        (date(1985, 4, 12), DATE_YEAR, "+001985"),
        (date(1985, 4, 12), DATE_CENTURY, "+0019"),
        (date(1985, 4, 12), DATE_EXT_WEEK_COMPLETE, "+001985-W15-5"),
    ],
)
def test_date_isoformat_expanded(d, fmt, expected):
    assert date_isoformat(d, fmt, 6) == expected


@pytest.mark.parametrize(
    "string, fmt",
    [
        ("1985-04-12", DATE_EXT_COMPLETE),
        ("19850412", DATE_BAS_COMPLETE),
        ("1985-102", DATE_EXT_ORD_COMPLETE),
        ("1985102", DATE_BAS_ORD_COMPLETE),
        ("1985-W15-5", DATE_EXT_WEEK_COMPLETE),
        ("1985W155", DATE_BAS_WEEK_COMPLETE),
        ("2009-W53-4", DATE_EXT_WEEK_COMPLETE),
        ("1985-W15", DATE_EXT_WEEK),
        ("1985-04", DATE_EXT_MONTH),
        ("1985", DATE_YEAR),
    ],
)
def test_date_roundtrip_formats(string, fmt):
    assert date_isoformat(parse_date(string), fmt) == string


def test_date_roundtrip_every_day_of_leap_year():
    d = date(2020, 1, 1)
    while d.year == 2020:
        for fmt in (
            DATE_EXT_COMPLETE,
            DATE_BAS_COMPLETE,
            DATE_EXT_ORD_COMPLETE,
            DATE_BAS_ORD_COMPLETE,
            DATE_EXT_WEEK_COMPLETE,
            DATE_BAS_WEEK_COMPLETE,
        ):
            s = date_isoformat(d, fmt)
            if fmt in (DATE_EXT_WEEK_COMPLETE, DATE_BAS_WEEK_COMPLETE):
                # week-year may differ from calendar year for %Y; only check
                # when they agree.
                if d.isocalendar()[0] != d.year:
                    continue
            assert parse_date(s) == d, (fmt, s)
        d += timedelta(days=1)


@pytest.mark.parametrize("year", range(2000, 2030))
def test_week_date_parsing_matches_fromisocalendar(year):
    for week in (1, 2, 10, 52):
        for day in (1, 4, 7):
            expected = date.fromisocalendar(year, week, day)
            assert parse_date("%04d-W%02d-%d" % (year, week, day)) == expected
            assert parse_date("%04dW%02d%d" % (year, week, day)) == expected
        assert parse_date("%04d-W%02d" % (year, week)) == date.fromisocalendar(year, week, 1)


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "string, expected",
    [
        ("232050", time(23, 20, 50)),
        ("23:20:50", time(23, 20, 50)),
        ("2320", time(23, 20)),
        ("23:20", time(23, 20)),
        ("23", time(23)),
        ("T232050", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("T2320", time(23, 20)),
        ("T23:20", time(23, 20)),
        ("T23", time(23)),
        ("232050,5", time(23, 20, 50, 500000)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.1234567", time(23, 20, 50, 123456)),
        ("23:20:50.1234569", time(23, 20, 50, 123456)),
        ("15:33:42.123456789", time(15, 33, 42, 123456)),
        ("2320,8", time(23, 20, 48)),
        ("23:20,8", time(23, 20, 48)),
        ("23:20.5", time(23, 20, 30)),
        ("23,3", time(23, 18)),
        ("23.5", time(23, 30)),
        ("12.25", time(12, 15)),
        ("00.0001", time(0, 0, 0, 360000)),
        ("00:00.001", time(0, 0, 0, 60000)),
        ("00:00:00", time(0, 0, 0)),
        ("00", time(0)),
        ("14:00:00.000001", time(14, 0, 0, 1)),
        ("12:30:00.9999999", time(12, 30, 0, 999999)),
    ],
)
def test_parse_time_naive(string, expected):
    result = parse_time(string)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "string, expected_offset",
    [
        ("152746+0100", timedelta(hours=1)),
        ("15:27:46+01:00", timedelta(hours=1)),
        ("15:27:46+01", timedelta(hours=1)),
        ("15:27:46-05:00", timedelta(hours=-5)),
        ("152746-0500", timedelta(hours=-5)),
        ("152746-05", timedelta(hours=-5)),
        ("15:27:46+05:30", timedelta(hours=5, minutes=30)),
        ("15:27:46-05:30", timedelta(hours=-5, minutes=-30)),
        ("15:27:46-0045", timedelta(minutes=-45)),
        ("1527+0100", timedelta(hours=1)),
        ("15:27+01:00", timedelta(hours=1)),
        ("15+0100", timedelta(hours=1)),
        ("15.5-02", timedelta(hours=-2)),
        ("15:27:46Z", ZERO),
        ("152746Z", ZERO),
        ("15Z", ZERO),
        ("15:27Z", ZERO),
    ],
)
def test_parse_time_tz(string, expected_offset):
    result = parse_time(string)
    assert result.tzinfo is not None
    assert result.utcoffset() == expected_offset


def test_parse_time_tz_values():
    t = parse_time("15:27:46.25+05:30")
    assert (t.hour, t.minute, t.second, t.microsecond) == (15, 27, 46, 250000)
    assert t.tzinfo.tzname(None) == "+05:30"
    t = parse_time("15:27:46Z")
    assert t.tzinfo is UTC
    t = parse_time("15.5-02")
    assert (t.hour, t.minute, t.second) == (15, 30, 0)


@pytest.mark.parametrize(
    "string",
    [
        "",
        "2",
        "232",
        "23:2",
        "23:20:5",
        "23-20-50",
        "232050+1",
        "23:20:50+01:0",
        "23:20:50 ",
        "23:20:50X",
        "T",
        "TT23",
        "23:20:50.",
        "23.",
        "ab:cd",
    ],
)
def test_parse_time_invalid(string):
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 time format"):
        parse_time(string)


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("25:00:00")
    with pytest.raises(ValueError):
        parse_time("23:60:00")


# ---------------------------------------------------------------------------
# time_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, fmt, expected",
    [
        (time(23, 20, 50), TIME_EXT_COMPLETE, "23:20:50"),
        (time(23, 20, 50), TIME_BAS_COMPLETE, "232050"),
        (time(23, 20, 50), TIME_BAS_MINUTE, "2320"),
        (time(23, 20, 50), TIME_EXT_MINUTE, "23:20"),
        (time(23, 20, 50), TIME_HOUR, "23"),
        (time(1, 2, 3), TIME_EXT_COMPLETE, "01:02:03"),
        (time(1, 2, 3, 4), TIME_EXT_COMPLETE + ".%f", "01:02:03.000004"),
        (time(1, 2, 3, 123456), "%f", "123456"),
        (time(1, 2, 3, tzinfo=UTC), TIME_EXT_COMPLETE + TZ_EXT, "01:02:03Z"),
        (time(1, 2, 3, tzinfo=UTC), TIME_BAS_COMPLETE + TZ_BAS, "010203Z"),
        (time(1, 2, 3, tzinfo=UTC), TIME_HOUR + TZ_HOUR, "01Z"),
        (
            time(1, 2, 3, tzinfo=FixedOffset(1, 0, "+01:00")),
            TIME_EXT_COMPLETE + TZ_EXT,
            "01:02:03+01:00",
        ),
        (
            time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x")),
            TIME_BAS_COMPLETE + TZ_BAS,
            "010203+0530",
        ),
        (
            time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x")),
            TIME_EXT_COMPLETE + TZ_EXT,
            "01:02:03-05:30",
        ),
        (time(1, 2, 3, tzinfo=FixedOffset(-5, 0, "x")), TIME_HOUR + TZ_HOUR, "01-05"),
        (time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x")), TZ_HOUR, "-05"),
        (time(1, 2, 3), TIME_EXT_COMPLETE + TZ_EXT, "01:02:03"),
    ],
)
def test_time_isoformat(t, fmt, expected):
    assert time_isoformat(t, fmt) == expected


def test_time_isoformat_default():
    assert time_isoformat(time(23, 20, 50)) == "23:20:50"
    assert time_isoformat(time(23, 20, 50, tzinfo=FixedOffset(1, 0, "x"))) == "23:20:50+01:00"


@pytest.mark.parametrize(
    "string, fmt",
    [
        ("232050", TIME_BAS_COMPLETE),
        ("23:20:50", TIME_EXT_COMPLETE),
        ("2320", TIME_BAS_MINUTE),
        ("23:20", TIME_EXT_MINUTE),
        ("23", TIME_HOUR),
        ("15:27:46+01:00", TIME_EXT_COMPLETE + TZ_EXT),
        ("15:27:46-05:30", TIME_EXT_COMPLETE + TZ_EXT),
        ("152746+0100", TIME_BAS_COMPLETE + TZ_BAS),
        ("152746-0530", TIME_BAS_COMPLETE + TZ_BAS),
        ("15:27:46Z", TIME_EXT_COMPLETE + TZ_EXT),
        ("15-05", TIME_HOUR + TZ_HOUR),
    ],
)
def test_time_roundtrip(string, fmt):
    assert time_isoformat(parse_time(string), fmt) == string


# ---------------------------------------------------------------------------
# parse_datetime / datetime_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "string, expected",
    [
        ("19850412T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985102T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985W155T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-W15-5T10:15", datetime(1985, 4, 12, 10, 15)),
        ("2012-10-30T08:55:22.1234567", datetime(2012, 10, 30, 8, 55, 22, 123456)),
        ("2012-10-30T08:55", datetime(2012, 10, 30, 8, 55)),
        ("2012-10-30T08", datetime(2012, 10, 30, 8)),
        ("1985-04-12T23:20:50.52", datetime(1985, 4, 12, 23, 20, 50, 520000)),
    ],
)
def test_parse_datetime_naive(string, expected):
    result = parse_datetime(string)
    assert result == expected
    assert result.tzinfo is None
    assert type(result) is datetime


def test_parse_datetime_tz():
    dt = parse_datetime("1985-04-12T10:15:30Z")
    assert dt == datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
    assert dt.tzinfo is UTC
    dt = parse_datetime("1985-04-12T10:15:30+04:00")
    assert dt.utcoffset() == timedelta(hours=4)
    assert dt == datetime(1985, 4, 12, 6, 15, 30, tzinfo=UTC)
    dt = parse_datetime("19850412T101530-0230")
    assert dt.utcoffset() == timedelta(hours=-2, minutes=-30)
    assert dt == datetime(1985, 4, 12, 12, 45, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    "string",
    ["1985-04-12", "1985-04-12 10:15", "1985-04-12T10:15T", "T10:15", "1985-04-12T", "xxxxTyy"],
)
def test_parse_datetime_invalid(string):
    with pytest.raises(ISO8601Error):
        parse_datetime(string)


def test_parse_datetime_missing_t_message():
    with pytest.raises(ISO8601Error, match="time designator 'T' missing"):
        parse_datetime("1985-04-12 10:15")


@pytest.mark.parametrize(
    "dt, fmt, expected",
    [
        (datetime(1985, 4, 12, 10, 15, 30), DT_EXT_COMPLETE, "1985-04-12T10:15:30"),
        (datetime(1985, 4, 12, 10, 15, 30), DT_BAS_COMPLETE, "19850412T101530"),
        (datetime(1985, 4, 12, 10, 15, 30), DT_EXT_ORD_COMPLETE, "1985-102T10:15:30"),
        (datetime(1985, 4, 12, 10, 15, 30), DT_BAS_ORD_COMPLETE, "1985102T101530"),
        (datetime(1985, 4, 12, 10, 15, 30), DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30"),
        (datetime(1985, 4, 12, 10, 15, 30), DT_BAS_WEEK_COMPLETE, "1985W155T101530"),
        (
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC),
            DT_EXT_COMPLETE,
            "1985-04-12T10:15:30Z",
        ),
        (
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "x")),
            DT_EXT_COMPLETE,
            "1985-04-12T10:15:30+04:00",
        ),
        (
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "x")),
            DT_BAS_COMPLETE,
            "19850412T101530+0400",
        ),
        (
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-2, -30, "x")),
            DT_BAS_COMPLETE,
            "19850412T101530-0230",
        ),
        (
            datetime(2012, 10, 30, 8, 55, 22, 123456),
            DT_EXT_COMPLETE + ".%f",
            "2012-10-30T08:55:22.123456",
        ),
    ],
)
def test_datetime_isoformat(dt, fmt, expected):
    assert datetime_isoformat(dt, fmt) == expected


def test_datetime_isoformat_default():
    assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30)) == "1985-04-12T10:15:30"
    assert (
        datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-1, 0, "x")))
        == "1985-04-12T10:15:30-01:00"
    )


@pytest.mark.parametrize(
    "string",
    [
        "1985-04-12T10:15:30",
        "1985-04-12T10:15:30Z",
        "1985-04-12T10:15:30+04:00",
        "1985-04-12T10:15:30-07:30",
    ],
)
def test_datetime_roundtrip(string):
    assert datetime_isoformat(parse_datetime(string)) == string


# ---------------------------------------------------------------------------
# tzinfo / parse_tzinfo / tz_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "string, offset",
    [
        ("Z", ZERO),
        ("+01:00", timedelta(hours=1)),
        ("+0100", timedelta(hours=1)),
        ("+01", timedelta(hours=1)),
        ("-01:00", timedelta(hours=-1)),
        ("-0130", timedelta(hours=-1, minutes=-30)),
        ("-01:30", timedelta(hours=-1, minutes=-30)),
        ("+12:45", timedelta(hours=12, minutes=45)),
        ("-12", timedelta(hours=-12)),
        ("+00:00", ZERO),
    ],
)
def test_parse_tzinfo(string, offset):
    tz = parse_tzinfo(string)
    assert tz.utcoffset(None) == offset
    assert tz.dst(None) == ZERO


def test_parse_tzinfo_z_is_utc_singleton():
    assert parse_tzinfo("Z") is UTC


def test_parse_tzinfo_name():
    assert parse_tzinfo("+01:00").tzname(None) == "+01:00"
    assert parse_tzinfo("-0130").tzname(None) == "-0130"


def test_parse_tzinfo_empty_is_none():
    assert parse_tzinfo("") is None


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("+03:15", "+", 3, 15)
    assert tz.utcoffset(None) == timedelta(hours=3, minutes=15)
    assert tz.tzname(None) == "+03:15"
    tz = build_tzinfo("-03:15", "-", 3, 15)
    assert tz.utcoffset(None) == timedelta(hours=-3, minutes=-15)
    tz = build_tzinfo("x", tzhour=2)
    assert tz.utcoffset(None) == timedelta(hours=2)


def test_tz_isoformat():
    naive = datetime(2000, 1, 1)
    assert tz_isoformat(naive) == ""
    assert tz_isoformat(naive, "%z") == ""
    utc = datetime(2000, 1, 1, tzinfo=UTC)
    assert tz_isoformat(utc) == "Z"
    assert tz_isoformat(utc, "%z") == "Z"
    assert tz_isoformat(utc, "%h") == "Z"
    zero_fixed = datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0, "UTC"))
    assert tz_isoformat(zero_fixed) == "Z"
    pos = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(pos) == "+05:30"
    assert tz_isoformat(pos, "%Z") == "+05:30"
    assert tz_isoformat(pos, "%z") == "+0530"
    assert tz_isoformat(pos, "%h") == "+05"
    neg = datetime(2000, 1, 1, tzinfo=FixedOffset(-11, -15, "x"))
    assert tz_isoformat(neg) == "-11:15"
    assert tz_isoformat(neg, "%z") == "-1115"
    assert tz_isoformat(neg, "%h") == "-11"
    small_neg = datetime(2000, 1, 1, tzinfo=FixedOffset(0, -1, "x"))
    assert tz_isoformat(small_neg) == "-00:01"
    big = datetime(2000, 1, 1, tzinfo=FixedOffset(23, 59, "x"))
    assert tz_isoformat(big) == "+23:59"


def test_tz_isoformat_unknown_format():
    with pytest.raises(ValueError, match="unknown format string"):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%X")


def test_tz_isoformat_large_offset():
    class Huge(FixedOffset):
        def utcoffset(self, dt):
            return timedelta(hours=100)

    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=Huge()))

    class Big(FixedOffset):
        def utcoffset(self, dt):
            return timedelta(hours=99)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Big())) == "+99:00"


def test_tz_isoformat_utcoffset_none():
    class NoneTZ(FixedOffset):
        def utcoffset(self, dt):
            return None

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=NoneTZ())) == ""


def test_tz_isoformat_zero_offset_with_dst_not_z():
    class ZeroWithDst(FixedOffset):
        def utcoffset(self, dt):
            return ZERO

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=ZeroWithDst())) == "+00:00"


def test_utc():
    assert UTC.utcoffset(None) == ZERO
    assert UTC.dst(None) == ZERO
    assert UTC.tzname(None) == "UTC"
    assert isinstance(UTC, Utc)


def test_utc_pickle_singleton():
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    dt = datetime(2000, 1, 1, tzinfo=UTC)
    assert pickle.loads(pickle.dumps(dt)).tzinfo is UTC


def test_fixed_offset():
    fo = FixedOffset()
    assert fo.utcoffset(None) == ZERO
    assert fo.tzname(None) == "UTC"
    assert fo.dst(None) == ZERO
    fo = FixedOffset(-3, -30, "NST")
    assert fo.utcoffset(None) == timedelta(hours=-3, minutes=-30)
    assert fo.tzname(None) == "NST"
    assert fo.dst(datetime(2000, 1, 1)) == ZERO
    assert repr(fo) == "<FixedOffset 'NST'>"
    assert FixedOffset(2).utcoffset(None) == timedelta(hours=2)
    assert FixedOffset(0, 45).utcoffset(None) == timedelta(minutes=45)


def test_local_timezone():
    import time as _time

    assert isinstance(LOCAL, LocalTimezone)
    dt = datetime(2020, 1, 15, 12, 0)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert LOCAL.dst(dt) in (ZERO, isodate.tzinfo.DSTDIFF)
    assert LOCAL.tzname(dt) in _time.tzname
    # offset matches the platform's conversion
    stamp = _time.mktime(dt.timetuple())
    expected = datetime.fromtimestamp(stamp) - datetime.fromtimestamp(stamp, UTC).replace(tzinfo=None)
    assert off == expected
    # dst consistent with utcoffset
    assert LOCAL.utcoffset(dt) - LOCAL.dst(dt) == isodate.tzinfo.STDOFFSET
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


def test_local_timezone_summer():
    import time as _time

    dt = datetime(2020, 7, 15, 12, 0)
    stamp = _time.mktime(dt.timetuple())
    expected = datetime.fromtimestamp(stamp) - datetime.fromtimestamp(stamp, UTC).replace(tzinfo=None)
    assert LOCAL.utcoffset(dt) == expected
    assert LOCAL.utcoffset(dt) - LOCAL.dst(dt) == isodate.tzinfo.STDOFFSET
    isdst = _time.localtime(stamp).tm_isdst > 0
    assert LOCAL.tzname(dt) == _time.tzname[isdst]


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val, low, high, expected",
    [
        (Decimal(1), 1, 13, (0, Decimal(1))),
        (Decimal(12), 1, 13, (0, Decimal(12))),
        (Decimal(13), 1, 13, (1, Decimal(1))),
        (Decimal(14), 1, 13, (1, Decimal(2))),
        (Decimal(25), 1, 13, (2, Decimal(1))),
        (Decimal(0), 1, 13, (-1, Decimal(12))),
        (Decimal(-1), 1, 13, (-1, Decimal(11))),
        (Decimal(-11), 1, 13, (-1, Decimal(1))),
        (Decimal(-12), 1, 13, (-2, Decimal(12))),
        (Decimal(7), 0, 5, (1, Decimal(2))),
        (Decimal(-7), 0, 5, (-2, Decimal(3))),
        (Decimal("2.5"), 0, 2, (1, Decimal("0.5"))),
    ],
)
def test_fquotmod(val, low, high, expected):
    assert fquotmod(val, low, high) == expected
    div, mod = fquotmod(val, low, high)
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
        (2004, 2, 29),
        (1900, 2, 28),
        (2000, 2, 29),
        (2100, 2, 28),
        (2400, 2, 29),
        (1996, 2, 29),
        (1997, 2, 28),
    ],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


def test_max_days_in_month_matches_calendar():
    import calendar

    for year in range(1890, 2110):
        for month in range(1, 13):
            assert max_days_in_month(year, month) == calendar.monthrange(year, month)[1]


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


def test_duration_init_defaults():
    d = Duration()
    assert d.years == 0
    assert d.months == 0
    assert d.tdelta == timedelta(0)
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)


def test_duration_init_components():
    d = Duration(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.months == Decimal(8)
    assert d.years == Decimal(9)
    d = Duration(months=1.5, years=Decimal("2.25"))
    assert d.months == Decimal("1.5")
    assert d.years == Decimal("2.25")


def test_duration_init_keyword_mapping():
    assert Duration(days=1).tdelta == timedelta(days=1)
    assert Duration(seconds=1).tdelta == timedelta(seconds=1)
    assert Duration(microseconds=1).tdelta == timedelta(microseconds=1)
    assert Duration(milliseconds=1).tdelta == timedelta(milliseconds=1)
    assert Duration(minutes=1).tdelta == timedelta(minutes=1)
    assert Duration(hours=1).tdelta == timedelta(hours=1)
    assert Duration(weeks=1).tdelta == timedelta(weeks=1)


def test_duration_getattr_delegates():
    d = Duration(days=3, seconds=5, microseconds=7, years=1)
    assert d.days == 3
    assert d.seconds == 5
    assert d.microseconds == 7
    assert d.total_seconds() == timedelta(days=3, seconds=5, microseconds=7).total_seconds()
    with pytest.raises(AttributeError):
        d.nonexistent_attribute


def test_duration_str():
    assert str(Duration(years=1, months=1)) == "1 years, 1 month, 0:00:00"
    assert str(Duration(years=1, months=2)) == "1 years, 2 months, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=Decimal("0.5"))) == "0.5 month, 0:00:00"
    assert str(Duration(months=-2)) == "-2 month, 0:00:00"
    assert str(Duration(days=1, hours=2)) == "1 day, 2:00:00"
    assert str(Duration(years=3, days=2)) == "3 years, 2 days, 0:00:00"
    assert str(Duration()) == "0:00:00"


def test_duration_repr():
    assert repr(Duration(1, 2, 3, years=4, months=5)) == (
        "isodate.duration.Duration(1, 2, 3, years=4, months=5)"
    )
    assert repr(Duration()) == "isodate.duration.Duration(0, 0, 0, years=0, months=0)"


def test_duration_hash():
    d1 = Duration(days=1, months=2, years=3)
    d2 = Duration(days=1, months=2, years=3)
    assert hash(d1) == hash(d2)
    assert len({d1, d2}) == 1
    d3 = Duration(days=1, months=3, years=3)
    assert len({d1, d3}) == 2
    assert hash(d1) == hash((timedelta(days=1), Decimal(2), Decimal(3)))
    assert {d1: "x"}[d2] == "x"


def test_duration_neg():
    d = Duration(days=1, hours=2, months=3, years=4)
    n = -d
    assert isinstance(n, Duration)
    assert n.years == -4
    assert n.months == -3
    assert n.tdelta == -timedelta(days=1, hours=2)
    assert -n == d
    # original unchanged
    assert d.years == 4


def test_duration_add_duration():
    d = Duration(days=1, months=2, years=3) + Duration(hours=5, months=4, years=1)
    assert isinstance(d, Duration)
    assert d.years == 4
    assert d.months == 6
    assert d.tdelta == timedelta(days=1, hours=5)


def test_duration_add_timedelta():
    d = Duration(days=1, months=2, years=3) + timedelta(hours=5)
    assert isinstance(d, Duration)
    assert (d.years, d.months, d.tdelta) == (3, 2, timedelta(days=1, hours=5))
    d = timedelta(hours=5) + Duration(days=1, months=2, years=3)
    assert isinstance(d, Duration)
    assert (d.years, d.months, d.tdelta) == (3, 2, timedelta(days=1, hours=5))


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) + 1
    with pytest.raises(TypeError):
        "x" + Duration(days=1)
    assert Duration(days=1).__add__("x") is NotImplemented


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 1, 15), Duration(months=1), date(2000, 2, 15)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2000, 1, 15), Duration(months=12), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(2001, 2, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 15), Duration(months=-13), date(1998, 12, 15)),
        (date(2000, 1, 15), Duration(years=1, months=1, days=1), date(2001, 2, 16)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (date(2000, 5, 31), Duration(months=1), date(2000, 6, 30)),
        (
            datetime(2000, 1, 31, 12, 30),
            Duration(months=1, hours=12),
            datetime(2000, 3, 1, 0, 30),
        ),
        (
            datetime(2000, 1, 15, 12, tzinfo=UTC),
            Duration(years=2, minutes=5),
            datetime(2002, 1, 15, 12, 5, tzinfo=UTC),
        ),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected
    assert type(start + dur) is type(start)


def test_duration_add_date_preserves_tz():
    tz = FixedOffset(3, 0, "x")
    result = datetime(2000, 1, 1, tzinfo=tz) + Duration(months=1)
    assert result.tzinfo is tz


def test_duration_add_fractional_to_date_raises():
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) + Duration(months=1.5)
    with pytest.raises(ValueError, match="fractional"):
        Duration(years=0.5) + date(2000, 1, 1)
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) - Duration(months=1.5)
    with pytest.raises(ValueError, match="fractional"):
        date(2000, 1, 1) - Duration(years=1.5)


def test_duration_add_integral_decimal_ok():
    assert date(2000, 1, 1) + Duration(months=Decimal("2.0")) == date(2000, 3, 1)
    assert date(2000, 1, 1) + Duration(years=2.0) == date(2002, 1, 1)


def test_duration_mul():
    d = Duration(days=1, hours=1, months=2, years=3)
    for r in (d * 3, 3 * d):
        assert isinstance(r, Duration)
        assert r.years == 9
        assert r.months == 6
        assert r.tdelta == timedelta(days=3, hours=3)
    z = d * 0
    assert z == Duration()
    neg = d * -1
    assert neg == -d
    with pytest.raises(TypeError):
        d * 1.5
    with pytest.raises(TypeError):
        d * "x"
    assert d.__mul__(1.5) is NotImplemented


def test_duration_sub_duration():
    d = Duration(days=3, months=5, years=4) - Duration(days=1, months=2, years=1)
    assert isinstance(d, Duration)
    assert (d.years, d.months, d.tdelta) == (3, 3, timedelta(days=2))


def test_duration_sub_timedelta():
    d = Duration(days=3, months=5, years=4) - timedelta(days=1)
    assert isinstance(d, Duration)
    assert (d.years, d.months, d.tdelta) == (4, 5, timedelta(days=2))


def test_duration_rsub_timedelta():
    d = timedelta(days=5) - Duration(days=1, months=2, years=3)
    assert isinstance(d, Duration)
    assert d.years == -3
    assert d.months == -2
    assert d.tdelta == timedelta(days=4)


def test_duration_sub_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) - 1
    with pytest.raises(TypeError):
        Duration(days=1) - date(2000, 1, 1)
    assert Duration(days=1).__sub__("x") is NotImplemented
    assert Duration(days=1).__rsub__(5) is NotImplemented
    with pytest.raises(TypeError):
        5 - Duration(days=1)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=1), date(2000, 11, 15)),
        (date(2000, 2, 15), Duration(months=1), date(2000, 1, 15)),
        (date(2000, 1, 15), Duration(months=13), date(1998, 12, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(2000, 2, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2004, 2, 29), Duration(years=4), date(2000, 2, 29)),
        (date(2000, 3, 1), Duration(months=1, days=1), date(2000, 1, 31)),
        (
            datetime(2000, 3, 1, 0, 30),
            Duration(months=1, hours=12),
            datetime(2000, 1, 31, 12, 30),
        ),
        (
            datetime(2002, 1, 15, 12, 5, tzinfo=UTC),
            Duration(years=2, minutes=5),
            datetime(2000, 1, 15, 12, tzinfo=UTC),
        ),
    ],
)
def test_date_sub_duration(start, dur, expected):
    result = start - dur
    assert result == expected
    assert type(result) is type(start)


def test_date_minus_duration_equals_plus_negated_duration():
    starts = [date(2000, 1, 31), date(2001, 7, 15), date(1999, 12, 31), date(2004, 2, 29)]
    durs = [Duration(months=m, years=y) for m in range(-14, 15, 3) for y in (-1, 0, 2)]
    for s in starts:
        for d in durs:
            assert s - d == s + (-d)


def test_duration_add_then_sub_month_ranges():
    for month in range(1, 13):
        start = date(2001, month, 15)
        for n in range(-25, 26):
            d = Duration(months=n)
            end = start + d
            total = (2001 * 12 + month - 1) + n
            assert (end.year, end.month, end.day) == (total // 12, total % 12 + 1, 15)
            assert end - d == start


def test_duration_eq():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(days=1) == Duration(hours=24)
    assert not (Duration(years=1) == Duration(months=11))
    assert not (Duration(days=1) == Duration(days=2))
    assert not (Duration(days=1, months=1) == Duration(days=2, months=1))
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert not (Duration(days=1) == timedelta(days=2))
    assert not (Duration(days=1, months=1) == timedelta(days=1))
    assert not (Duration(days=1, years=1) == timedelta(days=1))
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == Duration(days=1)
    assert not (Duration(days=1) == "x")


def test_duration_ne():
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) != Duration(days=2)
    assert Duration(days=1, months=1) != Duration(days=2, months=1)
    assert not (Duration(days=1, months=1) != Duration(days=1, months=1))
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(days=1, months=1) != timedelta(days=1)
    assert Duration(days=1, years=1) != timedelta(days=1)
    assert Duration(days=1) != "x"
    assert Duration(months=1) != "x"


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(end=date(2000, 1, 1)) == timedelta(days=365)
    d = Duration(days=1, hours=2)
    assert d.totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=1, hours=2)
    assert d.totimedelta(end=datetime(2000, 1, 1)) == timedelta(days=1, hours=2)
    with pytest.raises(ValueError, match="start or end required"):
        d.totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 1))


def test_duration_pickle():
    d = Duration(days=1, seconds=2, months=3, years=4)
    p = pickle.loads(pickle.dumps(d))
    assert p == d
    assert p.years == 4
    assert p.months == 3
    assert p.tdelta == timedelta(days=1, seconds=2)
    assert d.__getstate__() == d.__dict__


def test_duration_setstate():
    d = Duration()
    d.__setstate__({"years": Decimal(5), "months": Decimal(1), "tdelta": timedelta(1)})
    assert d.years == 5
    assert d.months == 1
    assert d.tdelta == timedelta(1)


def test_duration_copy():
    import copy

    d = Duration(days=1, months=2, years=3)
    assert copy.copy(d) == d
    assert copy.deepcopy(d) == d


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "string, expected",
    [
        ("P18Y9M4DT11H9M8S", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P2W", timedelta(weeks=2)),
        ("P3Y6M4DT12H30M5S", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P23DT23H", timedelta(hours=23, days=23)),
        ("P4Y", Duration(years=4)),
        ("P1M", Duration(months=1)),
        ("PT1M", timedelta(minutes=1)),
        ("P0.5Y", Duration(years=0.5)),
        ("P0,5Y", Duration(years=0.5)),
        ("P1,5M", Duration(months=1.5)),
        ("PT36H", timedelta(hours=36)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("+P11D", timedelta(days=11)),
        ("-P2W", timedelta(weeks=-2)),
        ("-P2.2W", timedelta(weeks=-2.2)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P1DT2H3S", timedelta(days=1, hours=2, seconds=4 - 1)),
        ("P1DT2H", timedelta(days=1, hours=2)),
        ("P1DT2M", timedelta(days=1, minutes=2)),
        ("P1D", timedelta(days=1)),
        ("PT2H", timedelta(hours=2)),
        ("PT2.5H", timedelta(hours=2.5)),
        ("PT2,5H", timedelta(hours=2.5)),
        ("PT1.5M", timedelta(minutes=1, seconds=30)),
        ("PT1.5S", timedelta(seconds=1.5)),
        ("PT0.000001S", timedelta(microseconds=1)),
        ("PT36H", timedelta(days=1, hours=12)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("P1W1D", timedelta(days=8)),
        ("-P1DT1H", timedelta(days=-1, hours=-1)),
        ("-PT1S", timedelta(seconds=-1)),
        ("P1Y2M3W4D", Duration(years=1, months=2, weeks=3, days=4)),
    ],
)
def test_parse_duration(string, expected):
    result = parse_duration(string)
    assert result == expected
    assert type(result) is type(expected)


def test_parse_duration_components():
    d = parse_duration("P3Y6M4DT12H30M5S")
    assert isinstance(d, Duration)
    assert d.years == 3
    assert d.months == 6
    assert d.tdelta == timedelta(days=4, hours=12, minutes=30, seconds=5)
    assert isinstance(d.years, Decimal)
    assert isinstance(d.months, Decimal)
    d = parse_duration("P0.5Y1,25M")
    assert d.years == Decimal("0.5")
    assert d.months == Decimal("1.25")


def test_parse_duration_negative_duration():
    d = parse_duration("-P3Y6M4DT12H30M5S")
    assert isinstance(d, Duration)
    assert d.years == -3
    assert d.months == -6
    assert d.tdelta == -timedelta(days=4, hours=12, minutes=30, seconds=5)
    d = parse_duration("-P1Y")
    assert d.years == -1
    assert d.months == 0
    assert d.tdelta == timedelta(0)
    d = parse_duration("+P1Y")
    assert d.years == 1


def test_parse_duration_as_timedelta_if_possible_false():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.years == 0 and d.months == 0
    assert d.tdelta == timedelta(days=1)
    d = parse_duration("-PT2H", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(hours=-2)
    d = parse_duration("P2W", as_timedelta_if_possible=False)
    assert d.tdelta == timedelta(weeks=2)
    d = parse_duration("P1Y", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.years == 1


def test_parse_duration_weeks_and_time_components():
    d = parse_duration("P1Y1W1DT1H1M1S")
    assert d.tdelta == timedelta(weeks=1, days=1, hours=1, minutes=1, seconds=1)
    d = parse_duration("P1MT1M")
    assert d.months == 1
    assert d.tdelta == timedelta(minutes=1)


@pytest.mark.parametrize(
    "string, expected",
    [
        ("P0003-06-04T12:30:05", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P00030604T123005", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P0018-09-04T11:09:08", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P0001-02-03T04:05:06.5", Duration(3, 6, 500000, 0, 5, 4, 0, 2, 1)),
    ],
)
def test_parse_duration_alternative(string, expected):
    d = parse_duration(string)
    assert isinstance(d, Duration)
    assert d == expected
    assert d.years == expected.years
    assert d.months == expected.months
    assert d.tdelta == expected.tdelta


def test_parse_duration_alternative_ordinal():
    # ordinal date: year 1, day 32 -> Feb 1 => months=2, day=1
    d = parse_duration("P0001-032T01:02:03")
    assert isinstance(d, Duration)
    assert d.years == 1
    assert d.months == 2
    assert d.tdelta == timedelta(days=1, hours=1, minutes=2, seconds=3)


@pytest.mark.parametrize(
    "string",
    [
        "",
        "P",
        "1Y",
        "P1",
        "PxY",
        "P1Y2D3M",
        "P1H",
        "PT1D",
        "P1S",
        "P 1D",
        "P1D ",
        "-P",
        "P1.Y",
        "P.5Y",
        "PT1H2D",
        "X1D",
        "--P1D",
        "P1D1D",
    ],
)
def test_parse_duration_invalid(string):
    with pytest.raises(ISO8601Error):
        parse_duration(string)


def test_parse_duration_empty_time_part_accepted():
    # current behaviour: a bare 'T' designator is tolerated
    assert parse_duration("PT") == timedelta(0)
    d = parse_duration("P1YT")
    assert isinstance(d, Duration)
    assert d.years == 1


def test_parse_duration_invalid_message():
    with pytest.raises(ISO8601Error, match="Unable to parse duration string"):
        parse_duration("garbage")


def test_parse_duration_type_error():
    with pytest.raises(TypeError, match="Expecting a string"):
        parse_duration(None)
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(b"P1D")


# ---------------------------------------------------------------------------
# duration_isoformat / strftime for durations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dur, expected",
    [
        (Duration(4, 8, 0, 0, 9, 11, 0, 9, 18), "P18Y9M4DT11H9M8S"),
        (timedelta(weeks=2), "P14D"),
        (Duration(4, 5, 0, 0, 30, 12, 0, 6, 3), "P3Y6M4DT12H30M5S"),
        (timedelta(days=23, hours=23), "P23DT23H"),
        (Duration(years=4), "P4Y"),
        (Duration(months=1), "P1M"),
        (timedelta(minutes=1), "PT1M"),
        (Duration(years=0.5), "P0.5Y"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(days=11), "P11D"),
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=59, microseconds=123450), "PT59.12345S"),
        (timedelta(hours=1, seconds=1), "PT1H1S"),
        (timedelta(hours=1, minutes=1), "PT1H1M"),
        (timedelta(days=1, seconds=1), "P1DT1S"),
        (timedelta(days=1, minutes=59, seconds=59), "P1DT59M59S"),
        (timedelta(hours=23, minutes=59, seconds=59), "PT23H59M59S"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-36), "-P1DT12H"),
        (timedelta(seconds=-1, microseconds=-500000), "-PT1.5S"),
        (Duration(years=-1, months=-2), "-P1Y2M"),
        (Duration(days=-1), "-P1D"),
        (Duration(months=-1), "-P1M"),
        (Duration(years=-1), "-P1Y"),
        (Duration(years=1, months=2, days=3), "P1Y2M3D"),
        (Duration(years=1, hours=1), "P1YT1H"),
    ],
)
def test_duration_isoformat(dur, expected):
    assert duration_isoformat(dur) == expected


def test_duration_isoformat_negative_parsed():
    assert duration_isoformat(parse_duration("-P3Y6M4DT12H30M5S")) == "-P3Y6M4DT12H30M5S"
    assert duration_isoformat(parse_duration("-P2W")) == "-P14D"
    assert duration_isoformat(parse_duration("-PT1H")) == "-PT1H"


@pytest.mark.parametrize(
    "string",
    [
        "P18Y9M4DT11H9M8S",
        "P3Y6M4DT12H30M5S",
        "P23DT23H",
        "P4Y",
        "P1M",
        "PT1M",
        "P0.5Y",
        "P1DT12H",
        "-P1DT12H",
        "PT1.5S",
        "PT0.000001S",
        "-P1Y2M",
        "P1Y2M3DT4H5M6.789S",
        "P0D",
    ],
)
def test_duration_roundtrip(string):
    assert duration_isoformat(parse_duration(string)) == string


@pytest.mark.parametrize(
    "dur, expected",
    [
        (timedelta(weeks=2), "P2W"),
        (timedelta(weeks=2, days=3), "P2W"),
        (timedelta(days=6), "P0W"),
        (timedelta(days=7), "P1W"),
        (timedelta(weeks=-3), "-P3W"),
        (timedelta(days=-1), "-P1W"),
        (Duration(weeks=5), "P5W"),
    ],
)
def test_duration_isoformat_weeks(dur, expected):
    assert duration_isoformat(dur, D_WEEK) == expected


def test_strftime_duration_directives():
    td = timedelta(days=12, hours=3, minutes=4, seconds=5, microseconds=6)
    assert strftime(td, "%d") == "12"
    assert strftime(td, "%H") == "03"
    assert strftime(td, "%M") == "04"
    assert strftime(td, "%S") == "05"
    assert strftime(td, "%f") == "000006"
    assert strftime(td, "%W") == "01"
    assert strftime(timedelta(days=21), "%W") == "03"
    assert strftime(timedelta(days=3), "%d") == "03"
    assert strftime(td, "%%") == "%"
    assert strftime(td, "100%%") == "100%"
    assert strftime(td, "%H:%M:%S.%f") == "03:04:05.000006"
    assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H%M%S") == "235959"
    assert strftime(timedelta(minutes=125), "%H:%M") == "02:05"
    assert strftime(td, "%q") == "%q"


def test_strftime_duration_with_years_months():
    d = Duration(days=3, hours=4, minutes=5, seconds=6, months=7, years=8)
    assert strftime(d, "%Y") == "0008"
    assert strftime(d, "%m") == "07"
    assert strftime(d, "%d") == "03"
    assert strftime(d, "%C") == "00"
    assert strftime(Duration(years=1234), "%C") == "12"
    assert strftime(Duration(years=1234), "%Y", 6) == "+001234"
    assert strftime(Duration(years=1234), "%C", 6) == "+0012"
    assert duration_isoformat(d, D_ALT_EXT) == "P0008-07-03T04:05:06"
    assert duration_isoformat(d, D_ALT_BAS) == "P00080703T040506"


def test_strftime_timedelta_has_no_months():
    with pytest.raises(AttributeError):
        strftime(timedelta(days=1), "%m")


def test_alternative_duration_roundtrip():
    d = parse_duration("P0003-06-04T12:30:05")
    assert duration_isoformat(d, D_ALT_EXT) == "P0003-06-04T12:30:05"
    d = parse_duration("P00030604T123005")
    assert duration_isoformat(d, D_ALT_BAS) == "P00030604T123005"


# ---------------------------------------------------------------------------
# strftime for dates / times
# ---------------------------------------------------------------------------


def test_strftime_date_directives():
    d = datetime(2003, 2, 5, 6, 7, 8, 9)
    assert strftime(d, "%d") == "05"
    assert strftime(d, "%m") == "02"
    assert strftime(d, "%H") == "06"
    assert strftime(d, "%M") == "07"
    assert strftime(d, "%S") == "08"
    assert strftime(d, "%f") == "000009"
    assert strftime(d, "%j") == "036"
    assert strftime(d, "%w") == "3"
    assert strftime(d, "%W") == "06"
    assert strftime(d, "%Y") == "2003"
    assert strftime(d, "%C") == "20"
    assert strftime(d, "%%") == "%"
    assert strftime(d, "%z") == ""
    assert strftime(d, "%Z") == ""
    assert strftime(d, "%h") == ""
    assert strftime(d, "literal") == "literal"
    assert strftime(d, "%q%Y") == "%q2003"
    assert strftime(d, "%%Y") == "%Y"


def test_strftime_weekday_iso():
    # Monday = 1, Sunday = 7
    assert strftime(date(2024, 1, 1), "%w") == "1"
    assert strftime(date(2024, 1, 7), "%w") == "7"


def test_strftime_ordinal():
    assert strftime(date(2024, 1, 1), "%j") == "001"
    assert strftime(date(2024, 12, 31), "%j") == "366"
    assert strftime(date(2023, 12, 31), "%j") == "365"
    assert strftime(date(2023, 3, 1), "%j") == "060"


def test_strftime_year_padding():
    assert strftime(date(5, 1, 1), "%Y") == "0005"
    assert strftime(date(5, 1, 1), "%Y", 6) == "+000005"
    assert strftime(date(1985, 1, 1), "%Y", 5) == "+01985"
    assert strftime(date(1985, 1, 1), "%C", 5) == "+019"
    assert strftime(date(150, 1, 1), "%C") == "01"
    assert strftime(date(9999, 1, 1), "%C") == "99"


def test_strftime_tz_directives():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-8, 0, "x"))
    assert strftime(dt, "%z") == "-0800"
    assert strftime(dt, "%Z") == "-08:00"
    assert strftime(dt, "%h") == "-08"
    dt = datetime(2000, 1, 1, tzinfo=UTC)
    assert strftime(dt, "%z%Z%h") == "ZZZ"


def test_strftime_time_object():
    t = time(4, 5, 6, 7)
    assert strftime(t, "%H|%M|%S|%f") == "04|05|06|000007"


def test_strftime_dispatch_duration_vs_date():
    # Same directive means different things for dates vs durations
    assert strftime(date(2000, 1, 9), "%d") == "09"
    assert strftime(timedelta(days=9), "%d") == "09"
    assert strftime(timedelta(days=100), "%d") == "100"
    # %P only for durations
    assert strftime(date(2000, 1, 1), "%P") == "%P"
    assert strftime(timedelta(days=1), "%P") == "1D"
    assert strftime(timedelta(days=14), "%p") == "2W"
    # %j unsupported for durations
    assert strftime(timedelta(days=1), "%j") == "%j"
