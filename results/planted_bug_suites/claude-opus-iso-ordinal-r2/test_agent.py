"""Test suite for isodate, written from the library source and ISO 8601 rules."""

import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import isodate
from isodate import (
    D_ALT_BAS,
    D_ALT_EXT,
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

# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("2000-02-29", date(2000, 2, 29)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        # week dates
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        ("2009-W01-1", date(2008, 12, 29)),  # week 1 starts in previous year
        ("2009-W53-7", date(2010, 1, 3)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2010-W01-1", date(2010, 1, 4)),  # Jan 1 2010 is in week 53 of 2009
        ("2010-W01", date(2010, 1, 4)),
        ("2015-W01-4", date(2015, 1, 1)),
        # ordinal dates
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-001", date(1985, 1, 1)),
        ("1985-365", date(1985, 12, 31)),
        ("2000-366", date(2000, 12, 31)),
        ("2000-060", date(2000, 2, 29)),
        # reduced accuracy
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
    ],
)
def test_parse_date_four_digit(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text, expected",
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
        ("+001985", date(1985, 1, 1)),
        ("+0019", date(1901, 1, 1)),
    ],
)
def test_parse_date_expanded_six_digits(text, expected):
    assert parse_date(text, 6) == expected


def test_parse_date_expanded_four_digit_with_sign():
    assert parse_date("+1985-04-12", 4, True) == date(1985, 4, 12)
    # without expanded=True, a sign is not accepted for 4-digit years
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")


def test_parse_date_yeardigits_forces_expanded():
    # a sign is mandatory when yeardigits != 4
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", 6)


def test_parse_date_negative_year_is_value_error():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985", date(1985, 6, 15)),
        ("1985-04", date(1985, 4, 15)),
        ("19", date(1901, 6, 15)),
        ("1985-04-12", date(1985, 4, 12)),  # complete date ignores defaults
    ],
)
def test_parse_date_defaults(text, expected):
    assert parse_date(text, defaultmonth=6, defaultday=15) == expected


@pytest.mark.parametrize(
    "text",
    ["", "1985-4-12", "85-04-12", "1985/04/12", "1985-04-12T", "abcd", "1985-W1", "1985-1", "1"],
)
def test_parse_date_invalid_format(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-29", "1985-04-31", "1985-04-00"])
def test_parse_date_invalid_values(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_iso8601error_is_value_error():
    assert issubclass(ISO8601Error, ValueError)


def test_parse_ordinal_all_days_of_year():
    for year in (1985, 2000, 2024, 1900):
        start = date(year, 1, 1)
        ndays = (date(year + 1, 1, 1) - start).days
        for n in range(1, ndays + 1):
            expected = start + timedelta(days=n - 1)
            assert parse_date("%04d-%03d" % (year, n)) == expected
            assert parse_date("%04d%03d" % (year, n)) == expected


def test_parse_week_dates_match_isocalendar():
    d = date(2003, 12, 1)
    while d < date(2011, 2, 1):
        y, w, wd = d.isocalendar()
        assert parse_date("%04d-W%02d-%d" % (y, w, wd)) == d
        assert parse_date("%04dW%02d%d" % (y, w, wd)) == d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# date formatting
# ---------------------------------------------------------------------------


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


def test_date_isoformat_default():
    assert date_isoformat(date(1985, 4, 12)) == "1985-04-12"
    assert date_isoformat(date(7, 3, 9)) == "0007-03-09"


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_EXT_COMPLETE, "+001985-04-12"),
        (DATE_BAS_COMPLETE, "+0019850412"),
        (DATE_EXT_ORD_COMPLETE, "+001985-102"),
        (DATE_YEAR, "+001985"),
        (DATE_CENTURY, "+0019"),
    ],
)
def test_date_isoformat_expanded(fmt, expected):
    assert date_isoformat(date(1985, 4, 12), fmt, 6) == expected


def test_ordinal_formatting_edges():
    assert date_isoformat(date(1985, 1, 1), DATE_EXT_ORD_COMPLETE) == "1985-001"
    assert date_isoformat(date(1985, 12, 31), DATE_EXT_ORD_COMPLETE) == "1985-365"
    assert date_isoformat(date(2000, 12, 31), DATE_EXT_ORD_COMPLETE) == "2000-366"


def test_week_formatting_uses_iso_year_weeks():
    assert date_isoformat(date(2008, 12, 29), "%W-%w") == "01-1"
    assert date_isoformat(date(2010, 1, 3), "%W-%w") == "53-7"


@pytest.mark.parametrize(
    "fmt",
    [
        DATE_EXT_COMPLETE,
        DATE_BAS_COMPLETE,
        DATE_EXT_ORD_COMPLETE,
        DATE_BAS_ORD_COMPLETE,
    ],
)
def test_date_roundtrip(fmt):
    d = date(1999, 1, 1)
    while d < date(2001, 3, 1):
        assert parse_date(date_isoformat(d, fmt)) == d
        d += timedelta(days=7)


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("232050", time(23, 20, 50)),
        ("23:20:50", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("2320", time(23, 20)),
        ("23:20", time(23, 20)),
        ("23", time(23)),
        ("00:00:00", time(0, 0)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50,5", time(23, 20, 50, 500000)),
        ("232050.5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.1234567", time(23, 20, 50, 123456)),  # truncated, not rounded
        ("23:20:50.9999999", time(23, 20, 50, 999999)),
        ("23:20.5", time(23, 20, 30)),
        ("2320,5", time(23, 20, 30)),
        ("23:20.25", time(23, 20, 15)),
        ("23.5", time(23, 30)),
        ("23,3", time(23, 18)),
        ("12.25", time(12, 15)),
        ("14.1", time(14, 6)),
        ("10.0125", time(10, 0, 45)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "text, offset",
    [
        ("232030Z", timedelta(0)),
        ("23:20:30Z", timedelta(0)),
        ("23:20:30+01:00", timedelta(hours=1)),
        ("23:20:30+0100", timedelta(hours=1)),
        ("23:20:30+01", timedelta(hours=1)),
        ("23:20:30-05:00", timedelta(hours=-5)),
        ("23:20:30-0530", timedelta(hours=-5, minutes=-30)),
        ("23:20:30+05:45", timedelta(hours=5, minutes=45)),
        ("2320-05", timedelta(hours=-5)),
        ("23+03:30", timedelta(hours=3, minutes=30)),
        ("23:20:30.5-01:15", timedelta(hours=-1, minutes=-15)),
    ],
)
def test_parse_time_tz(text, offset):
    result = parse_time(text)
    assert result.tzinfo is not None
    assert result.utcoffset() == offset


def test_parse_time_z_is_utc_singleton():
    assert parse_time("10:00Z").tzinfo is UTC


def test_parse_time_tz_name_preserved():
    t = parse_time("10:00:00+05:30")
    assert t.tzname() == "+05:30"
    assert isinstance(t.tzinfo, FixedOffset)


@pytest.mark.parametrize(
    "text",
    ["", "2", "23:2", "23:20:5", "232", "23:20:30+1", "23:20:30 Z", "T", "ab:cd", "23:20:30.", "23:20:30Z+01"],
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["24:00:01", "23:60", "23:20:60", "25"])
def test_parse_time_out_of_range(text):
    with pytest.raises(ValueError):
        parse_time(text)


# ---------------------------------------------------------------------------
# time formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (TIME_EXT_COMPLETE, "23:20:50"),
        (TIME_BAS_COMPLETE, "232050"),
        (TIME_EXT_MINUTE, "23:20"),
        (TIME_BAS_MINUTE, "2320"),
        (TIME_HOUR, "23"),
        (TIME_EXT_COMPLETE + ".%f", "23:20:50.000042"),
    ],
)
def test_time_isoformat_naive(fmt, expected):
    assert time_isoformat(time(23, 20, 50, 42), fmt) == expected


def test_time_isoformat_default():
    assert time_isoformat(time(1, 2, 3)) == "01:02:03"
    assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x"))) == "01:02:03-05:30"


@pytest.mark.parametrize(
    "tz, ext, bas, hour",
    [
        (UTC, "Z", "Z", "Z"),
        (FixedOffset(0, 0, "UTC"), "Z", "Z", "Z"),
        (FixedOffset(1, 0, "x"), "+01:00", "+0100", "+01"),
        (FixedOffset(5, 45, "x"), "+05:45", "+0545", "+05"),
        (FixedOffset(-5, -30, "x"), "-05:30", "-0530", "-05"),
        (FixedOffset(-12, 0, "x"), "-12:00", "-1200", "-12"),
        (FixedOffset(14, 0, "x"), "+14:00", "+1400", "+14"),
    ],
)
def test_tz_formatting(tz, ext, bas, hour):
    t = time(10, 0, 0, tzinfo=tz)
    assert time_isoformat(t, TIME_EXT_COMPLETE + TZ_EXT) == "10:00:00" + ext
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "100000" + bas
    assert time_isoformat(t, TIME_HOUR + TZ_HOUR) == "10" + hour


def test_tz_formatting_naive_empty():
    assert time_isoformat(time(10, 0), TIME_HOUR + TZ_EXT) == "10"
    assert time_isoformat(time(10, 0), TZ_BAS + TZ_HOUR) == ""


@pytest.mark.parametrize(
    "text",
    ["23:20:50", "23:20:50Z", "23:20:50+01:00", "23:20:50-05:30", "00:00:00+14:00", "12:34:56-00:30"],
)
def test_time_roundtrip(text):
    assert time_isoformat(parse_time(text)) == text


# ---------------------------------------------------------------------------
# tzinfo parsing / formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, offset",
    [
        ("+01:00", timedelta(hours=1)),
        ("+0100", timedelta(hours=1)),
        ("+01", timedelta(hours=1)),
        ("-03:30", timedelta(hours=-3, minutes=-30)),
        ("-0330", timedelta(hours=-3, minutes=-30)),
        ("+00:00", timedelta(0)),
        ("Z", timedelta(0)),
    ],
)
def test_parse_tzinfo(text, offset):
    tz = parse_tzinfo(text)
    assert tz.utcoffset(None) == offset
    assert tz.dst(None) == timedelta(0)


def test_parse_tzinfo_z_and_empty():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None


def test_tz_isoformat_direct():
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(2, 30, "x"))
    assert tz_isoformat(dt) == "+02:30"
    assert tz_isoformat(dt, "%Z") == "+02:30"
    assert tz_isoformat(dt, "%z") == "+0230"
    assert tz_isoformat(dt, "%h") == "+02"
    assert tz_isoformat(datetime(2020, 1, 1)) == ""
    assert tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%Q")


def test_tz_isoformat_negative_minutes_only():
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(0, -30, "x"))
    assert tz_isoformat(dt, "%Z") == "-00:30"
    assert tz_isoformat(dt, "%z") == "-0030"


# ---------------------------------------------------------------------------
# tzinfo classes
# ---------------------------------------------------------------------------


def test_utc_class():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"


def test_utc_pickle_returns_singleton():
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_fixed_offset():
    tz = FixedOffset(-2, -15, "my")
    assert tz.utcoffset(None) == timedelta(hours=-2, minutes=-15)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "my"
    assert repr(tz) == "<FixedOffset 'my'>"
    default = FixedOffset()
    assert default.utcoffset(None) == timedelta(0)
    assert default.tzname(None) == "UTC"


def test_local_timezone_consistent():
    dt = datetime(2020, 7, 1, 12, 0)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert isinstance(LOCAL.tzname(dt), str)
    assert LOCAL.dst(dt) in (timedelta(0), off - LOCAL.utcoffset(datetime(2020, 1, 1)), LOCAL.dst(dt))
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


# ---------------------------------------------------------------------------
# parse_datetime / datetime_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("19850412T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
        ("1985-04-12T10:15:30.5", datetime(1985, 4, 12, 10, 15, 30, 500000)),
        ("1985102T235030", datetime(1985, 4, 12, 23, 50, 30)),
        ("1985-102T23:50:30", datetime(1985, 4, 12, 23, 50, 30)),
        ("1985-W15-5T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985W155T1015", datetime(1985, 4, 12, 10, 15)),
        ("2012-10-12T08:29:46.069178", datetime(2012, 10, 12, 8, 29, 46, 69178)),
    ],
)
def test_parse_datetime_naive(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12T10:15:30Z", datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)),
        (
            "1985-04-12T10:15:30+04:00",
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone(timedelta(hours=4))),
        ),
        (
            "19850412T101530-0400",
            datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone(timedelta(hours=-4))),
        ),
        (
            "2012-10-30T08:55:22.1234567+05",
            datetime(2012, 10, 30, 8, 55, 22, 123456, tzinfo=timezone(timedelta(hours=5))),
        ),
    ],
)
def test_parse_datetime_aware(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15", "1985-04-12T10:15T", "T10:15", ""])
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DT_EXT_COMPLETE, "1985-04-12T10:15:30+04:00"),
        (DT_BAS_COMPLETE, "19850412T101530+0400"),
        (DT_EXT_ORD_COMPLETE, "1985-102T10:15:30+04:00"),
        (DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30+04:00"),
        (DATE_EXT_COMPLETE + "T" + TIME_EXT_MINUTE + TZ_HOUR, "1985-04-12T10:15+04"),
    ],
)
def test_datetime_isoformat(fmt, expected):
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "+04:00"))
    assert datetime_isoformat(dt, fmt) == expected


def test_datetime_isoformat_default():
    assert datetime_isoformat(datetime(2001, 2, 3, 4, 5, 6)) == "2001-02-03T04:05:06"
    assert datetime_isoformat(datetime(2001, 2, 3, 4, 5, 6, tzinfo=UTC)) == "2001-02-03T04:05:06Z"


@pytest.mark.parametrize(
    "text",
    [
        "1985-04-12T10:15:30Z",
        "1985-04-12T10:15:30+04:00",
        "2000-02-29T00:00:00-11:30",
        "0001-01-01T00:00:00",
    ],
)
def test_datetime_roundtrip(text):
    assert datetime_isoformat(parse_datetime(text)) == text


def test_datetime_ord_roundtrip():
    dt = datetime(2000, 12, 31, 1, 2, 3, tzinfo=UTC)
    s = datetime_isoformat(dt, DT_EXT_ORD_COMPLETE)
    assert s == "2000-366T01:02:03Z"
    assert parse_datetime(s) == dt


# ---------------------------------------------------------------------------
# strftime misc
# ---------------------------------------------------------------------------


def test_strftime_literal_percent_and_unknown():
    assert strftime(date(2020, 5, 6), "%%Y=%Y %q") == "%Y=2020 %q"
    assert strftime(timedelta(days=1), "%% %q") == "% %q"


def test_strftime_all_date_codes():
    dt = datetime(2021, 3, 4, 5, 6, 7, 89)
    assert strftime(dt, "%Y|%m|%d|%H|%M|%S|%f|%j|%w|%W|%C") == "2021|03|04|05|06|07|000089|063|4|09|20"


def test_strftime_duration_codes():
    td = timedelta(days=12, hours=3, minutes=4, seconds=5, microseconds=6)
    assert strftime(td, "%d %H %M %S %f") == "12 03 04 05 000006"
    assert strftime(timedelta(weeks=3), "%W") == "03"
    dur = Duration(years=3, months=7)
    assert strftime(dur, "%Y-%m") == "0003-07"
    assert strftime(Duration(years=1234), "%C") == "12"


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P18D", timedelta(days=18)),
        ("P2W", timedelta(weeks=2)),
        ("P3W1D", timedelta(days=22)),
        ("PT36H", timedelta(hours=36)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("PT1M", timedelta(minutes=1)),
        ("PT10S", timedelta(seconds=10)),
        ("PT1.5S", timedelta(seconds=1.5)),
        ("PT1,5S", timedelta(seconds=1.5)),
        ("PT0.5M", timedelta(seconds=30)),
        ("P0.5D", timedelta(hours=12)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("+P1D", timedelta(days=1)),
        ("-P1D", timedelta(days=-1)),
        ("-PT1S", timedelta(seconds=-1)),
        ("-P2DT3H", -timedelta(days=2, hours=3)),
        ("P0Y0M5D", timedelta(days=5)),
    ],
)
def test_parse_duration_timedelta(text, expected):
    result = parse_duration(text)
    assert isinstance(result, timedelta)
    assert result == expected


@pytest.mark.parametrize(
    "text, years, months, tdelta",
    [
        ("P1Y", 1, 0, timedelta(0)),
        ("P2M", 0, 2, timedelta(0)),
        ("P1Y2M", 1, 2, timedelta(0)),
        ("P3Y6M4DT12H30M5S", 3, 6, timedelta(days=4, hours=12, minutes=30, seconds=5)),
        ("P1Y2M3W", 1, 2, timedelta(weeks=3)),
        ("P1.5Y", Decimal("1.5"), 0, timedelta(0)),
        ("P2,5M", 0, Decimal("2.5"), timedelta(0)),
        ("P1M1DT1M", 0, 1, timedelta(days=1, minutes=1)),
    ],
)
def test_parse_duration_duration(text, years, months, tdelta):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result.years == years
    assert result.months == months
    assert result.tdelta == tdelta


def test_parse_duration_negative_duration():
    result = parse_duration("-P1Y2M3DT4H")
    assert isinstance(result, Duration)
    assert result.years == -1
    assert result.months == -2
    assert result.tdelta == -timedelta(days=3, hours=4)


def test_parse_duration_decimal_types():
    result = parse_duration("P1Y2M")
    assert isinstance(result.years, Decimal)
    assert isinstance(result.months, Decimal)


def test_parse_duration_not_as_timedelta():
    result = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result.years == 0 and result.months == 0
    assert result.tdelta == timedelta(days=1)
    neg = parse_duration("-PT1H", as_timedelta_if_possible=False)
    assert isinstance(neg, Duration)
    assert neg.tdelta == timedelta(hours=-1)


@pytest.mark.parametrize(
    "text, years, months, tdelta",
    [
        ("P0018-09-04T11:09:08", 18, 9, timedelta(days=4, hours=11, minutes=9, seconds=8)),
        ("P00180904T110908", 18, 9, timedelta(days=4, hours=11, minutes=9, seconds=8)),
        ("P0001-01-01T00:00:00", 1, 1, timedelta(days=1)),
    ],
)
def test_parse_duration_alternative(text, years, months, tdelta):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result.years == years
    assert result.months == months
    assert result.tdelta == tdelta


@pytest.mark.parametrize(
    "text",
    ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P1D2H", "Pfoo", "P-1D", "PxYT", "PP1D", "1Y"],
)
def test_parse_duration_invalid(text):
    with pytest.raises((ISO8601Error, ValueError)):
        parse_duration(text)


@pytest.mark.parametrize("text", ["", "P", "1D", "P1H", "PT1D", "-P", "P1Y1Y"])
def test_parse_duration_invalid_is_iso8601error(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_type_error():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


# ---------------------------------------------------------------------------
# duration_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(days=18), "P18D"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=10), "PT10S"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(microseconds=120), "PT0.00012S"),
        (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
        (timedelta(0), "P0D"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-36), "-P1DT12H"),
        (timedelta(seconds=-1), "-PT1S"),
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5), "P3Y6M4DT12H30M5S"),
        (Duration(years=-1, months=-2), "-P1Y2M"),
        (Duration(days=-3), "-P3D"),
        (Duration(0), "P0D"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


def test_duration_isoformat_week():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=15), D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(weeks=-3), D_WEEK) == "-P3W"


def test_duration_isoformat_alternative():
    dur = Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8)
    assert duration_isoformat(dur, D_ALT_EXT) == "P0018-09-04T11:09:08"
    assert duration_isoformat(dur, D_ALT_BAS) == "P00180904T110908"


def test_duration_isoformat_default_is_d_default():
    assert D_DEFAULT == "P%P"
    assert duration_isoformat(timedelta(days=2)) == duration_isoformat(timedelta(days=2), D_DEFAULT)


@pytest.mark.parametrize(
    "text",
    [
        "P18D",
        "P14D",
        "P1DT12H",
        "P1Y",
        "P1M",
        "P3Y6M4DT12H30M5S",
        "PT1.5S",
        "-P1D",
        "-P1Y2M",
        "P0D",
        "PT0.000001S",
    ],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


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
        (Decimal(-11), 1, 13, (-1, Decimal(1))),
        (Decimal(-12), 1, 13, (-2, Decimal(12))),
        (Decimal(7), 0, 5, (1, Decimal(2))),
        (Decimal(-1), 0, 5, (-1, Decimal(4))),
    ],
)
def test_fquotmod(val, low, high, expected):
    assert fquotmod(val, low, high) == expected


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
    ],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


def test_duration_init_and_attributes():
    d = Duration(1, 2, 3, 4, 5, 6, 7, months=8, years=9)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.months == Decimal(8)
    assert d.years == Decimal(9)
    assert isinstance(d.months, Decimal)
    assert isinstance(d.years, Decimal)
    # attribute delegation to tdelta
    assert d.days == d.tdelta.days
    assert d.seconds == d.tdelta.seconds
    assert d.microseconds == d.tdelta.microseconds
    assert d.total_seconds() == d.tdelta.total_seconds()
    with pytest.raises(AttributeError):
        d.nonexistent_attribute


def test_duration_float_months_converted_via_str():
    d = Duration(months=0.1, years=1.1)
    assert d.months == Decimal("0.1")
    assert d.years == Decimal("1.1")


def test_duration_str():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(years=2, months=3)) == "2 years, 3 months, 0:00:00"
    assert str(Duration(hours=1)) == "1:00:00"
    assert str(Duration(months=-2)) == "-2 month, 0:00:00"


def test_duration_repr():
    d = Duration(days=1, seconds=2, microseconds=3, years=4, months=5)
    assert repr(d) == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=1, months=2, days=3)
    c = Duration(years=1, months=2, days=4)
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2
    assert {a: 1}[b] == 1


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        loaded = pickle.loads(pickle.dumps(d, proto))
        assert loaded == d
        assert loaded.years == d.years
        assert loaded.months == d.months
        assert loaded.tdelta == d.tdelta


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3, hours=4)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == -timedelta(days=3, hours=4)
    assert -(-d) == d


def test_duration_add_duration():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=4, months=11, hours=5)
    r = a + b
    assert isinstance(r, Duration)
    assert r.years == 5
    assert r.months == 13
    assert r.tdelta == timedelta(days=3, hours=5)
    # equality normalises months into years
    assert r == Duration(years=6, months=1, days=3, hours=5)


def test_duration_add_timedelta_both_sides():
    d = Duration(years=1, days=1)
    td = timedelta(hours=6)
    for r in (d + td, td + d):
        assert isinstance(r, Duration)
        assert r.years == 1
        assert r.tdelta == timedelta(days=1, hours=6)


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) + 1
    with pytest.raises(TypeError):
        Duration(days=1) + "x"


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 4, 30)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=14), date(2002, 1, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 1, 15), Duration(months=-13), date(1998, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 1), Duration(years=1, months=2, days=3), date(2001, 3, 4)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (date(2000, 1, 1), Duration(days=-1), date(1999, 12, 31)),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


def test_duration_add_datetime_keeps_time_and_tz():
    start = datetime(2000, 1, 31, 10, 30, tzinfo=UTC)
    r = start + Duration(months=1, hours=2)
    assert r == datetime(2000, 2, 29, 12, 30, tzinfo=UTC)
    assert r.tzinfo is UTC
    assert isinstance(r, datetime)


def test_duration_add_parsed_example():
    assert parse_date("2003-01-31") + parse_duration("P1M") == date(2003, 2, 28)
    assert parse_datetime("2000-02-29T00:00:00") + parse_duration("P1Y") == datetime(2001, 2, 28)
    assert parse_date("1985-04-12") + parse_duration("P1Y2M3DT0H") == date(1986, 6, 15)


def test_duration_add_fractional_to_date_raises():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("1.5"))
    with pytest.raises(ValueError):
        Duration(years=Decimal("0.5")) + date(2000, 1, 1)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=Decimal("0.5"))


def test_duration_sub_duration():
    a = Duration(years=3, months=2, days=10)
    b = Duration(years=1, months=5, days=3)
    r = a - b
    assert isinstance(r, Duration)
    assert r.years == 2
    assert r.months == -3
    assert r.tdelta == timedelta(days=7)
    assert r == Duration(years=1, months=9, days=7)


def test_duration_sub_timedelta():
    r = Duration(years=1, days=10) - timedelta(days=3)
    assert isinstance(r, Duration)
    assert r.years == 1
    assert r.tdelta == timedelta(days=7)


def test_timedelta_sub_duration():
    r = timedelta(days=10) - Duration(years=1, days=3)
    assert isinstance(r, Duration)
    assert r.years == -1
    assert r.months == 0
    assert r.tdelta == timedelta(days=7)


def test_duration_sub_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) - 1
    with pytest.raises(TypeError):
        Duration(days=1) - date(2000, 1, 1)
    with pytest.raises(TypeError):
        1 - Duration(days=1)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2001, 1, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2001, 1, 15), Duration(months=13), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2001, 3, 4), Duration(years=1, months=2, days=3), date(2000, 1, 1)),
        (date(2000, 3, 1), Duration(days=1), date(2000, 2, 29)),
    ],
)
def test_date_sub_duration(start, dur, expected):
    assert start - dur == expected


def test_datetime_sub_duration():
    r = datetime(2000, 3, 31, 1, 0, tzinfo=UTC) - Duration(months=1, hours=2)
    assert r == datetime(2000, 2, 28, 23, 0, tzinfo=UTC)


def test_date_sub_matches_add_negative_for_whole_days():
    d = date(2010, 5, 31)
    dur = Duration(years=2, months=3, days=5)
    assert d - dur == d + (-dur)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3)
    for r in (d * 3, 3 * d):
        assert isinstance(r, Duration)
        assert r.years == 3
        assert r.months == 6
        assert r.tdelta == timedelta(days=9)
    zero = d * 0
    assert zero == Duration(0)
    neg = d * -1
    assert neg == -d
    with pytest.raises(TypeError):
        d * 1.5
    with pytest.raises(TypeError):
        d * d


def test_duration_mul_add_date():
    assert date(2000, 1, 31) + Duration(months=1) * 2 == date(2000, 3, 31)


def test_duration_equality():
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1) != Duration(months=13)
    assert Duration(years=1, days=1) != Duration(years=1)
    assert Duration(days=1) == Duration(hours=24)
    assert not (Duration(years=1) != Duration(months=12))
    assert not (Duration(years=1) == Duration(years=1, days=1))


def test_duration_equality_with_timedelta():
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert not (Duration(months=1) == timedelta(0))
    assert Duration(months=1) != timedelta(0)


def test_duration_equality_with_other_types():
    assert Duration(days=1) != 1
    assert not (Duration(days=1) == "P1D")
    assert Duration(years=1) != 1


def test_duration_totimedelta_start():
    assert Duration(months=1).totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert Duration(months=1).totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1, days=1).totimedelta(start=datetime(2001, 1, 1)) == timedelta(days=366)


def test_duration_totimedelta_end():
    assert Duration(months=1).totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert Duration(months=1).totimedelta(end=date(2001, 3, 1)) == timedelta(days=28)
    assert Duration(years=1).totimedelta(end=date(2001, 1, 1)) == timedelta(days=366)


def test_duration_totimedelta_args():
    with pytest.raises(ValueError):
        Duration(days=1).totimedelta()
    with pytest.raises(ValueError):
        Duration(days=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 2))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def test_public_api_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name), name
    assert isinstance(isodate.__version__, str)


def test_format_constants():
    assert DATE_BAS_COMPLETE == "%Y%m%d"
    assert DATE_EXT_COMPLETE == "%Y-%m-%d"
    assert DATE_EXT_ORD_COMPLETE == "%Y-%j"
    assert DATE_EXT_WEEK_COMPLETE == "%Y-W%W-%w"
    assert TIME_EXT_COMPLETE == "%H:%M:%S"
    assert TZ_EXT == "%Z"
    assert TZ_BAS == "%z"
    assert TZ_HOUR == "%h"
    assert DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert D_WEEK == "P%p"
