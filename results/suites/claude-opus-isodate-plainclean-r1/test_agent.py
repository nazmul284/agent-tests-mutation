"""Test suite for the isodate library."""

import pickle
from datetime import date, datetime, time, timedelta
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
    LOCAL,
    TIME_BAS_COMPLETE,
    TIME_BAS_MINUTE,
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
        # Jan 1 2009 is a Thursday -> week 1 starts in 2008
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W53-7", date(2010, 1, 3)),
        # Jan 1 2010 is a Friday -> belongs to week 53 of 2009
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W01", date(2010, 1, 4)),
        ("2004-W53-7", date(2005, 1, 2)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2015-W01-4", date(2015, 1, 1)),
    ],
)
def test_parse_date_basic_and_extended(text, expected):
    assert parse_date(text) == expected


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
def test_parse_date_expanded_six_digits(text, expected):
    assert parse_date(text, yeardigits=6) == expected


def test_parse_date_expanded_four_digits():
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")
    # without a sign the expanded format must not match
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", expanded=True)


def test_parse_date_yeardigits_forces_expanded():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", yeardigits=6)


def test_parse_date_negative_year_not_representable():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", yeardigits=6)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=15) == date(1985, 4, 15)
    assert parse_date("19", defaultmonth=6, defaultday=15) == date(1901, 6, 15)
    # complete date ignores defaults
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)


@pytest.mark.parametrize(
    "text",
    ["", "1985-4-12", "85-04-12", "1985/04/12", "1985-04-12x", "x1985", "1985-W1", "1985-W15-55", "198"],
)
def test_parse_date_invalid_format(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1985-04-31", "2001-02-29"])
def test_parse_date_invalid_values(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_iso8601error_is_value_error():
    assert issubclass(ISO8601Error, ValueError)


# ---------------------------------------------------------------------------
# date formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DATE_EXT_COMPLETE, "1985-04-12"),
        (DATE_BAS_COMPLETE, "19850412"),
        (DATE_EXT_ORD_COMPLETE, "1985-102"),
        (DATE_BAS_ORD_COMPLETE, "1985102"),
        (DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (DATE_BAS_WEEK_COMPLETE, "1985W155"),
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
    assert date_isoformat(date(1985, 4, 12), fmt, yeardigits=6) == expected


def test_date_isoformat_default_and_padding():
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
    assert date_isoformat(date(2000, 12, 31), DATE_EXT_ORD_COMPLETE) == "2000-366"
    assert date_isoformat(date(2000, 1, 1), DATE_EXT_ORD_COMPLETE) == "2000-001"


@pytest.mark.parametrize(
    "d",
    [date(1985, 4, 12), date(2000, 2, 29), date(1, 1, 1), date(9999, 12, 31), date(2004, 12, 31)],
)
@pytest.mark.parametrize(
    "fmt", [DATE_EXT_COMPLETE, DATE_BAS_COMPLETE, DATE_EXT_ORD_COMPLETE, DATE_BAS_ORD_COMPLETE]
)
def test_date_roundtrip(d, fmt):
    assert parse_date(date_isoformat(d, fmt)) == d


def test_date_week_roundtrip_mid_year():
    for d in [date(1985, 4, 12), date(2020, 6, 30), date(2021, 7, 4)]:
        assert parse_date(date_isoformat(d, DATE_EXT_WEEK_COMPLETE)) == d
        assert parse_date(date_isoformat(d, DATE_BAS_WEEK_COMPLETE)) == d


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
        ("T23", time(23)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50,5", time(23, 20, 50, 500000)),
        ("232050.5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.000001", time(23, 20, 50, 1)),
        ("23:20.5", time(23, 20, 30)),
        ("2320,5", time(23, 20, 30)),
        ("23:20.25", time(23, 20, 15)),
        ("23.5", time(23, 30)),
        ("23,3", time(23, 18)),
        ("10.25", time(10, 15)),
        ("00:00:00", time(0, 0, 0)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


def test_parse_time_fraction_truncates_beyond_microseconds():
    assert parse_time("23:20:50.1234567") == time(23, 20, 50, 123456)
    assert parse_time("14:43:59.9999997") == time(14, 43, 59, 999999)
    assert parse_time("10:15:30.0000009") == time(10, 15, 30, 0)


@pytest.mark.parametrize(
    "text, offset",
    [
        ("152746+0100", timedelta(hours=1)),
        ("15:27:46+01:00", timedelta(hours=1)),
        ("15:27:46+01", timedelta(hours=1)),
        ("15:27:46-05:00", timedelta(hours=-5)),
        ("152746-0500", timedelta(hours=-5)),
        ("15:27:46-05", timedelta(hours=-5)),
        ("15:27:46+05:30", timedelta(hours=5, minutes=30)),
        ("15:27:46-03:30", -timedelta(hours=3, minutes=30)),
        ("15:27+05:30", timedelta(hours=5, minutes=30)),
        ("15+05:30", timedelta(hours=5, minutes=30)),
    ],
)
def test_parse_time_with_offset(text, offset):
    result = parse_time(text)
    assert result.tzinfo is not None
    assert result.utcoffset() == offset
    assert result.tzinfo.dst(None) == timedelta(0)


def test_parse_time_values_with_tz():
    result = parse_time("15:27:46.5-05:00")
    assert (result.hour, result.minute, result.second, result.microsecond) == (15, 27, 46, 500000)
    result = parse_time("15.5+01")
    assert (result.hour, result.minute, result.second) == (15, 30, 0)


def test_parse_time_utc():
    result = parse_time("15:27:46Z")
    assert result.tzinfo is UTC
    assert result.utcoffset() == timedelta(0)
    assert result == time(15, 27, 46, tzinfo=UTC)
    assert parse_time("1527Z").tzinfo is UTC


def test_parse_time_tzname_kept():
    assert parse_time("10:00+05:30").tzinfo.tzname(None) == "+05:30"
    assert parse_time("10:00-0800").tzinfo.tzname(None) == "-0800"


@pytest.mark.parametrize("text", ["", "1:00", "1", "23:2", "23:20:5", "23:20:50Q", "23:20:50+1", "abc", "23:20:50 "])
def test_parse_time_invalid_format(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["25:00", "23:61", "23:20:61"])
def test_parse_time_invalid_values(text):
    with pytest.raises(ValueError):
        parse_time(text)


# ---------------------------------------------------------------------------
# time formatting
# ---------------------------------------------------------------------------


def test_time_isoformat_defaults():
    assert time_isoformat(time(10, 15, 30)) == "10:15:30"
    assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x"))) == "01:02:03-05:30"


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (TIME_BAS_COMPLETE + TZ_BAS, "231505+0100"),
        (TIME_BAS_MINUTE + TZ_HOUR, "2315+01"),
        (TIME_EXT_MINUTE + TZ_EXT, "23:15+01:00"),
        (TIME_HOUR, "23"),
        ("%H:%M:%S.%f", "23:15:05.000042"),
        ("%%H", "%H"),
    ],
)
def test_time_isoformat_formats(fmt, expected):
    t = time(23, 15, 5, 42, tzinfo=FixedOffset(1, 0, "+01:00"))
    assert time_isoformat(t, fmt) == expected


@pytest.mark.parametrize(
    "t",
    [
        time(0, 0),
        time(23, 59, 59),
        time(12, 30, 15, tzinfo=UTC),
        time(12, 30, 15, tzinfo=FixedOffset(-8, 0, "-08:00")),
    ],
)
def test_time_roundtrip(t):
    assert parse_time(time_isoformat(t)) == t


# ---------------------------------------------------------------------------
# parse_datetime / datetime_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
        ("19850412T101530", datetime(1985, 4, 12, 10, 15, 30)),
        ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985W155T10", datetime(1985, 4, 12, 10)),
        ("1985-04-12T10:15:30.5", datetime(1985, 4, 12, 10, 15, 30, 500000)),
        ("1985-04-12T10:15:30Z", datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)),
    ],
)
def test_parse_datetime(text, expected):
    assert parse_datetime(text) == expected


def test_parse_datetime_offsets():
    result = parse_datetime("1985-04-12T10:15:30+04:00")
    assert result.utcoffset() == timedelta(hours=4)
    assert result.replace(tzinfo=None) == datetime(1985, 4, 12, 10, 15, 30)
    assert result == datetime(1985, 4, 12, 6, 15, 30, tzinfo=UTC)
    result = parse_datetime("19850412T101530-0230")
    assert result.utcoffset() == -timedelta(hours=2, minutes=30)


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10:15T30", ""])
def test_parse_datetime_requires_single_t(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


def test_parse_datetime_invalid_parts():
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-4-12T10:15:30")
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12T1:15:30")


def test_datetime_isoformat():
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
    assert datetime_isoformat(dt) == "1985-04-12T10:15:30Z"
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "+04:00"))
    assert datetime_isoformat(dt) == "1985-04-12T10:15:30+04:00"
    assert datetime_isoformat(dt, DT_BAS_COMPLETE) == "19850412T101530+0400"
    assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30)) == "1985-04-12T10:15:30"
    assert DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"


@pytest.mark.parametrize(
    "dt",
    [
        datetime(1985, 4, 12, 10, 15, 30),
        datetime(2000, 2, 29, 23, 59, 59, tzinfo=UTC),
        datetime(2020, 1, 1, 0, 0, 0, tzinfo=FixedOffset(-3, -30, "-03:30")),
    ],
)
def test_datetime_roundtrip(dt):
    assert parse_datetime(datetime_isoformat(dt)) == dt
    assert parse_datetime(datetime_isoformat(dt, DT_BAS_COMPLETE)) == dt


# ---------------------------------------------------------------------------
# tzinfo
# ---------------------------------------------------------------------------


def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    assert parse_tzinfo("+01:00").utcoffset(None) == timedelta(hours=1)
    assert parse_tzinfo("+0130").utcoffset(None) == timedelta(hours=1, minutes=30)
    assert parse_tzinfo("-01").utcoffset(None) == timedelta(hours=-1)
    assert parse_tzinfo("-01:30").utcoffset(None) == -timedelta(hours=1, minutes=30)


def test_tz_isoformat():
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 30, "IST"))
    assert tz_isoformat(dt) == "+05:30"
    assert tz_isoformat(dt, "%Z") == "+05:30"
    assert tz_isoformat(dt, "%z") == "+0530"
    assert tz_isoformat(dt, "%h") == "+05"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-5, -45, "x"))
    assert tz_isoformat(dt, "%Z") == "-05:45"
    assert tz_isoformat(dt, "%z") == "-0545"
    assert tz_isoformat(dt, "%h") == "-05"
    dt = datetime(2000, 1, 1, tzinfo=FixedOffset(-11, 0, "x"))
    assert tz_isoformat(dt, "%Z") == "-11:00"


def test_tz_isoformat_utc_and_naive():
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC)) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1)) == ""


def test_tz_isoformat_bad_format():
    with pytest.raises(ValueError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%Q")


def test_utc_tzinfo():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_fixed_offset():
    fo = FixedOffset()
    assert fo.utcoffset(None) == timedelta(0)
    assert fo.tzname(None) == "UTC"
    fo = FixedOffset(-2, -15, "my")
    assert fo.utcoffset(None) == -timedelta(hours=2, minutes=15)
    assert fo.dst(None) == timedelta(0)
    assert fo.tzname(None) == "my"
    assert repr(fo) == "<FixedOffset 'my'>"


def test_local_timezone():
    from isodate.tzinfo import DSTOFFSET, STDOFFSET

    dt = datetime(2000, 1, 1, 12)
    assert LOCAL.utcoffset(dt) in (STDOFFSET, DSTOFFSET)
    assert LOCAL.dst(dt) in (timedelta(0), DSTOFFSET - STDOFFSET)
    assert isinstance(LOCAL.tzname(dt), str)
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


# ---------------------------------------------------------------------------
# strftime
# ---------------------------------------------------------------------------


def test_strftime_date_codes():
    d = date(1985, 4, 12)
    assert strftime(d, "%d/%m/%Y") == "12/04/1985"
    assert strftime(d, "%j") == "102"
    assert strftime(d, "%w") == "5"
    assert strftime(d, "%W") == "15"
    assert strftime(d, "%C") == "19"
    assert strftime(d, "100%%") == "100%"
    assert strftime(d, "%Q") == "%Q"
    assert strftime(date(2021, 1, 3), "%w") == "7"
    assert strftime(date(2021, 1, 4), "%w") == "1"


def test_strftime_datetime_codes():
    dt = datetime(2001, 2, 3, 4, 5, 6, 7)
    assert strftime(dt, "%Y-%m-%dT%H:%M:%S.%f") == "2001-02-03T04:05:06.000007"
    assert strftime(dt, "%Z%z%h") == ""


def test_strftime_timedelta_codes():
    td = timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=6)
    assert strftime(td, "%d %H:%M:%S.%f") == "02 03:04:05.000006"
    assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H:%M:%S") == "23:59:59"
    assert strftime(timedelta(days=14), "%W") == "02"
    assert strftime(timedelta(days=1), "%%") == "%"


def test_strftime_duration_codes():
    dur = Duration(years=3, months=4, days=5, hours=6, minutes=7, seconds=8)
    assert strftime(dur, "%Y-%m-%dT%H:%M:%S") == "0003-04-05T06:07:08"
    assert strftime(dur, "%Y", yeardigits=6) == "+000003"


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PT36H", timedelta(hours=36)),
        ("P1D", timedelta(days=1)),
        ("P2W", timedelta(weeks=2)),
        ("PT1M", timedelta(minutes=1)),
        ("P1M", Duration(months=1)),
        ("PT1S", timedelta(seconds=1)),
        ("PT0.5S", timedelta(milliseconds=500)),
        ("PT0,5S", timedelta(milliseconds=500)),
        ("P0.5D", timedelta(hours=12)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("P1W2D", timedelta(days=9)),
        ("PT1H30M", timedelta(hours=1, minutes=30)),
        ("+PT1H", timedelta(hours=1)),
        ("-PT1H", timedelta(hours=-1)),
        ("-P1DT1H", -timedelta(days=1, hours=1)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("P1Y", Duration(years=1)),
        ("P18Y9M4DT11H9M8S", Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8)),
        ("P1Y2M3DT4H5M6S", Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)),
        ("-P1Y2M", Duration(years=-1, months=-2)),
    ],
)
def test_parse_duration(text, expected):
    result = parse_duration(text)
    assert type(result) is type(expected)
    assert result == expected


def test_parse_duration_m_ambiguity():
    assert parse_duration("P1M") == Duration(months=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    result = parse_duration("P1MT1M")
    assert isinstance(result, Duration)
    assert result.months == 1
    assert result.tdelta == timedelta(minutes=1)


def test_parse_duration_fractional_years_months_are_decimal():
    result = parse_duration("P1.5Y")
    assert isinstance(result, Duration)
    assert result.years == Decimal("1.5")
    assert isinstance(result.years, Decimal)
    result = parse_duration("P2,5M")
    assert result.months == Decimal("2.5")


def test_parse_duration_negative_duration_components():
    result = parse_duration("-P1Y2M3D")
    assert result.years == -1
    assert result.months == -2
    assert result.tdelta == timedelta(days=-3)


def test_parse_duration_not_as_timedelta():
    result = parse_duration("PT1H", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result.tdelta == timedelta(hours=1)
    assert result.years == 0 and result.months == 0
    result = parse_duration("-P2D", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result.tdelta == timedelta(days=-2)


def test_parse_duration_alternative_format():
    expected = Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5)
    for text in ["P0003-06-04T12:30:05", "P00030604T123005"]:
        result = parse_duration(text)
        assert isinstance(result, Duration)
        assert result == expected
        assert result.years == 3
        assert result.months == 6
        assert result.tdelta == timedelta(days=4, hours=12, minutes=30, seconds=5)


def test_parse_duration_alternative_format_with_fraction():
    result = parse_duration("P0001-02-03T04:05:06.5")
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=500000)


@pytest.mark.parametrize("text", ["", "P", "1Y", "PX", "P1Y2", "P1S", "PT1D", "P1H", "T1H", "P1.Y", "P-1Y", "xP1Y"])
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


@pytest.mark.parametrize("value", [None, 1, b"P1Y", timedelta(1)])
def test_parse_duration_type_error(value):
    with pytest.raises(TypeError):
        parse_duration(value)


# ---------------------------------------------------------------------------
# duration_isoformat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(days=1), "P1D"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(milliseconds=500), "PT0.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=1, microseconds=120000), "PT1.12S"),
        (timedelta(hours=1, seconds=1), "PT1H1S"),
        (timedelta(days=-1), "-P1D"),
        (-timedelta(days=1, hours=1, minutes=30), "-P1DT1H30M"),
        (timedelta(seconds=-1), "-PT1S"),
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8), "P18Y9M4DT11H9M8S"),
        (Duration(years=-1, months=-2), "-P1Y2M"),
        (Duration(days=-3), "-P3D"),
        (-Duration(years=1, days=2), "-P1Y2D"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


def test_duration_isoformat_weeks():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=10), D_WEEK) == "P1W"
    assert duration_isoformat(timedelta(weeks=-2), D_WEEK) == "-P2W"


def test_duration_isoformat_alternative():
    dur = Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5)
    assert duration_isoformat(dur, D_ALT_EXT) == "P0003-06-04T12:30:05"
    assert duration_isoformat(dur, D_ALT_BAS) == "P00030604T123005"
    assert parse_duration(duration_isoformat(dur, D_ALT_EXT)) == dur
    assert D_DEFAULT == "P%P"


@pytest.mark.parametrize(
    "text",
    [
        "P18Y9M4DT11H9M8S",
        "P1Y",
        "P2M",
        "P3D",
        "PT4H",
        "PT5M",
        "PT6S",
        "PT0.5S",
        "P1DT2H3M4.25S",
        "-P1Y2M3DT4H",
        "-PT1S",
        "P1Y2MT3M",
    ],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


def test_duration_normalizes_on_format():
    assert duration_isoformat(parse_duration("PT36H")) == "P1DT12H"
    assert duration_isoformat(parse_duration("PT90M")) == "PT1H30M"
    assert duration_isoformat(parse_duration("P2W")) == "P14D"


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val, low, high, expected",
    [
        (Decimal(1), 1, 13, (0, Decimal(1))),
        (Decimal(12), 1, 13, (0, Decimal(12))),
        (Decimal(13), 1, 13, (1, Decimal(1))),
        (Decimal(25), 1, 13, (2, Decimal(1))),
        (Decimal(0), 1, 13, (-1, Decimal(12))),
        (Decimal(-11), 1, 13, (-1, Decimal(1))),
        (Decimal(-12), 1, 13, (-2, Decimal(12))),
        (Decimal(-1), 0, 60, (-1, Decimal(59))),
        (Decimal(7), 0, 5, (1, Decimal(2))),
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
        (2000, 2, 29),
        (1900, 2, 28),
        (2100, 2, 28),
        (1600, 2, 29),
    ],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


def test_duration_init_and_attributes():
    dur = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=1, months=7, years=8)
    assert dur.years == Decimal(8)
    assert dur.months == Decimal(7)
    assert isinstance(dur.years, Decimal)
    assert isinstance(dur.months, Decimal)
    assert dur.tdelta == timedelta(1, 2, 3, 4, 5, 6, 1)
    # attributes are forwarded to the timedelta
    assert dur.days == 8
    assert dur.seconds == 6 * 3600 + 5 * 60 + 2
    assert dur.microseconds == 4003
    assert Duration(years=1.5).years == Decimal("1.5")
    assert Duration(months=Decimal("0.25")).months == Decimal("0.25")


def test_duration_getattr_missing():
    with pytest.raises(AttributeError):
        Duration().nonexistent


def test_duration_str():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(years=2)) == "2 years, 0:00:00"
    assert str(Duration(hours=1)) == "1:00:00"
    assert str(Duration()) == "0:00:00"


def test_duration_repr():
    assert repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5)) == (
        "isodate.duration.Duration(1, 2, 3, years=4, months=5)"
    )
    assert repr(Duration()) == "isodate.duration.Duration(0, 0, 0, years=0, months=0)"


def test_duration_repr_roundtrip():
    dur = Duration(days=1, seconds=2, microseconds=3, years=4, months=5)
    assert eval(repr(dur), {"isodate": isodate}) == dur


def test_duration_hash():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    s = {Duration(years=1, days=2), Duration(years=1, days=2), Duration(months=1)}
    assert len(s) == 2


def test_duration_pickle():
    dur = Duration(years=1, months=2, days=3, hours=4)
    restored = pickle.loads(pickle.dumps(dur))
    assert restored == dur
    assert restored.years == dur.years
    assert restored.months == dur.months
    assert restored.tdelta == dur.tdelta


def test_duration_neg():
    dur = -Duration(years=1, months=2, days=3, hours=4)
    assert dur.years == -1
    assert dur.months == -2
    assert dur.tdelta == -timedelta(days=3, hours=4)
    assert -(-dur) == dur


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(days=1) == Duration(hours=24)
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert not (Duration(years=1) == timedelta(days=365))
    assert not (Duration(years=1) == Duration(years=1, days=1))
    assert not (Duration(years=1) == Duration(years=2))
    assert not (Duration(days=1) == 1)


def test_duration_inequality():
    assert Duration(years=1) != Duration(years=2)
    assert Duration(years=1) != Duration(years=1, seconds=1)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1) != timedelta(days=365)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert timedelta(days=2) != Duration(days=1)


def test_duration_add_duration():
    result = Duration(years=1, months=2, days=3) + Duration(years=4, months=11, hours=5)
    assert isinstance(result, Duration)
    assert result.years == 5
    assert result.months == 13
    assert result.tdelta == timedelta(days=3, hours=5)


def test_duration_add_timedelta():
    result = Duration(years=1, days=1) + timedelta(hours=1)
    assert isinstance(result, Duration)
    assert result.years == 1
    assert result.tdelta == timedelta(days=1, hours=1)
    result = timedelta(hours=1) + Duration(months=2)
    assert isinstance(result, Duration)
    assert result.months == 2
    assert result.tdelta == timedelta(hours=1)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 1, 1), Duration(months=13), date(2001, 2, 1)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 1), Duration(years=1, months=2, days=3), date(2001, 3, 4)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (date(2000, 5, 31), Duration(months=1), date(2000, 6, 30)),
        (date(2000, 1, 1), Duration(months=24), date(2002, 1, 1)),
        (date(2000, 1, 1), Duration(months=-24), date(1998, 1, 1)),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


def test_duration_add_datetime():
    dt = datetime(2000, 1, 31, 12, 0, tzinfo=UTC)
    result = dt + Duration(months=1, hours=13)
    assert result == datetime(2000, 3, 1, 1, 0, tzinfo=UTC)
    assert result.tzinfo is UTC
    assert Duration(years=1, minutes=30) + datetime(1999, 6, 1) == datetime(2000, 6, 1, 0, 30)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2001, 2, 1), Duration(months=13), date(2000, 1, 1)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2000, 3, 4), Duration(years=1, months=2, days=3), date(1999, 1, 1)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2000, 1, 1), Duration(months=12), date(1999, 1, 1)),
    ],
)
def test_date_sub_duration(start, dur, expected):
    assert start - dur == expected


def test_datetime_sub_duration():
    dt = datetime(2000, 3, 31, 12)
    assert dt - Duration(months=1, hours=13) == datetime(2000, 2, 28, 23)
    assert dt - Duration(years=1) == datetime(1999, 3, 31, 12)


def test_duration_sub_duration():
    result = Duration(years=3, months=2, days=5) - Duration(years=1, months=4, days=1)
    assert isinstance(result, Duration)
    assert result.years == 2
    assert result.months == -2
    assert result.tdelta == timedelta(days=4)


def test_duration_sub_timedelta():
    result = Duration(years=1, days=5) - timedelta(days=2)
    assert isinstance(result, Duration)
    assert result.years == 1
    assert result.tdelta == timedelta(days=3)


def test_timedelta_sub_duration():
    result = timedelta(days=10) - Duration(months=1, days=3)
    assert isinstance(result, Duration)
    assert result.months == -1
    assert result.years == 0
    assert result.tdelta == timedelta(days=7)


def test_duration_sub_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) - "x"
    with pytest.raises(TypeError):
        Duration(days=1) - date(2000, 1, 1)
    with pytest.raises(TypeError):
        "x" - Duration(days=1)
    with pytest.raises(TypeError):
        1 - Duration(days=1)


def test_duration_add_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) + "x"
    with pytest.raises(TypeError):
        1 + Duration(days=1)


def test_duration_fractional_with_date_raises():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("1.5"))
    with pytest.raises(ValueError):
        Duration(months=0.5) + date(2000, 1, 1)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("1.5"))


def test_duration_mul():
    dur = Duration(years=1, months=2, days=3)
    for result in (dur * 3, 3 * dur):
        assert isinstance(result, Duration)
        assert result.years == 3
        assert result.months == 6
        assert result.tdelta == timedelta(days=9)
    neg = dur * -1
    assert neg == -dur
    zero = dur * 0
    assert zero == Duration()


def test_duration_mul_unsupported():
    with pytest.raises(TypeError):
        Duration(days=1) * 1.5
    with pytest.raises(TypeError):
        Duration(days=1) * Duration(days=1)
    with pytest.raises(TypeError):
        Duration(days=1) * "a"


def test_duration_add_then_sub_is_identity_without_clamping():
    start = date(2000, 1, 15)
    dur = Duration(years=2, months=7, days=10)
    assert (start + dur) - dur == start


def test_totimedelta():
    assert Duration(months=1).totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert Duration(months=1).totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert Duration(months=1).totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=365)
    assert Duration(days=3, hours=1).totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=3, hours=1)
    assert Duration(months=1, hours=1).totimedelta(end=datetime(2000, 3, 1)) == timedelta(days=29, hours=1)


def test_totimedelta_arguments():
    with pytest.raises(ValueError):
        Duration(days=1).totimedelta()
    with pytest.raises(ValueError):
        Duration(days=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 2))


def test_parsed_duration_arithmetic():
    assert date(2000, 1, 31) + parse_duration("P1M") == date(2000, 2, 29)
    assert datetime(2000, 1, 1) + parse_duration("P1Y2M3DT4H5M6S") == datetime(2001, 3, 4, 4, 5, 6)
    assert datetime(2000, 1, 1) - parse_duration("-P1Y") == datetime(2001, 1, 1)
    assert datetime(2000, 1, 1) + parse_duration("-P1Y") == datetime(1999, 1, 1)


def test_public_api_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name), name
    assert isinstance(isodate.__version__, str)
