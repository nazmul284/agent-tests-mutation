"""Test suite for the isodate ISO 8601 parser and formatter."""

import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    D_ALT_BAS,
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


def offset(hours=0, minutes=0):
    return timedelta(hours=hours, minutes=minutes)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("19", date(1901, 1, 1)),
        ("1985", date(1985, 1, 1)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-102", date(1985, 4, 12)),
        ("1985-001", date(1985, 1, 1)),
        ("1985-365", date(1985, 12, 31)),
        ("2000-366", date(2000, 12, 31)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W15", date(1985, 4, 8)),
        ("1985-W15", date(1985, 4, 8)),
        ("1989-W15", date(1989, 4, 10)),
        ("1989-W15-5", date(1989, 4, 14)),
        ("1936-W01-1", date(1935, 12, 30)),  # week 1 starts in previous year
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2010-W01-1", date(2010, 1, 4)),  # Jan 1st 2010 is in week 53 of 2009
        ("2004-W53-6", date(2005, 1, 1)),
        ("2020-02-29", date(2020, 2, 29)),
    ],
)
def test_parse_date_4_digits(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+0019", date(1901, 1, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+001985-04", date(1985, 4, 1)),
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
def test_parse_date_6_digits(text, expected):
    assert parse_date(text, 6) == expected


def test_parse_date_expanded_4_digits():
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", expanded=True)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)


def test_parse_date_negative_year_not_representable():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)


@pytest.mark.parametrize(
    "text",
    ["", "abc", "1985-4-12", "85-04-12", "1985/04/12", "1985-04-12T", "19850412 ", "1985-W1"],
)
def test_parse_date_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1985-04-31"])
def test_parse_date_out_of_range(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_iso8601_error_is_value_error():
    assert issubclass(ISO8601Error, ValueError)


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (date(1985, 4, 12), DATE_EXT_COMPLETE, "1985-04-12"),
        (date(1985, 4, 12), DATE_BAS_COMPLETE, "19850412"),
        (date(1985, 4, 12), DATE_EXT_ORD_COMPLETE, "1985-102"),
        (date(1985, 4, 12), DATE_BAS_ORD_COMPLETE, "1985102"),
        (date(1985, 4, 12), DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
        (date(1985, 4, 12), DATE_BAS_WEEK_COMPLETE, "1985W155"),
        (date(1985, 4, 12), DATE_EXT_WEEK, "1985-W15"),
        (date(1985, 4, 12), DATE_BAS_WEEK, "1985W15"),
        (date(1985, 4, 12), DATE_EXT_MONTH, "1985-04"),
        (date(1985, 4, 12), DATE_BAS_MONTH, "198504"),
        (date(1985, 4, 12), DATE_YEAR, "1985"),
        (date(1985, 4, 12), DATE_CENTURY, "19"),
        (date(5, 1, 7), DATE_EXT_COMPLETE, "0005-01-07"),
        (date(2010, 1, 3), DATE_EXT_WEEK_COMPLETE, "2010-W53-7"),
        (date(2000, 12, 31), DATE_EXT_ORD_COMPLETE, "2000-366"),
        (date(2000, 1, 1), DATE_EXT_ORD_COMPLETE, "2000-001"),
    ],
)
def test_date_isoformat(value, fmt, expected):
    assert date_isoformat(value, fmt) == expected


def test_date_isoformat_default():
    assert date_isoformat(date(2021, 7, 9)) == "2021-07-09"


def test_date_isoformat_expanded():
    assert date_isoformat(date(1985, 4, 12), DATE_EXT_COMPLETE, 6) == "+001985-04-12"
    assert date_isoformat(date(1985, 4, 12), DATE_CENTURY, 6) == "+0019"


@pytest.mark.parametrize(
    "fmt",
    [DATE_EXT_COMPLETE, DATE_BAS_COMPLETE, DATE_EXT_ORD_COMPLETE, DATE_BAS_ORD_COMPLETE],
)
@pytest.mark.parametrize(
    "value",
    [date(1985, 4, 12), date(2008, 12, 29), date(2010, 1, 3), date(2000, 2, 29), date(1, 1, 1)],
)
def test_date_roundtrip(value, fmt):
    assert parse_date(date_isoformat(value, fmt)) == value


@pytest.mark.parametrize("fmt", [DATE_EXT_WEEK_COMPLETE, DATE_BAS_WEEK_COMPLETE])
@pytest.mark.parametrize(
    "value", [date(1985, 4, 12), date(2000, 2, 29), date(2021, 6, 6), date(2021, 1, 4)]
)
def test_week_date_roundtrip(value, fmt):
    # %Y is the calendar year, so only dates whose ISO year matches round-trip
    assert parse_date(date_isoformat(value, fmt)) == value


# ---------------------------------------------------------------------------
# Times
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
        ("232050,5", time(23, 20, 50, 500000)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.1234567", time(23, 20, 50, 123456)),  # truncated, not rounded
        ("23:20:50.9999999", time(23, 20, 50, 999999)),
        ("2320,8", time(23, 20, 48)),
        ("23:20,8", time(23, 20, 48)),
        ("23:20.25", time(23, 20, 15)),
        ("23,3", time(23, 18)),
        ("23.5", time(23, 30)),
        ("00:00:00", time(0, 0, 0)),
        ("14:39:22.000001", time(14, 39, 22, 1)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "text, expected_time, expected_offset",
    [
        ("232030Z", time(23, 20, 30), offset(0)),
        ("23:20:30Z", time(23, 20, 30), offset(0)),
        ("2320Z", time(23, 20), offset(0)),
        ("23Z", time(23), offset(0)),
        ("152746+0100", time(15, 27, 46), offset(1)),
        ("15:27:46+01:00", time(15, 27, 46), offset(1)),
        ("15:27:46+01", time(15, 27, 46), offset(1)),
        ("152746-0500", time(15, 27, 46), offset(-5)),
        ("15:27:46-05:00", time(15, 27, 46), offset(-5)),
        ("15:27:46-05", time(15, 27, 46), offset(-5)),
        ("15:27:46+05:30", time(15, 27, 46), offset(5, 30)),
        ("15:27:46-03:30", time(15, 27, 46), offset(-3, -30)),
        ("15:27:46.5+01:00", time(15, 27, 46, 500000), offset(1)),
        ("1527+0100", time(15, 27), offset(1)),
        ("15+0100", time(15), offset(1)),
    ],
)
def test_parse_time_with_tz(text, expected_time, expected_offset):
    result = parse_time(text)
    assert result.replace(tzinfo=None) == expected_time
    assert result.tzinfo is not None
    assert result.utcoffset() == expected_offset


@pytest.mark.parametrize(
    "text", ["", "2", "23:2", "232", "23:20:5", "23-20-50", "ab:cd", "23:20:50X", "23:20:50+1"]
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["25:00:00", "23:60:00", "23:59:61"])
def test_parse_time_out_of_range(text):
    with pytest.raises(ValueError):
        parse_time(text)


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (time(23, 20, 50), TIME_EXT_COMPLETE, "23:20:50"),
        (time(23, 20, 50), TIME_BAS_COMPLETE, "232050"),
        (time(23, 20, 50), TIME_EXT_MINUTE, "23:20"),
        (time(23, 20, 50), TIME_BAS_MINUTE, "2320"),
        (time(23, 20, 50), TIME_HOUR, "23"),
        (time(1, 2, 3, 45), TIME_EXT_COMPLETE + ".%f", "01:02:03.000045"),
        (time(23, 20, 50, tzinfo=UTC), TIME_EXT_COMPLETE + TZ_EXT, "23:20:50Z"),
        (time(23, 20, 50, tzinfo=UTC), TIME_EXT_COMPLETE + TZ_BAS, "23:20:50Z"),
        (time(23, 20, 50, tzinfo=FixedOffset(1, 0, "+01")), TIME_EXT_COMPLETE + TZ_EXT, "23:20:50+01:00"),
        (time(23, 20, 50, tzinfo=FixedOffset(1, 0, "+01")), TIME_BAS_COMPLETE + TZ_BAS, "232050+0100"),
        (time(23, 20, 50, tzinfo=FixedOffset(1, 0, "+01")), TIME_HOUR + TZ_HOUR, "23+01"),
        (time(23, 20, 50, tzinfo=FixedOffset(-5, 0, "-05")), TIME_EXT_COMPLETE + TZ_EXT, "23:20:50-05:00"),
        (time(23, 20, 50, tzinfo=FixedOffset(-5, -30, "x")), TIME_BAS_COMPLETE + TZ_BAS, "232050-0530"),
        (time(23, 20, 50, tzinfo=FixedOffset(5, 45, "x")), TIME_EXT_COMPLETE + TZ_EXT, "23:20:50+05:45"),
        (time(23, 20, 50), TIME_EXT_COMPLETE + TZ_EXT, "23:20:50"),
    ],
)
def test_time_isoformat(value, fmt, expected):
    assert time_isoformat(value, fmt) == expected


def test_time_isoformat_default():
    assert time_isoformat(time(9, 8, 7)) == "09:08:07"
    assert time_isoformat(time(9, 8, 7, tzinfo=FixedOffset(2, 0, "+02:00"))) == "09:08:07+02:00"


@pytest.mark.parametrize(
    "text",
    ["23:20:50Z", "23:20:50+01:00", "23:20:50-05:00", "00:00:00+14:00", "12:34:56-09:30"],
)
def test_time_tz_roundtrip(text):
    assert time_isoformat(parse_time(text), TIME_EXT_COMPLETE + TZ_EXT) == text


# ---------------------------------------------------------------------------
# Time zones
# ---------------------------------------------------------------------------


def test_parse_tzinfo_utc():
    assert parse_tzinfo("Z") is UTC


def test_parse_tzinfo_empty_is_none():
    assert parse_tzinfo("") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+01:00", offset(1)),
        ("+0100", offset(1)),
        ("+01", offset(1)),
        ("-01:00", offset(-1)),
        ("-0100", offset(-1)),
        ("-01", offset(-1)),
        ("+05:30", offset(5, 30)),
        ("-03:30", offset(-3, -30)),
        ("+00:00", offset(0)),
        ("-00:00", offset(0)),
        ("+14:00", offset(14)),
        ("-12:00", offset(-12)),
    ],
)
def test_parse_tzinfo_offsets(text, expected):
    tz = parse_tzinfo(text)
    assert tz.utcoffset(None) == expected
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == text


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    assert build_tzinfo("+02:30", "+", 2, 30).utcoffset(None) == offset(2, 30)
    assert build_tzinfo("-02:30", "-", 2, 30).utcoffset(None) == offset(-2, -30)


def test_parse_tzinfo_no_designator_prefix():
    # TZ_RE.match permits an empty match, so a string without a designator
    # yields "no time zone" rather than an offset.
    assert parse_tzinfo("garbage") is None


@pytest.mark.parametrize(
    "tz, fmt, expected",
    [
        (UTC, "%Z", "Z"),
        (UTC, "%z", "Z"),
        (UTC, "%h", "Z"),
        (FixedOffset(0, 0, "UTC"), "%Z", "Z"),
        (FixedOffset(1, 0), "%Z", "+01:00"),
        (FixedOffset(1, 0), "%z", "+0100"),
        (FixedOffset(1, 0), "%h", "+01"),
        (FixedOffset(-1, 0), "%Z", "-01:00"),
        (FixedOffset(-1, 0), "%z", "-0100"),
        (FixedOffset(-1, 0), "%h", "-01"),
        (FixedOffset(5, 30), "%Z", "+05:30"),
        (FixedOffset(-9, -30), "%Z", "-09:30"),
        (FixedOffset(-9, -30), "%z", "-0930"),
        (FixedOffset(12, 45), "%z", "+1245"),
    ],
)
def test_tz_isoformat(tz, fmt, expected):
    assert tz_isoformat(datetime(2012, 12, 25, 13, 30, tzinfo=tz), fmt) == expected


def test_tz_isoformat_naive():
    assert tz_isoformat(datetime(2012, 12, 25, 13, 30)) == ""


def test_tz_isoformat_bad_format():
    with pytest.raises(ValueError):
        tz_isoformat(datetime(2012, 1, 1, tzinfo=FixedOffset(1, 0)), "%x")


def test_tz_isoformat_overflow():
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2012, 1, 1, tzinfo=_BigOffset()))


class _BigOffset(FixedOffset):
    def utcoffset(self, dt):
        return timedelta(hours=100)


def test_fixed_offset():
    tz = FixedOffset(-5, -30, "EST-ish")
    assert tz.utcoffset(None) == offset(-5, -30)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "EST-ish"
    assert repr(tz) == "<FixedOffset 'EST-ish'>"
    default = FixedOffset()
    assert default.utcoffset(None) == timedelta(0)
    assert default.tzname(None) == "UTC"


def test_utc():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"


def test_utc_pickle_singleton():
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(UTC, proto)) is UTC


def test_local_timezone_consistency():
    dt = datetime(2020, 6, 15, 12, 0)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert isinstance(LOCAL.tzname(dt), str)
    assert isinstance(LOCAL.dst(dt), timedelta)
    # standard offset plus DST adjustment must be consistent with utcoffset
    winter = datetime(2020, 1, 15, 12, 0)
    for d in (dt, winter):
        assert LOCAL.utcoffset(d) - LOCAL.dst(d) in (
            LOCAL.utcoffset(dt) - LOCAL.dst(dt),
            LOCAL.utcoffset(winter) - LOCAL.dst(winter),
        )
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


def test_tz_aware_comparison_uses_sign():
    # 10:00+05:00 and 05:00Z are the same instant
    a = parse_datetime("2020-01-01T10:00:00+05:00")
    b = parse_datetime("2020-01-01T05:00:00Z")
    assert a == b
    c = parse_datetime("2020-01-01T10:00:00-05:00")
    d = parse_datetime("2020-01-01T15:00:00Z")
    assert c == d


# ---------------------------------------------------------------------------
# Datetimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("19850412T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985102T1015Z", datetime(1985, 4, 12, 10, 15, tzinfo=UTC)),
        ("1985-102T10:15Z", datetime(1985, 4, 12, 10, 15, tzinfo=UTC)),
        ("1985W155T1015+0400", datetime(1985, 4, 12, 10, 15, tzinfo=FixedOffset(4, 0))),
        ("1985-W15-5T10:15+04", datetime(1985, 4, 12, 10, 15, tzinfo=FixedOffset(4, 0))),
        ("1985-W15-5T10:15-0430", datetime(1985, 4, 12, 10, 15, tzinfo=FixedOffset(-4, -30))),
        ("2012-10-30T08:55:22.1234567Z", datetime(2012, 10, 30, 8, 55, 22, 123456, tzinfo=UTC)),
        ("2012-10-30T08:55:22.1234561Z", datetime(2012, 10, 30, 8, 55, 22, 123456, tzinfo=UTC)),
        ("2014-08-18T14:55:22.123456Z", datetime(2014, 8, 18, 14, 55, 22, 123456, tzinfo=UTC)),
    ],
)
def test_parse_datetime(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert (result.tzinfo is None) == (expected.tzinfo is None)
    if expected.tzinfo is not None:
        assert result.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15", "1985-04-12T10:15T", "x"])
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (datetime(1985, 4, 12, 10, 15, 30), DT_EXT_COMPLETE, "1985-04-12T10:15:30"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC), DT_EXT_COMPLETE, "1985-04-12T10:15:30Z"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0)), DT_EXT_COMPLETE, "1985-04-12T10:15:30+04:00"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0)), DT_BAS_COMPLETE, "19850412T101530+0400"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-4, 0)), DT_BAS_COMPLETE, "19850412T101530-0400"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC), DT_EXT_ORD_COMPLETE, "1985-102T10:15:30Z"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC), DT_BAS_ORD_COMPLETE, "1985102T101530Z"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC), DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30Z"),
        (datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC), DT_BAS_WEEK_COMPLETE, "1985W155T101530Z"),
        (datetime(2012, 10, 12, 8, 29, 46, 69178), DT_EXT_COMPLETE + ".%f", "2012-10-12T08:29:46.069178"),
        (datetime(2012, 10, 12, 8, 29, 46, 691780), "%Y-%m-%dT%H:%M:%S.%f", "2012-10-12T08:29:46.691780"),
        (datetime(2012, 1, 1), "100%% %Y", "100% 2012"),
    ],
)
def test_datetime_isoformat(value, fmt, expected):
    assert datetime_isoformat(value, fmt) == expected


def test_datetime_isoformat_default():
    dt = datetime(2021, 3, 4, 5, 6, 7, tzinfo=FixedOffset(-7, 0, "-07:00"))
    assert datetime_isoformat(dt) == "2021-03-04T05:06:07-07:00"


@pytest.mark.parametrize(
    "text",
    [
        "1985-04-12T10:15:30Z",
        "1985-04-12T10:15:30+04:00",
        "1985-04-12T10:15:30-04:00",
        "2000-02-29T00:00:00+05:30",
        "2000-02-29T23:59:59-11:45",
    ],
)
def test_datetime_roundtrip(text):
    assert datetime_isoformat(parse_datetime(text)) == text


def test_datetime_matches_stdlib():
    for text in ["2020-01-01T10:00:00+05:00", "2020-07-01T10:00:00-07:30", "2020-07-01T10:00:00Z"]:
        ours = parse_datetime(text)
        std = datetime.fromisoformat(text.replace("Z", "+00:00"))
        assert ours == std
        assert ours.utcoffset() == std.utcoffset()


# ---------------------------------------------------------------------------
# Durations: parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P18Y9M4DT11H9M8S", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P2W", timedelta(weeks=2)),
        ("P3Y6M4DT12H30M5S", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P23DT23H", timedelta(hours=23, days=23)),
        ("P4Y", Duration(years=4)),
        ("P1M", Duration(months=1)),
        ("PT1M", timedelta(minutes=1)),
        ("P0.5Y", Duration(years=0.5)),
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
        ("-P3Y6M4DT12H30M5S", Duration(-4, -5, 0, 0, -30, -12, 0, -6, -3)),
        ("-P1DT2H3M4S", timedelta(days=-1, hours=-2, minutes=-3, seconds=-4)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("P1Y2M", Duration(years=1, months=2)),
        ("P1,5Y", Duration(years=Decimal("1.5"))),
    ],
)
def test_parse_duration(text, expected):
    result = parse_duration(text)
    assert type(result) is type(expected)
    assert result == expected


def test_parse_duration_as_duration():
    result = parse_duration("P1DT2H", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result == Duration(days=1, hours=2)
    assert result.years == 0 and result.months == 0
    neg = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(neg, Duration)
    assert neg.tdelta == timedelta(days=-1)


def test_parse_duration_year_month_are_decimal():
    result = parse_duration("P1.5Y2.25M")
    assert result.years == Decimal("1.5")
    assert result.months == Decimal("2.25")
    assert isinstance(result.years, Decimal)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0018-09-04T11:09:08", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P00180904T110908", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P0003-06-04T12:30:05", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P0003-06-04T12:30:05.5", Duration(4, 5.5, 0, 0, 30, 12, 0, 6, 3)),
    ],
)
def test_parse_duration_alternative(text, expected):
    result = parse_duration(text)
    assert result == expected
    assert isinstance(result, Duration)


@pytest.mark.parametrize(
    "text", ["", "P", "1Y", "P1", "P1S", "PT1D", "P1Y1Y", "-", "P-1Y", "XP1Y", "P1H"]
)
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_non_string():
    with pytest.raises(TypeError):
        parse_duration(1)
    with pytest.raises(TypeError):
        parse_duration(None)


# ---------------------------------------------------------------------------
# Durations: formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(days=1), "P1D"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(days=23, hours=23), "P23DT23H"),
        (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
        (timedelta(days=-1, hours=-2, minutes=-3, seconds=-4), "-P1DT2H3M4S"),
        (timedelta(hours=-1), "-PT1H"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(weeks=2), "P14D"),
        (Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8), "P18Y9M4DT11H9M8S"),
        (Duration(years=3, months=6, days=4, hours=12, minutes=30, seconds=5), "P3Y6M4DT12H30M5S"),
        (Duration(years=4), "P4Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=-2), "-P2Y"),
        (Duration(months=-1), "-P1M"),
        (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
        (Duration(years=Decimal("0.5")), "P0.5Y"),
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
        "PT12H",
        "P1DT2H3M4S",
        "-P1DT2H3M4S",
        "-P3Y6M4DT12H30M5S",
        "PT22.22S",
        "-P2Y",
    ],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


def test_duration_week_format():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=15), D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(weeks=-3), D_WEEK) == "-P3W"


def test_duration_alternative_formats():
    dur = Duration(years=18, months=9, days=4, hours=11, minutes=9, seconds=8)
    assert duration_isoformat(dur, D_ALT_EXT) == "P0018-09-04T11:09:08"
    assert duration_isoformat(dur, D_ALT_BAS) == "P00180904T110908"
    assert parse_duration(duration_isoformat(dur, D_ALT_EXT)) == dur
    assert parse_duration(duration_isoformat(dur, D_ALT_BAS)) == dur


def test_strftime_duration_codes():
    td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    assert strftime(td, "%d %H %M %S %f") == "03 04 05 06 000007"
    assert strftime(timedelta(days=14), "%W") == "02"
    assert strftime(timedelta(0), "%%") == "%"
    dur = Duration(years=2021, months=7)
    assert strftime(dur, "%Y-%m") == "2021-07"
    assert strftime(dur, "%C") == "20"
    assert strftime(dur, "%Y", 6) == "+002021"
    # unknown codes pass through
    assert strftime(td, "%x") == "%x"


def test_strftime_date_codes():
    d = datetime(2021, 1, 3, 4, 5, 6, 7)
    assert strftime(d, "%d|%f|%H|%j|%m|%M|%S|%w|%W|%Y|%C|%%") == (
        "03|000007|04|003|01|05|06|7|53|2021|20|%"
    )
    assert strftime(date(2021, 1, 4), "%w %W") == "1 01"
    assert strftime(time(1, 2, 3), "%x") == "%x"


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


def test_duration_init_and_attrs():
    dur = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=7, months=8, years=9)
    assert dur.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert dur.months == Decimal(8)
    assert dur.years == Decimal(9)
    assert isinstance(dur.months, Decimal)
    # timedelta attributes are proxied
    assert dur.days == dur.tdelta.days
    assert dur.seconds == dur.tdelta.seconds
    assert dur.microseconds == dur.tdelta.microseconds
    assert dur.total_seconds() == dur.tdelta.total_seconds()


def test_duration_float_months_converted_exactly():
    assert Duration(months=0.1).months == Decimal("0.1")


def test_duration_str_repr():
    assert str(Duration(years=2, months=3, days=4)) == "2 years, 3 months, 4 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    assert repr(Duration(1, 2, 3, years=4, months=5)) == (
        "isodate.duration.Duration(1, 2, 3, years=4, months=5)"
    )


def test_duration_hash():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=1, months=2, days=3)
    c = Duration(years=1, months=2, days=4)
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2
    assert {a: 1}[b] == 1


def test_duration_neg():
    dur = Duration(years=1, months=2, days=3, hours=4)
    neg = -dur
    assert neg.years == -1
    assert neg.months == -2
    assert neg.tdelta == -timedelta(days=3, hours=4)
    assert -neg == dur
    # original unchanged
    assert dur.years == 1


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == Duration(hours=24)
    assert Duration(days=1) != Duration(days=2)
    assert Duration(years=1, days=1) != Duration(years=1, days=2)
    assert not (Duration(years=1) == Duration(years=1, days=1))
    assert not (Duration(years=1) != Duration(months=12))
    # comparison with timedelta
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1) != 1


def test_duration_add_duration():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=4, months=5, hours=6)
    s = a + b
    assert s.years == 5 and s.months == 7
    assert s.tdelta == timedelta(days=3, hours=6)
    # operands unchanged
    assert a.years == 1 and b.years == 4


def test_duration_add_timedelta():
    dur = Duration(years=1, days=1)
    for res in (dur + timedelta(hours=2), timedelta(hours=2) + dur):
        assert isinstance(res, Duration)
        assert res.years == 1
        assert res.tdelta == timedelta(days=1, hours=2)


def test_duration_sub():
    a = Duration(years=3, months=5, days=10)
    b = Duration(years=1, months=7, days=4)
    d = a - b
    assert d.years == 2 and d.months == -2
    assert d.tdelta == timedelta(days=6)
    assert d == Duration(months=22, days=6)
    e = a - timedelta(days=3)
    assert isinstance(e, Duration)
    assert e.years == 3 and e.months == 5 and e.tdelta == timedelta(days=7)
    f = timedelta(days=12) - a
    assert isinstance(f, Duration)
    assert f.years == -3 and f.months == -5 and f.tdelta == timedelta(days=2)


def test_duration_mul():
    dur = Duration(years=1, months=2, days=3)
    for res in (dur * 3, 3 * dur):
        assert res.years == 3 and res.months == 6
        assert res.tdelta == timedelta(days=9)
    assert dur * -1 == -dur
    with pytest.raises(TypeError):
        dur * 1.5
    with pytest.raises(TypeError):
        dur * dur


def test_duration_unsupported_ops():
    dur = Duration(years=1)
    with pytest.raises(TypeError):
        dur + "x"
    with pytest.raises(TypeError):
        dur - "x"
    with pytest.raises(TypeError):
        "x" - dur


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 1), Duration(years=1), date(2001, 1, 1)),
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 12, 15), Duration(months=13), date(2002, 1, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 1), Duration(months=11), date(2000, 12, 1)),
        (date(2000, 1, 1), Duration(months=12), date(2001, 1, 1)),
        (date(2000, 1, 1), Duration(years=1, months=2, days=3), date(2001, 3, 4)),
        (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
        (
            datetime(2000, 1, 31, 23, 0),
            Duration(months=1, hours=2),
            datetime(2000, 3, 1, 1, 0),
        ),
        (
            datetime(2000, 1, 1, tzinfo=UTC),
            Duration(years=1),
            datetime(2001, 1, 1, tzinfo=UTC),
        ),
    ],
)
def test_duration_add_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2001, 1, 1), Duration(years=1), date(2000, 1, 1)),
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2001, 1, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2001, 1, 15), Duration(months=13), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2001, 3, 4), Duration(years=1, months=2, days=3), date(2000, 1, 1)),
        (datetime(2000, 3, 1, 1, 0), Duration(hours=2), datetime(2000, 2, 29, 23, 0)),
    ],
)
def test_duration_sub_from_date(start, dur, expected):
    assert start - dur == expected


def test_duration_date_arithmetic_preserves_type():
    assert type(date(2000, 1, 1) + Duration(months=1)) is date
    assert type(datetime(2000, 1, 1) + Duration(months=1)) is datetime
    assert type(datetime(2000, 1, 1) - Duration(months=1)) is datetime


def test_duration_fractional_months_with_date():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=1.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=0.5)


def test_duration_from_parse_applied_to_date():
    assert parse_date("2000-01-31") + parse_duration("P1M") == date(2000, 2, 29)
    assert parse_datetime("2000-01-01T00:00:00Z") + parse_duration("P1Y2M3DT4H") == datetime(
        2001, 3, 4, 4, tzinfo=UTC
    )
    assert parse_date("2000-03-15") + parse_duration("-P1M") == date(2000, 2, 15)


def test_totimedelta():
    dur = Duration(months=1)
    assert dur.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert dur.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert dur.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert dur.totimedelta(end=date(2001, 1, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=366)
    assert Duration(days=2, hours=3).totimedelta(start=date(2000, 1, 1)) == timedelta(days=2)
    with pytest.raises(ValueError):
        dur.totimedelta()
    with pytest.raises(ValueError):
        dur.totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_duration_pickle():
    dur = Duration(years=1, months=2, days=3, hours=4)
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(dur, proto))
        assert restored == dur
        assert restored.years == dur.years
        assert restored.tdelta == dur.tdelta


# ---------------------------------------------------------------------------
# Helpers in duration module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "year, month, expected",
    [
        (2001, 1, 31),
        (2001, 2, 28),
        (2004, 2, 29),
        (1900, 2, 28),
        (2000, 2, 29),
        (2100, 2, 28),
        (2001, 4, 30),
        (2001, 6, 30),
        (2001, 9, 30),
        (2001, 11, 30),
        (2001, 12, 31),
        (2001, 7, 31),
        (2001, 8, 31),
    ],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


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
        (Decimal(7), 0, 5, (1, Decimal(2))),
        (Decimal(-1), 0, 5, (-1, Decimal(4))),
    ],
)
def test_fquotmod(val, low, high, expected):
    assert fquotmod(val, low, high) == expected


def test_public_api():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert isinstance(isodate.__version__, str)
