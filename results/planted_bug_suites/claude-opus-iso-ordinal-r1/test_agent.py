"""Test suite for isodate: parsing and formatting of ISO 8601 dates, times,
datetimes, time zones and durations, plus Duration arithmetic."""

import pickle
from datetime import date, datetime, time, timedelta
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
    D_ALT_EXT,
    D_DEFAULT,
    D_WEEK,
    DT_BAS_COMPLETE,
    DT_EXT_COMPLETE,
    DT_EXT_ORD_COMPLETE,
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


# --------------------------------------------------------------------------
# parse_date
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("19", date(1901, 1, 1)),  # century
        ("1985", date(1985, 1, 1)),  # year
        ("1985-04", date(1985, 4, 1)),  # extended month
        ("198504", date(1985, 4, 1)),  # basic month
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("1985-102", date(1985, 4, 12)),  # extended ordinal
        ("1985102", date(1985, 4, 12)),  # basic ordinal
        ("1985-001", date(1985, 1, 1)),
        ("1985001", date(1985, 1, 1)),
        ("1985-365", date(1985, 12, 31)),
        ("2000-060", date(2000, 2, 29)),
        ("2000-366", date(2000, 12, 31)),
        ("2001-060", date(2001, 3, 1)),
        ("1985-W15-5", date(1985, 4, 12)),  # extended week date
        ("1985W155", date(1985, 4, 12)),  # basic week date
        ("1985-W15", date(1985, 4, 8)),  # reduced week (Monday)
        ("1985W15", date(1985, 4, 8)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2009-W01-1", date(2008, 12, 29)),  # week 1 begins previous year
        ("2009-W53-7", date(2010, 1, 3)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2010-W01-1", date(2010, 1, 4)),  # Jan 1 2010 is in week 53 of 2009
        ("2010-W01", date(2010, 1, 4)),
        ("2015-W01-4", date(2015, 1, 1)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
    ],
)
def test_parse_date_basic_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("day", [1, 32, 59, 60, 100, 200, 365])
def test_parse_ordinal_date_matches_toordinal(day):
    expected = date.fromordinal(date(2023, 1, 1).toordinal() + day - 1)
    assert parse_date("2023-%03d" % day) == expected
    assert parse_date("2023%03d" % day) == expected


@pytest.mark.parametrize("year", [2000, 2004, 2008, 2009, 2010, 2015, 2020, 2021])
def test_parse_week_date_matches_isocalendar(year):
    d = date(year, 1, 1)
    while d.year == year:
        y, w, wd = d.isocalendar()
        assert parse_date("%04d-W%02d-%d" % (y, w, wd)) == d
        assert parse_date("%04dW%02d%d" % (y, w, wd)) == d
        d += timedelta(days=5)


@pytest.mark.parametrize(
    "text, yeardigits, expected",
    [
        ("+0019", 6, date(1901, 1, 1)),
        ("+001985", 6, date(1985, 1, 1)),
        ("+001985-04", 6, date(1985, 4, 1)),
        ("+001985-04-12", 6, date(1985, 4, 12)),
        ("+0019850412", 6, date(1985, 4, 12)),
        ("+001985-102", 6, date(1985, 4, 12)),
        ("+001985102", 6, date(1985, 4, 12)),
        ("+001985-W15-5", 6, date(1985, 4, 12)),
        ("+001985W155", 6, date(1985, 4, 12)),
        ("+001985-W15", 6, date(1985, 4, 8)),
        ("+01985-04-12", 5, date(1985, 4, 12)),
    ],
)
def test_parse_date_expanded(text, yeardigits, expected):
    assert parse_date(text, yeardigits) == expected


def test_parse_date_expanded_four_digit_years():
    assert parse_date("+1985-04-12", 4, expanded=True) == date(1985, 4, 12)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")


def test_parse_date_expanded_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", 6)


def test_parse_date_negative_year_not_representable():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", 6)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    # defaults must not affect complete dates
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "abc",
        "1",
        "198",
        "1985-4-12",
        "1985-04-12T",
        "1985/04/12",
        "1985-W1",
        "85-04-12",
        " 1985-04-12",
        "1985-04-12 ",
        "1985-04-1",
    ],
)
def test_parse_date_invalid_format(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1985-04-00", "2001-02-29"])
def test_parse_date_invalid_values(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_iso8601error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)


# --------------------------------------------------------------------------
# date formatting
# --------------------------------------------------------------------------

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


def test_date_isoformat_default():
    assert date_isoformat(date(2023, 1, 5)) == "2023-01-05"
    assert date_isoformat(date(5, 1, 5)) == "0005-01-05"


def test_date_isoformat_expanded():
    assert date_isoformat(date(1985, 4, 12), DATE_EXT_COMPLETE, 6) == "+001985-04-12"
    assert date_isoformat(date(1985, 4, 12), DATE_CENTURY, 6) == "+0019"


@pytest.mark.parametrize(
    "d, expected",
    [
        (date(2000, 1, 1), "2000-001"),
        (date(2000, 12, 31), "2000-366"),
        (date(2001, 12, 31), "2001-365"),
        (date(2000, 3, 1), "2000-061"),
    ],
)
def test_date_isoformat_ordinal(d, expected):
    assert date_isoformat(d, DATE_EXT_ORD_COMPLETE) == expected


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
    end = date(2002, 1, 10)
    while d < end:
        assert parse_date(date_isoformat(d, fmt)) == d
        d += timedelta(days=3)


@pytest.mark.parametrize("fmt", [DATE_EXT_WEEK_COMPLETE, DATE_BAS_WEEK_COMPLETE])
def test_week_date_roundtrip_away_from_year_boundary(fmt):
    # %Y is the calendar year, so only dates whose ISO year equals the
    # calendar year round-trip through the week formats.
    d = date(1999, 1, 1)
    while d < date(2003, 1, 1):
        if d.isocalendar()[0] == d.year:
            assert parse_date(date_isoformat(d, fmt)) == d
        d += timedelta(days=1)


def test_week_date_formatting_uses_iso_year_boundary():
    # 2008-12-29 is in ISO week 1 of 2009; the formatter uses calendar year.
    assert date_isoformat(date(2008, 12, 29), DATE_EXT_WEEK_COMPLETE) == "2008-W01-1"
    assert date_isoformat(date(2010, 1, 3), DATE_EXT_WEEK_COMPLETE) == "2010-W53-7"


# --------------------------------------------------------------------------
# parse_time
# --------------------------------------------------------------------------

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
        ("23:20.5", time(23, 20, 30)),
        ("23,3", time(23, 18)),
        ("23.5", time(23, 30)),
        ("00:00:00", time(0, 0, 0)),
        ("15:27:46.5", time(15, 27, 46, 500000)),
        ("14:00.0001", time(14, 0, 0, 6000)),
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
        ("23:20:30+00:00", timedelta(0)),
        ("23:20:30+01:00", timedelta(hours=1)),
        ("23:20:30+0100", timedelta(hours=1)),
        ("23:20:30+01", timedelta(hours=1)),
        ("23:20:30-05:00", timedelta(hours=-5)),
        ("23:20:30-0530", timedelta(hours=-5, minutes=-30)),
        ("23:20:30+05:30", timedelta(hours=5, minutes=30)),
        ("2320-0130", timedelta(hours=-1, minutes=-30)),
        ("23-07", timedelta(hours=-7)),
        ("152746+0100", timedelta(hours=1)),
        ("15:27:46.5+01:00", timedelta(hours=1)),
    ],
)
def test_parse_time_with_tz(text, offset):
    result = parse_time(text)
    assert result.tzinfo is not None
    assert result.utcoffset() == offset


def test_parse_time_z_is_utc_singleton():
    assert parse_time("10:00Z").tzinfo is UTC


def test_parse_time_tz_values():
    t = parse_time("23:20:50.5-05:30")
    assert (t.hour, t.minute, t.second, t.microsecond) == (23, 20, 50, 500000)


@pytest.mark.parametrize(
    "text", ["", "2", "23:2", "23:20:5", "232", "23:20:50Z+01", "T", "ab:cd", "23:20:50.", "23::20"]
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["24:00:01", "23:60:00", "25"])
def test_parse_time_out_of_range(text):
    with pytest.raises(ValueError):
        parse_time(text)


# --------------------------------------------------------------------------
# time formatting
# --------------------------------------------------------------------------

def test_time_isoformat_default():
    assert time_isoformat(time(23, 20, 50)) == "23:20:50"
    assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (TIME_EXT_COMPLETE, "23:20:50"),
        (TIME_BAS_COMPLETE, "232050"),
        (TIME_EXT_MINUTE, "23:20"),
        (TIME_BAS_MINUTE, "2320"),
        (TIME_HOUR, "23"),
        (TIME_EXT_COMPLETE + ".%f", "23:20:50.000042"),
        ("%%H", "%H"),
    ],
)
def test_time_isoformat_formats(fmt, expected):
    assert time_isoformat(time(23, 20, 50, 42), fmt) == expected


@pytest.mark.parametrize(
    "tz, fmt, expected",
    [
        (FixedOffset(-5, -30, "x"), TZ_EXT, "-05:30"),
        (FixedOffset(-5, -30, "x"), TZ_BAS, "-0530"),
        (FixedOffset(-5, -30, "x"), TZ_HOUR, "-05"),
        (FixedOffset(1, 0, "x"), TZ_EXT, "+01:00"),
        (FixedOffset(1, 0, "x"), TZ_BAS, "+0100"),
        (FixedOffset(1, 0, "x"), TZ_HOUR, "+01"),
        (UTC, TZ_EXT, "Z"),
        (UTC, TZ_BAS, "Z"),
        (FixedOffset(0, 0, "x"), TZ_EXT, "Z"),
        (None, TZ_EXT, ""),
    ],
)
def test_time_isoformat_tz(tz, fmt, expected):
    assert time_isoformat(time(12, 0, tzinfo=tz), "%H" + fmt) == "12" + expected


# --------------------------------------------------------------------------
# time zones
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, offset",
    [
        ("Z", timedelta(0)),
        ("+01:00", timedelta(hours=1)),
        ("+0100", timedelta(hours=1)),
        ("+01", timedelta(hours=1)),
        ("-01:30", timedelta(hours=-1, minutes=-30)),
        ("-0130", timedelta(hours=-1, minutes=-30)),
        ("+14:00", timedelta(hours=14)),
        ("-12", timedelta(hours=-12)),
    ],
)
def test_parse_tzinfo(text, offset):
    tz = parse_tzinfo(text)
    assert tz.utcoffset(None) == offset
    assert tz.dst(None) == timedelta(0)


def test_parse_tzinfo_empty_is_none():
    assert parse_tzinfo("") is None


def test_parse_tzinfo_name_preserved():
    assert parse_tzinfo("+05:30").tzname(None) == "+05:30"
    assert parse_tzinfo("Z") is UTC


def test_tz_isoformat_direct():
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(-8, 0, "PST"))
    assert tz_isoformat(dt) == "-08:00"
    assert tz_isoformat(dt, "%z") == "-0800"
    assert tz_isoformat(dt, "%h") == "-08"
    assert tz_isoformat(datetime(2020, 1, 1)) == ""
    assert tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%Q")


def test_utc_and_fixed_offset_classes():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert UTC.dst(None) == timedelta(0)
    fo = FixedOffset(2, 15, "CUST")
    assert fo.utcoffset(None) == timedelta(hours=2, minutes=15)
    assert fo.tzname(None) == "CUST"
    assert fo.dst(None) == timedelta(0)
    assert repr(fo) == "<FixedOffset 'CUST'>"
    assert FixedOffset().utcoffset(None) == timedelta(0)


def test_utc_pickle_preserves_singleton():
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_local_timezone_consistent():
    dt = datetime(2020, 7, 1, 12, 0)
    off = isodate.LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert isodate.LOCAL.dst(dt) in (timedelta(0), isodate.tzinfo.DSTDIFF)
    assert isinstance(isodate.LOCAL.tzname(dt), str)


# --------------------------------------------------------------------------
# parse_datetime / datetime_isoformat
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("19850412T1015", datetime(1985, 4, 12, 10, 15)),
        ("1985-04-12T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985102T1015Z", datetime(1985, 4, 12, 10, 15, tzinfo=UTC)),
        ("1985-102T10:15Z", datetime(1985, 4, 12, 10, 15, tzinfo=UTC)),
        ("1985W155T1015+0400", datetime(1985, 4, 12, 10, 15, tzinfo=FixedOffset(4, 0, "x"))),
        ("1985-W15-5T10:15+04", datetime(1985, 4, 12, 10, 15, tzinfo=FixedOffset(4, 0, "x"))),
        ("2012-10-30T08:55:22.7+05:30", datetime(2012, 10, 30, 8, 55, 22, 700000,
                                                 tzinfo=FixedOffset(5, 30, "x"))),
        ("2012-10-12T08:29:46.069178Z", datetime(2012, 10, 12, 8, 29, 46, 69178, tzinfo=UTC)),
        ("2012-10-12T08:29:46,069178Z", datetime(2012, 10, 12, 8, 29, 46, 69178, tzinfo=UTC)),
    ],
)
def test_parse_datetime(text, expected):
    result = parse_datetime(text)
    assert result == expected
    assert result.utcoffset() == expected.utcoffset()


def test_parse_datetime_fields_and_tz():
    dt = parse_datetime("2012-10-30T08:55:22.7-05:30")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond) == (
        2012, 10, 30, 8, 55, 22, 700000
    )
    assert dt.utcoffset() == timedelta(hours=-5, minutes=-30)


def test_parse_datetime_naive():
    assert parse_datetime("2020-02-29T00:00:00").tzinfo is None


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15", "1985-04-12T10:15T11", ""])
def test_parse_datetime_missing_or_extra_t(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


def test_parse_datetime_bad_parts():
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-1xT10:15")
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12T10:1x")


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DT_EXT_COMPLETE, "1985-04-12T10:15:30+04:00"),
        (DT_BAS_COMPLETE, "19850412T101530+0400"),
        (DT_EXT_ORD_COMPLETE, "1985-102T10:15:30+04:00"),
        (isodate.DT_BAS_ORD_COMPLETE, "1985102T101530+0400"),
        (isodate.DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30+04:00"),
        (isodate.DT_BAS_WEEK_COMPLETE, "1985W155T101530+0400"),
    ],
)
def test_datetime_isoformat(fmt, expected):
    dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 0, "x"))
    assert datetime_isoformat(dt, fmt) == expected
    assert parse_datetime(expected) == dt


def test_datetime_isoformat_default():
    assert datetime_isoformat(datetime(2020, 1, 2, 3, 4, 5)) == "2020-01-02T03:04:05"
    assert datetime_isoformat(datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2020-01-02T03:04:05Z"


def test_datetime_roundtrip_with_microseconds():
    dt = datetime(2021, 12, 31, 23, 59, 59, 999999, tzinfo=FixedOffset(-3, -30, "x"))
    s = datetime_isoformat(dt, DATE_EXT_COMPLETE + "T" + TIME_EXT_COMPLETE + ".%f" + TZ_EXT)
    assert s == "2021-12-31T23:59:59.999999-03:30"
    assert parse_datetime(s) == dt


# --------------------------------------------------------------------------
# parse_duration
# --------------------------------------------------------------------------

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
        ("P0.5Y", Duration(years=Decimal("0.5"))),
        ("P0,5Y", Duration(years=Decimal("0.5"))),
        ("PT36H", timedelta(hours=36)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("PT0.5S", timedelta(microseconds=500000)),
        ("PT1,5H", timedelta(hours=1, minutes=30)),
        ("P1W2D", timedelta(days=9)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("+P11D", timedelta(days=11)),
        ("-P2W", timedelta(weeks=-2)),
        ("-P2.2W", timedelta(weeks=-2.2)),
        ("-P3Y", Duration(years=-3)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("P5Y7M", Duration(years=5, months=7)),
        ("PT2H", timedelta(hours=2)),
        ("PT0.0001S", timedelta(microseconds=100)),
    ],
)
def test_parse_duration(text, expected):
    result = parse_duration(text)
    assert result == expected
    assert type(result) is type(expected)


def test_parse_duration_negative_full():
    d = parse_duration("-P1Y2M3DT4H5M6S")
    assert isinstance(d, Duration)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == -timedelta(days=3, hours=4, minutes=5, seconds=6)


def test_parse_duration_negative_timedelta():
    assert parse_duration("-P1DT2H") == -timedelta(days=1, hours=2)
    assert parse_duration("-PT1M") == timedelta(minutes=-1)


def test_parse_duration_decimal_years_months():
    d = parse_duration("P1.5Y2.5M")
    assert d.years == Decimal("1.5")
    assert d.months == Decimal("2.5")
    assert isinstance(d.years, Decimal)


def test_parse_duration_as_duration_forced():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=1)
    assert d.years == 0 and d.months == 0
    neg = parse_duration("-PT1H", as_timedelta_if_possible=False)
    assert isinstance(neg, Duration)
    assert neg.tdelta == timedelta(hours=-1)


def test_parse_duration_month_vs_minute_disambiguation():
    assert parse_duration("P1M") == Duration(months=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    d = parse_duration("P1MT1M")
    assert d.months == 1 and d.tdelta == timedelta(minutes=1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0018-09-04T11:09:08", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P00180904T110908", Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)),
        ("P0003-06-04T12:30:05", Duration(4, 5, 0, 0, 30, 12, 0, 6, 3)),
        ("P0003-06-04T12:30:05.5", Duration(4, 5, 500000, 0, 30, 12, 0, 6, 3)),
    ],
)
def test_parse_duration_alternative_format(text, expected):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result == expected


@pytest.mark.parametrize(
    "text", ["", "P", "1Y", "P1", "PxY", "P1Y2", "P1S", "PT1D", "P1H", "Q1D", "P1D ", " P1D", "P-1D"]
)
def test_parse_duration_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_duration(text)


def test_parse_duration_non_string():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


# --------------------------------------------------------------------------
# duration formatting
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (Duration(4, 8, 0, 0, 9, 11, 0, 9, 18), "P18Y9M4DT11H9M8S"),
        (timedelta(weeks=2), "P14D"),
        (timedelta(days=23, hours=23), "P23DT23H"),
        (Duration(years=4), "P4Y"),
        (Duration(months=1), "P1M"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(0), "P0D"),
        (Duration(), "P0D"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=59), "PT59S"),
        (timedelta(days=-2), "-P2D"),
        (timedelta(hours=-1, minutes=-30), "-PT1H30M"),
        (Duration(years=-1, months=-2), "-P1Y2M"),
        (Duration(days=-3), "-P3D"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


@pytest.mark.parametrize(
    "text",
    ["P18Y9M4DT11H9M8S", "P1DT12H", "PT1.5S", "-P2D", "P4Y", "P1M", "-P1Y2M3DT4H5M6S", "PT1H1M1S"],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


def test_duration_isoformat_week_format():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=17), D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(weeks=-3), D_WEEK) == "-P3W"


def test_duration_isoformat_alt_format():
    d = Duration(4, 8, 0, 0, 9, 11, 0, 9, 18)
    assert duration_isoformat(d, D_ALT_EXT) == "P0018-09-04T11:09:08"


def test_strftime_duration_codes():
    td = timedelta(days=5, hours=3, minutes=7, seconds=9, microseconds=12)
    assert strftime(td, "%d %H %M %S %f %%") == "05 03 07 09 000012 %"
    assert strftime(timedelta(weeks=3), "%W") == "03"
    d = Duration(years=12, months=3)
    assert strftime(d, "%Y-%m") == "0012-03"
    assert strftime(d, "%Y", 6) == "+000012"
    assert strftime(Duration(years=1985), "%C") == "19"


def test_strftime_unknown_code_passthrough():
    assert strftime(date(2020, 1, 1), "%Y %q") == "2020 %q"
    assert strftime(timedelta(days=1), "%d %q") == "01 %q"


def test_strftime_dt_codes():
    dt = datetime(2003, 7, 9, 4, 5, 6, 7)
    assert strftime(dt, "%Y|%m|%d|%H|%M|%S|%f|%j|%w|%W|%C|%%") == (
        "2003|07|09|04|05|06|000007|190|3|28|20|%"
    )


# --------------------------------------------------------------------------
# Duration class
# --------------------------------------------------------------------------

def test_duration_constructor_and_attrs():
    d = Duration(days=1, hours=2, weeks=1, months=3, years=4)
    assert d.years == Decimal(4)
    assert d.months == Decimal(3)
    assert d.tdelta == timedelta(days=8, hours=2)
    # timedelta attributes are proxied
    assert d.days == 8
    assert d.seconds == 7200
    assert d.microseconds == 0
    assert isinstance(d.months, Decimal) and isinstance(d.years, Decimal)
    assert Duration(months=1.5).months == Decimal("1.5")


def test_duration_getattr_missing():
    with pytest.raises(AttributeError):
        Duration().nonexistent


def test_duration_str_and_repr():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
    assert r == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash():
    a = Duration(days=1, months=2, years=3)
    b = Duration(days=1, months=2, years=3)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert hash(a) != hash(Duration(days=1, months=3, years=3))


def test_duration_pickle():
    d = Duration(days=1, hours=2, months=3, years=Decimal("4.5"))
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        loaded = pickle.loads(pickle.dumps(d, proto))
        assert loaded == d
        assert loaded.years == Decimal("4.5")


def test_duration_neg():
    d = -Duration(days=1, hours=2, months=3, years=4)
    assert d.years == -4
    assert d.months == -3
    assert d.tdelta == -timedelta(days=1, hours=2)
    assert -(-d) == d


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(days=1, years=1) != timedelta(days=1)
    assert not (Duration(days=1, years=1) == timedelta(days=1))
    assert Duration(days=1) != Duration(days=2)
    assert not (Duration(days=1) != Duration(days=1))
    assert Duration(months=1) != Duration(months=1, days=1)
    assert Duration() != "foo"
    assert not (Duration(days=1) == "x")


def test_duration_add_durations():
    a = Duration(days=1, months=1, years=1)
    b = Duration(hours=12, months=2, years=3)
    c = a + b
    assert c.years == 4
    assert c.months == 3
    assert c.tdelta == timedelta(days=1, hours=12)


def test_duration_add_timedelta_both_ways():
    d = Duration(days=1, months=1)
    r1 = d + timedelta(hours=1)
    r2 = timedelta(hours=1) + d
    for r in (r1, r2):
        assert isinstance(r, Duration)
        assert r.months == 1
        assert r.tdelta == timedelta(days=1, hours=1)


def test_duration_sub_durations_and_timedelta():
    a = Duration(days=5, months=3, years=2)
    b = Duration(days=2, months=1, years=1)
    c = a - b
    assert (c.years, c.months, c.tdelta) == (1, 2, timedelta(days=3))
    d = a - timedelta(days=1)
    assert (d.years, d.months, d.tdelta) == (2, 3, timedelta(days=4))
    e = timedelta(days=10) - Duration(days=3, months=1)
    assert isinstance(e, Duration)
    assert (e.years, e.months, e.tdelta) == (0, -1, timedelta(days=7))


def test_duration_mul():
    d = Duration(days=1, months=2, years=3)
    for r in (d * 3, 3 * d):
        assert r.years == 9
        assert r.months == 6
        assert r.tdelta == timedelta(days=3)
    assert (d * -1) == -d
    with pytest.raises(TypeError):
        d * 1.5
    with pytest.raises(TypeError):
        d * "x"


def test_duration_unsupported_ops():
    with pytest.raises(TypeError):
        Duration(days=1) + "x"
    with pytest.raises(TypeError):
        Duration(days=1) - "x"
    with pytest.raises(TypeError):
        "x" - Duration(days=1)
    with pytest.raises(TypeError):
        Duration(days=1) - date(2000, 1, 1)


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
        (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
        (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
        (date(2000, 11, 15), Duration(months=14), date(2002, 1, 15)),
        (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
        (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
        (date(2000, 1, 1), Duration(months=12), date(2001, 1, 1)),
        (date(2000, 1, 1), Duration(months=11), date(2000, 12, 1)),
        (date(2000, 1, 1), Duration(days=1, months=1, years=1), date(2001, 2, 2)),
        (date(2000, 1, 31), Duration(days=1, months=1), date(2000, 3, 1)),
        (datetime(2000, 1, 31, 12), Duration(hours=13, months=1), datetime(2000, 3, 1, 1)),
        (date(2000, 4, 30), Duration(months=1), date(2000, 5, 30)),
        (date(1999, 8, 31), Duration(months=6), date(2000, 2, 29)),
        (date(2003, 5, 31), Duration(months=1), date(2003, 6, 30)),
    ],
)
def test_duration_add_to_date(start, dur, expected):
    assert start + dur == expected
    assert dur + start == expected


@pytest.mark.parametrize(
    "start, dur, expected",
    [
        (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
        (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
        (date(2001, 1, 15), Duration(months=1), date(2000, 12, 15)),
        (date(2001, 1, 15), Duration(months=13), date(1999, 12, 15)),
        (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2001, 2, 2), Duration(days=1, months=1, years=1), date(2000, 1, 1)),
        (date(2000, 12, 1), Duration(months=12), date(1999, 12, 1)),
        (datetime(2000, 3, 1, 1), Duration(hours=13, months=1), datetime(2000, 1, 31, 12)),
    ],
)
def test_date_minus_duration(start, dur, expected):
    assert start - dur == expected


def test_duration_parse_and_add():
    assert date(2003, 1, 31) + parse_duration("P1M") == date(2003, 2, 28)
    assert datetime(2003, 2, 28, 12) + parse_duration("P1Y1M1DT1H") == datetime(2004, 3, 29, 13)
    assert date(2003, 2, 28) - parse_duration("P1Y1M1D") == date(2002, 1, 27)


def test_duration_fractional_years_with_dates_raise():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("0.5"))


def test_duration_add_to_tz_aware_datetime_preserves_tz():
    tz = FixedOffset(2, 0, "x")
    r = datetime(2000, 1, 31, tzinfo=tz) + Duration(months=1)
    assert r == datetime(2000, 2, 29, tzinfo=tz)
    assert r.tzinfo is tz


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(days=3, hours=1).totimedelta(start=datetime(2000, 1, 1)) == timedelta(
        days=3, hours=1
    )
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_max_days_in_month():
    expected = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for m, days in enumerate(expected, 1):
        assert max_days_in_month(2001, m) == days
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(2004, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2100, 2) == 28
    assert max_days_in_month(1600, 2) == 29


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
        (Decimal(-1), 0, 10, (-1, Decimal(9))),
        (Decimal(7), 0, 5, (1, Decimal(2))),
    ],
)
def test_fquotmod(val, low, high, expected):
    assert fquotmod(val, low, high) == expected


def test_public_api_exports():
    for name in (
        "parse_date", "parse_time", "parse_datetime", "parse_duration", "parse_tzinfo",
        "date_isoformat", "time_isoformat", "datetime_isoformat", "duration_isoformat",
        "tz_isoformat", "strftime", "Duration", "ISO8601Error", "UTC", "LOCAL", "FixedOffset",
        "D_DEFAULT",
    ):
        assert hasattr(isodate, name)
    assert D_DEFAULT == "P%P"
