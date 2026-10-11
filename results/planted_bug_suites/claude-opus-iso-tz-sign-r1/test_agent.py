"""Test suite for isodate, written from the library's documented behaviour."""

import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import isodate
from isodate import (
    D_ALT_BAS,
    D_ALT_EXT,
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


# --------------------------------------------------------------------------
# parse_date
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("2000-366", date(2000, 12, 31)),
        ("2001-001", date(2001, 1, 1)),
        ("2000-02-29", date(2000, 2, 29)),
    ],
)
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # 2009-01-01 is Thursday -> week 1 starts Monday 2008-12-29
        ("2009-W01-1", date(2008, 12, 29)),
        ("2009-W01-7", date(2009, 1, 4)),
        ("2009-W53-7", date(2010, 1, 3)),
        # 2010-01-01 is Friday -> belongs to 2009-W53; 2010-W01 starts 2010-01-04
        ("2010-W01-1", date(2010, 1, 4)),
        # 2008-01-01 is Tuesday -> week 1 starts 2007-12-31
        ("2008-W01-1", date(2007, 12, 31)),
        ("2008-W01", date(2007, 12, 31)),
        # 2007-01-01 is Monday
        ("2007-W01-1", date(2007, 1, 1)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2015-W01-1", date(2014, 12, 29)),
        ("2016-W01-1", date(2016, 1, 4)),
    ],
)
def test_parse_week_dates(text, expected):
    assert parse_date(text) == expected


def test_week_dates_agree_with_isocalendar():
    d = date(1995, 1, 1)
    end = date(2030, 12, 31)
    while d <= end:
        y, w, wd = d.isocalendar()
        assert parse_date("%04d-W%02d-%d" % (y, w, wd)) == d
        assert parse_date("%04dW%02d%d" % (y, w, wd)) == d
        d += timedelta(days=11)


def test_ordinal_dates_cover_whole_year():
    for year in (1999, 2000, 2024, 2100):
        start = date(year, 1, 1)
        n = 366 if max_days_in_month(year, 2) == 29 else 365
        for day in range(1, n + 1):
            assert parse_date("%04d-%03d" % (year, day)) == start + timedelta(days=day - 1)


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
        ("+001985-04", date(1985, 4, 1)),
        ("+001985", date(1985, 1, 1)),
        ("+0019", date(1901, 1, 1)),
    ],
)
def test_parse_date_expanded_six_digits(text, expected):
    assert parse_date(text, yeardigits=6) == expected


def test_parse_date_expanded_four_digits():
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    assert parse_date("+19850412", 4, True) == date(1985, 4, 12)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", expanded=True)


def test_parse_date_expanded_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", yeardigits=6)


def test_parse_date_negative_year_not_representable():
    with pytest.raises(ValueError):
        parse_date("-001985-04-12", yeardigits=6)


def test_parse_date_defaults():
    assert parse_date("1985-04", defaultday=15) == date(1985, 4, 15)
    assert parse_date("1985", defaultmonth=6, defaultday=10) == date(1985, 6, 10)
    assert parse_date("19", defaultmonth=3, defaultday=2) == date(1901, 3, 2)
    # complete dates ignore defaults
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=10) == date(1985, 4, 12)


@pytest.mark.parametrize(
    "text",
    ["", "1985-4-12", "85-04-12", "1985/04/12", "1985-04-12T", "abcd", "1985-W1", "1985-04-1", "x1985"],
)
def test_parse_date_invalid_format(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "1985-04-00", "1900-02-29"])
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
        ("23:20:50", time(23, 20, 50)),
        ("232050", time(23, 20, 50)),
        ("T23:20:50", time(23, 20, 50)),
        ("23:20", time(23, 20)),
        ("2320", time(23, 20)),
        ("23", time(23)),
        ("00:00:00", time(0)),
        ("23:20:50.5", time(23, 20, 50, 500000)),
        ("23:20:50,5", time(23, 20, 50, 500000)),
        ("232050.5", time(23, 20, 50, 500000)),
        ("23:20:50.123456", time(23, 20, 50, 123456)),
        ("23:20:50.1234567", time(23, 20, 50, 123456)),
        ("23:20:50.9999999", time(23, 20, 50, 999999)),
        ("23:20.5", time(23, 20, 30)),
        ("2320,5", time(23, 20, 30)),
        ("23:20.25", time(23, 20, 15)),
        ("23.5", time(23, 30)),
        ("23,5", time(23, 30)),
        ("23.25", time(23, 15)),
        ("00.0125", time(0, 0, 45)),
        ("T23.5", time(23, 30)),
    ],
)
def test_parse_time_naive(text, expected):
    result = parse_time(text)
    assert result == expected
    assert result.tzinfo is None


def test_parse_time_utc():
    for text in ("23:20:50Z", "232050Z", "23:20Z", "23Z", "23.5Z"):
        t = parse_time(text)
        assert t.tzinfo is UTC
        assert t.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "text, offset",
    [
        ("15:27:46+01:00", timedelta(hours=1)),
        ("15:27:46+0100", timedelta(hours=1)),
        ("15:27:46+01", timedelta(hours=1)),
        ("15:27:46-05:00", timedelta(hours=-5)),
        ("15:27:46-0500", timedelta(hours=-5)),
        ("15:27:46-05", timedelta(hours=-5)),
        ("152746+05:30", timedelta(hours=5, minutes=30)),
        ("15:27+05:45", timedelta(hours=5, minutes=45)),
        ("15-03:30", timedelta(hours=-3, minutes=-30)),
        ("15:27:46+00:00", timedelta(0)),
    ],
)
def test_parse_time_offsets(text, offset):
    t = parse_time(text)
    assert t.utcoffset() == offset


def test_parse_time_offset_preserves_wall_clock():
    t = parse_time("15:27:46.25+05:30")
    assert (t.hour, t.minute, t.second, t.microsecond) == (15, 27, 46, 250000)


@pytest.mark.parametrize(
    "text",
    ["", "2", "23:2", "23:20:5", "23:20:50.", "23:20:50+1", "23:20:50Y", "232", "1:00", "23-20-50", "T"],
)
def test_parse_time_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["24:00:01", "23:60:00", "23:59:60", "25"])
def test_parse_time_out_of_range(text):
    with pytest.raises(ValueError):
        parse_time(text)


# --------------------------------------------------------------------------
# time zones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, offset",
    [
        ("+01:00", timedelta(hours=1)),
        ("+0100", timedelta(hours=1)),
        ("+01", timedelta(hours=1)),
        ("-01:00", timedelta(hours=-1)),
        ("+05:30", timedelta(hours=5, minutes=30)),
        ("-03:30", timedelta(hours=-3, minutes=-30)),
        ("+14:00", timedelta(hours=14)),
        ("-12", timedelta(hours=-12)),
    ],
)
def test_parse_tzinfo_offsets(text, offset):
    tz = parse_tzinfo(text)
    assert tz.utcoffset(None) == offset
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == text


def test_parse_tzinfo_utc_and_empty():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None


def test_parse_tzinfo_matches_stdlib():
    for text, std in [
        ("+02:00", timezone(timedelta(hours=2))),
        ("-07:00", timezone(timedelta(hours=-7))),
        ("+09:30", timezone(timedelta(hours=9, minutes=30))),
    ]:
        mine = datetime(2020, 6, 1, 12, tzinfo=parse_tzinfo(text))
        theirs = datetime(2020, 6, 1, 12, tzinfo=std)
        assert mine == theirs
        assert mine.astimezone(timezone.utc) == theirs.astimezone(timezone.utc)


def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    assert build_tzinfo("+02:30", "+", 2, 30).utcoffset(None) == timedelta(hours=2, minutes=30)
    assert build_tzinfo("-02:30", "-", 2, 30).utcoffset(None) == timedelta(hours=-2, minutes=-30)


def test_fixed_offset():
    tz = FixedOffset(5, 30, "IST")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "IST"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedOffset 'IST'>"
    neg = FixedOffset(-8, 0, "PST")
    assert neg.utcoffset(None) == timedelta(hours=-8)
    default = FixedOffset()
    assert default.utcoffset(None) == timedelta(0)
    assert default.tzname(None) == "UTC"


def test_utc_object():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_local_timezone_consistent():
    dt = datetime(2020, 6, 1, 12, 0)
    off = isodate.LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert isinstance(isodate.LOCAL.tzname(dt), str)
    assert isodate.LOCAL.dst(dt) == off - isodate.tzinfo.STDOFFSET


@pytest.mark.parametrize(
    "tz, ext, bas, hour",
    [
        (UTC, "Z", "Z", "Z"),
        (FixedOffset(0, 0, "x"), "Z", "Z", "Z"),
        (FixedOffset(1, 0, "x"), "+01:00", "+0100", "+01"),
        (FixedOffset(-5, 0, "x"), "-05:00", "-0500", "-05"),
        (FixedOffset(5, 30, "x"), "+05:30", "+0530", "+05"),
        (FixedOffset(-3, -30, "x"), "-03:30", "-0330", "-03"),
        (timezone(timedelta(hours=12, minutes=45)), "+12:45", "+1245", "+12"),
        (timezone(timedelta(hours=-9, minutes=-30)), "-09:30", "-0930", "-09"),
    ],
)
def test_tz_isoformat(tz, ext, bas, hour):
    dt = datetime(2012, 12, 25, 13, 30, tzinfo=tz)
    assert tz_isoformat(dt) == ext
    assert tz_isoformat(dt, "%Z") == ext
    assert tz_isoformat(dt, "%z") == bas
    assert tz_isoformat(dt, "%h") == hour


def test_tz_isoformat_naive_and_errors():
    assert tz_isoformat(datetime(2012, 1, 1)) == ""
    with pytest.raises(ValueError):
        tz_isoformat(datetime(2012, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%Q")


@pytest.mark.parametrize("tz", ["+01:00", "-01:00", "+05:30", "-09:30", "+14:00", "-12:00"])
def test_tz_roundtrip(tz):
    dt = datetime(2012, 12, 25, 13, 30, tzinfo=parse_tzinfo(tz))
    assert tz_isoformat(dt) == tz


# --------------------------------------------------------------------------
# parse_datetime
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
        ("19850412T101530", datetime(1985, 4, 12, 10, 15, 30)),
        ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
        ("1985-W15-5T10", datetime(1985, 4, 12, 10)),
        ("1985-04-12T10:15:30.5", datetime(1985, 4, 12, 10, 15, 30, 500000)),
        ("1985-04-12T10.5", datetime(1985, 4, 12, 10, 30)),
    ],
)
def test_parse_datetime_naive(text, expected):
    dt = parse_datetime(text)
    assert dt == expected
    assert dt.tzinfo is None


def test_parse_datetime_utc():
    dt = parse_datetime("1985-04-12T10:15:30Z")
    assert dt.tzinfo is UTC
    assert dt == datetime(1985, 4, 12, 10, 15, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text, utc",
    [
        ("1985-04-12T10:15:30+04:00", datetime(1985, 4, 12, 6, 15, 30, tzinfo=timezone.utc)),
        ("1985-04-12T10:15:30-04:00", datetime(1985, 4, 12, 14, 15, 30, tzinfo=timezone.utc)),
        ("19850412T101530+0400", datetime(1985, 4, 12, 6, 15, 30, tzinfo=timezone.utc)),
        ("1985-04-12T01:00:00+05:30", datetime(1985, 4, 11, 19, 30, tzinfo=timezone.utc)),
        ("1985-04-12T23:00:00-02", datetime(1985, 4, 13, 1, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_datetime_offsets_convert_to_utc(text, utc):
    dt = parse_datetime(text)
    assert dt == utc
    assert dt.astimezone(timezone.utc).replace(tzinfo=None) == utc.replace(tzinfo=None)


def test_parse_datetime_matches_fromisoformat():
    for text in (
        "2021-03-04T05:06:07+03:00",
        "2021-03-04T05:06:07-08:00",
        "2021-03-04T05:06:07.123456+11:30",
    ):
        assert parse_datetime(text) == datetime.fromisoformat(text)
        assert parse_datetime(text).utcoffset() == datetime.fromisoformat(text).utcoffset()


@pytest.mark.parametrize("text", ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10T15", "T10:15"])
def test_parse_datetime_invalid(text):
    with pytest.raises(ISO8601Error):
        parse_datetime(text)


# --------------------------------------------------------------------------
# formatting dates/times/datetimes
# --------------------------------------------------------------------------


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


def test_date_isoformat_default_and_expanded():
    assert date_isoformat(date(1985, 4, 12)) == "1985-04-12"
    assert date_isoformat(date(1985, 4, 12), DATE_EXT_COMPLETE, 6) == "+001985-04-12"
    assert date_isoformat(date(1985, 4, 12), DATE_CENTURY, 6) == "+0019"
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"


@pytest.mark.parametrize(
    "text, fmt, yd",
    [
        ("1985-04-12", DATE_EXT_COMPLETE, 4),
        ("19850412", DATE_BAS_COMPLETE, 4),
        ("2009-W02-1", DATE_EXT_WEEK_COMPLETE, 4),
        ("2009W307", DATE_BAS_WEEK_COMPLETE, 4),
        ("2000-366", DATE_EXT_ORD_COMPLETE, 4),
        ("1985-04", DATE_EXT_MONTH, 4),
        ("1985", DATE_YEAR, 4),
        ("+001985-04-12", DATE_EXT_COMPLETE, 6),
    ],
)
def test_date_roundtrip(text, fmt, yd):
    assert date_isoformat(parse_date(text, yd), fmt, yd) == text


def test_week_format_uses_iso_week():
    assert date_isoformat(date(2010, 1, 3), DATE_EXT_WEEK_COMPLETE) == "2010-W53-7"
    assert date_isoformat(date(2012, 1, 2), DATE_EXT_WEEK_COMPLETE) == "2012-W01-1"


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (TIME_EXT_COMPLETE, "23:20:50"),
        (TIME_BAS_COMPLETE, "232050"),
        (TIME_EXT_MINUTE, "23:20"),
        (TIME_BAS_MINUTE, "2320"),
        (TIME_HOUR, "23"),
        (TIME_EXT_COMPLETE + ".%f", "23:20:50.000123"),
    ],
)
def test_time_isoformat_naive(fmt, expected):
    assert time_isoformat(time(23, 20, 50, 123), fmt) == expected


def test_time_isoformat_default_has_tz():
    assert time_isoformat(time(23, 20, 50)) == "23:20:50"
    assert time_isoformat(time(23, 20, 50, tzinfo=UTC)) == "23:20:50Z"
    assert time_isoformat(time(23, 20, 50, tzinfo=FixedOffset(-5, 0, "x"))) == "23:20:50-05:00"
    t = time(23, 20, 50, tzinfo=FixedOffset(1, 0, "x"))
    assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "232050+0100"
    assert time_isoformat(t, TIME_EXT_COMPLETE + TZ_HOUR) == "23:20:50+01"
    assert time_isoformat(t, TIME_EXT_COMPLETE + TZ_EXT) == "23:20:50+01:00"


@pytest.mark.parametrize(
    "text, fmt",
    [
        ("23:20:50", TIME_EXT_COMPLETE + TZ_EXT),
        ("23:20:50Z", TIME_EXT_COMPLETE + TZ_EXT),
        ("23:20:50+01:00", TIME_EXT_COMPLETE + TZ_EXT),
        ("23:20:50-05:30", TIME_EXT_COMPLETE + TZ_EXT),
        ("232050+0100", TIME_BAS_COMPLETE + TZ_BAS),
        ("232050-0800", TIME_BAS_COMPLETE + TZ_BAS),
        ("23:20+02", TIME_EXT_MINUTE + TZ_HOUR),
    ],
)
def test_time_roundtrip(text, fmt):
    assert time_isoformat(parse_time(text), fmt) == text


def test_datetime_isoformat():
    dt = datetime(1985, 4, 12, 10, 15, 30, 250, tzinfo=FixedOffset(2, 0, "x"))
    assert datetime_isoformat(dt) == "1985-04-12T10:15:30+02:00"
    assert datetime_isoformat(dt, DT_EXT_COMPLETE) == "1985-04-12T10:15:30+02:00"
    assert datetime_isoformat(dt, DT_BAS_COMPLETE) == "19850412T101530+0200"
    assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30)) == "1985-04-12T10:15:30"
    assert datetime_isoformat(datetime(1985, 4, 12, tzinfo=UTC)) == "1985-04-12T00:00:00Z"
    assert datetime_isoformat(dt, "%Y-%m-%dT%H:%M:%S.%f%Z") == "1985-04-12T10:15:30.000250+02:00"


@pytest.mark.parametrize(
    "text, fmt",
    [
        ("1985-04-12T10:15:30", DT_EXT_COMPLETE),
        ("1985-04-12T10:15:30Z", DT_EXT_COMPLETE),
        ("1985-04-12T10:15:30+04:00", DT_EXT_COMPLETE),
        ("1985-04-12T10:15:30-04:00", DT_EXT_COMPLETE),
        ("19850412T101530+0400", DT_BAS_COMPLETE),
        ("19850412T101530-1100", DT_BAS_COMPLETE),
        ("1985-102T10:15:30+05:30", isodate.DT_EXT_ORD_COMPLETE),
        ("1985-W15-5T10:15:30-01:00", isodate.DT_EXT_WEEK_COMPLETE),
    ],
)
def test_datetime_roundtrip(text, fmt):
    assert datetime_isoformat(parse_datetime(text), fmt) == text


def test_strftime_literal_and_unknown():
    d = date(2020, 2, 3)
    assert strftime(d, "100%%") == "100%"
    assert strftime(d, "%Y %q") == "2020 %q"
    assert strftime(d, "%j") == "034"
    assert strftime(date(2020, 12, 31), "%j") == "366"
    assert strftime(d, "%w") == "1"
    assert strftime(date(2020, 2, 9), "%w") == "7"


# --------------------------------------------------------------------------
# parse_duration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P1D", timedelta(days=1)),
        ("PT1H", timedelta(hours=1)),
        ("PT1M", timedelta(minutes=1)),
        ("PT1S", timedelta(seconds=1)),
        ("P1W", timedelta(weeks=1)),
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6S", timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("PT36H", timedelta(hours=36)),
        ("P1DT12H", timedelta(days=1, hours=12)),
        ("PT0.5S", timedelta(milliseconds=500)),
        ("PT0,5S", timedelta(milliseconds=500)),
        ("P0.5D", timedelta(hours=12)),
        ("PT1.5H", timedelta(minutes=90)),
        ("PT1.5M", timedelta(seconds=90)),
        ("P0D", timedelta(0)),
        ("PT0S", timedelta(0)),
        ("+P1D", timedelta(days=1)),
        ("-P1D", timedelta(days=-1)),
        ("-PT1H30M", -timedelta(hours=1, minutes=30)),
        ("-P1W", timedelta(weeks=-1)),
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
        ("P1M", 0, 1, timedelta(0)),
        ("P18M", 0, 18, timedelta(0)),
        ("P1Y2M3DT4H5M6S", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("P2Y1DT1S", 2, 0, timedelta(days=1, seconds=1)),
        ("P1.5Y", Decimal("1.5"), 0, timedelta(0)),
        ("P1,5M", 0, Decimal("1.5"), timedelta(0)),
        ("P1YT1M", 1, 0, timedelta(minutes=1)),
    ],
)
def test_parse_duration_duration(text, years, months, tdelta):
    result = parse_duration(text)
    assert isinstance(result, Duration)
    assert result.years == years
    assert result.months == months
    assert result.tdelta == tdelta


def test_parse_duration_negative_duration():
    result = parse_duration("-P1Y2M3D")
    assert isinstance(result, Duration)
    assert result.years == -1
    assert result.months == -2
    assert result.tdelta == timedelta(days=-3)


def test_parse_duration_force_duration():
    result = parse_duration("P1DT2H", as_timedelta_if_possible=False)
    assert isinstance(result, Duration)
    assert result.years == 0 and result.months == 0
    assert result.tdelta == timedelta(days=1, hours=2)
    assert result == timedelta(days=1, hours=2)


def test_parse_duration_alternative_formats():
    r = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(r, Duration)
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    r = parse_duration("P00010203T040506")
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    r = parse_duration("P0001-034T01:00:00")
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3, hours=1)


@pytest.mark.parametrize("text", ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P-1D", "xP1D", "P1D1Y"])
def test_parse_duration_invalid(text):
    with pytest.raises((ISO8601Error, ValueError)):
        parse_duration(text)


def test_parse_duration_type_error():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


# --------------------------------------------------------------------------
# duration_isoformat
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(days=1), "P1D"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(0), "P0D"),
        (timedelta(days=3, hours=4, minutes=5, seconds=6), "P3DT4H5M6S"),
        (timedelta(hours=36), "P1DT12H"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(days=-1), "-P1D"),
        (-timedelta(hours=1, minutes=30), "-PT1H30M"),
        (-timedelta(microseconds=250000), "-PT0.25S"),
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
        (Duration(years=-1, months=-2), "-P1Y2M"),
        (Duration(years=Decimal("1.5")), "P1.5Y"),
        (Duration(0), "P0D"),
    ],
)
def test_duration_isoformat(value, expected):
    assert duration_isoformat(value) == expected


def test_duration_isoformat_weeks_and_custom():
    assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=-14), D_WEEK) == "-P2W"
    td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    assert strftime(td, "%d %H:%M:%S.%f") == "03 04:05:06.000007"
    assert strftime(Duration(years=12, months=3), "%Y-%m") == "0012-03"
    assert strftime(timedelta(days=21), "%W") == "03"


def test_duration_isoformat_alt():
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d, D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(d, D_ALT_BAS) == "P00010203T040506"


@pytest.mark.parametrize(
    "text",
    ["P1D", "PT1H", "P3DT4H5M6S", "-P1D", "-PT1H30M", "P1Y", "P1Y2M3DT4H5M6S", "-P1Y2M", "PT1.5S", "P2M1D"],
)
def test_duration_roundtrip(text):
    assert duration_isoformat(parse_duration(text)) == text


# --------------------------------------------------------------------------
# Duration class
# --------------------------------------------------------------------------


def test_duration_init_and_attrs():
    d = Duration(days=1, hours=2, years=3, months=4, weeks=1)
    assert d.years == Decimal(3)
    assert d.months == Decimal(4)
    assert isinstance(d.years, Decimal)
    assert d.tdelta == timedelta(days=8, hours=2)
    # attribute delegation to timedelta
    assert d.days == 8
    assert d.seconds == 7200
    assert d.microseconds == 0
    assert Duration(years=0.5).years == Decimal("0.5")


def test_duration_str_repr():
    assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(days=2)) == "2 days, 0:00:00"
    r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
    assert r == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, months=1) == Duration(months=13)
    assert Duration(years=1) != Duration(years=2)
    assert Duration(years=1) != Duration(years=1, days=1)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert not (Duration(years=1, days=1) == timedelta(days=1))
    assert Duration(days=1) != "P1D"


def test_duration_hash():
    a = Duration(years=1, days=2)
    b = Duration(years=1, days=2)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert hash(Duration(years=1)) != hash(Duration(years=2)) or Duration(years=1) != Duration(years=2)


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3)
    assert d.years == -1 and d.months == -2
    assert d.tdelta == timedelta(days=-3)
    assert -(-d) == d


def test_duration_add_sub_durations():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=2, months=11, hours=5)
    s = a + b
    assert (s.years, s.months) == (3, 13)
    assert s.tdelta == timedelta(days=3, hours=5)
    diff = a - b
    assert (diff.years, diff.months) == (-1, -9)
    assert diff.tdelta == timedelta(days=3, hours=-5)


def test_duration_with_timedelta():
    d = Duration(years=1, days=1)
    r = d + timedelta(hours=1)
    assert isinstance(r, Duration)
    assert r.years == 1 and r.tdelta == timedelta(days=1, hours=1)
    r = timedelta(hours=1) + d
    assert isinstance(r, Duration)
    assert r.tdelta == timedelta(days=1, hours=1)
    r = d - timedelta(hours=1)
    assert r.years == 1 and r.tdelta == timedelta(hours=23)
    r = timedelta(days=2) - d
    assert isinstance(r, Duration)
    assert r.years == -1 and r.tdelta == timedelta(days=1)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3)
    for r in (d * 3, 3 * d):
        assert (r.years, r.months) == (3, 6)
        assert r.tdelta == timedelta(days=9)
    assert d * 0 == Duration(0)
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
        (date(2000, 1, 1), Duration(years=1, months=1, days=1), date(2001, 2, 2)),
        (date(2000, 1, 1), Duration(days=-1), date(1999, 12, 31)),
        (
            datetime(2000, 1, 31, 23, 0),
            Duration(months=1, hours=2),
            datetime(2000, 3, 1, 1, 0),
        ),
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
        (date(2000, 1, 15), Duration(months=1), date(1999, 12, 15)),
        (date(2000, 1, 15), Duration(months=13), date(1998, 12, 15)),
        (date(2000, 12, 15), Duration(months=12), date(1999, 12, 15)),
        (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
        (date(2000, 1, 1), Duration(years=1, months=1, days=1), date(1998, 11, 30)),
        (date(2000, 1, 15), Duration(months=-1), date(2000, 2, 15)),
        (datetime(2000, 3, 1, 1, 0), Duration(months=1, hours=2), datetime(2000, 1, 31, 23, 0)),
    ],
)
def test_date_minus_duration(start, dur, expected):
    assert start - dur == expected


def test_duration_date_preserves_tz():
    dt = datetime(2000, 1, 31, 12, tzinfo=UTC)
    r = dt + Duration(months=1)
    assert r == datetime(2000, 2, 29, 12, tzinfo=UTC)
    assert r.tzinfo is UTC


def test_duration_fractional_months_with_date():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=1.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=0.5)


def test_duration_with_parsed_values():
    assert parse_date("2000-01-31") + parse_duration("P1M") == date(2000, 2, 29)
    assert parse_datetime("2000-01-01T00:00:00") + parse_duration("P1Y2M3DT4H") == datetime(
        2001, 3, 4, 4
    )
    assert parse_datetime("2000-01-01T00:00:00") - parse_duration("-P1D") == datetime(2000, 1, 2)


def test_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2000, 1, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(days=2, hours=3).totimedelta(start=datetime(2000, 1, 1)) == timedelta(
        days=2, hours=3
    )
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, seconds=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d
    assert d2.tdelta == d.tdelta and d2.years == d.years and d2.months == d.months


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "year, month, days",
    [
        (2001, 1, 31), (2001, 2, 28), (2001, 3, 31), (2001, 4, 30), (2001, 5, 31),
        (2001, 6, 30), (2001, 7, 31), (2001, 8, 31), (2001, 9, 30), (2001, 10, 31),
        (2001, 11, 30), (2001, 12, 31), (2000, 2, 29), (2004, 2, 29), (1900, 2, 28),
        (2100, 2, 28), (2400, 2, 29),
    ],
)
def test_max_days_in_month(year, month, days):
    assert max_days_in_month(year, month) == days


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


def test_public_api_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert isinstance(isodate.__version__, str)
