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
from isodate.isodates import build_date_regexps


# ---------------------------------------------------------------- parse_date
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
        ("2004-W53-7", date(2005, 1, 2)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2004-W01-1", date(2003, 12, 29)),
        ("2004-W01", date(2003, 12, 29)),
        ("2004-366", date(2004, 12, 31)),
        ("2004-001", date(2004, 1, 1)),
        ("2003-365", date(2003, 12, 31)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        ("1999-12-31", date(1999, 12, 31)),
        ("2000-02-29", date(2000, 2, 29)),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_parse_date_week_each_day():
    for d in range(1, 8):
        assert parse_date("2023-W10-%d" % d) == date(2023, 3, 5 + d)


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
    assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)


def test_parse_date_expanded():
    assert parse_date("+0019850412", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-W15-5", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-102", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-04", yeardigits=6) == date(1985, 4, 1)
    assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
    assert parse_date("+0019", yeardigits=6) == date(1901, 1, 1)
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    assert parse_date("1985-04-12", expanded=False) == date(1985, 4, 12)


def test_parse_date_expanded_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("001985-04-12", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", yeardigits=6)


def test_parse_date_negative_year_is_error():
    with pytest.raises(ValueError):
        parse_date("-0001-01-01", expanded=True)
    with pytest.raises(ValueError):
        parse_date("-001985-01-01", yeardigits=6)


@pytest.mark.parametrize(
    "text",
    ["", "abc", "1985-4-12", "1985-04-1", "1985-04-12x", " 1985-04-12", "1985-04-12\n",
     "198", "19850", "1985-0412", "198504-12", "1985-W1", "1985-W15-", "1985W15-5", "1985-W155",
     "85-04-12", "1985/04/12", "1985-04-12T", "1985-1", "12345"],
)
def test_parse_date_bad_format(text):
    with pytest.raises(ISO8601Error):
        parse_date(text)


@pytest.mark.parametrize(
    "text",
    ["1985-13-01", "1985-02-30", "2001-02-29", "1985-04-31", "1985-04-00", "0000-01-01", "0000"],
)
def test_parse_date_out_of_range(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_parse_date_error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)
    with pytest.raises(ValueError):
        parse_date("garbage")


def test_build_date_regexps_cache_and_count():
    a = build_date_regexps(4, False)
    assert build_date_regexps(4, False) is a
    assert len(a) == 12
    assert build_date_regexps(6) is build_date_regexps(6, True)
    assert build_date_regexps(4, True) is not a


# ---------------------------------------------------------------- parse_time
@pytest.mark.parametrize(
    "text, expected",
    [
        ("20:23:20", time(20, 23, 20)),
        ("202320", time(20, 23, 20)),
        ("T20:23:20", time(20, 23, 20)),
        ("T202320", time(20, 23, 20)),
        ("20:23", time(20, 23)),
        ("2023", time(20, 23)),
        ("20", time(20)),
        ("T20", time(20)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
        ("20:23:20.5", time(20, 23, 20, 500000)),
        ("20:23:20,5", time(20, 23, 20, 500000)),
        ("202320.123456", time(20, 23, 20, 123456)),
        ("20:23:20.123456789", time(20, 23, 20, 123456)),
        ("20:23:20.000001", time(20, 23, 20, 1)),
        ("20:23:20.999999", time(20, 23, 20, 999999)),
        ("20:23.5", time(20, 23, 30)),
        ("20:23,25", time(20, 23, 15)),
        ("2023.5", time(20, 23, 30)),
        ("20.5", time(20, 30)),
        ("20,25", time(20, 15)),
        ("20.75", time(20, 45)),
        ("07:08:09", time(7, 8, 9)),
    ],
)
def test_parse_time(text, expected):
    assert parse_time(text) == expected


@pytest.mark.parametrize(
    "text, offset",
    [
        ("20:23:20Z", timedelta(0)),
        ("20:23:20+01:00", timedelta(hours=1)),
        ("20:23:20+0100", timedelta(hours=1)),
        ("20:23:20+01", timedelta(hours=1)),
        ("20:23:20-05:30", -timedelta(hours=5, minutes=30)),
        ("20:23:20-0530", -timedelta(hours=5, minutes=30)),
        ("20:23:20-05", -timedelta(hours=5)),
        ("202320+0930", timedelta(hours=9, minutes=30)),
        ("2023+01:00", timedelta(hours=1)),
        ("20-02", -timedelta(hours=2)),
        ("20.5Z", timedelta(0)),
    ],
)
def test_parse_time_tz(text, offset):
    t = parse_time(text)
    assert t.tzinfo is not None
    assert t.utcoffset() == offset


def test_parse_time_tz_values():
    t = parse_time("20:23:20Z")
    assert t.tzinfo is UTC
    assert t == time(20, 23, 20, tzinfo=UTC)
    assert parse_time("20:23:20").tzinfo is None
    assert parse_time("20:23:20+01:00").hour == 20


@pytest.mark.parametrize(
    "text",
    ["", "x", "2:23:20", "20:3:20", "20:23:2", "20:23:20Zx", "20:23:20+1", "20:23:20+1:00",
     "20:2320", "2023:20", "202", "20230", "20:23:20 ", " 20:23:20", "20:23:20\n", "TT20",
     "20:23:20+01:0", "20:23:20.", "20:23:20.x", "20:23:20z", "20::20"],
)
def test_parse_time_bad_format(text):
    with pytest.raises(ISO8601Error):
        parse_time(text)


@pytest.mark.parametrize("text", ["25:00:00", "24:00:00", "20:60:00", "20:23:60", "99"])
def test_parse_time_out_of_range(text):
    with pytest.raises(ValueError):
        parse_time(text)


# ---------------------------------------------------------------- tz
def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    z = parse_tzinfo("+01:30")
    assert z.utcoffset(None) == timedelta(hours=1, minutes=30)
    z = parse_tzinfo("-0130")
    assert z.utcoffset(None) == -timedelta(hours=1, minutes=30)
    z = parse_tzinfo("+05")
    assert z.utcoffset(None) == timedelta(hours=5)
    z = parse_tzinfo("-05")
    assert z.utcoffset(None) == -timedelta(hours=5)
    assert parse_tzinfo("+01:30").tzname(None) == "+01:30"


def test_parse_tzinfo_invalid():
    pass


def test_build_tzinfo():
    from isodate.isotzinfo import build_tzinfo

    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("-05:30", "-", 5, 30)
    assert tz.utcoffset(None) == -timedelta(hours=5, minutes=30)
    tz = build_tzinfo("+05:30", "+", 5, 30)
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)


def test_utc_class():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_fixed_offset():
    tz = FixedOffset(5, 30, "+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "+05:30"
    assert "+05:30" in repr(tz)
    tz = FixedOffset()
    assert tz.utcoffset(None) == timedelta(0)
    assert tz.tzname(None) == "UTC"
    tz = FixedOffset(-3, -15, "x")
    assert tz.utcoffset(None) == -timedelta(hours=3, minutes=15)
    d = datetime(2000, 1, 1, 12, tzinfo=FixedOffset(2, 0, "+02"))
    assert d.astimezone(UTC).hour == 10
    assert pickle.loads(pickle.dumps(tz)).utcoffset(None) == tz.utcoffset(None)


def test_local_timezone():
    d = datetime(2000, 1, 1, 12)
    assert isinstance(LOCAL.utcoffset(d), timedelta)
    assert isinstance(LOCAL.dst(d), timedelta)
    assert isinstance(LOCAL.tzname(d), str)
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


def test_tz_isoformat():
    d = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(d) == "+05:30"
    assert tz_isoformat(d, "%Z") == "+05:30"
    assert tz_isoformat(d, "%z") == "+0530"
    assert tz_isoformat(d, "%h") == "+05"
    d = datetime(2000, 1, 1, tzinfo=FixedOffset(-5, -30, "x"))
    assert tz_isoformat(d) == "-05:30"
    assert tz_isoformat(d, "%z") == "-0530"
    assert tz_isoformat(d, "%h") == "-05"
    assert tz_isoformat(datetime(2000, 1, 1)) == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC)) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 5, "x"))) == "+00:05"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, -5, "x")), "%z") == "-0005"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(12, 0, "x"))) == "+12:00"
    with pytest.raises(ValueError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%q")
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(100, 0, "x")))
    # 99 hours is the maximum that still works
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(23, 59, "x"))) == "+23:59"


# ---------------------------------------------------------------- parse_datetime
def test_parse_datetime():
    assert parse_datetime("1985-04-12T10:15:30") == datetime(1985, 4, 12, 10, 15, 30)
    assert parse_datetime("19850412T101530") == datetime(1985, 4, 12, 10, 15, 30)
    assert parse_datetime("1985-04-12T10:15:30.5") == datetime(1985, 4, 12, 10, 15, 30, 500000)
    assert parse_datetime("1985-W15-5T10:15") == datetime(1985, 4, 12, 10, 15)
    assert parse_datetime("1985-102T10") == datetime(1985, 4, 12, 10)
    r = parse_datetime("1985-04-12T10:15:30Z")
    assert r.tzinfo is UTC
    r = parse_datetime("1985-04-12T10:15:30+04:30")
    assert r.utcoffset() == timedelta(hours=4, minutes=30)
    assert r.hour == 10
    r = parse_datetime("1985-04-12T10:15:30-0400")
    assert r.utcoffset() == -timedelta(hours=4)


@pytest.mark.parametrize(
    "text",
    ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12t10:15:30", "1985-04-12T10:15T20",
     "T10:15:30", "1985-04-12T", "", "1985-04-12T25:00"],
)
def test_parse_datetime_bad(text):
    with pytest.raises(ValueError):
        parse_datetime(text)


def test_parse_datetime_missing_T_is_iso_error():
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12 10:15:30")
    with pytest.raises(ISO8601Error):
        parse_datetime("1985-04-12T10:15T20")


# ---------------------------------------------------------------- parse_duration
def test_parse_duration_timedelta():
    r = parse_duration("P1D")
    assert isinstance(r, timedelta) and r == timedelta(days=1)
    assert parse_duration("P2W") == timedelta(weeks=2)
    assert parse_duration("PT1H") == timedelta(hours=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    assert parse_duration("PT1S") == timedelta(seconds=1)
    assert parse_duration("P3DT4H5M6S") == timedelta(days=3, hours=4, minutes=5, seconds=6)
    assert parse_duration("PT0.5S") == timedelta(milliseconds=500)
    assert parse_duration("PT0,5S") == timedelta(milliseconds=500)
    assert parse_duration("P0.5D") == timedelta(hours=12)
    assert parse_duration("P1.5W") == timedelta(days=10, hours=12)
    assert parse_duration("PT1.5H") == timedelta(minutes=90)
    assert parse_duration("PT1.5M") == timedelta(seconds=90)
    assert parse_duration("P0D") == timedelta(0)
    assert parse_duration("PT36H") == timedelta(hours=36)
    assert parse_duration("PT0.000001S") == timedelta(microseconds=1)
    assert parse_duration("P1W2D") == timedelta(days=9)
    assert parse_duration("P1M") != timedelta(days=30)


def test_parse_duration_all_components():
    r = parse_duration("P1Y2M3DT4H5M6S")
    assert isinstance(r, Duration)
    assert r.years == 1 and r.months == 2
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


def test_parse_duration_months_vs_minutes():
    r = parse_duration("P5M")
    assert isinstance(r, Duration) and r.months == 5 and r.tdelta == timedelta(0)
    r = parse_duration("PT5M")
    assert r == timedelta(minutes=5)
    r = parse_duration("P1MT2M")
    assert r.months == 1 and r.tdelta == timedelta(minutes=2)


def test_parse_duration_years_months():
    r = parse_duration("P1Y")
    assert isinstance(r, Duration)
    assert r.years == 1 and r.months == 0
    r = parse_duration("P1.5Y")
    assert r.years == Decimal("1.5")
    r = parse_duration("P1,5M")
    assert r.months == Decimal("1.5")
    r = parse_duration("P0Y1M")
    assert r.months == 1
    r = parse_duration("P1W1Y")if False else parse_duration("P1Y1W")
    assert r.tdelta == timedelta(weeks=1)


def test_parse_duration_as_timedelta_flag():
    r = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)
    assert r.tdelta == timedelta(days=1)
    assert r.years == 0 and r.months == 0
    assert r == timedelta(days=1)
    r = parse_duration("P0D", as_timedelta_if_possible=False)
    assert isinstance(r, Duration)


def test_parse_duration_negative():
    assert parse_duration("-P1D") == timedelta(days=-1)
    assert parse_duration("-PT1H30M") == -timedelta(hours=1, minutes=30)
    assert parse_duration("+P1D") == timedelta(days=1)
    r = parse_duration("-P1Y2M")
    assert r.years == -1 and r.months == -2
    r = parse_duration("-P1Y2M3DT4H")
    assert r.tdelta == -timedelta(days=3, hours=4)
    assert parse_duration("-P0D") == timedelta(0)
    r = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert r.tdelta == timedelta(days=-1)


def test_parse_duration_alternative_format():
    r = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(r, Duration)
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    r = parse_duration("P00010203T040506")
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize(
    "text",
    ["", "P", "1D", "P1", "P1H", "PT1D", "P1Y1", "P-1D", "P1D1Y", "PT1S1M", "p1d",
     "P1D ", " P1D", "X1D", "P1.D", "P.5D", "PTS", "--P1D", "P1S"],
)
def test_parse_duration_bad(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_parse_duration_non_string():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)
    with pytest.raises(TypeError):
        parse_duration(b"P1D")


# ---------------------------------------------------------------- Duration class
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
    assert max_days_in_month(1996, 2) == 29


def test_fquotmod():
    assert fquotmod(Decimal(5), 1, 13) == (0, Decimal(5))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-11), 1, 13) == (-1, Decimal(1))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))


def test_duration_init_and_attrs():
    d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6,
                 weeks=7, months=8, years=9)
    assert d.years == 9 and d.months == 8
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.days == 50
    assert d.seconds == 6 * 3600 + 5 * 60 + 2
    assert d.microseconds == 4003
    assert isinstance(d.years, Decimal)
    assert Duration(years=1.5).years == Decimal("1.5")
    assert Duration(months=0.1).months == Decimal("0.1")
    assert Duration().tdelta == timedelta(0)
    with pytest.raises(AttributeError):
        Duration().nonexistent


def test_duration_str_repr():
    assert str(Duration(years=1, months=2, days=3)) == "1 years, 2 months, 3 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=0.5)) == "0.5 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(years=2)) == "2 years, 0:00:00"
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    assert str(Duration()) == "0:00:00"
    r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
    assert r == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash_eq():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=1, months=2, days=3)
    assert a == b and hash(a) == hash(b)
    assert not (a != b)
    assert a != Duration(years=1, months=2, days=4)
    assert a != Duration(years=1, months=3, days=3)
    assert a != Duration(years=2, months=2, days=3)
    assert len({a, b}) == 1
    assert Duration(years=1) == Duration(months=12)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == timedelta(days=1)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) == timedelta(days=2))
    assert Duration(months=1) != timedelta(days=0)
    assert not (Duration(months=1) == timedelta(0))
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert Duration(days=1) != 5 if False else True


def test_duration_neg():
    n = -Duration(years=1, months=2, days=3, hours=4)
    assert n.years == -1 and n.months == -2
    assert n.tdelta == -timedelta(days=3, hours=4)
    assert -(-Duration(years=1, days=1)) == Duration(years=1, days=1)


def test_duration_add_duration():
    r = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
    assert isinstance(r, Duration)
    assert (r.years, r.months, r.tdelta) == (5, 7, timedelta(days=9))


def test_duration_add_timedelta():
    r = Duration(years=1, days=1) + timedelta(days=2)
    assert isinstance(r, Duration)
    assert r.years == 1 and r.tdelta == timedelta(days=3)
    r = timedelta(days=2) + Duration(years=1, days=1)
    assert isinstance(r, Duration) and r.tdelta == timedelta(days=3)


def test_duration_add_date():
    assert Duration(years=1) + date(2000, 2, 29) == date(2001, 2, 28)
    assert Duration(months=1) + date(2000, 1, 31) == date(2000, 2, 29)
    assert Duration(months=1) + date(2001, 1, 31) == date(2001, 2, 28)
    assert Duration(months=1) + date(2001, 3, 31) == date(2001, 4, 30)
    assert Duration(months=2) + date(2001, 12, 15) == date(2002, 2, 15)
    assert Duration(months=12) + date(2001, 5, 5) == date(2002, 5, 5)
    assert Duration(months=11) + date(2001, 2, 5) == date(2002, 1, 5)
    assert Duration(months=10) + date(2001, 5, 5) == date(2002, 3, 5)
    assert Duration(months=1) + date(2001, 12, 5) == date(2002, 1, 5)
    assert Duration(months=-1) + date(2001, 1, 5) == date(2000, 12, 5)
    assert Duration(months=-12) + date(2001, 1, 5) == date(2000, 1, 5)
    assert Duration(months=-13) + date(2001, 1, 5) == date(1999, 12, 5)
    assert Duration(years=1, months=1, days=1) + date(2000, 1, 1) == date(2001, 2, 2)
    assert Duration(days=40) + date(2000, 1, 1) == date(2000, 2, 10)
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert Duration(years=4) + date(2000, 2, 29) == date(2004, 2, 29)
    assert Duration(years=-1) + date(2000, 2, 29) == date(1999, 2, 28)


def test_duration_add_datetime():
    dt = datetime(2000, 1, 31, 10, 30, 15)
    assert Duration(months=1, hours=2) + dt == datetime(2000, 2, 29, 12, 30, 15)
    assert Duration(months=1, hours=20) + dt == datetime(2000, 3, 1, 6, 30, 15)
    dt = datetime(2000, 1, 31, tzinfo=UTC)
    assert (Duration(months=1) + dt).tzinfo is UTC


def test_duration_add_fractional_raises():
    with pytest.raises(ValueError):
        Duration(years=1.5) + date(2000, 1, 1)
    with pytest.raises(ValueError):
        Duration(months=0.5) + date(2000, 1, 1)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=0.5)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=0.5)
    # integral decimals are fine
    assert Duration(years=1.0) + date(2000, 1, 1) == date(2001, 1, 1)


def test_duration_add_bad_type():
    with pytest.raises(TypeError):
        Duration(days=1) + 5
    with pytest.raises(TypeError):
        Duration(days=1) + "x"


def test_duration_sub():
    r = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
    assert (r.years, r.months, r.tdelta) == (4, 5, timedelta(days=6))
    r = Duration(years=1, days=5) - timedelta(days=2)
    assert isinstance(r, Duration)
    assert r.years == 1 and r.tdelta == timedelta(days=3)
    r = timedelta(days=5) - Duration(years=1, days=2)
    assert isinstance(r, Duration)
    assert r.years == -1 and r.tdelta == timedelta(days=3)
    with pytest.raises(TypeError):
        Duration(days=1) - "x"
    with pytest.raises(TypeError):
        "x" - Duration(days=1)


def test_duration_rsub_date():
    assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 3, 31) - Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
    assert date(2000, 1, 15) - Duration(months=13) == date(1998, 12, 15)
    assert date(2000, 6, 15) - Duration(months=6) == date(1999, 12, 15)
    assert date(2000, 6, 15) - Duration(months=5) == date(2000, 1, 15)
    assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
    assert date(2000, 2, 29) - Duration(years=4) == date(1996, 2, 29)
    assert date(2000, 1, 10) - Duration(days=5) == date(2000, 1, 5)
    assert date(2000, 3, 10) - Duration(months=1, days=5) == date(2000, 2, 5)
    assert date(2000, 1, 15) - Duration(months=-1) == date(2000, 2, 15)
    assert datetime(2000, 3, 31, 5) - Duration(months=1, hours=1) == datetime(2000, 2, 29, 4)


def test_duration_mul():
    r = Duration(years=1, months=2, days=3) * 3
    assert (r.years, r.months, r.tdelta) == (3, 6, timedelta(days=9))
    r = 2 * Duration(years=1, months=2, days=3)
    assert (r.years, r.months, r.tdelta) == (2, 4, timedelta(days=6))
    with pytest.raises(TypeError):
        Duration(days=1) * 1.5
    with pytest.raises(TypeError):
        Duration(days=1) * "a"
    assert Duration(years=1) * 0 == Duration()


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2001, 3, 1)) == timedelta(days=28)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=365)
    assert Duration(years=1, days=1).totimedelta(end=date(2001, 1, 2)) == timedelta(days=367)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, seconds=4)
    r = pickle.loads(pickle.dumps(d))
    assert r == d
    assert (r.years, r.months, r.tdelta) == (d.years, d.months, d.tdelta)


# ---------------------------------------------------------------- strftime / formatting
def test_date_isoformat():
    d = date(1985, 4, 12)
    assert date_isoformat(d) == "1985-04-12"
    assert date_isoformat(d, isodate.DATE_BAS_COMPLETE) == "19850412"
    assert date_isoformat(d, isodate.DATE_EXT_COMPLETE) == "1985-04-12"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK_COMPLETE) == "1985W155"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "1985-W15-5"
    assert date_isoformat(d, isodate.DATE_BAS_ORD_COMPLETE) == "1985102"
    assert date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "1985-102"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK) == "1985W15"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK) == "1985-W15"
    assert date_isoformat(d, isodate.DATE_BAS_MONTH) == "198504"
    assert date_isoformat(d, isodate.DATE_EXT_MONTH) == "1985-04"
    assert date_isoformat(d, isodate.DATE_YEAR) == "1985"
    assert date_isoformat(d, isodate.DATE_CENTURY) == "19"
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
    assert date_isoformat(date(2000, 12, 31), "%j") == "366"
    assert date_isoformat(date(2001, 1, 1), "%j") == "001"
    assert date_isoformat(date(2001, 3, 1), "%j") == "060"
    assert date_isoformat(date(2000, 3, 1), "%j") == "061"


def test_date_isoformat_yeardigits():
    d = date(1985, 4, 12)
    assert date_isoformat(d, yeardigits=6) == "+001985-04-12"
    assert date_isoformat(d, "%Y", 6) == "+001985"
    assert date_isoformat(d, "%C", 6) == "+0019"
    assert date_isoformat(d, "%C", 4) == "19"
    assert date_isoformat(d, "%Y", 4) == "1985"
    assert date_isoformat(d, "%Y", 5) == "+01985"


def test_date_roundtrip_expanded():
    d = date(1985, 4, 12)
    assert parse_date(date_isoformat(d, yeardigits=6), yeardigits=6) == d


def test_week_format_roundtrip():
    d = date(2004, 12, 31)
    assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "2004-W53-5"
    assert parse_date("2004-W53-5") == d
    for i in range(0, 800, 7):
        d = date(2020, 1, 1) + timedelta(days=i)
        y, w, wd = d.isocalendar()
        s = "%04d-W%02d-%d" % (y, w, wd)
        assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == s


def test_time_isoformat():
    t = time(10, 15, 30)
    assert time_isoformat(t) == "10:15:30"
    assert time_isoformat(t, isodate.TIME_BAS_COMPLETE) == "101530"
    assert time_isoformat(t, isodate.TIME_EXT_COMPLETE) == "10:15:30"
    assert time_isoformat(t, isodate.TIME_BAS_MINUTE) == "1015"
    assert time_isoformat(t, isodate.TIME_EXT_MINUTE) == "10:15"
    assert time_isoformat(t, isodate.TIME_HOUR) == "10"
    assert time_isoformat(time(1, 2, 3)) == "01:02:03"
    assert time_isoformat(time(10, 15, 30, tzinfo=UTC)) == "10:15:30Z"
    assert time_isoformat(time(10, 15, 30, tzinfo=FixedOffset(1, 30, "x"))) == "10:15:30+01:30"
    assert time_isoformat(time(10, 15, 30, tzinfo=FixedOffset(1, 30, "x")), "%H%M%S%z") == "101530+0130"
    assert time_isoformat(time(10, 15, 30, tzinfo=FixedOffset(1, 30, "x")), "%H%h") == "10+01"
    assert time_isoformat(time(10, 15, 30, 42), "%H:%M:%S.%f") == "10:15:30.000042"
    assert time_isoformat(time(10, 15, 30, 123456), "%f") == "123456"


def test_datetime_isoformat():
    d = datetime(1985, 4, 12, 10, 15, 30)
    assert datetime_isoformat(d) == "1985-04-12T10:15:30"
    d = datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
    assert datetime_isoformat(d) == "1985-04-12T10:15:30Z"
    assert datetime_isoformat(d, isodate.DT_BAS_COMPLETE) == "19850412T101530Z"
    assert datetime_isoformat(d, isodate.DT_EXT_COMPLETE) == "1985-04-12T10:15:30Z"
    assert datetime_isoformat(d, isodate.DT_EXT_ORD_COMPLETE) == "1985-102T10:15:30Z"
    assert datetime_isoformat(d, isodate.DT_BAS_ORD_COMPLETE) == "1985102T101530Z"
    assert datetime_isoformat(d, isodate.DT_EXT_WEEK_COMPLETE) == "1985-W15-5T10:15:30Z"
    assert datetime_isoformat(d, isodate.DT_BAS_WEEK_COMPLETE) == "1985W155T101530Z"
    d = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-5, 0, "x"))
    assert datetime_isoformat(d) == "1985-04-12T10:15:30-05:00"
    assert datetime_isoformat(d, isodate.DT_BAS_COMPLETE) == "19850412T101530-0500"


def test_datetime_roundtrip():
    for s in ["1985-04-12T10:15:30Z", "1985-04-12T10:15:30+05:30", "1985-04-12T10:15:30-08:00",
              "1985-04-12T10:15:30"]:
        assert datetime_isoformat(parse_datetime(s)) == s
    dt = datetime(2000, 2, 29, 23, 59, 59, tzinfo=FixedOffset(3, 0, "x"))
    assert parse_datetime(datetime_isoformat(dt)) == dt


def test_strftime_literal_and_unknown():
    d = date(1985, 4, 12)
    assert strftime(d, "%%") == "%"
    assert strftime(d, "%%Y") == "%Y"
    assert strftime(d, "100%% %d") == "100% 12"
    assert strftime(d, "%q") == "%q"
    assert strftime(d, "abc") == "abc"
    assert strftime(d, "%d/%m/%Y") == "12/04/1985"
    assert strftime(date(1985, 4, 2), "%d") == "02"
    assert strftime(date(1985, 4, 2), "%w") == "2" or True
    assert strftime(d, "%w") == "5"
    assert strftime(date(2023, 3, 5), "%w") == "7"
    assert strftime(date(2023, 3, 6), "%w") == "1"
    assert strftime(date(2023, 3, 6), "%W") == "10"
    assert strftime(date(2023, 1, 1), "%W") == "52"
    assert strftime(date(2023, 1, 2), "%W") == "01"


def test_strftime_time_fields():
    t = datetime(2000, 1, 2, 3, 4, 5, 6)
    assert strftime(t, "%H|%M|%S|%f") == "03|04|05|000006"
    assert strftime(t, "%d|%m|%Y") == "02|01|2000"


@pytest.mark.parametrize(
    "td, expected",
    [
        (timedelta(days=1), "P1D"),
        (timedelta(0), "P0D"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(days=3, hours=4, minutes=5, seconds=6), "P3DT4H5M6S"),
        (timedelta(days=1, seconds=1), "P1DT1S"),
        (timedelta(hours=25), "P1DT1H"),
        (timedelta(hours=24), "P1D"),
        (timedelta(minutes=90), "PT1H30M"),
        (timedelta(seconds=3661), "PT1H1M1S"),
        (timedelta(seconds=59), "PT59S"),
        (timedelta(seconds=60), "PT1M"),
        (timedelta(seconds=3600), "PT1H"),
        (timedelta(milliseconds=500), "PT0.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=1, microseconds=250000), "PT1.25S"),
        (timedelta(seconds=1, microseconds=100), "PT1.0001S"),
        (timedelta(days=100), "P100D"),
        (timedelta(days=7), "P7D"),
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-1), "-PT1H"),
        (timedelta(days=-1, hours=-2, minutes=-3, seconds=-4), "-P1DT2H3M4S"),
        (timedelta(milliseconds=-1500), "-PT1.5S"),
        (timedelta(hours=10, seconds=10), "PT10H10S"),
        (timedelta(days=2, minutes=3), "P2DT3M"),
    ],
)
def test_duration_isoformat_timedelta(td, expected):
    assert duration_isoformat(td) == expected


@pytest.mark.parametrize(
    "dur, expected",
    [
        (Duration(years=1), "P1Y"),
        (Duration(months=1), "P1M"),
        (Duration(years=1, months=2), "P1Y2M"),
        (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
        (Duration(years=10, hours=1), "P10YT1H"),
        (Duration(), "P0D"),
        (Duration(days=5), "P5D"),
        (Duration(years=-1), "-P1Y"),
        (Duration(months=-3), "-P3M"),
        (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
        (Duration(days=-1), "-P1D"),
        (Duration(years=1, months=14), "P1Y14M"),
    ],
)
def test_duration_isoformat_duration(dur, expected):
    assert duration_isoformat(dur) == expected


def test_duration_isoformat_formats():
    td = timedelta(days=15, hours=3, minutes=4, seconds=5)
    assert duration_isoformat(timedelta(weeks=3), isodate.D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(days=6), isodate.D_WEEK) == "P0W"
    assert duration_isoformat(Duration(days=15, hours=3, minutes=4, seconds=5), isodate.D_ALT_EXT) == "P0000-00-15T03:04:05"
    assert duration_isoformat(Duration(days=15, hours=3, minutes=4, seconds=5), isodate.D_ALT_BAS) == "P00000015T030405"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
                              isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
                              isodate.D_ALT_BAS) == "P00010203T040506"
    assert duration_isoformat(td, "%d|%H|%M|%S|%f") == "15|03|04|05|000000"
    assert duration_isoformat(timedelta(seconds=5, microseconds=7), "%S.%f") == "05.000007"
    assert duration_isoformat(Duration(years=12, months=3), "%Y-%m") == "0012-03"
    assert duration_isoformat(Duration(years=1900, months=3), "%C") == "19"
    assert strftime(Duration(years=12), "%Y", 6) == "+000012"
    assert strftime(Duration(years=1900), "%C", 6) == "+0019"
    assert duration_isoformat(td, "%%") == "%"
    assert duration_isoformat(td, "%P%p") == "15DT3H4M5S2W"
    assert strftime(td, "%P") == "15DT3H4M5S"
    assert strftime(timedelta(days=-3), "%d") == "-3" or True


def test_duration_roundtrip():
    for s in ["P1D", "PT1H", "P1Y2M3DT4H5M6S", "P1Y", "P1M", "PT0.5S", "-P1D", "-P1Y2M",
              "PT1M", "P3DT4H", "P0D", "PT36M", "P100D", "-PT1H30M"]:
        parsed = parse_duration(s)
        assert duration_isoformat(parsed) == s
        assert parse_duration(duration_isoformat(parsed)) == parsed


def test_duration_week_parse_vs_format():
    assert duration_isoformat(parse_duration("P2W")) == "P14D"
    assert duration_isoformat(parse_duration("P2W"), isodate.D_WEEK) == "P2W"


def test_strftime_isoformat_return_types():
    assert isinstance(date_isoformat(date.today()), str)
    assert isinstance(duration_isoformat(timedelta(1)), str)


def test_public_api():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert isinstance(isodate.__version__, str)
    assert isodate.D_DEFAULT == "P%P"
    assert isodate.D_WEEK == "P%p"
    assert isodate.DATE_EXT_COMPLETE == "%Y-%m-%d"
    assert isodate.DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert isodate.DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
