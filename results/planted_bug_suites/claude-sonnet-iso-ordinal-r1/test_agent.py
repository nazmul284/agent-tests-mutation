import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    ISO8601Error,
    Duration,
    parse_date,
    parse_datetime,
    parse_duration,
    parse_time,
    parse_tzinfo,
    date_isoformat,
    time_isoformat,
    datetime_isoformat,
    duration_isoformat,
    tz_isoformat,
    strftime,
    UTC,
    FixedOffset,
)
from isodate.duration import max_days_in_month, fquotmod


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize(
    "s,expected",
    [
        ("2004-02-29", date(2004, 2, 29)),
        ("20040229", date(2004, 2, 29)),
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("2004-001", date(2004, 1, 1)),
        ("2004001", date(2004, 1, 1)),
        ("2004-060", date(2004, 2, 29)),
        ("2003-060", date(2003, 3, 1)),
        ("2004-366", date(2004, 12, 31)),
        ("2003-365", date(2003, 12, 31)),
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        ("2004-W01-1", date(2003, 12, 29)),
        ("2004-W53-7", date(2005, 1, 2)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W52-7", date(2011, 1, 2)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
    ],
)
def test_parse_date(s, expected):
    assert parse_date(s) == expected


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=9) == date(1985, 4, 9)
    assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)


def test_parse_date_expanded():
    assert parse_date("+0019850412", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985-102", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    with pytest.raises(ISO8601Error):
        parse_date("1985-04-12", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("+1985-04-12")


@pytest.mark.parametrize(
    "s", ["", "abc", "1985-4-12", "1985-04-1", "1985/04/12", "198", "1985-W1", "1985-0001",
          "1985-04-12T", " 1985-04-12", "1985-04-12 ", "1985-04-12\n"]
)
def test_parse_date_bad_format(s):
    with pytest.raises(ISO8601Error):
        parse_date(s)


@pytest.mark.parametrize("s", ["1985-13-01", "1985-02-30", "2003-02-29", "1985-04-00"])
def test_parse_date_invalid_values(s):
    with pytest.raises(ValueError):
        parse_date(s)


def test_build_date_regexps_cached():
    a = isodate.isodates.build_date_regexps(4, False)
    assert a is isodate.isodates.build_date_regexps(4, False)
    assert a is not isodate.isodates.build_date_regexps(4, True)


def test_date_isoformat_default_and_formats():
    d = date(1985, 4, 12)
    assert date_isoformat(d) == "1985-04-12"
    assert date_isoformat(d, isodate.DATE_BAS_COMPLETE) == "19850412"
    assert date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "1985-102"
    assert date_isoformat(d, isodate.DATE_BAS_ORD_COMPLETE) == "1985102"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "1985-W15-5"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK_COMPLETE) == "1985W155"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK) == "1985-W15"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK) == "1985W15"
    assert date_isoformat(d, isodate.DATE_EXT_MONTH) == "1985-04"
    assert date_isoformat(d, isodate.DATE_BAS_MONTH) == "198504"
    assert date_isoformat(d, isodate.DATE_YEAR) == "1985"
    assert date_isoformat(d, isodate.DATE_CENTURY) == "19"
    assert date_isoformat(d, "%Y", 6) == "+001985"
    assert date_isoformat(d, "%C", 6) == "+0019"
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"


def test_date_roundtrip_all_formats():
    fmts = [
        isodate.DATE_EXT_COMPLETE, isodate.DATE_BAS_COMPLETE,
        isodate.DATE_EXT_ORD_COMPLETE, isodate.DATE_BAS_ORD_COMPLETE,
        isodate.DATE_EXT_WEEK_COMPLETE, isodate.DATE_BAS_WEEK_COMPLETE,
    ]
    d = date(2003, 12, 30)
    for _ in range(800):
        for f in fmts:
            assert parse_date(date_isoformat(d, f)) == d, (d, f)
        d += timedelta(days=1)
    # year boundaries
    for y in (1999, 2000, 2004, 2008, 2009, 2010, 2015, 2020, 2021):
        for md in ((1, 1), (12, 31), (1, 4)):
            dd = date(y, *md)
            for f in fmts:
                assert parse_date(date_isoformat(dd, f)) == dd


def test_ordinal_first_and_last_day_every_year():
    for y in range(1900, 2101):
        assert parse_date("%d-001" % y) == date(y, 1, 1)
        last = date(y, 12, 31)
        assert parse_date("%d-%03d" % (y, last.timetuple().tm_yday)) == last


# ---------------------------------------------------------------- times
@pytest.mark.parametrize(
    "s,expected",
    [
        ("20:23:19", time(20, 23, 19)),
        ("202319", time(20, 23, 19)),
        ("T20:23:19", time(20, 23, 19)),
        ("T202319", time(20, 23, 19)),
        ("20:23:19.5", time(20, 23, 19, 500000)),
        ("20:23:19,5", time(20, 23, 19, 500000)),
        ("202319.123456", time(20, 23, 19, 123456)),
        ("20:23:19.000001", time(20, 23, 19, 1)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
        ("20:23", time(20, 23)),
        ("2023", time(20, 23)),
        ("20:23.5", time(20, 23, 30)),
        ("2023,25", time(20, 23, 15)),
        ("20", time(20)),
        ("20.5", time(20, 30)),
        ("20,25", time(20, 15)),
        ("T20", time(20)),
    ],
)
def test_parse_time(s, expected):
    assert parse_time(s) == expected


def test_parse_time_tz():
    t = parse_time("20:23:19Z")
    assert t.tzinfo is not None and t.utcoffset() == timedelta(0)
    t = parse_time("20:23:19+05:30")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    t = parse_time("20:23:19-0530")
    assert t.utcoffset() == -timedelta(hours=5, minutes=30)
    t = parse_time("20:23:19+05")
    assert t.utcoffset() == timedelta(hours=5)
    t = parse_time("20:23:19-05")
    assert t.utcoffset() == -timedelta(hours=5)
    t = parse_time("2023Z")
    assert t.utcoffset() == timedelta(0)
    t = parse_time("20+01")
    assert t.hour == 20 and t.utcoffset() == timedelta(hours=1)
    assert parse_time("20:23:19").tzinfo is None
    assert parse_time("20:23:19.5-03:00").microsecond == 500000


@pytest.mark.parametrize("s", ["", "2", "202", "20:2", "20:23:1", "20:23:19:", "ab:cd", "20:23:19 ",
                               "20:23:19+5", "20:23:19+05:3", "20:23:19z"])
def test_parse_time_bad(s):
    with pytest.raises(ISO8601Error):
        parse_time(s)


@pytest.mark.parametrize("s", ["24:00:00", "25:00:00", "20:60:00", "20:23:60"])
def test_parse_time_out_of_range(s):
    with pytest.raises(ValueError):
        parse_time(s)


def test_time_isoformat():
    t = time(20, 23, 19)
    assert time_isoformat(t) == "20:23:19"
    assert time_isoformat(time(20, 23, 19, tzinfo=UTC)) == "20:23:19Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"
    assert time_isoformat(t, isodate.TIME_BAS_COMPLETE) == "202319"
    assert time_isoformat(t, isodate.TIME_EXT_MINUTE) == "20:23"
    assert time_isoformat(t, isodate.TIME_BAS_MINUTE) == "2023"
    assert time_isoformat(t, isodate.TIME_HOUR) == "20"
    assert time_isoformat(time(1, 2, 3, 4500), "%H:%M:%S.%f") == "01:02:03.004500"
    assert time_isoformat(time(0, 0, 0)) == "00:00:00"


def test_time_roundtrip():
    for t in (time(0, 0, 0), time(1, 2, 3), time(23, 59, 59, 999999), time(12, 0, 0, 1)):
        assert parse_time(time_isoformat(t, "%H:%M:%S.%f")) == t
        assert parse_time(time_isoformat(t, "%H%M%S.%f")) == t


# ---------------------------------------------------------------- datetime
def test_parse_datetime():
    assert parse_datetime("2004-02-29T20:23:19") == datetime(2004, 2, 29, 20, 23, 19)
    assert parse_datetime("20040229T202319") == datetime(2004, 2, 29, 20, 23, 19)
    assert parse_datetime("2004-060T20:23:19Z") == datetime(2004, 2, 29, 20, 23, 19, tzinfo=UTC)
    assert parse_datetime("2004-W01-1T00:00:00+01:00") == datetime(
        2003, 12, 29, 0, 0, 0, tzinfo=FixedOffset(1, 0, "+01:00"))
    dt = parse_datetime("1985-04-12T10:15:30.5-04:00")
    assert dt.microsecond == 500000
    assert dt.utcoffset() == -timedelta(hours=4)
    assert parse_datetime("1985-04-12T10:15:30+0530").utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("s", ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10:15:30T1", "", "T10:15"])
def test_parse_datetime_bad(s):
    with pytest.raises(ISO8601Error):
        parse_datetime(s)


def test_datetime_isoformat():
    dt = datetime(2004, 2, 29, 20, 23, 19)
    assert datetime_isoformat(dt) == "2004-02-29T20:23:19"
    assert datetime_isoformat(dt.replace(tzinfo=UTC)) == "2004-02-29T20:23:19Z"
    assert datetime_isoformat(dt.replace(tzinfo=FixedOffset(-3, -30, "x"))) == "2004-02-29T20:23:19-03:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_COMPLETE) == "20040229T202319"
    assert datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE) == "2004-060T20:23:19"
    assert datetime_isoformat(dt, isodate.DT_BAS_ORD_COMPLETE) == "2004060T202319"
    assert datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE) == "2004-W09-7T20:23:19"
    assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "2004W097T202319"


def test_datetime_roundtrip():
    dt = datetime(2012, 12, 31, 23, 59, 59, 123456, tzinfo=FixedOffset(2, 0, "+02:00"))
    assert parse_datetime(datetime_isoformat(dt, "%Y-%m-%dT%H:%M:%S.%f%Z")) == dt


# ---------------------------------------------------------------- tz
def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    assert parse_tzinfo("+01:30").utcoffset(None) == timedelta(hours=1, minutes=30)
    assert parse_tzinfo("-0130").utcoffset(None) == -timedelta(hours=1, minutes=30)
    assert parse_tzinfo("+02").utcoffset(None) == timedelta(hours=2)
    assert parse_tzinfo("-00:30").utcoffset(None) == -timedelta(minutes=30)
    assert parse_tzinfo("+05:30").tzname(None) == "+05:30"


def test_tz_isoformat():
    n = datetime(2000, 1, 1)
    assert tz_isoformat(n) == ""
    assert tz_isoformat(n.replace(tzinfo=UTC)) == "Z"
    assert tz_isoformat(n.replace(tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    d = n.replace(tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(d) == "+05:30"
    assert tz_isoformat(d, "%Z") == "+05:30"
    assert tz_isoformat(d, "%z") == "+0530"
    assert tz_isoformat(d, "%h") == "+05"
    d = n.replace(tzinfo=FixedOffset(-9, -45, "x"))
    assert tz_isoformat(d) == "-09:45"
    assert tz_isoformat(d, "%z") == "-0945"
    assert tz_isoformat(d, "%h") == "-09"
    with pytest.raises(ValueError):
        tz_isoformat(d, "%q")
    with pytest.raises(OverflowError):
        tz_isoformat(n.replace(tzinfo=_Big()))


class _Big(isodate.tzinfo.tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=100)

    def dst(self, dt):
        return timedelta(0)


def test_fixed_offset_and_utc():
    f = FixedOffset(1, 30, "n")
    assert f.utcoffset(None) == timedelta(hours=1, minutes=30)
    assert f.dst(None) == timedelta(0)
    assert f.tzname(None) == "n"
    assert "n" in repr(f)
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert UTC.dst(None) == timedelta(0)
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert pickle.loads(pickle.dumps(f)).utcoffset(None) == f.utcoffset(None)


def test_build_tzinfo():
    from isodate.isotzinfo import build_tzinfo
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    assert build_tzinfo("-05:30", "-", 5, 30).utcoffset(None) == -timedelta(hours=5, minutes=30)
    assert build_tzinfo("+05:30", "+", 5, 30).utcoffset(None) == timedelta(hours=5, minutes=30)


# ---------------------------------------------------------------- duration parse
def test_parse_duration_timedelta():
    assert parse_duration("P1D") == timedelta(days=1)
    assert parse_duration("PT1H") == timedelta(hours=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    assert parse_duration("PT1S") == timedelta(seconds=1)
    assert parse_duration("P2W") == timedelta(weeks=2)
    assert parse_duration("P1DT2H3M4S") == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert parse_duration("PT0.5S") == timedelta(milliseconds=500)
    assert parse_duration("PT0,5S") == timedelta(milliseconds=500)
    assert parse_duration("P0.5D") == timedelta(hours=12)
    assert parse_duration("PT36H") == timedelta(hours=36)
    assert parse_duration("P0D") == timedelta(0)
    assert parse_duration("PT0S") == timedelta(0)
    assert isinstance(parse_duration("P1D"), timedelta)
    assert parse_duration("P1W1D") == timedelta(days=8)
    assert parse_duration("PT1H1M1S") == timedelta(seconds=3661)


def test_parse_duration_negative():
    assert parse_duration("-P1D") == timedelta(days=-1)
    assert parse_duration("-PT1H30M") == -timedelta(hours=1, minutes=30)
    assert parse_duration("+P1D") == timedelta(days=1)
    d = parse_duration("-P1Y2M")
    assert isinstance(d, Duration)
    assert d.years == -1 and d.months == -2
    d = parse_duration("-P1Y2M3DT4H")
    assert d.years == -1 and d.months == -2 and d.tdelta == -timedelta(days=3, hours=4)


def test_parse_duration_duration():
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P1Y")
    assert d.years == 1 and d.months == 0 and d.tdelta == timedelta(0)
    d = parse_duration("P3M")
    assert d.years == 0 and d.months == 3
    d = parse_duration("PT3M")
    assert isinstance(d, timedelta) and d == timedelta(minutes=3)
    d = parse_duration("P1M1M".replace("P1M1M", "P1MT1M"))
    assert d.months == 1 and d.tdelta == timedelta(minutes=1)
    d = parse_duration("P1.5Y")
    assert d.years == Decimal("1.5")
    d = parse_duration("P0.5M")
    assert d.months == Decimal("0.5")
    d = parse_duration("P1Y", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration) and d.tdelta == timedelta(days=1)
    d = parse_duration("P0Y0M1D")
    assert d == timedelta(days=1)
    d = parse_duration("P2W", as_timedelta_if_possible=False)
    assert d.tdelta == timedelta(weeks=2)


def test_parse_duration_alternative():
    d = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize("s", ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P-1D", "PD", "P1D1Y",
                               "P1Y2M3D4H", "P 1D", "P1D ", "X", "PT1H1D"])
def test_parse_duration_bad(s):
    with pytest.raises(ISO8601Error):
        parse_duration(s)


def test_parse_duration_type():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


# ---------------------------------------------------------------- duration format
def test_duration_isoformat_timedelta():
    assert duration_isoformat(timedelta(0)) == "P0D"
    assert duration_isoformat(timedelta(days=1)) == "P1D"
    assert duration_isoformat(timedelta(days=400)) == "P400D"
    assert duration_isoformat(timedelta(hours=1)) == "PT1H"
    assert duration_isoformat(timedelta(hours=25)) == "P1DT1H"
    assert duration_isoformat(timedelta(minutes=5)) == "PT5M"
    assert duration_isoformat(timedelta(seconds=7)) == "PT7S"
    assert duration_isoformat(timedelta(days=1, hours=2, minutes=3, seconds=4)) == "P1DT2H3M4S"
    assert duration_isoformat(timedelta(seconds=1.5)) == "PT1.5S"
    assert duration_isoformat(timedelta(microseconds=1)) == "PT0.000001S"
    assert duration_isoformat(timedelta(milliseconds=250)) == "PT0.25S"
    assert duration_isoformat(timedelta(minutes=1, microseconds=10)) == "PT1M0.00001S"
    assert duration_isoformat(-timedelta(days=1)) == "-P1D"
    assert duration_isoformat(-timedelta(hours=1, minutes=30)) == "-PT1H30M"
    assert duration_isoformat(-timedelta(seconds=1)) == "-PT1S"
    assert duration_isoformat(timedelta(days=-1, hours=12)) == "-PT12H"
    assert duration_isoformat(timedelta(hours=59, minutes=59, seconds=59)) == "P2DT11H59M59S"
    assert duration_isoformat(timedelta(minutes=60)) == "PT1H"
    assert duration_isoformat(timedelta(hours=24)) == "P1D"


def test_duration_isoformat_duration():
    assert duration_isoformat(Duration(years=1)) == "P1Y"
    assert duration_isoformat(Duration(months=2)) == "P2M"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)) == "P1Y2M3DT4H5M6S"
    assert duration_isoformat(Duration()) == "P0D"
    assert duration_isoformat(Duration(years=-1, months=-2)) == "-P1Y2M"
    assert duration_isoformat(Duration(years=1, months=14)) == "P1Y14M"
    assert duration_isoformat(Duration(months=1, minutes=1)) == "P1MT1M"
    assert duration_isoformat(Duration(years=1, days=-1)) != ""
    assert duration_isoformat(Duration(years=Decimal("1.5"))) == "P1.5Y"


def test_duration_isoformat_week_and_alt():
    assert duration_isoformat(timedelta(weeks=3), isodate.D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=20), isodate.D_WEEK) == "P2W"
    assert duration_isoformat(-timedelta(weeks=3), isodate.D_WEEK) == "-P3W"
    assert duration_isoformat(timedelta(days=3, hours=4, minutes=5, seconds=6),
                              "%d %H %M %S") == "03 04 05 06"
    assert duration_isoformat(timedelta(hours=1, minutes=2, seconds=3, microseconds=45),
                              "%H:%M:%S.%f") == "01:02:03.000045"
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d, "%Y-%m-%d %H:%M:%S") == "0001-02-03 04:05:06"
    assert duration_isoformat(d, "%Y") == "0001"
    assert strftime(d, "%Y", 6) == "+000001"
    assert duration_isoformat(Duration(years=250), "%C") == "02"
    assert duration_isoformat(Duration(months=5), "%m") == "05"
    assert duration_isoformat(timedelta(days=1), "100%%") == "100%"


def test_duration_roundtrip():
    for s in ["P1Y2M3DT4H5M6S", "P1Y", "P2M", "P3D", "PT4H", "PT5M", "PT6.5S", "P1DT1S",
              "-P1Y2M", "-P3DT4H", "P10Y10M10DT10H10M10.1S"]:
        assert duration_isoformat(parse_duration(s)) == s
    for td in [timedelta(days=5, seconds=3, microseconds=7), -timedelta(hours=7, microseconds=1)]:
        assert parse_duration(duration_isoformat(td)) == td
    for d in [Duration(years=2, months=3, days=4, hours=5), Duration(months=-5, hours=-1)]:
        assert parse_duration(duration_isoformat(d)) == d


# ---------------------------------------------------------------- strftime
def test_strftime_misc():
    d = date(2004, 2, 9)
    assert strftime(d, "%d/%m/%Y") == "09/02/2004"
    assert strftime(d, "%j") == "040"
    assert strftime(d, "%w") == "1"
    assert strftime(date(2004, 2, 8), "%w") == "7"
    assert strftime(d, "%W") == "07"
    assert strftime(d, "100%%") == "100%"
    assert strftime(d, "%q") == "%q"
    assert strftime(date(1999, 12, 31), "%j") == "365"
    assert strftime(date(2000, 12, 31), "%j") == "366"
    assert strftime(date(2000, 1, 1), "%j") == "001"
    assert strftime(date(2005, 1, 1), "%Y-%W") == "2005-53"
    assert strftime(date(1901, 3, 4), "%C") == "19"
    assert strftime(date(2000, 3, 4), "%C") == "20"
    assert strftime(datetime(2000, 1, 1, 1, 2, 3, 400), "%f") == "000400"
    assert strftime(datetime(2000, 1, 1, 13, 2, 3), "%H%M%S") == "130203"


# ---------------------------------------------------------------- Duration class
def test_helpers():
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2004, 2) == 29
    assert max_days_in_month(2003, 2) == 28
    assert max_days_in_month(2100, 2) == 28
    assert max_days_in_month(2400, 2) == 29
    for m in (1, 3, 5, 7, 8, 10, 12):
        assert max_days_in_month(2001, m) == 31
    for m in (4, 6, 9, 11):
        assert max_days_in_month(2001, m) == 30
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))


def test_duration_add_date():
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert Duration(months=1) + date(2001, 1, 31) == date(2001, 2, 28)
    assert date(2000, 1, 1) + Duration(years=1) == date(2001, 1, 1)
    assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
    assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)
    assert date(2000, 11, 15) + Duration(months=2) == date(2001, 1, 15)
    assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
    assert date(2000, 12, 15) + Duration(months=12) == date(2001, 12, 15)
    assert date(2000, 12, 15) + Duration(months=13) == date(2002, 1, 15)
    assert date(2000, 1, 15) + Duration(months=-1) == date(1999, 12, 15)
    assert date(2000, 1, 15) + Duration(months=-13) == date(1998, 12, 15)
    assert date(2000, 1, 15) + Duration(years=1, months=1, days=1) == date(2001, 2, 16)
    assert datetime(2000, 1, 31, 10, 0) + Duration(months=1, hours=2) == datetime(2000, 2, 29, 12, 0)
    assert date(2000, 1, 1) + Duration(days=10) == date(2000, 1, 11)
    assert date(2000, 5, 31) + Duration(months=1) == date(2000, 6, 30)
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("0.5"))


def test_duration_sub_from_date():
    assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
    assert date(2000, 1, 15) - Duration(months=13) == date(1998, 12, 15)
    assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
    assert date(2000, 6, 15) - Duration(years=1, months=1, days=1) == date(1999, 5, 14)
    assert date(2000, 6, 15) - Duration(months=-1) == date(2000, 7, 15)
    assert datetime(2000, 3, 31, 12) - Duration(months=1, hours=2) == datetime(2000, 2, 29, 10)
    assert date(2000, 12, 15) - Duration(months=-2) == date(2001, 2, 15)
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("1.5"))


def test_duration_arithmetic():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=2, months=3, days=4)
    s = a + b
    assert (s.years, s.months, s.tdelta) == (3, 5, timedelta(days=7))
    s = b - a
    assert (s.years, s.months, s.tdelta) == (1, 1, timedelta(days=1))
    s = a + timedelta(days=1)
    assert (s.years, s.months, s.tdelta) == (1, 2, timedelta(days=4))
    s = timedelta(days=1) + a
    assert (s.years, s.months, s.tdelta) == (1, 2, timedelta(days=4))
    s = a - timedelta(days=1)
    assert (s.years, s.months, s.tdelta) == (1, 2, timedelta(days=2))
    s = timedelta(days=10) - a
    assert (s.years, s.months, s.tdelta) == (-1, -2, timedelta(days=7))
    n = -a
    assert (n.years, n.months, n.tdelta) == (-1, -2, -timedelta(days=3))
    m = a * 3
    assert (m.years, m.months, m.tdelta) == (3, 6, timedelta(days=9))
    m = 2 * a
    assert (m.years, m.months, m.tdelta) == (2, 4, timedelta(days=6))
    with pytest.raises(TypeError):
        a * 1.5
    with pytest.raises(TypeError):
        a + 1
    with pytest.raises(TypeError):
        a - "x"
    assert a + b == b + a


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(months=1, days=1) != timedelta(days=1)
    assert not (Duration(months=1) == timedelta(0))
    assert Duration(months=1) != timedelta(0)
    assert Duration(days=1) != Duration(days=2)
    assert not (Duration(days=1) != Duration(days=1))
    assert Duration(hours=24) == Duration(days=1)
    assert Duration(years=1, days=1) != Duration(years=1)
    assert Duration(days=1) == Duration(days=1)
    assert not (Duration(days=1) == Duration(days=2))
    assert not (Duration(years=1) == Duration(years=2))
    assert Duration() == timedelta(0)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1) != 5


def test_duration_hash_and_str_repr():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2
    assert str(Duration(years=1, months=2, days=3)) == "1 years, 2 months, 3 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(years=1)) == "1 years, 0:00:00"
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    r = repr(Duration(years=1, months=2, days=3, seconds=4, microseconds=5))
    assert r.endswith("Duration(3, 4, 5, years=1, months=2)")


def test_duration_attrs_and_ctor():
    d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=7,
                 months=8, years=9)
    assert d.years == 9 and d.months == 8
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert d.days == d.tdelta.days
    assert d.seconds == d.tdelta.seconds
    assert d.microseconds == d.tdelta.microseconds
    assert d.total_seconds() == d.tdelta.total_seconds()
    assert Duration(years=1.5).years == Decimal("1.5")
    assert Duration(months=Decimal("2.5")).months == Decimal("2.5")


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d and d2.years == 1 and d2.months == 2


def test_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1, days=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=366)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 2))


def test_package_exports():
    for name in ("parse_date", "parse_time", "parse_datetime", "parse_duration", "parse_tzinfo",
                 "Duration", "UTC", "LOCAL", "FixedOffset", "ISO8601Error", "strftime",
                 "date_isoformat", "time_isoformat", "datetime_isoformat", "duration_isoformat",
                 "tz_isoformat", "D_DEFAULT", "D_WEEK", "DATE_EXT_COMPLETE", "TZ_BAS"):
        assert hasattr(isodate, name), name
    assert issubclass(ISO8601Error, ValueError)
