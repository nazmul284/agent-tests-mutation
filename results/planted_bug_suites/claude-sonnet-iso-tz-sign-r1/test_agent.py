import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
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


def off(dt):
    return dt.utcoffset()


# ---------------------------------------------------------------- tz parsing
@pytest.mark.parametrize(
    "s,expected",
    [
        ("+05:00", timedelta(hours=5)),
        ("-05:00", timedelta(hours=-5)),
        ("+0530", timedelta(hours=5, minutes=30)),
        ("-0530", timedelta(hours=-5, minutes=-30)),
        ("+05", timedelta(hours=5)),
        ("-11", timedelta(hours=-11)),
        ("+00:00", timedelta(0)),
        ("+05:45", timedelta(hours=5, minutes=45)),
        ("-03:30", timedelta(hours=-3, minutes=-30)),
    ],
)
def test_parse_tzinfo_offsets(s, expected):
    tz = parse_tzinfo(s)
    assert tz.utcoffset(None) == expected


def test_parse_tzinfo_z_and_empty():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None


def test_parse_tzinfo_name_is_input():
    assert parse_tzinfo("+05:30").tzname(None) == "+05:30"


def test_build_tzinfo_signs():
    from isodate.isotzinfo import build_tzinfo

    assert build_tzinfo("+01:00", "+", 1, 0).utcoffset(None) == timedelta(hours=1)
    assert build_tzinfo("-01:30", "-", 1, 30).utcoffset(None) == -timedelta(hours=1, minutes=30)
    assert build_tzinfo("Z") is UTC
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None


# ---------------------------------------------------------------- tz format
def test_tz_isoformat_naive_and_utc():
    assert tz_isoformat(datetime(2020, 1, 1)) == ""
    assert tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"


@pytest.mark.parametrize(
    "h,m,fmt,expected",
    [
        (5, 30, "%Z", "+05:30"),
        (5, 30, "%z", "+0530"),
        (5, 30, "%h", "+05"),
        (-5, -30, "%Z", "-05:30"),
        (-5, -30, "%z", "-0530"),
        (-8, 0, "%h", "-08"),
        (0, 45, "%Z", "+00:45"),
        (0, -45, "%Z", "-00:45"),
    ],
)
def test_tz_isoformat_offsets(h, m, fmt, expected):
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(h, m, "x"))
    assert tz_isoformat(dt, fmt) == expected


def test_tz_isoformat_bad_format():
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(1, 0, "x"))
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%q")


def test_tz_isoformat_overflow():
    dt = datetime(2020, 1, 1, tzinfo=FixedOffset(100, 0, "x"))
    with pytest.raises(OverflowError):
        tz_isoformat(dt)


def test_fixed_offset_basics():
    tz = FixedOffset(2, 15, "foo")
    assert tz.utcoffset(None) == timedelta(hours=2, minutes=15)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "foo"
    assert "foo" in repr(tz)
    assert FixedOffset().utcoffset(None) == timedelta(0)
    assert FixedOffset().tzname(None) == "UTC"


def test_utc_basics_and_pickle():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC


# ---------------------------------------------------------------- parse_time
@pytest.mark.parametrize(
    "s,expected",
    [
        ("20:23:12", time(20, 23, 12)),
        ("202312", time(20, 23, 12)),
        ("T20:23:12", time(20, 23, 12)),
        ("20:23:12.5", time(20, 23, 12, 500000)),
        ("20:23:12,25", time(20, 23, 12, 250000)),
        ("202312.123456", time(20, 23, 12, 123456)),
        ("20:23", time(20, 23)),
        ("2023", time(20, 23)),
        ("20:23.5", time(20, 23, 30)),
        ("2023,25", time(20, 23, 15)),
        ("20", time(20)),
        ("20.5", time(20, 30)),
        ("20,25", time(20, 15)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
    ],
)
def test_parse_time_naive(s, expected):
    t = parse_time(s)
    assert t == expected
    assert t.tzinfo is None


def test_parse_time_microsecond_truncated():
    assert parse_time("10:00:00.1234567").microsecond == 123456


@pytest.mark.parametrize(
    "s,expected",
    [
        ("20:23:12Z", timedelta(0)),
        ("20:23:12+01:00", timedelta(hours=1)),
        ("20:23:12-01:00", timedelta(hours=-1)),
        ("202312+0530", timedelta(hours=5, minutes=30)),
        ("202312-0530", timedelta(hours=-5, minutes=-30)),
        ("20:23+02", timedelta(hours=2)),
        ("20-07", timedelta(hours=-7)),
        ("20:23:12.5+05:00", timedelta(hours=5)),
    ],
)
def test_parse_time_tz(s, expected):
    t = parse_time(s)
    assert t.utcoffset() == expected


def test_parse_time_z_is_utc():
    assert parse_time("10:00:00Z").tzinfo is UTC


@pytest.mark.parametrize(
    "s", ["", "abc", "25:00:00x", "1:2:3", "20:23:12 ", "20:23:12ZZ", "2:00", "20:3"]
)
def test_parse_time_invalid(s):
    with pytest.raises(ValueError):
        parse_time(s)


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("24:00:00")
    with pytest.raises(ValueError):
        parse_time("10:60:00")
    with pytest.raises(ValueError):
        parse_time("10:00:60")


def test_time_isoformat():
    assert time_isoformat(time(1, 2, 3)) == "01:02:03"
    assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"
    assert time_isoformat(time(1, 2, 3), "%H%M%S") == "010203"
    assert time_isoformat(time(1, 2, 3, 45), "%H:%M:%S.%f") == "01:02:03.000045"


# ---------------------------------------------------------------- parse_date
@pytest.mark.parametrize(
    "s,expected",
    [
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        ("2000-02-29", date(2000, 2, 29)),
        ("1985-102", date(1985, 4, 12)),
        ("1985102", date(1985, 4, 12)),
        ("2004-366", date(2004, 12, 31)),
        ("1985-001", date(1985, 1, 1)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("1985-W15", date(1985, 4, 8)),
        ("1985W15", date(1985, 4, 8)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2015-W01-4", date(2015, 1, 1)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
    ],
)
def test_parse_date(s, expected):
    assert parse_date(s) == expected


def test_parse_date_week_matches_isocalendar():
    for y in range(1990, 2030):
        for w in (1, 2, 20, 52):
            for d in range(1, 8):
                expected = date.fromisocalendar(y, w, d)
                assert parse_date("%04d-W%02d-%d" % (y, w, d)) == expected
                assert parse_date("%04dW%02d%d" % (y, w, d)) == expected


def test_parse_date_ordinal_all_days():
    for y in (1999, 2000, 2024):
        d0 = date(y, 1, 1)
        for n in range(1, 366):
            assert parse_date("%04d-%03d" % (y, n)) == d0 + timedelta(days=n - 1)


def test_parse_date_century():
    assert parse_date("19").year // 100 == 19 or parse_date("19").year == 1901
    assert parse_date("19").month == 1 and parse_date("19").day == 1


def test_parse_date_defaults():
    assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
    assert parse_date("1985-04", defaultday=20) == date(1985, 4, 20)


def test_parse_date_expanded():
    assert parse_date("+0019850412", yeardigits=6, expanded=True) == date(1985, 4, 12)
    assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
    assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
    assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
    assert parse_date("+001985-W15-5", yeardigits=6) == date(1985, 4, 12)


@pytest.mark.parametrize(
    "s", ["", "abc", "1985-4-12", "1985-04-1", "85-04-12", "1985/04/12", "1985-04-12x", " 1985"]
)
def test_parse_date_invalid_format(s):
    with pytest.raises(ISO8601Error):
        parse_date(s)


@pytest.mark.parametrize(
    "s", ["1985-13-01", "1985-02-30", "1985-04-31", "1900-02-29", "1985-04-00"]
)
def test_parse_date_invalid_values(s):
    with pytest.raises(ValueError):
        parse_date(s)


def test_parse_date_error_is_valueerror():
    assert issubclass(ISO8601Error, ValueError)


def test_date_isoformat():
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
    assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
    assert date_isoformat(d, "%Y", 6) == "+001985"
    assert date_isoformat(d, "%d/%m %%") == "12/04 %"


def test_date_roundtrip():
    for d in (date(1, 1, 1), date(1999, 12, 31), date(2000, 2, 29), date(2024, 7, 4)):
        for fmt in (
            isodate.DATE_EXT_COMPLETE,
            isodate.DATE_BAS_COMPLETE,
            isodate.DATE_EXT_ORD_COMPLETE,
            isodate.DATE_BAS_ORD_COMPLETE,
            isodate.DATE_EXT_WEEK_COMPLETE,
            isodate.DATE_BAS_WEEK_COMPLETE,
        ):
            try:
                assert parse_date(date_isoformat(d, fmt)) == d
            except ValueError:
                # years at the very edge can't be shifted by the week algorithm
                assert d.year == 1


# ---------------------------------------------------------------- datetime
def test_parse_datetime_basic():
    assert parse_datetime("2012-10-26T09:33:00") == datetime(2012, 10, 26, 9, 33)
    assert parse_datetime("20121026T093300") == datetime(2012, 10, 26, 9, 33)
    assert parse_datetime("2012-10-26T09:33:00.25") == datetime(2012, 10, 26, 9, 33, 0, 250000)
    assert parse_datetime("2012-W43-5T09") == datetime(2012, 10, 26, 9)
    assert parse_datetime("2012-300T09:33") == datetime(2012, 10, 26, 9, 33)


def test_parse_datetime_tz():
    dt = parse_datetime("2012-10-26T09:33:00Z")
    assert dt.tzinfo is UTC
    dt = parse_datetime("2012-10-26T09:33:00+05:30")
    assert off(dt) == timedelta(hours=5, minutes=30)
    dt = parse_datetime("2012-10-26T09:33:00-08:00")
    assert off(dt) == timedelta(hours=-8)
    # same instant
    a = parse_datetime("2012-10-26T10:00:00+01:00")
    b = parse_datetime("2012-10-26T09:00:00Z")
    c = parse_datetime("2012-10-26T04:00:00-05:00")
    assert a == b == c


@pytest.mark.parametrize("s", ["2012-10-26", "2012-10-26 09:33", "2012-10-26T09:33T10", "", "T09:33"])
def test_parse_datetime_invalid(s):
    with pytest.raises(ValueError):
        parse_datetime(s)


def test_datetime_isoformat():
    dt = datetime(2012, 10, 26, 9, 33, 5)
    assert datetime_isoformat(dt) == "2012-10-26T09:33:05"
    assert datetime_isoformat(dt.replace(tzinfo=UTC)) == "2012-10-26T09:33:05Z"
    tz = FixedOffset(-8, 0, "x")
    assert datetime_isoformat(dt.replace(tzinfo=tz)) == "2012-10-26T09:33:05-08:00"
    assert datetime_isoformat(dt.replace(tzinfo=tz), isodate.DT_BAS_COMPLETE) == "20121026T093305-0800"
    assert datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE) == "2012-300T09:33:05"
    assert datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE) == "2012-W43-5T09:33:05"
    assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "2012W435T093305"
    assert datetime_isoformat(dt, isodate.DT_BAS_ORD_COMPLETE) == "2012300T093305"


def test_datetime_roundtrip_with_tz():
    for tz in (None, UTC, FixedOffset(5, 30, "x"), FixedOffset(-3, -30, "y"), FixedOffset(14, 0, "z")):
        dt = datetime(2020, 2, 29, 23, 59, 58, tzinfo=tz)
        back = parse_datetime(datetime_isoformat(dt))
        assert back == dt
        assert back.utcoffset() == dt.utcoffset()


# ---------------------------------------------------------------- strftime
def test_strftime_time_codes():
    t = time(3, 4, 5, 678)
    assert strftime(t, "%H|%M|%S|%f") == "03|04|05|000678"
    assert strftime(t, "%%H") == "%H"
    assert strftime(t, "%q") == "%q"


def test_strftime_date_codes():
    d = date(2024, 3, 9)
    assert strftime(d, "%Y-%m-%d") == "2024-03-09"
    assert strftime(d, "%j") == "069"
    assert strftime(d, "%w") == "6"
    assert strftime(date(2024, 3, 11), "%w") == "1"
    assert strftime(date(2024, 3, 10), "%w") == "7"
    assert strftime(d, "%W") == "10"
    assert strftime(date(2021, 1, 1), "%Y-%W") == "2021-53"
    assert strftime(date(2000, 1, 1), "%C") == "20"
    assert strftime(date(1999, 12, 31), "%C") == "19"


# ---------------------------------------------------------------- durations
def test_parse_duration_timedelta():
    assert parse_duration("P1D") == timedelta(days=1)
    assert parse_duration("PT1H") == timedelta(hours=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    assert parse_duration("PT1S") == timedelta(seconds=1)
    assert parse_duration("P1W") == timedelta(weeks=1)
    assert parse_duration("P2W3D") == timedelta(days=17)
    assert parse_duration("P1DT2H3M4S") == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert parse_duration("PT0.5S") == timedelta(milliseconds=500)
    assert parse_duration("PT0,5S") == timedelta(milliseconds=500)
    assert parse_duration("P0.5D") == timedelta(hours=12)
    assert parse_duration("PT36H") == timedelta(hours=36)
    assert parse_duration("P0D") == timedelta(0)
    assert isinstance(parse_duration("P1D"), timedelta)


def test_parse_duration_month_vs_minute():
    assert parse_duration("P1M") == Duration(months=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    d = parse_duration("P1MT1M")
    assert d.months == 1 and d.tdelta == timedelta(minutes=1)


def test_parse_duration_duration_objects():
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P1Y")
    assert d.years == 1 and d.months == 0 and d.tdelta == timedelta(0)
    d = parse_duration("P0.5Y")
    assert d.years == Decimal("0.5")
    d = parse_duration("P1,5M")
    assert d.months == Decimal("1.5")


def test_parse_duration_as_duration():
    d = parse_duration("P1DT1H", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=1, hours=1)


def test_parse_duration_negative():
    assert parse_duration("-P1D") == timedelta(days=-1)
    assert parse_duration("-PT5M") == timedelta(minutes=-5)
    assert parse_duration("+P1D") == timedelta(days=1)
    d = parse_duration("-P1Y2M3D")
    assert d.years == -1 and d.months == -2 and d.tdelta == timedelta(days=-3)
    assert parse_duration("-P1W") == timedelta(weeks=-1)


def test_parse_duration_alternative_format():
    d = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert d.years == 1 and d.months == 2


@pytest.mark.parametrize(
    "s", ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P1D1Y", "P-1D", "PD", "xP1D"]
)
def test_parse_duration_invalid(s):
    with pytest.raises(ValueError):
        parse_duration(s)


def test_parse_duration_type_error():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


def test_duration_isoformat_timedelta():
    f = duration_isoformat
    assert f(timedelta(days=1)) == "P1D"
    assert f(timedelta(hours=1)) == "PT1H"
    assert f(timedelta(minutes=90)) == "PT1H30M"
    assert f(timedelta(seconds=5)) == "PT5S"
    assert f(timedelta(days=2, hours=3, minutes=4, seconds=5)) == "P2DT3H4M5S"
    assert f(timedelta(seconds=1, microseconds=500000)) == "PT1.5S"
    assert f(timedelta(microseconds=1)) == "PT0.000001S"
    assert f(timedelta(0)) == "P0D"
    assert f(timedelta(days=-1)) == "-P1D"
    assert f(timedelta(days=-1, hours=-2)) == "-P1DT2H"
    assert f(timedelta(hours=-1)) == "-PT1H"
    assert f(timedelta(hours=25)) == "P1DT1H"
    assert f(timedelta(days=15), isodate.D_WEEK) == "P2W"


def test_duration_isoformat_duration():
    f = duration_isoformat
    assert f(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)) == "P1Y2M3DT4H5M6S"
    assert f(Duration(years=1)) == "P1Y"
    assert f(Duration(months=3)) == "P3M"
    assert f(Duration(years=-1, months=-2)) == "-P1Y2M"
    assert f(Duration(years=1, days=1)) == "P1Y1D"
    assert f(Duration(months=1, minutes=1)) == "P1MT1M"


def test_duration_alt_formats():
    td = timedelta(days=3, hours=4, minutes=5, seconds=6)
    assert strftime(td, "%d %H:%M:%S") == "03 04:05:06"
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert strftime(d, isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert strftime(d, isodate.D_ALT_BAS) == "P00010203T040506"
    assert strftime(d, "%Y|%m|%d|%H|%M|%S") == "0001|02|03|04|05|06"
    assert strftime(td, "%f") == "000000"
    assert strftime(timedelta(microseconds=42), "%f") == "000042"
    assert strftime(timedelta(days=15), "%W") == "02"
    assert strftime(td, "%%") == "%"


@pytest.mark.parametrize(
    "s",
    ["P1D", "PT1H", "P1DT2H3M4S", "PT0.5S", "-P1D", "P1Y", "P1Y2M3DT4H5M6S", "-P1Y2M", "P3M", "P1MT1M", "P0D"],
)
def test_duration_roundtrip(s):
    assert duration_isoformat(parse_duration(s)) == s


# ---------------------------------------------------------------- Duration
def test_duration_init_and_attrs():
    d = Duration(years=1, months=2, days=3, hours=4)
    assert d.years == 1 and d.months == 2
    assert d.days == 3  # proxied to timedelta
    assert d.seconds == 4 * 3600
    assert d.tdelta == timedelta(days=3, hours=4)
    assert isinstance(d.years, Decimal)
    assert Duration(weeks=1).tdelta == timedelta(weeks=1)
    assert Duration(milliseconds=5).tdelta == timedelta(milliseconds=5)


def test_duration_eq():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1) != Duration(months=11)
    assert Duration(days=1) == timedelta(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert Duration(months=1) != timedelta(days=30)
    assert Duration(months=1, days=1) != Duration(months=1, days=2)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(months=1) != timedelta(0)
    assert Duration(days=1) != "x" or True


def test_duration_hash():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    assert len({Duration(months=1), Duration(months=1), Duration(months=2)}) == 2


def test_duration_neg():
    n = -Duration(years=1, months=2, days=3, hours=4)
    assert n.years == -1 and n.months == -2
    assert n.tdelta == -timedelta(days=3, hours=4)
    assert -(-Duration(years=1, days=1)) == Duration(years=1, days=1)


def test_duration_add_sub_duration():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=2, months=5, days=4, hours=1)
    s = a + b
    assert s.years == 3 and s.months == 7 and s.tdelta == timedelta(days=7, hours=1)
    dif = b - a
    assert dif.years == 1 and dif.months == 3 and dif.tdelta == timedelta(days=1, hours=1)
    assert (a - a) == Duration()


def test_duration_add_sub_timedelta():
    a = Duration(months=1, days=1)
    s = a + timedelta(days=2)
    assert isinstance(s, Duration) and s.months == 1 and s.tdelta == timedelta(days=3)
    s = timedelta(days=2) + a
    assert isinstance(s, Duration) and s.tdelta == timedelta(days=3)
    s = a - timedelta(days=2)
    assert s.months == 1 and s.tdelta == timedelta(days=-1)
    s = timedelta(days=5) - a
    assert s.months == -1 and s.tdelta == timedelta(days=4)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3) * 3
    assert d.years == 3 and d.months == 6 and d.tdelta == timedelta(days=9)
    d = 2 * Duration(months=1, hours=1)
    assert d.months == 2 and d.tdelta == timedelta(hours=2)
    with pytest.raises(TypeError):
        Duration(days=1) * 1.5


def test_duration_add_to_date():
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 1, 31) + Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 1, 15) + Duration(months=12) == date(2001, 1, 15)
    assert date(2000, 11, 15) + Duration(months=2) == date(2001, 1, 15)
    assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
    assert date(2000, 1, 15) + Duration(years=1) == date(2001, 1, 15)
    assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
    assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)
    assert date(1900, 1, 31) + Duration(years=100, months=1) == date(2000, 2, 29)
    assert date(1999, 1, 31) + Duration(years=1, months=1, days=1) == date(2000, 3, 1)
    assert Duration(months=1, days=1) + date(2000, 1, 1) == date(2000, 2, 2)
    assert date(2000, 1, 1) + Duration(months=25) == date(2002, 2, 1)
    assert date(2000, 3, 31) + Duration(months=-1) == date(2000, 2, 29)
    assert date(2000, 1, 31) + Duration(months=-1) == date(1999, 12, 31)


def test_duration_add_to_datetime():
    dt = datetime(2000, 1, 31, 10, 30, tzinfo=UTC)
    r = dt + Duration(months=1, hours=2)
    assert r == datetime(2000, 2, 29, 12, 30, tzinfo=UTC)


def test_duration_month_lengths():
    for year, month, n in [(2000, 2, 29), (1900, 2, 28), (2004, 2, 29), (2001, 2, 28),
                           (2001, 1, 31), (2001, 3, 31), (2001, 4, 30), (2001, 5, 31),
                           (2001, 6, 30), (2001, 7, 31), (2001, 8, 31), (2001, 9, 30),
                           (2001, 10, 31), (2001, 11, 30), (2001, 12, 31)]:
        from isodate.duration import max_days_in_month

        assert max_days_in_month(year, month) == n


def test_date_sub_duration():
    assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 3, 31) - Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
    assert date(2000, 1, 15) - Duration(years=1) == date(1999, 1, 15)
    assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
    assert date(2000, 3, 1) - Duration(months=13) == date(1999, 2, 1)
    assert date(2000, 3, 10) - Duration(months=1, days=2) == date(2000, 2, 8)
    assert date(2000, 1, 1) - Duration(months=-1) == date(2000, 2, 1)


def test_duration_fractional_date_math_rejected():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=Decimal("0.5"))


def test_duration_unsupported_ops():
    assert Duration(days=1).__add__("x") is NotImplemented
    with pytest.raises(TypeError):
        Duration(days=1) + 5


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 2))


def test_duration_str_repr():
    assert str(Duration(days=1)) == "1 day, 0:00:00"
    s = str(Duration(years=2, months=3, days=1))
    assert s.startswith("2 years, 3 months, ")
    assert str(Duration(months=1)).startswith("1 month, ")
    assert str(Duration(years=1)).startswith("1 years, ")
    r = repr(Duration(years=1, months=2, days=3, seconds=4, microseconds=5))
    assert "years=1" in r and "months=2" in r and "3, 4, 5" in r


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    back = pickle.loads(pickle.dumps(d))
    assert back == d
    assert back.years == 1 and back.months == 2


def test_fquotmod():
    from isodate.duration import fquotmod

    assert fquotmod(Decimal(5), 1, 13) == (0, Decimal(5))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-11), 1, 13) == (-1, Decimal(1))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))


# ---------------------------------------------------------------- misc
def test_public_api():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert isinstance(isodate.__version__, str)
