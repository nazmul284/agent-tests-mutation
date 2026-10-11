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


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize(
    "s,expected",
    [
        ("2023-05-17", date(2023, 5, 17)),
        ("20230517", date(2023, 5, 17)),
        ("2024-02-29", date(2024, 2, 29)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        ("2023-05", date(2023, 5, 1)),
        ("202305", date(2023, 5, 1)),
        ("2023", date(2023, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("19", date(1901, 1, 1)),
    ],
)
def test_parse_date_calendar(s, expected):
    assert parse_date(s) == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("2023-001", date(2023, 1, 1)),
        ("2023001", date(2023, 1, 1)),
        ("2023-032", date(2023, 2, 1)),
        ("2023-365", date(2023, 12, 31)),
        ("2024-060", date(2024, 2, 29)),
        ("2024-366", date(2024, 12, 31)),
        ("2024366", date(2024, 12, 31)),
        ("2023-137", date(2023, 5, 17)),
    ],
)
def test_parse_date_ordinal(s, expected):
    assert parse_date(s) == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("2004-W53-6", date(2005, 1, 1)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2023-W20-3", date(2023, 5, 17)),
        ("2023W203", date(2023, 5, 17)),
        ("2023-W20", date(2023, 5, 15)),
        ("2023W20", date(2023, 5, 15)),
        ("2023-W01-1", date(2023, 1, 2)),
        ("2023-W01-7", date(2023, 1, 8)),
        ("2020-W01-1", date(2019, 12, 30)),
        ("2021-W01-1", date(2021, 1, 4)),
    ],
)
def test_parse_date_week(s, expected):
    assert parse_date(s) == expected
    # consistent with the stdlib
    assert expected.isocalendar()[:2] == (
        int(s[:4]),
        int(s.replace("-", "").replace("W", "")[4:6]),
    )


def test_parse_date_week_matches_stdlib_for_many_dates():
    d = date(2000, 1, 1)
    while d < date(2030, 1, 1):
        y, w, wd = d.isocalendar()
        assert parse_date("%04d-W%02d-%d" % (y, w, wd)) == d
        assert parse_date("%04dW%02d%d" % (y, w, wd)) == d
        d += timedelta(days=1)


def test_parse_date_ordinal_matches_stdlib_for_many_dates():
    d = date(1999, 1, 1)
    while d < date(2026, 1, 1):
        doy = d.timetuple().tm_yday
        assert parse_date("%04d-%03d" % (d.year, doy)) == d
        assert parse_date("%04d%03d" % (d.year, doy)) == d
        d += timedelta(days=1)


def test_parse_date_defaults():
    assert parse_date("2023", defaultmonth=6, defaultday=15) == date(2023, 6, 15)
    assert parse_date("2023-05", defaultday=20) == date(2023, 5, 20)
    assert parse_date("20", defaultmonth=3, defaultday=4) == date(2001, 3, 4)


def test_parse_date_expanded():
    assert parse_date("+002023-05-17", yeardigits=6) == date(2023, 5, 17)
    assert parse_date("+0020230517", yeardigits=6) == date(2023, 5, 17)
    assert parse_date("+2023-05-17", expanded=True) == date(2023, 5, 17)
    assert parse_date("+002023-137", yeardigits=6) == date(2023, 5, 17)
    assert parse_date("+0020", yeardigits=6) == date(2001, 1, 1)
    assert parse_date("+002023", yeardigits=6) == date(2023, 1, 1)
    assert parse_date("+002023-W20-3", yeardigits=6) == date(2023, 5, 17)


def test_parse_date_expanded_requires_sign():
    with pytest.raises(ISO8601Error):
        parse_date("002023-05-17", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("2023-05-17", expanded=True)


def test_parse_date_sign_not_allowed_by_default():
    with pytest.raises(ISO8601Error):
        parse_date("+2023-05-17")


def test_parse_date_negative_year_unrepresentable():
    with pytest.raises(ValueError):
        parse_date("-2023-05-17", expanded=True)


@pytest.mark.parametrize(
    "s",
    ["", "abc", "2023-5-17", "2023/05/17", "23-05-17", "2023-05-17 ", " 2023-05-17",
     "2023-05-17\n", "20230", "2023-W5", "12345", "2023-05-1", "2023-W20-3-1"],
)
def test_parse_date_invalid_format(s):
    with pytest.raises(ISO8601Error):
        parse_date(s)


@pytest.mark.parametrize("s", ["2023-13-01", "2023-02-30", "2023-04-31",
                               "2023-02-29", "2023-05-00"])
def test_parse_date_invalid_value(s):
    with pytest.raises(ValueError):
        parse_date(s)


def test_build_date_regexps_cached():
    from isodate.isodates import build_date_regexps

    assert build_date_regexps(4, False) is build_date_regexps(4, False)
    assert build_date_regexps(6) is build_date_regexps(6, True)
    assert build_date_regexps(4, True) is not build_date_regexps(4, False)


# ----------------------------------------------------------------- time
@pytest.mark.parametrize(
    "s,expected",
    [
        ("14:30:15", time(14, 30, 15)),
        ("143015", time(14, 30, 15)),
        ("T14:30:15", time(14, 30, 15)),
        ("T143015", time(14, 30, 15)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
        ("14:30:15.5", time(14, 30, 15, 500000)),
        ("14:30:15,5", time(14, 30, 15, 500000)),
        ("14:30:15.123456", time(14, 30, 15, 123456)),
        ("14:30:15.000001", time(14, 30, 15, 1)),
        ("143015.25", time(14, 30, 15, 250000)),
        ("14:30", time(14, 30)),
        ("1430", time(14, 30)),
        ("14:30.5", time(14, 30, 30)),
        ("1430,25", time(14, 30, 15)),
        ("14", time(14)),
        ("14.5", time(14, 30)),
        ("14,25", time(14, 15)),
        ("14.75", time(14, 45)),
    ],
)
def test_parse_time(s, expected):
    assert parse_time(s) == expected
    assert parse_time(s).tzinfo is None


def test_parse_time_truncates_microseconds():
    assert parse_time("14:30:15.1234569") == time(14, 30, 15, 123456)


def test_parse_time_timezones():
    t = parse_time("14:30:15Z")
    assert t.tzinfo is UTC
    assert t.utcoffset() == timedelta(0)
    t = parse_time("14:30:15+05:30")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    t = parse_time("14:30:15-0530")
    assert t.utcoffset() == -timedelta(hours=5, minutes=30)
    t = parse_time("14:30:15+02")
    assert t.utcoffset() == timedelta(hours=2)
    t = parse_time("143015-08")
    assert t.utcoffset() == -timedelta(hours=8)
    t = parse_time("14:30Z")
    assert t == time(14, 30, tzinfo=UTC)
    t = parse_time("14Z")
    assert t == time(14, tzinfo=UTC)
    t = parse_time("14:30:15.5+01:00")
    assert t.microsecond == 500000
    assert t.utcoffset() == timedelta(hours=1)


@pytest.mark.parametrize(
    "s",
    ["", "1", "14:", "14:30:", "14:3", "14:30:1", "abc", "14:30:15 ", " 14:30:15",
     "14:30:15Zx", "14:30:15+5", "14:30:15+0", "T", "14-30-15", "14:30:15.", "1:30:15"],
)
def test_parse_time_invalid_format(s):
    with pytest.raises(ISO8601Error):
        parse_time(s)


@pytest.mark.parametrize("s", ["24:00:00", "25:00:00", "14:60:00", "14:30:60", "99"])
def test_parse_time_invalid_value(s):
    with pytest.raises(ValueError):
        parse_time(s)


# ------------------------------------------------------------- tzinfo
def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    tz = parse_tzinfo("-0530")
    assert tz.utcoffset(None) == -timedelta(hours=5, minutes=30)
    tz = parse_tzinfo("+05")
    assert tz.utcoffset(None) == timedelta(hours=5)
    tz = parse_tzinfo("-11")
    assert tz.utcoffset(None) == timedelta(hours=-11)
    assert tz.tzname(None) == "-11"


def test_build_tzinfo():
    from isodate.isotzinfo import build_tzinfo

    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("-03:30", "-", 3, 30)
    assert tz.utcoffset(None) == -timedelta(hours=3, minutes=30)
    tz = build_tzinfo("+03:30", "+", 3, 30)
    assert tz.utcoffset(None) == timedelta(hours=3, minutes=30)


def test_tz_isoformat():
    dt = datetime(2023, 1, 1)
    assert tz_isoformat(dt) == ""
    assert tz_isoformat(dt.replace(tzinfo=UTC)) == "Z"
    assert tz_isoformat(dt.replace(tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    dt = datetime(2023, 1, 1, tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(dt) == "+05:30"
    assert tz_isoformat(dt, "%Z") == "+05:30"
    assert tz_isoformat(dt, "%z") == "+0530"
    assert tz_isoformat(dt, "%h") == "+05"
    dt = datetime(2023, 1, 1, tzinfo=FixedOffset(-5, -30, "x"))
    assert tz_isoformat(dt) == "-05:30"
    assert tz_isoformat(dt, "%z") == "-0530"
    assert tz_isoformat(dt, "%h") == "-05"
    with pytest.raises(ValueError):
        tz_isoformat(dt, "%q")


def test_fixed_offset_and_utc():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.dst(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    fo = FixedOffset(-5, -30, "EST?")
    assert fo.utcoffset(None) == timedelta(hours=-5, minutes=-30)
    assert fo.dst(None) == timedelta(0)
    assert fo.tzname(None) == "EST?"
    assert "EST?" in repr(fo)
    assert FixedOffset().utcoffset(None) == timedelta(0)
    assert FixedOffset().tzname(None) == "UTC"


def test_utc_pickle_returns_singleton():
    assert pickle.loads(pickle.dumps(UTC)) is UTC


def test_local_timezone_consistent():
    dt = datetime(2023, 1, 15, 12, 0, tzinfo=LOCAL)
    off = LOCAL.utcoffset(dt)
    assert isinstance(off, timedelta)
    assert LOCAL.dst(dt) in (timedelta(0), off - LOCAL.utcoffset(datetime(2023, 1, 15)))
    assert isinstance(LOCAL.tzname(dt), str)
    with pytest.raises(Exception):
        LOCAL.utcoffset(None)


# ------------------------------------------------------------ datetime
def test_parse_datetime():
    assert parse_datetime("2023-05-17T14:30:15") == datetime(2023, 5, 17, 14, 30, 15)
    assert parse_datetime("20230517T143015") == datetime(2023, 5, 17, 14, 30, 15)
    assert parse_datetime("2023-137T14:30") == datetime(2023, 5, 17, 14, 30)
    assert parse_datetime("2023-W20-3T14:30:15Z") == datetime(
        2023, 5, 17, 14, 30, 15, tzinfo=UTC
    )
    dt = parse_datetime("2023-05-17T14:30:15.25+05:30")
    assert dt.microsecond == 250000
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert parse_datetime("2023-05-17T14") == datetime(2023, 5, 17, 14)


@pytest.mark.parametrize("s", ["2023-05-17", "2023-05-17 14:30:15", "", "T14:30:15",
                               "2023-05-17T14:30:15T", "2023-05-17t14:30:15"])
def test_parse_datetime_invalid(s):
    with pytest.raises(ISO8601Error):
        parse_datetime(s)


def test_parse_datetime_trailing_t():
    with pytest.raises(ISO8601Error):
        parse_datetime("2023-05-17T")


# ------------------------------------------------------------ strftime
D = date(2023, 5, 17)
DT = datetime(2023, 5, 7, 4, 3, 2, 1234, tzinfo=FixedOffset(5, 30, "x"))


def test_date_formats():
    assert date_isoformat(D) == "2023-05-17"
    assert date_isoformat(D, isodate.DATE_BAS_COMPLETE) == "20230517"
    assert date_isoformat(D, isodate.DATE_EXT_ORD_COMPLETE) == "2023-137"
    assert date_isoformat(D, isodate.DATE_BAS_ORD_COMPLETE) == "2023137"
    assert date_isoformat(D, isodate.DATE_EXT_WEEK_COMPLETE) == "2023-W20-3"
    assert date_isoformat(D, isodate.DATE_BAS_WEEK_COMPLETE) == "2023W203"
    assert date_isoformat(D, isodate.DATE_EXT_WEEK) == "2023-W20"
    assert date_isoformat(D, isodate.DATE_BAS_WEEK) == "2023W20"
    assert date_isoformat(D, isodate.DATE_EXT_MONTH) == "2023-05"
    assert date_isoformat(D, isodate.DATE_BAS_MONTH) == "202305"
    assert date_isoformat(D, isodate.DATE_YEAR) == "2023"
    assert date_isoformat(D, isodate.DATE_CENTURY) == "20"
    assert date_isoformat(date(1, 1, 1)) == "0001-01-01"
    assert date_isoformat(date(2023, 1, 1), "%j") == "001"
    assert date_isoformat(date(2024, 12, 31), "%j") == "366"
    assert date_isoformat(date(2023, 12, 31), "%j") == "365"
    assert date_isoformat(date(2005, 1, 1), "%Y-W%W-%w") == "2005-W53-6"
    assert date_isoformat(date(2023, 5, 21), "%w") == "7"
    assert date_isoformat(date(2023, 5, 15), "%w") == "1"


def test_date_format_expanded_year():
    assert date_isoformat(D, "%Y", 6) == "+002023"
    assert date_isoformat(D, "%C", 6) == "+0020"
    assert date_isoformat(D, "%Y-%m-%d", 4) == "2023-05-17"


def test_date_format_roundtrip_all_formats():
    d = date(2024, 3, 9)
    for fmt in (
        isodate.DATE_BAS_COMPLETE,
        isodate.DATE_EXT_COMPLETE,
        isodate.DATE_BAS_ORD_COMPLETE,
        isodate.DATE_EXT_ORD_COMPLETE,
        isodate.DATE_BAS_WEEK_COMPLETE,
        isodate.DATE_EXT_WEEK_COMPLETE,
    ):
        assert parse_date(date_isoformat(d, fmt)) == d


def test_time_formats():
    t = time(4, 3, 2, 1234)
    assert time_isoformat(t) == "04:03:02"
    assert time_isoformat(t, isodate.TIME_BAS_COMPLETE) == "040302"
    assert time_isoformat(t, isodate.TIME_EXT_MINUTE) == "04:03"
    assert time_isoformat(t, isodate.TIME_BAS_MINUTE) == "0403"
    assert time_isoformat(t, isodate.TIME_HOUR) == "04"
    assert time_isoformat(t, "%H:%M:%S.%f") == "04:03:02.001234"
    assert time_isoformat(time(4, 3, 2, tzinfo=UTC)) == "04:03:02Z"
    assert time_isoformat(time(4, 3, 2, tzinfo=FixedOffset(-5, 0, "x"))) == "04:03:02-05:00"
    assert time_isoformat(time(23, 59, 59)) == "23:59:59"
    assert time_isoformat(time(0, 0, 0)) == "00:00:00"


def test_datetime_formats():
    assert datetime_isoformat(DT) == "2023-05-07T04:03:02+05:30"
    assert datetime_isoformat(datetime(2023, 5, 7, 4, 3, 2)) == "2023-05-07T04:03:02"
    assert datetime_isoformat(DT, isodate.DT_EXT_COMPLETE) == "2023-05-07T04:03:02+05:30"
    assert datetime_isoformat(DT, isodate.DT_BAS_COMPLETE) == "20230507T040302+0530"
    assert datetime_isoformat(DT, isodate.DT_EXT_ORD_COMPLETE) == "2023-127T04:03:02+05:30"
    assert datetime_isoformat(DT, isodate.DT_BAS_ORD_COMPLETE) == "2023127T040302+0530"
    assert datetime_isoformat(DT, isodate.DT_EXT_WEEK_COMPLETE) == "2023-W18-7T04:03:02+05:30"
    assert datetime_isoformat(DT, isodate.DT_BAS_WEEK_COMPLETE) == "2023W187T040302+0530"
    assert datetime_isoformat(DT.replace(tzinfo=UTC)) == "2023-05-07T04:03:02Z"
    assert datetime_isoformat(DT, "%h") == "+05"


def test_datetime_roundtrip():
    for dt in (
        datetime(2023, 5, 7, 4, 3, 2, tzinfo=UTC),
        datetime(2023, 5, 7, 4, 3, 2, tzinfo=FixedOffset(-3, -30, "x")),
        datetime(1999, 12, 31, 23, 59, 59),
    ):
        assert parse_datetime(datetime_isoformat(dt)) == dt
        assert parse_datetime(datetime_isoformat(dt)).utcoffset() == dt.utcoffset()
    dt = datetime(2023, 5, 7, 4, 3, 2, tzinfo=FixedOffset(2, 0, "x"))
    assert parse_datetime(datetime_isoformat(dt, isodate.DT_BAS_COMPLETE)) == dt
    assert parse_datetime(datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE)) == dt
    assert parse_datetime(datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE)) == dt


def test_strftime_percent_and_unknown():
    assert strftime(D, "100%%") == "100%"
    assert strftime(D, "%q") == "%q"
    assert strftime(D, "plain") == "plain"
    assert strftime(timedelta(days=1), "%%") == "%"


def test_strftime_timedelta_directives():
    td = timedelta(days=15, hours=3, minutes=7, seconds=9, microseconds=42)
    assert strftime(td, "%d") == "15"
    assert strftime(td, "%H") == "03"
    assert strftime(td, "%M") == "07"
    assert strftime(td, "%S") == "09"
    assert strftime(td, "%f") == "000042"
    assert strftime(Duration(years=3, months=4), "%Y-%m") == "0003-04"
    assert strftime(Duration(years=3, months=4), "%Y", 6) == "+000003"
    assert strftime(Duration(years=300), "%C") == "03"


# ------------------------------------------------------------ duration
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
    assert parse_duration("P1W1D") == timedelta(days=8)
    assert isinstance(parse_duration("P1D"), timedelta)


def test_parse_duration_negative_and_sign():
    assert parse_duration("-P1D") == timedelta(days=-1)
    assert parse_duration("-PT1H30M") == -timedelta(hours=1, minutes=30)
    assert parse_duration("+P1D") == timedelta(days=1)
    d = parse_duration("-P1Y2M")
    assert d == Duration(years=-1, months=-2)
    assert parse_duration("-P1Y2M3DT4H") == Duration(years=-1, months=-2, days=-3, hours=-4)


def test_parse_duration_months_vs_minutes():
    assert parse_duration("PT5M") == timedelta(minutes=5)
    d = parse_duration("P5M")
    assert isinstance(d, Duration)
    assert d.months == 5 and d.tdelta == timedelta(0)
    d = parse_duration("P1M1DT1M")
    assert d.months == 1 and d.tdelta == timedelta(days=1, minutes=1)


def test_parse_duration_duration():
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P1Y")
    assert d.years == 1 and d.months == 0
    d = parse_duration("P1.5Y")
    assert d.years == Decimal("1.5")
    d = parse_duration("P0.5M")
    assert d.months == Decimal("0.5")
    d = parse_duration("P2Y1W")
    assert d.years == 2 and d.tdelta == timedelta(weeks=1)


def test_parse_duration_as_timedelta_if_possible_false():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d == timedelta(days=1)
    d = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=-1)


def test_parse_duration_alternative_format():
    d = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize(
    "s",
    ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P1Y1Y", "P1D1Y", "PTM", "P-1D",
     "xP1D", "P1D ", "garbage", "P1H1D"],
)
def test_parse_duration_invalid(s):
    with pytest.raises(ISO8601Error):
        parse_duration(s)


def test_parse_duration_type_error():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)


def test_duration_isoformat_timedelta():
    assert duration_isoformat(timedelta(days=1)) == "P1D"
    assert duration_isoformat(timedelta(0)) == "P0D"
    assert duration_isoformat(timedelta(hours=1)) == "PT1H"
    assert duration_isoformat(timedelta(minutes=1)) == "PT1M"
    assert duration_isoformat(timedelta(seconds=1)) == "PT1S"
    assert duration_isoformat(timedelta(days=1, hours=2, minutes=3, seconds=4)) == "P1DT2H3M4S"
    assert duration_isoformat(timedelta(days=40)) == "P40D"
    assert duration_isoformat(timedelta(hours=25)) == "P1DT1H"
    assert duration_isoformat(timedelta(seconds=90)) == "PT1M30S"
    assert duration_isoformat(timedelta(microseconds=500000)) == "PT0.5S"
    assert duration_isoformat(timedelta(seconds=1, microseconds=10)) == "PT1.00001S"
    assert duration_isoformat(timedelta(microseconds=1)) == "PT0.000001S"
    assert duration_isoformat(timedelta(minutes=2, microseconds=1)) == "PT2M0.000001S"
    assert duration_isoformat(timedelta(days=1, seconds=1)) == "P1DT1S"
    assert duration_isoformat(timedelta(hours=1, seconds=1)) == "PT1H1S"


def test_duration_isoformat_negative():
    assert duration_isoformat(timedelta(days=-1)) == "-P1D"
    assert duration_isoformat(-timedelta(hours=1, minutes=30)) == "-PT1H30M"
    assert duration_isoformat(timedelta(seconds=-1, microseconds=-500000)) == "-PT1.5S"
    assert duration_isoformat(Duration(years=-1)) == "-P1Y"
    assert duration_isoformat(Duration(months=-2)) == "-P2M"
    assert duration_isoformat(Duration(days=-2)) == "-P2D"
    assert duration_isoformat(Duration(years=1, months=2)) == "P1Y2M"


def test_duration_isoformat_duration():
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d) == "P1Y2M3DT4H5M6S"
    assert duration_isoformat(Duration(years=1)) == "P1Y"
    assert duration_isoformat(Duration(months=3)) == "P3M"
    assert duration_isoformat(Duration()) == "P0D"
    assert duration_isoformat(Duration(years=1, minutes=1)) == "P1YT1M"
    assert duration_isoformat(Duration(years=Decimal("1.5"))) == "P1.5Y"


def test_duration_isoformat_weeks():
    assert duration_isoformat(timedelta(weeks=3), isodate.D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(days=6), isodate.D_WEEK) == "P0W"


def test_duration_isoformat_alternative():
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
                              isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
                              isodate.D_ALT_BAS) == "P00010203T040506"


def test_duration_roundtrip():
    for s in ["P1D", "PT1H", "P1DT2H3M4S", "P1Y2M3DT4H5M6S", "P1Y", "P3M", "PT0.25S",
              "-P1D", "-P1Y2M", "P0D", "PT1M30S", "P10W"]:
        d = parse_duration(s)
        if s == "P10W":
            assert d == timedelta(weeks=10)
            continue
        assert parse_duration(duration_isoformat(d)) == d
    td = timedelta(days=3, hours=5, seconds=7, microseconds=123456)
    assert parse_duration(duration_isoformat(td)) == td


# ------------------------------------------------------ Duration class
def test_duration_init_and_str_repr():
    d = Duration(days=1, hours=2, years=3, months=4)
    assert d.years == 3 and d.months == 4
    assert d.tdelta == timedelta(days=1, hours=2)
    assert isinstance(d.years, Decimal)
    assert d.days == 1  # delegated to timedelta
    assert d.seconds == 7200
    assert str(d) == "3 years, 4 months, 1 day, 2:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(years=1)) == "1 years, 0:00:00"
    assert str(Duration(days=2)) == "2 days, 0:00:00"
    r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
    assert r.endswith("Duration(1, 2, 3, years=4, months=5)")
    with pytest.raises(AttributeError):
        Duration().nonexistent


def test_duration_equality():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, days=1) == Duration(months=12, days=1)
    assert Duration(years=1) != Duration(years=1, days=1)
    assert Duration(years=1) != Duration(months=11)
    assert not (Duration(years=1) != Duration(months=12))
    assert not (Duration(years=1) == Duration(years=2))
    assert Duration(days=1) == timedelta(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(years=1) != timedelta(days=365)
    assert not (Duration(years=1) == timedelta(days=365))
    assert Duration(months=1) != timedelta(0)
    assert Duration(days=1, years=1) != Duration(days=2, years=1)


def test_duration_hash():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2


def test_duration_neg():
    n = -Duration(years=1, months=2, days=3, hours=4)
    assert n.years == -1 and n.months == -2
    assert n.tdelta == -timedelta(days=3, hours=4)
    assert -Duration() == Duration()


def test_duration_add_duration_and_timedelta():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=2, months=3, days=4)
    s = a + b
    assert (s.years, s.months, s.tdelta) == (3, 5, timedelta(days=7))
    s = a + timedelta(days=1)
    assert isinstance(s, Duration)
    assert (s.years, s.months, s.tdelta) == (1, 2, timedelta(days=4))
    s = timedelta(days=1) + a
    assert isinstance(s, Duration)
    assert s.tdelta == timedelta(days=4)
    with pytest.raises(TypeError):
        a + 5
    with pytest.raises(TypeError):
        a + "x"


def test_duration_add_to_date():
    assert date(2023, 1, 31) + Duration(months=1) == date(2023, 2, 28)
    assert date(2024, 1, 31) + Duration(months=1) == date(2024, 2, 29)
    assert Duration(months=1) + date(2023, 1, 31) == date(2023, 2, 28)
    assert date(2023, 5, 17) + Duration(years=1) == date(2024, 5, 17)
    assert date(2024, 2, 29) + Duration(years=1) == date(2025, 2, 28)
    assert date(2024, 2, 29) + Duration(years=4) == date(2028, 2, 29)
    assert date(2023, 11, 15) + Duration(months=2) == date(2024, 1, 15)
    assert date(2023, 12, 15) + Duration(months=1) == date(2024, 1, 15)
    assert date(2023, 1, 15) + Duration(months=12) == date(2024, 1, 15)
    assert date(2023, 1, 15) + Duration(months=11) == date(2023, 12, 15)
    assert date(2023, 1, 15) + Duration(months=24) == date(2025, 1, 15)
    assert date(2023, 1, 15) + Duration(months=-1) == date(2022, 12, 15)
    assert date(2023, 1, 15) + Duration(months=-13) == date(2021, 12, 15)
    assert date(2023, 3, 31) + Duration(months=-1) == date(2023, 2, 28)
    assert date(2023, 5, 17) + Duration(months=1, days=20) == date(2023, 7, 7)
    assert date(1900, 1, 31) + Duration(months=1) == date(1900, 2, 28)
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert date(2023, 5, 31) + Duration(months=1) == date(2023, 6, 30)
    assert date(2023, 5, 17) + Duration() == date(2023, 5, 17)


def test_duration_add_to_datetime():
    dt = datetime(2023, 1, 31, 10, 20, 30, tzinfo=UTC)
    r = dt + Duration(months=1, hours=2)
    assert r == datetime(2023, 2, 28, 12, 20, 30, tzinfo=UTC)
    assert r.tzinfo is UTC


def test_duration_fractional_date_math_rejected():
    with pytest.raises(ValueError):
        date(2023, 1, 1) + Duration(months=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2023, 1, 1) + Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2023, 1, 1) - Duration(years=Decimal("0.5"))
    assert date(2023, 1, 1) + Duration(years=Decimal("1.0")) == date(2024, 1, 1)


def test_duration_sub():
    a = Duration(years=3, months=5, days=7)
    b = Duration(years=1, months=2, days=3)
    s = a - b
    assert (s.years, s.months, s.tdelta) == (2, 3, timedelta(days=4))
    s = a - timedelta(days=2)
    assert isinstance(s, Duration)
    assert (s.years, s.months, s.tdelta) == (3, 5, timedelta(days=5))
    s = timedelta(days=10) - Duration(days=3)
    assert isinstance(s, Duration)
    assert s.tdelta == timedelta(days=7)
    with pytest.raises(TypeError):
        a - 5


def test_duration_rsub_date():
    assert date(2023, 3, 31) - Duration(months=1) == date(2023, 2, 28)
    assert date(2023, 1, 15) - Duration(months=1) == date(2022, 12, 15)
    assert date(2023, 1, 15) - Duration(months=13) == date(2021, 12, 15)
    assert date(2023, 5, 17) - Duration(years=1) == date(2022, 5, 17)
    assert date(2024, 2, 29) - Duration(years=1) == date(2023, 2, 28)
    assert date(2023, 5, 17) - Duration(months=1, days=7) == date(2023, 4, 10)
    assert date(2023, 5, 17) - Duration(months=-1) == date(2023, 6, 17)
    dt = datetime(2023, 3, 31, 12, tzinfo=UTC)
    assert dt - Duration(months=1, hours=2) == datetime(2023, 2, 28, 10, tzinfo=UTC)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3, hours=1) * 3
    assert (d.years, d.months, d.tdelta) == (3, 6, timedelta(days=9, hours=3))
    d = 2 * Duration(years=1, months=2, days=3)
    assert (d.years, d.months, d.tdelta) == (2, 4, timedelta(days=6))
    assert Duration(years=1) * 0 == Duration()
    with pytest.raises(TypeError):
        Duration(years=1) * 1.5
    with pytest.raises(TypeError):
        Duration(years=1) * "a"


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2023, 1, 1)) == timedelta(days=31)
    assert d.totimedelta(start=date(2023, 2, 1)) == timedelta(days=28)
    assert d.totimedelta(start=date(2024, 2, 1)) == timedelta(days=29)
    assert d.totimedelta(end=date(2023, 3, 1)) == timedelta(days=28)
    assert d.totimedelta(end=date(2023, 4, 1)) == timedelta(days=31)
    assert Duration(years=1).totimedelta(start=date(2024, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(start=date(2023, 1, 1)) == timedelta(days=365)
    assert Duration(years=1, days=1).totimedelta(start=date(2023, 1, 1)) == timedelta(days=366)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2023, 1, 1), end=date(2023, 2, 1))


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d
    assert (d2.years, d2.months, d2.tdelta) == (d.years, d.months, d.tdelta)


def test_helpers():
    assert max_days_in_month(2023, 1) == 31
    assert max_days_in_month(2023, 3) == 31
    assert max_days_in_month(2023, 5) == 31
    assert max_days_in_month(2023, 7) == 31
    assert max_days_in_month(2023, 8) == 31
    assert max_days_in_month(2023, 10) == 31
    assert max_days_in_month(2023, 12) == 31
    assert max_days_in_month(2023, 4) == 30
    assert max_days_in_month(2023, 6) == 30
    assert max_days_in_month(2023, 9) == 30
    assert max_days_in_month(2023, 11) == 30
    assert max_days_in_month(2023, 2) == 28
    assert max_days_in_month(2024, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(2100, 2) == 28

    assert fquotmod(Decimal(5), 1, 13) == (0, Decimal(5))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-1), 1, 13) == (-1, Decimal(11))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))


def test_public_api():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert isinstance(isodate.__version__, str)
    assert issubclass(ISO8601Error, ValueError)
