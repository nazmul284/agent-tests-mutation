"""Behavioral tests for isodate, written against this implementation."""

import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate.duration import Duration, fquotmod, max_days_in_month
from isodate.isoerror import ISO8601Error
from isodate.isodates import build_date_regexps
from isodate.isotime import build_time_regexps
from isodate.isotzinfo import build_tzinfo
from isodate.tzinfo import FixedOffset, UTC, Utc, ZERO


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-02-29", date(2024, 2, 29)),
        ("19991231", date(1999, 12, 31)),
        ("2020-060", date(2020, 2, 29)),
        ("2019365", date(2019, 12, 31)),
        ("2020-W01-1", date(2019, 12, 30)),
        ("2020W537", date(2021, 1, 3)),
        ("2021-W01", date(2021, 1, 4)),
        ("2024-07", date(2024, 7, 1)),
        ("2024", date(2024, 1, 1)),
        ("19", date(1901, 1, 1)),
    ],
)
def test_parse_date_supported_forms(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_expanded_and_failures():
    assert isodate.parse_date("2024", defaultmonth=6, defaultday=15) == date(2024, 6, 15)
    assert isodate.parse_date("20", defaultmonth=12, defaultday=31) == date(2001, 12, 31)
    assert isodate.parse_date("+002024-03-04", yeardigits=6) == date(2024, 3, 4)
    assert isodate.parse_date("+2024-03-04", expanded=True) == date(2024, 3, 4)
    with pytest.raises(ISO8601Error):
        isodate.parse_date("2024/03/04")
    with pytest.raises(ValueError):
        isodate.parse_date("2023-02-29")
    with pytest.raises(ValueError):
        isodate.parse_date("-000001-01-01", yeardigits=6)


def test_date_regex_cache_and_formatting():
    assert build_date_regexps() is build_date_regexps()
    assert build_date_regexps(6, False) is build_date_regexps(6, True)
    d = date(2020, 12, 31)
    assert isodate.date_isoformat(d) == "2020-12-31"
    assert isodate.date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "2020-366"
    assert isodate.date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "2020-W53-4"
    assert isodate.date_isoformat(d, yeardigits=6) == "+002020-12-31"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("23:59:58", time(23, 59, 58)),
        ("T235958", time(23, 59, 58)),
        ("12:30.5", time(12, 30, 30)),
        ("1230,25", time(12, 30, 15)),
        ("12.5", time(12, 30)),
        ("01,010203", time(1, 0, 36, 730800)),
        ("10:20:30.1234569", time(10, 20, 30, 123456)),
    ],
)
def test_parse_time_forms_and_fractions(text, expected):
    assert isodate.parse_time(text) == expected


def test_parse_time_timezones_errors_and_cache():
    assert build_time_regexps() is build_time_regexps()
    utc = isodate.parse_time("12:00:00Z")
    assert utc.tzinfo is UTC and utc.utcoffset() == ZERO
    east = isodate.parse_time("12:00+05:30")
    west = isodate.parse_time("1200-0330")
    assert east.utcoffset() == timedelta(hours=5, minutes=30)
    assert west.utcoffset() == -timedelta(hours=3, minutes=30)
    assert isodate.time_isoformat(time(1, 2, 3, tzinfo=east.tzinfo)) == "01:02:03+05:30"
    with pytest.raises(ISO8601Error):
        isodate.parse_time("1:02")
    with pytest.raises(ValueError):
        isodate.parse_time("24:00:00")


def test_datetime_parsing_and_formatting():
    value = isodate.parse_datetime("2020-02-29T23:45:01.5-02:30")
    assert value == datetime(2020, 2, 29, 23, 45, 1, 500000, value.tzinfo)
    assert value.utcoffset() == -timedelta(hours=2, minutes=30)
    assert isodate.datetime_isoformat(value) == "2020-02-29T23:45:01-02:30"
    with pytest.raises(ISO8601Error, match="designator 'T' missing"):
        isodate.parse_datetime("2020-01-01 12:00:00")
    with pytest.raises(ISO8601Error):
        isodate.parse_datetime("2020-01-01T12:00:00Textra")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0D", timedelta(0)),
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.25S", timedelta(days=3, hours=4, minutes=5, seconds=6.25)),
        ("PT0.5H", timedelta(minutes=30)),
        ("P0,5D", timedelta(hours=12)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
    ],
)
def test_parse_duration_timedelta_forms(text, expected):
    assert isodate.parse_duration(text) == expected


def test_parse_duration_calendar_components_and_options():
    value = isodate.parse_duration("P1.5Y2.25M3DT4H")
    assert isinstance(value, Duration)
    assert value.years == Decimal("1.5")
    assert value.months == Decimal("2.25")
    assert value.tdelta == timedelta(days=3, hours=4)
    neg = isodate.parse_duration("-P1Y2M3D")
    assert neg == Duration(years=-1, months=-2, days=-3)
    forced = isodate.parse_duration("P2D", as_timedelta_if_possible=False)
    assert isinstance(forced, Duration) and forced == Duration(days=2)
    with pytest.raises(TypeError):
        isodate.parse_duration(3)  # type: ignore[arg-type]
    assert isodate.parse_duration("PT") == timedelta(0)
    for invalid in ("P", "hello", "P1Q"):
        with pytest.raises(ISO8601Error):
            isodate.parse_duration(invalid)


def test_duration_isoformat_components_precision_and_sign():
    assert isodate.duration_isoformat(timedelta(0)) == "P0D"
    assert isodate.duration_isoformat(timedelta(days=8, hours=2, minutes=3, seconds=4)) == "P8DT2H3M4S"
    assert isodate.duration_isoformat(timedelta(microseconds=120000)) == "PT0.12S"
    assert isodate.duration_isoformat(-timedelta(seconds=1)) == "-PT1S"
    assert isodate.duration_isoformat(Duration(years=2, months=3, days=4)) == "P2Y3M4D"
    assert isodate.duration_isoformat(Duration(months=-2)) == "-P2M"
    assert isodate.duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"


def test_duration_helpers_cover_month_and_leap_boundaries():
    assert fquotmod(Decimal("13"), 1, 13) == (1, Decimal("1"))
    assert fquotmod(Decimal("0"), 1, 13) == (-1, Decimal("12"))
    assert [max_days_in_month(2023, m) for m in range(1, 13)] == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert max_days_in_month(2000, 2) == 29
    assert max_days_in_month(1900, 2) == 28
    assert max_days_in_month(2024, 2) == 29


def test_duration_calendar_arithmetic_clamps_and_carries():
    assert date(2020, 1, 31) + Duration(months=1) == date(2020, 2, 29)
    assert date(2021, 1, 31) + Duration(months=1) == date(2021, 2, 28)
    assert date(2020, 12, 31) + Duration(months=2, days=1) == date(2021, 3, 1)
    assert date(2020, 3, 31) - Duration(months=1) == date(2020, 2, 29)
    dt = datetime(2020, 2, 29, 3, 4, 5)
    assert dt + Duration(years=1) == datetime(2021, 2, 28, 3, 4, 5)
    with pytest.raises(ValueError, match="fractional"):
        date(2020, 1, 1) + Duration(months=Decimal("0.5"))


def test_duration_operators_equality_and_conversion():
    a = Duration(years=1, months=2, days=3, seconds=4)
    b = Duration(months=14, days=3, seconds=4)
    assert a == b and not (a != b) and hash(a) == hash(Duration(years=1, months=2, days=3, seconds=4))
    assert a + Duration(months=1, days=2) == Duration(years=1, months=3, days=5, seconds=4)
    assert a - Duration(months=1, days=2) == Duration(years=1, months=1, days=1, seconds=4)
    assert -Duration(months=2, days=1) == Duration(months=-2, days=-1)
    assert Duration(days=2) * 3 == Duration(days=6)
    assert 2 * Duration(months=3) == Duration(months=6)
    assert Duration(days=1) + timedelta(hours=2) == Duration(days=1, hours=2)
    assert timedelta(days=3) - Duration(days=1) == Duration(days=2)
    assert Duration(days=2) == timedelta(days=2)
    assert Duration(months=1) != timedelta(days=30)
    start = date(2020, 1, 31)
    assert Duration(months=1).totimedelta(start=start) == timedelta(days=29)
    assert Duration(months=1).totimedelta(end=date(2020, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError, match="start or end"):
        Duration().totimedelta()
    with pytest.raises(ValueError, match="only start or end"):
        Duration().totimedelta(start=start, end=start)


def test_duration_string_repr_state_and_unsupported_operations():
    value = Duration(years=2, months=1, days=3, seconds=4)
    assert str(value) == "2 years, 1 month, 3 days, 0:00:04"
    assert "Duration(3, 4, 0, years=2, months=1)" in repr(value)
    assert value.days == 3 and value.seconds == 4
    assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(TypeError):
        _ = value * 1.5
    with pytest.raises(TypeError):
        _ = value + object()


def test_timezone_construction_parsing_and_formatting():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    assert isodate.parse_tzinfo("Z") is UTC
    east = isodate.parse_tzinfo("+05:45")
    west = isodate.parse_tzinfo("-0330")
    assert east.utcoffset(None) == timedelta(hours=5, minutes=45)
    assert east.tzname(None) == "+05:45" and east.dst(None) == ZERO
    assert west.utcoffset(None) == -timedelta(hours=3, minutes=30)
    aware = datetime(2020, 1, 1, tzinfo=east)
    assert isodate.tz_isoformat(aware, "%Z") == "+05:45"
    assert isodate.tz_isoformat(aware, "%z") == "+0545"
    assert isodate.tz_isoformat(aware, "%h") == "+05"
    assert isodate.tz_isoformat(datetime(2020, 1, 1)) == ""
    assert isodate.tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"
    # The timezone parser deliberately accepts a missing designator as local time.
    assert isodate.parse_tzinfo("") is None
    with pytest.raises(ValueError, match="unknown format"):
        isodate.tz_isoformat(aware, "%q")


def test_timezone_classes_repr_and_pickle_identity():
    assert Utc().utcoffset(None) == ZERO
    assert Utc().dst(None) == ZERO
    assert Utc().tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    fixed = FixedOffset(-2, -30, "X")
    assert repr(fixed) == "<FixedOffset 'X'>"
    assert fixed.utcoffset(None) == -timedelta(hours=2, minutes=30)


def test_strftime_directives_and_unknown_text():
    dt = datetime(2024, 2, 29, 5, 6, 7, 890, tzinfo=FixedOffset(-4, -30, "x"))
    assert isodate.strftime(dt, "%Y %C %m %d %j %W %w %H %M %S %f %% %q") == (
        "2024 20 02 29 060 09 4 05 06 07 000890 % %q"
    )
    td = timedelta(days=8, hours=2, minutes=3, seconds=4, microseconds=5)
    assert isodate.strftime(td, "%d %H %M %S %f %W %%") == "08 02 03 04 000005 01 %"
