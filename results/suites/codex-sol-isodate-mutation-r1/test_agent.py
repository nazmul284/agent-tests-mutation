import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate.duration import Duration, fquotmod, max_days_in_month
from isodate.isoerror import ISO8601Error
from isodate.isostrf import _strfduration
from isodate.isotzinfo import build_tzinfo
from isodate.tzinfo import FixedOffset, UTC, Utc


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-02-29", date(2024, 2, 29)),
        ("20240229", date(2024, 2, 29)),
        ("2024-060", date(2024, 2, 29)),
        ("2024060", date(2024, 2, 29)),
        ("2020-W53-7", date(2021, 1, 3)),
        ("2020W537", date(2021, 1, 3)),
        ("2021-W01-1", date(2021, 1, 4)),
        ("2021W011", date(2021, 1, 4)),
        ("2020-W53", date(2020, 12, 28)),
        ("2020W53", date(2020, 12, 28)),
        ("1985-04", date(1985, 4, 1)),
        ("198504", date(1985, 4, 1)),
        ("1985", date(1985, 1, 1)),
        ("19", date(1901, 1, 1)),
    ],
)
def test_parse_date_supported_forms(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_and_expanded_years():
    assert isodate.parse_date("2000", defaultmonth=7, defaultday=9) == date(2000, 7, 9)
    assert isodate.parse_date("20", defaultmonth=2, defaultday=3) == date(2001, 2, 3)
    assert isodate.parse_date("2000-00", defaultmonth=6, defaultday=4) == date(2000, 6, 4)
    assert isodate.parse_date("+002024-03-09", yeardigits=6) == date(2024, 3, 9)
    assert isodate.parse_date("+2024-03-09", expanded=True) == date(2024, 3, 9)


@pytest.mark.parametrize("text", ["", "2024/01/02", "2024-1-02", "2024-01-2", "abcd", "202401020"])
def test_parse_date_rejects_bad_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_date(text)


@pytest.mark.parametrize("text", ["2023-02-29", "2024-13-01", "2024-01-32", "0000-01-01"])
def test_parse_date_rejects_unrepresentable_values(text):
    with pytest.raises(ValueError):
        isodate.parse_date(text)


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (date(2024, 2, 29), isodate.DATE_EXT_COMPLETE, "2024-02-29"),
        (date(2024, 2, 29), isodate.DATE_BAS_COMPLETE, "20240229"),
        (date(2024, 2, 29), isodate.DATE_EXT_ORD_COMPLETE, "2024-060"),
        (date(2021, 1, 3), isodate.DATE_EXT_WEEK_COMPLETE, "2021-W53-7"),
        (date(2024, 2, 29), "%C %%", "20 %"),
    ],
)
def test_date_format_directives(value, fmt, expected):
    assert isodate.date_isoformat(value, fmt) == expected


def test_date_format_expanded_year():
    assert isodate.date_isoformat(date(42, 3, 5), yeardigits=6) == "+000042-03-05"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("23:14:05", time(23, 14, 5)),
        ("T231405", time(23, 14, 5)),
        ("12:30.5", time(12, 30, 30)),
        ("1230,25", time(12, 30, 15)),
        ("12.5", time(12, 30)),
        ("01,010101", time(1, 0, 36, 363600)),
        ("10:20:30.1234569", time(10, 20, 30, 123456)),
    ],
)
def test_parse_time_forms_and_fractions(text, expected):
    assert isodate.parse_time(text) == expected


def test_parse_time_zones():
    zulu = isodate.parse_time("10:20:30Z")
    assert zulu.tzinfo is UTC
    assert zulu.utcoffset() == timedelta(0)
    assert isodate.parse_time("10:20:30+05:45").utcoffset() == timedelta(hours=5, minutes=45)
    assert isodate.parse_time("10:20:30-0330").utcoffset() == -timedelta(hours=3, minutes=30)
    assert isodate.parse_time("10:20:30+02").utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize("text", ["", "1:02", "12:1", "12:00:00junk", "T"])
def test_parse_time_rejects_bad_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_time(text)


@pytest.mark.parametrize("text", ["24:00:00", "23:60:00", "23:59:60"])
def test_parse_time_rejects_bad_values(text):
    with pytest.raises(ValueError):
        isodate.parse_time(text)


def test_time_and_datetime_format_and_parse():
    aware = datetime(2024, 2, 29, 3, 4, 5, tzinfo=FixedOffset(-3, -30, "-03:30"))
    assert isodate.datetime_isoformat(aware) == "2024-02-29T03:04:05-03:30"
    assert isodate.datetime_isoformat(aware, isodate.DT_BAS_COMPLETE) == "20240229T030405-0330"
    parsed = isodate.parse_datetime("2024-060T03:04:05.25+02:30")
    assert parsed.replace(tzinfo=None) == datetime(2024, 2, 29, 3, 4, 5, 250000)
    assert parsed.utcoffset() == timedelta(hours=2, minutes=30)
    assert isodate.time_isoformat(time(3, 4, 5, tzinfo=UTC)) == "03:04:05Z"
    assert isodate.time_isoformat(time(3, 4, 5)) == "03:04:05"


@pytest.mark.parametrize("text", ["2024-01-01 12:00:00", "2024-01-01", "2024-01-01T12:00T00"])
def test_datetime_requires_exactly_one_designator(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_datetime(text)


def test_timezone_parse_build_and_format():
    assert isodate.parse_tzinfo("") is None
    assert isodate.parse_tzinfo("Z") is UTC
    zone = isodate.parse_tzinfo("-09:30")
    dt = datetime(2020, 1, 1, tzinfo=zone)
    assert zone.utcoffset(dt) == -timedelta(hours=9, minutes=30)
    assert zone.dst(dt) == timedelta(0)
    assert zone.tzname(dt) == "-09:30"
    assert repr(zone) == "<FixedOffset '-09:30'>"
    assert isodate.tz_isoformat(dt) == "-09:30"
    assert isodate.tz_isoformat(dt, "%z") == "-0930"
    assert isodate.tz_isoformat(dt, "%h") == "-09"
    assert isodate.tz_isoformat(datetime(2020, 1, 1)) == ""
    assert isodate.tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"
    assert build_tzinfo(None) is None
    assert build_tzinfo("", "-", 3, 4) is None


def test_timezone_errors_and_utc_protocol():
    # The public parser deliberately accepts an empty prefix as local time.
    assert isodate.parse_tzinfo("hello") is None
    with pytest.raises(ValueError):
        isodate.tz_isoformat(datetime(2020, 1, 1, tzinfo=FixedOffset(1)), "%q")
    utc = Utc()
    assert utc.utcoffset(None) == utc.dst(None) == timedelta(0)
    assert utc.tzname(None) == "UTC"
    assert pickle.loads(pickle.dumps(utc)) is UTC


@pytest.mark.parametrize(
    "text, expected",
    [
        ("P0D", timedelta(0)),
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6S", timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ("PT1.5H", timedelta(hours=1.5)),
        ("PT2,25M", timedelta(minutes=2.25)),
        ("PT0.000001S", timedelta(microseconds=1)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
    ],
)
def test_parse_timedelta_duration(text, expected):
    result = isodate.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected


def test_parse_calendar_duration_and_force_duration():
    value = isodate.parse_duration("P1.5Y2.25M3DT4H")
    assert isinstance(value, Duration)
    assert value.years == Decimal("1.5")
    assert value.months == Decimal("2.25")
    assert value.tdelta == timedelta(days=3, hours=4)
    forced = isodate.parse_duration("P2D", as_timedelta_if_possible=False)
    assert isinstance(forced, Duration)
    assert forced == Duration(days=2)
    assert isodate.parse_duration("-P1Y2M3D") == -Duration(years=1, months=2, days=3)


@pytest.mark.parametrize("value", [None, 3, []])
def test_duration_parser_requires_string(value):
    with pytest.raises(TypeError):
        isodate.parse_duration(value)


@pytest.mark.parametrize("text", ["", "P", "1D", "P1Q", "P-1D", "P1DT2"])
def test_duration_rejects_invalid_strings(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_duration(text)


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(0), "P0D"),
        (timedelta(days=14), "P14D"),
        (timedelta(days=2, hours=3, minutes=4, seconds=5), "P2DT3H4M5S"),
        (timedelta(microseconds=120000), "PT0.12S"),
        (-timedelta(seconds=1), "-PT1S"),
        (Duration(years=2, months=3, days=4, seconds=5), "P2Y3M4DT5S"),
    ],
)
def test_duration_default_format(value, expected):
    assert isodate.duration_isoformat(value) == expected


def test_duration_week_and_custom_directives():
    assert isodate.duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
    value = Duration(years=3, months=4, days=2, hours=5, minutes=6, seconds=7, microseconds=8)
    assert _strfduration(value, "%Y/%m/%d %H:%M:%S.%f %%") == "0003/04/02 05:06:07.000008 %"


@pytest.mark.parametrize(
    "year, month, expected",
    [(2000, 2, 29), (1900, 2, 28), (2024, 2, 29), (2023, 2, 28), (2024, 1, 31), (2024, 4, 30)],
)
def test_max_days_in_month(year, month, expected):
    assert max_days_in_month(year, month) == expected


def test_fquotmod_wraps_both_directions():
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-1), 1, 13) == (-1, Decimal(11))


def test_duration_date_arithmetic_clamps_and_carries():
    assert date(2020, 1, 31) + Duration(months=1) == date(2020, 2, 29)
    assert date(2021, 1, 31) + Duration(months=1) == date(2021, 2, 28)
    assert date(2020, 12, 31) + Duration(months=2) == date(2021, 2, 28)
    assert date(2020, 1, 31) - Duration(months=2) == date(2019, 11, 30)
    assert Duration(years=1, days=2) + date(2020, 2, 29) == date(2021, 3, 2)
    dt = datetime(2020, 1, 31, 12, 30) + Duration(months=1, hours=2)
    assert dt == datetime(2020, 2, 29, 14, 30)


def test_fractional_calendar_units_cannot_modify_dates():
    with pytest.raises(ValueError):
        date(2020, 1, 1) + Duration(months=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2020, 1, 1) - Duration(years=Decimal("1.5"))


def test_duration_arithmetic_equality_hash_and_attributes():
    a = Duration(years=1, months=2, days=3, seconds=4)
    b = Duration(months=14, days=3, seconds=4)
    assert a == b and not (a != b)
    assert a.days == 3 and a.seconds == 4
    assert a + Duration(months=1, days=2) == Duration(years=1, months=3, days=5, seconds=4)
    assert a - Duration(months=1, days=2) == Duration(years=1, months=1, days=1, seconds=4)
    assert a + timedelta(days=2) == Duration(years=1, months=2, days=5, seconds=4)
    assert timedelta(days=10) - Duration(days=3) == Duration(days=7)
    assert 2 * Duration(months=2, days=3) == Duration(months=4, days=6)
    assert -Duration(months=2, days=3) == Duration(months=-2, days=-3)
    assert Duration(days=2) == timedelta(days=2)
    assert Duration(months=1) != timedelta(0)
    assert len({a, Duration(years=1, months=2, days=3, seconds=4)}) == 1


def test_duration_conversion_requires_one_endpoint():
    dur = Duration(months=1, days=2)
    assert dur.totimedelta(start=date(2020, 1, 31)) == timedelta(days=31)
    assert dur.totimedelta(end=date(2020, 3, 31)) == timedelta(days=33)
    with pytest.raises(ValueError):
        dur.totimedelta()
    with pytest.raises(ValueError):
        dur.totimedelta(start=date(2020, 1, 1), end=date(2020, 2, 1))


def test_duration_string_repr_and_pickle_state():
    value = Duration(years=2, months=1, days=3, seconds=4)
    assert str(value) == "2 years, 1 month, 3 days, 0:00:04"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert repr(value) == "isodate.duration.Duration(3, 4, 0, years=2, months=1)"
    assert pickle.loads(pickle.dumps(value)) == value


def test_unsupported_duration_operators_return_notimplemented_via_typeerror():
    with pytest.raises(TypeError):
        Duration() + object()
    with pytest.raises(TypeError):
        Duration() * 1.5
    with pytest.raises(TypeError):
        Duration() - object()
