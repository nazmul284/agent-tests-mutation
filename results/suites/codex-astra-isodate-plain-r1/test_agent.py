"""Regression tests written from the public API and implementation contracts."""

import calendar
import pickle
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest

import isodate as iso
from isodate.duration import fquotmod, max_days_in_month


@pytest.mark.parametrize('text, expected', [
    ('20240229', date(2024, 2, 29)), ('2024-02-29', date(2024, 2, 29)),
    ('2024060', date(2024, 2, 29)), ('2024-060', date(2024, 2, 29)),
    ('2023-365', date(2023, 12, 31)), ('2024-366', date(2024, 12, 31)),
    ('2020-W01-1', date(2019, 12, 30)), ('2020W537', date(2021, 1, 3)),
    ('2021-W01', date(2021, 1, 4)), ('2021W01', date(2021, 1, 4)),
    ('2024-02', date(2024, 2, 1)), ('202402', date(2024, 2, 1)),
    ('2024', date(2024, 1, 1)), ('20', date(2001, 1, 1)),
    ('0001-01-01', date.min), ('9999-12-31', date.max),
])
def test_parse_date(text, expected):
    assert iso.parse_date(text) == expected


def test_date_defaults_and_expanded_years():
    assert iso.parse_date('2024', defaultmonth=6, defaultday=15) == date(2024, 6, 15)
    assert iso.parse_date('2024-02', defaultday=29) == date(2024, 2, 29)
    assert iso.parse_date('20', defaultmonth=6, defaultday=15) == date(2001, 6, 15)
    assert iso.parse_date('+002024-02-29', yeardigits=6) == date(2024, 2, 29)
    assert iso.parse_date('+2024-02-29', expanded=True) == date(2024, 2, 29)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('2024-02-29', expanded=True)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('+2024-02-29')
    # Changing the cached regex configuration must not affect later calls.
    assert iso.parse_date('2024-02-29') == date(2024, 2, 29)


@pytest.mark.parametrize('text', ['', '2024/02/29', '2024-2-29', '2024-02-29x',
                                  ' 2024-02-29', '2024-02-29\n'])
def test_bad_date_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text', ['2023-02-29', '1900-02-29', '2024-04-31',
                                  '2024-13-01', '2024-01-00', '0000-01-01'])
def test_unrepresentable_dates(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


@pytest.mark.parametrize('year', [1999, 2000, 2004, 2015, 2020, 2021, 2100])
def test_week_and_ordinal_parsing_against_stdlib(year):
    for week in (1, 2, 26, 52):
        for day in (1, 4, 7):
            assert iso.parse_date(f'{year}-W{week:02d}-{day}') == date.fromisocalendar(year, week, day)
    for ordinal in (1, 59, 60, 61, 365):
        assert iso.parse_date(f'{year}-{ordinal:03d}') == date(year, 1, 1) + timedelta(days=ordinal - 1)


@pytest.mark.parametrize('text, expected', [
    ('12', time(12)), ('T12', time(12)), ('1234', time(12, 34)),
    ('12:34', time(12, 34)), ('123456', time(12, 34, 56)),
    ('T12:34:56', time(12, 34, 56)), ('12.5', time(12, 30)),
    ('12,125', time(12, 7, 30)), ('12:34.5', time(12, 34, 30)),
    ('1234,125', time(12, 34, 7, 500000)),
    ('12:34:56.1234569', time(12, 34, 56, 123456)),
    ('12:34:59.9999999', time(12, 34, 59, 999999)),
    ('12:34.0000001', time(12, 34, 0, 6)),
    ('00:00:00,000001', time(0, 0, 0, 1)),
])
def test_parse_time(text, expected):
    assert iso.parse_time(text) == expected


@pytest.mark.parametrize('suffix, minutes', [('Z', 0), ('+00', 0), ('-00:30', -30),
    ('+05:45', 345), ('-0330', -210), ('+06', 360)])
def test_timezones(suffix, minutes):
    zone = iso.parse_tzinfo(suffix)
    assert zone.utcoffset(None) == timedelta(minutes=minutes)
    assert zone.dst(None) == timedelta(0)
    assert iso.parse_time('12:34:56' + suffix).utcoffset() == timedelta(minutes=minutes)
    parsed = iso.parse_datetime('2024-02-29T12:34:56' + suffix)
    assert parsed == datetime(2024, 2, 29, 12, 34, 56, tzinfo=timezone(timedelta(minutes=minutes)))
    assert iso.parse_datetime(iso.datetime_isoformat(parsed)) == parsed


@pytest.mark.parametrize('text', ['', '1:23', '12:3', '12:34:5', '12:34junk',
                                  '12:34\n', '12:34+5', '12:34:56.', '24:00', '12:60', '12:34:60'])
def test_invalid_time(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize('text', ['', '2024-02-29', '2024-02-29 12:00',
                                  '2024-02-29TT12:00', 'T12:00', '2024-02-29T'])
def test_invalid_datetime(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


def test_mixed_datetime_forms():
    assert iso.parse_datetime('2020-W01-1T123045.25') == datetime(2019, 12, 30, 12, 30, 45, 250000)
    assert iso.parse_datetime('2024060T12.5') == datetime(2024, 2, 29, 12, 30)


@pytest.mark.parametrize('fmt, expected', [
    (iso.DATE_EXT_COMPLETE, '2024-02-29'), (iso.DATE_BAS_COMPLETE, '20240229'),
    (iso.DATE_EXT_ORD_COMPLETE, '2024-060'), (iso.DATE_BAS_ORD_COMPLETE, '2024060'),
    (iso.DATE_EXT_WEEK_COMPLETE, '2024-W09-4'), (iso.DATE_BAS_WEEK_COMPLETE, '2024W094'),
    (iso.DATE_EXT_WEEK, '2024-W09'), (iso.DATE_BAS_WEEK, '2024W09'),
    (iso.DATE_EXT_MONTH, '2024-02'), (iso.DATE_BAS_MONTH, '202402'),
    (iso.DATE_YEAR, '2024'), (iso.DATE_CENTURY, '20'),
])
def test_date_formatting(fmt, expected):
    assert iso.date_isoformat(date(2024, 2, 29), fmt) == expected


def test_custom_formatting():
    dt = datetime(7, 2, 3, 4, 5, 6, 123, tzinfo=iso.UTC)
    assert iso.datetime_isoformat(dt) == '0007-02-03T04:05:06Z'
    assert iso.strftime(dt, '%Y %m %d %H %M %S %f %% %Q') == '0007 02 03 04 05 06 000123 % %Q'
    assert iso.date_isoformat(dt, yeardigits=6) == '+000007-02-03'
    assert iso.time_isoformat(time(4, 5, 6)) == '04:05:06'
    assert iso.time_isoformat(dt, iso.TIME_BAS_COMPLETE) == '040506'
    assert iso.time_isoformat(dt, iso.TIME_EXT_MINUTE) == '04:05'
    assert iso.time_isoformat(dt, iso.TIME_BAS_MINUTE) == '0405'
    assert iso.time_isoformat(dt, iso.TIME_HOUR) == '04'


@pytest.mark.parametrize('minutes, extended, basic, hour', [
    (345, '+05:45', '+0545', '+05'), (-210, '-03:30', '-0330', '-03'),
    (-30, '-00:30', '-0030', '-00'), (0, '+00:00', '+0000', '+00'),
])
def test_timezone_formatting(minutes, extended, basic, hour):
    dt = datetime(2024, 1, 1, tzinfo=timezone(timedelta(minutes=minutes)))
    for fmt, expected in [(iso.TZ_EXT, extended), (iso.TZ_BAS, basic), (iso.TZ_HOUR, hour)]:
        assert iso.tz_isoformat(dt, fmt) == expected


def test_timezone_special_cases():
    assert iso.parse_tzinfo('') is None
    assert iso.parse_tzinfo('Z') is iso.UTC
    assert iso.UTC.tzname(None) == 'UTC'
    for fmt in (iso.TZ_EXT, iso.TZ_BAS, iso.TZ_HOUR):
        assert iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=iso.UTC), fmt) == 'Z'
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ''
    zone = iso.FixedOffset(-3, -30, 'west')
    assert zone.tzname(None) == 'west'
    assert zone.utcoffset(None) == timedelta(hours=-3, minutes=-30)
    with pytest.raises(ValueError):
        iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=zone), '%unknown')
    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None
    assert iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=NoOffset())) == ''


@pytest.mark.parametrize('text, expected', [
    ('P0D', timedelta()), ('PT0S', timedelta()), ('P2W', timedelta(days=14)),
    ('P1.5W', timedelta(days=10, hours=12)), ('P1,5D', timedelta(hours=36)),
    ('PT1.5H', timedelta(minutes=90)), ('PT1,5M', timedelta(seconds=90)),
    ('PT0.000001S', timedelta(microseconds=1)),
    ('P2DT3H4M5.25S', timedelta(days=2, hours=3, minutes=4, seconds=5.25)),
    ('-P2DT3H', -timedelta(days=2, hours=3)), ('+PT1S', timedelta(seconds=1)),
    ('P0Y0M1D', timedelta(days=1)),
])
def test_fixed_duration_parsing(text, expected):
    result = iso.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected
    forced = iso.parse_duration(text, as_timedelta_if_possible=False)
    assert isinstance(forced, iso.Duration)
    assert forced.years == forced.months == 0
    assert forced.tdelta == expected


@pytest.mark.parametrize('sign', ['', '+', '-'])
def test_calendar_duration_parsing(sign):
    factor = -1 if sign == '-' else 1
    result = iso.parse_duration(sign + 'P1.5Y2,25M3DT4H5M6.125S')
    assert isinstance(result, iso.Duration)
    assert result.years == Decimal('1.5') * factor
    assert result.months == Decimal('2.25') * factor
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.125) * factor


@pytest.mark.parametrize('text', ['P00010203T040506', 'P0001-02-03T04:05:06'])
def test_alternative_duration(text):
    value = iso.parse_duration(text)
    assert value.years == 1 and value.months == 2
    assert value.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize('text', ['', 'P', '1D', 'P-1D', 'PT1D', 'P1S', 'P1Dgarbage', 'P1M1Y', 'pt1s'])
def test_invalid_duration(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 1, b'P1D', timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value, expected', [
    (timedelta(), 'P0D'), (timedelta(days=2), 'P2D'),
    (timedelta(hours=1), 'PT1H'), (timedelta(minutes=1), 'PT1M'),
    (timedelta(microseconds=1), 'PT0.000001S'),
    (timedelta(seconds=1, microseconds=120000), 'PT1.12S'),
    (-timedelta(microseconds=1), '-PT0.000001S'),
    (timedelta(days=-2, seconds=-3), '-P2DT3S'),
    (iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), 'P1Y2M3DT4H5M6S'),
    (iso.Duration(years=Decimal('1.5')), 'P1.5Y'),
    (iso.Duration(months=-2), '-P2M'),
])
def test_duration_formatting_and_roundtrip(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value


def test_duration_alternative_formatting():
    value = iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert iso.duration_isoformat(value, iso.D_ALT_EXT) == 'P0001-02-03T04:05:06'
    assert iso.duration_isoformat(value, iso.D_ALT_BAS) == 'P00010203T040506'
    assert iso.duration_isoformat(timedelta(days=14), iso.D_WEEK) == 'P2W'


@pytest.mark.parametrize('start, months, years, expected', [
    (date(2023, 1, 31), 1, 0, date(2023, 2, 28)),
    (date(2024, 1, 31), 1, 0, date(2024, 2, 29)),
    (date(2024, 3, 31), -1, 0, date(2024, 2, 29)),
    (date(2024, 2, 29), 0, 1, date(2025, 2, 28)),
    (date(2000, 2, 29), 0, 100, date(2100, 2, 28)),
    (date(2024, 1, 31), -13, 0, date(2022, 12, 31)),
    (date(2024, 12, 31), 14, 0, date(2026, 2, 28)),
])
def test_calendar_arithmetic(start, months, years, expected):
    duration = iso.Duration(months=months, years=years)
    assert start + duration == expected
    assert duration + start == expected
    assert start - (-duration) == expected


def test_datetime_arithmetic_preserves_clock_and_zone():
    start = datetime(2024, 1, 31, 23, 30, 1, 123, tzinfo=iso.UTC)
    delta = iso.Duration(months=1, days=1, hours=2)
    assert start + delta == datetime(2024, 3, 2, 1, 30, 1, 123, tzinfo=iso.UTC)
    assert (start + delta).tzinfo is iso.UTC
    assert start - delta == datetime(2023, 12, 30, 21, 30, 1, 123, tzinfo=iso.UTC)


def test_duration_operators_and_no_mutation():
    a = iso.Duration(years=2, months=3, days=4, seconds=5)
    b = iso.Duration(years=1, months=2, days=1, seconds=2)
    assert a + b == iso.Duration(years=3, months=5, days=5, seconds=7)
    assert a - b == iso.Duration(years=1, months=1, days=3, seconds=3)
    assert -a == iso.Duration(years=-2, months=-3, days=-4, seconds=-5)
    assert a * 2 == 2 * a == iso.Duration(years=4, months=6, days=8, seconds=10)
    assert a * 0 == timedelta(0)
    td = timedelta(days=2)
    assert a + td == td + a == iso.Duration(years=2, months=3, days=6, seconds=5)
    assert a - td == iso.Duration(years=2, months=3, days=2, seconds=5)
    assert td - a == iso.Duration(years=-2, months=-3, days=-2, seconds=-5)
    assert a.years == 2 and a.months == 3 and a.tdelta == timedelta(days=4, seconds=5)
    assert pickle.loads(pickle.dumps(a)) == a


@pytest.mark.parametrize('a, b, equal', [
    (iso.Duration(years=1), iso.Duration(months=12), True),
    (iso.Duration(days=2), timedelta(days=2), True),
    (iso.Duration(years=1), timedelta(days=365), False),
    (iso.Duration(months=1), iso.Duration(months=2), False),
    (iso.Duration(months=1), iso.Duration(months=1, seconds=1), False),
    (iso.Duration(), object(), False),
])
def test_duration_equality(a, b, equal):
    assert (a == b) is equal
    assert (a != b) is (not equal)


@pytest.mark.parametrize('duration', [iso.Duration(years=0.5), iso.Duration(months=0.5)])
def test_fractional_calendar_arithmetic_rejected(duration):
    with pytest.raises(ValueError, match='fractional'):
        date(2024, 1, 1) + duration
    with pytest.raises(ValueError, match='fractional'):
        date(2024, 1, 1) - duration


def test_duration_conversion_needs_one_anchor():
    duration = iso.Duration(months=1)
    assert duration.totimedelta(start=date(2024, 2, 1)) == timedelta(days=29)
    assert duration.totimedelta(start=date(2023, 2, 1)) == timedelta(days=28)
    assert duration.totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError):
        duration.totimedelta()
    with pytest.raises(ValueError):
        duration.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))


def test_unsupported_duration_operands():
    duration = iso.Duration(months=1)
    with pytest.raises(TypeError):
        duration + 1
    with pytest.raises(TypeError):
        duration - '1'
    with pytest.raises(TypeError):
        duration * 1.5


@pytest.mark.parametrize('year', [1900, 2000, 2023, 2024, 2100, 2400])
def test_month_lengths(year):
    for month in range(1, 13):
        assert max_days_in_month(year, month) == calendar.monthrange(year, month)[1]


@pytest.mark.parametrize('value', [-25, -12, -1, 0, 1, 12, 13, 25, 37])
def test_month_normalization(value):
    quotient, remainder = fquotmod(Decimal(value), 1, 13)
    assert (quotient, remainder) == (divmod(value - 1, 12)[0], divmod(value - 1, 12)[1] + 1)
