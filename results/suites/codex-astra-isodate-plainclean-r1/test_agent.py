"""Behavioral tests written from the public API and implementation.

Expected values are explicit or computed independently with the standard library;
round trips supplement, rather than replace, assertions about parsed values.
"""
import calendar
import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
import isodate as iso


@pytest.mark.parametrize('text, expected', [
    ('20240229', date(2024, 2, 29)), ('2024-02-29', date(2024, 2, 29)),
    ('2024060', date(2024, 2, 29)), ('2023-060', date(2023, 3, 1)),
    ('2024-366', date(2024, 12, 31)), ('2024', date(2024, 1, 1)),
    ('2024-02', date(2024, 2, 1)), ('202402', date(2024, 2, 1)),
    ('20', date(2001, 1, 1)), ('00', date(1, 1, 1)),
    ('0001-01-01', date.min), ('9999-12-31', date.max),
    ('2020-W53-7', date(2021, 1, 3)), ('2020W537', date(2021, 1, 3)),
    ('2020-W01', date(2019, 12, 30)), ('2020W01', date(2019, 12, 30)),
])
def test_parse_dates(text, expected):
    assert iso.parse_date(text) == expected


@pytest.mark.parametrize('year', [1999, 2000, 2004, 2015, 2016, 2019, 2020, 2021, 2100])
def test_week_dates_against_standard_library(year):
    for week in (1, 26, date(year, 12, 28).isocalendar().week):
        for weekday in range(1, 8):
            expected = date.fromisocalendar(year, week, weekday)
            assert iso.parse_date(f'{year}-W{week:02}-{weekday}') == expected
            assert iso.parse_date(f'{year}W{week:02}{weekday}') == expected


def test_date_defaults_and_expanded_years_are_independent():
    assert iso.parse_date('2024', defaultmonth=6, defaultday=12) == date(2024, 6, 12)
    assert iso.parse_date('2024-02', defaultday=14) == date(2024, 2, 14)
    assert iso.parse_date('20', defaultmonth=3, defaultday=4) == date(2001, 3, 4)
    assert iso.parse_date('+002024-02-29', yeardigits=6) == date(2024, 2, 29)
    assert iso.parse_date('+2024-02-29', expanded=True) == date(2024, 2, 29)
    assert iso.parse_date('2024-02-29') == date(2024, 2, 29)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('+2024-02-29')


@pytest.mark.parametrize('text', ['', '2024/02/29', '2024-2-29', ' 2024-02-29',
                                       '2024-02-29x', '2024-02-29\n'])
def test_invalid_date_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text', ['2023-02-29', '1900-02-29', '2024-04-31',
                                       '2024-13-01', '2024-01-00', '0000-01-01'])
def test_impossible_calendar_dates(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


@pytest.mark.parametrize('text, expected', [
    ('12', time(12)), ('1234', time(12, 34)), ('12:34', time(12, 34)),
    ('123456', time(12, 34, 56)), ('T12:34:56', time(12, 34, 56)),
    ('12.5', time(12, 30)), ('12,125', time(12, 7, 30)),
    ('12:34.5', time(12, 34, 30)), ('1234,125', time(12, 34, 7, 500000)),
    ('12:34:56,123456', time(12, 34, 56, 123456)),
    ('12:34:56.1234569', time(12, 34, 56, 123456)),
    ('12:34.0000001', time(12, 34, 0, 6)),
    ('23:59:59.9999999', time(23, 59, 59, 999999)),
])
def test_time_precision(text, expected):
    actual = iso.parse_time(text)
    assert actual == expected
    assert actual.tzinfo is None


@pytest.mark.parametrize('suffix, minutes', [('Z', 0), ('+00:00', 0), ('-00', 0),
    ('+05:45', 345), ('-0330', -210), ('+09', 540), ('-00:30', -30)])
def test_timezones(suffix, minutes):
    zone = iso.parse_tzinfo(suffix)
    assert zone.utcoffset(None) == timedelta(minutes=minutes)
    assert zone.dst(None) == timedelta(0)
    parsed = iso.parse_time('12:34:56' + suffix)
    assert parsed.replace(tzinfo=None) == time(12, 34, 56)
    assert parsed.utcoffset() == timedelta(minutes=minutes)
    assert parsed.tzname() == ('UTC' if suffix == 'Z' else suffix)


@pytest.mark.parametrize('text', ['', '1:23', '12:3', '12:34:', '12:34:56x',
    '12:34:56\n', '12:34:56z', '12:34+5:30', '24:00', '23:60', '23:59:60'])
def test_invalid_times(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize('text, expected', [
    ('2024-02-29T12:34:56.123456Z', datetime(2024, 2, 29, 12, 34, 56, 123456, timezone.utc)),
    ('2024060T1234-0330', datetime(2024, 2, 29, 12, 34, tzinfo=timezone(timedelta(minutes=-210)))),
    ('2020-W53-7T12.5', datetime(2021, 1, 3, 12, 30)),
])
def test_datetime_composition(text, expected):
    actual = iso.parse_datetime(text)
    assert actual == expected
    assert actual.replace(tzinfo=None) == expected.replace(tzinfo=None)
    assert actual.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize('text', ['2024-02-29 12:30', '2024-02-29',
    '2024-02-29TT12:30', 'T12:30', '2024-02-29T'])
def test_invalid_datetimes(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


@pytest.mark.parametrize('text, expected', [
    ('P0D', timedelta()), ('PT0S', timedelta()), ('P2W', timedelta(days=14)),
    ('P1.5W', timedelta(days=10, hours=12)), ('P1,5D', timedelta(hours=36)),
    ('PT1.5H', timedelta(minutes=90)), ('PT1,5M', timedelta(seconds=90)),
    ('PT0.000001S', timedelta(microseconds=1)),
    ('P2DT3H4M5.25S', timedelta(days=2, hours=3, minutes=4, seconds=5.25)),
    ('-P2DT3H', -timedelta(days=2, hours=3)), ('+PT30S', timedelta(seconds=30)),
    ('P0Y0M3D', timedelta(days=3)),
])
def test_timedelta_durations(text, expected):
    actual = iso.parse_duration(text)
    assert type(actual) is timedelta
    assert actual == expected
    forced = iso.parse_duration(text, as_timedelta_if_possible=False)
    assert isinstance(forced, iso.Duration)
    assert forced.years == forced.months == 0
    assert forced.tdelta == expected


@pytest.mark.parametrize('text, years, months, delta', [
    ('P1Y2M3DT4H5M6S', '1', '2', timedelta(days=3, hours=4, minutes=5, seconds=6)),
    ('P1.25Y2,5M', '1.25', '2.5', timedelta()),
    ('-P1Y2M3DT4H', '-1', '-2', -timedelta(days=3, hours=4)),
    ('P00010203T040506', '1', '2', timedelta(days=3, hours=4, minutes=5, seconds=6)),
    ('P0001-02-03T04:05:06', '1', '2', timedelta(days=3, hours=4, minutes=5, seconds=6)),
])
def test_calendar_duration_components(text, years, months, delta):
    actual = iso.parse_duration(text)
    assert isinstance(actual, iso.Duration)
    assert actual.years == Decimal(years)
    assert actual.months == Decimal(months)
    assert actual.tdelta == delta


@pytest.mark.parametrize('text', ['', 'P', '1D', 'P1S', 'PT1Y', 'P1D2Y', 'P-1D', 'PT1Hjunk'])
def test_invalid_durations(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 1, b'P1D', timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value, expected', [
    (timedelta(), 'P0D'), (timedelta(days=14), 'P14D'),
    (timedelta(microseconds=1), 'PT0.000001S'),
    (timedelta(seconds=1.12), 'PT1.12S'),
    (-timedelta(microseconds=1), '-PT0.000001S'),
    (-timedelta(days=1, seconds=2), '-P1DT2S'),
    (timedelta(days=2, hours=3, minutes=4, seconds=5), 'P2DT3H4M5S'),
    (iso.Duration(years=1, months=2, days=3), 'P1Y2M3D'),
    (iso.Duration(years=Decimal('1.5')), 'P1.5Y'),
    (iso.Duration(months=-2, hours=-4), '-P2MT4H'),
])
def test_duration_format_and_roundtrip(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value


@pytest.mark.parametrize('fmt, expected', [
    (iso.DATE_EXT_COMPLETE, '2024-02-29'), (iso.DATE_BAS_COMPLETE, '20240229'),
    (iso.DATE_EXT_ORD_COMPLETE, '2024-060'), (iso.DATE_BAS_ORD_COMPLETE, '2024060'),
    (iso.DATE_EXT_WEEK_COMPLETE, '2024-W09-4'), (iso.DATE_BAS_WEEK_COMPLETE, '2024W094'),
    (iso.DATE_EXT_WEEK, '2024-W09'), (iso.DATE_BAS_WEEK, '2024W09'),
    (iso.DATE_EXT_MONTH, '2024-02'), (iso.DATE_BAS_MONTH, '202402'),
    (iso.DATE_YEAR, '2024'), (iso.DATE_CENTURY, '20'),
])
def test_date_formats(fmt, expected):
    assert iso.date_isoformat(date(2024, 2, 29), fmt) == expected


def test_formatting_directives_and_precision():
    dt = datetime(7, 2, 3, 4, 5, 6, 789, iso.FixedOffset(-3, -30, 'west'))
    assert iso.datetime_isoformat(dt) == '0007-02-03T04:05:06-03:30'
    assert iso.datetime_isoformat(dt, iso.DT_BAS_COMPLETE) == '00070203T040506-0330'
    assert iso.strftime(dt, '%Y/%m/%d %H:%M:%S.%f %% %q') == '0007/02/03 04:05:06.000789 % %q'
    assert iso.date_isoformat(dt, yeardigits=6) == '+000007-02-03'
    assert iso.time_isoformat(time(4, 5, 6)) == '04:05:06'
    assert iso.time_isoformat(time(4, 5, 6, tzinfo=iso.UTC)) == '04:05:06Z'
    assert iso.time_isoformat(time(4, 5), iso.TIME_BAS_MINUTE) == '0405'
    assert iso.duration_isoformat(timedelta(weeks=3), iso.D_WEEK) == 'P3W'
    assert iso.duration_isoformat(iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), iso.D_ALT_EXT) == 'P0001-02-03T04:05:06'


@pytest.mark.parametrize('offset, extended, basic, hours', [
    (345, '+05:45', '+0545', '+05'), (-210, '-03:30', '-0330', '-03'),
    (-30, '-00:30', '-0030', '-00'), (0, 'Z', 'Z', 'Z'),
])
def test_timezone_formatting(offset, extended, basic, hours):
    dt = datetime(2024, 1, 1, tzinfo=iso.FixedOffset(0, offset))
    assert iso.tz_isoformat(dt) == extended
    assert iso.tz_isoformat(dt, '%z') == basic
    assert iso.tz_isoformat(dt, '%h') == hours


def test_naive_and_utc_zones():
    assert iso.parse_tzinfo('') is None
    assert iso.parse_tzinfo('Z') is iso.UTC
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ''
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    with pytest.raises(ValueError):
        iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=iso.FixedOffset(1)), '%Q')


@pytest.mark.parametrize('year', [1900, 2000, 2023, 2024, 2100])
def test_month_arithmetic_clamps_to_calendar(year):
    for month in range(1, 13):
        start = date(year, month, calendar.monthrange(year, month)[1])
        for shift in (-25, -12, -1, 1, 12, 25):
            total = year * 12 + month - 1 + shift
            target_year, target_month = divmod(total, 12)
            target_month += 1
            expected = date(target_year, target_month, min(start.day, calendar.monthrange(target_year, target_month)[1]))
            duration = iso.Duration(months=shift)
            assert start + duration == expected
            assert duration + start == expected
            assert start - iso.Duration(months=-shift) == expected


def test_calendar_arithmetic_order_and_datetime_preservation():
    duration = iso.Duration(years=1, days=1, seconds=2)
    start = datetime(2024, 2, 29, 12, 34, 56, 789, iso.UTC)
    assert start + duration == datetime(2025, 3, 1, 12, 34, 58, 789, iso.UTC)
    assert start - duration == datetime(2023, 2, 27, 12, 34, 54, 789, iso.UTC)
    assert (start + duration).tzinfo is iso.UTC
    assert date(2023, 1, 31) + iso.Duration(months=1, days=1) == date(2023, 3, 1)


def test_duration_arithmetic_does_not_mutate_operands():
    a = iso.Duration(years=2, months=3, days=4, seconds=5)
    b = iso.Duration(years=1, months=2, days=3, seconds=4)
    assert a + b == iso.Duration(years=3, months=5, days=7, seconds=9)
    assert a - b == iso.Duration(years=1, months=1, days=1, seconds=1)
    assert -a == iso.Duration(years=-2, months=-3, days=-4, seconds=-5)
    assert 2 * a == a * 2 == iso.Duration(years=4, months=6, days=8, seconds=10)
    assert a * 0 == iso.Duration()
    delta = timedelta(days=1)
    assert a + delta == delta + a == iso.Duration(years=2, months=3, days=5, seconds=5)
    assert a - delta == iso.Duration(years=2, months=3, days=3, seconds=5)
    assert delta - a == iso.Duration(years=-2, months=-3, days=-3, seconds=-5)
    assert (a.years, a.months, a.tdelta) == (2, 3, timedelta(days=4, seconds=5))
    assert (b.years, b.months, b.tdelta) == (1, 2, timedelta(days=3, seconds=4))
    restored = pickle.loads(pickle.dumps(a))
    assert restored == a
    assert restored is not a


@pytest.mark.parametrize('a,b,equal', [
    (iso.Duration(years=1), iso.Duration(months=12), True),
    (iso.Duration(years=1), iso.Duration(months=11), False),
    (iso.Duration(months=1), iso.Duration(months=1, seconds=1), False),
    (iso.Duration(days=1), timedelta(days=1), True),
    (iso.Duration(years=1), timedelta(days=365), False),
    (iso.Duration(), None, False),
])
def test_duration_equality(a, b, equal):
    assert (a == b) is equal
    assert (a != b) is (not equal)


@pytest.mark.parametrize('duration', [iso.Duration(months=Decimal('0.5')), iso.Duration(years=Decimal('1.5'))])
def test_fractional_calendar_arithmetic_rejected(duration):
    with pytest.raises(ValueError):
        date(2024, 1, 1) + duration
    with pytest.raises(ValueError):
        date(2024, 1, 1) - duration


def test_duration_conversion_needs_exactly_one_anchor():
    duration = iso.Duration(months=1)
    assert duration.totimedelta(start=date(2024, 2, 1)) == timedelta(days=29)
    assert duration.totimedelta(start=date(2023, 2, 1)) == timedelta(days=28)
    assert duration.totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError):
        duration.totimedelta()
    with pytest.raises(ValueError):
        duration.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))


@pytest.mark.parametrize('operation', [lambda d: d + 1, lambda d: 1 + d,
    lambda d: d - 'x', lambda d: 'x' - d, lambda d: d * 1.5])
def test_unsupported_duration_operands(operation):
    with pytest.raises(TypeError):
        operation(iso.Duration(days=1))
