"""Public API regression tests, with independent datetime/calendar oracles.

The ordinal parser's known one-day error is recorded as a strict xfail;
all other tests must pass normally. No external fixtures or dependencies.
"""
import calendar
import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import isodate as iso


@pytest.mark.parametrize('text, expected', [
    ('20240229', date(2024, 2, 29)), ('2024-02-29', date(2024, 2, 29)),
    ('0001-01-01', date.min), ('99991231', date.max),
    ('2024-02', date(2024, 2, 1)), ('202402', date(2024, 2, 1)),
    ('2024', date(2024, 1, 1)), ('20', date(2001, 1, 1)),
    ('00', date(1, 1, 1)),
])
def test_calendar_dates(text, expected):
    assert iso.parse_date(text) == expected


def test_date_defaults_and_expanded_years():
    assert iso.parse_date('2024', defaultmonth=6, defaultday=15) == date(2024, 6, 15)
    assert iso.parse_date('2024-02', defaultday=29) == date(2024, 2, 29)
    assert iso.parse_date('20', defaultmonth=3, defaultday=4) == date(2001, 3, 4)
    assert iso.parse_date('+002024-02-29', yeardigits=6) == date(2024, 2, 29)
    assert iso.parse_date('+20240229', expanded=True) == date(2024, 2, 29)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('+2024-02-29')
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('2024-02-29', expanded=True)


@pytest.mark.parametrize('year', [1999, 2000, 2004, 2015, 2016, 2020, 2021, 2024])
def test_week_dates_against_standard_library(year):
    for week in (1, 2, 52):
        for day in range(1, 8):
            expected = date.fromisocalendar(year, week, day)
            assert iso.parse_date(f'{year}-W{week:02}-{day}') == expected
            assert iso.parse_date(f'{year}W{week:02}{day}') == expected
        assert iso.parse_date(f'{year}-W{week:02}') == date.fromisocalendar(year, week, 1)
        assert iso.parse_date(f'{year}W{week:02}') == date.fromisocalendar(year, week, 1)
    assert iso.parse_date('2020-W53-7') == date(2021, 1, 3)


@pytest.mark.xfail(strict=True, reason='Ordinal parsing adds day number instead of day number minus one')
@pytest.mark.parametrize('text, expected', [
    ('2024-001', date(2024, 1, 1)), ('2024060', date(2024, 2, 29)),
    ('2024-366', date(2024, 12, 31)), ('2023-365', date(2023, 12, 31)),
])
def test_ordinal_dates(text, expected):
    assert iso.parse_date(text) == expected


@pytest.mark.parametrize('text', ['', '2024/02/29', '2024-2-29', ' 2024-02-29',
                                  '2024-02-29\n', '2024-02-29junk', 'abcd'])
def test_date_syntax_errors(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text', ['2023-02-29', '1900-02-29', '2024-04-31',
                                  '2024-13-01', '2024-01-32', '0000-01-01'])
def test_unrepresentable_dates(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


@pytest.mark.parametrize('text, expected', [
    ('00', time()), ('23', time(23)), ('1234', time(12, 34)),
    ('12:34', time(12, 34)), ('123456', time(12, 34, 56)),
    ('T12:34:56', time(12, 34, 56)), ('12.5', time(12, 30)),
    ('12,25', time(12, 15)), ('12:34.5', time(12, 34, 30)),
    ('1234,25', time(12, 34, 15)),
    ('12:34:56.1234569', time(12, 34, 56, 123456)),
    ('23:59:59,9999999', time(23, 59, 59, 999999)),
    ('00:00:00.000001', time(0, 0, 0, 1)),
])
def test_time_components_and_fractions(text, expected):
    assert iso.parse_time(text) == expected


@pytest.mark.parametrize('suffix, minutes', [('Z', 0), ('+00:00', 0), ('-00', 0),
    ('+05:45', 345), ('+0545', 345), ('-03:30', -210), ('-0330', -210), ('+06', 360)])
def test_timezones(suffix, minutes):
    tz = iso.parse_tzinfo(suffix)
    assert tz.utcoffset(None) == timedelta(minutes=minutes)
    assert tz.dst(None) == timedelta(0)
    result = iso.parse_time('12:34:56' + suffix)
    assert result == time(12, 34, 56, tzinfo=timezone(timedelta(minutes=minutes)))
    assert result.utcoffset() == timedelta(minutes=minutes)
    if suffix == 'Z':
        assert tz is iso.UTC
    else:
        assert tz.tzname(None) == suffix


def test_naive_and_utc_timezone():
    assert iso.parse_tzinfo('') is None
    assert iso.parse_time('12:34').tzinfo is None
    assert iso.UTC.tzname(None) == 'UTC'
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC


@pytest.mark.parametrize('text', ['', '1:23', '12:3', '12:34:', '12:34:56junk',
                                  '12:34:56\n', '12:34z', '12:34+5:30'])
def test_invalid_time_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_time(text)


@pytest.mark.parametrize('text', ['24:00', '12:60', '12:34:60'])
def test_time_range_errors(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize('text', ['2024-02-29T12:34:56', '20240229T123456',
                                  '2024-W09-4T12:34:56'])
def test_datetime_composition(text):
    assert iso.parse_datetime(text) == datetime(2024, 2, 29, 12, 34, 56)
    assert iso.parse_datetime(text + '-03:30') == datetime(
        2024, 2, 29, 16, 4, 56, tzinfo=timezone.utc)


@pytest.mark.parametrize('text', ['2024-02-29 12:34:56', '2024-02-29',
                                  '2024-02-29TT12:34:56', 'T12:34', '2024-02-29T'])
def test_invalid_datetime(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


@pytest.mark.parametrize('text, expected', [
    ('P0D', timedelta()), ('PT0S', timedelta()), ('P2W', timedelta(days=14)),
    ('P1.5W', timedelta(days=10, hours=12)), ('P1,5D', timedelta(hours=36)),
    ('PT1.5H', timedelta(minutes=90)), ('PT1,5M', timedelta(seconds=90)),
    ('PT0.000001S', timedelta(microseconds=1)),
    ('P2DT3H4M5.125S', timedelta(days=2, hours=3, minutes=4, seconds=5, milliseconds=125)),
    ('P1W2D', timedelta(days=9)), ('P0Y0M1D', timedelta(days=1)),
])
@pytest.mark.parametrize('sign, factor', [('', 1), ('+', 1), ('-', -1)])
def test_timedelta_duration_parsing(text, expected, sign, factor):
    result = iso.parse_duration(sign + text)
    assert type(result) is timedelta
    assert result == expected * factor


@pytest.mark.parametrize('sign, factor', [('', 1), ('-', -1)])
def test_calendar_duration_parsing(sign, factor):
    result = iso.parse_duration(sign + 'P1.5Y2,25M3DT4H5M6.125S')
    assert isinstance(result, iso.Duration)
    assert result.years == Decimal('1.5') * factor
    assert result.months == Decimal('2.25') * factor
    assert result.tdelta == factor * timedelta(days=3, hours=4, minutes=5, seconds=6.125)


def test_forced_duration_and_alternative_format():
    result = iso.parse_duration('PT2H', as_timedelta_if_possible=False)
    assert type(result) is iso.Duration
    assert result.years == result.months == 0
    assert result.tdelta == timedelta(hours=2)
    for text in ('P0001-02-03T04:05:06', 'P00010203T040506'):
        result = iso.parse_duration(text)
        assert result.years == 1
        assert result.months == 2
        assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize('text', ['', 'P', '1D', 'P-1D', 'PT1Q', 'P1S', 'P1D2Y', 'PT1Hjunk'])
def test_invalid_duration(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 42, b'P1D', timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value, expected', [
    (timedelta(), 'P0D'), (timedelta(days=2), 'P2D'),
    (timedelta(microseconds=1), 'PT0.000001S'),
    (timedelta(seconds=1, milliseconds=120), 'PT1.12S'),
    (timedelta(days=2, hours=3, minutes=4, seconds=5), 'P2DT3H4M5S'),
    (timedelta(microseconds=-1), '-PT0.000001S'),
    (timedelta(hours=-25), '-P1DT1H'),
    (iso.Duration(years=1, months=2, days=3), 'P1Y2M3D'),
    (iso.Duration(months=Decimal('1.5')), 'P1.5M'),
    (iso.Duration(years=-1, months=-2, days=-3), '-P1Y2M3D'),
])
def test_duration_formatting(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value


@pytest.mark.parametrize('year', [1, 1900, 2000, 2023, 2024, 9998])
def test_calendar_month_arithmetic(year):
    for month in range(1, 13):
        start = date(year, month, calendar.monthrange(year, month)[1])
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        expected = date(next_year, next_month, min(start.day, calendar.monthrange(next_year, next_month)[1]))
        delta = iso.Duration(months=1)
        assert start + delta == expected
        assert delta + start == expected
        assert delta.totimedelta(start=start) == expected - start


def test_duration_subtraction_and_leap_years():
    assert date(2024, 3, 31) - iso.Duration(months=1) == date(2024, 2, 29)
    assert date(2024, 1, 31) + iso.Duration(months=-2) == date(2023, 11, 30)
    assert date(2024, 2, 29) + iso.Duration(years=1) == date(2025, 2, 28)
    assert date(2000, 2, 29) - iso.Duration(years=100) == date(1900, 2, 28)
    assert iso.Duration(months=1).totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    start = datetime(2024, 1, 31, 23, 30, tzinfo=timezone.utc)
    assert start + iso.Duration(months=1, hours=2) == datetime(2024, 3, 1, 1, 30, tzinfo=timezone.utc)


def test_duration_algebra_and_no_operand_mutation():
    a = iso.Duration(years=1, months=2, days=3, seconds=4)
    b = iso.Duration(months=3, days=5)
    assert a + b == iso.Duration(years=1, months=5, days=8, seconds=4)
    assert a - b == iso.Duration(years=1, months=-1, days=-2, seconds=4)
    assert a * 2 == 2 * a == iso.Duration(years=2, months=4, days=6, seconds=8)
    assert a * 0 == timedelta(0)
    assert a + (-a) == timedelta(0)
    assert a + timedelta(days=1) == timedelta(days=1) + a == iso.Duration(years=1, months=2, days=4, seconds=4)
    assert timedelta(days=7) - b == iso.Duration(months=-3, days=2)
    assert a - timedelta(days=1) == iso.Duration(years=1, months=2, days=2, seconds=4)
    assert (a.years, a.months, a.tdelta) == (1, 2, timedelta(days=3, seconds=4))
    assert pickle.loads(pickle.dumps(a)) == a


def test_duration_equality():
    assert iso.Duration(years=1) == iso.Duration(months=12)
    assert not (iso.Duration(years=1) != iso.Duration(months=12))
    assert iso.Duration(days=1) == timedelta(days=1)
    assert iso.Duration(months=1) != timedelta(days=30)
    assert iso.Duration(months=1) != iso.Duration(months=2)
    assert iso.Duration(days=1) != iso.Duration(days=2)
    assert iso.Duration() != object()


@pytest.mark.parametrize('kwargs', [{}, {'start': date(2024, 1, 1), 'end': date(2024, 2, 1)}])
def test_totimedelta_requires_exactly_one_anchor(kwargs):
    with pytest.raises(ValueError):
        iso.Duration(months=1).totimedelta(**kwargs)


@pytest.mark.parametrize('duration', [iso.Duration(months=Decimal('0.5')), iso.Duration(years=Decimal('1.5'))])
def test_fractional_calendar_arithmetic_rejected(duration):
    with pytest.raises(ValueError):
        date(2024, 1, 1) + duration
    with pytest.raises(ValueError):
        date(2024, 1, 1) - duration


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


def test_format_precision_and_literals():
    value = datetime(7, 2, 3, 4, 5, 6, 123, tzinfo=iso.UTC)
    assert iso.datetime_isoformat(value) == '0007-02-03T04:05:06Z'
    assert iso.strftime(value, '%Y-%m-%dT%H:%M:%S.%f%Z %% %Q') == '0007-02-03T04:05:06.000123Z % %Q'
    assert iso.date_isoformat(value, yeardigits=6) == '+000007-02-03'
    assert iso.time_isoformat(value) == '04:05:06Z'
    assert iso.time_isoformat(value, iso.TIME_BAS_COMPLETE) == '040506'
    assert iso.time_isoformat(value, iso.TIME_EXT_MINUTE) == '04:05'
    assert iso.time_isoformat(value, iso.TIME_BAS_MINUTE) == '0405'
    assert iso.time_isoformat(value, iso.TIME_HOUR) == '04'
    assert iso.duration_isoformat(timedelta(weeks=3), iso.D_WEEK) == 'P3W'
    assert iso.duration_isoformat(iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), iso.D_ALT_EXT) == 'P0001-02-03T04:05:06'


@pytest.mark.parametrize('minutes', [-210, -30, 0, 345, 780])
def test_timezone_formatting(minutes):
    value = datetime(2024, 1, 1, tzinfo=iso.FixedOffset(0, minutes, 'test'))
    sign = '-' if minutes < 0 else '+'
    hours, mins = divmod(abs(minutes), 60)
    for fmt, expected in [('%Z', f'{sign}{hours:02}:{mins:02}'),
                          ('%z', f'{sign}{hours:02}{mins:02}'), ('%h', f'{sign}{hours:02}')]:
        assert iso.tz_isoformat(value, fmt) == ('Z' if minutes == 0 else expected)
    if minutes:
        with pytest.raises(ValueError):
            iso.tz_isoformat(value, '%Q')
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ''


@pytest.mark.parametrize('value', [datetime(1, 1, 1), datetime(2024, 2, 29, 23, 59, 59),
    datetime(2020, 12, 31, 12, 34, 56, tzinfo=iso.UTC),
    datetime(2024, 1, 1, tzinfo=iso.FixedOffset(-3, -30))])
def test_calendar_datetime_roundtrips(value):
    for fmt in (iso.DT_EXT_COMPLETE, iso.DT_BAS_COMPLETE):
        assert iso.parse_datetime(iso.datetime_isoformat(value, fmt)) == value
