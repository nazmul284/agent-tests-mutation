"""Behavioral tests derived from the public parsers, formatters and Duration."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest
import isodate as iso


@pytest.mark.parametrize('text, expected', [
    ('2024-02-29', date(2024, 2, 29)), ('20240229', date(2024, 2, 29)),
    ('2024-060', date(2024, 2, 29)), ('2024060', date(2024, 2, 29)),
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
    assert iso.parse_date('2024', defaultmonth=7, defaultday=19) == date(2024, 7, 19)
    assert iso.parse_date('2024-02', defaultday=9) == date(2024, 2, 9)
    assert iso.parse_date('20', defaultmonth=3, defaultday=2) == date(2001, 3, 2)
    assert iso.parse_date('+002024-02-29', yeardigits=6) == date(2024, 2, 29)
    assert iso.parse_date('+2024-02-29', expanded=True) == date(2024, 2, 29)
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('+2024-02-29')
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date('002024-02-29', yeardigits=6)


@pytest.mark.parametrize('text', ['', '2024/02/29', '2024-2-29', ' 2024-02-29',
                                  '2024-02-29\n', '2024-02-29junk'])
def test_date_rejects_bad_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text', ['2023-02-29', '1900-02-29', '2024-13-01',
                                  '2024-04-31', '2024-01-00', '0000-01-01'])
def test_date_rejects_impossible_calendar_dates(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


@pytest.mark.parametrize('text, expected', [
    ('12', time(12)), ('T12', time(12)), ('1234', time(12, 34)),
    ('12:34', time(12, 34)), ('123456', time(12, 34, 56)),
    ('T12:34:56', time(12, 34, 56)), ('12.5', time(12, 30)),
    ('12,125', time(12, 7, 30)), ('12:34.5', time(12, 34, 30)),
    ('1234,125', time(12, 34, 7, 500000)),
    ('12:34:56.000001', time(12, 34, 56, 1)),
    ('12:34:56,1234569', time(12, 34, 56, 123456)),
    ('12:34.0000001', time(12, 34, 0, 6)),
    ('23:59:59.9999999', time(23, 59, 59, 999999)),
])
def test_parse_time(text, expected):
    assert iso.parse_time(text) == expected


@pytest.mark.parametrize('text', ['', '1:23', '12:3', '12:34:5', '12:34:56junk',
                                  '12:34:56\n', '12.5:30', '12:34z'])
def test_time_rejects_bad_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_time(text)


@pytest.mark.parametrize('text', ['24:00', '12:60', '12:34:60'])
def test_time_rejects_out_of_range_values(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize('suffix, minutes', [('Z', 0), ('+00', 0), ('-00:00', 0),
    ('+05:45', 345), ('+0545', 345), ('-03:30', -210), ('-0330', -210),
    ('+06', 360), ('-00:30', -30)])
def test_timezone_parsing_and_time_attachment(suffix, minutes):
    zone = iso.parse_tzinfo(suffix)
    assert zone.utcoffset(None) == timedelta(minutes=minutes)
    assert zone.dst(None) == timedelta(0)
    assert iso.parse_time('12:34:56' + suffix).utcoffset() == timedelta(minutes=minutes)
    if suffix == 'Z':
        assert zone is iso.UTC
    else:
        assert zone.tzname(None) == suffix


def test_naive_and_utc_zones():
    assert iso.parse_tzinfo('') is None
    assert iso.parse_time('12:34').tzinfo is None
    assert iso.UTC.tzname(None) == 'UTC'
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ''
    assert iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=iso.UTC)) == 'Z'


@pytest.mark.parametrize('fmt, expected', [(iso.TZ_EXT, '-03:30'),
    (iso.TZ_BAS, '-0330'), (iso.TZ_HOUR, '-03')])
def test_timezone_format(fmt, expected):
    dt = datetime(2024, 1, 1, tzinfo=iso.FixedOffset(-3, -30))
    assert iso.tz_isoformat(dt, fmt) == expected
    with pytest.raises(ValueError):
        iso.tz_isoformat(dt, '%invalid')


@pytest.mark.parametrize('text', ['2024-02-29T12:34:56.123456+05:45',
                                  '20240229T123456.123456+0545',
                                  '2024-060T12:34:56.123456+05:45'])
def test_datetime_combines_date_time_and_zone(text):
    result = iso.parse_datetime(text)
    assert result.replace(tzinfo=None) == datetime(2024, 2, 29, 12, 34, 56, 123456)
    assert result.utcoffset() == timedelta(hours=5, minutes=45)


@pytest.mark.parametrize('text', ['2024-02-29', '2024-02-29 12:34',
                                  '2024-02-29TT12:34', 'T12:34', '2024-02-29T'])
def test_invalid_datetime(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


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


def test_custom_and_default_formatting():
    dt = datetime(7, 2, 3, 4, 5, 6, 123, iso.UTC)
    assert iso.date_isoformat(dt) == '0007-02-03'
    assert iso.date_isoformat(dt, yeardigits=6) == '+000007-02-03'
    assert iso.time_isoformat(dt) == '04:05:06Z'
    assert iso.datetime_isoformat(dt) == '0007-02-03T04:05:06Z'
    assert iso.datetime_isoformat(dt, iso.DT_BAS_COMPLETE) == '00070203T040506Z'
    assert iso.strftime(dt, '%Y %% %H:%M:%S.%f %q') == '0007 % 04:05:06.000123 %q'


@pytest.mark.parametrize('year', [1900, 2000, 2015, 2016, 2020, 2021, 2024])
def test_week_and_ordinal_parsing_against_standard_library(year):
    # Exercise every day around ISO week-year transitions, plus leap days.
    for offset in list(range(-7, 8)) + [58, 59, 60, 364, 365]:
        day = date(year, 1, 1) + timedelta(days=offset)
        wy, week, weekday = day.isocalendar()
        assert iso.parse_date(f'{wy:04d}-W{week:02d}-{weekday}') == day
        ordinal = (day - date(day.year, 1, 1)).days + 1
        assert iso.parse_date(f'{day.year:04d}-{ordinal:03d}') == day


@pytest.mark.parametrize('text, expected', [
    ('P0D', timedelta()), ('PT0S', timedelta()), ('P2W', timedelta(days=14)),
    ('P1.5W', timedelta(days=10, hours=12)), ('P1,5D', timedelta(hours=36)),
    ('PT1.5H', timedelta(minutes=90)), ('PT1,5M', timedelta(seconds=90)),
    ('PT0.000001S', timedelta(microseconds=1)),
    ('P2DT3H4M5.25S', timedelta(days=2, hours=3, minutes=4, seconds=5.25)),
    ('-PT0.5S', timedelta(microseconds=-500000)), ('+P3D', timedelta(days=3)),
    ('P0Y0M2D', timedelta(days=2)),
])
def test_timedelta_duration_parsing(text, expected):
    result = iso.parse_duration(text)
    assert type(result) is timedelta
    assert result == expected
    forced = iso.parse_duration(text, as_timedelta_if_possible=False)
    assert isinstance(forced, iso.Duration)
    assert forced.years == forced.months == 0
    assert forced.tdelta == expected


@pytest.mark.parametrize('sign', ['', '+', '-'])
def test_calendar_duration_components(sign):
    result = iso.parse_duration(sign + 'P1.5Y2,25M3DT4H5M6.125S')
    factor = -1 if sign == '-' else 1
    assert isinstance(result, iso.Duration)
    assert result.years == Decimal('1.5') * factor
    assert result.months == Decimal('2.25') * factor
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.125) * factor


@pytest.mark.parametrize('text', ['P00010203T040506', 'P0001-02-03T04:05:06'])
def test_alternative_duration(text):
    result = iso.parse_duration(text)
    assert result.years == 1
    assert result.months == 2
    assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)


@pytest.mark.parametrize('text', ['', 'P', '1D', 'P-1D', 'PT1Q', 'P1D2Y', 'P1Dgarbage'])
def test_invalid_duration(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 42, b'P1D', timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value, expected', [
    (timedelta(), 'P0D'), (timedelta(days=2), 'P2D'),
    (timedelta(hours=25, minutes=2, seconds=3), 'P1DT1H2M3S'),
    (timedelta(microseconds=1), 'PT0.000001S'),
    (timedelta(seconds=1.25), 'PT1.25S'),
    (timedelta(microseconds=-1), '-PT0.000001S'),
    (timedelta(days=-2, seconds=-3), '-P2DT3S'),
    (iso.Duration(years=1, months=2, days=3, hours=4), 'P1Y2M3DT4H'),
    (iso.Duration(years=-1, months=-2, days=-3), '-P1Y2M3D'),
    (iso.Duration(months=Decimal('1.5')), 'P1.5M'),
])
def test_duration_format_and_roundtrip(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value


def test_duration_special_formats():
    assert iso.duration_isoformat(timedelta(weeks=3), iso.D_WEEK) == 'P3W'
    value = iso.Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert iso.duration_isoformat(value, iso.D_ALT_EXT) == 'P0001-02-03T04:05:06'
    assert iso.duration_isoformat(value, iso.D_ALT_BAS) == 'P00010203T040506'


@pytest.mark.parametrize('start, duration, expected', [
    (date(2024, 1, 31), iso.Duration(months=1), date(2024, 2, 29)),
    (date(2023, 1, 31), iso.Duration(months=1), date(2023, 2, 28)),
    (date(2000, 2, 29), iso.Duration(years=1), date(2001, 2, 28)),
    (date(1899, 2, 28), iso.Duration(years=1, days=1), date(1900, 3, 1)),
    (date(2024, 1, 31), iso.Duration(months=-2), date(2023, 11, 30)),
    (date(2023, 12, 31), iso.Duration(months=14), date(2025, 2, 28)),
    (date(2024, 1, 31), iso.Duration(months=1, days=1), date(2024, 3, 1)),
])
def test_calendar_addition_clips_before_adding_days(start, duration, expected):
    assert start + duration == expected
    assert duration + start == expected
    assert start - (-duration) == expected


def test_datetime_arithmetic_preserves_clock_and_zone():
    start = datetime(2024, 3, 31, 23, 45, 12, 123, iso.UTC)
    result = start - iso.Duration(months=1, hours=2)
    assert result == datetime(2024, 2, 29, 21, 45, 12, 123, iso.UTC)
    assert result.tzinfo is iso.UTC


def test_duration_arithmetic_does_not_mutate_operands():
    a = iso.Duration(years=2, months=3, days=4, seconds=5)
    b = iso.Duration(years=1, months=2, days=1, seconds=2)
    assert a + b == iso.Duration(years=3, months=5, days=5, seconds=7)
    assert a - b == iso.Duration(years=1, months=1, days=3, seconds=3)
    assert a * 2 == 2 * a == iso.Duration(years=4, months=6, days=8, seconds=10)
    assert -a == iso.Duration(years=-2, months=-3, days=-4, seconds=-5)
    td = timedelta(days=2)
    assert a + td == td + a == iso.Duration(years=2, months=3, days=6, seconds=5)
    assert a - td == iso.Duration(years=2, months=3, days=2, seconds=5)
    assert td - a == iso.Duration(years=-2, months=-3, days=-2, seconds=-5)
    assert a * 0 == timedelta(0)
    assert (a.years, a.months, a.tdelta) == (2, 3, timedelta(days=4, seconds=5))
    assert pickle.loads(pickle.dumps(a)) == a


def test_duration_equality():
    assert iso.Duration(years=1) == iso.Duration(months=12)
    assert not (iso.Duration(years=1) != iso.Duration(months=12))
    assert iso.Duration(days=3) == timedelta(days=3)
    assert iso.Duration(years=1) != timedelta(days=365)
    assert iso.Duration(months=1) != iso.Duration(months=2)
    assert iso.Duration(days=1) != iso.Duration(days=2)
    assert iso.Duration() != object()


@pytest.mark.parametrize('duration', [iso.Duration(months=Decimal('0.5')),
                                      iso.Duration(years=Decimal('1.5'))])
def test_fractional_calendar_units_cannot_be_added_to_dates(duration):
    with pytest.raises(ValueError):
        date(2024, 1, 1) + duration
    with pytest.raises(ValueError):
        date(2024, 1, 1) - duration


def test_duration_conversion_requires_one_anchor():
    duration = iso.Duration(months=1)
    assert duration.totimedelta(start=date(2024, 1, 31)) == timedelta(days=29)
    assert duration.totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError):
        duration.totimedelta()
    with pytest.raises(ValueError):
        duration.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))


@pytest.mark.parametrize('operation', [lambda d: d + 1, lambda d: d - 1,
                                      lambda d: d * 1.5, lambda d: 1 - d])
def test_unsupported_duration_arithmetic(operation):
    with pytest.raises(TypeError):
        operation(iso.Duration(months=1))
