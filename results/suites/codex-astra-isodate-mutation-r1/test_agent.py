"""Behavioral tests for ISO parsing, formatting, and calendar durations."""
import calendar
import pickle
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest
import isodate as iso
from isodate.duration import fquotmod, max_days_in_month
from isodate.isotzinfo import build_tzinfo
import isodate.tzinfo as zones


@pytest.mark.parametrize('text,expected', [
    ('20240229', date(2024, 2, 29)), ('2024-02-29', date(2024, 2, 29)),
    ('2024060', date(2024, 2, 29)), ('2024-060', date(2024, 2, 29)),
    ('2023365', date(2023, 12, 31)), ('2024-366', date(2024, 12, 31)),
    ('2024', date(2024, 1, 1)), ('20', date(2001, 1, 1)),
    ('00', date(1, 1, 1)), ('2024-02', date(2024, 2, 1)),
    ('202402', date(2024, 2, 1)), ('0001-01-01', date.min),
    ('9999-12-31', date.max),
])
def test_dates(text, expected):
    assert iso.parse_date(text) == expected


@pytest.mark.parametrize('year', range(1998, 2027))
def test_iso_week_dates_against_calendar(year):
    for week in (1, 2, 26, 52):
        for weekday in (1, 4, 7):
            expected = date.fromisocalendar(year, week, weekday)
            assert iso.parse_date(f'{year}-W{week:02}-{weekday}') == expected
            assert iso.parse_date(f'{year}W{week:02}{weekday}') == expected
        monday = date.fromisocalendar(year, week, 1)
        assert iso.parse_date(f'{year}-W{week:02}') == monday
        assert iso.parse_date(f'{year}W{week:02}') == monday
    if date(year, 12, 28).isocalendar().week == 53:
        assert iso.parse_date(f'{year}-W53-7') == date.fromisocalendar(year, 53, 7)


@pytest.mark.parametrize('text,kwargs,expected', [
    ('2024', dict(defaultmonth=7, defaultday=9), date(2024, 7, 9)),
    ('20', dict(defaultmonth=7, defaultday=9), date(2001, 7, 9)),
    ('2024-03', dict(defaultmonth=7, defaultday=9), date(2024, 3, 9)),
    ('2024-03-02', dict(defaultmonth=7, defaultday=9), date(2024, 3, 2)),
    ('+2024-02-29', dict(expanded=True), date(2024, 2, 29)),
    ('+0020240229', dict(yeardigits=6), date(2024, 2, 29)),
    ('+002024-060', dict(yeardigits=6), date(2024, 2, 29)),
    ('+002024-W01-1', dict(yeardigits=6), date(2024, 1, 1)),
    ('+0020', dict(yeardigits=6), date(2001, 1, 1)),
])
def test_date_options(text, kwargs, expected):
    assert iso.parse_date(text, **kwargs) == expected


@pytest.mark.parametrize('text', ['', '2024/02/29', '2024-2-29', ' 2024', '2024\n',
                                       '2024-02-29junk', '+2024-02-29', '2024-w01-1'])
def test_bad_date_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text,kwargs', [('2023-02-29', {}), ('2024-13-01', {}),
    ('2024-01-00', {}), ('0000-01-01', {}), ('-2024-01-01', {'expanded': True}),
    ('2024-01-01', {'expanded': True}), ('002024-01-01', {'yeardigits': 6})])
def test_invalid_date_values(text, kwargs):
    with pytest.raises(ValueError):
        iso.parse_date(text, **kwargs)


@pytest.mark.parametrize('text,expected', [
    ('00', time(0)), ('23', time(23)), ('1234', time(12, 34)),
    ('12:34', time(12, 34)), ('123456', time(12, 34, 56)),
    ('12:34:56', time(12, 34, 56)), ('T12:34:56', time(12, 34, 56)),
    ('12.5', time(12, 30)), ('T12,5125', time(12, 30, 45)),
    ('12:34.125', time(12, 34, 7, 500000)), ('1234,125', time(12, 34, 7, 500000)),
    ('12:34:56.1234569', time(12, 34, 56, 123456)),
    ('123456,0000019', time(12, 34, 56, 1)),
    ('12:34.0000001', time(12, 34, 0, 6)),
    ('12.000001', time(12, 0, 0, 3600)),
    ('23:59:59.9999999', time(23, 59, 59, 999999)),
])
def test_times(text, expected):
    assert iso.parse_time(text) == expected


@pytest.mark.parametrize('suffix,minutes', [('Z', 0), ('+00', 0), ('-00:30', -30),
    ('+05:45', 345), ('-0330', -210), ('+14', 840), ('-12', -720)])
@pytest.mark.parametrize('clock', ['12', '1234', '12:34:56.125'])
def test_time_zones(clock, suffix, minutes):
    result = iso.parse_time(clock + suffix)
    assert result.replace(tzinfo=None) == iso.parse_time(clock)
    assert result.utcoffset() == timedelta(minutes=minutes)
    assert result.tzinfo.dst(None) == timedelta(0)
    assert result.tzname() == ('UTC' if suffix == 'Z' else suffix)


@pytest.mark.parametrize('text', ['', '1', '12:3', '12:34:', '12.5:30', '24:00',
    '12:60', '12:00:60', '12:00:00.', '12:00:00z', '12:00:00 junk', '12\n'])
def test_invalid_times(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


@pytest.mark.parametrize('text,expected', [
    ('2024-02-29T12:34:56.123456+05:45', datetime(2024, 2, 29, 12, 34, 56, 123456,
                                               timezone(timedelta(hours=5, minutes=45)))),
    ('2024060T123456Z', datetime(2024, 2, 29, 12, 34, 56, tzinfo=timezone.utc)),
    ('2020-W01-1T12.5', datetime(2019, 12, 30, 12, 30)),
])
def test_datetimes(text, expected):
    assert iso.parse_datetime(text) == expected


@pytest.mark.parametrize('text', ['2024-02-29 12:00', '2024-02-29',
                                       '2024-02-29TT12:00', 'T12:00', '2024-02-29T'])
def test_invalid_datetimes(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_datetime(text)


@pytest.mark.parametrize('text,expected', [
    ('P0D', timedelta(0)), ('PT0S', timedelta(0)), ('P2W', timedelta(days=14)),
    ('P1.5W', timedelta(days=10, hours=12)), ('P2.5D', timedelta(days=2, hours=12)),
    ('PT1.5H', timedelta(hours=1, minutes=30)), ('PT1,5M', timedelta(seconds=90)),
    ('PT0.000001S', timedelta(microseconds=1)),
    ('P2W3DT4H5M6.125S', timedelta(days=17, hours=4, minutes=5, seconds=6.125)),
    ('P0Y0M2D', timedelta(days=2)),
])
def test_timedelta_parse(text, expected):
    for sign, multiplier in [('', 1), ('+', 1), ('-', -1)]:
        result = iso.parse_duration(sign + text)
        assert type(result) is timedelta
        assert result == expected * multiplier
        forced = iso.parse_duration(sign + text, as_timedelta_if_possible=False)
        assert type(forced) is iso.Duration
        assert forced.years == 0 and forced.months == 0
        assert forced.tdelta == expected * multiplier


@pytest.mark.parametrize('text,years,months,delta', [
    ('P1Y', '1', '0', timedelta(0)), ('P2M', '0', '2', timedelta(0)),
    ('P1.5Y2,25M3W4DT5H6M7.125S', '1.5', '2.25',
     timedelta(days=25, hours=5, minutes=6, seconds=7.125)),
])
def test_calendar_duration_parse(text, years, months, delta):
    for sign, factor in [('', 1), ('+', 1), ('-', -1)]:
        result = iso.parse_duration(sign + text)
        assert type(result) is iso.Duration
        assert result.years == Decimal(years) * factor
        assert result.months == Decimal(months) * factor
        assert result.tdelta == delta * factor


@pytest.mark.parametrize('text', ['P0002-03-04T05:06:07', 'P00020304T050607',
                                       'P0002-063T05:06:07', 'P0002063T050607'])
def test_alternative_durations(text):
    result = iso.parse_duration(text)
    assert (result.years, result.months, result.tdelta) == (
        2, 3, timedelta(days=4, hours=5, minutes=6, seconds=7))


@pytest.mark.parametrize('text', ['', 'P', '1D', 'P-1D', 'P1Q', 'P1D2Y', 'PT1M2H', 'P1S', 'junk'])
def test_invalid_durations(text):
    with pytest.raises(ValueError):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 123, b'P1D', timedelta(days=1)])
def test_duration_requires_string(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value,expected', [
    (timedelta(), 'P0D'), (timedelta(days=1), 'P1D'),
    (timedelta(hours=1), 'PT1H'), (timedelta(minutes=1), 'PT1M'),
    (timedelta(seconds=1), 'PT1S'), (timedelta(microseconds=1), 'PT0.000001S'),
    (timedelta(seconds=1, microseconds=120000), 'PT1.12S'),
    (timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=600), 'P2DT3H4M5.0006S'),
    (iso.Duration(years=2), 'P2Y'), (iso.Duration(months=3), 'P3M'),
    (iso.Duration(years=Decimal('1.5'), months=Decimal('2.25'), days=4,
                  hours=5, minutes=6, seconds=7), 'P1.5Y2.25M4DT5H6M7S'),
])
def test_duration_format(value, expected):
    assert iso.duration_isoformat(value) == expected
    assert iso.parse_duration(expected) == value
    if value != timedelta(0):
        assert iso.duration_isoformat(-value) == '-' + expected
        assert iso.parse_duration('-' + expected) == -value


@pytest.mark.parametrize('name,expected', [
    ('DATE_BAS_COMPLETE', '20240229'), ('DATE_EXT_COMPLETE', '2024-02-29'),
    ('DATE_BAS_WEEK_COMPLETE', '2024W094'), ('DATE_EXT_WEEK_COMPLETE', '2024-W09-4'),
    ('DATE_BAS_ORD_COMPLETE', '2024060'), ('DATE_EXT_ORD_COMPLETE', '2024-060'),
    ('DATE_BAS_WEEK', '2024W09'), ('DATE_EXT_WEEK', '2024-W09'),
    ('DATE_BAS_MONTH', '202402'), ('DATE_EXT_MONTH', '2024-02'),
    ('DATE_YEAR', '2024'), ('DATE_CENTURY', '20'),
])
def test_date_formats(name, expected):
    assert iso.date_isoformat(date(2024, 2, 29), getattr(iso, name)) == expected


@pytest.mark.parametrize('name,expected', [
    ('TIME_BAS_COMPLETE', '030405'), ('TIME_EXT_COMPLETE', '03:04:05'),
    ('TIME_BAS_MINUTE', '0304'), ('TIME_EXT_MINUTE', '03:04'), ('TIME_HOUR', '03'),
])
def test_time_formats(name, expected):
    assert iso.time_isoformat(time(3, 4, 5, 123), getattr(iso, name)) == expected


@pytest.mark.parametrize('name,expected', [
    ('DT_EXT_COMPLETE', '2024-02-29T03:04:05-03:30'),
    ('DT_BAS_COMPLETE', '20240229T030405-0330'),
    ('DT_EXT_ORD_COMPLETE', '2024-060T03:04:05-03:30'),
    ('DT_BAS_ORD_COMPLETE', '2024060T030405-0330'),
    ('DT_EXT_WEEK_COMPLETE', '2024-W09-4T03:04:05-03:30'),
    ('DT_BAS_WEEK_COMPLETE', '2024W094T030405-0330'),
])
def test_datetime_formats(name, expected):
    dt = datetime(2024, 2, 29, 3, 4, 5, tzinfo=iso.FixedOffset(-3, -30))
    assert iso.datetime_isoformat(dt, getattr(iso, name)) == expected


def test_default_and_custom_formats():
    dt = datetime(7, 2, 3, 4, 5, 6, 12, tzinfo=iso.UTC)
    assert iso.date_isoformat(dt) == '0007-02-03'
    assert iso.date_isoformat(dt, yeardigits=6) == '+000007-02-03'
    assert iso.strftime(dt, '%C/%Y', 6) == '+0000/+000007'
    assert iso.time_isoformat(dt) == '04:05:06Z'
    assert iso.datetime_isoformat(dt) == '0007-02-03T04:05:06Z'
    assert iso.strftime(dt, '%% %f %j %w %W %q') == '% 000012 034 6 05 %q'
    assert iso.time_isoformat(time(0)) == '00:00:00'
    assert iso.strftime(date(2024, 1, 1), '%j') == '001'
    assert iso.strftime(date(2024, 12, 31), '%j') == '366'


def test_duration_format_directives():
    value = iso.Duration(years=2001, months=3, days=16, hours=4, minutes=5,
                         seconds=6, microseconds=7)
    assert iso.strftime(value, '%Y %C %m %d %H %M %S %f %W %% %q') == (
        '2001 20 03 16 04 05 06 000007 02 % %q')
    assert iso.strftime(value, '%Y %C', 6) == '+002001 +0020'
    assert iso.duration_isoformat(value, iso.D_ALT_EXT) == 'P2001-03-16T04:05:06'
    assert iso.duration_isoformat(value, iso.D_ALT_BAS) == 'P20010316T040506'
    assert iso.duration_isoformat(timedelta(days=21), iso.D_WEEK) == 'P3W'
    assert iso.duration_isoformat(timedelta(days=-21), iso.D_WEEK) == '-P3W'


@pytest.mark.parametrize('text,minutes', [('+05:45', 345), ('-03:30', -210),
    ('+0530', 330), ('-0045', -45), ('+02', 120), ('-12', -720), ('Z', 0)])
def test_timezone_parser(text, minutes):
    zone = iso.parse_tzinfo(text)
    assert zone.utcoffset(None) == timedelta(minutes=minutes)
    assert zone.dst(None) == timedelta(0)
    assert zone.tzname(None) == ('UTC' if text == 'Z' else text)


@pytest.mark.parametrize('minutes,ext,basic,hour', [(345, '+05:45', '+0545', '+05'),
    (-210, '-03:30', '-0330', '-03'), (-30, '-00:30', '-0030', '-00'),
    (60, '+01:00', '+0100', '+01'), (0, 'Z', 'Z', 'Z')])
def test_timezone_formats(minutes, ext, basic, hour):
    dt = datetime(2024, 1, 1, tzinfo=iso.FixedOffset(0, minutes, 'test'))
    for fmt, expected in [(iso.TZ_EXT, ext), (iso.TZ_BAS, basic), (iso.TZ_HOUR, hour)]:
        assert iso.tz_isoformat(dt, fmt) == expected
    assert iso.tz_isoformat(dt) == ext


def test_timezone_special_cases():
    assert iso.parse_tzinfo('') is None
    assert iso.parse_tzinfo('Z') is iso.UTC
    assert build_tzinfo(None) is None
    assert build_tzinfo('') is None
    assert build_tzinfo('Z', '-', 12, 30) is iso.UTC
    assert build_tzinfo('west', '-', 2, 15).utcoffset(None) == -timedelta(hours=2, minutes=15)
    assert iso.tz_isoformat(datetime(2024, 1, 1)) == ''
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    assert repr(iso.FixedOffset(2, 30, 'east')) == "<FixedOffset 'east'>"
    with pytest.raises(ValueError):
        iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=iso.FixedOffset(1)), '%q')

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None
    assert iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=NoOffset())) == ''

    class HugeOffset(tzinfo):
        def utcoffset(self, dt):
            return timedelta(hours=100)
        def dst(self, dt):
            return timedelta(0)
    with pytest.raises(OverflowError):
        iso.tz_isoformat(datetime(2024, 1, 1, tzinfo=HugeOffset()))


@pytest.mark.parametrize('year', [1, 1899, 1900, 1996, 2000, 2001, 2024, 2100, 2400])
def test_month_lengths(year):
    for month in range(1, 13):
        assert max_days_in_month(year, month) == calendar.monthrange(year, month)[1]


@pytest.mark.parametrize('value', ['-25', '-12', '0', '1', '12', '13', '25', '2.5', '-0.5'])
@pytest.mark.parametrize('low,high', [(1, 13), (-3, 5), (0, 7)])
def test_floor_quotient_remainder(value, low, high):
    number = Decimal(value)
    q, r = fquotmod(number, low, high)
    assert low <= r < high
    assert number == q * (high - low) + r
    assert isinstance(q, int)


def parts(value):
    return value.years, value.months, value.tdelta


def test_duration_construction_and_state():
    d = iso.Duration(days=2, seconds=3, microseconds=4, milliseconds=5,
                     minutes=6, hours=7, weeks=8, months=1.2, years=2.3)
    expected = timedelta(days=2, seconds=3, microseconds=4, milliseconds=5,
                         minutes=6, hours=7, weeks=8)
    assert parts(d) == (Decimal('2.3'), Decimal('1.2'), expected)
    assert d.days == expected.days and d.seconds == expected.seconds
    assert d.microseconds == expected.microseconds
    assert d.total_seconds() == expected.total_seconds()
    assert parts(pickle.loads(pickle.dumps(d))) == parts(d)
    assert hash(d) == hash((expected, Decimal('1.2'), Decimal('2.3')))
    assert str(iso.Duration(years=2, months=1, days=3)) == '2 years, 1 month, 3 days, 0:00:00'
    assert str(iso.Duration(months=2)) == '2 months, 0:00:00'
    assert str(iso.Duration()) == '0:00:00'
    assert repr(iso.Duration(years=2, months=3, days=4, seconds=5, microseconds=6)) == (
        'isodate.duration.Duration(4, 5, 6, years=2, months=3)')


def test_duration_arithmetic():
    a = iso.Duration(years=2, months=3, days=4, seconds=5, microseconds=6)
    b = iso.Duration(years=5, months=7, days=8, seconds=9, microseconds=10)
    before = parts(a), parts(b)
    assert parts(a + b) == (7, 10, timedelta(days=12, seconds=14, microseconds=16))
    assert parts(a - b) == (-3, -4, timedelta(days=-4, seconds=-4, microseconds=-4))
    assert parts(-a) == (-2, -3, -a.tdelta)
    for factor in (-2, 0, 1, 3):
        assert parts(a * factor) == (2 * factor, 3 * factor, a.tdelta * factor)
        assert parts(factor * a) == parts(a * factor)
    delta = timedelta(days=3, seconds=4)
    assert parts(a + delta) == (2, 3, a.tdelta + delta)
    assert parts(delta + a) == parts(a + delta)
    assert parts(a - delta) == (2, 3, a.tdelta - delta)
    assert parts(delta - a) == (-2, -3, delta - a.tdelta)
    assert (parts(a), parts(b)) == before
    assert a.__add__(object()) is NotImplemented
    assert a.__sub__(object()) is NotImplemented
    assert a.__rsub__(object()) is NotImplemented
    assert a.__mul__(1.5) is NotImplemented
    with pytest.raises(TypeError):
        _ = a + 1
    with pytest.raises(TypeError):
        _ = a * 1.5


@pytest.mark.parametrize('a,b,equal', [
    (iso.Duration(years=1), iso.Duration(months=12), True),
    (iso.Duration(years=1, months=2, days=3), iso.Duration(months=14, days=3), True),
    (iso.Duration(years=1), iso.Duration(years=1, seconds=1), False),
    (iso.Duration(months=1), iso.Duration(months=2), False),
    (iso.Duration(days=2), timedelta(days=2), True),
    (iso.Duration(days=2), timedelta(days=3), False),
    (iso.Duration(years=1), timedelta(0), False),
    (iso.Duration(months=1), timedelta(0), False),
    (iso.Duration(), None, False),
])
def test_duration_equality(a, b, equal):
    assert (a == b) is equal
    assert (a != b) is (not equal)
    assert (b == a) is equal
    assert (b != a) is (not equal)


@pytest.mark.parametrize('start,years,months,days,expected', [
    (date(2024, 1, 31), 0, 1, 0, date(2024, 2, 29)),
    (date(2023, 1, 31), 0, 1, 0, date(2023, 2, 28)),
    (date(2024, 2, 29), 1, 0, 0, date(2025, 2, 28)),
    (date(2000, 2, 29), 100, 0, 0, date(2100, 2, 28)),
    (date(2024, 3, 31), 0, -1, 0, date(2024, 2, 29)),
    (date(2024, 1, 15), -1, -14, 2, date(2021, 11, 17)),
    (date(2023, 12, 31), 1, 14, 1, date(2026, 3, 1)),
    (date(2024, 5, 30), 0, 1, 0, date(2024, 6, 30)),
    (date(2024, 5, 31), 0, 1, 0, date(2024, 6, 30)),
])
def test_calendar_addition(start, years, months, days, expected):
    d = iso.Duration(years=years, months=months, days=days)
    assert start + d == expected
    assert d + start == expected
    # Subtraction of the negated duration applies the same calendar adjustment.
    assert start - (-d) == expected
    assert d.totimedelta(start=start) == expected - start


def test_datetime_duration_and_conversion():
    start = datetime(2024, 1, 31, 23, 30, 40, 123456, iso.UTC)
    d = iso.Duration(months=1, hours=2, seconds=3, microseconds=4)
    assert start + d == datetime(2024, 3, 1, 1, 30, 43, 123460, iso.UTC)
    assert start - d == datetime(2023, 12, 31, 21, 30, 37, 123452, iso.UTC)
    assert (start + d).tzinfo is iso.UTC
    assert iso.Duration(months=1).totimedelta(start=date(2024, 2, 1)) == timedelta(days=29)
    assert iso.Duration(months=1).totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    assert iso.Duration(months=1).totimedelta(end=date(2024, 3, 1)) == timedelta(days=29)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=start, end=start)
    # date subtraction must retain timedelta's behavior for fractional days.
    assert date(2024, 3, 10) - iso.Duration(seconds=1) == date(2024, 3, 10)
    assert date(2024, 3, 10) + iso.Duration(seconds=-1) == date(2024, 3, 9)


@pytest.mark.parametrize('kwargs', [{'years': 1.5}, {'months': 1.5}, {'years': 1.5, 'months': 1.5}])
def test_fractional_calendar_arithmetic_rejected(kwargs):
    d = iso.Duration(**kwargs)
    with pytest.raises(ValueError):
        _ = date(2024, 1, 1) + d
    with pytest.raises(ValueError):
        _ = date(2024, 1, 1) - d


def test_local_timezone_branches(monkeypatch):
    zone = zones.LocalTimezone()
    standard, daylight = timedelta(hours=2), timedelta(hours=3)
    monkeypatch.setattr(zones, 'STDOFFSET', standard)
    monkeypatch.setattr(zones, 'DSTOFFSET', daylight)
    monkeypatch.setattr(zones, 'DSTDIFF', daylight - standard)
    monkeypatch.setattr(zones.time, 'tzname', ('standard', 'summer'))
    for flag in (False, True):
        monkeypatch.setattr(zone, '_isdst', lambda dt, flag=flag: flag)
        assert zone.utcoffset(None) == (daylight if flag else standard)
        assert zone.dst(None) == (timedelta(hours=1) if flag else timedelta(0))
        assert zone.tzname(None) == ('summer' if flag else 'standard')


def test_local_dst_detection(monkeypatch):
    zone = zones.LocalTimezone()
    dt = datetime(2024, 7, 14, 12, 34, 56)
    seen = []
    def mktime(fields):
        seen.append(fields)
        return 12345
    monkeypatch.setattr(zones.time, 'mktime', mktime)
    for flag in (-1, 0, 1):
        def localtime(stamp):
            assert stamp == 12345
            return zones.time.struct_time((2024, 7, 14, 12, 34, 56, 6, 196, flag))
        monkeypatch.setattr(zones.time, 'localtime', localtime)
        assert zone._isdst(dt) is (flag > 0)
    assert seen == [(2024, 7, 14, 12, 34, 56, 6, 0, -1)] * 3
    with pytest.raises(Exception, match='None'):
        zone._isdst(None)


def test_parser_cache_rebuild(monkeypatch):
    import isodate.isodates as dates
    import isodate.isotime as times
    monkeypatch.setattr(dates, 'DATE_REGEX_CACHE', {})
    monkeypatch.setattr(times, 'TIME_REGEX_CACHE', [])
    normal = dates.build_date_regexps()
    assert len(normal) == 12
    assert dates.build_date_regexps() is normal
    expanded = dates.build_date_regexps(4, True)
    assert expanded is not normal
    assert iso.parse_date('+2024-03-04', expanded=True) == date(2024, 3, 4)
    assert iso.parse_date('2024-03-04') == date(2024, 3, 4)
    six = dates.build_date_regexps(6)
    assert dates.build_date_regexps(6, True) is six
    assert iso.parse_date('+002024-03-04', yeardigits=6) == date(2024, 3, 4)
    clock = times.build_time_regexps()
    assert len(clock) == 5
    assert times.build_time_regexps() is clock
    assert len(clock) == 5
    assert iso.parse_time('12:34:56') == time(12, 34, 56)


def test_date_defaults_apply_only_to_missing_fields():
    assert iso.parse_date('2024-00-02', defaultmonth=5) == date(2024, 5, 2)
    assert iso.parse_date('202400', defaultmonth=5, defaultday=3) == date(2024, 5, 3)
    assert iso.parse_date('2024-W01', defaultday=7, defaultmonth=8) == date(2024, 1, 1)
    assert iso.parse_date('2024-060', defaultday=7, defaultmonth=8) == date(2024, 2, 29)


def test_timezone_nonstandard_offsets():
    class Offset(tzinfo):
        def __init__(self, delta, dst):
            self.delta, self.adjustment = delta, dst
        def utcoffset(self, dt):
            return self.delta
        def dst(self, dt):
            return self.adjustment
    # A zero total offset with a DST adjustment is formatted numerically.
    dt = datetime(2024, 1, 1, tzinfo=Offset(timedelta(0), timedelta(hours=1)))
    assert iso.tz_isoformat(dt) == '+00:00'
    for hours in (-99, 99):
        dt = datetime(2024, 1, 1, tzinfo=Offset(timedelta(hours=hours), timedelta(0)))
        assert iso.tz_isoformat(dt) == ('-' if hours < 0 else '+') + '99:00'
    dt = datetime(2024, 1, 1, tzinfo=Offset(timedelta(hours=1, minutes=2, seconds=3), None))
    assert iso.tz_isoformat(dt) == '+01:02'


@pytest.mark.parametrize('value,expected', [
    (iso.Duration(years=-1, months=2), '-P1Y2M'),
    (iso.Duration(years=1, months=-2), '-P1Y2M'),
    (iso.Duration(years=1, seconds=-1), '-P1YT1S'),
    (iso.Duration(years=0, months=0), 'P0D'),
])
def test_mixed_sign_duration_format(value, expected):
    # The formatter chooses a leading minus if any component is negative.
    assert iso.duration_isoformat(value) == expected


def test_duration_near_day_boundaries():
    assert iso.duration_isoformat(timedelta(microseconds=-1)) == '-PT0.000001S'
    assert iso.duration_isoformat(timedelta(days=1, microseconds=-1)) == 'PT23H59M59.999999S'
    assert iso.duration_isoformat(timedelta(seconds=60)) == 'PT1M'
    assert iso.duration_isoformat(timedelta(seconds=3600)) == 'PT1H'
    assert iso.duration_isoformat(timedelta(seconds=86400)) == 'P1D'
    assert str(iso.Duration(months=Decimal('0.5'))) == '0.5 month, 0:00:00'
    assert str(iso.Duration(months=Decimal('1.5'))) == '1.5 months, 0:00:00'
