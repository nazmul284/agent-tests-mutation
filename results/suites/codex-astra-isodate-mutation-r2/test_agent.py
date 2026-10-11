"""Behavioral tests derived from the parsers and calendar arithmetic."""
import calendar
import math
import pickle
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest
import isodate as iso
from isodate.duration import Duration, fquotmod, max_days_in_month
from isodate.isotzinfo import build_tzinfo


@pytest.mark.parametrize('text,expected', [
    ('20240229', date(2024,2,29)), ('2024-02-29', date(2024,2,29)),
    ('2024060', date(2024,2,29)), ('2024-060', date(2024,2,29)),
    ('2023-365', date(2023,12,31)), ('2024-366', date(2024,12,31)),
    ('2024-02', date(2024,2,1)), ('202402', date(2024,2,1)),
    ('2024', date(2024,1,1)), ('20', date(2001,1,1)), ('00', date(1,1,1)),
])
def test_dates(text, expected):
    assert iso.parse_date(text) == expected


@pytest.mark.parametrize('year', range(2015,2026))
@pytest.mark.parametrize('week,day', [(1,1),(1,7),(2,3),(52,7)])
def test_week_dates(year, week, day):
    expected = date.fromisocalendar(year, week, day)
    assert iso.parse_date(f'{year}-W{week:02}-{day}') == expected
    assert iso.parse_date(f'{year}W{week:02}{day}') == expected
    if day == 1:
        assert iso.parse_date(f'{year}-W{week:02}') == expected
        assert iso.parse_date(f'{year}W{week:02}') == expected


def test_date_options():
    for text in ['2024', '20']:
        assert iso.parse_date(text, defaultmonth=5, defaultday=17) == date(2024 if text == '2024' else 2001,5,17)
    assert iso.parse_date('2024-03', defaultday=17) == date(2024,3,17)
    assert iso.parse_date('2024-00-03', defaultmonth=5) == date(2024,5,3)
    assert iso.parse_date('+002024-02-29', yeardigits=6) == date(2024,2,29)
    assert iso.parse_date('+2024-02-29', expanded=True) == date(2024,2,29)
    assert iso.parse_date('+0020', yeardigits=6) == date(2001,1,1)
    for text, kwargs in [('002024-02-29', {'yeardigits':6}), ('2024', {'expanded':True}), ('+2024', {})]:
        with pytest.raises(iso.ISO8601Error):
            iso.parse_date(text, **kwargs)
    with pytest.raises(ValueError):
        iso.parse_date('-002024-01-01', yeardigits=6)


@pytest.mark.parametrize('text', ['', '2024-2-01', '2024/02/01', 'x2024-02-01', '2024-02-01x', '2024-02-01\n'])
def test_bad_date_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_date(text)


@pytest.mark.parametrize('text', ['2023-02-29','2024-13-01','2024-01-32','0000-01-01'])
def test_bad_calendar_date(text):
    with pytest.raises(ValueError):
        iso.parse_date(text)


@pytest.mark.parametrize('text,expected', [
    ('12', time(12)), ('T12', time(12)), ('12.125', time(12,7,30)),
    ('12,000001', time(12,0,0,3600)), ('1234', time(12,34)),
    ('12:34.125', time(12,34,7,500000)), ('1234,000001', time(12,34,0,60)),
    ('12:34:56.1234569', time(12,34,56,123456)),
    ('123456,0000099', time(12,34,56,9)), ('T00:00:00', time(0)),
    ('23:59:59.9999999', time(23,59,59,999999)),
])
def test_times(text, expected):
    assert iso.parse_time(text) == expected


@pytest.mark.parametrize('suffix,minutes', [('Z',0),('+00',0),('+05',300),('+0530',330),('+05:45',345),('-03:30',-210),('-0045',-45)])
def test_time_zones(suffix, minutes):
    parsed = iso.parse_time('T12:34:56.123456' + suffix)
    assert parsed.replace(tzinfo=None) == time(12,34,56,123456)
    assert parsed.utcoffset() == timedelta(minutes=minutes)
    tz = iso.parse_tzinfo(suffix)
    assert tz.utcoffset(None) == timedelta(minutes=minutes)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == ('UTC' if suffix == 'Z' else suffix)


@pytest.mark.parametrize('text', ['', '1', '12:3', '12:34:', '12:34:56x','12:34:56\n','12:34:56+1'])
def test_bad_time_syntax(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_time(text)


@pytest.mark.parametrize('text', ['24:00:00','12:60:00','12:00:60'])
def test_bad_time_values(text):
    with pytest.raises(ValueError):
        iso.parse_time(text)


def test_datetime():
    assert iso.parse_datetime('2024-060T12:34:56.123456-03:30') == datetime(2024,2,29,12,34,56,123456,timezone(-timedelta(hours=3,minutes=30)))
    assert iso.parse_datetime('2024W094T1234') == datetime(2024,2,29,12,34)
    for text in ['2024-01-01 12:00', '2024-01-01TT12:00', 'T12:00', '2024-01-01T']:
        with pytest.raises(iso.ISO8601Error):
            iso.parse_datetime(text)


@pytest.mark.parametrize('text,expected', [
    ('P0D',timedelta()), ('PT0S',timedelta()), ('P2W',timedelta(days=14)),
    ('P1.5W',timedelta(days=10,hours=12)), ('P2DT3H4M5.123456S',timedelta(days=2,hours=3,minutes=4,seconds=5,microseconds=123456)),
    ('PT1,5H',timedelta(minutes=90)), ('PT2.5M',timedelta(seconds=150)),
    ('P1,25D',timedelta(days=1,hours=6)), ('P1W2DT3H',timedelta(days=9,hours=3)),
    ('P0Y0M3D',timedelta(days=3)),
])
def test_timedelta_parse(text, expected):
    for prefix, sign in [('',1),('+',1),('-',-1)]:
        result = iso.parse_duration(prefix + text)
        assert type(result) is timedelta
        assert result == expected * sign
        result = iso.parse_duration(prefix + text, as_timedelta_if_possible=False)
        assert type(result) is Duration
        assert result.years == result.months == 0
        assert result.tdelta == expected * sign


@pytest.mark.parametrize('text,years,months,delta', [
    ('P1Y',1,0,timedelta()), ('P2M',0,2,timedelta()),
    ('P1.25Y2,5M3W4DT5H6M7.25S',Decimal('1.25'),Decimal('2.5'),timedelta(days=25,hours=5,minutes=6,seconds=7.25)),
    ('P0001-02-03T04:05:06',1,2,timedelta(days=3,hours=4,minutes=5,seconds=6)),
    ('P00010203T040506',1,2,timedelta(days=3,hours=4,minutes=5,seconds=6)),
])
def test_calendar_duration_parse(text, years, months, delta):
    result = iso.parse_duration(text)
    assert type(result) is Duration
    assert (result.years,result.months,result.tdelta) == (years,months,delta)
    if '-' not in text and '0001' not in text:
        negative = iso.parse_duration('-'+text)
        assert (negative.years,negative.months,negative.tdelta) == (-years,-months,-delta)


@pytest.mark.parametrize('text', ['', 'P','1D','P-1D','PT1Q','P1S','P1Dgarbage'])
def test_bad_duration(text):
    with pytest.raises(iso.ISO8601Error):
        iso.parse_duration(text)


@pytest.mark.parametrize('value', [None, 1, b'P1D', timedelta(days=1)])
def test_duration_input_type(value):
    with pytest.raises(TypeError):
        iso.parse_duration(value)


@pytest.mark.parametrize('value,expected', [
    (timedelta(), 'P0D'), (timedelta(days=2),'P2D'),
    (timedelta(hours=3),'PT3H'), (timedelta(minutes=4),'PT4M'),
    (timedelta(seconds=5),'PT5S'), (timedelta(microseconds=1),'PT0.000001S'),
    (timedelta(seconds=1,microseconds=120000),'PT1.12S'),
    (timedelta(days=2,hours=3,minutes=4,seconds=5,microseconds=6),'P2DT3H4M5.000006S'),
    (Duration(years=2,months=3,days=4,hours=5),'P2Y3M4DT5H'),
    (Duration(years=Decimal('1.5')), 'P1.5Y'), (Duration(months=2),'P2M'),
])
def test_duration_format(value, expected):
    assert iso.duration_isoformat(value) == expected
    if value != timedelta():
        assert iso.duration_isoformat(-value) == '-' + expected
    assert iso.parse_duration(expected) == value


def test_format_directives():
    dt = datetime(2024,2,29,13,14,15,1234,iso.FixedOffset(-3,-30,'west'))
    assert iso.strftime(dt, '%Y %C %m %d %j %W %w %H %M %S %f %z %Z %h %% %Q') == '2024 20 02 29 060 09 4 13 14 15 001234 -0330 -03:30 -03 % %Q'
    assert iso.strftime(date(7,1,1), '%Y %C %j') == '0007 00 001'
    assert iso.strftime(dt, '%Y %C', 6) == '+002024 +0020'
    assert iso.date_isoformat(dt) == '2024-02-29'
    assert iso.date_isoformat(dt, yeardigits=6) == '+002024-02-29'
    assert iso.time_isoformat(dt) == '13:14:15-03:30'
    assert iso.datetime_isoformat(dt) == '2024-02-29T13:14:15-03:30'
    d = Duration(years=123,months=4,days=15,hours=13,minutes=14,seconds=15,microseconds=1234)
    assert iso.strftime(d, '%Y %C %m %d %W %H %M %S %f %% %Q') == '0123 01 04 15 02 13 14 15 001234 % %Q'
    assert iso.strftime(d, '%Y %C',6) == '+000123 +0001'
    assert iso.duration_isoformat(timedelta(days=21),iso.D_WEEK) == 'P3W'
    assert iso.duration_isoformat(Duration(years=1,months=2,days=3,hours=4,minutes=5,seconds=6), iso.D_ALT_EXT) == 'P0001-02-03T04:05:06'


@pytest.mark.parametrize('constant,expected', [
    ('DATE_BAS_COMPLETE','20240229'),('DATE_EXT_COMPLETE','2024-02-29'),
    ('DATE_BAS_WEEK_COMPLETE','2024W094'),('DATE_EXT_WEEK_COMPLETE','2024-W09-4'),
    ('DATE_BAS_ORD_COMPLETE','2024060'),('DATE_EXT_ORD_COMPLETE','2024-060'),
    ('DATE_BAS_WEEK','2024W09'),('DATE_EXT_WEEK','2024-W09'),
    ('DATE_BAS_MONTH','202402'),('DATE_EXT_MONTH','2024-02'),('DATE_YEAR','2024'),('DATE_CENTURY','20'),
    ('TIME_BAS_COMPLETE','131415'),('TIME_EXT_COMPLETE','13:14:15'),('TIME_BAS_MINUTE','1314'),('TIME_EXT_MINUTE','13:14'),('TIME_HOUR','13'),
    ('DT_BAS_COMPLETE','20240229T131415Z'),('DT_EXT_COMPLETE','2024-02-29T13:14:15Z'),
    ('DT_BAS_ORD_COMPLETE','2024060T131415Z'),('DT_EXT_ORD_COMPLETE','2024-060T13:14:15Z'),
    ('DT_BAS_WEEK_COMPLETE','2024W094T131415Z'),('DT_EXT_WEEK_COMPLETE','2024-W09-4T13:14:15Z'),
])
def test_format_constants(constant,expected):
    assert iso.strftime(datetime(2024,2,29,13,14,15,tzinfo=iso.UTC),getattr(iso,constant)) == expected


def parts(d):
    return d.years, d.months, d.tdelta


def test_duration_operators():
    a = Duration(years=2,months=3,days=4,seconds=5,microseconds=6)
    b = Duration(years=1,months=2,days=3,seconds=4,microseconds=5)
    assert parts(a+b) == (3,5,timedelta(days=7,seconds=9,microseconds=11))
    assert parts(a-b) == (1,1,timedelta(days=1,seconds=1,microseconds=1))
    assert parts(-a) == (-2,-3,-timedelta(days=4,seconds=5,microseconds=6))
    td = timedelta(days=2,seconds=7)
    assert parts(a+td) == parts(td+a) == (2,3,a.tdelta+td)
    assert parts(a-td) == (2,3,a.tdelta-td)
    assert parts(td-a) == (-2,-3,td-a.tdelta)
    for n in [-2,0,3]:
        assert parts(a*n) == parts(n*a) == (2*n,3*n,a.tdelta*n)
    assert parts(a) == (2,3,timedelta(days=4,seconds=5,microseconds=6))
    assert a.days == 4 and a.seconds == 5 and a.microseconds == 6
    assert a.total_seconds() == a.tdelta.total_seconds()
    for operation in [lambda:a+1,lambda:1+a,lambda:a-1,lambda:1-a,lambda:a*1.5,lambda:1.5*a]:
        with pytest.raises(TypeError):
            operation()


@pytest.mark.parametrize('a,b,equal', [
    (Duration(years=1),Duration(months=12),True),
    (Duration(years=1,days=1),Duration(months=12,days=2),False),
    (Duration(years=1),Duration(months=11),False),
    (Duration(days=2),timedelta(days=2),True),
    (Duration(days=2),timedelta(days=3),False),
    (Duration(years=1),timedelta(),False),
    (Duration(months=1),timedelta(),False),
    (Duration(),object(),False),
])
def test_duration_equality(a,b,equal):
    assert (a == b) is equal
    assert (a != b) is (not equal)


@pytest.mark.parametrize('start,years,months,days,expected', [
    (date(2023,1,31),0,1,0,date(2023,2,28)),
    (date(2024,1,31),0,1,0,date(2024,2,29)),
    (date(2024,2,29),1,0,0,date(2025,2,28)),
    (date(2023,12,15),1,14,2,date(2026,2,17)),
    (date(2024,1,31),0,-1,0,date(2023,12,31)),
    (date(2024,3,31),0,-13,1,date(2023,3,1)),
    (date(2024,1,28),0,1,0,date(2024,2,28)),
])
def test_calendar_arithmetic(start,years,months,days,expected):
    d = Duration(years=years,months=months,days=days)
    assert start+d == d+start == expected
    assert start-(-d) == expected
    assert d.totimedelta(start=start) == expected-start
    assert (-d).totimedelta(end=start) == start-expected


def test_duration_datetime_and_conversion():
    dt = datetime(2024,1,31,23,59,58,123,tzinfo=iso.UTC)
    d = Duration(months=1,seconds=3,microseconds=7)
    assert dt+d == datetime(2024,3,1,0,0,1,130,tzinfo=iso.UTC)
    assert dt-d == datetime(2023,12,31,23,59,55,116,tzinfo=iso.UTC)
    for kwargs in [{}, {'start':dt,'end':dt}]:
        with pytest.raises(ValueError):
            d.totimedelta(**kwargs)
    for d in [Duration(years=.5),Duration(months=Decimal('1.5'))]:
        for op in [lambda:dt+d,lambda:dt-d]:
            with pytest.raises(ValueError):
                op()
    all_units = Duration(days=1,seconds=2,microseconds=3,milliseconds=4,minutes=5,hours=6,weeks=7,months=.1,years=.2)
    assert parts(all_units) == (Decimal('.2'),Decimal('.1'),timedelta(days=50,hours=6,minutes=5,seconds=2,microseconds=4003))
    assert parts(pickle.loads(pickle.dumps(all_units))) == parts(all_units)
    assert hash(all_units) == hash((all_units.tdelta,all_units.months,all_units.years))
    assert str(Duration(years=2,months=1,days=3)) == '2 years, 1 month, 3 days, 0:00:00'
    assert str(Duration(months=2)) == '2 months, 0:00:00'
    assert repr(Duration(years=2,months=3,days=4,seconds=5,microseconds=6)) == 'isodate.duration.Duration(4, 5, 6, years=2, months=3)'


@pytest.mark.parametrize('year',[1900,2000,2023,2024,2100,2400])
@pytest.mark.parametrize('month',range(1,13))
def test_month_lengths(year,month):
    assert max_days_in_month(year,month) == calendar.monthrange(year,month)[1]


@pytest.mark.parametrize('value',[-25,-12,-1,0,1,12,13,24,25,Decimal('13.5')])
def test_floor_divmod(value):
    q,r = fquotmod(Decimal(value),1,13)
    assert q == math.floor((Decimal(value) - 1) / 12)
    assert Decimal(value) == q*12+r
    assert 1 <= r < 13


def test_timezone_objects():
    assert build_tzinfo(None) is None
    assert build_tzinfo('') is None
    assert iso.parse_tzinfo('') is None
    assert build_tzinfo('Z') is iso.UTC
    assert pickle.loads(pickle.dumps(iso.UTC)) is iso.UTC
    assert iso.UTC.utcoffset(None) == iso.UTC.dst(None) == timedelta()
    assert iso.UTC.tzname(None) == 'UTC'
    zone = iso.FixedOffset(5,45,'Nepal')
    assert zone.utcoffset(None) == timedelta(hours=5,minutes=45)
    assert zone.dst(None) == timedelta()
    assert zone.tzname(None) == 'Nepal'
    assert repr(zone) == "<FixedOffset 'Nepal'>"
    for zone, expected in [(None,''),(iso.UTC,'Z'),(iso.FixedOffset(),'Z'),(iso.FixedOffset(-3,-30),' -03:30'.strip()),(iso.FixedOffset(0,-30),'-00:30'),(iso.FixedOffset(5,45),'+05:45')]:
        dt = datetime(2024,1,1,tzinfo=zone)
        assert iso.tz_isoformat(dt) == expected
        if expected not in ('','Z'):
            assert iso.tz_isoformat(dt,'%z') == expected.replace(':','')
            assert iso.tz_isoformat(dt,'%h') == expected[:3]
    with pytest.raises(ValueError):
        iso.tz_isoformat(datetime(2024,1,1,tzinfo=iso.FixedOffset(1)), '%bad')


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None


def test_unspecified_offset():
    assert iso.tz_isoformat(datetime(2024,1,1,tzinfo=NoOffset())) == ''


@pytest.mark.parametrize('isdst', [-1, 0, 1])
def test_local_timezone(monkeypatch, isdst):
    import importlib
    import time as stdtime
    module = importlib.import_module('isodate.tzinfo')
    standard = timedelta(hours=2)
    daylight = timedelta(hours=3)
    monkeypatch.setattr(module, 'STDOFFSET', standard)
    monkeypatch.setattr(module, 'DSTOFFSET', daylight)
    monkeypatch.setattr(module, 'DSTDIFF', timedelta(hours=1))
    monkeypatch.setattr(module.time, 'tzname', ('standard', 'summer'))
    dt = datetime(2024, 7, 14, 12, 34, 56)
    def mktime(fields):
        assert fields == (2024, 7, 14, 12, 34, 56, 6, 0, -1)
        return 12345
    def localtime(stamp):
        assert stamp == 12345
        return stdtime.struct_time((2024, 7, 14, 12, 34, 56, 6, 196, isdst))
    monkeypatch.setattr(module.time, 'mktime', mktime)
    monkeypatch.setattr(module.time, 'localtime', localtime)
    zone = module.LocalTimezone()
    assert zone._isdst(dt) is (isdst > 0)
    assert zone.utcoffset(dt) == (daylight if isdst > 0 else standard)
    assert zone.dst(dt) == (timedelta(hours=1) if isdst > 0 else timedelta())
    assert zone.tzname(dt) == ('summer' if isdst > 0 else 'standard')
    with pytest.raises(Exception, match='datetime object dt was None'):
        zone._isdst(None)


def test_regex_caches():
    from isodate.isodates import build_date_regexps
    from isodate.isotime import build_time_regexps
    assert build_date_regexps() is build_date_regexps()
    assert build_date_regexps(6, False) is build_date_regexps(6, True)
    assert build_date_regexps(4, False) is not build_date_regexps(4, True)
    assert build_time_regexps() is build_time_regexps()
    assert len(build_date_regexps()) == 12
    assert len(build_time_regexps()) == 5
