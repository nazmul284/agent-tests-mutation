import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
import parse


@pytest.mark.parametrize('spec,text,value', [
    ('d','0',0), ('d','-19',-19), ('d','+23',23), ('d',' 8',8),
    ('d','0XAf',175), ('d','-0b101',-5), ('d','+0o17',15),
    (',d','1,234,567',1234567), ('_d','1_234',1234),
    ('b','0B101',5), ('b','-101',-5), ('o','0O71',57),
    ('x','-0xAF',-175), ('X','FF',255), ('n','1,234,567',1234567),
    ('n','-12.345',-12345), ('%','12.5%',.125), ('%','-2%',-.02),
    ('f','-.25',-.25), ('f','+1.25',1.25), ('.0f','-20',-20.),
    ('F','1.23456789',Decimal('1.23456789')), ('.0F','23',Decimal(23)),
    ('e','1.25e-2',.0125), ('E','-2.5E+2',-250.),
    ('g','42',42.), ('g','-2.5e3',-2500.), ('G','3E-2',.03),
])
def test_numeric_conversions_and_capture_indices(spec, text, value):
    result = parse.parse('{}|{:'+spec+'}|{value:'+spec+'}|{:d}', 'a|'+text+'|'+text+'|7')
    assert result.fixed == ('a', value, 7)
    assert result.named == {'value': value}
    assert type(result[1]) is type(value)
    assert result.spans == {0:(0,1), 1:(2,2+len(text)), 'value':(3+len(text),3+2*len(text)), 2:(4+2*len(text),5+2*len(text))}


@pytest.mark.parametrize('spec,text', [('b','2'),('o','8'),('x','zz'),('d','abc'),('d','1.5'),('n','1234'),('n','1,23'),('f','12'),('.0f','12.5'),('e','12'),('%','10'),('l','abc1'),('w','a-b'),('D','12'),('S','a b')])
def test_invalid_typed_inputs(spec,text):
    assert parse.parse('{:'+spec+'}',text) is None


@pytest.mark.parametrize('spec', ['f','F','e','E','g','G'])
@pytest.mark.parametrize('text', ['nan','NAN','inf','-inf','+INF'])
def test_special_numbers(spec,text):
    value = parse.parse('{:'+spec+'}',text)[0]
    if 'nan' in text.lower():
        assert math.isnan(value)
    else:
        assert value == (float('-inf') if text.startswith('-') else float('inf'))


@pytest.mark.parametrize('spec,text', [('w','abc_123'),('W','!? '),('s',' \t\n'),('S','abc!?'),('D','abc!'),('l','AbCd')])
def test_character_classes(spec,text):
    assert parse.parse('{:'+spec+'}',text).fixed == (text,)


def test_literals_escaping_and_full_match():
    literal = '.?[]()!*+^$|' + chr(92)
    assert parse.parse(literal,literal).fixed == ()
    assert parse.parse('{{{x}}}', '{hello}')['x'] == 'hello'
    assert parse.parse('abc','ABC') is not None
    assert parse.parse('abc','ABC',case_sensitive=True) is None
    for text in ('xabc','abcx','abc\n'):
        assert parse.parse('abc',text) is None
    assert parse.parse('{}','') is None
    assert parse.parse('{}:{}','a:b:c').fixed == ('a','b:c')
    assert parse.parse('{}','a\nb')[0] == 'a\nb'
    assert parse.parse('','').named == {}


@pytest.mark.parametrize('spec,text,expected,span', [
    ('>','  abc','abc',(2,5)), ('<','abc  ','abc',(0,3)),
    ('^','  abc  ','abc',(2,5)), ('.>','...abc','abc',(3,6)),
    ('*<','abc***','abc',(0,3)), ('_^','__abc__','abc',(2,5)),
    ('5d','   12',12,(3,5)), ('05d','00012',12,(0,5)),
    ('=d','-0012',-12,(0,5)), ('0=6d','+00012',12,(0,6)),
    ('3','abc','abc',(0,3)), ('.3','ab','ab',(0,2)),
    ('2.4','abc','abc',(0,3)), ('+08.2f','00012.50',12.5,(0,8)),
])
def test_padding_and_lengths(spec,text,expected,span):
    result = parse.parse('{:'+spec+'}',text)
    assert result[0] == expected
    assert result.spans[0] == span


@pytest.mark.parametrize('spec,text', [('3','ab'),('.3','abcd'),('2.4','a'),('2.4','abcde'),('2d','123')])
def test_width_precision_limits(spec,text):
    assert parse.parse('{:'+spec+'}',text) is None


def test_names_repetition_nested_and_collisions():
    r = parse.parse('{a.b}|{a_b}|{a-b}|{a[b][c]:d}|{a[b][d]}', 'one|two|three|4|five')
    assert r.named == {'a.b':'one','a_b':'two','a-b':'three','a':{'b':{'c':4,'d':'five'}}}
    assert r.spans['a[b][c]'] == (14,15)
    assert parse.parse('{x:d}/{x:d}/{:d}', '12/12/3').fixed == (3,)
    assert parse.parse('{x:d}/{x:d}', '12/13') is None
    assert parse.parse('{x}/{x}', 'abc/ABC')['x'] == 'abc'
    assert parse.parse('{x}/{x}', 'abc/ABC',case_sensitive=True) is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile('{x:d}/{x:f}')
    assert parse.parse('{2}/{0}/{}', 'a/b/c').fixed == ('a','b','c')


def test_result_protocol_and_parser_metadata():
    p = parse.compile('{}:{name:d}:{}')
    assert p.format == '{}:{name:d}:{}'
    assert p.named_fields == ['name']
    assert p.fixed_fields == [0,2]
    p.named_fields.append('bad')
    p.fixed_fields.clear()
    assert p.named_fields == ['name'] and p.fixed_fields == [0,2]
    r = p.parse('a:42:b')
    assert r[:] == ('a','b') and r[-1] == 'b'
    assert r['name'] == 42 and 'name' in r and 'missing' not in r
    with pytest.raises(KeyError):
        r['missing']
    with pytest.raises(IndexError):
        r[2]
    assert repr(r) == "<Result ('a', 'b') {'name': 42}>"
    assert repr(parse.compile('x'*20)) == "<Parser '"+'x'*20+"'>"
    assert repr(parse.compile('x'*21)) == "<Parser '"+'x'*17+"...'>"
    assert p.parse('c:9:d').fixed == ('c','d')


def test_search_findall_bounds_and_lazy_conversion():
    calls = []
    @parse.with_pattern(r'(\d+)',regex_group_count=1)
    def twice(s):
        calls.append(s)
        return int(s)*2
    assert twice.pattern == r'(\d+)' and twice.regex_group_count == 1
    types = {'twice':twice}
    text = 'x A1! a22! A333! z'
    fmt = 'A{:twice}!'
    m = parse.search(fmt,text,extra_types=types,evaluate_result=False)
    assert isinstance(m,parse.Match) and calls == []
    assert m.match.span() == (2,5)
    assert m.evaluate_result().fixed == (2,) and calls == ['1']
    assert parse.search(fmt,text,pos=5,endpos=10,extra_types=types).fixed == (44,)
    assert parse.search(fmt,text,pos=5,endpos=9,extra_types=types) is None
    assert parse.search(fmt,text,pos=5,case_sensitive=True,extra_types=types).fixed == (666,)
    it = parse.findall(fmt,text,pos=5,endpos=16,extra_types=types,evaluate_result=False)
    assert iter(it) is it
    assert next(it).evaluate_result()[0] == 44
    assert it.next().evaluate_result()[0] == 666
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)
    assert [r[0] for r in parse.findall(fmt,text,extra_types=types,case_sensitive=True)] == [2,666]
    assert list(parse.findall('NO',text)) == []
    p = parse.compile('{:twice}|{:d}',extra_types=types)
    m = p.parse('12|7',evaluate_result=False)
    assert m.parser is p and m.evaluate_result().fixed == (24,7)
    assert parse.parse('{:twice}','2',extra_types=types,evaluate_result=False).evaluate_result()[0] == 4


def test_custom_types_and_errors():
    @parse.with_pattern('[a-z]+')
    def upper(s):
        return s.upper()
    assert upper.regex_group_count is None
    assert parse.parse('{:word}|{:d}','hello|9',{'word':upper}).fixed == ('HELLO',9)
    assert parse.parse('{x:raw}', 'abc', {'raw':lambda s:list(s)})['x'] == ['a','b','c']
    assert parse.parse('{:d}','abc',{'d':upper})[0] == 'ABC'
    with pytest.raises(ValueError,match='not recognised'):
        parse.compile('{:unknown}')
    with pytest.raises(ValueError):
        parse.parse('{:bad}','anything',{'bad':int})


@pytest.mark.parametrize('spec,text,expected', [
    ('ti','2024-02-29',datetime(2024,2,29)),
    ('ti','2024-02-29T03:04:05.1234567Z',datetime(2024,2,29,3,4,5,123456,tzinfo=parse.FixedTzOffset(0,'UTC'))),
    ('ti','2024-02-29 03:04+0530',datetime(2024,2,29,3,4,tzinfo=parse.FixedTzOffset(330,'+0530'))),
    ('ti','2024-02-29T03:04:05-02:30',datetime(2024,2,29,3,4,5,tzinfo=parse.FixedTzOffset(-150,'-02:30'))),
    ('tg','29/February/2024 12:04 AM',datetime(2024,2,29,0,4)),
    ('tg','29-2-2024 12:04 PM',datetime(2024,2,29,12,4)),
    ('ta','February/29/2024 1:04 PM +530',datetime(2024,2,29,13,4,tzinfo=parse.FixedTzOffset(330,'+530'))),
    ('ta','2-29-2024',datetime(2024,2,29)),
    ('te','Thu, 29 Feb 2024 03:04:05 +0000',datetime(2024,2,29,3,4,5,tzinfo=parse.FixedTzOffset(0,'+0000'))),
    ('te','29 Feb 2024 03:04 -230',datetime(2024,2,29,3,4,tzinfo=parse.FixedTzOffset(-150,'-230'))),
    ('th','29/Feb/2024:03:04:05 +06:00',datetime(2024,2,29,3,4,5,tzinfo=parse.FixedTzOffset(360,'+06:00'))),
    ('tc','Thu Feb 29 03:04:05 2024',datetime(2024,2,29,3,4,5)),
    ('tt','12:34 AM',time(0,34)), ('tt','12:34 PM',time(12,34)),
    ('tt','1:34 PM',time(13,34)), ('tt','1:34 AM',time(1,34)),
    ('tt','03:04:05.12 -0130',time(3,4,5,120000,tzinfo=parse.FixedTzOffset(-90,'-0130'))),
    ('tt','',time(0,0)),
    ('ts','Feb 12 03:04:05',datetime(datetime.today().year,2,12,3,4,5)),
    ('ts','Dec 12 ',datetime(datetime.today().year,12,12)),
])
def test_builtin_dates_with_surrounding_groups(spec,text,expected):
    r = parse.parse('{:d}|{:'+spec+'}|{when:'+spec+'}|{:d}', '9|'+text+'|'+text+'|8')
    assert r.fixed == (9,expected,8)
    assert r['when'] == expected
    assert r[1].tzinfo == expected.tzinfo
    assert r.spans[1] == (2,2+len(text))


@pytest.mark.parametrize('spec,text,expected', [
    ('%Y-%m-%d','2024-02-29',date(2024,2,29)),
    ('%d/%b/%Y %H:%M:%S','29/Feb/2024 03:04:05',datetime(2024,2,29,3,4,5)),
    ('%H:%M:%S.%f','03:04:05.12',time(3,4,5,120000)),
    ('%I:%M %p','01:04 PM',time(13,4)),
    ('%m-%d','02-12',date(datetime.today().year,2,12)),
    ('%Y %j','2024 060',date(2024,2,29)),
    ('%y/%m/%d','24/02/29',date(2024,2,29)),
])
def test_strptime_formats(spec,text,expected):
    r = parse.parse('{:'+spec+'}|{:d}',text+'|7')
    assert r.fixed == (expected,7)
    assert type(r[0]) is type(expected)


@pytest.mark.parametrize('month,num', [(name, i) for i, pair in enumerate(zip(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'), ('January','February','March','April','May','June','July','August','September','October','November','December')), 1) for name in pair])
def test_month_names(month,num):
    assert parse.parse('{:tg}', '12/'+month+'/2024')[0] == datetime(2024,num,12)


def test_timezone_protocol():
    a = parse.FixedTzOffset(90,'east')
    assert a.utcoffset(None) == timedelta(minutes=90)
    assert a.tzname(None) == 'east'
    assert a.dst(None) == timedelta(0)
    assert repr(a) == '<FixedTzOffset east 1:30:00>'
    assert a == parse.FixedTzOffset(90,'east')
    assert a != parse.FixedTzOffset(91,'east')
    assert a != parse.FixedTzOffset(90,'west')
    assert a != object()
    assert a.__eq__(object()) is NotImplemented


@pytest.mark.parametrize('base,text,expected', [(None,'0',0),(None,'-0Xaf',-175),(None,'+0O17',15),(None,'0B11',3),(None,'012',12),(16,'FE',254),(36,'z',35),(10,'-1,234',-1234)])
def test_integer_converter(base,text,expected):
    assert parse.int_convert(base)(text,None) == expected


def test_extract_format_components():
    f = parse.extract_format('*>+012,.3f',{})
    assert {k:f[k] for k in ('fill','align','zero','width','grouping','precision','type')} == dict(fill='*',align='>',zero=True,width='12',grouping=',',precision='3',type='f')
    f = parse.extract_format('d',{})
    assert (f['fill'],f['align'],f['zero'],f['width'],f['type']) == (None,None,False,'','d')


@pytest.mark.parametrize('names', [('a_b','a.b'), ('a_b','a-b'), ('a.b','a_b'), ('a_b','a__b','a.b'), ('a[b]','a_b_')])
def test_name_collision_order(names):
    fmt = '|'.join('{'+name+'}' for name in names)
    values = ['first','second','third'][:len(names)]
    result = parse.parse(fmt,'|'.join(values))
    for name,value in zip(names,values):
        if name == 'a[b]':
            assert result['a']['b'] == value
        else:
            assert result[name] == value


def test_compiled_case_sensitivity_and_cached_patterns():
    p = parse.compile('ABC{}!',case_sensitive=True)
    assert p.parse('abcx!') is None
    assert p.search('abcx!') is None
    assert list(p.findall('abcx!')) == []
    assert p.parse('ABCx\ny!')[0] == 'x\ny'
    assert p.search('...ABCz!')[0] == 'z'
    assert [r[0] for r in p.findall('ABC1!ABC2!')] == ['1','2']
    assert p._search_re is p._search_re
    assert p._match_re is p._match_re


@pytest.mark.parametrize('spec,text', [('ti','2023-02-29'), ('tg','32/Jan/2024'), ('tt','25:00')])
def test_invalid_calendar_values_raise(spec,text):
    with pytest.raises(ValueError):
        parse.parse('{:'+spec+'}',text)


@pytest.mark.parametrize('spec,text,expected', [
    ('%A %B %d %Y','Thursday February 29 2024',date(2024,2,29)),
    ('%a %b %d %Y','Thu Feb 29 2024',date(2024,2,29)),
    ('%Y %U %w','2024 08 4',date(2024,2,29)),
    ('%Y %W %w','2024 09 4',date(2024,2,29)),
])
def test_additional_calendar_directives(spec,text,expected):
    assert parse.parse('{:'+spec+'}',text)[0] == expected


def test_strptime_timezone_and_date_time_return_types():
    r = parse.parse('{:%H:%M%z}', '13:45+0530')[0]
    assert isinstance(r,time)
    assert (r.hour,r.minute,r.utcoffset()) == (13,45,timedelta(minutes=330))
    r = parse.parse('{:%Y-%m-%d %H:%M%z}', '2024-02-29 13:45-0230')[0]
    assert isinstance(r,datetime)
    assert (r.year,r.month,r.day,r.hour,r.minute,r.utcoffset()) == (2024,2,29,13,45,timedelta(minutes=-150))
