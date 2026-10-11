"""Behavioral tests derived from the parser implementation and its API docstrings."""
import math
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


@pytest.mark.parametrize('literal', ['', 'a.b*+?^$|()[]!\\', '{{hello}}', 'café বাংলা'])
def test_literals_are_not_regular_expressions(literal):
    expected = literal.replace('{{', '{').replace('}}', '}')
    result = parse.parse(literal, expected)
    assert result is not None
    assert result.fixed == ()
    assert result.named == {}
    assert result.spans == {}
    assert parse.parse(literal, expected + 'x') is None
    assert parse.parse(literal, 'x' + expected) is None


def test_mixed_fields_result_access_and_absolute_spans():
    text = 'prefix [Ada:17:blue] suffix'
    result = parse.search('[{}:{age:d}:{color}]', text)
    assert result.fixed == ('Ada',)
    assert result.named == {'age': 17, 'color': 'blue'}
    assert result[0] == result[-1] == 'Ada'
    assert result[:] == ('Ada',)
    assert result['age'] == 17
    assert 'age' in result and 'absent' not in result and 0 not in result
    assert result.spans == {0: (8, 11), 'age': (12, 14), 'color': (15, 19)}
    with pytest.raises(KeyError):
        result['absent']
    with pytest.raises(IndexError):
        result[1]


def test_braces_multiline_and_nongreedy_fields():
    result = parse.parse('{{{}|{name}}}', '{first|second\nthird}')
    assert result.fixed == ('first',)
    assert result.named == {'name': 'second\nthird'}
    assert parse.parse('{}:{}', 'a:b:c').fixed == ('a', 'b:c')
    assert parse.parse('{}', '') is None
    assert parse.parse('literal', 'literal\n') is None


@pytest.mark.parametrize('api', [parse.parse, parse.search, parse.findall])
def test_case_sensitivity_is_forwarded(api):
    def results(**kwargs):
        value = api('TAG={value:w}', 'tag=MiXeD', **kwargs)
        return list(value) if api is parse.findall else ([] if value is None else [value])
    assert results()[0]['value'] == 'MiXeD'
    assert results(case_sensitive=True) == []


def test_search_and_findall_bounds_and_exhaustion():
    text = 'x=1; x=22; x=333;'
    parser = parse.compile('x={:d};')
    for search in [parser.search, lambda s, **kw: parse.search(parser.format, s, **kw)]:
        assert search(text)[0] == 1
        assert search(text, pos=4)[0] == 22
        assert search(text, pos=4, endpos=9) is None
        result = search(text, pos=4, endpos=10)
        assert result[0] == 22
        assert result.spans[0] == (7, 9)
        assert search(text, pos=len(text)) is None
    for findall in [parser.findall, lambda s, **kw: parse.findall(parser.format, s, **kw)]:
        iterator = findall(text)
        assert iter(iterator) is iterator
        assert [r[0] for r in iterator] == [1, 22, 333]
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
        assert [r[0] for r in findall(text, pos=4, endpos=10)] == [22]
        assert list(findall('no matches')) == []


def test_repeated_names_require_same_text_and_format():
    parser = parse.compile('{item:d}/{item:d}/{other}')
    result = parser.parse('12/12/end')
    assert result.named == {'item': 12, 'other': 'end'}
    assert result.spans['item'] == (0, 2)
    assert parser.parse('12/13/end') is None
    assert parser.parse('12/012/end') is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile('{x:d} {x:f}')
    with pytest.raises(parse.RepeatedNameError):
        parse.compile('{x} {x:w}')
    assert parse.parse('{x}/{x}', 'Ab/aB')['x'] == 'Ab'
    assert parse.parse('{x}/{x}', 'Ab/aB', case_sensitive=True) is None


def test_named_field_collisions_and_nested_dictionaries():
    result = parse.parse('{a.b}|{a_b}|{a-b}|{user[name]}|{user[stats][score]:d}',
                         'one|two|three|Ada|42')
    assert result.named == {'a.b': 'one', 'a_b': 'two', 'a-b': 'three',
                            'user': {'name': 'Ada', 'stats': {'score': 42}}}
    assert result.spans['user[stats][score]'] == (18, 20)


@pytest.mark.parametrize('spec,text,expected', [
    ('d', '-42', -42), ('d', '+42', 42), ('d', '00042', 42),
    ('d', '0x2a', 42), ('d', '-0B101', -5), ('d', '+0o17', 15),
    (',d', '1,234,567', 1234567), ('_d', '-1_234', -1234),
    ('n', '1,234,567', 1234567), ('n', '-1.234', -1234),
    ('b', '0B1011', 11), ('b', '-1011', -11),
    ('o', '0O17', 15), ('o', '-17', -15),
    ('x', '0xFf', 255), ('X', '-FF', -255),
    ('f', '-12.50', -12.5), ('f', '.25', .25),
    ('.0f', '-20', -20.), ('.0F', '+20', Decimal('20')),
    ('F', '123.4500', Decimal('123.4500')),
    ('e', '-1.25e+2', -125.), ('E', '1.25E-2', .0125),
    ('g', '12', 12.), ('g', '-1.2e3', -1200.), ('G', '1.2E-3', .0012),
    ('%', '12.5%', .125), ('%', '-50%', -.5),
])
def test_numeric_conversions_and_group_accounting(spec, text, expected):
    # Surround every numeric field with other captures to detect group shifts.
    result = parse.parse('{}|{value:' + spec + '}|{:d}|{}', 'left|' + text + '|7|right')
    assert result is not None
    assert result['value'] == expected
    assert type(result['value']) is type(expected)
    assert result.fixed == ('left', 7, 'right')
    assert result.spans['value'] == (5, 5 + len(text))
    result = parse.parse('{:' + spec + '}|{}', text + '|tail')
    assert result.fixed == (expected, 'tail')


@pytest.mark.parametrize('spec', ['f', 'F', 'e', 'E', 'g', 'G'])
@pytest.mark.parametrize('text', ['nan', 'NAN', 'inf', '-inf', '+INF'])
def test_special_floating_point_values(spec, text):
    value = parse.parse('{:' + spec + '}', text)[0]
    assert isinstance(value, Decimal if spec == 'F' else float)
    if text.lower() == 'nan':
        assert math.isnan(value)
    else:
        assert math.isinf(value)
        assert (value < 0) == text.startswith('-')


@pytest.mark.parametrize('spec,text', [('d', '12.3'), ('d', '0xZZ'), ('b', '102'),
    ('o', '89'), ('x', 'GG'), ('n', '12,34'), ('f', '12'), ('.0f', '12.0'),
    ('e', '1e3'), ('%', '50'), ('l', 'abc1'), ('w', 'two words'), ('D', 'a1')])
def test_invalid_typed_values_do_not_match(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


@pytest.mark.parametrize('spec,text', [('w', 'abc_123'), ('W', '!@ '), ('s', ' \t\n'),
    ('S', 'abc!?'), ('D', 'abc!?'), ('l', 'AbCd')])
def test_character_classes_preserve_text(spec, text):
    assert parse.parse('{:' + spec + '}', text)[0] == text


@pytest.mark.parametrize('spec,value', [('08d', 42), ('+08d', -42), ('>8d', 42),
    ('<8d', 42), ('^9d', -42), ('*>8d', 42), ('.>8d', 42),
    ('08.2f', -12.5), (' =8d', -42), ('0=8d', 42)])
def test_numeric_format_roundtrip(spec, value):
    text = format(value, spec)
    assert parse.parse('{:' + spec + '}', text)[0] == value


@pytest.mark.parametrize('spec,text,expected', [('<', 'abc   ', 'abc'),
    ('>', '   abc', 'abc'), ('^', '  abc  ', 'abc'), ('*^', '**abc**', 'abc'),
    ('.>', '...abc', 'abc'), ('.3', 'abc', 'abc'), ('2.4', 'abcd', 'abcd')])
def test_text_padding_and_precision(spec, text, expected):
    result = parse.parse('{:' + spec + '}', text)
    assert result[0] == expected
    start, end = result.spans[0]
    assert text[start:end] == expected


def test_width_and_precision_bounds():
    assert parse.parse('{:2.4}', 'a') is None
    assert parse.parse('{:2.4}', 'abcde') is None
    assert parse.parse('{:3}', 'ab') is None
    assert parse.parse('{:3}', 'abcd')[0] == 'abcd'
    assert parse.parse('{:2d}{:2d}', '1234').fixed == (12, 34)


@pytest.mark.parametrize('spec,text,expected', [
    ('ti', '2024-02-29', datetime(2024, 2, 29)),
    ('ti', '2024-02-29T13:14:15.1234567Z', datetime(2024, 2, 29, 13, 14, 15, 123456, parse.FixedTzOffset(0, 'UTC'))),
    ('ti', '2024-02-29 13:14+05:30', datetime(2024, 2, 29, 13, 14, tzinfo=parse.FixedTzOffset(330, '+05:30'))),
    ('tg', '29/February/2024 1:14 PM -0330', datetime(2024, 2, 29, 13, 14, tzinfo=parse.FixedTzOffset(-210, '-0330'))),
    ('ta', 'February-29-2024 12:14 AM', datetime(2024, 2, 29, 0, 14)),
    ('te', 'Thu, 29 Feb 2024 13:14:15 +0000', datetime(2024, 2, 29, 13, 14, 15, tzinfo=parse.FixedTzOffset(0, '+0000'))),
    ('th', '29/Feb/2024:13:14:15 -0530', datetime(2024, 2, 29, 13, 14, 15, tzinfo=parse.FixedTzOffset(-330, '-0530'))),
    ('tc', 'Thu Feb 29 13:14:15 2024', datetime(2024, 2, 29, 13, 14, 15)),
    ('ts', 'Feb 12 13:14:15', datetime(datetime.today().year, 2, 12, 13, 14, 15)),
    ('%Y-%m-%d', '2024-02-29', date(2024, 2, 29)),
    ('%H:%M:%S.%f', '13:14:15.123', time(13, 14, 15, 123000)),
    ('%Y/%m/%d %H:%M', '2024/02/29 13:14', datetime(2024, 2, 29, 13, 14)),
    ('%m-%d', '02-12', date(datetime.today().year, 2, 12)),
])
def test_dates_with_surrounding_captures(spec, text, expected):
    for field in ['{when:' + spec + '}', '{:' + spec + '}']:
        result = parse.parse('{}|' + field + '|{:d}', 'before|' + text + '|99')
        actual = result['when'] if result.named else result[1]
        assert actual == expected
        assert type(actual) is type(expected)
        assert result[0] == 'before'
        assert result[-1] == 99
        span_key = 'when' if result.named else 1
        assert result.spans[span_key] == (7, 7 + len(text))


@pytest.mark.parametrize('text,hour', [('12:15 AM', 0), ('12:15 PM', 12), ('1:15 PM', 13), ('1:15 AM', 1)])
def test_am_pm_conversion(text, hour):
    assert parse.parse('{:ta}', '2/12/2024 ' + text)[0] == datetime(2024, 2, 12, hour, 15)


def test_invalid_calendar_date_raises_at_evaluation():
    match = parse.parse('{:ti}', '2023-02-29', evaluate_result=False)
    assert isinstance(match, parse.Match)
    with pytest.raises(ValueError):
        match.evaluate_result()
    with pytest.raises(ValueError):
        parse.parse('{:%Y-%m-%d}', '2023-02-29')


def test_custom_pattern_captures_and_type_override():
    @parse.with_pattern(r'(\d+)-(\d+)', regex_group_count=2)
    def pair(text):
        return tuple(map(int, text.split('-')))
    assert pair.pattern == r'(\d+)-(\d+)'
    assert pair.regex_group_count == 2
    result = parse.parse('{:pair}/{x:pair}/{:d}/{}', '1-2/3-4/5/end', {'pair': pair})
    assert result.fixed == ((1, 2), 5, 'end')
    assert result.named == {'x': (3, 4)}
    assert result.spans == {0: (0, 3), 'x': (4, 7), 1: (8, 9), 2: (10, 13)}
    assert parse.parse('{:pair}', 'no', {'pair': pair}) is None
    assert parse.parse('{:d}', 'abc', {'d': str.upper})[0] == 'ABC'


@pytest.mark.parametrize('api', [parse.parse, parse.search, parse.findall])
def test_custom_conversions_are_deferred(api):
    calls = []
    @parse.with_pattern(r'\d+')
    def convert(text):
        calls.append(text)
        return int(text) + 1
    raw = api('v={:custom};', 'v=12;', extra_types={'custom': convert}, evaluate_result=False)
    assert calls == []
    if api is parse.findall:
        raw = next(raw)
    assert isinstance(raw, parse.Match)
    assert calls == []
    assert raw.evaluate_result().fixed == (13,)
    assert calls == ['12']
    assert raw.parser.evaluate_result(raw.match).fixed == (13,)


def test_converter_exception_is_not_swallowed():
    def fail(text):
        raise RuntimeError('conversion failed')
    parser = parse.compile('{:bad}', {'bad': fail})
    with pytest.raises(RuntimeError, match='conversion failed'):
        parser.parse('anything')
    assert parser.parse('', evaluate_result=False) is None


@pytest.mark.parametrize('spec', ['q', 'unknown', '.2q'])
def test_unknown_format_is_rejected(spec):
    with pytest.raises(ValueError, match='not recognised'):
        parse.compile('{:' + spec + '}')


def test_compiled_parser_reuse_properties_and_pickle():
    parser = parse.compile('ID={name:w}:{:d}', case_sensitive=True)
    assert parser.format == 'ID={name:w}:{:d}'
    named, fixed = parser.named_fields, parser.fixed_fields
    named.clear()
    fixed.clear()
    assert len(parser.named_fields) == len(parser.fixed_fields) == 1
    assert parser.parse('ID=one:1').named == {'name': 'one'}
    assert parser.search('prefix ID=two:2').fixed == (2,)
    restored = pickle.loads(pickle.dumps(parser))
    assert restored.parse('ID=three:3').fixed == (3,)
    assert restored.parse('id=three:3') is None
    assert [r[0] for r in restored.findall('ID=a:4 ID=b:5')] == [4, 5]


def test_fixed_timezone_contract():
    zone = parse.FixedTzOffset(-90, 'custom')
    assert zone.utcoffset(None) == timedelta(minutes=-90)
    assert zone.dst(None) == timedelta(0)
    assert zone.tzname(None) == 'custom'
    assert zone == parse.FixedTzOffset(-90, 'custom')
    assert zone != parse.FixedTzOffset(90, 'custom')
    assert zone != parse.FixedTzOffset(-90, 'other')
    assert zone != object()


def test_numeric_equal_alignment_defaults_to_zero_fill():
    assert parse.parse('{:=8d}', '-0000042')[0] == -42
    assert parse.parse('{:0=8d}', '+0000042')[0] == 42
