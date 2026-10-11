"""Behavioral tests written from the parser implementation and format semantics."""
import math
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import parse


@pytest.mark.parametrize('literal', ['', 'hello', '.*+?[]()^$!|\\', '{hello}', 'a\nb'])
def test_literal_matching_and_regex_escaping(literal):
    fmt = literal.replace('{', '{{').replace('}', '}}')
    result = parse.parse(fmt, literal)
    assert result.fixed == ()
    assert result.named == {}
    assert result.spans == {}
    assert parse.parse(fmt, literal + '!') is None
    assert parse.parse(fmt, '!' + literal) is None


def test_mixed_fields_result_access_and_spans():
    result = parse.parse('{}:{name}:{:d}', 'alpha:beta:42')
    assert result.fixed == ('alpha', 42)
    assert result.named == {'name': 'beta'}
    assert result[0] == 'alpha'
    assert result[-1] == 42
    assert result[:] == ('alpha', 42)
    assert result['name'] == 'beta'
    assert 'name' in result and 'absent' not in result
    assert result.spans == {0: (0, 5), 'name': (6, 10), 1: (11, 13)}
    with pytest.raises(KeyError):
        result['absent']
    with pytest.raises(IndexError):
        result[2]


def test_fields_are_nonempty_lazy_and_multiline():
    assert parse.parse('{}|{}', 'one|two|three').fixed == ('one', 'two|three')
    assert parse.parse('<{}>', '<a\nb>').fixed == ('a\nb',)
    assert parse.parse('<{}>', '<>') is None
    assert parse.parse('abc', 'abc\n') is None


@pytest.mark.parametrize('api', [parse.parse, parse.search, lambda f, s, **kw: next(parse.findall(f, s, **kw), None)])
def test_case_sensitivity(api):
    assert api('ID={:d}', 'id=7').fixed == (7,)
    assert api('ID={:d}', 'id=7', case_sensitive=True) is None
    assert api('ID={:d}', 'ID=7', case_sensitive=True).fixed == (7,)


@pytest.mark.parametrize('spec,text,expected', [
    ('d', '0', 0), ('d', '-42', -42), ('d', '+17', 17), ('d', ' 17', 17),
    ('d', '0xAf', 175), ('d', '-0B101', -5), ('d', '0o17', 15),
    (',d', '1,234,567', 1234567), ('_d', '-1_234', -1234),
    ('b', '1011', 11), ('b', '-0B101', -5), ('o', '+0o17', 15),
    ('x', '0xAf', 175), ('X', '-FF', -255), ('n', '1,234,567', 1234567),
    ('n', '-12.345', -12345), ('f', '-.125', -.125), ('f', '+12.5', 12.5),
    ('F', '0.125', Decimal('0.125')), ('.0f', '-12', -12.0),
    ('.0F', '+12', Decimal('12')), ('e', '-1.25e+2', -125.0),
    ('E', '1.25E-2', .0125), ('g', '12', 12.0), ('G', '-12.5E2', -1250.0),
])
def test_numeric_conversions_and_capture_accounting(spec, text, expected):
    result = parse.parse('{:' + spec + '}|{tail:d}|{}', text + '|9|end')
    assert result.fixed == (expected, 'end')
    assert type(result[0]) is type(expected)
    assert result.named == {'tail': 9}
    assert result.spans == {0: (0, len(text)), 'tail': (len(text)+1, len(text)+2), 1: (len(text)+3, len(text)+6)}


@pytest.mark.parametrize('spec', ['f', 'F', 'e', 'E', 'g', 'G'])
@pytest.mark.parametrize('text', ['nan', 'NAN', 'inf', '-inf', '+INF'])
def test_nonfinite_numbers(spec, text):
    value = parse.parse('{:' + spec + '}', text)[0]
    if text.lower() == 'nan':
        assert math.isnan(value)
    else:
        assert value == float(text)


@pytest.mark.parametrize('spec,text', [('d', '12.5'), ('b', '102'), ('o', '8'), ('x', 'xyz'), ('n', '12,34'), ('f', '12'), ('e', '1.2'), ('%', '12'), ('l', 'abc1'), ('w', 'a-b'), ('s', 'x'), ('D', '12'), ('W', 'abc'), ('S', ' ')])
def test_invalid_typed_values_do_not_match(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


@pytest.mark.parametrize('spec,text', [('l', 'Abc'), ('w', 'abc_12'), ('W', '!@'), ('D', 'abc!'), ('s', '\t \n'), ('S', 'abc!')])
def test_character_classes(spec, text):
    assert parse.parse('{:' + spec + '}', text).fixed == (text,)


@pytest.mark.parametrize('spec,value', [('08d', -42), ('>8d', 42), ('<8d', 42), ('^8d', 42), ('0=8d', -42), ('*>8d', 42), ('.<8d', 42), ('+d', 42), ('.2f', -1.25)])
def test_numeric_format_roundtrip(spec, value):
    assert parse.parse('{:' + spec + '}', format(value, spec))[0] == value


@pytest.mark.parametrize('spec,text,expected', [('>','   abc','abc'), ('<','abc   ','abc'), ('^','  abc  ','abc'), ('*^','**abc**','abc'), ('.3','abc','abc'), ('2.4','abcd','abcd'), ('3','abcd','abcd')])
def test_string_alignment_and_lengths(spec, text, expected):
    assert parse.parse('{:' + spec + '}', text)[0] == expected


@pytest.mark.parametrize('spec,text', [('.3', 'abcd'), ('2.4', 'a'), ('2.4', 'abcde'), ('3', 'ab'), ('2d', '123')])
def test_width_and_precision_boundaries(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


def test_repeated_names_require_identical_text_and_type():
    result = parse.parse('{x:d}/{x:d}', '12/12')
    assert result.named == {'x': 12}
    assert result.spans == {'x': (0, 2)}
    assert parse.parse('{x:d}/{x:d}', '12/13') is None
    assert parse.parse('{x:d}/{x:d}', '12/012') is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile('{x:d}/{x:f}')


def test_nested_keys_and_group_name_collisions():
    result = parse.parse('{a.b}/{a_b}/{a-b}/{user[address][city]}/{user[id]:d}', 'one/two/three/Dhaka/7')
    assert result.named == {'a.b': 'one', 'a_b': 'two', 'a-b': 'three', 'user': {'address': {'city': 'Dhaka'}, 'id': 7}}
    assert result.spans['user[id]'] == (20, 21)


def test_search_bounds_and_absolute_spans():
    text = 'xx <12> yy <34> zz'
    parser = parse.compile('<{n:d}>')
    assert parse.parse('<{n:d}>', text) is None
    assert parser.search(text)['n'] == 12
    result = parse.search('<{n:d}>', text, pos=7, endpos=15)
    assert result.named == {'n': 34}
    assert result.spans == {'n': (12, 14)}
    assert parser.search(text, endpos=5) is None
    assert parser.search(text, pos=14) is None
    assert parser.search('missing') is None


def test_findall_bounds_nonoverlap_and_exhaustion():
    text = '[1] [22] [333]'
    iterator = parse.findall('[{:d}]', text, pos=4, endpos=8)
    assert iter(iterator) is iterator
    assert next(iterator).fixed == (22,)
    with pytest.raises(StopIteration):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    assert [r.fixed for r in parse.compile('[{:d}]').findall(text)] == [(1,), (22,), (333,)]
    assert list(parse.findall('[{:d}]', 'none')) == []
    assert len(list(parse.findall('aba', 'ababa'))) == 1


def test_custom_converter_nested_groups_and_deferred_evaluation():
    calls = []

    @parse.with_pattern(r'(\d+)-(\d+)', regex_group_count=2)
    def pair(text):
        calls.append(text)
        return tuple(map(int, text.split('-')))

    parser = parse.compile('{:pair}|{:d}|{named:pair}|{}', {'pair': pair})
    match = parser.parse('1-2|3|4-5|end', evaluate_result=False)
    assert isinstance(match, parse.Match)
    assert calls == []
    result = match.evaluate_result()
    assert result.fixed == ((1, 2), 3, 'end')
    assert result.named == {'named': (4, 5)}
    assert calls == ['1-2', '4-5']
    assert result.spans == {0: (0, 3), 1: (4, 5), 'named': (6, 9), 2: (10, 13)}
    assert pair.pattern == r'(\d+)-(\d+)'
    assert pair.regex_group_count == 2


@pytest.mark.parametrize('api', [parse.parse, parse.search, lambda f,s,**kw: next(parse.findall(f,s,**kw))])
def test_wrappers_forward_custom_types_and_evaluation(api):
    calls = []
    def convert(text):
        calls.append(text)
        return text.upper()
    match = api('<{:custom}>', '<abc>', extra_types={'custom': convert}, evaluate_result=False)
    assert isinstance(match, parse.Match)
    assert calls == []
    assert match.evaluate_result().fixed == ('ABC',)
    assert calls == ['abc']
    assert api('<{:custom}>', '<abc>', extra_types={'custom': convert}).fixed == ('ABC',)


def test_custom_pattern_and_conversion_error():
    @parse.with_pattern(r'\d+')
    def reject(text):
        raise ValueError('conversion rejected')
    parser = parse.compile('{:bad}', {'bad': reject})
    assert parser.parse('abc') is None
    assert isinstance(parser.parse('12', evaluate_result=False), parse.Match)
    with pytest.raises(ValueError, match='conversion rejected'):
        parser.parse('12')


def test_compiled_parser_reuse_and_metadata_copies():
    parser = parse.compile('{}-{name:d}', case_sensitive=True)
    assert parser.format == '{}-{name:d}'
    fixed, named = parser.fixed_fields, parser.named_fields
    fixed.clear()
    named.clear()
    assert parser.fixed_fields == [0]
    assert parser.named_fields == ['name']
    assert parser.parse('a-1').fixed == ('a',)
    assert parser.parse('b-2').named == {'name': 2}
    assert parser.search('xx c-3').named == {'name': 3}


@pytest.mark.parametrize('spec', ['q', 'unknown', '12.2q'])
def test_unknown_format_rejected(spec):
    with pytest.raises(ValueError, match='not recognised'):
        parse.compile('{:' + spec + '}')


@pytest.mark.parametrize('spec,text,expected', [
    ('ti', '2024-02-29', datetime(2024, 2, 29)),
    ('ti', '2024-02-29T13:14:15.1234567Z', datetime(2024, 2, 29, 13, 14, 15, 123456, timezone.utc)),
    ('ti', '2024-02-29 13:14+05:30', datetime(2024, 2, 29, 13, 14, tzinfo=timezone(timedelta(hours=5, minutes=30)))),
    ('tg', '29/February/2024 12:15 AM', datetime(2024, 2, 29, 0, 15)),
    ('ta', 'Feb/29/2024 12:15 PM', datetime(2024, 2, 29, 12, 15)),
    ('tg', '29-02-2024 1:15 PM -0330', datetime(2024, 2, 29, 13, 15, tzinfo=timezone(-timedelta(hours=3, minutes=30)))),
    ('te', 'Thu, 29 Feb 2024 13:14:15 +0000', datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone.utc)),
    ('th', '29/Feb/2024:13:14:15 +0000', datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone.utc)),
    ('tc', 'Thu Feb 29 13:14:15 2024', datetime(2024, 2, 29, 13, 14, 15)),
    ('tt', '12:30 AM', time(0, 30)), ('tt', '12:30 PM', time(12, 30)),
    ('tt', '1:30:45.12 PM +0530', time(13, 30, 45, 120000, timezone(timedelta(minutes=330)))),
    ('%Y-%m-%d', '2024-02-29', date(2024, 2, 29)),
    ('%H:%M:%S.%f', '13:14:15.12', time(13, 14, 15, 120000)),
    ('%Y/%m/%d %H:%M', '2024/02/29 13:14', datetime(2024, 2, 29, 13, 14)),
])
def test_datetime_types_and_following_capture(spec, text, expected):
    result = parse.parse('{}|{when:' + spec + '}|{:d}', 'start|' + text + '|7')
    assert result.fixed == ('start', 7)
    assert result['when'] == expected
    assert type(result['when']) is type(expected)
    assert result.spans['when'] == (6, 6 + len(text))


def test_implicit_year_dates():
    before = datetime.today().year
    result = parse.parse('{:ts}', 'Jan 2 03:04:05')[0]
    simple = parse.parse('{:%m-%d}', '01-02')[0]
    after = datetime.today().year
    assert result.year in {before, after}
    assert (result.month, result.day, result.hour, result.minute, result.second) == (1, 2, 3, 4, 5)
    assert simple.year in {before, after}
    assert (simple.month, simple.day) == (1, 2)


def test_invalid_calendar_date_raises_during_evaluation():
    match = parse.parse('{:ti}', '2023-02-29', evaluate_result=False)
    with pytest.raises(ValueError):
        match.evaluate_result()


def test_fixed_timezone_contract():
    tz = parse.FixedTzOffset(-90, 'local')
    assert tz.utcoffset(None) == timedelta(minutes=-90)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == 'local'
    assert tz == parse.FixedTzOffset(-90, 'local')
    assert tz != parse.FixedTzOffset(90, 'local')
    assert tz != parse.FixedTzOffset(-90, 'other')
    assert tz != object()


@pytest.mark.xfail(strict=True, reason='Known defect: percentage() divides by 10 instead of 100')
@pytest.mark.parametrize('value', [0.125, -0.5, 1.0])
def test_percentage_format_roundtrip(value):
    result = parse.parse('{:%}|{:d}', format(value, '.1%') + '|7')
    assert result.fixed == (value, 7)


def test_zero_percentage_and_following_capture():
    assert parse.parse('{:%}|{:d}', '0%|7').fixed == (0.0, 7)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason='Known defect: numeric = alignment does not escape regex fill characters')
def test_numeric_sign_alignment_with_literal_star_fill():
    assert parse.parse('{:*=8d}', format(-42, '*=8d'))[0] == -42
