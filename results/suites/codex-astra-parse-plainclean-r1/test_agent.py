"""Behavioral regression tests written from the parser implementation."""

import math
import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import parse


@pytest.mark.parametrize('literal', ['', 'plain text', r'.*+?^$[]()\|!', '{braces}'])
def test_literals_are_not_regex(literal):
    fmt = literal.replace('{', '{{').replace('}', '}}')
    result = parse.parse(fmt, literal)
    assert result.fixed == ()
    assert result.named == {}
    assert result.spans == {}
    assert parse.parse(fmt, literal + '\n') is None
    assert parse.parse(fmt, 'x' + literal) is None


def test_mixed_results_and_spans():
    text = 'red:007/blue/42'
    result = parse.parse('{}:{count:d}/{}/{}', text)
    assert result.fixed == ('red', 'blue', '42')
    assert result.named == {'count': 7}
    assert result[0] == 'red'
    assert result[-1] == '42'
    assert result[1:] == ('blue', '42')
    assert result['count'] == 7
    assert 'count' in result and 'red' not in result
    assert result.spans == {0: (0, 3), 'count': (4, 7), 1: (8, 12), 2: (13, 15)}
    with pytest.raises(KeyError):
        _ = result['missing']
    with pytest.raises(IndexError):
        _ = result[3]


def test_untyped_fields_are_nonempty_lazy_and_multiline():
    assert parse.parse('{}|{}', 'a|b|c').fixed == ('a', 'b|c')
    assert parse.parse('start{}end', 'starta\nbend')[0] == 'a\nb'
    assert parse.parse('{}', '') is None
    assert parse.parse('{{{value}}}', '{hello}')['value'] == 'hello'


@pytest.mark.parametrize('api', [parse.parse, parse.search, parse.findall])
def test_case_sensitivity(api):
    result = api('ITEM={:d}', 'item=12')
    if api is parse.findall:
        result = list(result)[0]
    assert result[0] == 12
    sensitive = api('ITEM={:d}', 'item=12', case_sensitive=True)
    if api is parse.findall:
        assert list(sensitive) == []
    else:
        assert sensitive is None


def test_search_bounds_and_absolute_spans():
    text = 'x[12]x[345]x[6]'
    parser = parse.compile('[{:d}]')
    for search in (parser.search, lambda s, **kw: parse.search('[{:d}]', s, **kw)):
        assert search(text)[0] == 12
        result = search(text, pos=5, endpos=11)
        assert result.fixed == (345,)
        assert result.spans == {0: (7, 10)}
        assert search(text, pos=5, endpos=9) is None
        assert search(text, pos=len(text)) is None
    assert parser.parse(text) is None


def test_findall_bounds_nonoverlap_and_exhaustion():
    text = '[1][22][333]'
    for iterator in (parse.findall('[{:d}]', text, pos=3, endpos=7),
                     parse.compile('[{:d}]').findall(text, pos=3, endpos=7)):
        assert iter(iterator) is iterator
        result = next(iterator)
        assert result.fixed == (22,)
        assert result.spans == {0: (4, 6)}
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
    assert [r[0] for r in parse.findall('[{:d}]', text)] == [1, 22, 333]
    assert len(list(parse.findall('aba', 'ababa'))) == 1


@pytest.mark.parametrize('spec,text,expected', [
    ('d', '42', 42), ('d', '-42', -42), ('d', '+42', 42),
    ('d', '0xff', 255), ('d', '-0B101', -5), ('d', '+0o17', 15),
    ('d', '009', 9), (',d', '1,234,567', 1234567), ('_d', '-1_234', -1234),
    ('b', '0B1011', 11), ('b', '-101', -5), ('o', '0o71', 57),
    ('o', '+17', 15), ('x', '-0xFa', -250), ('X', 'FF', 255),
    ('n', '1,234,567', 1234567), ('n', '-12.345', -12345),
    ('f', '-.25', -0.25), ('f', '+12.50', 12.5),
    ('.0f', '-20', -20.0), ('.0F', '+20', Decimal('20')),
    ('F', '123.450', Decimal('123.450')),
    ('e', '-1.25e+2', -125.0), ('E', '1.25E-2', 0.0125),
    ('g', '12', 12.0), ('g', '1.5e3', 1500.0), ('G', '-2E-2', -0.02),
    ('%', '-12.5%', -0.125), ('%', '100%', 1.0),
])
def test_numeric_conversion_and_capture_accounting(spec, text, expected):
    # Surrounding fields catch incorrect indexing of internal regex groups.
    result = parse.parse('{:d}|{:' + spec + '}|{tail:d}|{}', '9|' + text + '|17|end')
    assert result.fixed == (9, expected, 'end')
    assert type(result[1]) is type(expected)
    assert result.named == {'tail': 17}
    assert result.spans[1] == (2, 2 + len(text))


@pytest.mark.parametrize('spec', ['f', 'F', '.0f', '.0F', 'e', 'E', 'g', 'G'])
@pytest.mark.parametrize('text', ['nan', 'INF', '-inf', '+inf'])
def test_special_numbers(spec, text):
    value = parse.parse('{:' + spec + '}', text)[0]
    if text == 'nan':
        assert math.isnan(value)
    else:
        assert value == float(text)


@pytest.mark.parametrize('spec,text', [
    ('d', '12z'), ('d', '1.2'), ('b', '102'), ('o', '89'), ('x', 'fg'),
    ('n', '12,34'), ('n', '1234'), ('f', '12'), ('.0f', '12.1'),
    ('e', '1.2'), ('%', '12'), ('%', '12%%'), ('g', '1e'),
])
def test_invalid_numeric_text_does_not_match(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


@pytest.mark.parametrize('spec,text', [
    ('w', 'abc_42'), ('W', '!@-'), ('D', 'abc!'), ('s', ' \t\n'),
    ('S', 'a_9!'), ('l', 'AbCd'),
])
def test_character_classes(spec, text):
    assert parse.parse('{:' + spec + '}', text)[0] == text


@pytest.mark.parametrize('spec,text', [('w', 'a-'), ('W', '!a'), ('D', 'a1'),
                                        ('s', ' a'), ('S', 'a b'), ('l', 'abc1')])
def test_character_class_rejections(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


@pytest.mark.parametrize('spec,value', [('05d', -12), ('+08d', 42), ('>8d', 17),
    ('<8d', 17), ('^9d', 17), ('_=8d', -12), ('08x', 255),
    ('*>8d', 12), ('.<8d', 12), ('*^8d', 12), ('.2f', -12.25)])
def test_numeric_format_roundtrips(spec, value):
    fmt = '{:' + spec + '}'
    assert parse.parse(fmt, fmt.format(value))[0] == value


@pytest.mark.parametrize('spec,text,expected,span', [
    ('>', '   cat', 'cat', (3, 6)), ('<', 'cat   ', 'cat', (0, 3)),
    ('^', '  cat  ', 'cat', (2, 5)), ('*>', '***cat', 'cat', (3, 6)),
    ('.<', 'cat...', 'cat', (0, 3)), ('.3', 'cat', 'cat', (0, 3)),
    ('2.4', 'cats', 'cats', (0, 4)),
])
def test_text_padding_precision_and_spans(spec, text, expected, span):
    result = parse.parse('{word:' + spec + '}', text)
    assert result['word'] == expected
    assert result.spans['word'] == span


@pytest.mark.parametrize('spec,text', [('.3', 'four'), ('2.4', 'a'), ('4', 'abc'), ('2d', '123')])
def test_width_and_precision_limits(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


def test_repeated_names_match_original_text():
    result = parse.parse('{n:d}/{n:d}/{:d}', '007/007/9')
    assert result.named == {'n': 7}
    assert result.fixed == (9,)
    assert result.spans['n'] == (0, 3)
    assert parse.parse('{n:d}/{n:d}', '007/7') is None
    assert parse.parse('{n}/{n}', 'one/two') is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile('{n:d}/{n:f}')


def test_names_with_collisions_and_nested_dictionaries():
    result = parse.parse('{a.b}/{a_b}/{a-b}/{user[id]:d}/{user[info][name]}/{user[0]}',
                         'dot/under/dash/7/Ada/zero')
    assert result.named == {'a.b': 'dot', 'a_b': 'under', 'a-b': 'dash',
                            'user': {'id': 7, 'info': {'name': 'Ada'}, '0': 'zero'}}
    assert result.spans['user[id]'] == (15, 16)


@pytest.mark.parametrize('fmt', ['{:unknown}', '{x:Q}', '{:.2q}'])
def test_unknown_types_fail_at_compile_time(fmt):
    with pytest.raises(ValueError, match='not recognised'):
        parse.compile(fmt)


def test_custom_pattern_with_nested_groups():
    @parse.with_pattern(r'([A-Z]+)-(\d+)', regex_group_count=2)
    def identifier(text):
        name, number = text.split('-')
        return name.lower(), int(number)

    assert identifier.pattern == r'([A-Z]+)-(\d+)'
    assert identifier.regex_group_count == 2
    result = parse.parse('{:ID}|{:d}|{key:ID}|{}', 'AB-12|7|CD-34|end', {'ID': identifier})
    assert result.fixed == (('ab', 12), 7, 'end')
    assert result.named == {'key': ('cd', 34)}
    assert result.spans == {0: (0, 5), 1: (6, 7), 'key': (8, 13), 2: (14, 17)}
    assert parse.parse('{:ID}', 'bad', {'ID': identifier}) is None


@pytest.mark.parametrize('api', ['parse', 'search', 'findall'])
def test_deferred_conversion(api):
    calls = []

    @parse.with_pattern(r'\d+')
    def twice(text):
        calls.append(text)
        return int(text) * 2

    result = getattr(parse, api)('[{x:twice}]', '[12]', extra_types={'twice': twice},
                                 evaluate_result=False)
    if api == 'findall':
        result = next(result)
    assert isinstance(result, parse.Match)
    assert calls == []
    assert result.evaluate_result().named == {'x': 24}
    assert calls == ['12']


def test_unpatterned_converter_override_and_errors():
    assert parse.parse('{:upper}', 'abc', {'upper': str.upper})[0] == 'ABC'
    assert parse.parse('{:d}', 'word', {'d': str.upper})[0] == 'WORD'

    def reject(text):
        raise ValueError('conversion rejected')

    match = parse.parse('{:bad}', 'anything', {'bad': reject}, evaluate_result=False)
    with pytest.raises(ValueError, match='conversion rejected'):
        match.evaluate_result()


@pytest.mark.parametrize('spec,text,expected', [
    ('ti', '2024-02-29', datetime(2024, 2, 29)),
    ('ti', '2024-02-29T23:04:05.1234567Z', datetime(2024, 2, 29, 23, 4, 5, 123456, timezone.utc)),
    ('ti', '2024-02-29 23:04+05:30', datetime(2024, 2, 29, 23, 4, tzinfo=timezone(timedelta(hours=5, minutes=30)))),
    ('tg', '29/February/2024 12:04 AM', datetime(2024, 2, 29, 0, 4)),
    ('ta', 'Feb-29-2024 12:04 PM', datetime(2024, 2, 29, 12, 4)),
    ('tg', '29-2-2024 1:04 PM -0330', datetime(2024, 2, 29, 13, 4, tzinfo=timezone(-timedelta(hours=3, minutes=30)))),
    ('te', 'Thu, 29 Feb 2024 13:04:05 +0000', datetime(2024, 2, 29, 13, 4, 5, tzinfo=timezone.utc)),
    ('th', '29/Feb/2024:13:04:05 -0500', datetime(2024, 2, 29, 13, 4, 5, tzinfo=timezone(-timedelta(hours=5)))),
    ('tc', 'Thu Feb 29 13:04:05 2024', datetime(2024, 2, 29, 13, 4, 5)),
    ('tt', '12:04 AM', time(0, 4)), ('tt', '12:04 PM', time(12, 4)),
    ('tt', '1:04:05.12 PM +530', time(13, 4, 5, 120000, timezone(timedelta(hours=5, minutes=30)))),
    ('%Y-%m-%d', '2024-02-29', date(2024, 2, 29)),
    ('%H:%M:%S', '23:04:05', time(23, 4, 5)),
    ('%Y/%m/%d %H:%M', '2024/02/29 23:04', datetime(2024, 2, 29, 23, 4)),
])
def test_dates_and_group_accounting(spec, text, expected):
    result = parse.parse('{:d}|{when:' + spec + '}|{:d}', '8|' + text + '|9')
    assert result.fixed == (8, 9)
    assert result['when'] == expected
    assert type(result['when']) is type(expected)
    assert result.spans['when'] == (2, len(text) + 2)


def test_dates_without_year_use_current_year():
    before = datetime.today().year
    syslog = parse.parse('{:ts}', 'Jan  2 03:04:05')[0]
    short = parse.parse('{:%m/%d}', '01/02')[0]
    after = datetime.today().year
    assert syslog.year in (before, after)
    assert (syslog.month, syslog.day, syslog.hour, syslog.minute, syslog.second) == (1, 2, 3, 4, 5)
    assert short.year in (before, after)
    assert (short.month, short.day) == (1, 2)


@pytest.mark.parametrize('spec,text', [('ti', '2023-02-29'), ('tt', '25:00'), ('%Y-%m-%d', '2024-13-01')])
def test_invalid_calendar_values_raise_on_evaluation(spec, text):
    result = parse.parse('{:' + spec + '}', text, evaluate_result=False)
    assert isinstance(result, parse.Match)
    with pytest.raises(ValueError):
        result.evaluate_result()


def test_fixed_timezone_contract():
    tz = parse.FixedTzOffset(-90, 'west')
    assert tz.utcoffset(None) == timedelta(minutes=-90)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == 'west'
    assert tz == parse.FixedTzOffset(-90, 'west')
    assert tz != parse.FixedTzOffset(90, 'west')
    assert tz != parse.FixedTzOffset(-90, 'other')
    assert tz != object()


def test_compiled_parser_reuse_metadata_and_pickle():
    parser = parse.compile('ID={id:d};{:g}', case_sensitive=True)
    assert parser.format == 'ID={id:d};{:g}'
    names, fixed = parser.named_fields, parser.fixed_fields
    names.clear()
    fixed.clear()
    assert parser.named_fields == ['id']
    assert len(parser.fixed_fields) == 1
    for candidate in (parser, pickle.loads(pickle.dumps(parser))):
        assert candidate.parse('ID=3;2.5').named == {'id': 3}
        assert candidate.parse('ID=8;4').fixed == (4.0,)
        assert candidate.parse('id=3;2.5') is None
        assert candidate.search('prefix ID=4;1.5 suffix')['id'] == 4
