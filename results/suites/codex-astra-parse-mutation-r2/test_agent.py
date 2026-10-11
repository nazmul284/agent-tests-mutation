import math
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
import parse


@pytest.mark.parametrize('spec,text,expected', [
    ('d', '0', 0), ('d', '7', 7), ('d', '-42', -42), ('d', '+19', 19),
    ('d', ' 23', 23), ('d', '0012', 12), ('d', '0XAf', 175),
    ('d', '-0b101', -5), ('d', '+0O17', 15), ('d', '0x0', 0),
    (',d', '1,234,567', 1234567), ('_d', '-1_234', -1234),
    ('b', '101', 5), ('b', '-0B110', -6), ('o', '17', 15),
    ('o', '+0O70', 56), ('x', 'aF', 175), ('X', '-0XAB', -171),
    ('n', '1,234,567', 1234567), ('n', '-12.345', -12345), ('n', '123', 123),
    ('%', '-12.5%', -.125), ('%', '+50%', .5),
    ('f', '-.25', -.25), ('f', '+12.50', 12.5), ('.0f', '-12', -12.),
    ('F', '12.345', Decimal('12.345')), ('.0F', '+23', Decimal('23')),
    ('e', '-1.25e+2', -125.), ('E', '2.5E-2', .025),
    ('g', '12', 12.), ('g', '-1.25e2', -125.), ('G', '+2E-3', .002),
])
def test_numeric(spec, text, expected):
    # Surrounding fields detect incorrect accounting for internal regex groups.
    r = parse.parse('{}|{value:' + spec + '}|{:d}|{}', 'left|' + text + '|73|right')
    assert r is not None
    assert r.fixed == ('left', 73, 'right')
    assert r.named == {'value': expected}
    assert type(r['value']) is type(expected)
    assert r.spans == {0: (0, 4), 'value': (5, 5 + len(text)),
                       1: (6 + len(text), 8 + len(text)),
                       2: (9 + len(text), 14 + len(text))}
    r = parse.parse('{:' + spec + '}|{}', text + '|tail')
    assert r.fixed == (expected, 'tail')


@pytest.mark.parametrize('spec', ['f', 'F', '.0f', '.0F', 'e', 'E', 'g', 'G'])
@pytest.mark.parametrize('text', ['nan', 'NAN', 'inf', '-inf', '+INF'])
def test_special_numbers(spec, text):
    value = parse.parse('{:' + spec + '}', text)[0]
    if 'nan' in text.lower():
        assert math.isnan(value)
    else:
        assert value == (float('-inf') if text.startswith('-') else float('inf'))


@pytest.mark.parametrize('spec,text', [('b', '102'), ('o', '8'), ('x', 'xyz'),
    ('d', '1.2'), ('d', '0b12'), ('d', '0x'), ('n', '1234'), ('n', '1,23'),
    ('%', '12'), ('%', '.5%'), ('f', '12'), ('.0f', '12.0'), ('e', '12e3'),
    ('g', '.5'), ('l', 'abc2'), ('w', 'a-b'), ('D', 'a2'), ('S', 'a b')])
def test_rejected_values(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


@pytest.mark.parametrize('spec,text', [('w', 'abc_12'), ('W', '! -'), ('s', ' \t\n'),
    ('S', 'ab_!'), ('D', 'ab!'), ('l', 'AbCd')])
def test_character_classes(spec, text):
    assert parse.parse('{:' + spec + '}', text).fixed == (text,)


def test_literal_escaping_and_anchors():
    literal = r'?\.[x](y)*+^$!|'
    assert parse.parse(literal, literal).fixed == ()
    assert parse.parse('{{{x}}}', '{hello}')['x'] == 'hello'
    assert parse.parse('{{}}', '{}').named == {}
    assert parse.parse('abc', 'ABC') is not None
    assert parse.parse('abc', 'ABC', case_sensitive=True) is None
    for text in ['xabc', 'abcx', 'abc\n']:
        assert parse.parse('abc', text) is None
    assert parse.parse('', '') is not None
    assert parse.parse('{}', '') is None
    assert parse.parse('a{}z', 'aone\ntwoz')[0] == 'one\ntwo'
    assert parse.parse('{}:{}', 'a:b:c').fixed == ('a', 'b:c')


def test_results_and_nested_names():
    r = parse.parse('{2}|{}|{user[name]}|{user[age]:d}|{tree[a][b]}', 'A|B|Jo|28|leaf')
    assert r.fixed == ('A', 'B')
    assert r.named == {'user': {'name': 'Jo', 'age': 28}, 'tree': {'a': {'b': 'leaf'}}}
    assert r[0] == 'A' and r[-1] == 'B' and r[:] == ('A', 'B')
    assert r['user']['age'] == 28
    assert 'user' in r and 'missing' not in r and 0 not in r
    assert r.spans['user[age]'] == (7, 9)
    with pytest.raises(KeyError):
        _ = r['missing']
    with pytest.raises(IndexError):
        _ = r[2]
    assert repr(parse.parse('{} {x:d}', 'a 2')) == "<Result ('a',) {'x': 2}>"


@pytest.mark.parametrize('names', [('a.b', 'a_b', 'a__b'), ('a_b', 'a.b', 'a__b'),
                                   ('a_b', 'a-b', 'a__b')])
def test_name_collisions(names):
    fmt = '|'.join('{' + n + '}' for n in names)
    r = parse.parse(fmt + '|{' + names[0] + '}', 'one|two|three|one')
    assert r.named == dict(zip(names, ['one', 'two', 'three']))
    assert r.spans[names[0]] == (0, 3)
    assert parse.parse(fmt + '|{' + names[0] + '}', 'one|two|three|other') is None


def test_repeated_names():
    assert parse.parse('{x:d}/{x:d}/{}', '12/12/end').fixed == ('end',)
    assert parse.parse('{x:d}/{x:d}', '12/012') is None
    with pytest.raises(parse.RepeatedNameError, match='does not match'):
        parse.compile('{x:d}/{x:x}')


@pytest.mark.parametrize('spec,text,value', [('<', 'abc   ', 'abc'), ('>', '   abc', 'abc'),
    ('^', '  abc   ', 'abc'), ('.^', '...abc..', 'abc'), ('*>', '***abc', 'abc'),
    ('3', 'abcde', 'abcde'), ('.3', 'ab', 'ab'), ('2.4', 'abc', 'abc'),
    ('05d', '00023', 23), ('=d', '-00023', -23), ('_=d', '+___23', 23),
    ('6d', '    23', 23), ('<6d', '23    ', 23), ('^6d', ' 23 ', 23),
    ('+08.2f', '+0012.50', 12.5)])
def test_padding_width_precision(spec, text, value):
    r = parse.parse('{:' + spec + '}', text)
    assert r is not None and r[0] == value


@pytest.mark.parametrize('spec,text', [('3', 'ab'), ('.3', 'abcd'), ('2.4', 'a'),
                                       ('2.4', 'abcde'), ('2d', '123')])
def test_width_limits(spec, text):
    assert parse.parse('{:' + spec + '}', text) is None


def test_extract_format():
    result = parse.extract_format('*>+08,.2f', {})
    assert {k: result[k] for k in ('fill', 'align', 'zero', 'width', 'grouping', 'precision', 'type')} == {
        'fill': '*', 'align': '>', 'zero': True, 'width': '8', 'grouping': ',', 'precision': '2', 'type': 'f'}
    assert parse.extract_format('d', {})['zero'] is False
    assert parse.extract_format('_d', {})['grouping'] == '_'
    with pytest.raises(ValueError, match='not recognised'):
        parse.compile('{:unknown}')


def test_search_and_iterator_boundaries():
    text = 'xA1! A22! A333!z'
    p = parse.compile('A{:d}!')
    assert p.search(text).fixed == (1,)
    assert p.search(text, 3).fixed == (22,)
    assert p.search(text, 3, 8) is None
    assert p.search(text, 3, 9).spans == {0: (6, 8)}
    assert p.search(text, 14) is None
    it = p.findall(text)
    assert iter(it) is it
    assert next(it).fixed == (1,)
    assert it.next().fixed == (22,)
    assert next(it).fixed == (333,)
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)
    assert [r[0] for r in parse.findall('A{:d}!', text, 3, 9)] == [22]
    assert parse.search('a{:d}!', text, pos=3, endpos=9)[0] == 22
    assert parse.search('a{:d}!', text, case_sensitive=True) is None
    assert list(parse.findall('a{:d}!', text, case_sensitive=True)) == []


def test_custom_types_and_lazy_results():
    calls = []
    @parse.with_pattern(r'(\d+)-(\d+)', regex_group_count=2)
    def pair(s):
        calls.append(s)
        return tuple(map(int, s.split('-')))
    assert pair.pattern == r'(\d+)-(\d+)' and pair.regex_group_count == 2
    types = {'pair': pair}
    p = parse.compile('{:pair}|{n:pair}|{:d}', types)
    m = p.parse('12-34|5-6|78', evaluate_result=False)
    assert isinstance(m, parse.Match) and m.parser is p
    assert calls == []
    r = m.evaluate_result()
    assert r.fixed == ((12, 34), 78) and r.named == {'n': (5, 6)}
    assert r.spans == {0: (0, 5), 'n': (6, 9), 1: (10, 12)}
    assert calls == ['12-34', '5-6']
    for m in [parse.parse('{:pair}', '1-2', types, False),
              parse.search('{:pair}', 'x1-2y', extra_types=types, evaluate_result=False),
              next(parse.findall('{:pair}', 'x1-2y', extra_types=types, evaluate_result=False))]:
        assert isinstance(m, parse.Match)
        assert m.evaluate_result()[0] == (1, 2)
    @parse.with_pattern('[a-z]+')
    def upper(s):
        return s.upper()
    assert upper.regex_group_count is None
    assert parse.parse('{:u}|{:d}', 'ab|2', {'u': upper}).fixed == ('AB', 2)
    assert parse.parse('{:u}', 'abc', {'u': str.upper})[0] == 'ABC'
    assert parse.parse('{:d}', 'abc', {'d': str.upper})[0] == 'ABC'


def test_parser_properties_and_pickle():
    p = parse.compile('{} {x:d}', case_sensitive=True)
    assert p.format == '{} {x:d}'
    assert p.fixed_fields == [0] and p.named_fields == ['x']
    p.fixed_fields.append(99)
    p.named_fields.clear()
    assert p.fixed_fields == [0] and p.named_fields == ['x']
    assert repr(p) == "<Parser '{} {x:d}'>"
    assert repr(parse.compile('a' * 20)) == "<Parser '" + 'a' * 20 + "'>"
    assert repr(parse.compile('a' * 21)) == "<Parser '" + 'a' * 17 + "...'>"
    for parser in [p, pickle.loads(pickle.dumps(p))]:
        assert parser.parse('a 12').fixed == ('a',)
        assert parser.search('a 12')['x'] == 12
        assert parser.parse('b 23')['x'] == 23


@pytest.mark.parametrize('spec,text,expected,offset', [
    ('ti', '2024-02-29', datetime(2024, 2, 29), None),
    ('ti', '2024-02-29T14:03:04.1234567Z', datetime(2024, 2, 29, 14, 3, 4, 123456), 0),
    ('ti', '2024-02-29 14:03+0530', datetime(2024, 2, 29, 14, 3), 330),
    ('ti', '2024-02-29T14:03:04.2-03:45', datetime(2024, 2, 29, 14, 3, 4, 200000), -225),
    ('tg', '29/February/2024 12:03 AM +530', datetime(2024, 2, 29, 0, 3), 330),
    ('tg', '29-2-2024 12:03 PM', datetime(2024, 2, 29, 12, 3), None),
    ('tg', '29/Feb/2024 1:03 PM -3:30', datetime(2024, 2, 29, 13, 3), -210),
    ('tg', '29/2/2024', datetime(2024, 2, 29), None),
    ('ta', 'February/29/2024 1:03 AM', datetime(2024, 2, 29, 1, 3), None),
    ('ta', '2-29-2024 2:03:04 PM +0000', datetime(2024, 2, 29, 14, 3, 4), 0),
    ('ta', 'Feb/29/2024', datetime(2024, 2, 29), None),
    ('te', 'Thu, 29 Feb 2024 14:03:04 +0530', datetime(2024, 2, 29, 14, 3, 4), 330),
    ('te', '29 Feb 2024 14:03 -0330', datetime(2024, 2, 29, 14, 3), -210),
    ('th', '29/Feb/2024:14:03:04 +0530', datetime(2024, 2, 29, 14, 3, 4), 330),
    ('tc', 'Thu Feb 29 14:03:04 2024', datetime(2024, 2, 29, 14, 3, 4), None),
    ('tt', '12:03 AM', time(0, 3), None), ('tt', '12:03 PM', time(12, 3), None),
    ('tt', '2:03:04.012 PM -0330', time(14, 3, 4, 12000), -210),
    ('tt', '14:03 +0000', time(14, 3), 0),
    ('ts', 'Feb 12 14:03:04', datetime(datetime.today().year, 2, 12, 14, 3, 4), None),
])
def test_builtin_dates(spec, text, expected, offset):
    for field in ['{:' + spec + '}', '{date:' + spec + '}']:
        r = parse.parse('{:d}|' + field + '|{:d}', '17|' + text + '|29')
        assert r is not None
        actual = r['date'] if r.named else r[1]
        assert type(actual) is type(expected)
        assert actual.replace(tzinfo=None) == expected
        if offset is None:
            assert actual.tzinfo is None
        else:
            assert actual.utcoffset() == timedelta(minutes=offset)
            assert actual.dst() == timedelta(0)
        assert r[0] == 17 and r[-1] == 29
        assert r.spans['date' if r.named else 1] == (3, 3 + len(text))


@pytest.mark.parametrize('month,number', [
    ('Jan', 1), ('January', 1), ('Feb', 2), ('February', 2), ('Mar', 3), ('March', 3),
    ('Apr', 4), ('April', 4), ('May', 5), ('Jun', 6), ('June', 6), ('Jul', 7), ('July', 7),
    ('Aug', 8), ('August', 8), ('Sep', 9), ('September', 9), ('Oct', 10), ('October', 10),
    ('Nov', 11), ('November', 11), ('Dec', 12), ('December', 12)])
def test_months(month, number):
    assert parse.parse('{:tg}', '5/' + month + '/2023')[0] == datetime(2023, number, 5)


@pytest.mark.parametrize('spec,text,expected', [
    ('%Y-%m-%d', '2024-02-29', date(2024, 2, 29)),
    ('%y/%m/%d', '24/02/29', date(2024, 2, 29)),
    ('%B %d', 'March 12', date(datetime.today().year, 3, 12)),
    ('%H:%M:%S.%f', '14:03:04.12', time(14, 3, 4, 120000)),
    ('%I:%M %p', '02:03 PM', time(14, 3)),
    ('%Y-%m-%d %H:%M', '2024-02-29 14:03', datetime(2024, 2, 29, 14, 3)),
    ('%Y %j', '2024 060', date(2024, 2, 29)),
    ('%a %b %d %Y', 'Thu Feb 29 2024', date(2024, 2, 29)),
    ('%A %B %d %Y', 'Thursday February 29 2024', date(2024, 2, 29)),
])
def test_strptime_dates(spec, text, expected):
    r = parse.parse('{}|{v:' + spec + '}|{:d}', 'x|' + text + '|19')
    assert r is not None and r.fixed == ('x', 19)
    assert r['v'] == expected and type(r['v']) is type(expected)


def test_timezone_and_invalid_date():
    z = parse.FixedTzOffset(90, 'east')
    assert z.utcoffset(None) == timedelta(minutes=90)
    assert z.tzname(None) == 'east' and z.dst(None) == timedelta(0)
    assert repr(z) == '<FixedTzOffset east 1:30:00>'
    assert z == parse.FixedTzOffset(90, 'east')
    assert z != parse.FixedTzOffset(91, 'east')
    assert z != parse.FixedTzOffset(90, 'other')
    assert z != object()
    assert z.__eq__(object()) is NotImplemented
    assert parse.parse('{:ti}', '2024-01-02Z')[0].tzname() == 'UTC'
    assert parse.parse('{:ti}', '2024-01-02+0130')[0].tzname() == '+0130'
    with pytest.raises(ValueError):
        parse.parse('{:ti}', '2023-02-29')
    m = parse.parse('{:ti}', '2023-02-29', evaluate_result=False)
    with pytest.raises(ValueError):
        m.evaluate_result()
    assert parse.parse('{:%Y-%m-%d}', '24-02-29') is None


@pytest.mark.parametrize('base,text,expected', [(None, '0', 0), (None, '+0', 0),
    (None, '-0', 0), (None, '0B11', 3), (None, '0O11', 9), (None, '0X11', 17),
    (None, '011', 11), (10, '-1,234', -1234), (36, 'z', 35), (16, 'FF', 255)])
def test_integer_converter(base, text, expected):
    assert parse.int_convert(base)(text, None) == expected


@pytest.mark.parametrize('fill', list(r'.\+?*[]()$'))
@pytest.mark.parametrize('align', ['<', '>', '^'])
def test_regex_fill_characters(fill, align):
    text = {'<': 'abc' + fill * 3, '>': fill * 3 + 'abc',
            '^': fill * 2 + 'abc' + fill * 3}[align]
    r = parse.parse('{:' + fill + align + '}', text)
    assert r[0] == 'abc'
    start = text.index('abc')
    assert r.spans[0] == (start, start + 3)


@pytest.mark.parametrize('spec,text', [('3', 'abc'), ('.3', 'abc'), ('.3', 'a'),
    ('2.4', 'ab'), ('2.4', 'abcd'), ('2d', '12'), ('2d', '1'),
    ('2d', '0xff'), ('2d', '-0b11')])
def test_inclusive_width_boundaries(spec, text):
    assert parse.parse('{:' + spec + '}', text) is not None


def test_custom_converter_failure_is_deferred():
    def fail(value):
        raise ValueError('converter called')
    p = parse.compile('v={:bad}', {'bad': fail})
    assert p.parse('other') is None
    assert p.search('other', evaluate_result=False) is None
    m = p.search('v=text', evaluate_result=False)
    with pytest.raises(ValueError, match='converter called'):
        m.evaluate_result()
    with pytest.raises(ValueError, match='converter called'):
        next(p.findall('v=text'))
    lazy = next(p.findall('v=text', evaluate_result=False))
    assert isinstance(lazy, parse.Match)


def test_strptime_timezone_and_regex():
    r = parse.parse('{:%H:%M %z}', '14:03 -0330')[0]
    assert isinstance(r, time)
    assert (r.hour, r.minute, r.utcoffset()) == (14, 3, timedelta(minutes=-210))
    assert parse.get_regex_for_datetime_format('%Y-%m-%d') == '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}'
    with pytest.raises(ValueError, match='neither|nor'):
        parse.strf_date_convert('%', None, '%%')


def test_large_number_of_fields():
    # Modern Python supports more than the old 100-group limit.
    fmt = '|'.join('{field%d:d}' % n for n in range(110))
    text = '|'.join(str(n) for n in range(110))
    r = parse.parse(fmt, text)
    assert r.fixed == ()
    assert r.named == {'field%d' % n: n for n in range(110)}


def test_search_case_and_multiline_and_repeated_backreference():
    p = parse.compile('START{x}END', case_sensitive=True)
    assert p.search('startaend') is None
    assert p.search('xxSTARTa\nbENDyy')['x'] == 'a\nb'
    assert [r['x'] for r in p.findall('STARTaEND startbEND STARTcEND')] == ['a', 'c']
    assert parse.parse('{a}/{a}', 'Hi/hi')['a'] == 'Hi'
    assert parse.parse('{a}/{a}', 'Hi/hi', case_sensitive=True) is None


def test_date_groups_after_complex_types():
    r = parse.parse('{:g}|{:ti}|{:n}|{when:tt}|{}',
                    '1.5e2|2024-02-29T01:02:03Z|1,234|2:30 PM|end')
    assert r.fixed[0] == 150.0
    assert r.fixed[1].replace(tzinfo=None) == datetime(2024, 2, 29, 1, 2, 3)
    assert r.fixed[1].utcoffset() == timedelta(0)
    assert r.fixed[2:] == (1234, 'end')
    assert r.named == {'when': time(14, 30)}
