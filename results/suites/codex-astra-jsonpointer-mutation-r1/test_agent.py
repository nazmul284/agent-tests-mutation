"""Behavioral tests for jsonpointer, written against the local implementation."""
from collections import UserDict, UserList
from types import MappingProxyType

import pytest

import jsonpointer as jp


@pytest.mark.parametrize('raw, encoded', [
    ('', ''), ('plain', 'plain'), ('/', '~1'), ('~', '~0'),
    ('~/~01', '~0~1~001'), ('a/b~c', 'a~1b~0c'),
    ('雪 %20 #\\', '雪 %20 #\\'),
])
def test_escape_and_unescape(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


def test_unescape_order_and_literal_characters():
    assert jp.unescape('~01/~10/~00/~11') == '~1//0/~0//1'
    assert jp.unescape('%2F+~2') == '%2F+~2'


@pytest.mark.parametrize('path, parts', [
    ('', []), ('/', ['']), ('//', ['', '']), ('/a/', ['a', '']),
    ('/~0/~1/~01', ['~', '/', '~1']), ('/a//b', ['a', '', 'b']),
    ('/雪/a%20b/a b', ['雪', 'a%20b', 'a b']),
])
def test_parsing_and_rendering(path, parts):
    ptr = jp.JsonPointer(path)
    assert ptr.parts == parts
    assert ptr.get_parts() == parts
    assert ptr.path == path
    assert str(ptr) == path
    assert repr(ptr) == 'JsonPointer(' + repr(path) + ')'
    assert jp.JsonPointer.from_parts(iter(parts)) == ptr


@pytest.mark.parametrize('path', ['foo', 'a/b', '#/a', ' /a'])
def test_missing_initial_slash(path):
    with pytest.raises(jp.JsonPointerException, match='Location must start with /'):
        jp.JsonPointer(path)


@pytest.mark.parametrize('path, bad', [
    ('/~', '~'), ('/a~2b', '~2'), ('/a/~9', '~9'),
    ('/~0/late~x', '~x'), ('/~~0', '~~'), ('/a~\nb', '~\n'),
])
def test_invalid_escape(path, bad):
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.JsonPointer(path)
    assert str(exc.value) == 'Found invalid escape ' + bad


def test_from_parts_stringifies_and_preserves_subclasses():
    class Pointer(jp.JsonPointer):
        pass
    ptr = Pointer.from_parts(['a/b', '~1', 0, 12, '', None])
    assert type(ptr) is Pointer
    assert ptr.path == '/a~1b/~01/0/12//None'
    assert ptr.parts == ['a/b', '~1', '0', '12', '', 'None']
    assert repr(Pointer('/x')) == "Pointer('/x')"
    assert type(ptr.join('/tail')) is Pointer


@pytest.fixture
def document():
    return {'foo': ['bar', {'deep': [False, None, 0]}], '': {'': 5},
            'a/b': 6, '~1': 7, '~': 8, 'a%20b': 9, 'a b': 10,
            '01': 11, '-': 12}


@pytest.mark.parametrize('path, expected', [
    ('/foo/0', 'bar'), ('/foo/1/deep/0', False),
    ('/foo/1/deep/1', None), ('/foo/1/deep/2', 0),
    ('//', 5), ('/a~1b', 6), ('/~01', 7), ('/~0', 8),
    ('/a%20b', 9), ('/a b', 10), ('/01', 11), ('/-', 12),
])
def test_resolution(document, path, expected):
    marker = object()
    assert jp.resolve_pointer(document, path, marker) == expected
    ptr = jp.JsonPointer(path)
    assert ptr.resolve(document, marker) == expected
    assert ptr.get(document, marker) == expected


@pytest.mark.parametrize('doc', [None, False, 0, 'text', [], {}])
def test_root_resolution(doc):
    assert jp.resolve_pointer(doc, '') is doc
    assert jp.JsonPointer('').to_last(doc) == (doc, None)


def test_resolution_returns_original_object(document):
    assert jp.resolve_pointer(document, '/foo') is document['foo']
    parent, key = jp.JsonPointer('/foo/1/deep/2').to_last(document)
    assert parent is document['foo'][1]['deep']
    assert key == 2 and type(key) is int
    parent, key = jp.JsonPointer('/01').to_last(document)
    assert parent is document and key == '01'


@pytest.mark.parametrize('doc, path', [
    ({}, '/missing'), ({'a': {}}, '/a/missing/child'),
    ([], '/0'), ([1], '/1'), ([1], '/9'),
    ({'a': None}, '/a/b'), ('abc', '/0'),
    ([1], '/-1'), ([1], '/01'), ([1], '/-/x'),
])
def test_failed_resolution_and_defaults(doc, path):
    ptr = jp.JsonPointer(path)
    with pytest.raises(jp.JsonPointerException):
        ptr.resolve(doc)
    for default in (None, False, 0, '', [], object()):
        assert ptr.resolve(doc, default) is default
        assert jp.resolve_pointer(doc, path, default) is default


def test_invalid_pointer_is_not_hidden_by_default():
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer({}, '/~2', None)


@pytest.mark.parametrize('token', ['', '00', '01', '0abc', '1a', '-1', '+1', ' 1', '1 ', '1.0', '1\n', '١'])
def test_invalid_sequence_indices(token):
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.JsonPointer.get_part([], token)
    assert str(exc.value) == "'%s' is not a valid sequence index" % token


@pytest.mark.parametrize('token, expected', [('0', 0), ('1', 1), ('10', 10), ('123', 123), (2, 2)])
def test_sequence_indices(token, expected):
    result = jp.JsonPointer.get_part([], token)
    assert result == expected and type(result) is int


@pytest.mark.parametrize('doc', [[3, 4], (3, 4), UserList([3, 4])])
def test_sequences_and_end_of_list(doc):
    assert jp.resolve_pointer(doc, '/1') == 4
    end = jp.resolve_pointer(doc, '/-')
    assert isinstance(end, jp.EndOfList)
    assert end.list_ is doc
    assert repr(end) == 'EndOfList(' + repr(doc) + ')'
    assert jp.JsonPointer.get_part(doc, '-') == '-'
    with pytest.raises(jp.JsonPointerException, match="index '2' is out of bounds"):
        jp.resolve_pointer(doc, '/2')


@pytest.mark.parametrize('doc', [UserDict({'0': 7}), MappingProxyType({'0': 7})])
def test_mapping_protocol(doc):
    assert jp.resolve_pointer(doc, '/0') == 7
    assert jp.JsonPointer.get_part(doc, '0') == '0'


@pytest.mark.parametrize('doc', [None, 1, 1.5, True, object()])
def test_non_indexable(doc):
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.JsonPointer.get_part(doc, 'a')
    assert str(exc.value) == "Document '%s' does not support indexing, must be mapping/sequence or support __getitem__" % type(doc)


def test_string_is_not_a_container():
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.resolve_pointer('abc', '/0')
    assert str(exc.value) == "Cannot apply token '0' to non-container type <class 'str'>"


@pytest.mark.parametrize('verbose', [True, False])
def test_missing_member_messages(monkeypatch, verbose):
    monkeypatch.setattr(jp, 'VERBOSE_EXCEPTIONS', verbose)
    doc = {'secret': 'contents'}
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.resolve_pointer(doc, '/absent')
    expected = "member 'absent' not found"
    if verbose:
        expected += ' in ' + str(doc)
    assert str(exc.value) == expected


def test_duck_typing_and_unrelated_errors():
    class Indexed:
        def __init__(self):
            self.values = {'key': 3}
        def __getitem__(self, key):
            if key == 'boom':
                raise RuntimeError('unrelated')
            return self.values[key]
        def __setitem__(self, key, value):
            self.values[key] = value
    doc = Indexed()
    assert jp.resolve_pointer(doc, '/key') == 3
    assert jp.resolve_pointer(doc, '/missing', None) is None
    assert jp.set_pointer(doc, '/key', 8) is doc
    assert doc.values == {'key': 8}
    with pytest.raises(RuntimeError, match='unrelated'):
        jp.resolve_pointer(doc, '/boom', None)


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('path', ['/a/0/x', '/a/0/new', '/a/1', '/a/-', '/new', '/a~1b'])
def test_set_nested_and_copy(inplace, path):
    doc = {'a': [{'x': 1}, 2], 'untouched': {'items': []}}
    value = {'value': []}
    result = jp.set_pointer(doc, path, value, inplace=inplace)
    assert (result is doc) is inplace
    expected = {'a': [{'x': 1}, 2], 'untouched': {'items': []}}
    if path == '/a/0/x':
        expected['a'][0]['x'] = value
    elif path == '/a/0/new':
        expected['a'][0]['new'] = value
    elif path == '/a/1':
        expected['a'][1] = value
    elif path == '/a/-':
        expected['a'].append(value)
    else:
        expected['new' if path == '/new' else 'a/b'] = value
    assert result == expected
    actual_path = '/a/2' if path == '/a/-' else path
    assert jp.resolve_pointer(result, actual_path) is value
    if not inplace:
        assert doc == {'a': [{'x': 1}, 2], 'untouched': {'items': []}}
        assert result['a'] is not doc['a']
        assert result['untouched']['items'] is not doc['untouched']['items']


def test_default_inplace_and_dictionary_dash():
    doc = {'': 1}
    assert jp.set_pointer(doc, '/', 2) is doc
    assert jp.JsonPointer('/-').set(doc, 3) is doc
    assert doc == {'': 2, '-': 3}


@pytest.mark.parametrize('doc', [{}, [], None, 'text'])
def test_set_root(doc):
    value = []
    assert jp.set_pointer(doc, '', value, inplace=False) is value
    with pytest.raises(jp.JsonPointerException, match='^Cannot set root in place$'):
        jp.set_pointer(doc, '', value)


@pytest.mark.parametrize('doc, path', [([], '/0'), ([1], '/1'), ((1,), '/0'), (MappingProxyType({'x': 1}), '/x')])
def test_invalid_assignment_target(doc, path):
    with pytest.raises(jp.JsonPointerException, match='^Invalid assignment target: '):
        jp.set_pointer(doc, path, 9)


@pytest.mark.parametrize('doc, path', [({}, '/missing/child'), ({'a': None}, '/a/x'), ('text', '/0'), ([1], '/01')])
def test_set_cannot_create_intermediate_containers(doc, path):
    with pytest.raises(jp.JsonPointerException):
        jp.set_pointer(doc, path, 9)


@pytest.mark.parametrize('suffix', ['/c~1d/~0', jp.JsonPointer('/c~1d/~0'), ['c/d', '~'], ('c/d', '~')])
def test_join_and_division(suffix):
    ptr = jp.JsonPointer('/a~1b')
    expected = jp.JsonPointer('/a~1b/c~1d/~0')
    assert ptr.join(suffix) == expected
    assert ptr / suffix == expected
    assert ptr.parts == ['a/b']


def test_join_empty_generator_and_invalid():
    ptr = jp.JsonPointer('/a')
    for suffix in ('', [], jp.JsonPointer('')):
        result = ptr.join(suffix)
        assert result == ptr and result is not ptr
    assert ptr.join(x for x in [0, '', '~']) == jp.JsonPointer('/a/0//~0')
    for suffix in (None, 1, object()):
        with pytest.raises(jp.JsonPointerException, match='^Invalid suffix$'):
            ptr.join(suffix)
    with pytest.raises(jp.JsonPointerException, match='Location must start with /'):
        ptr.join('bad')


@pytest.mark.parametrize('path, prefix, expected', [
    ('/a/b', '/a', True), ('/a', '/a/b', False), ('/a', '/a', True),
    ('/ab', '/a', False), ('/a/b', '/b', False),
    ('/a', '', True), ('', '/a', False), ('', '', True),
])
def test_contains(path, prefix, expected):
    ptr, other = jp.JsonPointer(path), jp.JsonPointer(prefix)
    assert ptr.contains(other) is expected
    assert (other in ptr) is expected


def test_equality_and_hash():
    left = jp.JsonPointer('/a/~0')
    right = jp.JsonPointer.from_parts(['a', '~'])
    assert left == right
    assert hash(left) == hash(right) == hash(('a', '~'))
    assert len({left, right, jp.JsonPointer('/b')}) == 2
    assert {left: 'value'}[right] == 'value'
    for other in ('/a/~0', ['a', '~'], None, jp.JsonPointer('/a'), jp.JsonPointer('/a/~1')):
        assert (left == other) is False
    assert jp.JsonPointer('') != jp.JsonPointer('/')


@pytest.mark.parametrize('items, expected', [([], []), ([1], []), ([1, 2], [(1, 2)]), ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])])
def test_pairwise(items, expected):
    assert list(jp.pairwise(iter(items))) == expected


def test_to_last_stops_before_missing_leaf():
    child = {}
    doc = {'a': child, 'list': []}
    parent, token = jp.JsonPointer('/a/new').to_last(doc)
    assert parent is child and token == 'new'
    parent, token = jp.JsonPointer('/list/-').to_last(doc)
    assert parent is doc['list'] and token == '-'
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer('/missing/leaf').to_last(doc)


@pytest.mark.parametrize('inplace', [True, False])
def test_append_to_empty_list(inplace):
    doc = []
    value = object()
    result = jp.JsonPointer('/-').set(doc, value, inplace=inplace)
    assert result == [value]
    assert result[0] is value
    assert (result is doc) is inplace
    if not inplace:
        assert doc == []


def test_large_sequence_index_conversion_error():
    import sys
    if not hasattr(sys, 'get_int_max_str_digits') or not sys.get_int_max_str_digits():
        pytest.skip('Interpreter does not limit integer string conversion')
    token = '9' * (sys.get_int_max_str_digits() + 1)
    with pytest.raises(jp.JsonPointerException, match='^Invalid sequence index: '):
        jp.JsonPointer.get_part([], token)
