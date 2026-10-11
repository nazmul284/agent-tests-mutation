"""Behavioral tests written against the local jsonpointer implementation."""
from collections import UserDict, UserList

import pytest

import jsonpointer as jp
from jsonpointer import JsonPointer, JsonPointerException, resolve_pointer, set_pointer


@pytest.mark.parametrize('path, expected', [
    ('/foo', ['bar', 'baz']), ('/foo/0', 'bar'), ('/', 0),
    ('/a~1b', 1), ('/c%d', 2), ('/e^f', 3), ('/g|h', 4),
    ('/i\\j', 5), ('/k"l', 6), ('/ ', 7), ('/m~0n', 8),
    ('/~01', 9), ('/a%20b', 10), ('/বাংলা', 11),
])
def test_literal_and_escaped_members(path, expected):
    doc = {'foo': ['bar', 'baz'], '': 0, 'a/b': 1, 'c%d': 2,
           'e^f': 3, 'g|h': 4, 'i\\j': 5, 'k"l': 6, ' ': 7,
           'm~n': 8, '~1': 9, 'a%20b': 10, 'বাংলা': 11}
    assert resolve_pointer(doc, path) == expected
    assert JsonPointer(path).get(doc) == expected


@pytest.mark.parametrize('doc', [None, False, 0, 'text', [], {}])
def test_root_is_document(doc):
    assert resolve_pointer(doc, '') is doc
    assert JsonPointer('').to_last(doc) == (doc, None)


def test_empty_tokens_and_reference_identity():
    target = []
    doc = {'': {'': target}, 'items': [{'leaf': target}]}
    assert resolve_pointer(doc, '//') is target
    assert resolve_pointer(doc, '/items/0/leaf') is target


@pytest.mark.parametrize('path', ['foo', 'foo/bar', '#/foo', ' /foo',
                                  '/~', '/~2', '/abc~x', '/~~0', '/a/~9/b'])
def test_invalid_pointer_syntax(path):
    with pytest.raises(JsonPointerException):
        JsonPointer(path)
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, path, default=None)


@pytest.mark.parametrize('token', ['00', '01', '0x', '0\n', '-1', '+1', ' 1',
                                    '1 ', '1.0', '1e0', '', 'one', '١'])
def test_invalid_array_indices(token):
    with pytest.raises(JsonPointerException):
        resolve_pointer([10, 20], '/' + token)
    with pytest.raises(JsonPointerException):
        set_pointer([10, 20], '/' + token, 30)
    assert resolve_pointer({token: 42}, '/' + token) == 42


@pytest.mark.parametrize('doc,path', [({}, '/absent'), ({'a': {}}, '/a/b/c'),
    ([], '/0'), ([1], '/1'), ([1], '/100'), ('abc', '/0'),
    ({'a': None}, '/a/b'), (True, '/x'), (12, '/x'), (1.5, '/x'),
    ([1], '/-/child')])
def test_resolution_errors_and_defaults(doc, path):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, path)
    marker = object()
    assert resolve_pointer(doc, path, marker) is marker
    assert JsonPointer(path).resolve(doc, None) is None


@pytest.mark.parametrize('value', [None, False, 0, '', [], {}])
def test_existing_falsey_value_does_not_use_default(value):
    assert resolve_pointer({'a': value}, '/a', object()) is value


def test_sequence_and_mapping_protocols():
    assert resolve_pointer(('a', {'b': 2}), '/1/b') == 2
    assert resolve_pointer(UserList([3, 4]), '/1') == 4
    assert resolve_pointer(UserDict({'01': 7}), '/01') == 7
    assert resolve_pointer({'0': 'string', 0: 'integer', '-': 8}, '/0') == 'string'
    assert resolve_pointer({'-': 8}, '/-') == 8


def test_custom_indexing_and_assignment():
    class Container:
        def __init__(self):
            self.data = {'key': {'child': 3}}

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

    doc = Container()
    assert resolve_pointer(doc, '/key/child') == 3
    assert set_pointer(doc, '/new', 4) is doc
    assert doc.data['new'] == 4
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, '/missing')


def test_end_of_list_marker():
    doc = [1, 2]
    end = resolve_pointer(doc, '/-')
    assert isinstance(end, jp.EndOfList)
    assert end.list_ is doc
    assert repr(end) == 'EndOfList([1, 2])'
    assert isinstance(resolve_pointer([], '/-'), jp.EndOfList)


def test_to_last_returns_parent_and_typed_token():
    items = [{'x': 1}]
    doc = {'items': items}
    parent, token = JsonPointer('/items/0').to_last(doc)
    assert parent is items
    assert token == 0 and type(token) is int
    parent, token = JsonPointer('/items/0/new~1key').to_last(doc)
    assert parent is items[0]
    assert token == 'new/key'
    assert JsonPointer('/items/-').to_last(doc) == (items, '-')
    assert JsonPointer('/01').to_last({}) == ({}, '01')


def test_set_updates_adds_and_appends_in_place():
    doc = {'items': [{'x': 1}, 2], 'other': []}
    assert set_pointer(doc, '/items/0/x', 9) is doc
    assert doc['items'][0] == {'x': 9}
    assert JsonPointer('/items/1').set(doc, 7) is doc
    value = {'new': []}
    assert set_pointer(doc, '/items/-', value) is doc
    assert doc['items'] == [{'x': 9}, 7, value]
    assert doc['items'][-1] is value
    set_pointer(doc, '/a~1b~0c', 5)
    set_pointer(doc, '/', 6)
    set_pointer(doc, '/-', 8)
    assert doc['a/b~c'] == 5 and doc[''] == 6 and doc['-'] == 8


@pytest.mark.parametrize('path,value', [('/items/0/x', 9), ('/items/-', {'y': 2}),
                                      ('/new', 3)])
def test_non_inplace_set_deep_copies_document(path, value):
    doc = {'items': [{'x': 1}], 'untouched': {'nested': []}}
    result = set_pointer(doc, path, value, inplace=False)
    assert doc == {'items': [{'x': 1}], 'untouched': {'nested': []}}
    assert result is not doc
    assert result['items'] is not doc['items']
    assert result['items'][0] is not doc['items'][0]
    assert result['untouched']['nested'] is not doc['untouched']['nested']
    target = '/items/1' if path.endswith('/-') else path
    assert resolve_pointer(result, target) is value


def test_root_assignment_requires_replacement():
    doc = {'old': []}
    value = {'new': []}
    with pytest.raises(JsonPointerException, match='root'):
        set_pointer(doc, '', value)
    assert set_pointer(doc, '', value, inplace=False) is value
    assert doc == {'old': []}


@pytest.mark.parametrize('doc,path', [({}, '/missing/child'), ([1], '/1'),
    ([], '/0'), ((1, 2), '/0'), ('abc', '/0'), ({'a': 'abc'}, '/a/0'),
    ({'a': None}, '/a/x')])
def test_invalid_assignment_is_pointer_exception(doc, path):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, path, 5)


@pytest.mark.parametrize('raw,encoded', [('', ''), ('~', '~0'), ('/', '~1'),
    ('~/~1', '~0~1~01'), ('a/b~c', 'a~1b~0c'), ('%20 বাংলা', '%20 বাংলা')])
def test_escape_and_unescape(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


def test_from_parts_round_trip_and_representation():
    parts = ['a/b', '~1', '', 0, 'বাংলা']
    ptr = JsonPointer.from_parts(iter(parts))
    assert ptr.parts == ['a/b', '~1', '', '0', 'বাংলা']
    assert ptr.get_parts() == ptr.parts
    assert ptr.path == '/a~1b/~01//0/বাংলা'
    assert str(ptr) == ptr.path
    assert repr(ptr) == "JsonPointer('/a~1b/~01//0/বাংলা')"
    assert JsonPointer(ptr.path) == ptr
    assert JsonPointer.from_parts([]).path == ''


@pytest.mark.parametrize('suffix', ['/b~1c/~0', JsonPointer('/b~1c/~0'), ['b/c', '~']])
def test_join_and_division(suffix):
    base = JsonPointer('/a')
    assert base.join(suffix) == JsonPointer('/a/b~1c/~0')
    assert base / suffix == JsonPointer('/a/b~1c/~0')
    assert base.path == '/a'


def test_join_empty_generator_and_subclass():
    class SpecialPointer(JsonPointer):
        pass

    base = SpecialPointer('/a')
    assert isinstance(base.join('/b'), SpecialPointer)
    assert isinstance(SpecialPointer.from_parts(['x']), SpecialPointer)
    assert base.join('') == base
    assert base.join('') is not base
    assert JsonPointer('').join('/a') == base
    assert base.join(x for x in ['', 2]).path == '/a//2'


@pytest.mark.parametrize('suffix', [None, 42, '/bad~2', 'no-leading-slash'])
def test_invalid_join_suffix(suffix):
    with pytest.raises(JsonPointerException):
        JsonPointer('/a').join(suffix)


def test_equality_and_hashing():
    a = JsonPointer('/a~1b/0')
    b = JsonPointer.from_parts(['a/b', 0])
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b, JsonPointer('/a/b/0')}) == 2
    assert {a: 'value'}[b] == 'value'
    assert a != a.path
    assert a != None  # noqa: E711
    assert a != JsonPointer('/a~1b/00')


@pytest.mark.parametrize('path,prefix,expected', [('/a/b', '/a', True),
    ('/a/b', '/a/b', True), ('/a/b', '', True), ('', '/a', False),
    ('/a', '/a/b', False), ('/ab', '/a', False), ('/a~1b', '/a', False),
    ('/a/c', '/a/b', False)])
def test_contains_uses_token_prefix(path, prefix, expected):
    # The implementation tests whether the argument is a prefix of self.
    ptr, ancestor = JsonPointer(path), JsonPointer(prefix)
    assert ptr.contains(ancestor) is expected
    assert (ancestor in ptr) is expected


@pytest.mark.parametrize('verbose', [True, False])
def test_exception_verbosity(monkeypatch, verbose):
    monkeypatch.setattr(jp, 'VERBOSE_EXCEPTIONS', verbose)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({'private': 'unique-secret'}, '/missing')
    assert 'missing' in str(exc.value)
    assert ('unique-secret' in str(exc.value)) is verbose


@pytest.mark.parametrize('items,expected', [([], []), ([1], []),
    ([1, 2], [(1, 2)]), ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])])
def test_pairwise_accepts_one_shot_iterators(items, expected):
    assert list(jp.pairwise(iter(items))) == expected
