"""Behavioral tests derived from the local jsonpointer implementation."""

from collections import UserDict
from copy import deepcopy

import pytest

import jsonpointer as jp
from jsonpointer import JsonPointer, JsonPointerException, resolve_pointer, set_pointer


@pytest.mark.parametrize('value', [None, False, 0, '', [], {}, {'a': [1]}])
def test_root_returns_original(value):
    assert resolve_pointer(value, '') is value
    assert JsonPointer('').get(value) is value
    parent, token = JsonPointer('').to_last(value)
    assert parent is value
    assert token is None


@pytest.mark.parametrize(('path', 'expected'), [
    ('/', 'empty'), ('/a~1b', 'slash'), ('/m~0n', 'tilde'),
    ('/~01', 'literal escape'), ('/a%20b', 'percent'),
    ('/a b', 'space'), ('/বাংলা', 'unicode'), ('//', 'nested empty'),
])
def test_literal_keys_and_escaping(path, expected):
    doc = {'': {'': 'nested empty'}, 'a/b': 'slash', 'm~n': 'tilde',
           '~1': 'literal escape', 'a%20b': 'percent', 'a b': 'space',
           'বাংলা': 'unicode'}
    if path == '/':
        doc[''] = 'empty'
    assert resolve_pointer(doc, path) == expected


@pytest.mark.parametrize('path', ['a', 'a/b', '#/a', ' /a', '/~', '/~2',
                                  '/a~x', '/a/~10~', '/~~0'])
def test_invalid_pointer_syntax(path):
    with pytest.raises(JsonPointerException):
        JsonPointer(path)
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, path, default=None)


@pytest.mark.parametrize(('raw', 'encoded'), [
    ('', ''), ('a/b', 'a~1b'), ('~', '~0'), ('~1', '~01'),
    ('~/~0', '~0~1~00'), ('%20 বাংলা', '%20 বাংলা'),
])
def test_escape_helpers(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


def test_nested_resolution_preserves_identity_and_document():
    target = {'value': [None, False, 0, '']}
    doc = {'items': [42, {'a/b': target}]}
    before = deepcopy(doc)
    pointer = JsonPointer('/items/1/a~1b')
    assert pointer.resolve(doc) is target
    assert pointer.get(doc) is target
    assert pointer.walk(doc, 'items') is doc['items']
    assert doc == before
    assert pointer.to_last(doc) == (doc['items'][1], 'a/b')
    parent, token = JsonPointer('/items/1').to_last(doc)
    assert parent is doc['items']
    assert token == 1
    assert type(token) is int


@pytest.mark.parametrize('token', ['00', '01', '-1', '+1', ' 1', '1 ',
                                   '1.0', '1e0', '', '0x0', '1\n', '١'])
def test_noncanonical_array_indices_rejected(token):
    with pytest.raises(JsonPointerException):
        resolve_pointer([10, 20], '/' + token)
    assert resolve_pointer([10, 20], '/' + token, 'missing') == 'missing'
    assert resolve_pointer({token: 'literal'}, '/' + token) == 'literal'


@pytest.mark.parametrize('doc', [[4, 5], (4, 5)])
def test_sequence_indices_and_bounds(doc):
    assert resolve_pointer(doc, '/0') == 4
    assert resolve_pointer(doc, '/1') == 5
    for index in [2, 100]:
        with pytest.raises(JsonPointerException, match='out of bounds'):
            resolve_pointer(doc, '/' + str(index))


@pytest.mark.parametrize('doc,path', [({}, '/missing'), ({'a': {}}, '/a/b'),
    ([], '/0'), ({'a': [1]}, '/a/2'), (None, '/x'), (3, '/0'),
    (True, '/x'), ('abc', '/0'), ({'a': None}, '/a/b')])
def test_failures_and_explicit_defaults(doc, path):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, path)
    for fallback in [None, False, 0, '', object()]:
        assert resolve_pointer(doc, path, fallback) is fallback
        assert JsonPointer(path).get(doc, fallback) is fallback


@pytest.mark.parametrize('value', [None, False, 0, '', [], {}])
def test_existing_falsy_values_do_not_use_default(value):
    assert resolve_pointer({'key': value}, '/key', object()) is value


def test_end_of_list_is_a_marker_not_last_item():
    items = [1, 2]
    marker = resolve_pointer(items, '/-')
    assert isinstance(marker, jp.EndOfList)
    assert marker.list_ is items
    assert repr(marker) == 'EndOfList([1, 2])'
    assert resolve_pointer({'-': 7}, '/-') == 7
    with pytest.raises(JsonPointerException):
        resolve_pointer(items, '/-/child')


@pytest.mark.parametrize('use_wrapper', [False, True])
def test_set_replaces_adds_and_appends_in_place(use_wrapper):
    doc = {'items': [{'x': 1}], 'other': [9]}
    def assign(path, value):
        if use_wrapper:
            return set_pointer(doc, path, value)
        return JsonPointer(path).set(doc, value)
    assert assign('/items/0/x', 2) is doc
    assert assign('/items/0/a~1b', 3) is doc
    inserted = {'new': True}
    assert assign('/items/-', inserted) is doc
    assert doc == {'items': [{'x': 2, 'a/b': 3}, {'new': True}], 'other': [9]}
    assert doc['items'][1] is inserted
    assert assign('/items/0', None) is doc
    assert doc['items'] == [None, inserted]
    assert assign('/-', 4) is doc
    assert doc['-'] == 4


@pytest.mark.parametrize('path,value', [('/a/0/x', 8), ('/a/-', {'x': 3}), ('/new', 4)])
def test_non_inplace_set_deepcopies_document(path, value):
    doc = {'a': [{'x': 1}], 'untouched': {'nested': []}}
    before = deepcopy(doc)
    result = set_pointer(doc, path, value, inplace=False)
    assert result is not doc
    assert doc == before
    assert result['a'] is not doc['a']
    assert result['a'][0] is not doc['a'][0]
    result['untouched']['nested'].append(2)
    assert doc['untouched']['nested'] == []
    if path.endswith('-'):
        assert result['a'][-1] is value
    else:
        assert resolve_pointer(result, path) == value


def test_root_assignment():
    doc = {'old': [1]}
    value = {'replacement': []}
    with pytest.raises(JsonPointerException, match='root'):
        set_pointer(doc, '', value)
    assert set_pointer(doc, '', value, inplace=False) is value
    assert doc == {'old': [1]}


@pytest.mark.parametrize('doc,path', [({'a': []}, '/a/0'), ({'a': [1]}, '/a/1'),
    ({}, '/missing/child'), ({'a': 'abc'}, '/a/0'),
    ({'a': (1, 2)}, '/a/0'), ({'a': 3}, '/a/x'), ({'a': [1]}, '/a/-1')])
def test_invalid_assignments_do_not_modify_document(doc, path):
    before = deepcopy(doc)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, path, 9)
    assert doc == before


def test_parts_and_representations():
    pointer = JsonPointer.from_parts(iter(['', 'a/b', '~1', 0, 'বাংলা']))
    assert pointer.parts == ['', 'a/b', '~1', '0', 'বাংলা']
    assert pointer.get_parts() == pointer.parts
    assert pointer.path == '//a~1b/~01/0/বাংলা'
    assert str(pointer) == pointer.path
    assert repr(pointer) == 'JsonPointer(' + repr(pointer.path) + ')'
    assert JsonPointer(pointer.path) == pointer
    assert JsonPointer.from_parts([]).path == ''
    assert JsonPointer('/a') != JsonPointer('/b')
    assert JsonPointer('/a') != '/a'
    assert not (JsonPointer('/a') == None)
    assert len({JsonPointer('/a'), JsonPointer.from_parts(['a'])}) == 1
    assert {pointer: 'found'}[JsonPointer(pointer.path)] == 'found'


@pytest.mark.parametrize('suffix', ['/b~1c/~0', JsonPointer('/b~1c/~0'), ['b/c', '~']])
def test_join_and_division(suffix):
    base = JsonPointer('/a')
    assert base.join(suffix).path == '/a/b~1c/~0'
    assert (base / suffix).parts == ['a', 'b/c', '~']
    assert base.path == '/a'
    assert base.join('') == base
    assert base.join('') is not base
    assert JsonPointer('').join(base) == base


@pytest.mark.parametrize('suffix', [None, 42, '/bad~2', 'missing-slash'])
def test_invalid_join(suffix):
    with pytest.raises(JsonPointerException):
        JsonPointer('/a').join(suffix)


def test_subclass_construction_and_join():
    class Pointer(JsonPointer):
        pass
    pointer = Pointer.from_parts(['a'])
    assert isinstance(pointer, Pointer)
    assert isinstance(pointer.join('/b'), Pointer)
    assert repr(pointer) == "Pointer('/a')"


@pytest.mark.parametrize('path,prefix,expected', [
    ('/a/b', '/a', True), ('/a', '/a', True), ('/a', '', True),
    ('', '/a', False), ('/a', '/a/b', False), ('/ab', '/a', False),
    ('/a~1b', '/a', False), ('/a/b', '/b', False),
])
def test_contains_uses_complete_prefix_tokens(path, prefix, expected):
    # This API tests whether the argument is a prefix of self.
    pointer, ancestor = JsonPointer(path), JsonPointer(prefix)
    assert pointer.contains(ancestor) is expected
    assert (ancestor in pointer) is expected


def test_mapping_and_duck_typed_containers():
    assert resolve_pointer(UserDict({'0': ['ok']}), '/0/0') == 'ok'
    class Container:
        def __init__(self):
            self.data = {'key': [7]}
        def __getitem__(self, key):
            return self.data[key]
        def __setitem__(self, key, value):
            self.data[key] = value
    doc = Container()
    assert resolve_pointer(doc, '/key/0') == 7
    assert set_pointer(doc, '/new', 8) is doc
    assert doc.data['new'] == 8
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, '/absent')


@pytest.mark.parametrize('verbose', [True, False])
def test_exception_document_visibility(monkeypatch, verbose):
    monkeypatch.setattr(jp, 'VERBOSE_EXCEPTIONS', verbose)
    doc = {'secret': 'sensitive content'}
    with pytest.raises(JsonPointerException) as error:
        resolve_pointer(doc, '/missing')
    message = str(error.value)
    assert 'missing' in message
    assert ('sensitive content' in message) is verbose


@pytest.mark.parametrize('items,expected', [([], []), ([1], []),
    ([1, 2], [(1, 2)]), ([1, 2, 3], [(1, 2), (2, 3)])])
def test_pairwise_accepts_one_shot_iterators(items, expected):
    assert list(jp.pairwise(iter(items))) == expected
