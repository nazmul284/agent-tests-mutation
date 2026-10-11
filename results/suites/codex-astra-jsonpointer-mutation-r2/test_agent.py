import copy
from collections import UserDict

import pytest

import jsonpointer as jp


@pytest.mark.parametrize('raw,encoded', [('', ''), ('abc', 'abc'), ('/', '~1'), ('~', '~0'), ('~/~1', '~0~1~01'), ('a/b~c', 'a~1b~0c'), ('é %20', 'é %20')])
def test_escaping(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


@pytest.mark.parametrize('path,parts', [('', []), ('/', ['']), ('//', ['', '']), ('/a//b/', ['a', '', 'b', '']), ('/~01/~10/~00', ['~1', '/0', '~0']), ('/a%20b/雪', ['a%20b', '雪'])])
def test_parse_and_render(path, parts):
    ptr = jp.JsonPointer(path)
    assert ptr.parts == parts
    assert ptr.get_parts() == parts
    assert ptr.path == path
    assert str(ptr) == path
    assert repr(ptr) == 'JsonPointer(%r)' % path
    assert jp.JsonPointer.from_parts(parts) == ptr


@pytest.mark.parametrize('path', ['a', 'a/b', '#/a', ' /a', '/~', '/a~2b', '/~x', '/~~0', '/ok/~3/end'])
def test_bad_pointer(path):
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer(path)


def test_from_parts_and_subclass():
    class Pointer(jp.JsonPointer):
        pass
    ptr = Pointer.from_parts(iter(['a/b', '~', 0, '', '雪']))
    assert type(ptr) is Pointer
    assert ptr.path == '/a~1b/~0/0//雪'
    assert repr(ptr) == "Pointer('/a~1b/~0/0//雪')"
    assert isinstance(ptr.join('/end'), Pointer)


@pytest.mark.parametrize('path,expected', [('/a/0/x', 9), ('/a/1', None), ('/', 'empty'), ('/a~1b', 'slash'), ('/m~0n', 'tilde'), ('/a%20b', 'percent'), ('/a b', 'space'), ('/01', 'numeric-key'), ('/-', 'dash')])
def test_resolve_document(path, expected):
    doc = {'a': [{'x': 9}, None], '': 'empty', 'a/b': 'slash', 'm~n': 'tilde', 'a%20b': 'percent', 'a b': 'space', '01': 'numeric-key', '-': 'dash'}
    assert jp.resolve_pointer(doc, path) == expected
    assert jp.JsonPointer(path).get(doc) == expected


@pytest.mark.parametrize('doc', [{}, [], None, 0, False, 'abc'])
def test_root_identity(doc):
    assert jp.resolve_pointer(doc, '') is doc
    parent, part = jp.JsonPointer('').to_last(doc)
    assert parent is doc
    assert part is None


@pytest.mark.parametrize('doc,path', [({}, '/missing'), ({'a': {}}, '/a/missing/deeper'), ([], '/0'), ([1], '/1'), ([1], '/-1'), ({'a': None}, '/a/x'), ('abc', '/0'), (12, '/a'), (False, '/x')])
def test_lookup_failures_and_defaults(doc, path):
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(doc, path)
    for default in [None, False, 0, '', object()]:
        assert jp.resolve_pointer(doc, path, default) is default
        assert jp.JsonPointer(path).resolve(doc, default) is default


def test_defaults_do_not_replace_values_or_hide_syntax():
    for value in [None, False, 0, '', [], {}]:
        assert jp.resolve_pointer({'a': value}, '/a', object()) is value
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer({}, 'invalid', None)


@pytest.mark.parametrize('token', ['00', '01', '-1', '+1', ' 1', '1 ', '1.0', '1x', '0x', '', '١', '1\n'])
def test_invalid_array_tokens(token):
    with pytest.raises(jp.JsonPointerException, match='not a valid sequence index'):
        jp.JsonPointer.get_part([0, 1], token)


@pytest.mark.parametrize('token,expected', [('0', 0), ('1', 1), ('10', 10), ('123', 123), (0, 0), (12, 12), ('-', '-')])
def test_array_token_conversion(token, expected):
    assert jp.JsonPointer.get_part([], token) == expected


def test_mapping_and_sequence_types():
    doc = UserDict({'01': [4, 5], '-': 3})
    assert jp.JsonPointer.get_part(doc, '01') == '01'
    assert jp.resolve_pointer(doc, '/01/1') == 5
    assert jp.resolve_pointer(doc, '/-') == 3
    assert jp.resolve_pointer((8, 9), '/0') == 8
    assert jp.resolve_pointer(b'ab', '/1') == 98


def test_end_of_list():
    doc = [1, {'a': 2}]
    end = jp.resolve_pointer(doc, '/-')
    assert isinstance(end, jp.EndOfList)
    assert end.list_ is doc
    assert repr(end) == "EndOfList([1, {'a': 2}])"
    assert jp.resolve_pointer(doc, '/-/x', None) is None


def test_to_last():
    child = [4, {'x': 2}]
    doc = {'a': child}
    parent, part = jp.JsonPointer('/a/1/x').to_last(doc)
    assert parent is child[1]
    assert part == 'x'
    parent, part = jp.JsonPointer('/a/0').to_last(doc)
    assert parent is child and part == 0
    assert jp.JsonPointer('/a/-').to_last(doc) == (child, '-')
    assert jp.JsonPointer('/missing').to_last(doc) == (doc, 'missing')
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer('/missing/x').to_last(doc)


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('path,value,expected', [('/a/0/x', 7, {'a': [{'x': 7}, 2], 'other': []}), ('/a/1', 8, {'a': [{'x': 1}, 8], 'other': []}), ('/a/-', 9, {'a': [{'x': 1}, 2, 9], 'other': []}), ('/new~1key', 3, {'a': [{'x': 1}, 2], 'other': [], 'new/key': 3})])
def test_set_and_copy(inplace, path, value, expected):
    doc = {'a': [{'x': 1}, 2], 'other': []}
    before = copy.deepcopy(doc)
    result = jp.set_pointer(doc, path, value, inplace=inplace)
    assert result == expected
    if inplace:
        assert result is doc
    else:
        assert doc == before
        assert result is not doc
        assert result['a'] is not doc['a']
        assert result['a'][0] is not doc['a'][0]
        assert result['other'] is not doc['other']


def test_set_defaults_and_root():
    doc = {'a': 1}
    value = []
    assert jp.JsonPointer('/a').set(doc, value) is doc
    assert doc['a'] is value
    assert jp.set_pointer(doc, '/-', 4) is doc
    assert doc['-'] == 4
    for setter in [lambda: jp.set_pointer(doc, '', value), lambda: jp.JsonPointer('').set(doc, value)]:
        with pytest.raises(jp.JsonPointerException, match='Cannot set root in place'):
            setter()
    assert jp.set_pointer(doc, '', value, inplace=False) is value
    assert doc == {'a': [], '-': 4}


@pytest.mark.parametrize('doc,path', [([1], '/1'), ([1], '/-1'), ((1, 2), '/0'), ({'a': 'abc'}, '/a/0'), ({}, '/missing/x'), ({'a': 4}, '/a/x')])
def test_set_invalid_targets(doc, path):
    before = copy.deepcopy(doc)
    with pytest.raises(jp.JsonPointerException):
        jp.set_pointer(doc, path, 9)
    assert doc == before


class Indexed:
    def __init__(self):
        self.values = {'key': {'x': 7}}

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value


def test_duck_indexing():
    doc = Indexed()
    assert jp.resolve_pointer(doc, '/key/x') == 7
    assert jp.JsonPointer.get_part(doc, '01') == '01'
    assert jp.set_pointer(doc, '/new', 4) is doc
    assert doc.values['new'] == 4
    with pytest.raises(jp.JsonPointerException, match="member 'absent' not found"):
        jp.resolve_pointer(doc, '/absent')


def test_unrelated_exceptions_propagate():
    class Broken:
        def __getitem__(self, key):
            raise RuntimeError('broken')
    with pytest.raises(RuntimeError, match='broken'):
        jp.resolve_pointer(Broken(), '/x', None)


@pytest.mark.parametrize('verbose', [True, False])
def test_missing_member_message(monkeypatch, verbose):
    monkeypatch.setattr(jp, 'VERBOSE_EXCEPTIONS', verbose)
    doc = {'secret': 123}
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.resolve_pointer(doc, '/missing')
    expected = "member 'missing' not found"
    if verbose:
        expected += ' in ' + str(doc)
    assert str(exc.value) == expected


@pytest.mark.parametrize('suffix', ['/b~1c/~0', jp.JsonPointer('/b~1c/~0'), ['b/c', '~'], ('b/c', '~')])
def test_join(suffix):
    ptr = jp.JsonPointer('/a')
    result = ptr.join(suffix)
    assert result.path == '/a/b~1c/~0'
    assert (ptr / suffix) == result
    assert ptr.path == '/a'
    assert result is not ptr


def test_join_empty_generator_and_invalid():
    ptr = jp.JsonPointer('/a')
    assert ptr.join('').path == '/a'
    assert ptr.join(iter([0, '', '/'])).path == '/a/0//~1'
    assert jp.JsonPointer('').join('/a') == ptr
    for suffix in [None, 3, object()]:
        with pytest.raises(jp.JsonPointerException, match='Invalid suffix'):
            ptr.join(suffix)
    with pytest.raises(jp.JsonPointerException):
        ptr.join('b')


@pytest.mark.parametrize('left,right,expected', [('', '', True), ('/a', '', True), ('', '/a', False), ('/a/b', '/a', True), ('/a', '/a/b', False), ('/ab', '/a', False), ('/a/b', '/a/c', False), ('/a', '/a', True)])
def test_containment(left, right, expected):
    a, b = jp.JsonPointer(left), jp.JsonPointer(right)
    assert a.contains(b) is expected
    assert (b in a) is expected


def test_equality_and_hash():
    a = jp.JsonPointer('/a/~1')
    b = jp.JsonPointer.from_parts(['a', '/'])
    assert a == b
    assert hash(a) == hash(b) == hash(('a', '/'))
    assert len({a, b, jp.JsonPointer('/a')}) == 2
    assert a != jp.JsonPointer('/a/~0')
    assert a != '/a/~1'
    assert a != None
    assert jp.JsonPointer('') != jp.JsonPointer('/')


@pytest.mark.parametrize('items,expected', [([], []), ([1], []), ([1, 2], [(1, 2)]), ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])])
def test_pairwise(items, expected):
    assert list(jp.pairwise(iter(items))) == expected
