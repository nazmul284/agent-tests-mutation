import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jsonpointer
from jsonpointer import (
    EndOfList, JsonPointer, JsonPointerException, escape, pairwise,
    resolve_pointer, set_pointer, unescape,
)


DOC = {
    'foo': ['bar', 'baz'],
    '': 0,
    'a/b': 1,
    'c%d': 2,
    'e^f': 3,
    'g|h': 4,
    'i\\j': 5,
    'k"l': 6,
    ' ': 7,
    'm~n': 8,
    'arr': [{'prop': 44}, [1, 2, 3]],
    'nested': {'x': {'y': 'z'}},
}


# ---------- RFC examples ----------

@pytest.mark.parametrize('ptr,expected', [
    ('/foo', ['bar', 'baz']),
    ('/foo/0', 'bar'),
    ('/foo/1', 'baz'),
    ('/', 0),
    ('/a~1b', 1),
    ('/c%d', 2),
    ('/e^f', 3),
    ('/g|h', 4),
    ('/i\\j', 5),
    ('/k"l', 6),
    ('/ ', 7),
    ('/m~0n', 8),
    ('/arr/0/prop', 44),
    ('/arr/1/2', 3),
    ('/nested/x/y', 'z'),
])
def test_rfc_resolve(ptr, expected):
    assert resolve_pointer(DOC, ptr) == expected
    assert JsonPointer(ptr).resolve(DOC) == expected
    assert JsonPointer(ptr).get(DOC) == expected


def test_resolve_root():
    assert resolve_pointer(DOC, '') is DOC


# ---------- construction / validation ----------

def test_must_start_with_slash():
    for bad in ['foo', 'a/b', ' /a', '~0']:
        with pytest.raises(JsonPointerException, match='Location must start with /'):
            JsonPointer(bad)


@pytest.mark.parametrize('bad', ['/~', '/a~', '/~2', '/~a', '/a~b/c', '/a/~', '/~ ', '/x~/y', '/~01~'])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException, match='Found invalid escape'):
        JsonPointer(bad)


def test_invalid_escape_message():
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer('/a~2')
    assert str(ei.value) == 'Found invalid escape ~2'
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer('/a~')
    assert str(ei.value) == 'Found invalid escape ~'


def test_invalid_escape_checked_before_start():
    with pytest.raises(JsonPointerException, match='invalid escape'):
        JsonPointer('a~')


def test_valid_escapes():
    assert JsonPointer('/~0').parts == ['~']
    assert JsonPointer('/~1').parts == ['/']
    assert JsonPointer('/~01').parts == ['~1']
    assert JsonPointer('/~10').parts == ['/0']
    assert JsonPointer('/~0~1/~1~0').parts == ['~/', '/~']


def test_parts_parsing():
    assert JsonPointer('').parts == []
    assert JsonPointer('/').parts == ['']
    assert JsonPointer('//').parts == ['', '']
    assert JsonPointer('/a/b/c').parts == ['a', 'b', 'c']
    assert JsonPointer('/a//b').parts == ['a', '', 'b']
    assert JsonPointer('/a/').parts == ['a', '']


def test_get_parts():
    p = JsonPointer('/a/b~1c/0')
    assert p.get_parts() == ['a', 'b/c', '0']
    assert p.get_parts() is p.parts


# ---------- escape / unescape ----------

def test_escape():
    assert escape('') == ''
    assert escape('abc') == 'abc'
    assert escape('~') == '~0'
    assert escape('/') == '~1'
    assert escape('~/') == '~0~1'
    assert escape('/~') == '~1~0'
    assert escape('a/b~c') == 'a~1b~0c'
    assert escape('~1') == '~01'
    assert escape('//~~') == '~1~1~0~0'


def test_unescape():
    assert unescape('') == ''
    assert unescape('abc') == 'abc'
    assert unescape('~0') == '~'
    assert unescape('~1') == '/'
    assert unescape('~01') == '~1'
    assert unescape('~00') == '~0'
    assert unescape('~10') == '/0'
    assert unescape('a~1b~0c') == 'a/b~c'
    assert unescape('~1~1~0~0') == '//~~'


@pytest.mark.parametrize('s', ['', '~', '/', '~1', '~0', '~01', '/~', 'a~/b~~//'])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


# ---------- resolve errors / defaults ----------

def test_missing_key_raises():
    with pytest.raises(JsonPointerException, match="member 'nope' not found"):
        resolve_pointer(DOC, '/nope')


def test_missing_key_message_verbose():
    d = {'a': 1}
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(d, '/b')
    assert str(ei.value) == "member 'b' not found in {'a': 1}"


def test_missing_key_message_not_verbose(monkeypatch):
    monkeypatch.setattr(jsonpointer, 'VERBOSE_EXCEPTIONS', False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({'a': 1}, '/b')
    assert str(ei.value) == "member 'b' not found"


def test_default_returned():
    assert resolve_pointer(DOC, '/nope', 'dflt') == 'dflt'
    assert resolve_pointer(DOC, '/nope', None) is None
    assert resolve_pointer(DOC, '/foo/9', 'd') == 'd'
    assert resolve_pointer(DOC, '/foo/x', 'd') == 'd'
    assert resolve_pointer(DOC, '/foo/0/0/0', 'd') == 'd'
    assert JsonPointer('/nope').resolve(DOC, 0) == 0
    assert JsonPointer('/nope').get(DOC, default=False) is False


def test_default_not_used_when_found():
    assert resolve_pointer(DOC, '/foo/0', 'd') == 'bar'
    assert resolve_pointer({'a': None}, '/a', 'd') is None
    assert resolve_pointer(DOC, '', 'd') is DOC


def test_index_out_of_bounds():
    with pytest.raises(JsonPointerException, match="index '2' is out of bounds"):
        resolve_pointer(DOC, '/foo/2')
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([], '/0')
    assert resolve_pointer(['a'], '/0') == 'a'


def test_index_message():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([1], '/5')
    assert str(ei.value) == "index '5' is out of bounds"


@pytest.mark.parametrize('idx', ['01', '00', '-1', '+1', '1.0', ' 1', '1 ', 'a', '', '0x1', '1e0', '1a', '٣'])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException, match='is not a valid sequence index'):
        resolve_pointer([0, 1, 2, 3, 4], '/' + idx)


def test_invalid_array_index_message():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([1, 2], '/01')
    assert str(ei.value) == "'01' is not a valid sequence index"


def test_valid_array_indices():
    lst = list(range(20))
    assert resolve_pointer(lst, '/0') == 0
    assert resolve_pointer(lst, '/10') == 10
    assert resolve_pointer(lst, '/19') == 19
    assert resolve_pointer(lst, '/9') == 9
    assert resolve_pointer(lst, '/1') == 1


def test_negative_index_not_allowed():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], '/-1')


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    r = resolve_pointer(lst, '/-')
    assert isinstance(r, EndOfList)
    assert r.list_ is lst
    assert repr(r) == 'EndOfList([1, 2])'


def test_dash_on_mapping_is_key():
    assert resolve_pointer({'-': 5}, '/-') == 5


def test_non_container():
    with pytest.raises(JsonPointerException):
        resolve_pointer(5, '/a')
    with pytest.raises(JsonPointerException, match='does not support indexing'):
        resolve_pointer({'a': 5}, '/a/b')
    with pytest.raises(JsonPointerException):
        resolve_pointer({'a': None}, '/a/b')


def test_string_not_indexable():
    with pytest.raises(JsonPointerException, match="Cannot apply token 'a' to non-container"):
        resolve_pointer({'s': 'abc'}, '/s/a')
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        resolve_pointer({'s': 'abc'}, '/s/0')
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        resolve_pointer('abc', '/0')


def test_tuple_is_sequence():
    assert resolve_pointer({'t': (1, 2)}, '/t/1') == 2
    with pytest.raises(JsonPointerException):
        resolve_pointer((1,), '/1')
    assert isinstance(resolve_pointer((1,), '/-'), EndOfList)


def test_duck_typed_getitem():
    class Duck:
        def __getitem__(self, k):
            if k == 'a':
                return 'found'
            raise KeyError(k)

    assert resolve_pointer(Duck(), '/a') == 'found'
    with pytest.raises(JsonPointerException):
        resolve_pointer(Duck(), '/b')
    # tokens are not converted to int for duck types
    class Duck2:
        def __getitem__(self, k):
            return repr(k)
    assert resolve_pointer(Duck2(), '/0') == "'0'"


def test_custom_mapping():
    from collections import OrderedDict, UserDict
    assert resolve_pointer(OrderedDict(a=1), '/a') == 1
    assert resolve_pointer(UserDict({'0': 'x'}), '/0') == 'x'


def test_empty_key():
    assert resolve_pointer({'': {'': 1}}, '//') == 1
    assert resolve_pointer({'': 1}, '/') == 1


def test_resolve_mapping_with_numeric_string_keys():
    assert resolve_pointer({'0': 'a', '01': 'b'}, '/01') == 'b'


def test_walk_and_get_part_direct():
    p = JsonPointer('')
    assert p.walk({'a': 1}, 'a') == 1
    assert p.walk([5, 6], '1') == 6
    assert isinstance(p.walk([5], '-'), EndOfList)
    assert JsonPointer.get_part({'a': 1}, 'a') == 'a'
    assert JsonPointer.get_part([1], '0') == 0
    assert JsonPointer.get_part([1], '-') == '-'
    assert JsonPointer.get_part([1], 0) == 0
    assert isinstance(JsonPointer.get_part([1], '12'), int)
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(1, 'a')
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part('s', 'a')


def test_walk_non_str_index():
    p = JsonPointer('')
    assert p.walk([5, 6], 1) == 6


# ---------- to_last ----------

def test_to_last():
    d = {'a': {'b': [1, 2]}}
    parent, last = JsonPointer('/a/b/1').to_last(d)
    assert parent is d['a']['b']
    assert last == 1
    parent, last = JsonPointer('/a/b').to_last(d)
    assert parent is d['a']
    assert last == 'b'
    parent, last = JsonPointer('/a').to_last(d)
    assert parent is d
    assert last == 'a'
    parent, last = JsonPointer('').to_last(d)
    assert parent is d
    assert last is None
    parent, last = JsonPointer('/a/b/-').to_last(d)
    assert parent is d['a']['b']
    assert last == '-'


def test_to_last_missing_intermediate():
    with pytest.raises(JsonPointerException):
        JsonPointer('/x/y').to_last({'a': 1})


def test_to_last_nonexistent_final_ok_for_mapping():
    parent, last = JsonPointer('/a/new').to_last({'a': {}})
    assert parent == {}
    assert last == 'new'


# ---------- set ----------

def test_set_mapping_inplace():
    d = {'a': {'b': 1}}
    r = set_pointer(d, '/a/b', 2)
    assert r is d
    assert d == {'a': {'b': 2}}


def test_set_new_key():
    d = {'a': {}}
    r = set_pointer(d, '/a/new', 'v')
    assert r is d
    assert d == {'a': {'new': 'v'}}


def test_set_list_index():
    d = {'l': [1, 2, 3]}
    set_pointer(d, '/l/1', 'x')
    assert d == {'l': [1, 'x', 3]}
    set_pointer(d, '/l/0', 'y')
    set_pointer(d, '/l/2', 'z')
    assert d == {'l': ['y', 'x', 'z']}


def test_set_list_append():
    d = {'l': [1]}
    r = set_pointer(d, '/l/-', 2)
    assert r is d
    assert d == {'l': [1, 2]}
    set_pointer(d, '/l/-', 3)
    assert d == {'l': [1, 2, 3]}


def test_set_list_append_toplevel():
    l = []
    assert set_pointer(l, '/-', 'a') is l
    assert l == ['a']


def test_set_list_index_out_of_range():
    with pytest.raises(JsonPointerException, match='Invalid assignment target'):
        set_pointer([1], '/5', 'x')
    with pytest.raises(JsonPointerException, match='Invalid assignment target'):
        set_pointer([], '/0', 'x')


def test_set_list_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], '/01', 'x')
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], '/a', 'x')
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], '/-1', 'x')


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException, match='Cannot set root in place'):
        set_pointer({}, '', 1)
    with pytest.raises(JsonPointerException, match='Cannot set root in place'):
        set_pointer({}, '', 1, inplace=True)


def test_set_root_not_inplace():
    d = {'a': 1}
    assert set_pointer(d, '', 5, inplace=False) == 5
    assert d == {'a': 1}
    new = {'b': 2}
    assert set_pointer(d, '', new, inplace=False) is new


def test_set_not_inplace_copies():
    d = {'a': {'b': [1, 2]}}
    orig = copy.deepcopy(d)
    r = set_pointer(d, '/a/b/0', 99, inplace=False)
    assert r == {'a': {'b': [99, 2]}}
    assert d == orig
    assert r is not d
    r2 = set_pointer(d, '/a/c', 1, inplace=False)
    assert r2['a']['c'] == 1
    assert d == orig
    r3 = set_pointer(d, '/a/b/-', 3, inplace=False)
    assert r3['a']['b'] == [1, 2, 3]
    assert d == orig


def test_set_inplace_default_true():
    d = {'a': 1}
    r = JsonPointer('/a').set(d, 2)
    assert r is d and d['a'] == 2


def test_set_inplace_explicit_false_method():
    d = {'a': 1}
    r = JsonPointer('/a').set(d, 2, False)
    assert r is not d and r == {'a': 2} and d == {'a': 1}


def test_set_in_string():
    with pytest.raises(JsonPointerException, match='Cannot apply token'):
        set_pointer({'s': 'abc'}, '/s/0', 'x')


def test_set_string_dash():
    with pytest.raises(JsonPointerException):
        set_pointer({'s': 'abc'}, '/s/-', 'x')


def test_set_missing_intermediate():
    with pytest.raises(JsonPointerException):
        set_pointer({'a': {}}, '/a/b/c', 1)


def test_set_on_immutable_tuple():
    with pytest.raises(JsonPointerException):
        set_pointer({'t': (1, 2)}, '/t/0', 1)


def test_set_tuple_append():
    # tuple has no append -> AttributeError is not wrapped
    with pytest.raises(AttributeError):
        set_pointer({'t': (1,)}, '/t/-', 1)


def test_set_value_types():
    d = {}
    set_pointer(d, '/a', None)
    assert d == {'a': None}
    set_pointer(d, '/b', [1])
    assert d['b'] == [1]


def test_set_dash_on_mapping_sets_key():
    d = {}
    set_pointer(d, '/-', 1)
    assert d == {'-': 1}


def test_set_unhashable_type_error():
    class D:
        def __getitem__(self, k):
            return self
        def __setitem__(self, k, v):
            raise TypeError('nope')
    with pytest.raises(JsonPointerException, match='Invalid assignment target: nope'):
        set_pointer(D(), '/a', 1)


def test_set_invalid_pointer():
    with pytest.raises(JsonPointerException):
        set_pointer({}, 'a', 1)


# ---------- contains / join ----------

def test_contains():
    a = JsonPointer('/a')
    ab = JsonPointer('/a/b')
    assert ab.contains(a)
    assert a in ab
    assert not a.contains(ab)
    assert ab.contains(ab)
    assert ab.contains(JsonPointer(''))
    assert JsonPointer('').contains(JsonPointer(''))
    assert not JsonPointer('').contains(a)
    assert not ab.contains(JsonPointer('/b'))
    assert not ab.contains(JsonPointer('/a/c'))
    assert not JsonPointer('/a/b/c').contains(JsonPointer('/a/c'))
    assert JsonPointer('/a/b/c').contains(JsonPointer('/a/b'))
    assert not ab.contains(JsonPointer('/a/b/c'))
    assert JsonPointer('/ab').contains(JsonPointer('/ab'))
    assert not JsonPointer('/ab').contains(JsonPointer('/a'))
    assert JsonPointer('/a') not in JsonPointer('/b')


def test_join_str():
    p = JsonPointer('/a')
    r = p.join('/b/c')
    assert r.parts == ['a', 'b', 'c']
    assert p.parts == ['a']
    assert isinstance(r, JsonPointer)
    assert p.join('').parts == ['a']
    assert JsonPointer('').join('/x').parts == ['x']


def test_join_pointer():
    r = JsonPointer('/a').join(JsonPointer('/b~1c'))
    assert r.parts == ['a', 'b/c']
    assert r.path == '/a/b~1c'


def test_join_list():
    r = JsonPointer('/a').join(['b', 'c/d', 0])
    assert r.parts == ['a', 'b', 'c/d', '0']
    assert r.path == '/a/b/c~1d/0'


def test_join_invalid():
    with pytest.raises(JsonPointerException):
        JsonPointer('/a').join('b')
    with pytest.raises(JsonPointerException, match='Invalid suffix'):
        JsonPointer('/a').join(5)
    with pytest.raises(JsonPointerException, match='Invalid suffix'):
        JsonPointer('/a').join(None)


def test_truediv():
    p = JsonPointer('/a') / 'b' if False else JsonPointer('/a') / '/b'
    assert p.parts == ['a', 'b']
    assert (JsonPointer('/a') / JsonPointer('/b')).parts == ['a', 'b']
    assert (JsonPointer('/a') / ['x', 'y']).parts == ['a', 'x', 'y']
    assert (JsonPointer('') / '/a' / '/b').path == '/a/b'


def test_join_subclass_preserved():
    class Sub(JsonPointer):
        pass
    assert type(Sub('/a').join('/b')) is Sub


# ---------- path / str / repr / eq / hash ----------

def test_path():
    assert JsonPointer('').path == ''
    assert JsonPointer('/').path == '/'
    assert JsonPointer('/a/b').path == '/a/b'
    assert JsonPointer('/~0/0/~1').path == '/~0/0/~1'
    assert JsonPointer('//').path == '//'
    assert JsonPointer('/a~01').path == '/a~01'


def test_str_repr():
    p = JsonPointer('/a~1b/c')
    assert str(p) == '/a~1b/c'
    assert repr(p) == "JsonPointer('/a~1b/c')"
    assert repr(JsonPointer('')) == "JsonPointer('')"

    class Sub(JsonPointer):
        pass
    assert repr(Sub('/x')) == "Sub('/x')"


def test_eq():
    assert JsonPointer('/a/b') == JsonPointer('/a/b')
    assert JsonPointer('/a/b') != JsonPointer('/a/c')
    assert JsonPointer('/a/b') != JsonPointer('/a')
    assert JsonPointer('') == JsonPointer('')
    assert JsonPointer('/a') != '/a'
    assert not (JsonPointer('/a') == '/a')
    assert JsonPointer('/a') != None  # noqa
    assert JsonPointer('/a') != ['a']
    assert JsonPointer('/~01') == JsonPointer('/~01')
    assert JsonPointer('/~01') != JsonPointer('/~10')


def test_hash():
    assert hash(JsonPointer('/a/b')) == hash(JsonPointer('/a/b'))
    assert hash(JsonPointer('/a/b')) == hash(('a', 'b'))
    assert hash(JsonPointer('')) == hash(())
    s = {JsonPointer('/a'), JsonPointer('/a'), JsonPointer('/b')}
    assert len(s) == 2


# ---------- from_parts ----------

def test_from_parts():
    assert JsonPointer.from_parts(['a', '~', '/', 0]).path == '/a/~0/~1/0'
    assert JsonPointer.from_parts([]).path == ''
    assert JsonPointer.from_parts([]).parts == []
    assert JsonPointer.from_parts(['']).parts == ['']
    assert JsonPointer.from_parts(['a', 'b']).parts == ['a', 'b']
    assert JsonPointer.from_parts(['a/b', '~1']).parts == ['a/b', '~1']
    assert JsonPointer.from_parts([1, 2]).parts == ['1', '2']
    assert JsonPointer.from_parts(iter(['x', 'y'])).parts == ['x', 'y']


def test_from_parts_subclass():
    class Sub(JsonPointer):
        pass
    assert type(Sub.from_parts(['a'])) is Sub


def test_roundtrip_parts():
    parts = ['a', '', '~', '/', '~1', '~0', 'x/y~z']
    p = JsonPointer.from_parts(parts)
    assert p.parts == parts
    assert JsonPointer(p.path).parts == parts


# ---------- pairwise ----------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter('abc'))) == [('a', 'b'), ('b', 'c')]


# ---------- misc ----------

def test_module_metadata_and_flags():
    assert jsonpointer.VERBOSE_EXCEPTIONS is True
    assert jsonpointer.__version__ == '3.2.1'


def test_end_of_list_repr():
    assert repr(EndOfList([])) == 'EndOfList([])'
    assert repr(EndOfList([1, 'a'])) == "EndOfList([1, 'a'])"


def test_resolve_does_not_mutate():
    d = copy.deepcopy(DOC)
    resolve_pointer(d, '/foo/0')
    assert d == DOC


def test_resolve_deep_list_in_list():
    assert resolve_pointer([[1, 2], [3, 4]], '/1/0') == 3


def test_resolve_error_midway_with_default():
    assert resolve_pointer({'a': [1]}, '/a/0/b', 'd') == 'd'
    assert resolve_pointer({'a': [1]}, '/b/0', 'd') == 'd'
