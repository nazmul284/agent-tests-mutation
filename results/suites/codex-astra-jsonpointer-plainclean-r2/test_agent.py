"""Behavioral tests for the public JSON Pointer API."""

from collections import UserDict, UserList
from copy import deepcopy

import pytest

import jsonpointer as jp


@pytest.mark.parametrize("doc", [{}, [], None, False, 0, "text"])
def test_empty_pointer_returns_document_itself(doc):
    assert jp.resolve_pointer(doc, "") is doc
    assert jp.JsonPointer("").get(doc) is doc
    parent, token = jp.JsonPointer("").to_last(doc)
    assert parent is doc
    assert token is None


@pytest.mark.parametrize("key, encoded", [
    ("", ""), ("a/b", "a~1b"), ("a~b", "a~0b"),
    ("~1", "~01"), ("~/", "~0~1"), ("~0", "~00"),
    ("a%20b", "a%20b"), ("a b", "a b"), ("雪", "雪"),
    ('a"b', 'a"b'), ("a\\b", "a\\b"), ("\n", "\n"),
])
def test_literal_keys_and_escape_round_trip(key, encoded):
    value = object()
    pointer = jp.JsonPointer("/" + encoded)
    assert pointer.resolve({key: value}) is value
    assert pointer.get_parts() == [key]
    assert pointer.path == "/" + encoded
    assert str(pointer) == pointer.path
    assert jp.escape(key) == encoded
    assert jp.unescape(encoded) == key
    assert jp.JsonPointer.from_parts([key]) == pointer


@pytest.mark.parametrize("path", ["a", "a/b", "#/a", " /a", "/~", "/~2", "/a~x", "/~~0", "/ok/bad~"])
def test_invalid_syntax_is_rejected_even_with_default(path):
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer(path)
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer({}, path, None)


def test_nested_lookup_preserves_identity_and_empty_tokens():
    leaf = {"answer": 42}
    doc = {"": {"": leaf}, "rows": [{"a/b": leaf}]}
    assert jp.resolve_pointer(doc, "//") is leaf
    assert jp.resolve_pointer(doc, "/rows/0/a~1b") is leaf
    pointer = jp.JsonPointer("/rows/0/a~1b")
    parent, token = pointer.to_last(doc)
    assert parent is doc["rows"][0]
    assert token == "a/b"
    parent, token = jp.JsonPointer("/rows/0").to_last(doc)
    assert parent is doc["rows"]
    assert token == 0


@pytest.mark.parametrize("token", ["00", "01", "0x", "0\n", "1\n", "-1", "+1", " 1", "1 ", "1.0", "", "١", "x"])
def test_array_indices_are_strict_but_mapping_keys_are_literal(token):
    path = "/" + token
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer([10, 20], path)
    assert jp.resolve_pointer({token: 99}, path) == 99


@pytest.mark.parametrize("sequence", [[10, 20], (10, 20), UserList([10, 20])])
def test_sequence_lookup_and_end_marker(sequence):
    assert jp.resolve_pointer(sequence, "/0") == 10
    assert jp.resolve_pointer(sequence, "/1") == 20
    marker = jp.resolve_pointer(sequence, "/-")
    assert isinstance(marker, jp.EndOfList)
    assert marker.list_ is sequence
    assert repr(marker) == "EndOfList(" + repr(sequence) + ")"
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(sequence, "/2")


@pytest.mark.parametrize("doc, path", [
    ({}, "/missing"), ({"a": {}}, "/a/b/c"), ([], "/0"),
    ([1], "/9"), ([1], "/bad"), ([1], "/-/x"),
    (None, "/x"), (True, "/x"), (7, "/x"), (1.5, "/x"),
    ("abc", "/0"), ({"a": None}, "/a/b"),
])
def test_resolution_failures_and_explicit_defaults(doc, path):
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(doc, path)
    for default in [None, False, 0, "", [], object()]:
        assert jp.resolve_pointer(doc, path, default) is default
        assert jp.JsonPointer(path).get(doc, default) is default


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_existing_falsey_values_do_not_trigger_default(value):
    assert jp.resolve_pointer({"a": value}, "/a", object()) is value


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("path", ["/rows/0/a~1b", "/rows/0/new", "/rows/1", "/rows/-", "/new"])
def test_assignment_and_copy_isolation(path, inplace):
    doc = {"rows": [{"a/b": 1}, 2], "untouched": {"nested": []}}
    before = deepcopy(doc)
    value = {"replacement": []}
    result = jp.set_pointer(doc, path, value, inplace=inplace)
    expected = deepcopy(before)
    if path == "/rows/0/a~1b":
        expected["rows"][0]["a/b"] = value
    elif path == "/rows/0/new":
        expected["rows"][0]["new"] = value
    elif path == "/rows/1":
        expected["rows"][1] = value
    elif path == "/rows/-":
        expected["rows"].append(value)
    else:
        expected["new"] = value
    assert result == expected
    target = "/rows/2" if path == "/rows/-" else path
    assert jp.resolve_pointer(result, target) is value
    if inplace:
        assert result is doc
    else:
        assert doc == before
        assert result is not doc
        assert result["rows"] is not doc["rows"]
        result["untouched"]["nested"].append(1)
        assert doc == before


def test_root_replacement_requires_non_inplace_mode():
    doc = {"a": 1}
    value = []
    with pytest.raises(jp.JsonPointerException):
        jp.set_pointer(doc, "", value)
    assert jp.set_pointer(doc, "", value, inplace=False) is value
    assert doc == {"a": 1}


def test_assignment_to_empty_and_dash_mapping_keys_and_empty_list():
    doc = {}
    assert jp.JsonPointer("/").set(doc, 1) is doc
    jp.set_pointer(doc, "/-", 2)
    assert doc == {"": 1, "-": 2}
    items = []
    assert jp.set_pointer(items, "/-", "first") is items
    assert items == ["first"]


@pytest.mark.parametrize("doc, path", [
    ({}, "/missing/child"), ([1], "/1"), ([], "/0"),
    ([1], "/-1"), ([1], "/01"), ((1, 2), "/0"),
    ({"a": "abc"}, "/a/0"), ({"a": None}, "/a/b"),
    ([1], "/-/child"),
])
def test_invalid_assignment_does_not_change_document(doc, path):
    before = deepcopy(doc)
    with pytest.raises(jp.JsonPointerException):
        jp.set_pointer(doc, path, "new")
    assert doc == before


def test_mapping_and_duck_typed_containers():
    class Container:
        def __init__(self):
            self.values = {"01": {"x": 7}}

        def __getitem__(self, key):
            return self.values[key]

        def __setitem__(self, key, value):
            self.values[key] = value

    for doc in [UserDict({"01": {"x": 7}}), Container()]:
        assert jp.resolve_pointer(doc, "/01/x") == 7
        assert jp.set_pointer(doc, "/new", 8) is doc
        assert doc["new"] == 8
        with pytest.raises(jp.JsonPointerException):
            jp.resolve_pointer(doc, "/absent")


@pytest.mark.parametrize("verbose", [True, False])
def test_exception_verbosity_setting(monkeypatch, verbose):
    monkeypatch.setattr(jp, "VERBOSE_EXCEPTIONS", verbose)
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.resolve_pointer({"secret": "sensitive-content"}, "/missing")
    assert "missing" in str(exc.value)
    assert ("sensitive-content" in str(exc.value)) is verbose


@pytest.mark.parametrize("suffix", ["/a~1b/~0/0", jp.JsonPointer("/a~1b/~0/0"), ["a/b", "~", 0]])
def test_join_and_division_accept_supported_suffix_types(suffix):
    base = jp.JsonPointer("/base")
    expected = jp.JsonPointer("/base/a~1b/~0/0")
    assert base.join(suffix) == expected
    assert base / suffix == expected
    assert base.path == "/base"


def test_from_parts_generator_and_join_identity():
    pointer = jp.JsonPointer.from_parts(p for p in ["", "a/b", "~1", 12])
    assert pointer.path == "//a~1b/~01/12"
    assert pointer.parts == ["", "a/b", "~1", "12"]
    assert jp.JsonPointer.from_parts([]).path == ""
    assert pointer.join("") == pointer
    assert pointer.join([]) is not pointer
    assert jp.JsonPointer("").join(pointer) == pointer
    assert pointer.join(iter(["last"])).path == pointer.path + "/last"


@pytest.mark.parametrize("suffix", [None, 12, object(), "relative", "/bad~2"])
def test_invalid_join_suffix(suffix):
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer("/base").join(suffix)


@pytest.mark.parametrize("prefix, expected", [("", True), ("/a", True), ("/a/b", True), ("/a/b/c", False), ("/ab", False), ("/b", False)])
def test_containment_compares_whole_token_prefixes(prefix, expected):
    pointer = jp.JsonPointer("/a/b")
    candidate = jp.JsonPointer(prefix)
    assert pointer.contains(candidate) is expected
    assert (candidate in pointer) is expected


def test_equality_hash_and_repr():
    pointer = jp.JsonPointer("/a~1b/~01")
    equal = jp.JsonPointer.from_parts(["a/b", "~1"])
    assert pointer == equal
    assert hash(pointer) == hash(equal)
    assert {pointer: "value"}[equal] == "value"
    assert len({pointer, equal}) == 1
    assert pointer != jp.JsonPointer("/a/b/~01")
    assert pointer != pointer.path
    assert pointer != None
    assert repr(pointer) == "JsonPointer('/a~1b/~01')"


def test_factories_preserve_subclass():
    class Pointer(jp.JsonPointer):
        pass

    pointer = Pointer.from_parts(["a"])
    assert isinstance(pointer, Pointer)
    assert isinstance(pointer.join("/b"), Pointer)
    assert repr(pointer) == "Pointer('/a')"


@pytest.mark.parametrize("items, expected", [([], []), ([1], []), ([1, 2, 3], [(1, 2), (2, 3)])])
def test_pairwise_supports_single_pass_iterables(items, expected):
    assert list(jp.pairwise(iter(items))) == expected
