"""Behavioral regression tests for the local JSON Pointer implementation."""

from collections import UserDict
from itertools import product

import pytest

import jsonpointer as jp
from jsonpointer import JsonPointer, JsonPointerException, resolve_pointer, set_pointer


@pytest.fixture
def document():
    return {
        "": "empty key",
        "foo": ["first", {"answer": 42}],
        "a/b": "slash",
        "m~n": "tilde",
        "~1": "literal escape",
        "a%20b": "percent",
        "a b": "space",
        "雪": "unicode",
        "quote\"\\": "punctuation",
    }


@pytest.mark.parametrize("path, expected", [
    ("/", "empty key"), ("/foo/0", "first"), ("/foo/1/answer", 42),
    ("/a~1b", "slash"), ("/m~0n", "tilde"), ("/~01", "literal escape"),
    ("/a%20b", "percent"), ("/a b", "space"), ("/雪", "unicode"),
    ('/quote"\\', "punctuation"),
])
def test_resolve_tokens(document, path, expected):
    pointer = JsonPointer(path)
    assert resolve_pointer(document, path) == expected
    assert pointer.resolve(document) == expected
    assert pointer.get(document) == expected
    assert pointer.path == path


@pytest.mark.parametrize("doc", [None, False, 0, "hello", [], {}])
def test_empty_pointer_returns_original_root(doc):
    assert resolve_pointer(doc, "") is doc
    assert JsonPointer("").to_last(doc) == (doc, None)
    assert JsonPointer("").parts == []


def test_nested_empty_tokens_and_reference_identity():
    leaf = []
    doc = {"": {"": leaf}}
    assert resolve_pointer(doc, "//") is leaf
    assert JsonPointer("//").parts == ["", ""]


@pytest.mark.parametrize("path", ["foo", "foo/bar", "#", "#/foo", "/~", "/~2", "/a~x", "/a/~10~"])
def test_invalid_pointer_syntax_even_with_default(path):
    with pytest.raises(JsonPointerException):
        JsonPointer(path)
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, path, default=None)


@pytest.mark.parametrize("token", ["00", "01", "-1", "+1", " 1", "1 ", "1.0", "1e0", "", "١", "0x0", "1\n"])
def test_noncanonical_array_indices_rejected_but_valid_mapping_keys(token):
    path = JsonPointer.from_parts([token]).path
    with pytest.raises(JsonPointerException):
        resolve_pointer([10, 20], path)
    assert resolve_pointer({token: "value"}, path) == "value"


@pytest.mark.parametrize("doc,path", [
    ({}, "/missing"), ({"a": {}}, "/a/missing/child"),
    ([], "/0"), ([1], "/1"), ([1], "/9999999999999999999999"),
    (None, "/x"), (True, "/x"), (12, "/x"), (1.5, "/x"),
    ("abc", "/0"), ({"a": None}, "/a/x"), ([1], "/-/child"),
])
def test_resolution_errors_and_default(doc, path):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, path)
    sentinel = object()
    assert resolve_pointer(doc, path, sentinel) is sentinel
    assert JsonPointer(path).resolve(doc, None) is None


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_existing_falsey_value_does_not_use_default(value):
    assert resolve_pointer({"a": value}, "/a", object()) is value


def test_sequence_and_mapping_protocols():
    assert resolve_pointer(("a", "b"), "/1") == "b"
    assert resolve_pointer(UserDict({"a": [7]}), "/a/0") == 7
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part({"0": 1}, "0") == "0"
    assert resolve_pointer({"-": 4}, "/-") == 4
    assert resolve_pointer(list(range(15)), "/12") == 12


def test_duck_typed_indexing():
    class Indexable:
        def __getitem__(self, key):
            return {"item": {"value": 8}}[key]

    assert resolve_pointer(Indexable(), "/item/value") == 8
    with pytest.raises(JsonPointerException):
        resolve_pointer(Indexable(), "/absent")


def test_end_of_list_marker_keeps_original_sequence():
    values = [1, 2]
    marker = resolve_pointer(values, "/-")
    assert isinstance(marker, jp.EndOfList)
    assert marker.list_ is values
    assert repr(marker) == "EndOfList([1, 2])"
    values.append(3)
    assert marker.list_ == [1, 2, 3]


def test_to_last_returns_parent_and_converted_key():
    doc = {"items": [{"a/b": 1}]}
    parent, key = JsonPointer("/items/0/a~1b").to_last(doc)
    assert parent is doc["items"][0]
    assert key == "a/b"
    parent, key = JsonPointer("/items/0").to_last(doc)
    assert parent is doc["items"]
    assert key == 0
    assert JsonPointer("/items/-").to_last(doc) == (doc["items"], "-")
    assert JsonPointer("/new").to_last(doc) == (doc, "new")


@pytest.mark.parametrize("use_wrapper", [False, True])
def test_set_replaces_adds_and_appends_in_place(use_wrapper):
    doc = {"items": [{"value": 1}], "other": []}
    original_items = doc["items"]
    def assign(path, value):
        if use_wrapper:
            return set_pointer(doc, path, value)
        return JsonPointer(path).set(doc, value)
    assert assign("/items/0/value", 2) is doc
    assert assign("/a~1b", 3) is doc
    assert assign("/items/-", {"value": 4}) is doc
    assert assign("/items/0", 5) is doc
    assert assign("/", 6) is doc
    assert doc == {"items": [5, {"value": 4}], "other": [], "a/b": 3, "": 6}
    assert doc["items"] is original_items


@pytest.mark.parametrize("path,value,expected", [
    ("/items/0/value", 9, {"items": [{"value": 9}], "other": []}),
    ("/items/-", 2, {"items": [{"value": 1}, 2], "other": []}),
])
def test_set_without_inplace_deepcopies_document(path, value, expected):
    doc = {"items": [{"value": 1}], "other": []}
    result = set_pointer(doc, path, value, inplace=False)
    assert result == expected
    assert doc == {"items": [{"value": 1}], "other": []}
    assert result is not doc
    assert result["items"] is not doc["items"]
    assert result["items"][0] is not doc["items"][0]
    assert result["other"] is not doc["other"]


def test_root_replacement_requires_non_inplace():
    doc = {"old": []}
    replacement = ["new"]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "", replacement)
    assert set_pointer(doc, "", replacement, inplace=False) is replacement
    assert doc == {"old": []}


@pytest.mark.parametrize("doc,path", [
    ({}, "/missing/child"), ([], "/0"), ([1], "/1"), ([1], "/-1"),
    ([1], "/01"), ((1,), "/0"), ("abc", "/0"),
    ({"a": "abc"}, "/a/0"), ({"a": None}, "/a/x"),
])
def test_invalid_set_targets(doc, path):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, path, "new")


def test_dash_is_normal_mapping_key_when_setting():
    doc = {}
    assert set_pointer(doc, "/-", 3) == {"-": 3}


@pytest.mark.parametrize("raw,encoded", [
    ("", ""), ("/", "~1"), ("~", "~0"), ("~/", "~0~1"),
    ("~1", "~01"), ("/~0", "~1~00"), ("a%20b 雪", "a%20b 雪"),
])
def test_escape_and_unescape(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


def test_short_tokens_roundtrip_without_collisions():
    tokens = ["".join(chars) for n in range(4) for chars in product("a/~01", repeat=n)]
    paths = set()
    for token in tokens:
        ptr = JsonPointer.from_parts([token])
        assert ptr.get_parts() == [token]
        assert JsonPointer(ptr.path) == ptr
        assert ptr.resolve({token: 123}) == 123
        paths.add(ptr.path)
    assert len(paths) == len(tokens)


def test_from_parts_converts_values_and_consumes_generator():
    ptr = JsonPointer.from_parts(part for part in ["a/b", "~", 0, ""])
    assert ptr.path == "/a~1b/~0/0/"
    assert ptr.parts == ["a/b", "~", "0", ""]
    assert JsonPointer.from_parts([]).path == ""


@pytest.mark.parametrize("suffix", ["/b~1c/~0", JsonPointer("/b~1c/~0"), ["b/c", "~"], ("b/c", "~")])
def test_join_and_division_preserve_tokens_and_base(suffix):
    base = JsonPointer("/a")
    assert base.join(suffix) == JsonPointer("/a/b~1c/~0")
    assert base / suffix == JsonPointer("/a/b~1c/~0")
    assert base.path == "/a"
    assert base.join("") == base
    assert base.join("") is not base
    assert JsonPointer("").join(base) == base


@pytest.mark.parametrize("suffix", [None, 3, "not/a/pointer", "/~bad"])
def test_invalid_join(suffix):
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(suffix)


def test_pointer_equality_hash_and_representations():
    a = JsonPointer("/a~1b/~0/0")
    b = JsonPointer.from_parts(["a/b", "~", 0])
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert {a: "found"}[b] == "found"
    assert a != a.path
    assert a != None  # noqa: E711
    assert a != JsonPointer("/a/b/~0/0")
    assert str(a) == "/a~1b/~0/0"
    assert repr(a) == "JsonPointer('/a~1b/~0/0')"


def test_subclass_construction_and_join():
    class Pointer(JsonPointer):
        pass
    ptr = Pointer.from_parts(["a"])
    assert isinstance(ptr, Pointer)
    assert isinstance(ptr.join(["b"]), Pointer)
    assert repr(ptr) == "Pointer('/a')"


@pytest.mark.parametrize("path,prefix,expected", [
    ("/a/b", "/a", True), ("/a", "/a/b", False),
    ("/ab", "/a", False), ("/a", "/a", True),
    ("/a", "", True), ("", "/a", False), ("", "", True),
    ("/a~1b", "/a", False),
])
def test_contains_uses_token_prefix(path, prefix, expected):
    # contains() tests whether its argument is a prefix of this pointer.
    ptr, other = JsonPointer(path), JsonPointer(prefix)
    assert ptr.contains(other) is expected
    assert (other in ptr) is expected


@pytest.mark.parametrize("verbose", [True, False])
def test_exception_verbosity(monkeypatch, verbose):
    monkeypatch.setattr(jp, "VERBOSE_EXCEPTIONS", verbose)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({"private": "sensitive-payload"}, "/missing")
    assert "missing" in str(exc.value)
    assert ("sensitive-payload" in str(exc.value)) is verbose


@pytest.mark.parametrize("values,expected", [([], []), ([1], []), ([1, 2], [(1, 2)]), ([1, 2, 3], [(1, 2), (2, 3)])])
def test_pairwise_supports_one_shot_iterators(values, expected):
    assert list(jp.pairwise(iter(values))) == expected
