"""Behavioral tests derived from the local implementation and its API contract."""

import copy
from collections import UserDict

import pytest

import jsonpointer
from jsonpointer import (
    EndOfList, JsonPointer, JsonPointerException, escape, pairwise,
    resolve_pointer, set_pointer, unescape,
)


@pytest.mark.parametrize("pointer, expected", [
    ("/foo", ["bar", "baz"]), ("/foo/0", "bar"), ("/foo/1", "baz"),
    ("/", 0), ("/a~1b", 1), ("/c%d", 2), ("/e^f", 3),
    ("/g|h", 4), ('/i\\j', 5), ('/k"l', 6), ("/ ", 7),
    ("/m~0n", 8), ("/~01", 9), ("/a%20b", 10), ("/বাংলা", 11),
])
def test_literal_keys_and_escaped_tokens(pointer, expected):
    doc = {"foo": ["bar", "baz"], "": 0, "a/b": 1, "c%d": 2,
           "e^f": 3, "g|h": 4, "i\\j": 5, 'k"l': 6, " ": 7,
           "m~n": 8, "~1": 9, "a%20b": 10, "বাংলা": 11}
    assert resolve_pointer(doc, pointer) == expected
    assert JsonPointer(pointer).get(doc) == expected


@pytest.mark.parametrize("doc", [{}, [], None, False, 0, "text"])
def test_root_returns_original_object(doc):
    assert resolve_pointer(doc, "") is doc
    assert JsonPointer("").to_last(doc) == (doc, None)


@pytest.mark.parametrize("path", ["a", "a/b", "#/a", " /a", "/~", "/~2", "/a~x", "/~~0", "/ok/~9/x"])
def test_invalid_pointer_syntax(path):
    with pytest.raises(JsonPointerException):
        JsonPointer(path)
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, path, default=None)


@pytest.mark.parametrize("raw, encoded", [
    ("", ""), ("/", "~1"), ("~", "~0"), ("~/", "~0~1"),
    ("~1", "~01"), ("a/b~c", "a~1b~0c"), ("%20 বাংলা", "%20 বাংলা"),
])
def test_escape_order_and_roundtrip(raw, encoded):
    assert escape(raw) == encoded
    assert unescape(encoded) == raw
    ptr = JsonPointer.from_parts([raw])
    assert ptr.path == "/" + encoded
    assert ptr.get_parts() == [raw]


def test_empty_tokens_are_not_discarded():
    ptr = JsonPointer("//a/")
    assert ptr.parts == ["", "a", ""]
    assert ptr.resolve({"": {"a": {"": 42}}}) == 42
    assert ptr.path == "//a/"


@pytest.mark.parametrize("doc, path", [
    ({}, "/missing"), ({"a": {}}, "/a/b/c"), ([], "/0"),
    ([1], "/1"), ([1], "/99999999999999999999999999"),
    ({"a": None}, "/a/b"), ({"a": "abc"}, "/a/0"),
    (False, "/x"), (3.5, "/x"),
])
def test_resolution_failures_and_default(doc, path):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, path)
    sentinel = object()
    assert resolve_pointer(doc, path, sentinel) is sentinel
    assert JsonPointer(path).resolve(doc, None) is None


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_default_does_not_replace_present_falsy_values(value):
    assert resolve_pointer({"value": value}, "/value", object()) is value


@pytest.mark.parametrize("token", ["-1", "+1", " 1", "1 ", "1.0", "1e0", "", "a", "١", "1\n"])
def test_invalid_sequence_indices(token):
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2], "/" + token)
    with pytest.raises(JsonPointerException):
        set_pointer([0, 1, 2], "/" + token, 9)


@pytest.mark.xfail(strict=True, reason="Known bug: array index validation accepts leading zeros")
@pytest.mark.parametrize("token", ["00", "01", "000"])
@pytest.mark.parametrize("operation", ["resolve", "set"])
def test_array_indices_reject_leading_zeros(token, operation):
    with pytest.raises(JsonPointerException):
        if operation == "resolve":
            resolve_pointer([10, 20], "/" + token)
        else:
            set_pointer([10, 20], "/" + token, 30)


@pytest.mark.parametrize("key", ["0", "01", "-", "-1", "+1"])
def test_numeric_and_dash_mapping_keys_remain_strings(key):
    doc = {key: "old"}
    assert resolve_pointer(doc, "/" + key) == "old"
    assert set_pointer(doc, "/" + key, "new") is doc
    assert doc == {key: "new"}


def test_nested_resolution_keeps_identity_and_does_not_mutate():
    doc = {"a": [{"b": [1, 2]}]}
    before = copy.deepcopy(doc)
    assert resolve_pointer(doc, "/a/0/b") is doc["a"][0]["b"]
    assert doc == before
    parent, part = JsonPointer("/a/0/b/1").to_last(doc)
    assert parent is doc["a"][0]["b"]
    assert part == 1
    parent, part = JsonPointer("/a/0/new").to_last(doc)
    assert parent is doc["a"][0]
    assert part == "new"


def test_set_replaces_adds_and_appends_in_place():
    doc = {"items": [{"value": 1}]}
    assert set_pointer(doc, "/items/0/value", 2) is doc
    assert JsonPointer("/items/0/a~1b").set(doc, 3) is doc
    value = {"value": 4}
    assert set_pointer(doc, "/items/-", value) is doc
    assert doc == {"items": [{"value": 2, "a/b": 3}, {"value": 4}]}
    assert doc["items"][1] is value


@pytest.mark.parametrize("path, value, expected", [
    ("/items/0/x", 9, {"items": [{"x": 9}], "other": []}),
    ("/items/-", 2, {"items": [{"x": 1}, 2], "other": []}),
])
def test_non_inplace_set_deepcopies_document(path, value, expected):
    doc = {"items": [{"x": 1}], "other": []}
    result = set_pointer(doc, path, value, inplace=False)
    assert result == expected
    assert doc == {"items": [{"x": 1}], "other": []}
    assert result is not doc
    assert result["items"] is not doc["items"]
    assert result["items"][0] is not doc["items"][0]
    assert result["other"] is not doc["other"]


def test_root_replacement_requires_non_inplace():
    doc = {"old": [1]}
    replacement = [2]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "", replacement)
    assert set_pointer(doc, "", replacement, inplace=False) is replacement
    assert doc == {"old": [1]}


@pytest.mark.parametrize("doc, path", [
    ({}, "/missing/child"), ({"a": []}, "/a/0"),
    ({"a": [1]}, "/a/1"), ({"a": "abc"}, "/a/0"),
    ({"a": 1}, "/a/x"), ({"a": (1, 2)}, "/a/0"),
])
def test_invalid_assignment_does_not_change_document(doc, path):
    before = copy.deepcopy(doc)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, path, 99)
    assert doc == before


def test_end_of_list_sentinel_and_append():
    doc = [1]
    end = resolve_pointer(doc, "/-")
    assert isinstance(end, EndOfList)
    assert end.list_ is doc
    assert repr(end) == "EndOfList([1])"
    assert JsonPointer("/-").to_last(doc) == (doc, "-")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/-/x")
    assert set_pointer(doc, "/-", 2) == [1, 2]
    assert set_pointer([], "/-", 1) == [1]


def test_alternative_containers_and_duck_typing():
    assert resolve_pointer(UserDict({"a": (10, 20)}), "/a/1") == 20

    class Indexed:
        def __getitem__(self, key):
            if key == "name":
                return "custom"
            raise KeyError(key)

    assert resolve_pointer(Indexed(), "/name") == "custom"
    with pytest.raises(JsonPointerException):
        resolve_pointer(Indexed(), "/missing")


def test_pointer_construction_and_representations():
    ptr = JsonPointer.from_parts(iter(["a/b", "~1", 0, ""]))
    assert ptr.parts == ["a/b", "~1", "0", ""]
    assert ptr.path == "/a~1b/~01/0/"
    assert str(ptr) == ptr.path
    assert repr(ptr) == "JsonPointer('/a~1b/~01/0/')"
    assert JsonPointer.from_parts([]) == JsonPointer("")
    assert ptr == JsonPointer(ptr.path)
    assert hash(ptr) == hash(JsonPointer(ptr.path))
    assert len({ptr, JsonPointer(ptr.path)}) == 1
    assert ptr != JsonPointer("/other")
    assert ptr != ptr.path
    assert ptr != None  # noqa: E711 -- exercise cross-type equality


@pytest.mark.parametrize("suffix", ["/b~1c/~0", JsonPointer("/b~1c/~0"), ["b/c", "~"], ("b/c", "~")])
def test_join_and_division_preserve_token_boundaries(suffix):
    base = JsonPointer("/a")
    assert base.join(suffix) == JsonPointer("/a/b~1c/~0")
    assert base / suffix == JsonPointer("/a/b~1c/~0")
    assert base.path == "/a"
    assert base.join("") == base
    assert base.join("") is not base


@pytest.mark.parametrize("suffix", [None, 42, "/bad~2", "no-slash"])
def test_invalid_join_suffix(suffix):
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(suffix)


def test_subclass_factories_preserve_type():
    class Pointer(JsonPointer):
        pass

    assert isinstance(Pointer.from_parts(["a"]), Pointer)
    assert isinstance(Pointer("/a").join("/b"), Pointer)


@pytest.mark.parametrize("path, prefix, expected", [
    ("/a/b", "/a", True), ("/a", "/a", True), ("/a", "", True),
    ("/a", "/a/b", False), ("/ab", "/a", False),
    ("/a~1b", "/a", False), ("", "", True), ("", "/", False),
])
def test_containment_uses_decoded_token_prefix(path, prefix, expected):
    # The implementation defines contains as testing whether its argument is a prefix.
    ptr, candidate = JsonPointer(path), JsonPointer(prefix)
    assert ptr.contains(candidate) is expected
    assert (candidate in ptr) is expected


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_exceptions_toggle(monkeypatch, verbose):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", verbose)
    with pytest.raises(JsonPointerException) as error:
        resolve_pointer({"secret": "private-content"}, "/missing")
    assert "missing" in str(error.value)
    assert ("private-content" in str(error.value)) is verbose


@pytest.mark.parametrize("items, expected", [([], []), ([1], []), ([1, 2, 3], [(1, 2), (2, 3)])])
def test_pairwise_supports_single_pass_iterators(items, expected):
    assert list(pairwise(iter(items))) == expected
