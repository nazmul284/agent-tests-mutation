"""Behavioral tests derived from the local jsonpointer implementation."""

from collections import UserDict
from copy import deepcopy
from itertools import product

import pytest

import jsonpointer as jp


@pytest.fixture
def document():
    return {
        "": "empty key",
        "items": [{"name": "first"}, {"name": None}],
        "a/b": {"~key": 42},
        "a%20b": "literal percent",
        "a b": "space",
        "雪": "unicode",
        "0": "string key",
        "-": "dash key",
    }


@pytest.mark.parametrize("path, expected", [
    ("/", "empty key"),
    ("/items/0/name", "first"),
    ("/items/1/name", None),
    ("/a~1b/~0key", 42),
    ("/a%20b", "literal percent"),
    ("/a b", "space"),
    ("/雪", "unicode"),
    ("/0", "string key"),
    ("/-", "dash key"),
])
def test_resolution(document, path, expected):
    before = deepcopy(document)
    pointer = jp.JsonPointer(path)
    assert jp.resolve_pointer(document, path) == expected
    assert pointer.resolve(document) == expected
    assert pointer.get(document) == expected
    assert document == before


@pytest.mark.parametrize("doc", [{}, [], None, False, 0, "text"])
def test_empty_pointer_returns_document_itself(doc):
    assert jp.resolve_pointer(doc, "") is doc
    assert jp.JsonPointer("").to_last(doc) == (doc, None)


def test_resolution_preserves_identity(document):
    assert jp.resolve_pointer(document, "/items") is document["items"]
    assert jp.resolve_pointer(document, "/items/0") is document["items"][0]


@pytest.mark.parametrize("path", ["items", "#/items", " /items", "/~", "/a~2", "/~x", "/ok/bad~"])
def test_invalid_pointer_syntax(path):
    with pytest.raises(jp.JsonPointerException):
        jp.JsonPointer(path)
    # Defaults handle lookup failures, not malformed pointers.
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer({}, path, default=None)


@pytest.mark.parametrize("token", ["00", "01", "0x", "-1", "+1", " 1", "1 ", "1.0", "1e0", "", "١", "1\n"])
def test_sequence_indices_are_strict_but_mapping_keys_are_literal(token):
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(["zero", "one"], "/" + token)
    assert jp.resolve_pointer({token: "value"}, "/" + token) == "value"


@pytest.mark.parametrize("doc,path", [
    ({}, "/missing"), ({"a": {}}, "/a/b/c"), ([], "/0"),
    ([1], "/1"), ([1], "/999999"), (None, "/a"),
    (True, "/a"), (123, "/0"), ("abc", "/0"),
    ({"a": "abc"}, "/a/0"), ([], "/-/child"),
])
def test_lookup_errors_and_defaults(doc, path):
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(doc, path)
    sentinel = object()
    assert jp.resolve_pointer(doc, path, sentinel) is sentinel
    assert jp.JsonPointer(path).resolve(doc, None) is None


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_existing_falsy_value_does_not_use_default(value):
    assert jp.resolve_pointer({"key": value}, "/key", object()) is value


def test_end_of_list_marker():
    values = [1, 2]
    marker = jp.resolve_pointer(values, "/-")
    assert isinstance(marker, jp.EndOfList)
    assert marker.list_ is values
    assert repr(marker) == "EndOfList([1, 2])"


def test_to_last_converts_only_sequence_indices(document):
    parent, part = jp.JsonPointer("/items/0").to_last(document)
    assert parent is document["items"]
    assert part == 0 and isinstance(part, int)
    parent, part = jp.JsonPointer("/0").to_last(document)
    assert parent is document and part == "0"
    parent, part = jp.JsonPointer("/items/-").to_last(document)
    assert parent is document["items"] and part == "-"


@pytest.mark.parametrize("path,value", [
    ("/items/0/name", "changed"), ("/items/1", {"new": True}),
    ("/new", [1]), ("/a~1b/~0key", 99), ("/", "new empty"),
])
def test_set_in_place(document, path, value):
    result = jp.set_pointer(document, path, value)
    assert result is document
    assert jp.resolve_pointer(document, path) is value
    assert document["雪"] == "unicode"


def test_set_copy_is_deep_and_keeps_original(document):
    before = deepcopy(document)
    value = {"replacement": []}
    result = jp.JsonPointer("/items/0").set(document, value, inplace=False)
    assert document == before
    assert result is not document
    assert result["items"] is not document["items"]
    assert result["items"][1] is not document["items"][1]
    assert result["a/b"] is not document["a/b"]
    assert result["items"][0] is value


@pytest.mark.parametrize("inplace", [True, False])
def test_append(inplace):
    original = {"values": [1]}
    value = {"new": 2}
    result = jp.set_pointer(original, "/values/-", value, inplace=inplace)
    assert result["values"] == [1, value]
    assert result["values"][-1] is value
    assert (result is original) is inplace
    if not inplace:
        assert original == {"values": [1]}


def test_root_replacement():
    original = {"old": [1]}
    replacement = ["new"]
    with pytest.raises(jp.JsonPointerException, match="root"):
        jp.set_pointer(original, "", replacement)
    assert jp.set_pointer(original, "", replacement, inplace=False) is replacement
    assert original == {"old": [1]}


@pytest.mark.parametrize("doc,path", [
    ({}, "/missing/child"), ([1], "/1"), ([1], "/-1"),
    ([1], "/01"), ("abc", "/0"), ((1, 2), "/0"),
    ({"a": None}, "/a/b"),
])
def test_invalid_assignments_raise_without_mutation(doc, path):
    before = deepcopy(doc)
    with pytest.raises(jp.JsonPointerException):
        jp.set_pointer(doc, path, "replacement")
    assert doc == before


@pytest.mark.parametrize("raw,encoded", [
    ("", ""), ("/", "~1"), ("~", "~0"), ("~/", "~0~1"),
    ("a/b~c", "a~1b~0c"), ("雪 %", "雪 %"), ("~~", "~0~0"),
])
def test_escape_and_unescape(raw, encoded):
    assert jp.escape(raw) == encoded
    assert jp.unescape(encoded) == raw


@pytest.mark.xfail(strict=True, reason="unescape decodes ~0 before ~1, decoding ~01 twice")
def test_escaped_tilde_followed_by_one_is_not_a_slash():
    assert jp.unescape("~01") == "~1"
    assert jp.resolve_pointer({"~1": "literal", "/": "slash"}, "/~01") == "literal"


def test_from_parts_round_trip():
    for parts in product(["", "a", "/", "~", "雪", "0"], repeat=2):
        pointer = jp.JsonPointer.from_parts(iter(parts))
        assert pointer.get_parts() == list(parts)
        assert jp.JsonPointer(pointer.path) == pointer
    pointer = jp.JsonPointer.from_parts(["a/b", "~", 0])
    assert pointer.path == "/a~1b/~0/0"
    assert pointer.parts == ["a/b", "~", "0"]
    assert jp.JsonPointer.from_parts([]).path == ""


@pytest.mark.parametrize("suffix", ["/b~1c/~0", jp.JsonPointer("/b~1c/~0"), ["b/c", "~"], ("b/c", "~")])
def test_join_and_division(suffix):
    base = jp.JsonPointer("/a")
    assert base.join(suffix).path == "/a/b~1c/~0"
    assert base / suffix == jp.JsonPointer("/a/b~1c/~0")
    assert base.path == "/a"
    assert base.join("") == base
    assert base.join("") is not base


@pytest.mark.parametrize("suffix", [None, 1, object()])
def test_invalid_join_suffix(suffix):
    with pytest.raises(jp.JsonPointerException, match="Invalid suffix"):
        jp.JsonPointer("/a").join(suffix)


def test_equality_hash_and_representations():
    pointer = jp.JsonPointer("/a~1b/~0")
    equivalent = jp.JsonPointer.from_parts(["a/b", "~"])
    assert pointer == equivalent
    assert hash(pointer) == hash(equivalent)
    assert len({pointer, equivalent}) == 1
    assert pointer != jp.JsonPointer("/a/b/~0")
    assert pointer != pointer.path
    assert pointer != object()
    assert str(pointer) == pointer.path
    assert repr(pointer) == "JsonPointer('/a~1b/~0')"


def test_contains_uses_complete_token_prefix():
    # This API checks whether its argument is a prefix of self.
    pointer = jp.JsonPointer("/a/b")
    for prefix in ["", "/a", "/a/b"]:
        assert pointer.contains(jp.JsonPointer(prefix))
        assert jp.JsonPointer(prefix) in pointer
    for other in ["/a/b/c", "/a/bb", "/ab", "/x"]:
        assert not pointer.contains(jp.JsonPointer(other))


def test_mapping_sequence_and_duck_typed_containers():
    assert jp.resolve_pointer(UserDict({"0": "mapping"}), "/0") == "mapping"
    assert jp.resolve_pointer(("a", "b"), "/1") == "b"

    class Lookup:
        def __getitem__(self, key):
            if key == "name":
                return {"value": 7}
            raise KeyError(key)

    assert jp.resolve_pointer(Lookup(), "/name/value") == 7
    with pytest.raises(jp.JsonPointerException):
        jp.resolve_pointer(Lookup(), "/missing")


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_exception_setting(monkeypatch, verbose):
    monkeypatch.setattr(jp, "VERBOSE_EXCEPTIONS", verbose)
    with pytest.raises(jp.JsonPointerException) as exc:
        jp.resolve_pointer({"secret": "private-value"}, "/absent")
    assert "absent" in str(exc.value)
    assert ("private-value" in str(exc.value)) is verbose


@pytest.mark.parametrize("values,expected", [([], []), ([1], []), ([1, 2, 3], [(1, 2), (2, 3)])])
def test_pairwise_accepts_single_pass_iterators(values, expected):
    assert list(jp.pairwise(iter(values))) == expected
