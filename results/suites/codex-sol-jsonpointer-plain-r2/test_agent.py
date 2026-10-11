"""Tests for the public behaviour of :mod:`jsonpointer`.

The examples here are deliberately independent of any upstream test suite.
They exercise the RFC 6901 rules as well as the additional convenience API
implemented by this module.
"""

import pytest

import jsonpointer
from jsonpointer import (
    EndOfList,
    JsonPointer,
    JsonPointerException,
    escape,
    pairwise,
    resolve_pointer,
    set_pointer,
    unescape,
)


RFC_DOCUMENT = {
    "foo": ["bar", "baz"],
    "": 0,
    "a/b": 1,
    "c%d": 2,
    "e^f": 3,
    "g|h": 4,
    "i\\j": 5,
    'k"l': 6,
    " ": 7,
    "m~n": 8,
}


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        ("", RFC_DOCUMENT),
        ("/foo", ["bar", "baz"]),
        ("/foo/0", "bar"),
        ("/", 0),
        ("/a~1b", 1),
        ("/c%d", 2),
        ("/e^f", 3),
        ("/g|h", 4),
        ("/i\\j", 5),
        ('/k"l', 6),
        ("/ ", 7),
        ("/m~0n", 8),
    ],
)
def test_rfc_6901_examples(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_numeric_object_members_are_not_treated_as_array_indexes():
    doc = {"0": "zero", "01": "leading zero", "-": "hyphen"}
    assert resolve_pointer(doc, "/0") == "zero"
    assert resolve_pointer(doc, "/01") == "leading zero"
    assert resolve_pointer(doc, "/-") == "hyphen"


@pytest.mark.parametrize("pointer", ["foo", "#", "relative/path"])
def test_pointer_must_be_empty_or_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/a~xb", "/ok/~9/end"])
def test_invalid_tilde_escapes_are_rejected(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


def test_escape_decoding_order_handles_literal_tilde_one():
    # ~01 means the literal token "~1", not "/".  Reversing replacement
    # order in unescape would silently get this wrong.
    doc = {"~1": "literal", "/": "slash"}
    assert resolve_pointer(doc, "/~01") == "literal"
    assert unescape("~01~1~0") == "~1/~"


@pytest.mark.parametrize(
    ("raw", "encoded"),
    [("plain", "plain"), ("~", "~0"), ("/", "~1"), ("~/~", "~0~1~0")],
)
def test_escape_and_unescape(raw, encoded):
    assert escape(raw) == encoded
    assert unescape(encoded) == raw


@pytest.mark.parametrize("index", ["00", "01", "+1", "-1", " 1", "1 ", "1.0", ""])
def test_array_indexes_use_strict_json_pointer_syntax(index):
    with pytest.raises(JsonPointerException, match="valid sequence index"):
        resolve_pointer(["zero", "one"], "/" + index)


def test_array_index_bounds_and_scalar_traversal_are_reported():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([1], "/1")
    with pytest.raises(JsonPointerException, match="non-container"):
        resolve_pointer({"value": "text"}, "/value/0")
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer({"value": 3}, "/value/anything")


def test_missing_member_default_and_explicit_none_default():
    sentinel = object()
    assert resolve_pointer({}, "/missing", sentinel) is sentinel
    assert resolve_pointer({}, "/missing", None) is None
    with pytest.raises(JsonPointerException, match="member 'missing' not found"):
        resolve_pointer({}, "/missing")


def test_default_applies_to_failure_at_any_depth():
    assert resolve_pointer({"a": []}, "/a/5/value", "fallback") == "fallback"


def test_hyphen_resolves_to_an_end_of_list_marker():
    values = [1, 2]
    marker = resolve_pointer(values, "/-")
    assert isinstance(marker, EndOfList)
    assert marker.list_ is values
    assert repr(marker) == "EndOfList([1, 2])"


def test_set_replaces_and_adds_mapping_members_in_place():
    doc = {"outer": {"old": 1}}
    result = set_pointer(doc, "/outer/old", 2)
    assert result is doc
    assert doc == {"outer": {"old": 2}}

    set_pointer(doc, "/outer/a~1b", 3)
    assert doc["outer"]["a/b"] == 3


def test_set_updates_list_index_and_appends_with_hyphen():
    doc = {"items": ["first"]}
    set_pointer(doc, "/items/0", "changed")
    set_pointer(doc, "/items/-", "last")
    assert doc == {"items": ["changed", "last"]}


def test_non_inplace_set_deep_copies_before_modifying():
    doc = {"nested": [{"value": 1}], "untouched": []}
    result = set_pointer(doc, "/nested/0/value", 9, inplace=False)
    assert result == {"nested": [{"value": 9}], "untouched": []}
    assert doc == {"nested": [{"value": 1}], "untouched": []}
    assert result is not doc
    assert result["untouched"] is not doc["untouched"]


def test_setting_root_requires_non_inplace_mode():
    original = {"a": 1}
    replacement = {"b": 2}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(original, "", replacement)
    assert set_pointer(original, "", replacement, inplace=False) is replacement
    assert original == {"a": 1}


def test_invalid_assignment_targets_are_wrapped():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer([1], "/2", 9)
    with pytest.raises(JsonPointerException, match="non-container"):
        set_pointer({"s": "abc"}, "/s/0", "x")


def test_to_last_returns_parent_and_converted_final_part():
    doc = {"items": [10, 20]}
    parent, part = JsonPointer("/items/1").to_last(doc)
    assert parent is doc["items"]
    assert part == 1
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_from_parts_and_path_round_trip_special_characters_and_numbers():
    pointer = JsonPointer.from_parts(["a/b", "m~n", "", 0])
    assert pointer.path == "/a~1b/m~0n//0"
    assert pointer.get_parts() == ["a/b", "m~n", "", "0"]
    assert str(pointer) == pointer.path
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n//0')"


def test_join_accepts_pointer_string_and_parts_iterable():
    base = JsonPointer("/a~1b")
    assert (base / JsonPointer("/c~0d")).path == "/a~1b/c~0d"
    assert base.join("/x/y").path == "/a~1b/x/y"
    assert base.join(["literal/slash", 2]).path == "/a~1b/literal~1slash/2"
    assert base.path == "/a~1b"


def test_invalid_join_suffix_is_reported_consistently():
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        JsonPointer("/a").join(42)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("not-a-pointer")


def test_pointer_equality_hashing_and_ancestor_containment():
    same_a = JsonPointer("/a")
    same_a_again = JsonPointer.from_parts(["a"])
    descendant = JsonPointer("/a/b")
    assert same_a == same_a_again
    assert hash(same_a) == hash(same_a_again)
    assert same_a != "/a"
    assert same_a in descendant
    assert JsonPointer("") in descendant
    assert descendant not in same_a
    assert JsonPointer("/other") not in descendant


def test_custom_indexable_objects_are_supported_by_duck_typing():
    class Indexable:
        def __getitem__(self, key):
            if key == "answer":
                return 42
            raise KeyError(key)

    assert resolve_pointer(Indexable(), "/answer") == 42
    with pytest.raises(JsonPointerException, match="member 'missing' not found"):
        resolve_pointer(Indexable(), "/missing")


def test_verbose_exception_flag_controls_document_disclosure(monkeypatch):
    doc = {"secret": "do not include me"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as caught:
        resolve_pointer(doc, "/missing")
    assert str(caught.value) == "member 'missing' not found"

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as caught:
        resolve_pointer(doc, "/missing")
    assert "secret" in str(caught.value)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], []), ([1], []), ([1, 2], [(1, 2)]), ([1, 2, 3], [(1, 2), (2, 3)])],
)
def test_pairwise(values, expected):
    assert list(pairwise(iter(values))) == expected
