"""Behavioral tests for the public :mod:`jsonpointer` API.

The examples here are intentionally independent of the project's historical
test suite.  Most of the resolution cases come directly from the document in
RFC 6901, with additional tests for this implementation's mutation helpers.
"""

import pytest

import jsonpointer
from jsonpointer import (
    EndOfList,
    JsonPointer,
    JsonPointerException,
    escape,
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
def test_resolves_rfc_6901_examples(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_empty_pointer_selects_the_entire_document_by_identity():
    assert resolve_pointer(RFC_DOCUMENT, "") is RFC_DOCUMENT


@pytest.mark.parametrize("pointer", ["foo", "#", "foo/bar"])
def test_pointer_must_be_empty_or_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/ok/~x/bad"])
def test_invalid_tilde_escape_is_rejected(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


def test_escape_and_unescape_preserve_the_required_replacement_order():
    token = "~/~01"
    encoded = "~0~1~001"
    assert escape(token) == encoded
    assert unescape(encoded) == token


def test_pointer_path_and_parts_round_trip_reserved_and_empty_tokens():
    ptr = JsonPointer.from_parts(["a/b", "m~n", "", 3])
    assert ptr.path == "/a~1b/m~0n//3"
    assert ptr.get_parts() == ["a/b", "m~n", "", "3"]
    assert JsonPointer(ptr.path) == ptr
    assert str(ptr) == ptr.path
    assert repr(ptr) == "JsonPointer('/a~1b/m~0n//3')"


@pytest.mark.parametrize("token", ["00", "01", "-1", "+1", " 1", "1 ", "1.0"])
def test_array_indices_use_the_strict_json_pointer_grammar(token):
    with pytest.raises(JsonPointerException, match="valid sequence index"):
        resolve_pointer(["zero", "one"], "/" + token)


def test_array_index_zero_and_multi_digit_indices_are_valid():
    values = list(range(12))
    assert resolve_pointer(values, "/0") == 0
    assert resolve_pointer(values, "/10") == 10


def test_missing_members_and_out_of_range_indices_raise_library_exception():
    with pytest.raises(JsonPointerException, match="member 'missing' not found"):
        resolve_pointer({}, "/missing")
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([1], "/1")


def test_default_is_returned_for_any_failed_resolution_step():
    marker = object()
    assert resolve_pointer({"a": {}}, "/a/missing/deeper", marker) is marker
    assert JsonPointer("/4").get([1, 2], marker) is marker
    # A falsey default must not be mistaken for an omitted default.
    assert resolve_pointer({}, "/missing", None) is None


def test_cannot_walk_through_a_scalar_value():
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer({"answer": 42}, "/answer/unit")
    with pytest.raises(JsonPointerException, match="non-container"):
        resolve_pointer({"word": "abc"}, "/word/0")


def test_dash_resolves_to_an_end_of_list_marker():
    values = [1, 2]
    result = resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_set_replaces_mapping_and_sequence_members_in_place():
    doc = {"obj": {"old": 1}, "items": [10, 20]}
    assert set_pointer(doc, "/obj/old", 2) is doc
    assert set_pointer(doc, "/obj/new", 3) is doc
    assert set_pointer(doc, "/items/1", 99) is doc
    assert doc == {"obj": {"old": 2, "new": 3}, "items": [10, 99]}


def test_set_dash_appends_to_a_list():
    doc = {"items": [1]}
    assert set_pointer(doc, "/items/-", 2) is doc
    assert doc == {"items": [1, 2]}


def test_non_inplace_set_deep_copies_before_mutating():
    original = {"outer": {"items": [{"value": 1}]}}
    changed = set_pointer(original, "/outer/items/0/value", 9, inplace=False)
    assert changed == {"outer": {"items": [{"value": 9}]}}
    assert original == {"outer": {"items": [{"value": 1}]}}
    assert changed is not original
    assert changed["outer"] is not original["outer"]


def test_setting_root_requires_non_inplace_mode():
    original = {"a": 1}
    replacement = {"b": 2}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(original, "", replacement)
    assert set_pointer(original, "", replacement, inplace=False) is replacement
    assert original == {"a": 1}


def test_invalid_set_targets_are_reported_as_pointer_errors():
    with pytest.raises(JsonPointerException, match="non-container"):
        set_pointer({"value": "abc"}, "/value/0", "x")
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer([1], "/2", "x")


def test_to_last_returns_parent_and_typed_final_component():
    doc = {"items": ["a"]}
    parent, part = JsonPointer("/items/0").to_last(doc)
    assert parent is doc["items"]
    assert part == 0
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_join_and_division_compose_without_mutating_inputs():
    base = JsonPointer("/a~1b")
    suffix = JsonPointer("/m~0n/0")
    assert base.join(suffix).path == "/a~1b/m~0n/0"
    assert (base / "/child").path == "/a~1b/child"
    assert (base / ["x/y", 2]).path == "/a~1b/x~1y/2"
    assert base.path == "/a~1b"
    assert suffix.path == "/m~0n/0"


def test_pointer_equality_hashing_and_ancestor_containment():
    same_a = JsonPointer("/a/~1")
    same_b = JsonPointer.from_parts(["a", "/"])
    child = same_a / "/child"
    assert same_a == same_b
    assert hash(same_a) == hash(same_b)
    assert same_a != same_a.path
    assert same_a in child
    assert child not in same_a


def test_verbose_exception_flag_controls_document_disclosure(monkeypatch):
    secret_doc = {"visible": "secret-value"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as caught:
        resolve_pointer(secret_doc, "/missing")
    assert "secret-value" not in str(caught.value)
    assert str(caught.value) == "member 'missing' not found"
