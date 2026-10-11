"""Behavioral tests for the public jsonpointer API."""

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
    'k\"l': 6,
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
        ('/k\"l', 6),
        ("/ ", 7),
        ("/m~0n", 8),
    ],
)
def test_rfc_6901_examples(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


@pytest.mark.parametrize("pointer", ["foo", "#", "a/b"])
def test_pointer_must_be_empty_or_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="must start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/abc~x", "/a/~9/b"])
def test_invalid_tilde_escapes_are_rejected(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


def test_escape_round_trip_and_order_of_replacements():
    raw = "a~/b~1"
    encoded = "a~0~1b~01"
    assert escape(raw) == encoded
    assert unescape(encoded) == raw
    assert JsonPointer.from_parts([raw, 0]).path == "/a~0~1b~01/0"


def test_empty_tokens_and_numeric_mapping_keys_are_not_array_indices():
    document = {"": {"01": {"0": "value"}}}
    assert resolve_pointer(document, "//01/0") == "value"


@pytest.mark.parametrize("token", ["+1", "-1", " 1", "1 ", "1.0", "one"])
def test_invalid_array_indices_are_rejected(token):
    with pytest.raises(JsonPointerException, match="not a valid sequence index"):
        resolve_pointer(["zero", "one"], "/" + token)


def test_array_index_out_of_bounds_has_pointer_exception():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([1], "/1")


def test_missing_member_uses_default_even_when_default_is_none():
    document = {"present": None}
    marker = object()
    assert resolve_pointer(document, "/absent", marker) is marker
    assert resolve_pointer(document, "/absent", None) is None
    assert resolve_pointer(document, "/present", marker) is None
    with pytest.raises(JsonPointerException, match="member 'absent' not found"):
        resolve_pointer(document, "/absent")


def test_default_covers_failure_at_any_depth():
    assert resolve_pointer({"a": []}, "/a/3/x", "fallback") == "fallback"


def test_dash_resolves_to_end_of_list_marker():
    values = [1, 2]
    result = resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_set_replaces_mapping_and_sequence_values_in_place():
    document = {"object": {"old": 1}, "array": [10, 20]}
    returned = set_pointer(document, "/object/new", 2)
    assert returned is document
    assert document == {"object": {"old": 1, "new": 2}, "array": [10, 20]}
    set_pointer(document, "/array/1", 99)
    assert document["array"] == [10, 99]


def test_set_dash_appends_to_list():
    document = {"items": [1]}
    assert set_pointer(document, "/items/-", 2) is document
    assert document == {"items": [1, 2]}


def test_non_inplace_set_deep_copies_before_changing_value():
    original = {"outer": {"items": [{"value": 1}]}}
    changed = set_pointer(original, "/outer/items/0/value", 2, inplace=False)
    assert changed == {"outer": {"items": [{"value": 2}]}}
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


@pytest.mark.parametrize(
    ("document", "pointer", "message"),
    [
        ({"a": "text"}, "/a/0", "Cannot apply token '0' to non-container type"),
        ({"a": [1]}, "/a/2", "Invalid assignment target"),
        ({"a": 1}, "/a/x", "does not support indexing"),
    ],
)
def test_invalid_assignment_targets_raise_pointer_exception(document, pointer, message):
    with pytest.raises(JsonPointerException, match=message):
        set_pointer(document, pointer, 9)


def test_pointer_path_parts_repr_equality_and_hash():
    pointer = JsonPointer("/a~1b/m~0n")
    same = JsonPointer.from_parts(["a/b", "m~n"])
    assert pointer.get_parts() == ["a/b", "m~n"]
    assert pointer.path == "/a~1b/m~0n"
    assert str(pointer) == "/a~1b/m~0n"
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n')"
    assert pointer == same
    assert pointer != "/a~1b/m~0n"
    assert len({pointer, same}) == 1


def test_join_accepts_pointer_string_and_unescaped_parts():
    base = JsonPointer("/a")
    assert (base / JsonPointer("/b")).path == "/a/b"
    assert base.join("/b~1c").path == "/a/b~1c"
    assert base.join(["b/c", "~", 0]).path == "/a/b~1c/~0/0"
    assert base.path == "/a"


def test_join_wraps_bad_suffix_as_pointer_exception():
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        JsonPointer("/a").join(42)


def test_contains_describes_prefix_relationship():
    root = JsonPointer("")
    parent = JsonPointer("/a")
    child = JsonPointer("/a/b")
    assert child.contains(parent)
    assert parent in child
    assert root in child
    assert child not in parent
    assert JsonPointer("/ab") not in child


class CustomIndexable:
    def __init__(self):
        self.values = {"key": "value"}

    def __getitem__(self, key):
        return self.values[key]


def test_custom_indexable_objects_are_supported():
    assert resolve_pointer(CustomIndexable(), "/key") == "value"


def test_verbose_exception_setting_controls_document_disclosure(monkeypatch):
    secret = {"secret": "do not print"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as caught:
        resolve_pointer(secret, "/missing")
    assert str(caught.value) == "member 'missing' not found"
    assert "do not print" not in str(caught.value)
