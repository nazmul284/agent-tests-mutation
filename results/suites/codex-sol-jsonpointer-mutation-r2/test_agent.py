import copy

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
def test_rfc_examples(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


@pytest.mark.parametrize("pointer", ["x", "foo/bar", "~0"])
def test_pointer_must_be_empty_or_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/a~x", "/a/~9/b"])
def test_invalid_tilde_escapes_are_rejected(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


def test_escape_unescape_order_handles_literal_escape_like_text():
    assert escape("~/~1~0") == "~0~1~01~00"
    assert unescape("~01~10") == "~1/0"
    original = "a~/b"
    assert unescape(escape(original)) == original


def test_parts_path_and_from_parts_round_trip_without_exposing_encoded_parts():
    pointer = JsonPointer.from_parts(["a/b", "m~n", 0, ""])
    assert pointer.get_parts() == ["a/b", "m~n", "0", ""]
    assert pointer.parts == ["a/b", "m~n", "0", ""]
    assert pointer.path == "/a~1b/m~0n/0/"
    assert str(pointer) == "/a~1b/m~0n/0/"
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n/0/')"


def test_pairwise_for_empty_singleton_and_generator():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise(x for x in range(4))) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("bad_index", ["00", "01", "-1", "+1", " 1", "1 ", "1.0", "a", ""])
def test_sequence_indices_have_strict_json_pointer_syntax(bad_index):
    with pytest.raises(JsonPointerException, match="not a valid sequence index"):
        resolve_pointer([10, 20], "/" + bad_index)


def test_sequence_index_and_bounds_and_dash_resolution():
    assert resolve_pointer([10, 20], "/0") == 10
    assert resolve_pointer([10, 20], "/1") == 20
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([10, 20], "/2")
    marker = resolve_pointer([10, 20], "/-")
    assert isinstance(marker, EndOfList)
    assert marker.list_ == [10, 20]
    assert repr(marker) == "EndOfList([10, 20])"


def test_numeric_mapping_keys_remain_strings():
    assert resolve_pointer({"0": "string", 0: "integer"}, "/0") == "string"


def test_strings_are_not_treated_as_indexable_sequences():
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        resolve_pointer("abc", "/0")
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        set_pointer({"x": "abc"}, "/x/0", "z")


@pytest.mark.parametrize("document", [None, 3, 1.5, True])
def test_non_container_document_is_rejected(document):
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer(document, "/x")


class DuckContainer:
    def __init__(self):
        self.data = {"key": "value"}

    def __getitem__(self, key):
        return self.data[key]


class BrokenContainer:
    def __getitem__(self, key):
        raise TypeError("broken lookup")


def test_getitem_duck_typing_is_supported():
    assert resolve_pointer(DuckContainer(), "/key") == "value"


def test_missing_default_distinguishes_no_default_from_explicit_none():
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, "/missing")
    assert resolve_pointer({}, "/missing", None) is None
    sentinel = object()
    assert JsonPointer("/missing/deeper").resolve({}, sentinel) is sentinel
    # Defaults only suppress pointer resolution failures, not arbitrary user errors.
    with pytest.raises(TypeError, match="broken lookup"):
        resolve_pointer(BrokenContainer(), "/wrong", None)


def test_verbose_exception_switch_controls_document_disclosure(monkeypatch):
    document = {"secret": "do not print", "present": 1}
    with pytest.raises(JsonPointerException) as verbose:
        resolve_pointer(document, "/absent")
    assert "secret" in str(verbose.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as terse:
        resolve_pointer(document, "/absent")
    assert str(terse.value) == "member 'absent' not found"


def test_to_last_for_root_mapping_and_sequence():
    root = {"a": {"b": 1}, "items": [3]}
    assert JsonPointer("").to_last(root) == (root, None)
    assert JsonPointer("/a/b").to_last(root) == (root["a"], "b")
    assert JsonPointer("/items/0").to_last(root) == (root["items"], 0)


def test_set_mapping_existing_and_new_keys_in_place():
    document = {"a": {"old": 1}}
    returned = set_pointer(document, "/a/old", 2)
    assert returned is document
    assert document == {"a": {"old": 2}}
    set_pointer(document, "/a/new~1key", 3)
    assert document["a"]["new/key"] == 3


def test_set_sequence_replaces_and_dash_appends():
    document = {"items": [1, 2]}
    assert set_pointer(document, "/items/0", 9) is document
    assert document["items"] == [9, 2]
    set_pointer(document, "/items/-", 3)
    assert document["items"] == [9, 2, 3]
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer(document, "/items/9", 4)


def test_non_inplace_set_deep_copies_and_root_replacement():
    original = {"nested": {"values": [1]}}
    snapshot = copy.deepcopy(original)
    changed = set_pointer(original, "/nested/values/0", 8, inplace=False)
    assert changed == {"nested": {"values": [8]}}
    assert changed is not original
    assert changed["nested"] is not original["nested"]
    assert original == snapshot

    replacement = {"new": []}
    assert set_pointer(original, "", replacement, inplace=False) is replacement
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(original, "", replacement)


def test_equality_hash_and_comparison_to_unrelated_values():
    a = JsonPointer("/a~1b")
    same = JsonPointer.from_parts(["a/b"])
    different = JsonPointer("/a/b")
    assert a == same
    assert hash(a) == hash(same)
    assert a != different
    assert a != "/a~1b"
    assert len({a, same, different}) == 2


def test_contains_is_prefix_based_with_correct_direction():
    root = JsonPointer("")
    parent = JsonPointer("/a")
    child = JsonPointer("/a/b")
    sibling = JsonPointer("/ab")
    assert root.contains(root)
    assert child.contains(root)
    assert root in child
    assert child.contains(parent)
    assert parent in child
    assert not parent.contains(child)
    assert not root.contains(child)
    assert not parent.contains(sibling)


def test_join_accepts_pointer_string_and_iterable_and_preserves_subclass():
    class SpecializedPointer(JsonPointer):
        pass

    base = SpecializedPointer("/a")
    assert base.join(JsonPointer("/b~1c")).path == "/a/b~1c"
    assert (base / "/b").path == "/a/b"
    joined = base.join(["x/y", "~"])
    assert isinstance(joined, SpecializedPointer)
    assert joined.parts == ["a", "x/y", "~"]
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        base.join(42)


def test_get_alias_calls_resolution():
    assert JsonPointer("/a").get({"a": 7}) == 7
