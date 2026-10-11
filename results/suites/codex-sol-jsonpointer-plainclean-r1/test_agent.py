"""Behavioral tests for the public :mod:`jsonpointer` API.

The examples deliberately include keys which look like pointer syntax.  JSON
Pointer only decodes ``~0`` and ``~1``; it does not perform URL decoding.
"""

from collections import UserDict

import pytest

import jsonpointer
from jsonpointer import EndOfList, JsonPointer, JsonPointerException


RFC_DOCUMENT = {
    "foo": ["bar", "baz"],
    "": 0,
    "a/b": 1,
    "c%d": 2,
    "e^f": 3,
    "g|h": 4,
    "i\\j": 5,
    "k\"l": 6,
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
def test_resolves_rfc_examples(pointer, expected):
    assert jsonpointer.resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_tokens_are_unescaped_once_and_are_not_percent_decoded():
    document = {"~1": "literal escape", "a%20b": "percent", "a b": "space"}
    assert jsonpointer.resolve_pointer(document, "/~01") == "literal escape"
    assert jsonpointer.resolve_pointer(document, "/a%20b") == "percent"
    assert jsonpointer.resolve_pointer(document, "/a b") == "space"


@pytest.mark.parametrize("pointer", ["foo", "#", "a/b"])
def test_pointer_must_start_with_slash_unless_it_is_root(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/ok/bad~x"])
def test_rejects_invalid_tilde_escapes(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


@pytest.mark.parametrize("token", ["00", "01", "+1", "-1", " 1", "1 ", "1.0"])
def test_rejects_noncanonical_array_indices(token):
    with pytest.raises(JsonPointerException, match="valid sequence index"):
        jsonpointer.resolve_pointer(["zero", "one"], "/" + token)


def test_array_index_failures_and_scalar_traversal_are_domain_errors():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        jsonpointer.resolve_pointer([], "/0")
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        jsonpointer.resolve_pointer({"value": 1}, "/value/child")


def test_default_is_returned_for_any_unresolvable_step():
    marker = object()
    assert jsonpointer.resolve_pointer({}, "/missing", marker) is marker
    assert jsonpointer.resolve_pointer([], "/3", marker) is marker
    assert jsonpointer.resolve_pointer(1, "/child", marker) is marker


def test_dash_resolves_to_end_of_list_marker():
    values = [1, 2]
    result = jsonpointer.resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_set_pointer_mutates_nested_mapping_and_replaces_list_item():
    document = {"items": [{"name": "old"}]}
    returned = jsonpointer.set_pointer(document, "/items/0/name", "new")
    assert returned is document
    assert document == {"items": [{"name": "new"}]}


def test_set_pointer_dash_appends_to_list():
    document = {"items": [1]}
    assert jsonpointer.set_pointer(document, "/items/-", 2) is document
    assert document == {"items": [1, 2]}


def test_non_inplace_set_deep_copies_and_can_replace_root():
    original = {"nested": {"value": 1}, "untouched": []}
    changed = jsonpointer.set_pointer(original, "/nested/value", 2, inplace=False)
    assert changed == {"nested": {"value": 2}, "untouched": []}
    assert original == {"nested": {"value": 1}, "untouched": []}
    assert changed is not original
    assert changed["untouched"] is not original["untouched"]

    replacement = {"new": True}
    assert jsonpointer.set_pointer(original, "", replacement, inplace=False) is replacement
    with pytest.raises(JsonPointerException, match="root"):
        jsonpointer.set_pointer(original, "", replacement)


def test_invalid_assignment_targets_raise_pointer_exception():
    with pytest.raises(JsonPointerException, match="non-container"):
        jsonpointer.set_pointer({"text": "abc"}, "/text/0", "x")
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        jsonpointer.set_pointer([], "/1", "x")


def test_pointer_round_trip_and_construction_from_parts():
    pointer = JsonPointer.from_parts(["a/b", "m~n", "", 0])
    assert pointer.get_parts() == ["a/b", "m~n", "", "0"]
    assert pointer.path == "/a~1b/m~0n//0"
    assert str(JsonPointer(pointer.path)) == pointer.path
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n//0')"


def test_join_division_equality_hash_and_containment():
    base = JsonPointer("/a")
    child = base.join("/b~1c")
    assert child == base / JsonPointer("/b~1c")
    assert child.path == "/a/b~1c"
    assert base in child
    assert child not in base
    assert len({child, JsonPointer("/a/b~1c")}) == 1
    assert child != "/a/b~1c"


def test_join_accepts_unescaped_parts_and_wraps_invalid_suffixes():
    assert JsonPointer("/a").join(["b/c", "~"]).path == "/a/b~1c/~0"
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        JsonPointer("/a").join(123)


def test_mapping_subclasses_and_duck_typed_indexing_are_supported():
    wrapped = UserDict({"key": "value"})
    assert jsonpointer.resolve_pointer(wrapped, "/key") == "value"

    class Indexable:
        def __getitem__(self, key):
            if key == "answer":
                return 42
            raise KeyError(key)

    assert jsonpointer.resolve_pointer(Indexable(), "/answer") == 42


def test_pairwise_handles_empty_singleton_and_generator_inputs():
    assert list(jsonpointer.pairwise([])) == []
    assert list(jsonpointer.pairwise([1])) == []
    assert list(jsonpointer.pairwise(i for i in range(4))) == [(0, 1), (1, 2), (2, 3)]


def test_verbose_exception_switch_controls_document_disclosure(monkeypatch):
    secret = {"password": "do-not-log"}
    with pytest.raises(JsonPointerException) as verbose:
        jsonpointer.resolve_pointer(secret, "/missing")
    assert "do-not-log" in str(verbose.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as quiet:
        jsonpointer.resolve_pointer(secret, "/missing")
    assert "do-not-log" not in str(quiet.value)
