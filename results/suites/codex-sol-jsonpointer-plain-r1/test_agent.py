"""Behavioral tests for the public API in :mod:`jsonpointer`.

The examples deliberately include values whose Python truthiness could hide a
bad implementation (zero, ``False``, and ``None``), and tokens where changing
the order of escaping operations changes the result.
"""

from collections import UserDict

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
def test_resolves_rfc_6901_examples(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_empty_pointer_returns_the_exact_root_object():
    doc = {"value": []}
    assert resolve_pointer(doc, "") is doc


@pytest.mark.parametrize("value", [None, False, 0, "", []])
def test_resolve_preserves_falsey_values(value):
    assert resolve_pointer({"value": value}, "/value") == value


def test_resolve_default_is_returned_for_failure_at_any_depth():
    marker = object()
    doc = {"present": {"items": [1]}}
    assert resolve_pointer(doc, "/missing/child", marker) is marker
    assert resolve_pointer(doc, "/present/items/9/child", marker) is marker


def test_missing_value_raises_without_an_explicit_default():
    with pytest.raises(JsonPointerException, match="member 'missing' not found"):
        resolve_pointer({}, "/missing")


@pytest.mark.parametrize("pointer", ["foo", "#", "not/a/pointer"])
def test_pointer_must_be_empty_or_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/a~xb", "/ok/bad~9"])
def test_rejects_invalid_tilde_escapes(pointer):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(pointer)


def test_escape_and_unescape_use_rfc_ordering():
    raw = "a~/b~1c"
    encoded = "a~0~1b~01c"
    assert escape(raw) == encoded
    assert unescape(encoded) == raw
    assert JsonPointer.from_parts([raw]).path == "/" + encoded


def test_path_round_trip_keeps_empty_and_escaped_tokens_distinct():
    pointer = JsonPointer("//~0/~1/~01")
    assert pointer.get_parts() == ["", "~", "/", "~1"]
    assert pointer.path == "//~0/~1/~01"
    assert str(pointer) == pointer.path
    assert repr(pointer) == "JsonPointer('//~0/~1/~01')"


@pytest.mark.parametrize("bad_index", ["00", "01", "+1", "-1", " 1", "1 ", "1.0", ""])
def test_sequence_indices_have_strict_json_pointer_syntax(bad_index):
    with pytest.raises(JsonPointerException, match="not a valid sequence index"):
        resolve_pointer(["zero", "one"], "/" + bad_index)


def test_sequence_index_zero_and_multi_digit_indices_work():
    values = list(range(12))
    assert resolve_pointer(values, "/0") == 0
    assert resolve_pointer(values, "/10") == 10


def test_out_of_bounds_sequence_index_has_library_exception():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([], "/0")


def test_dash_resolves_to_an_end_of_list_marker():
    values = [1, 2]
    result = resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_dash_is_an_ordinary_key_in_a_mapping():
    assert resolve_pointer({"-": "value"}, "/-") == "value"


def test_cannot_walk_through_scalar_values():
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer({"answer": 42}, "/answer/child")
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        resolve_pointer("abc", "/0")


def test_mapping_subclasses_and_duck_typed_objects_are_supported():
    class Lookup:
        def __getitem__(self, key):
            if key != "answer":
                raise KeyError(key)
            return 42

    assert resolve_pointer(UserDict({"x": 1}), "/x") == 1
    assert resolve_pointer(Lookup(), "/answer") == 42


def test_set_replaces_mapping_and_list_values_in_place():
    doc = {"obj": {"old": 1}, "items": [10, 20]}
    returned = set_pointer(doc, "/obj/old", 2)
    assert returned is doc
    assert doc == {"obj": {"old": 2}, "items": [10, 20]}
    set_pointer(doc, "/items/1", 99)
    assert doc["items"] == [10, 99]


def test_set_can_add_mapping_member_and_append_to_list():
    doc = {"obj": {}, "items": [1]}
    set_pointer(doc, "/obj/a~1b", "slash")
    set_pointer(doc, "/items/-", 2)
    assert doc == {"obj": {"a/b": "slash"}, "items": [1, 2]}


def test_non_inplace_set_deep_copies_before_assignment():
    original = {"nested": {"items": [{"value": 1}]}}
    changed = set_pointer(original, "/nested/items/0/value", 7, inplace=False)
    assert changed == {"nested": {"items": [{"value": 7}]}}
    assert original == {"nested": {"items": [{"value": 1}]}}
    assert changed is not original
    assert changed["nested"] is not original["nested"]


def test_root_replacement_requires_non_inplace_mode():
    root = {"old": True}
    replacement = {"new": True}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(root, "", replacement)
    assert set_pointer(root, "", replacement, inplace=False) is replacement
    assert root == {"old": True}


def test_invalid_assignment_targets_are_wrapped():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer([1], "/2", 3)
    with pytest.raises(JsonPointerException, match="Cannot apply token"):
        set_pointer({"text": "abc"}, "/text/0", "x")


def test_to_last_returns_parent_and_converted_final_part():
    parent, part = JsonPointer("/items/1").to_last({"items": [10, 20]})
    assert parent == [10, 20]
    assert part == 1
    root = {"x": 1}
    assert JsonPointer("").to_last(root) == (root, None)


def test_join_accepts_pointer_string_and_iterable_suffixes():
    base = JsonPointer("/a~1b")
    assert base.join(JsonPointer("/c~0d")).path == "/a~1b/c~0d"
    assert (base / "/c").path == "/a~1b/c"
    assert base.join(["x/y", "~", 3]).path == "/a~1b/x~1y/~0/3"
    assert base.path == "/a~1b"


def test_invalid_join_suffix_raises_library_exception():
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        JsonPointer("/a").join(42)
    with pytest.raises(JsonPointerException, match="Location must start with /"):
        JsonPointer("/a").join("not-a-pointer")


def test_containment_uses_ancestor_direction_and_token_boundaries():
    root = JsonPointer("")
    ancestor = JsonPointer("/a")
    descendant = JsonPointer("/a/b")
    sibling_prefix = JsonPointer("/ab")
    assert descendant.contains(ancestor)
    assert ancestor in descendant
    assert descendant.contains(root)
    assert not ancestor.contains(descendant)
    assert not descendant.contains(sibling_prefix)


def test_equality_and_hash_are_based_on_unescaped_parts():
    first = JsonPointer("/a~1b/~0")
    same = JsonPointer.from_parts(["a/b", "~"])
    different = JsonPointer("/a/b/~0")
    assert first == same
    assert hash(first) == hash(same)
    assert first != different
    assert first != first.path
    assert len({first, same, different}) == 2


def test_verbose_exception_flag_controls_document_disclosure(monkeypatch):
    secret_doc = {"secret": "do not print", "safe": 1}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as verbose:
        resolve_pointer(secret_doc, "/missing")
    assert "secret" in str(verbose.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as terse:
        resolve_pointer(secret_doc, "/missing")
    assert "secret" not in str(terse.value)
    assert str(terse.value) == "member 'missing' not found"


@pytest.mark.parametrize(
    ("items", "expected"),
    [([], []), ([1], []), ([1, 2], [(1, 2)]), ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]),
    ],
)
def test_pairwise(items, expected):
    assert list(pairwise(items)) == expected


def test_pairwise_consumes_a_generator_lazily_and_once():
    seen = []

    def values():
        for value in range(4):
            seen.append(value)
            yield value

    pairs = pairwise(values())
    # pairwise primes the second tee iterator so that the first produced pair
    # is offset by one item.
    assert seen == [0]
    assert next(pairs) == (0, 1)
    assert seen == [0, 1]
    assert list(pairs) == [(1, 2), (2, 3)]
    assert seen == [0, 1, 2, 3]
