"""Behavioral tests for the local RFC 6901 JSON Pointer implementation."""

from collections.abc import Mapping

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
def test_rfc_examples_resolve(pointer, expected):
    assert resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_tokens_are_not_url_decoded():
    doc = {"a%20b": "encoded", "a b": "space"}
    assert resolve_pointer(doc, "/a%20b") == "encoded"
    assert resolve_pointer(doc, "/a b") == "space"


@pytest.mark.parametrize("pointer", ["foo", "#", "a/b"])
def test_nonempty_pointer_must_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match=r"^Location must start with /$"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/x~9y", "/ok/~a"])
def test_invalid_tilde_escapes_are_rejected(pointer):
    with pytest.raises(JsonPointerException, match=r"^Found invalid escape ~"):
        JsonPointer(pointer)


def test_valid_escapes_are_decoded_in_the_required_order():
    # ~01 means a literal "~1", not "/"; replacement order is significant.
    assert JsonPointer("/~01/~10/~00").get_parts() == ["~1", "/0", "~0"]
    assert unescape("~01~1~0") == "~1/~"
    assert escape("~/~1") == "~0~1~01"


def test_path_and_from_parts_round_trip_strings_and_non_strings():
    pointer = JsonPointer.from_parts(["a/b", "m~n", "", 0, False])
    assert pointer.parts == ["a/b", "m~n", "", "0", "False"]
    assert pointer.path == "/a~1b/m~0n//0/False"
    assert str(pointer) == pointer.path
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n//0/False')"
    assert JsonPointer(pointer.path) == pointer


@pytest.mark.parametrize(
    "token", ["00", "01", "-1", "+1", "1.0", " 1", "1 ", "1x", "x1"]
)
def test_invalid_array_indices_are_rejected(token):
    with pytest.raises(JsonPointerException, match="not a valid sequence index"):
        resolve_pointer(["zero", "one"], "/" + token)


@pytest.mark.parametrize(("token", "expected"), [("0", "a"), ("1", "b"), ("10", "k")])
def test_valid_array_indices(token, expected):
    assert resolve_pointer(list("abcdefghijk"), "/" + token) == expected


def test_missing_members_and_indices_raise_pointer_exceptions():
    with pytest.raises(JsonPointerException, match=r"index '2' is out of bounds"):
        resolve_pointer([0], "/2")
    with pytest.raises(JsonPointerException, match=r"member 'missing' not found"):
        resolve_pointer({"present": 1}, "/missing")


def test_default_is_returned_only_for_pointer_lookup_failures():
    marker = object()
    assert resolve_pointer({}, "/missing", marker) is marker
    assert resolve_pointer([], "/5", marker) is marker
    assert resolve_pointer(3, "/anything", marker) is marker
    assert resolve_pointer({"x": None}, "/x", marker) is None
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, "/missing")


def test_resolution_does_not_swallow_errors_from_custom_getitem():
    class Broken:
        def __getitem__(self, key):
            raise RuntimeError(key)

    with pytest.raises(RuntimeError, match="boom"):
        resolve_pointer(Broken(), "/boom", "default")


def test_dash_on_sequence_returns_end_marker_for_same_sequence():
    values = [1, 2]
    result = resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_mapping_tokens_remain_strings_even_when_numeric():
    assert resolve_pointer({"0": "string key", 0: "integer key"}, "/0") == "string key"


def test_strings_are_not_treated_as_indexable_sequences():
    with pytest.raises(JsonPointerException, match="Cannot apply token '0' to non-container type"):
        resolve_pointer("abc", "/0")


def test_duck_typed_indexing_is_supported():
    class Indexable:
        def __getitem__(self, key):
            return (key, "looked up")

    assert resolve_pointer(Indexable(), "/raw-token") == ("raw-token", "looked up")


def test_custom_mapping_is_recognized_as_mapping():
    class TinyMapping(Mapping):
        def __getitem__(self, key):
            return {"2": "two"}[key]

        def __iter__(self):
            return iter(("2",))

        def __len__(self):
            return 1

    assert JsonPointer.get_part(TinyMapping(), "2") == "2"
    assert resolve_pointer(TinyMapping(), "/2") == "two"


def test_non_container_reports_document_type():
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        JsonPointer.get_part(42, "x")


def test_verbose_exception_setting_controls_document_disclosure(monkeypatch):
    secret = {"secret": "do-not-log"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as verbose:
        resolve_pointer(secret, "/absent")
    assert "do-not-log" in str(verbose.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as terse:
        resolve_pointer(secret, "/absent")
    assert str(terse.value) == "member 'absent' not found"
    assert "do-not-log" not in str(terse.value)


def test_set_existing_and_new_mapping_members_in_place():
    doc = {"outer": {"old": 1}}
    returned = set_pointer(doc, "/outer/old", 2)
    assert returned is doc
    assert doc == {"outer": {"old": 2}}
    set_pointer(doc, "/outer/a~1b", 3)
    assert doc["outer"]["a/b"] == 3


def test_set_list_item_and_append_with_dash():
    doc = {"items": ["a", "b"]}
    assert set_pointer(doc, "/items/1", "B") is doc
    assert set_pointer(doc, "/items/-", "c") is doc
    assert doc == {"items": ["a", "B", "c"]}


def test_non_inplace_set_deep_copies_all_nested_data():
    doc = {"items": [{"value": 1}], "untouched": [1]}
    changed = set_pointer(doc, "/items/0/value", 9, inplace=False)
    assert changed == {"items": [{"value": 9}], "untouched": [1]}
    assert doc == {"items": [{"value": 1}], "untouched": [1]}
    assert changed is not doc
    assert changed["items"] is not doc["items"]
    assert changed["untouched"] is not doc["untouched"]


def test_root_can_only_be_replaced_out_of_place():
    original = {"a": 1}
    replacement = {"b": 2}
    assert set_pointer(original, "", replacement, inplace=False) is replacement
    assert original == {"a": 1}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(original, "", replacement)


def test_invalid_assignment_targets_are_wrapped():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer([], "/1", "outside")
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        set_pointer({"number": 1}, "/number/x", 2)
    with pytest.raises(JsonPointerException, match="Cannot apply token '0' to non-container type"):
        set_pointer({"text": "abc"}, "/text/0", "A")


def test_set_defensively_rejects_a_string_parent(monkeypatch):
    pointer = JsonPointer("/x")
    monkeypatch.setattr(pointer, "to_last", lambda doc: ("abc", 0))
    with pytest.raises(JsonPointerException, match=r"^Cannot set value in a string$"):
        pointer.set({}, "A")


def test_to_last_returns_parent_and_typed_final_part():
    doc = {"rows": [{"name": "Ada"}]}
    assert JsonPointer("").to_last(doc) == (doc, None)
    parent, part = JsonPointer("/rows/0/name").to_last(doc)
    assert parent is doc["rows"][0]
    assert part == "name"
    parent, part = JsonPointer("/rows/0").to_last(doc)
    assert parent is doc["rows"]
    assert part == 0


def test_contains_uses_ancestor_direction_and_token_boundaries():
    root = JsonPointer("")
    a = JsonPointer("/a")
    ab = JsonPointer("/a/b")
    abc = JsonPointer("/a/b/c")
    assert ab.contains(a)
    assert a in ab
    assert abc.contains(ab)
    assert abc.contains(root)
    assert ab.contains(ab)
    assert not a.contains(ab)
    assert not JsonPointer("/ab").contains(a)
    assert not ab.contains(JsonPointer("/x"))


def test_join_accepts_pointer_string_and_iterable_without_mutating_inputs():
    base = JsonPointer("/a~1b")
    suffix = JsonPointer("/c~0d")
    assert base.join(suffix).path == "/a~1b/c~0d"
    assert base.join("/e/f").path == "/a~1b/e/f"
    assert base.join(iter(["g/h", 2])).path == "/a~1b/g~1h/2"
    assert (base / "/z").path == "/a~1b/z"
    assert base.path == "/a~1b"
    assert suffix.path == "/c~0d"


def test_join_normalizes_failures_to_pointer_exception():
    with pytest.raises(JsonPointerException, match=r"^Invalid suffix$"):
        JsonPointer("/a").join(3)
    with pytest.raises(JsonPointerException, match=r"^Location must start with /$"):
        JsonPointer("/a").join("not-a-pointer")


def test_equality_hashing_and_get_alias():
    first = JsonPointer("/a~1b")
    equivalent = JsonPointer.from_parts(["a/b"])
    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert len({first, equivalent}) == 1
    assert first != JsonPointer("/a/b")
    assert first != "/a~1b"
    assert JsonPointer("/x").get({"x": 4}) == 4


def test_pairwise_for_empty_singleton_sequence_and_iterator():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]
