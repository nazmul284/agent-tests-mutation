"""Behavioral tests for the public jsonpointer API.

The examples deliberately use only JSON-compatible containers except where a
custom indexable object is needed to exercise the documented duck-typing path.
"""

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
def test_resolve_rfc_examples(pointer, expected):
    assert jsonpointer.resolve_pointer(RFC_DOCUMENT, pointer) == expected


def test_pointer_is_not_a_uri_fragment_or_percent_decoder():
    document = {"a%20b": "encoded", "a b": "space"}
    assert jsonpointer.resolve_pointer(document, "/a%20b") == "encoded"
    assert jsonpointer.resolve_pointer(document, "/a b") == "space"
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer("#/a%20b")


@pytest.mark.parametrize("pointer", ["a/b", "~", "/a~", "/a~2b", "/~~0"])
def test_invalid_pointer_syntax_is_rejected(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


@pytest.mark.parametrize("token", ["", "plain", "a/b", "m~n", "~/~", "0"])
def test_escape_round_trip(token):
    assert jsonpointer.unescape(jsonpointer.escape(token)) == token


@pytest.mark.xfail(
    reason="known RFC 6901 bug: replacement must not decode the newly formed '~1'",
)
def test_unescape_does_not_process_an_escape_twice():
    assert jsonpointer.unescape("~01") == "~1"


@pytest.mark.parametrize("index", ["00", "01", "+1", "-1", " 1", "1.0", "1e0"])
def test_noncanonical_array_indices_are_rejected(index):
    with pytest.raises(JsonPointerException, match="valid sequence index"):
        jsonpointer.resolve_pointer(["zero", "one"], "/" + index)


def test_missing_members_and_indices_raise_or_return_explicit_default():
    marker = object()
    with pytest.raises(JsonPointerException, match="member 'missing' not found"):
        jsonpointer.resolve_pointer({}, "/missing")
    with pytest.raises(JsonPointerException, match="out of bounds"):
        jsonpointer.resolve_pointer([], "/0")
    assert jsonpointer.resolve_pointer({}, "/missing", marker) is marker
    assert jsonpointer.resolve_pointer({}, "/missing", None) is None


def test_dash_resolves_to_end_of_list_marker():
    values = [1, 2]
    result = jsonpointer.resolve_pointer(values, "/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is values
    assert repr(result) == "EndOfList([1, 2])"


def test_set_replaces_mapping_and_sequence_values_in_place():
    document = {"items": [{"name": "old"}], "untouched": True}
    returned = jsonpointer.set_pointer(document, "/items/0/name", "new")
    assert returned is document
    assert document == {"items": [{"name": "new"}], "untouched": True}


def test_set_can_add_mapping_member_and_append_to_list():
    document = {"mapping": {}, "items": [1]}
    jsonpointer.set_pointer(document, "/mapping/a~1b", 2)
    jsonpointer.set_pointer(document, "/items/-", 3)
    assert document == {"mapping": {"a/b": 2}, "items": [1, 3]}


def test_non_inplace_set_deep_copies_and_can_replace_root():
    original = {"nested": {"value": 1}}
    changed = jsonpointer.set_pointer(original, "/nested/value", 2, inplace=False)
    assert changed == {"nested": {"value": 2}}
    assert original == {"nested": {"value": 1}}
    assert changed is not original
    assert changed["nested"] is not original["nested"]
    replacement = {"new": []}
    assert jsonpointer.set_pointer(original, "", replacement, inplace=False) is replacement


def test_setting_root_in_place_is_rejected_without_mutation():
    document = {"value": 1}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        jsonpointer.set_pointer(document, "", 2)
    assert document == {"value": 1}


@pytest.mark.parametrize(
    ("document", "pointer", "message"),
    [
        ({"x": "text"}, "/x/0", "Cannot apply token.*non-container"),
        ({"x": []}, "/x/1", "Invalid assignment target"),
        ({"x": 1}, "/x/y", "does not support indexing"),
    ],
)
def test_invalid_set_targets_raise_domain_exception(document, pointer, message):
    with pytest.raises(JsonPointerException, match=message):
        jsonpointer.set_pointer(document, pointer, "value")


def test_pointer_representation_parts_equality_and_hashing():
    pointer = JsonPointer.from_parts(["a/b", "m~n", 0, ""])
    assert pointer.path == "/a~1b/m~0n/0/"
    assert str(pointer) == pointer.path
    assert repr(pointer) == "JsonPointer('/a~1b/m~0n/0/')"
    assert pointer.get_parts() == ["a/b", "m~n", "0", ""]
    assert pointer == JsonPointer("/a~1b/m~0n/0/")
    assert pointer != pointer.path
    assert len({pointer, JsonPointer(pointer.path)}) == 1


def test_join_division_and_containment_do_not_mutate_operands():
    base = JsonPointer("/a~1b")
    suffix = JsonPointer("/c~0d")
    joined = base.join(suffix)
    assert joined == base / ["c~d"]
    assert joined.path == "/a~1b/c~0d"
    assert base.path == "/a~1b"
    assert suffix.path == "/c~0d"
    assert base.contains(joined) is False
    assert joined.contains(base) is True
    assert base in joined


def test_to_last_returns_typed_final_component():
    document = {"items": ["zero", "one"]}
    parent, component = JsonPointer("/items/1").to_last(document)
    assert parent is document["items"]
    assert component == 1
    assert JsonPointer("").to_last(document) == (document, None)


def test_custom_indexable_objects_are_supported():
    class Indexable:
        def __getitem__(self, key):
            if key == "answer":
                return 42
            raise KeyError(key)

    assert JsonPointer("/answer").resolve(Indexable()) == 42


def test_pairwise_handles_empty_single_and_generator_inputs():
    assert list(jsonpointer.pairwise([])) == []
    assert list(jsonpointer.pairwise([1])) == []
    assert list(jsonpointer.pairwise(x for x in range(4))) == [(0, 1), (1, 2), (2, 3)]
