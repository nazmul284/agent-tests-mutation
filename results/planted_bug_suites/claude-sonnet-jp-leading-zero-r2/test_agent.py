import pytest

import jsonpointer
from jsonpointer import (
    JsonPointer, JsonPointerException, EndOfList,
    resolve_pointer, set_pointer, escape, unescape, pairwise,
)

# RFC 6901 section 5 example document
RFC_DOC = {
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

RFC_CASES = [
    ("", RFC_DOC),
    ("/foo", ["bar", "baz"]),
    ("/foo/0", "bar"),
    ("/", 0),
    ("/a~1b", 1),
    ("/c%d", 2),
    ("/e^f", 3),
    ("/g|h", 4),
    ("/i\\j", 5),
    ("/k\"l", 6),
    ("/ ", 7),
    ("/m~0n", 8),
]


@pytest.mark.parametrize("ptr,expected", RFC_CASES)
def test_rfc_examples(ptr, expected):
    assert resolve_pointer(RFC_DOC, ptr) == expected


def test_root_returns_same_object():
    assert resolve_pointer(RFC_DOC, "") is RFC_DOC


# ---- construction / validation ----

@pytest.mark.parametrize("bad", ["a", "foo/bar", "~0", " /a"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~/b", "/~ "])
def test_invalid_escapes(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_parts_and_unescape():
    p = JsonPointer("/a~1b/m~0n/~01/~10")
    assert p.parts == ["a/b", "m~n", "~1", "/0"]
    assert p.get_parts() == p.parts


def test_empty_token_parts():
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("").parts == []


def test_escape_unescape():
    assert escape("a/b~c") == "a~1b~0c"
    assert unescape("a~1b~0c") == "a/b~c"
    # order matters: ~01 is literally "~1", not "/"
    assert unescape("~01") == "~1"
    assert escape("~1") == "~01"
    for s in ["", "~", "/", "~/~", "~0~1", "//~~"]:
        assert unescape(escape(s)) == s


def test_path_roundtrip():
    for s in ["", "/", "//", "/a~1b/m~0n", "/~01/~10", "/foo/0"]:
        assert JsonPointer(s).path == s
        assert str(JsonPointer(s)) == s


def test_repr():
    assert repr(JsonPointer("/a~1b")) == "JsonPointer('/a~1b')"


# ---- resolve ----

def test_resolve_nested():
    doc = {"a": [{"b": [10, 20, {"c": "x"}]}]}
    assert resolve_pointer(doc, "/a/0/b/2/c") == "x"
    assert resolve_pointer(doc, "/a/0/b/1") == 20


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_default():
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/b", 0) == 0
    assert resolve_pointer({"a": [1]}, "/a/5", "d") == "d"
    assert resolve_pointer({"a": [1]}, "/a/x", "d") == "d"
    assert resolve_pointer({"a": 1}, "/a", "d") == 1


def test_resolve_default_through_scalar():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/a/b")
    assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"
    assert resolve_pointer({"a": "str"}, "/a/0", "d") == "d"


def test_get_alias():
    assert JsonPointer("/a").get({"a": 5}) == 5


def test_string_not_indexable():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")


def test_list_index_out_of_bounds():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")


def test_list_last_index_ok():
    assert resolve_pointer([1, 2, 3], "/2") == 3
    assert resolve_pointer([1, 2, 3], "/0") == 1


@pytest.mark.parametrize("tok", ["-1", "+1", " 1", "1 ", "1.0", "a", "", "1e0", "0x1"])
def test_invalid_list_index_tokens(tok):
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/" + tok)


@pytest.mark.parametrize("tok", ["00", "01", "001", "012"])
def test_leading_zero_index_rejected(tok):
    # RFC 6901: array indices must not have leading zeros
    doc = list(range(20))
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/" + tok)
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(doc, tok)


def test_leading_zero_index_rejected_on_set():
    doc = [1, 2, 3]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/01", 9)
    assert doc == [1, 2, 3]


def test_zero_index_ok():
    assert JsonPointer.get_part([1], "0") == 0
    assert resolve_pointer([7], "/0") == 7
    assert JsonPointer.get_part([0] * 11, "10") == 10


def test_non_ascii_digits_rejected():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/١")  # arabic-indic digit one


def test_dash_resolves_to_end_of_list():
    doc = [1, 2]
    r = resolve_pointer(doc, "/-")
    assert isinstance(r, EndOfList)
    assert r.list_ is doc
    assert "EndOfList" in repr(r)


def test_digit_key_in_dict_stays_string():
    assert resolve_pointer({"0": "a", "01": "b"}, "/0") == "a"
    assert resolve_pointer({"0": "a", "01": "b"}, "/01") == "b"
    assert resolve_pointer({0: "int"}, "/0", "d") == "d"


def test_tuple_is_sequence():
    assert resolve_pointer({"a": (1, 2)}, "/a/1") == 2


def test_duck_typed_getitem():
    class Thing:
        def __getitem__(self, k):
            if k == "x":
                return 42
            raise KeyError(k)

    assert resolve_pointer({"t": Thing()}, "/t/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer({"t": Thing()}, "/t/y")


def test_non_indexable_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 5}, "/a/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": None}, "/a/b")


def test_verbose_exceptions_toggle(monkeypatch):
    doc = {"secret": "SECRETVALUE"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(doc, "/nope")
    assert "SECRETVALUE" in str(ei.value)
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(doc, "/nope")
    assert "SECRETVALUE" not in str(ei.value)
    assert "nope" in str(ei.value)


# ---- to_last / get_part ----

def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(doc) == (doc, None)
    assert JsonPointer("/a").to_last(doc) == (doc, "a")
    sub, last = JsonPointer("/a/b/1").to_last(doc)
    assert sub is doc["a"]["b"] and last == 1
    sub, last = JsonPointer("/a/b/-").to_last(doc)
    assert sub is doc["a"]["b"] and last == "-"


def test_get_part_types():
    assert JsonPointer.get_part({}, "5") == "5"
    assert JsonPointer.get_part([1] * 10, "5") == 5
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": {"b": 1}}
    result = set_pointer(doc, "/a/b", 2)
    assert result is doc
    assert doc == {"a": {"b": 2}}


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/new", [1])
    assert doc == {"a": {"new": [1]}}


def test_set_list_element():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", "x")
    assert doc == {"a": [1, "x", 3]}


def test_set_list_append_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    set_pointer(doc, "/a/-", 3)
    assert doc == {"a": [1, 2, 3]}


def test_set_list_out_of_range():
    doc = [1]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/5", 0)
    assert doc == [1]


def test_set_list_bad_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1], "/x", 0)


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "", 5)


def test_set_root_not_inplace_returns_value():
    doc = {"a": 1}
    assert set_pointer(doc, "", 5, inplace=False) == 5
    assert doc == {"a": 1}


def test_set_not_inplace_copies():
    doc = {"a": {"b": [1, 2]}}
    out = set_pointer(doc, "/a/b/0", 9, inplace=False)
    assert out == {"a": {"b": [9, 2]}}
    assert doc == {"a": {"b": [1, 2]}}
    assert out is not doc
    assert out["a"] is not doc["a"]


def test_set_not_inplace_append():
    doc = [1]
    assert set_pointer(doc, "/-", 2, inplace=False) == [1, 2]
    assert doc == [1]


def test_set_in_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_through_missing_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 1)


def test_set_scalar_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    assert doc == {"a/b": 1, "m~n": 2}


def test_set_percent_literal():
    doc = {"foo": {}}
    set_pointer(doc, "/foo/a%20b", "x")
    assert doc == {"foo": {"a%20b": "x"}}


# ---- contains / join / eq / hash ----

def test_contains():
    assert JsonPointer("/a/b/c") in JsonPointer("/a/b/c")
    p = JsonPointer("/a/b")
    assert JsonPointer("/a/b/c").contains(p)
    assert p in JsonPointer("/a/b/c")
    assert JsonPointer("/a/b") in JsonPointer("/a/b/c")
    assert not JsonPointer("/a/b/c").contains(JsonPointer("/a/x"))
    assert JsonPointer("/a/b").contains(JsonPointer("/a/b/c")) is False
    assert JsonPointer("/a/x") not in JsonPointer("/a/b/c")
    assert JsonPointer("") in JsonPointer("/a")
    assert JsonPointer("/a") not in JsonPointer("")


def test_contains_is_token_based():
    assert JsonPointer("/ab") not in JsonPointer("/abc")
    assert JsonPointer("/a") not in JsonPointer("/ab")


def test_join_variants():
    base = JsonPointer("/a")
    assert base.join("/b/c").path == "/a/b/c"
    assert base.join(JsonPointer("/b")).path == "/a/b"
    assert base.join(["b", "c/d"]).path == "/a/b/c~1d"
    assert (base / "/b").path == "/a/b"
    assert (base / JsonPointer("/b")).path == "/a/b"
    assert JsonPointer("").join("/x").path == "/x"
    assert base.join("").path == "/a"


def test_join_does_not_mutate():
    base = JsonPointer("/a")
    base.join("/b")
    assert base.path == "/a"


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_join_subclass_preserved():
    class Sub(JsonPointer):
        pass
    assert isinstance(Sub("/a").join("/b"), Sub)


def test_equality_and_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != "/a"
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert JsonPointer("/") != JsonPointer("")


# ---- from_parts ----

def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    p = JsonPointer.from_parts(["a/b", "m~n", "~1"])
    assert p.parts == ["a/b", "m~n", "~1"]


def test_from_parts_roundtrip_resolve():
    doc = {"a/b": {"m~n": [5, 6]}}
    p = JsonPointer.from_parts(["a/b", "m~n", 1])
    assert p.resolve(doc) == 6


# ---- pairwise ----

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]
