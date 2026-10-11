import copy

import pytest

import jsonpointer
from jsonpointer import (
    EndOfList, JsonPointer, JsonPointerException, escape, pairwise,
    resolve_pointer, set_pointer, unescape,
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


# ---- escaping ----

def test_escape_basic():
    assert escape("a/b") == "a~1b"
    assert escape("m~n") == "m~0n"
    assert escape("~/") == "~0~1"
    assert escape("plain") == "plain"


def test_unescape_basic():
    assert unescape("a~1b") == "a/b"
    assert unescape("m~0n") == "m~n"
    assert unescape("~0~1") == "~/"


def test_unescape_tilde_one_literal():
    # "~01" must decode to "~1" (not "/"), per RFC 6901 section 4
    assert unescape("~01") == "~1"


def test_unescape_tilde_zero_after_slash():
    assert unescape("~1~0") == "/~"
    assert unescape("~0~0") == "~~"


def test_pointer_literal_tilde_one_key():
    doc = {"~1": "tilde-one", "/": "slash"}
    assert resolve_pointer(doc, "/~01") == "tilde-one"
    assert resolve_pointer(doc, "/~1") == "slash"


@pytest.mark.parametrize("s", ["", "a", "~", "/", "~1", "~0", "~01", "~10",
                               "~~//", "~0~1~01~10", "a~b/c", "//~~1"])
def test_escape_unescape_roundtrip(s):
    assert unescape(escape(s)) == s


@pytest.mark.parametrize("path", ["", "/a", "/~0", "/~1", "/~01", "/~10",
                                  "/a/~0/0/~1", "//", "/~0~1/~1~0"])
def test_path_roundtrip(path):
    assert JsonPointer(path).path == path
    assert str(JsonPointer(path)) == path


# ---- construction ----

def test_must_start_with_slash():
    with pytest.raises(JsonPointerException):
        JsonPointer("foo")
    with pytest.raises(JsonPointerException):
        JsonPointer("foo/bar")


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~b/c", "/a/~"])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_empty_pointer_has_no_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("").path == ""


def test_parts():
    p = JsonPointer("/a/b~1c/")
    assert p.get_parts() == ["a", "b/c", ""]


def test_empty_key_parts():
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]


# ---- resolve ----

def test_resolve_nested():
    doc = {"a": {"b": [1, {"c": [10, 20]}]}}
    assert resolve_pointer(doc, "/a/b/1/c/1") == 20
    assert resolve_pointer(doc, "/a/b/0") == 1


def test_resolve_root_returns_same_object():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_missing_key_default():
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/b", 0) == 0
    assert resolve_pointer({"a": 1}, "/a/b/c", "d") == "d"


def test_resolve_default_not_used_when_found():
    assert resolve_pointer({"a": 0}, "/a", "x") == 0
    assert resolve_pointer({"a": None}, "/a", "x") is None


def test_resolve_index_out_of_range():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    assert resolve_pointer([1, 2], "/2", "d") == "d"


@pytest.mark.parametrize("idx", ["01", "+1", "-1", "1.0", " 1", "1 ", "a", "", "00", "0x1", "1e0"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2], "/" + idx)


def test_invalid_array_index_default():
    assert resolve_pointer([0, 1], "/01", "d") == "d"


def test_valid_array_indices():
    lst = list(range(20))
    assert resolve_pointer(lst, "/0") == 0
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_array_index_with_trailing_garbage():
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2], "/1a")
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2], "/10x")


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    r = resolve_pointer(lst, "/-")
    assert isinstance(r, EndOfList)
    assert r.list_ is lst
    assert "EndOfList" in repr(r)


def test_tuple_is_sequence():
    assert resolve_pointer((1, 2, 3), "/2") == 3


def test_string_is_not_container():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")
    assert resolve_pointer({"a": "hello"}, "/a/0", "d") == "d"


def test_non_container_scalar():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 5}, "/a/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer(None, "/a")


def test_mapping_numeric_key_is_string():
    assert resolve_pointer({"1": "x"}, "/1") == "x"
    with pytest.raises(JsonPointerException):
        resolve_pointer({1: "x"}, "/1")


def test_duck_typed_getitem():
    class Duck:
        def __getitem__(self, k):
            if k == "x":
                return 42
            raise KeyError(k)

    assert resolve_pointer(Duck(), "/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer(Duck(), "/y")


def test_get_alias():
    p = JsonPointer("/a")
    assert p.get({"a": 3}) == 3
    assert p.resolve({"a": 3}) == 3


def test_verbose_exceptions_flag(monkeypatch):
    doc = {"secret": 1}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer(doc, "/missing")
    assert "secret" in str(e.value)
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer(doc, "/missing")
    assert "secret" not in str(e.value)
    assert "missing" in str(e.value)


# ---- set ----

def test_set_existing_key():
    doc = {"a": {"b": 1}}
    res = set_pointer(doc, "/a/b", 2)
    assert res is doc
    assert doc == {"a": {"b": 2}}


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/z", 9)
    assert doc == {"a": {"z": 9}}


def test_set_list_index():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", "x")
    assert doc == {"a": [1, "x", 3]}


def test_set_list_append_with_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    assert doc == {"a": [1, 2]}


def test_set_list_out_of_range():
    with pytest.raises(JsonPointerException):
        set_pointer([1], "/5", 0)


def test_set_list_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 0)


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "", 5)


def test_set_root_not_inplace_returns_value():
    doc = {"a": 1}
    assert set_pointer(doc, "", 5, inplace=False) == 5
    assert doc == {"a": 1}


def test_set_not_inplace_copies():
    doc = {"a": {"b": [1, 2]}}
    orig = copy.deepcopy(doc)
    res = set_pointer(doc, "/a/b/0", 99, inplace=False)
    assert doc == orig
    assert res == {"a": {"b": [99, 2]}}
    assert res is not doc


def test_set_in_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_missing_intermediate_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_on_scalar_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    assert doc == {"a/b": 1, "m~n": 2}


def test_set_percent_not_decoded():
    doc = {}
    set_pointer(doc, "/a%20b", "x")
    assert doc == {"a%20b": "x"}


def test_set_empty_key():
    doc = {}
    set_pointer(doc, "/", 1)
    assert doc == {"": 1}


def test_set_tuple_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 5)


def test_set_dash_through_nested_list():
    doc = [[1], [2]]
    set_pointer(doc, "/1/-", 3)
    assert doc == [[1], [2, 3]]


# ---- to_last / get_part / walk ----

def test_to_last():
    doc = {"a": {"b": 1}}
    assert JsonPointer("/a/b").to_last(doc) == ({"b": 1}, "b")
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_to_last_list_converts_index():
    doc = {"a": [5, 6]}
    sub, last = JsonPointer("/a/1").to_last(doc)
    assert sub == [5, 6]
    assert last == 1 and isinstance(last, int)


def test_get_part():
    assert JsonPointer.get_part({}, "k") == "k"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([1], "x")


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([7, 8], "1") == 8
    with pytest.raises(JsonPointerException):
        p.walk([7], "1")
    with pytest.raises(JsonPointerException):
        p.walk({}, "a")


# ---- contains / join / eq / hash ----

def test_contains():
    p = JsonPointer("/a/b")
    assert p.contains(JsonPointer("/a"))
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer(""))
    assert not p.contains(JsonPointer("/a/b/c"))
    assert not p.contains(JsonPointer("/b"))
    assert not p.contains(JsonPointer("/a/c"))
    assert JsonPointer("/a") in p
    assert JsonPointer("/x") not in p


def test_join_variants():
    p = JsonPointer("/a")
    assert p.join("/b/c") == JsonPointer("/a/b/c")
    assert p.join(JsonPointer("/b")) == JsonPointer("/a/b")
    assert p.join(["b", "c"]) == JsonPointer("/a/b/c")
    assert p.join(["~", "/"]).path == "/a/~0/~1"
    assert p.join("").path == "/a"


def test_join_does_not_mutate():
    p = JsonPointer("/a")
    p.join("/b")
    assert p.path == "/a"


def test_truediv():
    assert (JsonPointer("/a") / "/b") == JsonPointer("/a/b")
    assert (JsonPointer("/a") / "/b" / "/c").path == "/a/b/c"


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_eq_and_hash():
    a, b = JsonPointer("/a/b"), JsonPointer("/a/b")
    assert a == b
    assert hash(a) == hash(b)
    assert a != JsonPointer("/a/c")
    assert a != "/a/b"
    assert len({a, b, JsonPointer("/x")}) == 2
    assert JsonPointer("/~01") != JsonPointer("/~1")


def test_repr_str():
    p = JsonPointer("/a~1b")
    assert str(p) == "/a~1b"
    assert repr(p) == "JsonPointer('/a~1b')"


def test_from_parts():
    p = JsonPointer.from_parts(["a", "~", "/", 0])
    assert p.path == "/a/~0/~1/0"
    assert p.parts == ["a", "~", "/", "0"]
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]
    assert JsonPointer.from_parts(["~01"]).parts == ["~01"]


def test_from_parts_roundtrip_resolve():
    doc = {"a/b": {"~1": [10, 20]}}
    p = JsonPointer.from_parts(["a/b", "~1", 1])
    assert p.resolve(doc) == 20


def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
