import copy

import pytest

import jsonpointer
from jsonpointer import (
    JsonPointer, JsonPointerException, EndOfList, escape, unescape,
    resolve_pointer, set_pointer, pairwise,
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
    ('/k"l', 6),
    ("/ ", 7),
    ("/m~0n", 8),
]


@pytest.mark.parametrize("ptr,expected", RFC_CASES)
def test_rfc_examples(ptr, expected):
    assert resolve_pointer(RFC_DOC, ptr) == expected


# ---- escape / unescape ----

def test_escape_basic():
    assert escape("a/b") == "a~1b"
    assert escape("m~n") == "m~0n"
    assert escape("~/") == "~0~1"
    assert escape("plain") == "plain"


def test_unescape_basic():
    assert unescape("a~1b") == "a/b"
    assert unescape("m~0n") == "m~n"
    assert unescape("~0~1") == "~/"


def test_unescape_order_tilde_one():
    # "~01" must become "~1", not "/"
    assert unescape("~01") == "~1"
    assert unescape("~00") == "~0"
    assert unescape("~0~01") == "~~1"


@pytest.mark.parametrize("s", ["~1", "~0", "~01", "~10", "/~", "~/~0~1", "a~b/c", "~~", "//", ""])
def test_escape_unescape_roundtrip(s):
    assert unescape(escape(s)) == s


def test_pointer_with_tilde_zero_one_resolves_literal_key():
    doc = {"~1": "tilde-one", "/": "slash"}
    assert resolve_pointer(doc, "/~01") == "tilde-one"
    assert resolve_pointer(doc, "/~1") == "slash"
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/~10").parts == ["/0"]


# ---- construction ----

def test_must_start_with_slash():
    with pytest.raises(JsonPointerException):
        JsonPointer("foo")
    with pytest.raises(JsonPointerException):
        JsonPointer("foo/bar")


@pytest.mark.parametrize("bad", ["/~", "/~2", "/a~", "/~x/b", "/a/~", "/a~b", "/~ "])
def test_invalid_escapes(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_empty_pointer_has_no_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("").get_parts() == []


def test_parts_parsing():
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b/c").parts == ["a", "b", "c"]
    assert JsonPointer("/a~1b/c~0d").get_parts() == ["a/b", "c~d"]


# ---- resolve ----

def test_resolve_root():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc


def test_resolve_nested():
    doc = {"a": {"b": [10, {"c": "x"}]}}
    assert resolve_pointer(doc, "/a/b/1/c") == "x"
    assert resolve_pointer(doc, "/a/b/0") == 10
    assert resolve_pointer(doc, "/a/b") == [10, {"c": "x"}]


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_default():
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/b", 5) == 5
    assert resolve_pointer({"a": [1]}, "/a/3", "d") == "d"
    assert resolve_pointer({"a": [1]}, "/a/x", "d") == "d"
    assert resolve_pointer({"a": 1}, "/a", "d") == 1


def test_resolve_default_falsy_value_not_replaced():
    assert resolve_pointer({"a": 0}, "/a", "d") == 0
    assert resolve_pointer({"a": None}, "/a", "d") is None


def test_resolve_index_out_of_bounds():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")


def test_resolve_negative_index_rejected():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/-1")


@pytest.mark.parametrize("idx", ["01", "00", "+1", " 1", "1 ", "1.0", "a", "", "0x1", "1e0", "١"])
def test_resolve_invalid_array_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/" + idx)


def test_resolve_index_zero_and_multi_digit():
    lst = list(range(20))
    assert resolve_pointer(lst, "/0") == 0
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_resolve_dash_gives_end_of_list():
    lst = [1, 2]
    res = resolve_pointer(lst, "/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst
    assert "EndOfList" in repr(res)


def test_resolve_into_string_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")


def test_resolve_into_scalar_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 5}, "/a/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer(None, "/a")


def test_resolve_mapping_key_numeric_string():
    assert resolve_pointer({"0": "zero"}, "/0") == "zero"


def test_resolve_tuple_is_sequence():
    assert resolve_pointer((1, 2, 3), "/1") == 2


def test_resolve_duck_typed_getitem():
    class Duck:
        def __getitem__(self, k):
            if k == "x":
                return 42
            raise KeyError(k)

    assert resolve_pointer(Duck(), "/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer(Duck(), "/y")


def test_verbose_exceptions_flag():
    old = jsonpointer.VERBOSE_EXCEPTIONS
    try:
        jsonpointer.VERBOSE_EXCEPTIONS = True
        with pytest.raises(JsonPointerException) as e:
            resolve_pointer({"secret": 1}, "/nope")
        assert "secret" in str(e.value)
        jsonpointer.VERBOSE_EXCEPTIONS = False
        with pytest.raises(JsonPointerException) as e:
            resolve_pointer({"secret": 1}, "/nope")
        assert "secret" not in str(e.value)
        assert "nope" in str(e.value)
    finally:
        jsonpointer.VERBOSE_EXCEPTIONS = old


def test_get_alias():
    p = JsonPointer("/a")
    assert p.get({"a": 3}) == 3
    assert p.get({}, "d") == "d"


def test_pointer_object_resolve():
    assert JsonPointer("/foo/1").resolve(RFC_DOC) == "baz"


# ---- to_last / walk / get_part ----

def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    assert JsonPointer("/a/b/1").to_last(doc) == ([1, 2], 1)
    assert JsonPointer("/a/b").to_last(doc) == (doc["a"], "b")
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_get_part():
    assert JsonPointer.get_part({}, "a") == "a"
    assert JsonPointer.get_part([], "3") == 3
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([], "01")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk([5, 6], "2")
    with pytest.raises(JsonPointerException):
        p.walk({}, "a")


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": 1}
    res = set_pointer(doc, "/a", 2)
    assert doc == {"a": 2}
    assert res is doc


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/b", 1)
    assert doc == {"a": {"b": 1}}


def test_set_list_index():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", 9)
    assert doc == {"a": [1, 9, 3]}


def test_set_list_append_with_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    assert doc == {"a": [1, 2]}


def test_set_list_out_of_range_raises():
    doc = [1]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/5", 1)
    assert doc == [1]


def test_set_missing_intermediate_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "", 5)


def test_set_root_not_inplace_returns_value():
    assert set_pointer({"a": 1}, "", 5, inplace=False) == 5


def test_set_not_inplace_copies():
    doc = {"a": {"b": [1, 2]}}
    orig = copy.deepcopy(doc)
    res = set_pointer(doc, "/a/b/0", 99, inplace=False)
    assert doc == orig
    assert res == {"a": {"b": [99, 2]}}
    assert res is not doc


def test_set_in_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "hello"}, "/a/0", "x")


def test_set_in_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 5)


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    set_pointer(doc, "/~01", 3)
    assert doc == {"a/b": 1, "m~n": 2, "~1": 3}


def test_set_invalid_index_on_list_raises():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 5)
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/x", 5)


def test_set_on_scalar_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


# ---- path / str / repr / eq / hash ----

@pytest.mark.parametrize("p", ["", "/", "//", "/a", "/a/b", "/a~1b", "/m~0n", "/~01", "/~10", "/0/1", "/ /%"])
def test_path_roundtrip(p):
    ptr = JsonPointer(p)
    assert ptr.path == p
    assert str(ptr) == p
    assert JsonPointer(ptr.path) == ptr


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"


def test_eq_and_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("/a") != "/a"
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert JsonPointer("/~01") != JsonPointer("/~1")


# ---- from_parts ----

def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]
    assert JsonPointer.from_parts(["a/b", "c~d"]).parts == ["a/b", "c~d"]


# ---- join / contains ----

def test_join_variants():
    base = JsonPointer("/a")
    assert base.join("/b/c").path == "/a/b/c"
    assert base.join(JsonPointer("/b")).path == "/a/b"
    assert base.join(["b", "c/d"]).path == "/a/b/c~1d"
    assert (base / "/b").path == "/a/b"
    assert (base / JsonPointer("/b")) == JsonPointer("/a/b")
    assert JsonPointer("").join("/x").path == "/x"
    assert base.join("").path == "/a"


def test_join_does_not_mutate():
    base = JsonPointer("/a")
    base.join("/b")
    assert base.path == "/a"


def test_join_escaped_parts():
    assert JsonPointer("/a").join("/~01").parts == ["a", "~1"]
    assert JsonPointer("/a").join(JsonPointer("/b~1c")).parts == ["a", "b/c"]


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_contains():
    # self contains ptr when ptr's parts are a prefix of self's parts
    a = JsonPointer("/a")
    ab = JsonPointer("/a/b")
    assert a in ab
    assert ab.contains(a)
    assert ab in ab
    assert ab not in a
    assert not a.contains(ab)
    assert JsonPointer("/b") not in ab
    assert JsonPointer("/a") not in JsonPointer("/ab")


def test_contains_root():
    assert JsonPointer("") in JsonPointer("/a")
    assert JsonPointer("/a") not in JsonPointer("")


# ---- pairwise ----

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]
