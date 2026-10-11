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
    assert JsonPointer(ptr).resolve(RFC_DOC) == expected
    assert JsonPointer(ptr).get(RFC_DOC) == expected


# ---- escape / unescape ----

def test_escape_unescape_basic():
    assert escape("a/b") == "a~1b"
    assert escape("a~b") == "a~0b"
    assert escape("~/") == "~0~1"
    assert escape("plain") == "plain"
    assert unescape("a~1b") == "a/b"
    assert unescape("a~0b") == "a~b"


def test_unescape_order():
    # "~01" must become "~1", not "/"
    assert unescape("~01") == "~1"
    assert JsonPointer("/~01").parts == ["~1"]
    assert escape("~1") == "~01"


@pytest.mark.parametrize("s", ["", "~", "/", "~0", "~1", "~01", "~10", "a~/b", "//", "~~//"])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


# ---- construction / validation ----

def test_root_pointer_has_no_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("").path == ""


def test_must_start_with_slash():
    for bad in ["foo", "foo/bar", "0"]:
        with pytest.raises(JsonPointerException):
            JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~x/b", "/a/b~", "/~ ", "/~-"])
def test_invalid_escapes(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_trailing_and_empty_segments():
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/").parts == ["a", ""]
    assert JsonPointer("/a//b").parts == ["a", "", "b"]


def test_get_parts():
    assert JsonPointer("/a/b~1c/0").get_parts() == ["a", "b/c", "0"]


def test_path_and_str_repr():
    p = JsonPointer("/a~1b/~0/0")
    assert p.path == "/a~1b/~0/0"
    assert str(p) == "/a~1b/~0/0"
    assert repr(p) == "JsonPointer('/a~1b/~0/0')"
    assert JsonPointer("").path == ""


# ---- resolve ----

def test_resolve_nested():
    doc = {"a": {"b": [10, {"c": "d"}]}}
    assert resolve_pointer(doc, "/a/b/1/c") == "d"
    assert resolve_pointer(doc, "/a/b/0") == 10
    assert resolve_pointer(doc, "/a/b") == [10, {"c": "d"}]


def test_resolve_last_index():
    assert resolve_pointer([1, 2, 3], "/2") == 3
    assert resolve_pointer([1, 2, 3], "/0") == 1


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_missing_with_default():
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/b", 5) == 5
    assert resolve_pointer({"a": 1}, "/a/b/c", "d") == "d"
    assert resolve_pointer([1], "/5", "d") == "d"
    assert resolve_pointer([1], "/x", "d") == "d"


def test_default_not_used_when_found():
    assert resolve_pointer({"a": 0}, "/a", "d") == 0
    assert resolve_pointer({"a": None}, "/a", "d") is None


def test_default_falsy_values():
    assert resolve_pointer({}, "/a", 0) == 0
    assert resolve_pointer({}, "/a", "") == ""
    assert resolve_pointer({}, "/a", False) is False


def test_resolve_root_returns_same_object():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc
    assert resolve_pointer(doc, "", "d") is doc


def test_resolve_empty_key():
    assert resolve_pointer({"": {"": 1}}, "//") == 1


def test_resolve_array_index_out_of_range():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")


@pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", "1.0", " 1", "1 ", "a", "", "1e1", "0x1"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2], "/" + idx)


def test_index_with_trailing_garbage_rejected():
    # regex alternation must not let "0..." through
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/0a")
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/0 ")
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/00")


def test_multi_digit_index():
    lst = list(range(20))
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    res = resolve_pointer(lst, "/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst
    assert repr(res) == "EndOfList([1, 2])"


def test_dash_in_mapping_is_plain_key():
    assert resolve_pointer({"-": 1}, "/-") == 1


def test_numeric_key_in_dict_is_string():
    assert resolve_pointer({"0": "x"}, "/0") == "x"
    with pytest.raises(JsonPointerException):
        resolve_pointer({0: "x"}, "/0")


def test_tuple_is_sequence():
    assert resolve_pointer((1, 2, 3), "/1") == 2


def test_cannot_index_string():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer("hello", "/0")


def test_cannot_index_scalar():
    for scalar in (1, 1.5, None, True):
        with pytest.raises(JsonPointerException):
            resolve_pointer({"a": scalar}, "/a/b")


def test_walk_into_scalar_default():
    assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"


def test_duck_typed_getitem():
    class Custom:
        def __getitem__(self, key):
            if key == "k":
                return "v"
            raise KeyError(key)

    assert resolve_pointer(Custom(), "/k") == "v"
    with pytest.raises(JsonPointerException):
        resolve_pointer(Custom(), "/missing")


def test_verbose_exceptions_flag(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"secret": 1}, "/nope")
    assert "secret" in str(ei.value)
    assert "nope" in str(ei.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"secret": 1}, "/nope")
    assert "secret" not in str(ei.value)
    assert "nope" in str(ei.value)


def test_escaped_keys_resolve():
    doc = {"a/b": {"c~d": 1}}
    assert resolve_pointer(doc, "/a~1b/c~0d") == 1


def test_percent_not_decoded():
    doc = {"a%20b": 1, "a b": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/a b") == 2


# ---- to_last / walk / get_part ----

def test_to_last():
    doc = {"a": {"b": 1}}
    parent, last = JsonPointer("/a/b").to_last(doc)
    assert parent is doc["a"]
    assert last == "b"


def test_to_last_root():
    doc = {"a": 1}
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_to_last_list_index_converted():
    doc = {"a": [1, 2, 3]}
    parent, last = JsonPointer("/a/1").to_last(doc)
    assert parent is doc["a"]
    assert last == 1 and isinstance(last, int)


def test_to_last_dash():
    parent, last = JsonPointer("/-").to_last([1])
    assert last == "-"


def test_get_part():
    assert JsonPointer.get_part({}, "a") == "a"
    assert JsonPointer.get_part({}, "1") == "1"
    assert JsonPointer.get_part([], "1") == 1
    assert JsonPointer.get_part([], "0") == 0
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([], "01")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk({"a": 1}, "b")
    with pytest.raises(JsonPointerException):
        p.walk([5], "3")


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": 1}
    res = set_pointer(doc, "/a", 2)
    assert res is doc
    assert doc == {"a": 2}


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/b", 1)
    assert doc == {"a": {"b": 1}}


def test_set_list_element():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", 20)
    assert doc == {"a": [1, 20, 3]}


def test_set_list_append_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    set_pointer(doc, "/a/-", 3)
    assert doc == {"a": [1, 2, 3]}


def test_set_dash_on_dict_is_key():
    doc = {}
    set_pointer(doc, "/-", 1)
    assert doc == {"-": 1}


def test_set_list_out_of_range():
    with pytest.raises(JsonPointerException):
        set_pointer([1], "/5", 1)
    with pytest.raises(JsonPointerException):
        set_pointer([], "/0", 1)


def test_set_list_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 1)
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/-1", 1)


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
    assert res == {"a": {"b": [99, 2]}}
    assert doc == orig
    assert res is not doc
    assert res["a"] is not doc["a"]


def test_set_not_inplace_append():
    doc = [1]
    res = set_pointer(doc, "/-", 2, inplace=False)
    assert res == [1, 2]
    assert doc == [1]


def test_set_into_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_into_missing_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_into_scalar_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "/a/b", 1)


def test_set_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 1)


def test_set_escaped_key():
    doc = {"a/b": {}}
    set_pointer(doc, "/a~1b/~0", 1)
    assert doc == {"a/b": {"~": 1}}


def test_set_returns_doc_and_pointer_object():
    doc = {"a": 1}
    assert JsonPointer("/a").set(doc, 3) is doc
    assert doc["a"] == 3


# ---- contains / join / eq / hash ----

def test_contains():
    p = JsonPointer("/a/b")
    assert JsonPointer("/a/b/c") in JsonPointer("/a/b/c")
    assert JsonPointer("/a") in p
    assert JsonPointer("/a/b") in p
    assert JsonPointer("") in p
    assert JsonPointer("/a/b/c") not in p
    assert JsonPointer("/b") not in p
    assert JsonPointer("/a/c") not in p
    assert p.contains(JsonPointer("/a"))
    assert not p.contains(JsonPointer("/x"))


def test_contains_is_by_parts_not_string_prefix():
    assert JsonPointer("/ab") not in JsonPointer("/a")
    assert JsonPointer("/a") not in JsonPointer("/ab")
    assert JsonPointer("/ab") not in JsonPointer("/abc")


def test_join_str_pointer_parts():
    p = JsonPointer("/a")
    assert p.join("/b/c").path == "/a/b/c"
    assert p.join(JsonPointer("/b")).path == "/a/b"
    assert p.join(["b", "c/d"]).path == "/a/b/c~1d"
    assert p.join([1, 2]).path == "/a/1/2"
    assert p.path == "/a"  # unchanged


def test_join_root():
    assert JsonPointer("").join("/a").path == "/a"
    assert JsonPointer("/a").join("").path == "/a"
    assert JsonPointer("/a").join(JsonPointer("")).path == "/a"


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")  # no leading slash
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_truediv():
    assert (JsonPointer("/a") / "/b").path == "/a/b"
    assert (JsonPointer("/a") / JsonPointer("/b")).path == "/a/b"
    assert (JsonPointer("/a") / ["x"]).path == "/a/x"


def test_join_escapes_special_chars():
    j = JsonPointer("/a").join(["~", "/"])
    assert j.path == "/a/~0/~1"
    assert j.parts == ["a", "~", "/"]


def test_eq_and_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/a/b") != "/a/b"
    assert JsonPointer("/a") != None  # noqa: E711
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert JsonPointer("/a~1b") == JsonPointer("/a~1b")
    assert JsonPointer("/a/b") != JsonPointer("/a~1b")


# ---- from_parts ----

def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["a", "b"]).parts == ["a", "b"]
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]


def test_from_parts_roundtrip():
    parts = ["a", "", "b/c", "~", "~1", "0"]
    assert JsonPointer.from_parts(parts).parts == parts
    p = JsonPointer("/x~0~1y/0//z")
    assert JsonPointer.from_parts(p.parts) == p
    assert JsonPointer.from_parts(p.parts).path == p.path


def test_from_parts_subclass():
    class Sub(JsonPointer):
        pass

    assert isinstance(Sub.from_parts(["a"]), Sub)
    assert isinstance(Sub("/a").join("/b"), Sub)


# ---- pairwise ----

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---- module-level helpers ----

def test_resolve_pointer_invalid_pointer_raises_even_with_default():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "a", "default")


def test_exception_is_exception():
    assert issubclass(JsonPointerException, Exception)
