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


def test_escape_unescape():
    assert escape("a/b~c") == "a~1b~0c"
    assert unescape("a~1b~0c") == "a/b~c"
    assert unescape("~01") == "~1"
    assert escape("~1") == "~01"
    assert unescape(escape("~/~1/~0")) == "~/~1/~0"


def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2, 3])) == [(1, 2), (2, 3)]


def test_must_start_with_slash():
    for bad in ["foo", "a/b", " /a", "#/a"]:
        with pytest.raises(JsonPointerException):
            JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~b/c", "/~ "])
def test_invalid_escapes(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_parts_parsing():
    assert JsonPointer("").parts == []
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]
    assert JsonPointer("/a~1b/~0").parts == ["a/b", "~"]


def test_path_roundtrip():
    for p in ["", "/", "//", "/a/b", "/~0/0/~1", "/a~01", "/~01~10"]:
        assert JsonPointer(p).path == p
        assert str(JsonPointer(p)) == p


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"


def test_resolve_nested():
    doc = {"a": [{"b": [10, 20, {"c": "x"}]}]}
    assert resolve_pointer(doc, "/a/0/b/2/c") == "x"
    assert resolve_pointer(doc, "/a/0/b/1") == 20
    assert resolve_pointer(doc, "/a/0/b/0") == 10


def test_resolve_missing_key():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_default():
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/b", "d") == "d"
    assert resolve_pointer({"a": [1]}, "/a/5", 9) == 9
    assert resolve_pointer({"a": [1]}, "/a/x", 9) == 9
    assert resolve_pointer({"a": 1}, "/a", 9) == 1
    # falsy values are still found
    assert resolve_pointer({"a": 0}, "/a", 9) == 0


def test_resolve_does_not_unquote_percent():
    doc = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/c d") == 2
    assert resolve_pointer(doc, "/c%20d", None) is None


def test_index_out_of_range():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    assert resolve_pointer([1, 2], "/1") == 2


def test_negative_index_rejected():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/-1")


@pytest.mark.parametrize("idx", ["00", "01", "007", "000", "+1", " 1", "1 ",
                                 "1.0", "0x1", "1e0", "a", "", "١"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "/" + idx)
    assert resolve_pointer([0, 1, 2], "/" + idx, "dflt") == "dflt"


def test_leading_zero_index_on_set():
    lst = [1, 2, 3]
    with pytest.raises(JsonPointerException):
        set_pointer(lst, "/01", 9)
    with pytest.raises(JsonPointerException):
        set_pointer(lst, "/00", 9)
    assert lst == [1, 2, 3]


def test_get_part_leading_zero():
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([1, 2], "01")
    assert JsonPointer.get_part([1, 2], "0") == 0
    assert JsonPointer.get_part([1, 2], "10") == 10


def test_valid_multi_digit_index():
    lst = list(range(20))
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19
    assert resolve_pointer(lst, "/0") == 0


def test_numeric_key_in_dict_is_string():
    assert resolve_pointer({"0": "a", "01": "b"}, "/0") == "a"
    assert resolve_pointer({"0": "a", "01": "b"}, "/01") == "b"


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    res = resolve_pointer(lst, "/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst
    assert "EndOfList" in repr(res)


def test_dash_on_dict_is_key():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_non_container():
    for doc in [5, None, 1.5, True]:
        with pytest.raises(JsonPointerException):
            resolve_pointer(doc, "/a")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "str"}, "/a/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer("abc", "/0")


def test_tuple_sequence():
    assert resolve_pointer((1, 2, 3), "/1") == 2


def test_duck_typed_getitem():
    class Thing:
        def __getitem__(self, k):
            if k == "x":
                return 42
            raise KeyError(k)

    assert resolve_pointer(Thing(), "/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer(Thing(), "/y")


def test_verbose_exceptions(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"secret": 1}, "/zzz")
    assert "secret" in str(e.value)
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"secret": 1}, "/zzz")
    assert "secret" not in str(e.value)
    assert "zzz" in str(e.value)


def test_get_alias():
    p = JsonPointer("/a")
    assert p.get({"a": 3}) == 3
    assert p.get({}, "d") == "d"


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": {"b": 1}}
    res = set_pointer(doc, "/a/b", 2)
    assert res is doc
    assert doc == {"a": {"b": 2}}


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/b", 1)
    assert doc == {"a": {"b": 1}}


def test_set_list_element():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", 20)
    assert doc == {"a": [1, 20, 3]}


def test_set_append_with_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    assert doc == {"a": [1, 2]}


def test_set_dash_on_dict_is_key():
    doc = {}
    set_pointer(doc, "/-", 1)
    assert doc == {"-": 1}


def test_set_list_out_of_range():
    with pytest.raises(JsonPointerException):
        set_pointer([1], "/5", 0)


def test_set_missing_intermediate():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_root_inplace_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "", 5)


def test_set_root_not_inplace():
    assert set_pointer({"a": 1}, "", 5, inplace=False) == 5


def test_set_not_inplace_copies():
    doc = {"a": [1, {"b": 2}]}
    res = set_pointer(doc, "/a/1/b", 3, inplace=False)
    assert res == {"a": [1, {"b": 3}]}
    assert doc == {"a": [1, {"b": 2}]}
    assert res is not doc
    assert res["a"] is not doc["a"]


def test_set_not_inplace_append():
    doc = [1]
    res = set_pointer(doc, "/-", 2, inplace=False)
    assert res == [1, 2]
    assert doc == [1]


def test_set_in_string_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_in_tuple_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 9)


def test_set_escaped_key():
    doc = {"a/b": {}}
    set_pointer(doc, "/a~1b/~0", 1)
    assert doc == {"a/b": {"~": 1}}


def test_to_last():
    doc = {"a": [1, 2]}
    assert JsonPointer("").to_last(doc) == (doc, None)
    parent, last = JsonPointer("/a/1").to_last(doc)
    assert parent is doc["a"] and last == 1
    parent, last = JsonPointer("/a").to_last(doc)
    assert parent is doc and last == "a"
    parent, last = JsonPointer("/a/-").to_last(doc)
    assert last == "-"


# ---- contains / join / eq / hash ----

def test_contains():
    p = JsonPointer("/a/b")
    assert p.contains(JsonPointer("/a"))
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer(""))
    assert not p.contains(JsonPointer("/a/b/c"))
    assert not p.contains(JsonPointer("/b"))
    assert not p.contains(JsonPointer("/a/c"))
    assert JsonPointer("/a") in JsonPointer("/a/b")
    assert JsonPointer("/a/b") not in JsonPointer("/a")


def test_join():
    p = JsonPointer("/a")
    assert p.join("/b/c").path == "/a/b/c"
    assert p.join(JsonPointer("/b")).path == "/a/b"
    assert p.join(["b", "c"]).path == "/a/b/c"
    assert p.join(["x/y", "~"]).path == "/a/x~1y/~0"
    assert p.join([1, 2]).path == "/a/1/2"
    assert JsonPointer("").join("/a").path == "/a"
    assert p.join("").path == "/a"
    assert p.path == "/a"  # unchanged


def test_join_invalid():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_truediv():
    assert (JsonPointer("/a") / "/b").path == "/a/b"
    assert (JsonPointer("/a") / ["b"]).path == "/a/b"


def test_join_subclass():
    class Sub(JsonPointer):
        pass
    assert isinstance(Sub("/a").join("/b"), Sub)


def test_eq_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("/a") != "/a"
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/") != JsonPointer("")
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2


def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["a", "b"]).parts == ["a", "b"]
    assert JsonPointer.from_parts(["~01"]).parts == ["~01"]


def test_root_resolve_returns_same_object():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc
