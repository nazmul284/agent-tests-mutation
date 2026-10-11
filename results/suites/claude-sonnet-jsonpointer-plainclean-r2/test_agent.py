import pytest

import jsonpointer
from jsonpointer import (
    EndOfList, JsonPointer, JsonPointerException, escape, pairwise,
    resolve_pointer, set_pointer, unescape,
)

# RFC 6901 section 5 example document
RFC = {
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


@pytest.mark.parametrize("ptr,expected", [
    ("", RFC),
    ("/foo", ["bar", "baz"]),
    ("/foo/0", "bar"),
    ("/foo/1", "baz"),
    ("/", 0),
    ("/a~1b", 1),
    ("/c%d", 2),
    ("/e^f", 3),
    ("/g|h", 4),
    ("/i\\j", 5),
    ('/k"l', 6),
    ("/ ", 7),
    ("/m~0n", 8),
])
def test_rfc_examples(ptr, expected):
    assert resolve_pointer(RFC, ptr) == expected


def test_escape_unescape():
    assert escape("a/b~c") == "a~1b~0c"
    assert unescape("a~1b~0c") == "a/b~c"
    # order matters: ~01 must decode to ~1, not /
    assert unescape("~01") == "~1"
    assert escape("~1") == "~01"
    assert escape("") == ""


def test_parts_parsing():
    assert JsonPointer("").parts == []
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/~10").parts == ["/0"]


@pytest.mark.parametrize("bad", ["a", "a/b", "foo", " /a"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/~2", "/a~", "/~a", "/a~/b", "/a/~x", "~"])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_resolve_errors_and_default():
    doc = {"a": [1, 2], "s": "str", "n": 5}
    for p in ["/b", "/a/2", "/a/x", "/a/01", "/a/-1", "/a/+1", "/a/1.0",
              "/a/ 1", "/a/1 ", "/s/0", "/n/0", "/a/0/x"]:
        with pytest.raises(JsonPointerException):
            resolve_pointer(doc, p)
        assert resolve_pointer(doc, p, "D") == "D"
    assert resolve_pointer(doc, "/b", None) is None


def test_default_not_used_when_found():
    assert resolve_pointer({"a": 0}, "/a", "D") == 0
    assert resolve_pointer({"a": None}, "/a", "D") is None


def test_array_index_validation():
    lst = list(range(20))
    assert resolve_pointer(lst, "/0") == 0
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19
    for bad in ["/00", "/01", "/-0", "/1e0", "/0x1", "/١"]:
        with pytest.raises(JsonPointerException):
            resolve_pointer(lst, bad)
    # trailing garbage must be rejected even after valid digits
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/1a")
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/10\n")
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/")


def test_tuple_sequence():
    assert resolve_pointer((1, 2, 3), "/2") == 3
    with pytest.raises(JsonPointerException):
        resolve_pointer((1, 2, 3), "/3")


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    r = resolve_pointer(lst, "/-")
    assert isinstance(r, EndOfList)
    assert r.list_ is lst
    assert "EndOfList([1, 2])" == repr(r)
    # dash on a mapping is just a key
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_string_is_not_container():
    with pytest.raises(JsonPointerException):
        resolve_pointer("abc", "/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "abc"}, "/a/0")


def test_mapping_keys_are_strings():
    assert resolve_pointer({"1": "x"}, "/1") == "x"
    with pytest.raises(JsonPointerException):
        resolve_pointer({1: "x"}, "/1")


def test_duck_typed_getitem():
    class Custom:
        def __getitem__(self, k):
            if k == "k":
                return "v"
            raise KeyError(k)
    assert resolve_pointer(Custom(), "/k") == "v"
    with pytest.raises(JsonPointerException):
        resolve_pointer(Custom(), "/zz")


def test_non_indexable():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/a/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer(None, "/a")
    assert resolve_pointer(None, "/a", "D") == "D"


def test_empty_pointer_returns_doc():
    d = {"a": 1}
    assert resolve_pointer(d, "") is d
    assert JsonPointer("").resolve(5) == 5
    assert JsonPointer("").get(5) == 5


def test_verbose_exceptions(monkeypatch):
    doc = {"secret": "SENSITIVE"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer(doc, "/missing")
    assert "SENSITIVE" in str(e.value)
    assert "missing" in str(e.value)
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer(doc, "/missing")
    assert "SENSITIVE" not in str(e.value)
    assert "missing" in str(e.value)


def test_set_dict_inplace():
    d = {"a": {"b": 1}}
    res = set_pointer(d, "/a/b", 2)
    assert res is d
    assert d == {"a": {"b": 2}}
    set_pointer(d, "/a/c", 3)
    assert d == {"a": {"b": 2, "c": 3}}
    set_pointer(d, "/x~1y", 9)
    assert d["x/y"] == 9


def test_set_list():
    d = {"a": [1, 2, 3]}
    set_pointer(d, "/a/0", 10)
    set_pointer(d, "/a/2", 30)
    assert d["a"] == [10, 2, 30]
    set_pointer(d, "/a/-", 4)
    assert d["a"] == [10, 2, 30, 4]


def test_set_list_errors():
    d = {"a": [1, 2, 3]}
    for p in ["/a/3", "/a/01", "/a/x", "/a/-1"]:
        with pytest.raises(JsonPointerException):
            set_pointer(d, p, 0)
    assert d == {"a": [1, 2, 3]}


def test_set_dash_on_dict_is_key():
    d = {}
    set_pointer(d, "/-", 1)
    assert d == {"-": 1}


def test_set_missing_intermediate():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)
    with pytest.raises(JsonPointerException):
        set_pointer({"a": [1]}, "/a/5/b", 1)


def test_set_in_string_and_immutable():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 9)
    with pytest.raises(JsonPointerException):
        set_pointer(5, "/a", 9)


def test_set_root():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "", 1)
    assert set_pointer({"a": 1}, "", 7, inplace=False) == 7


def test_set_not_inplace():
    d = {"a": [1, {"b": 2}]}
    res = set_pointer(d, "/a/1/b", 99, inplace=False)
    assert res is not d
    assert res == {"a": [1, {"b": 99}]}
    assert d == {"a": [1, {"b": 2}]}
    res = set_pointer(d, "/a/-", 5, inplace=False)
    assert res["a"] == [1, {"b": 2}, 5]
    assert d["a"] == [1, {"b": 2}]


def test_set_not_inplace_error_leaves_doc():
    d = {"a": [1]}
    with pytest.raises(JsonPointerException):
        set_pointer(d, "/a/9", 1, inplace=False)
    assert d == {"a": [1]}


def test_to_last():
    d = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(d) == (d, None)
    sub, last = JsonPointer("/a/b/1").to_last(d)
    assert sub is d["a"]["b"] and last == 1
    sub, last = JsonPointer("/a/b/-").to_last(d)
    assert sub is d["a"]["b"] and last == "-"
    sub, last = JsonPointer("/a/new").to_last(d)
    assert sub is d["a"] and last == "new"


def test_get_part():
    assert JsonPointer.get_part({}, "5") == "5"
    assert JsonPointer.get_part([], "5") == 5
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("s", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(3, "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([], "")


def test_walk():
    assert JsonPointer("").walk({"a": 1}, "a") == 1
    assert JsonPointer("").walk([5, 6], "1") == 6
    assert isinstance(JsonPointer("").walk([5], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        JsonPointer("").walk([5], "1")
    with pytest.raises(JsonPointerException):
        JsonPointer("").walk({}, "a")


def test_path_roundtrip():
    for p in ["", "/", "//", "/a/b", "/a~1b/c~0d", "/~01", "/~10", "/0/-"]:
        assert JsonPointer(p).path == p
        assert str(JsonPointer(p)) == p


def test_repr():
    assert repr(JsonPointer("/a~1b")) == "JsonPointer('/a~1b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_eq_and_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("/a") != "/a"
    assert JsonPointer("/a") != None  # noqa: E711
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert JsonPointer("/a/b") != JsonPointer("/a~1b")


def test_contains():
    base = JsonPointer("/a/b")
    assert JsonPointer("/a/b/c") in JsonPointer("/a/b/c")
    assert base.contains(JsonPointer("/a"))
    assert JsonPointer("/a") in base
    assert JsonPointer("") in base
    assert base in base
    assert base not in JsonPointer("/a")
    assert JsonPointer("/b") not in base
    assert JsonPointer("/a/c") not in base
    assert JsonPointer("/a/b/c") not in base
    assert JsonPointer("/") not in JsonPointer("/a")


def test_join():
    p = JsonPointer("/a")
    assert p.join("/b/c") == JsonPointer("/a/b/c")
    assert p.join(JsonPointer("/b")) == JsonPointer("/a/b")
    assert p.join(["b", "c/d"]).path == "/a/b/c~1d"
    assert p.join([1, 2]).path == "/a/1/2"
    assert (p / "/x").path == "/a/x"
    assert (p / JsonPointer("/x")).path == "/a/x"
    assert JsonPointer("").join("/x").path == "/x"
    assert p.join("").path == "/a"
    assert p.path == "/a"  # unchanged
    with pytest.raises(JsonPointerException):
        p.join("noslash")
    with pytest.raises(JsonPointerException):
        p.join(5)


def test_from_parts():
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]
    assert JsonPointer.from_parts(iter(["a", "b"])).parts == ["a", "b"]


def test_from_parts_roundtrip():
    parts = ["a/b", "~", "~0", "~1", "", "x y", "0"]
    assert JsonPointer.from_parts(parts).parts == parts


def test_subclass_join_preserves_class():
    class Sub(JsonPointer):
        pass
    assert isinstance(Sub("/a").join("/b"), Sub)
    assert isinstance(Sub("/a").join(["b"]), Sub)


def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


def test_pointer_object_reusable():
    p = JsonPointer("/a/0")
    assert p.resolve({"a": [1]}) == 1
    assert p.resolve({"a": [2]}) == 2
    assert p.resolve({}, "D") == "D"
    d = {"a": [0]}
    assert p.set(d, 5) is d and d == {"a": [5]}


def test_resolve_pointer_accepts_default_none_and_falsey():
    assert resolve_pointer({}, "/x", 0) == 0
    assert resolve_pointer({}, "/x", "") == ""
    assert resolve_pointer({}, "/x", False) is False


def test_nested_deep():
    d = {"a": [{"b": [10, {"c": "deep"}]}]}
    assert resolve_pointer(d, "/a/0/b/1/c") == "deep"
    set_pointer(d, "/a/0/b/1/c", "new")
    assert d["a"][0]["b"][1]["c"] == "new"
