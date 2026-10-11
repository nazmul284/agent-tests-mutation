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
    ("/k\"l", 6),
    ("/ ", 7),
    ("/m~0n", 8),
]


@pytest.mark.parametrize("ptr,expected", RFC_CASES)
def test_rfc_examples(ptr, expected):
    assert resolve_pointer(RFC_DOC, ptr) == expected


# ---- escape / unescape ----

def test_escape():
    assert escape("a/b") == "a~1b"
    assert escape("m~n") == "m~0n"
    assert escape("~/") == "~0~1"
    assert escape("plain") == "plain"


def test_unescape():
    assert unescape("a~1b") == "a/b"
    assert unescape("m~0n") == "m~n"
    assert unescape("~01") == "~1"
    assert unescape("~00") == "~0"
    assert unescape("~0~1") == "~/"


@pytest.mark.parametrize("s", ["", "a", "~", "/", "~0", "~1", "~/~", "~01", "a~b/c//"])
def test_escape_unescape_roundtrip(s):
    assert unescape(escape(s)) == s


# ---- construction ----

def test_root_pointer_has_no_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("").path == ""


def test_parts_parsed_and_unescaped():
    assert JsonPointer("/a/b/c").parts == ["a", "b", "c"]
    assert JsonPointer("/a~1b/m~0n").parts == ["a/b", "m~n"]
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]


@pytest.mark.parametrize("bad", ["a", "a/b", "foo", " /a", "0"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/a~b", "/~a/b", "/a/~", "/~ ", "~"])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_invalid_escape_message():
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer("/a~2")


# ---- resolve ----

def test_resolve_root_returns_doc_itself():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc
    assert resolve_pointer([1], "") == [1]
    assert resolve_pointer(5, "") == 5


def test_resolve_nested():
    doc = {"a": {"b": [10, {"c": "x"}]}}
    assert resolve_pointer(doc, "/a/b/1/c") == "x"
    assert resolve_pointer(doc, "/a/b/0") == 10
    assert resolve_pointer(doc, "/a/b") is doc["a"]["b"]


def test_resolve_list_indices():
    lst = list(range(20))
    assert resolve_pointer(lst, "/0") == 0
    assert resolve_pointer(lst, "/9") == 9
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_resolve_tuple():
    assert resolve_pointer({"a": (1, 2, 3)}, "/a/2") == 3


def test_resolve_falsy_values():
    doc = {"a": None, "b": 0, "c": "", "d": [], "e": False}
    assert resolve_pointer(doc, "/a") is None
    assert resolve_pointer(doc, "/b") == 0
    assert resolve_pointer(doc, "/c") == ""
    assert resolve_pointer(doc, "/d") == []
    assert resolve_pointer(doc, "/e") is False
    # falsy values must not be replaced by the default
    assert resolve_pointer(doc, "/b", "dflt") == 0
    assert resolve_pointer(doc, "/a", "dflt") is None


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": {"b": 1}}, "/a/c")


def test_resolve_missing_key_default():
    assert resolve_pointer({"a": 1}, "/b", "d") == "d"
    assert resolve_pointer({"a": 1}, "/b", None) is None
    assert resolve_pointer({"a": 1}, "/a/b/c", 7) == 7


def test_resolve_default_on_root_returns_doc():
    assert resolve_pointer({"a": 1}, "", "d") == {"a": 1}


def test_resolve_index_out_of_range():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")
    assert resolve_pointer([1, 2], "/2", "d") == "d"
    assert resolve_pointer([1, 2], "/1", "d") == 2


@pytest.mark.parametrize("tok", ["-1", "+1", "01", "00", "1.0", " 1", "1 ", "a", "", "1e0", "0x1", "１"])
def test_resolve_invalid_list_index(tok):
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/" + tok)
    assert resolve_pointer([1, 2, 3], "/" + tok, "d") == "d"


def test_resolve_negative_index_does_not_wrap():
    with pytest.raises(JsonPointerException):
        resolve_pointer(["a", "b"], "/-1")


def test_resolve_trailing_garbage_in_index():
    # "1x" and "10a" must not be accepted as indices
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/1x")
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/10a")
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/0a")


def test_resolve_dash_gives_end_of_list():
    lst = [1, 2]
    res = resolve_pointer(lst, "/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst
    assert "EndOfList" in repr(res)
    assert "[1, 2]" in repr(res)


def test_resolve_dash_on_dict_is_plain_key():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_resolve_numeric_key_on_dict_is_string():
    assert resolve_pointer({"0": "x"}, "/0") == "x"
    with pytest.raises(JsonPointerException):
        resolve_pointer({0: "x"}, "/0")


def test_resolve_empty_key():
    assert resolve_pointer({"": {"": 1}}, "/") == {"": 1}
    assert resolve_pointer({"": {"": 1}}, "//") == 1


def test_resolve_into_string_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer("hello", "/0")
    assert resolve_pointer({"a": "hello"}, "/a/0", "d") == "d"


def test_resolve_into_scalar_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 5}, "/a/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": None}, "/a/b")
    assert resolve_pointer({"a": 5}, "/a/b", "d") == "d"


def test_resolve_no_percent_decoding():
    doc = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/c d") == 2
    assert resolve_pointer(doc, "/c%20d", None) is None


def test_resolve_tilde_keys():
    doc = {"~1": "a", "/": "b", "~": "c", "~0": "d"}
    assert resolve_pointer(doc, "/~01") == "a"
    assert resolve_pointer(doc, "/~1") == "b"
    assert resolve_pointer(doc, "/~0") == "c"
    assert resolve_pointer(doc, "/~00") == "d"


def test_get_is_alias_of_resolve():
    p = JsonPointer("/a")
    assert p.get({"a": 3}) == 3
    assert p.get({}, "d") == "d"


def test_resolve_custom_getitem():
    class Custom:
        def __getitem__(self, k):
            if k == "x":
                return 42
            raise KeyError(k)

    assert resolve_pointer(Custom(), "/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer(Custom(), "/y")


def test_resolve_object_without_getitem():
    with pytest.raises(JsonPointerException):
        resolve_pointer(object(), "/a")
    assert resolve_pointer({"a": object()}, "/a/b", "d") == "d"


def test_resolve_mapping_subclass():
    from collections import OrderedDict
    assert resolve_pointer(OrderedDict(a=OrderedDict(b=2)), "/a/b") == 2


def test_verbose_exceptions(monkeypatch):
    doc = {"secret": "SENSITIVEVALUE"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(doc, "/missing")
    assert "SENSITIVEVALUE" in str(ei.value)
    assert "missing" in str(ei.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(doc, "/missing")
    assert "SENSITIVEVALUE" not in str(ei.value)
    assert "missing" in str(ei.value)


# ---- walk / get_part / to_last ----

def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk({"a": 1}, "b")
    with pytest.raises(JsonPointerException):
        p.walk([5, 6], "2")


def test_get_part():
    assert JsonPointer.get_part({"a": 1}, "a") == "a"
    assert JsonPointer.get_part({"a": 1}, "0") == "0"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "-") == "-"
    assert JsonPointer.get_part([1] * 20, "15") == 15
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([1], "01")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")


def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(doc) == (doc, None)
    sub, last = JsonPointer("/a/b/1").to_last(doc)
    assert sub is doc["a"]["b"]
    assert last == 1
    sub, last = JsonPointer("/a/b").to_last(doc)
    assert sub is doc["a"]
    assert last == "b"
    sub, last = JsonPointer("/a/b/-").to_last(doc)
    assert sub is doc["a"]["b"]
    assert last == "-"
    sub, last = JsonPointer("/x").to_last(doc)
    assert sub is doc and last == "x"


def test_to_last_missing_intermediate():
    with pytest.raises(JsonPointerException):
        JsonPointer("/x/y").to_last({"a": 1})


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": {"b": 1}}
    res = set_pointer(doc, "/a/b", 2)
    assert res is doc
    assert doc == {"a": {"b": 2}}


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/new", 1)
    assert doc == {"a": {"new": 1}}


def test_set_list_element():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", "x")
    assert doc == {"a": [1, "x", 3]}
    set_pointer(doc, "/a/0", "y")
    set_pointer(doc, "/a/2", "z")
    assert doc == {"a": ["y", "x", "z"]}


def test_set_list_append_with_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    set_pointer(doc, "/a/-", 3)
    assert doc == {"a": [1, 2, 3]}


def test_set_dash_toplevel_list():
    doc = []
    set_pointer(doc, "/-", "x")
    assert doc == ["x"]


def test_set_list_out_of_range():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/5", "x")
    with pytest.raises(JsonPointerException):
        set_pointer([], "/0", "x")


def test_set_list_index_equal_to_length_fails():
    doc = [1, 2]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/2", "x")
    assert doc == [1, 2]


@pytest.mark.parametrize("tok", ["01", "-1", "a", "1.5", ""])
def test_set_list_invalid_index(tok):
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2, 3], "/" + tok, "x")


def test_set_missing_intermediate_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


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
    assert res["a"]["b"] is not doc["a"]["b"]


def test_set_not_inplace_append():
    doc = {"a": [1]}
    res = set_pointer(doc, "/a/-", 2, inplace=False)
    assert res == {"a": [1, 2]}
    assert doc == {"a": [1]}


def test_set_into_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "hello"}, "/a/0", "x")
    with pytest.raises(JsonPointerException):
        set_pointer("hello", "/0", "x")
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "hello"}, "/a/-", "x")


def test_set_into_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", "x")


def test_set_into_scalar_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    assert doc == {"a/b": 1, "m~n": 2}


def test_set_no_percent_decoding():
    doc = {"foo": {}}
    set_pointer(doc, "/foo/a%20b", "x")
    assert doc == {"foo": {"a%20b": "x"}}


def test_set_empty_key():
    doc = {}
    set_pointer(doc, "/", 1)
    assert doc == {"": 1}


def test_set_value_none_and_falsy():
    doc = {"a": 1}
    set_pointer(doc, "/a", None)
    assert doc == {"a": None}
    set_pointer(doc, "/b", 0)
    assert doc == {"a": None, "b": 0}


def test_set_then_resolve():
    doc = {"a": [{}]}
    set_pointer(doc, "/a/0/k", [1, 2])
    assert resolve_pointer(doc, "/a/0/k/1") == 2


def test_set_mutates_only_target():
    doc = {"a": [1, 2, 3], "b": {"c": 1}}
    set_pointer(doc, "/a/1", 9)
    assert doc == {"a": [1, 9, 3], "b": {"c": 1}}


# ---- contains / join / eq / hash / str ----

def test_contains():
    p = JsonPointer("/a/b")
    # self contains ptr if ptr is a prefix of self
    assert p.contains(JsonPointer("/a"))
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer(""))
    assert not p.contains(JsonPointer("/a/b/c"))
    assert not p.contains(JsonPointer("/b"))
    assert not p.contains(JsonPointer("/a/c"))
    assert JsonPointer("/a") in JsonPointer("/a/b")
    assert JsonPointer("/a/b") not in JsonPointer("/a")


def test_contains_whole_parts_only():
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))
    assert not JsonPointer("/a/b").contains(JsonPointer("/a/b/"))


def test_join_pointer_string_and_parts():
    p = JsonPointer("/a")
    assert p.join(JsonPointer("/b/c")).path == "/a/b/c"
    assert p.join("/b/c").path == "/a/b/c"
    assert p.join(["b", "c"]).path == "/a/b/c"
    assert p.join(["b", 0]).path == "/a/b/0"
    assert p.join("").path == "/a"
    assert JsonPointer("").join("/x").path == "/x"
    assert (p / "/b").path == "/a/b"
    assert (p / JsonPointer("/b")).parts == ["a", "b"]
    assert p.path == "/a"  # original unchanged


def test_join_escapes_parts():
    p = JsonPointer("/a")
    assert p.join(["b/c", "d~e"]).path == "/a/b~1c/d~0e"
    assert p.join(["b/c"]).parts == ["a", "b/c"]


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_join_subclass_preserved():
    class Sub(JsonPointer):
        pass

    assert isinstance(Sub("/a").join("/b"), Sub)


def test_path_and_str_and_repr():
    p = JsonPointer("/a~1b/m~0n/0/")
    assert p.path == "/a~1b/m~0n/0/"
    assert str(p) == "/a~1b/m~0n/0/"
    assert repr(p) == "JsonPointer('/a~1b/m~0n/0/')"
    assert str(JsonPointer("")) == ""
    assert repr(JsonPointer("")) == "JsonPointer('')"


@pytest.mark.parametrize("s", ["", "/", "//", "/a", "/a~1b", "/~01", "/~0~1", "/a/0/-"])
def test_path_roundtrip(s):
    assert JsonPointer(s).path == s


def test_eq_and_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/a") != "/a"
    assert not (JsonPointer("/a") == ["a"])
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert JsonPointer("/a/b") != JsonPointer("/b/a")


def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([]).parts == []
    assert JsonPointer.from_parts([""]).parts == [""]
    assert JsonPointer.from_parts(["a", "b"]) == JsonPointer("/a/b")
    assert JsonPointer.from_parts(["a/b", "c~d"]).parts == ["a/b", "c~d"]
    assert JsonPointer.from_parts(iter(["x", "y"])).path == "/x/y"


def test_from_parts_get_parts_roundtrip():
    parts = ["a", "b/c", "~", "~1", "", " "]
    assert JsonPointer.from_parts(parts).get_parts() == parts


# ---- pairwise ----

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---- misc ----

def test_resolve_pointer_accepts_only_str_pointer_syntax():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "a")


def test_resolve_does_not_modify_doc():
    doc = {"a": [1, {"b": 2}]}
    orig = copy.deepcopy(doc)
    resolve_pointer(doc, "/a/1/b")
    resolve_pointer(doc, "/a/-")
    assert doc == orig
