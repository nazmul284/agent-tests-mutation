import copy
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


# Example document from RFC 6901 section 5
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


# ---------------------------------------------------------------- escaping

@pytest.mark.parametrize("raw, escaped", [
    ("", ""),
    ("abc", "abc"),
    ("~", "~0"),
    ("/", "~1"),
    ("a/b", "a~1b"),
    ("m~n", "m~0n"),
    ("~1", "~01"),
    ("/~", "~1~0"),
    ("~/", "~0~1"),
])
def test_escape_unescape(raw, escaped):
    assert escape(raw) == escaped
    assert unescape(escaped) == raw


def test_unescape_order_matters():
    # "~01" must become "~1", not "/"
    assert unescape("~01") == "~1"
    assert unescape("~10") == "/0"


# ----------------------------------------------------------- RFC examples

@pytest.mark.parametrize("pointer, expected", [
    ("", RFC_DOC),
    ("/foo", ["bar", "baz"]),
    ("/foo/0", "bar"),
    ("/foo/1", "baz"),
    ("/", 0),
    ("/a~1b", 1),
    ("/c%d", 2),
    ("/e^f", 3),
    ("/g|h", 4),
    ("/i\\j", 5),
    ("/k\"l", 6),
    ("/ ", 7),
    ("/m~0n", 8),
])
def test_rfc_examples(pointer, expected):
    assert resolve_pointer(RFC_DOC, pointer) == expected
    assert JsonPointer(pointer).resolve(RFC_DOC) == expected
    assert JsonPointer(pointer).get(RFC_DOC) == expected


def test_root_returns_same_object():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc


def test_resolve_returns_identity_of_subobject():
    inner = {"x": 1}
    doc = {"a": [inner]}
    assert resolve_pointer(doc, "/a/0") is inner


def test_nested_resolution():
    doc = {"a": {"b": [{"c": [10, 20, 30]}]}}
    assert resolve_pointer(doc, "/a/b/0/c/2") == 30


def test_percent_not_decoded():
    doc = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/c d") == 2
    assert resolve_pointer(doc, "/c%20d", None) is None


def test_empty_key_chain():
    doc = {"": {"": 5}}
    assert resolve_pointer(doc, "//") == 5
    assert JsonPointer("//").parts == ["", ""]


def test_trailing_slash_means_empty_key():
    doc = {"a": {"": "empty"}}
    assert resolve_pointer(doc, "/a/") == "empty"


# ------------------------------------------------------------ parsing

@pytest.mark.parametrize("pointer, parts", [
    ("", []),
    ("/", [""]),
    ("/a", ["a"]),
    ("/a/b", ["a", "b"]),
    ("/a~1b", ["a/b"]),
    ("/m~0n", ["m~n"]),
    ("/~01", ["~1"]),
    ("/~10", ["/0"]),
    ("/0/-", ["0", "-"]),
])
def test_parts(pointer, parts):
    ptr = JsonPointer(pointer)
    assert ptr.parts == parts
    assert ptr.get_parts() == parts


@pytest.mark.parametrize("pointer", ["a", "a/b", " /a", "#/a"])
def test_must_start_with_slash(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer", ["/~", "/~2", "/a~", "/~a", "/foo~/bar", "/~~", "/a/~x/b"])
def test_invalid_escape(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


# ------------------------------------------------------- array indices

@pytest.mark.parametrize("index", ["01", "00", "007", "-1", "+1", "1.0", " 1", "1 ", "a", "1e2", "", "0x1"])
def test_invalid_array_index(index):
    doc = {"arr": list(range(20))}
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/arr/" + index)
    assert resolve_pointer(doc, "/arr/" + index, "dflt") == "dflt"


@pytest.mark.parametrize("index", ["١", "１", "²"])
def test_non_ascii_digits_rejected(index):
    doc = list(range(5))
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/" + index)


def test_zero_and_multidigit_indices_valid():
    doc = list(range(20))
    assert resolve_pointer(doc, "/0") == 0
    assert resolve_pointer(doc, "/10") == 10
    assert resolve_pointer(doc, "/19") == 19


def test_index_out_of_bounds():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")
    assert resolve_pointer([1, 2], "/5", None) is None


def test_get_part_types():
    assert JsonPointer.get_part({"a": 1}, "a") == "a"
    assert JsonPointer.get_part({"0": 1}, "0") == "0"
    assert JsonPointer.get_part([1], "0") == 0
    assert isinstance(JsonPointer.get_part([1], "12"), int)
    assert JsonPointer.get_part([1], "-") == "-"
    assert JsonPointer.get_part((1,), "0") == 0


def test_get_part_on_string_raises():
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("abc", "0")


def test_get_part_on_unindexable_raises():
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(None, "a")


def test_mapping_with_numeric_string_keys():
    doc = {"0": "zero", "01": "zero-one"}
    assert resolve_pointer(doc, "/0") == "zero"
    assert resolve_pointer(doc, "/01") == "zero-one"


def test_tuple_is_sequence():
    assert resolve_pointer({"t": (1, 2, 3)}, "/t/2") == 3


# --------------------------------------------------- end-of-list "-"

def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    result = resolve_pointer({"a": lst}, "/a/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is lst


def test_end_of_list_repr():
    assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"


def test_dash_on_mapping_is_key():
    assert resolve_pointer({"-": 3}, "/-") == 3


def test_walk_past_dash_fails():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": [1]}, "/a/-/0")


# ---------------------------------------------------- failure modes

def test_missing_member_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_missing_member_default():
    sentinel = object()
    assert resolve_pointer({"a": 1}, "/b", sentinel) is sentinel
    assert resolve_pointer({"a": 1}, "/b/c/d", sentinel) is sentinel
    assert JsonPointer("/b").resolve({"a": 1}, None) is None


def test_default_falsy_values_returned():
    for d in (None, 0, "", False, []):
        assert resolve_pointer({}, "/x", d) == d


def test_default_not_used_when_found():
    assert resolve_pointer({"a": None}, "/a", 5) is None
    assert resolve_pointer({"a": 0}, "/a", 5) == 0


def test_traverse_into_scalar_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/a/b")
    assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"


def test_traverse_into_string_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")
    assert resolve_pointer({"a": "hello"}, "/a/0", "d") == "d"


def test_traverse_into_none_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": None}, "/a/b")


def test_verbose_exceptions(monkeypatch):
    doc = {"secret": "content"}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer(doc, "/missing")
    assert "missing" in str(exc.value)
    assert "content" in str(exc.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer(doc, "/missing")
    assert "missing" in str(exc.value)
    assert "content" not in str(exc.value)


# ---------------------------------------------- duck-typed containers

class Indexable:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


def test_ducktyped_getitem():
    obj = Indexable({"a": Indexable({"b": 42})})
    assert resolve_pointer(obj, "/a/b") == 42
    assert JsonPointer.get_part(obj, "a") == "a"


def test_ducktyped_missing_key():
    obj = Indexable({"a": 1})
    with pytest.raises(JsonPointerException):
        resolve_pointer(obj, "/b")
    assert resolve_pointer(obj, "/b", None) is None


def test_userdict_mapping():
    doc = UserDict({"a": [1, 2]})
    assert resolve_pointer(doc, "/a/1") == 2


# -------------------------------------------------------------- set

def test_set_pointer_inplace_mapping():
    doc = {"foo": {"bar": 1}}
    result = set_pointer(doc, "/foo/bar", 2)
    assert result is doc
    assert doc == {"foo": {"bar": 2}}


def test_set_adds_new_member():
    doc = {"foo": {}}
    set_pointer(doc, "/foo/new", "x")
    assert doc == {"foo": {"new": "x"}}


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    assert doc == {"a/b": 1, "m~n": 2}


def test_set_list_index():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", 20)
    assert doc == {"a": [1, 20, 3]}


def test_set_list_append_dash():
    doc = {"a": [1, 2]}
    set_pointer(doc, "/a/-", 3)
    assert doc == {"a": [1, 2, 3]}


def test_set_top_level_list():
    doc = [1, 2]
    set_pointer(doc, "/0", 9)
    set_pointer(doc, "/-", 3)
    assert doc == [9, 2, 3]


def test_set_list_out_of_bounds():
    doc = [1, 2]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/2", 3)
    assert doc == [1, 2]


@pytest.mark.parametrize("index", ["01", "-1", "x", "1.5"])
def test_set_list_invalid_index(index):
    doc = [1, 2, 3]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/" + index, 0)
    assert doc == [1, 2, 3]


def test_set_into_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"t": (1, 2)}, "/t/0", 5)


def test_set_into_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"s": "abc"}, "/s/0", "x")


def test_set_missing_intermediate_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "", 2)
    with pytest.raises(JsonPointerException):
        JsonPointer("").set({}, 1)


def test_set_root_not_inplace_returns_value():
    doc = {"a": 1}
    assert set_pointer(doc, "", 2, inplace=False) == 2
    assert doc == {"a": 1}


def test_set_not_inplace_copies():
    doc = {"a": {"b": [1, 2]}}
    original = copy.deepcopy(doc)
    result = set_pointer(doc, "/a/b/0", 99, inplace=False)
    assert result == {"a": {"b": [99, 2]}}
    assert doc == original
    assert result is not doc
    assert result["a"] is not doc["a"]


def test_set_not_inplace_append():
    doc = [1]
    result = JsonPointer("/-").set(doc, 2, inplace=False)
    assert result == [1, 2]
    assert doc == [1]


def test_set_value_identity():
    doc = {}
    val = [1, 2]
    set_pointer(doc, "/x", val)
    assert doc["x"] is val


def test_set_then_resolve_roundtrip():
    doc = {"a": [{"b": {}}]}
    ptr = JsonPointer("/a/0/b/c~1d")
    ptr.set(doc, "v")
    assert ptr.resolve(doc) == "v"
    assert doc["a"][0]["b"]["c/d"] == "v"


def test_set_dash_on_mapping_is_key():
    doc = {}
    set_pointer(doc, "/-", 1)
    assert doc == {"-": 1}


# ----------------------------------------------------------- to_last

def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(doc) == (doc, None)
    parent, part = JsonPointer("/a/b/1").to_last(doc)
    assert parent is doc["a"]["b"]
    assert part == 1 and isinstance(part, int)
    parent, part = JsonPointer("/a/b").to_last(doc)
    assert parent is doc["a"]
    assert part == "b"
    parent, part = JsonPointer("/a/b/-").to_last(doc)
    assert part == "-"


# ---------------------------------------------------------- path/str

@pytest.mark.parametrize("pointer", [
    "", "/", "/a", "/a/b", "/a~1b", "/m~0n", "/~01", "/~0/0/~1", "//", "/a/", "/0/-",
])
def test_path_roundtrip(pointer):
    ptr = JsonPointer(pointer)
    assert ptr.path == pointer
    assert str(ptr) == pointer
    assert JsonPointer(ptr.path) == ptr


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_repr_subclass():
    class MyPointer(JsonPointer):
        pass
    assert repr(MyPointer("/x")) == "MyPointer('/x')"


# ---------------------------------------------------------- from_parts

def test_from_parts():
    ptr = JsonPointer.from_parts(["a", "~", "/", 0])
    assert ptr.path == "/a/~0/~1/0"
    assert ptr.parts == ["a", "~", "/", "0"]


def test_from_parts_empty():
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([]) == JsonPointer("")


def test_from_parts_empty_string_part():
    assert JsonPointer.from_parts([""]).path == "/"


def test_from_parts_roundtrip():
    parts = ["a/b", "~x", "", "-", "0"]
    assert JsonPointer.from_parts(parts).parts == parts


def test_from_parts_subclass():
    class MyPointer(JsonPointer):
        pass
    assert type(MyPointer.from_parts(["a"])) is MyPointer


# --------------------------------------------------------- eq / hash

def test_equality():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/") != JsonPointer("")


def test_equality_with_non_pointer():
    assert JsonPointer("/a") != "/a"
    assert not (JsonPointer("/a") == "/a")
    assert JsonPointer("/a") != ["a"]
    assert JsonPointer("") != None  # noqa: E711


def test_hash():
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
    assert len(s) == 2
    d = {JsonPointer("/x~1y"): 1}
    assert d[JsonPointer.from_parts(["x/y"])] == 1


# ------------------------------------------------------------ contains

def test_contains():
    p = JsonPointer("/a/b/c")
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer("/a"))
    assert p.contains(JsonPointer(""))
    assert p.contains(JsonPointer("/a/b/c"))
    assert not p.contains(JsonPointer("/a/b/c/d"))
    assert not p.contains(JsonPointer("/a/c"))
    assert not p.contains(JsonPointer("/b"))
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))


def test_in_operator():
    p = JsonPointer("/a/b")
    assert JsonPointer("/a") in p
    assert JsonPointer("/b") not in p
    assert JsonPointer("/a/b/c") not in p


# ---------------------------------------------------------------- join

def test_join_string():
    assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")


def test_join_pointer():
    assert JsonPointer("/a").join(JsonPointer("/b")) == JsonPointer("/a/b")


def test_join_list_of_parts():
    joined = JsonPointer("/a").join(["b/c", "~", 0])
    assert joined.parts == ["a", "b/c", "~", "0"]
    assert joined.path == "/a/b~1c/~0/0"


def test_join_empty():
    assert JsonPointer("/a").join("") == JsonPointer("/a")
    assert JsonPointer("").join("/a") == JsonPointer("/a")
    assert JsonPointer("/a").join([]) == JsonPointer("/a")


def test_join_does_not_mutate():
    p = JsonPointer("/a")
    p.join("/b")
    p / "/c"
    assert p.parts == ["a"]


def test_truediv():
    assert JsonPointer("/a") / "/b" == JsonPointer("/a/b")
    assert JsonPointer("/a") / ["b", "c"] == JsonPointer("/a/b/c")
    assert JsonPointer("") / JsonPointer("/x") / "/y" == JsonPointer("/x/y")


def test_join_invalid_string_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a") / "/~2"


def test_join_invalid_non_iterable():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a") / None


def test_join_subclass():
    class MyPointer(JsonPointer):
        pass
    assert type(MyPointer("/a").join("/b")) is MyPointer


# ----------------------------------------------------------- pairwise

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# -------------------------------------------------------- misc / walk

def test_walk_direct():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk([5], "1")
    with pytest.raises(JsonPointerException):
        p.walk({}, "x")


def test_resolve_does_not_mutate():
    doc = {"a": [1, {"b": 2}]}
    original = copy.deepcopy(doc)
    resolve_pointer(doc, "/a/1/b")
    resolve_pointer(doc, "/a/-")
    resolve_pointer(doc, "/zz", None)
    assert doc == original


def test_pointer_reusable_across_docs():
    p = JsonPointer("/x/0")
    assert p.resolve({"x": [1]}) == 1
    assert p.resolve({"x": ["a"]}) == "a"
    assert p.resolve({"x": {"0": "m"}}) == "m"


def test_version_metadata():
    assert isinstance(jsonpointer.__version__, str)
