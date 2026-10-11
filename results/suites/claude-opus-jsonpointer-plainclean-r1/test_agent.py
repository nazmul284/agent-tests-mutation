import copy
from collections import UserDict, OrderedDict

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


# The example document from RFC 6901, section 5.
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


@pytest.fixture
def doc():
    return {
        "foo": {
            "anArray": [{"prop": 44}, {"prop": 45}],
            "another prop": {"baz": "A string"},
        },
        "list": [1, [2, 3], {"x": None}],
        "zero": 0,
        "none": None,
        "false": False,
    }


# ---------------------------------------------------------------------------
# RFC 6901 compliance
# ---------------------------------------------------------------------------

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


def test_root_resolves_to_same_object():
    d = {"a": 1}
    assert resolve_pointer(d, "") is d


def test_resolve_returns_identical_objects(doc):
    assert resolve_pointer(doc, "/foo") is doc["foo"]
    assert resolve_pointer(doc, "/foo/anArray/1") is doc["foo"]["anArray"][1]


def test_resolve_nested(doc):
    assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44
    assert resolve_pointer(doc, "/foo/anArray/1/prop") == 45
    assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"
    assert resolve_pointer(doc, "/list/1/1") == 3
    assert resolve_pointer(doc, "/list/2/x") is None


@pytest.mark.parametrize("key", ["zero", "none", "false"])
def test_resolve_falsy_values(doc, key):
    assert resolve_pointer(doc, "/" + key) == doc[key]
    assert resolve_pointer(doc, "/" + key, "default") == doc[key]


def test_no_url_decoding():
    d = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(d, "/a%20b") == 1
    assert resolve_pointer(d, "/c d") == 2
    assert resolve_pointer(d, "/a b", None) is None
    assert resolve_pointer(d, "/c%20d", None) is None


def test_empty_key_vs_root():
    d = {"": {"": "deep"}}
    assert resolve_pointer(d, "/") == {"": "deep"}
    assert resolve_pointer(d, "//") == "deep"


def test_trailing_slash_means_empty_key():
    d = {"a": {"": "empty"}}
    assert resolve_pointer(d, "/a/") == "empty"


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

def test_escape():
    assert escape("a/b") == "a~1b"
    assert escape("m~n") == "m~0n"
    assert escape("~/") == "~0~1"
    assert escape("/~") == "~1~0"
    assert escape("~1") == "~01"
    assert escape("plain") == "plain"
    assert escape("") == ""


def test_unescape():
    assert unescape("a~1b") == "a/b"
    assert unescape("m~0n") == "m~n"
    assert unescape("~0~1") == "~/"
    assert unescape("plain") == "plain"


def test_unescape_order():
    # "~01" must become "~1", not "/" (RFC 6901 section 4).
    assert unescape("~01") == "~1"
    assert unescape("~0~01") == "~~1"


@pytest.mark.parametrize("s", ["", "a", "~", "/", "~0", "~1", "~01", "a/b~c", "//~~", "~/~/"])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


def test_pointer_with_tilde01_key():
    d = {"~1": "tilde-one", "/": "slash"}
    assert resolve_pointer(d, "/~01") == "tilde-one"
    assert resolve_pointer(d, "/~1") == "slash"
    assert JsonPointer("/~01").parts == ["~1"]


@pytest.mark.parametrize("pointer", ["/~", "/a~", "/~2", "/a~b", "/~~", "/foo/~x/bar", "~"])
def test_invalid_escapes(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


def test_invalid_escape_message():
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer("/a~2")


# ---------------------------------------------------------------------------
# Construction / parts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pointer", ["foo", "a/b", " /a", "0"])
def test_must_start_with_slash(pointer):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(pointer)


@pytest.mark.parametrize("pointer, parts", [
    ("", []),
    ("/", [""]),
    ("//", ["", ""]),
    ("/a", ["a"]),
    ("/a/b", ["a", "b"]),
    ("/a~1b/c~0d", ["a/b", "c~d"]),
    ("/0/-", ["0", "-"]),
    ("/a/", ["a", ""]),
])
def test_parts(pointer, parts):
    ptr = JsonPointer(pointer)
    assert ptr.parts == parts
    assert ptr.get_parts() == parts


def test_parts_are_strings():
    assert JsonPointer("/0/1").parts == ["0", "1"]


# ---------------------------------------------------------------------------
# path / str / repr
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pointer", ["", "/", "//", "/a", "/a/b", "/~0/0/~1", "/a~1b/c~0d", "/~01", "/ /%20"])
def test_path_roundtrip(pointer):
    ptr = JsonPointer(pointer)
    assert ptr.path == pointer
    assert str(ptr) == pointer
    assert JsonPointer(ptr.path) == ptr


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"
    assert repr(JsonPointer("/a~1b")) == "JsonPointer('/a~1b')"


def test_repr_subclass():
    class MyPointer(JsonPointer):
        pass
    assert repr(MyPointer("/x")) == "MyPointer('/x')"


# ---------------------------------------------------------------------------
# from_parts
# ---------------------------------------------------------------------------

def test_from_parts():
    ptr = JsonPointer.from_parts(["a", "~", "/", 0])
    assert ptr.path == "/a/~0/~1/0"
    assert ptr.parts == ["a", "~", "/", "0"]


def test_from_parts_empty():
    ptr = JsonPointer.from_parts([])
    assert ptr.path == ""
    assert ptr.parts == []


def test_from_parts_empty_string_part():
    assert JsonPointer.from_parts([""]).path == "/"


def test_from_parts_roundtrip():
    parts = ["a/b", "c~d", "~1", "", "x"]
    ptr = JsonPointer.from_parts(parts)
    assert ptr.parts == parts
    assert JsonPointer(ptr.path).parts == parts


def test_from_parts_accepts_iterator():
    assert JsonPointer.from_parts(iter(["a", "b"])).path == "/a/b"


def test_from_parts_subclass():
    class MyPointer(JsonPointer):
        pass
    assert type(MyPointer.from_parts(["a"])) is MyPointer


# ---------------------------------------------------------------------------
# Equality / hashing
# ---------------------------------------------------------------------------

def test_equality():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("/") != JsonPointer("")
    assert JsonPointer.from_parts(["a/b"]) == JsonPointer("/a~1b")


def test_equality_other_types():
    assert JsonPointer("/a") != "/a"
    assert not (JsonPointer("/a") == "/a")
    assert JsonPointer("") != []
    assert JsonPointer("/a") != ["a"]
    assert JsonPointer("/a") != None  # noqa: E711


def test_hash():
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
    assert len(s) == 2
    d = {JsonPointer("/x/0"): 1}
    assert d[JsonPointer.from_parts(["x", 0])] == 1


# ---------------------------------------------------------------------------
# contains / join / truediv
# ---------------------------------------------------------------------------

def test_contains():
    p1 = JsonPointer("/a/b/c")
    p2 = JsonPointer("/a/b")
    p3 = JsonPointer("/b")
    root = JsonPointer("")
    assert p1.contains(p2)
    assert p1.contains(p1)
    assert p1.contains(root)
    assert not p1.contains(p3)
    assert not p2.contains(p1)
    assert root.contains(root)
    assert not root.contains(p2)


def test_contains_is_part_based_not_string_prefix():
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))
    assert not JsonPointer("/a/bc").contains(JsonPointer("/a/b"))


def test_in_operator():
    assert JsonPointer("/a/b") in JsonPointer("/a/b/c")
    assert JsonPointer("/a/b/c") not in JsonPointer("/a/b")
    assert JsonPointer("/x") not in JsonPointer("/a/b")


def test_join_string():
    assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")
    assert JsonPointer("").join("/b") == JsonPointer("/b")
    assert JsonPointer("/a").join("") == JsonPointer("/a")


def test_join_pointer():
    assert JsonPointer("/a").join(JsonPointer("/b~1c")) == JsonPointer("/a/b~1c")


def test_join_list():
    result = JsonPointer("/a").join(["b/c", "~", 0])
    assert result.parts == ["a", "b/c", "~", "0"]
    assert result.path == "/a/b~1c/~0/0"


def test_join_does_not_mutate():
    ptr = JsonPointer("/a")
    ptr.join("/b")
    ptr / "/c"
    assert ptr.parts == ["a"]


def test_join_invalid():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("/~2")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_truediv():
    assert JsonPointer("/a") / "/b" == JsonPointer("/a/b")
    assert JsonPointer("/a") / ["b", "c"] == JsonPointer("/a/b/c")
    assert JsonPointer("/a") / JsonPointer("/b") / "/c" == JsonPointer("/a/b/c")
    assert (JsonPointer("") / ["x/y"]).path == "/x~1y"


def test_join_preserves_subclass():
    class MyPointer(JsonPointer):
        pass
    assert type(MyPointer("/a").join("/b")) is MyPointer


# ---------------------------------------------------------------------------
# resolve errors and defaults
# ---------------------------------------------------------------------------

def test_missing_member_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/nope")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/foo/nope/deeper")


def test_missing_member_message(doc, monkeypatch):
    with pytest.raises(JsonPointerException, match="member 'nope' not found in"):
        resolve_pointer({"a": 1}, "/nope")
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({"secret": 1}, "/nope")
    assert str(exc.value) == "member 'nope' not found"
    assert "secret" not in str(exc.value)


def test_verbose_exception_includes_doc():
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({"secret": 1}, "/nope")
    assert "secret" in str(exc.value)


def test_default_returned(doc):
    assert resolve_pointer(doc, "/nope", None) is None
    assert resolve_pointer(doc, "/nope", 42) == 42
    assert resolve_pointer(doc, "/foo/anArray/99", "d") == "d"
    assert resolve_pointer(doc, "/foo/anArray/x", "d") == "d"
    assert resolve_pointer(doc, "/zero/x", "d") == "d"
    sentinel = object()
    assert JsonPointer("/a/b/c").resolve({}, sentinel) is sentinel
    assert JsonPointer("/a/b/c").get({}, sentinel) is sentinel


def test_default_false_values():
    assert resolve_pointer({}, "/a", False) is False
    assert resolve_pointer({}, "/a", 0) == 0
    assert resolve_pointer({}, "/a", "") == ""


def test_index_out_of_bounds():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": []}, "/a/0")


@pytest.mark.parametrize("index", ["01", "00", "-1", "+1", "1.0", " 1", "1 ", "a", "1e2", "0x1", "", "١"])
def test_invalid_array_index(index):
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(200)), "/" + index)


def test_invalid_array_index_message():
    with pytest.raises(JsonPointerException, match="not a valid sequence index"):
        resolve_pointer([1], "/01")


def test_invalid_index_with_default():
    assert resolve_pointer([1, 2], "/01", "d") == "d"
    assert resolve_pointer([1, 2], "/-1", "d") == "d"


@pytest.mark.parametrize("index, value", [("0", "a"), ("1", "b"), ("10", "k"), ("25", "z")])
def test_valid_array_index(index, value):
    letters = [chr(ord("a") + i) for i in range(26)]
    assert resolve_pointer(letters, "/" + index) == value


def test_tuple_is_sequence():
    assert resolve_pointer((10, 20, 30), "/2") == 30
    with pytest.raises(JsonPointerException):
        resolve_pointer((10, 20), "/2")


def test_string_is_not_container():
    with pytest.raises(JsonPointerException, match="non-container"):
        resolve_pointer({"s": "hello"}, "/s/0")
    assert resolve_pointer({"s": "hello"}, "/s/0", "d") == "d"


@pytest.mark.parametrize("value", [1, 1.5, None, True, object()])
def test_scalar_not_indexable(value):
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer({"v": value}, "/v/x")
    assert resolve_pointer({"v": value}, "/v/x", "d") == "d"


def test_end_of_list_resolve():
    lst = [1, 2]
    result = resolve_pointer({"a": lst}, "/a/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is lst


def test_end_of_list_repr():
    assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"


def test_end_of_list_cannot_be_traversed():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1], "/-/0")


def test_dash_on_mapping_is_key():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_numeric_string_keys_in_mapping():
    d = {"0": "zero", "01": "zero-one", "-1": "neg"}
    assert resolve_pointer(d, "/0") == "zero"
    assert resolve_pointer(d, "/01") == "zero-one"
    assert resolve_pointer(d, "/-1") == "neg"


def test_mapping_int_key_not_found():
    # Tokens are strings; an int key is not matched.
    with pytest.raises(JsonPointerException):
        resolve_pointer({0: "x"}, "/0")


def test_custom_mapping():
    d = UserDict({"a": UserDict({"b": 2})})
    assert resolve_pointer(d, "/a/b") == 2
    od = OrderedDict([("x", [1, 2])])
    assert resolve_pointer(od, "/x/1") == 2


def test_ducktyped_getitem():
    class Thing:
        def __getitem__(self, key):
            if key == "ok":
                return "found"
            raise KeyError(key)

    t = Thing()
    assert resolve_pointer({"t": t}, "/t/ok") == "found"
    with pytest.raises(JsonPointerException):
        resolve_pointer({"t": t}, "/t/bad")
    assert resolve_pointer({"t": t}, "/t/bad", "d") == "d"


def test_resolve_does_not_modify(doc):
    original = copy.deepcopy(doc)
    resolve_pointer(doc, "/foo/anArray/0/prop")
    resolve_pointer(doc, "/foo/nope", None)
    assert doc == original


# ---------------------------------------------------------------------------
# to_last / walk / get_part
# ---------------------------------------------------------------------------

def test_to_last_root():
    d = {"a": 1}
    parent, part = JsonPointer("").to_last(d)
    assert parent is d
    assert part is None


def test_to_last(doc):
    parent, part = JsonPointer("/foo/anArray/1").to_last(doc)
    assert parent is doc["foo"]["anArray"]
    assert part == 1
    assert isinstance(part, int)

    parent, part = JsonPointer("/foo/another prop").to_last(doc)
    assert parent is doc["foo"]
    assert part == "another prop"

    parent, part = JsonPointer("/list/-").to_last(doc)
    assert parent is doc["list"]
    assert part == "-"


def test_get_part():
    assert JsonPointer.get_part({}, "a") == "a"
    assert JsonPointer.get_part({}, "0") == "0"
    assert JsonPointer.get_part([], "0") == 0
    assert JsonPointer.get_part([], "12") == 12
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([], "01")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("abc", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")


def test_walk():
    ptr = JsonPointer("")
    assert ptr.walk({"a": 1}, "a") == 1
    assert ptr.walk([5, 6], "1") == 6
    assert isinstance(ptr.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        ptr.walk([5, 6], "2")
    with pytest.raises(JsonPointerException):
        ptr.walk({"a": 1}, "b")


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

def test_set_existing_inplace(doc):
    result = set_pointer(doc, "/foo/anArray/0/prop", 55)
    assert result is doc
    assert doc["foo"]["anArray"][0]["prop"] == 55
    assert doc["foo"]["anArray"][1]["prop"] == 45


def test_set_new_key(doc):
    set_pointer(doc, "/foo/yet another prop", "added")
    assert doc["foo"]["yet another prop"] == "added"
    assert doc["foo"]["another prop"] == {"baz": "A string"}


def test_set_no_url_decoding():
    d = {"foo": {}}
    assert set_pointer(d, "/foo/a%20b", "x") == {"foo": {"a%20b": "x"}}


def test_set_escaped_key():
    d = {}
    set_pointer(d, "/a~1b", 1)
    set_pointer(d, "/c~0d", 2)
    set_pointer(d, "/~01", 3)
    assert d == {"a/b": 1, "c~d": 2, "~1": 3}


def test_set_empty_key():
    d = {}
    set_pointer(d, "/", "v")
    assert d == {"": "v"}


def test_set_list_index():
    d = {"a": [1, 2, 3]}
    set_pointer(d, "/a/1", "x")
    assert d == {"a": [1, "x", 3]}


def test_set_list_append():
    d = {"a": [1, 2]}
    set_pointer(d, "/a/-", 3)
    assert d == {"a": [1, 2, 3]}
    JsonPointer("/a/-").set(d, 4)
    assert d == {"a": [1, 2, 3, 4]}


def test_set_top_level_list():
    lst = [1, 2]
    assert set_pointer(lst, "/0", 9) is lst
    assert lst == [9, 2]
    set_pointer(lst, "/-", 3)
    assert lst == [9, 2, 3]


def test_set_list_out_of_bounds():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer([1, 2], "/2", 3)


def test_set_list_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 3)
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/x", 3)


def test_set_in_string_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"s": "abc"}, "/s/0", "x")


def test_set_in_tuple_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"t": (1, 2)}, "/t/0", "x")


def test_set_on_scalar_fails():
    with pytest.raises(JsonPointerException):
        set_pointer({"n": 5}, "/n/x", 1)


def test_set_missing_intermediate_fails():
    d = {"a": {}}
    with pytest.raises(JsonPointerException):
        set_pointer(d, "/a/b/c", 1)
    assert d == {"a": {}}


def test_set_root_inplace_fails():
    with pytest.raises(JsonPointerException, match="root"):
        set_pointer({"a": 1}, "", 5)
    with pytest.raises(JsonPointerException):
        JsonPointer("").set({}, 5, inplace=True)


def test_set_root_not_inplace():
    d = {"a": 1}
    assert set_pointer(d, "", "new", inplace=False) == "new"
    assert JsonPointer("").set(d, [1], inplace=False) == [1]
    assert d == {"a": 1}


def test_set_not_inplace(doc):
    original = copy.deepcopy(doc)
    result = set_pointer(doc, "/foo/anArray/0/prop", 99, inplace=False)
    assert doc == original
    assert result is not doc
    assert result["foo"]["anArray"][0]["prop"] == 99
    # deep copy: nested objects are not shared
    assert result["foo"] is not doc["foo"]
    assert result["foo"]["anArray"] is not doc["foo"]["anArray"]
    result["foo"]["another prop"]["baz"] = "changed"
    assert doc["foo"]["another prop"]["baz"] == "A string"


def test_set_not_inplace_append():
    d = {"a": [1]}
    result = set_pointer(d, "/a/-", 2, inplace=False)
    assert result == {"a": [1, 2]}
    assert d == {"a": [1]}


def test_set_then_resolve_roundtrip():
    d = {"a": {"b": [0, {"c": None}]}}
    for ptr in ["/a/b/0", "/a/b/1/c", "/a/new", "/a/b/1/d~1e"]:
        set_pointer(d, ptr, ptr)
        assert resolve_pointer(d, ptr) == ptr


def test_set_custom_mapping():
    d = UserDict()
    set_pointer(d, "/k", "v")
    assert d["k"] == "v"


# ---------------------------------------------------------------------------
# pairwise
# ---------------------------------------------------------------------------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


def test_verbose_exceptions_default():
    assert jsonpointer.VERBOSE_EXCEPTIONS is True
