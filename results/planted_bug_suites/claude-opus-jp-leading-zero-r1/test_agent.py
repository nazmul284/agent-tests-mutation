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


# RFC 6901 section 5 example document
RFC_DOC = {
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


@pytest.fixture
def doc():
    return {
        "foo": {"anArray": [{"prop": 44}], "another prop": {"baz": "A string"}},
        "a%20b": 1,
        "c d": 2,
    }


# --- RFC examples -----------------------------------------------------------

@pytest.mark.parametrize(
    "pointer, expected",
    [
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
        ('/k"l', 6),
        ("/ ", 7),
        ("/m~0n", 8),
    ],
)
def test_rfc_examples(pointer, expected):
    assert resolve_pointer(RFC_DOC, pointer) == expected
    assert JsonPointer(pointer).resolve(RFC_DOC) == expected
    assert JsonPointer(pointer).get(RFC_DOC) == expected


def test_root_resolves_to_same_object():
    d = {"a": 1}
    assert resolve_pointer(d, "") is d


# --- resolve ----------------------------------------------------------------

def test_resolve_nested(doc):
    assert resolve_pointer(doc, "/foo") is doc["foo"]
    assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"
    assert resolve_pointer(doc, "/foo/anArray/0") == {"prop": 44}
    assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44


def test_no_url_decoding(doc):
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/c d") == 2
    assert resolve_pointer(doc, "/a b", None) is None
    assert resolve_pointer(doc, "/c%20d", None) is None


def test_missing_member_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/some/path")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/foo/missing")


def test_default_returned_on_failure(doc):
    sentinel = object()
    assert resolve_pointer(doc, "/some/path", sentinel) is sentinel
    assert resolve_pointer(doc, "/foo/anArray/5", sentinel) is sentinel
    assert resolve_pointer(doc, "/foo/anArray/x", sentinel) is sentinel
    assert JsonPointer("/nope").resolve(doc, sentinel) is sentinel


def test_default_none_and_falsy_are_honoured(doc):
    assert resolve_pointer(doc, "/nope", None) is None
    assert resolve_pointer(doc, "/nope", 0) == 0
    assert resolve_pointer(doc, "/nope", False) is False
    assert resolve_pointer(doc, "/nope", "") == ""


def test_default_not_used_on_success(doc):
    assert resolve_pointer(doc, "/c d", "default") == 2


def test_falsy_values_resolve():
    d = {"a": 0, "b": None, "c": False, "d": "", "e": [], "f": {}}
    for k in d:
        assert resolve_pointer(d, "/" + k, "default") == d[k]


def test_out_of_bounds_index_raises():
    with pytest.raises(JsonPointerException, match="out of bounds"):
        resolve_pointer({"a": [1, 2]}, "/a/2")


def test_last_valid_index():
    assert resolve_pointer({"a": [1, 2, 3]}, "/a/2") == 3


def test_index_zero():
    assert resolve_pointer([10, 20], "/0") == 10


def test_multi_digit_index():
    lst = list(range(20))
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_top_level_list():
    assert resolve_pointer([[1, [2, 3]]], "/0/1/1") == 3


def test_tuple_is_sequence():
    assert resolve_pointer({"t": (5, 6)}, "/t/1") == 6


@pytest.mark.parametrize("bad", ["-1", "+1", " 1", "1 ", "1.0", "1e2", "a", "", "0x1"])
def test_invalid_array_indices(bad):
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/" + bad)
    assert resolve_pointer([1, 2, 3], "/" + bad, "dflt") == "dflt"


@pytest.mark.parametrize("bad", ["01", "00", "007"])
def test_leading_zero_indices_rejected(bad):
    # RFC 6901: array-index = %x30 / ( %x31-39 *(%x30-39) )
    # i.e. "0" or digits without a leading zero.
    lst = list(range(10))
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/" + bad)
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(lst, bad)


def test_unicode_digits_not_index():
    # Non-ASCII digits must not be accepted as array indices
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/١")


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    result = resolve_pointer({"a": lst}, "/a/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is lst


def test_end_of_list_repr():
    assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"


def test_dash_on_dict_is_key():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_numeric_key_on_dict_is_string():
    d = {"0": "zero", 0: "int zero"}
    assert resolve_pointer(d, "/0") == "zero"
    assert resolve_pointer({"01": "x"}, "/01") == "x"


def test_resolve_into_string_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "hello"}, "/a/0")
    assert resolve_pointer({"a": "hello"}, "/a/0", None) is None


def test_resolve_into_scalar_raises():
    for value in (1, 1.5, None, True):
        with pytest.raises(JsonPointerException):
            resolve_pointer({"a": value}, "/a/b")
        assert resolve_pointer({"a": value}, "/a/b", "d") == "d"


def test_resolve_custom_getitem():
    class Thing:
        def __getitem__(self, key):
            if key == "x":
                return 42
            raise KeyError(key)

    assert resolve_pointer({"t": Thing()}, "/t/x") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer({"t": Thing()}, "/t/y")


def test_resolve_user_mapping():
    d = UserDict({"a": {"b": 1}})
    assert resolve_pointer(d, "/a/b") == 1


def test_verbose_exceptions(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({"secret": 1}, "/nope")
    assert "nope" in str(exc.value)
    assert "secret" in str(exc.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer({"secret": 1}, "/nope")
    assert "nope" in str(exc.value)
    assert "secret" not in str(exc.value)


# --- parsing / validation ---------------------------------------------------

@pytest.mark.parametrize("bad", ["foo", "a/b", " /a", "#/a"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/a~b", "/~~", "/ok/~x/ok"])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(bad)


def test_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]
    assert JsonPointer("/a~1b/m~0n").parts == ["a/b", "m~n"]


def test_unescape_order():
    # "~01" must become "~1", not "/"
    assert JsonPointer("/~01").parts == ["~1"]
    assert unescape("~01") == "~1"
    assert JsonPointer("/~10").parts == ["/0"]


def test_escape_order():
    assert escape("~1") == "~01"
    assert escape("/") == "~1"
    assert escape("~") == "~0"
    assert escape("a/b~c") == "a~1b~0c"


def test_escape_unescape_roundtrip():
    for s in ["", "a", "~", "/", "~1", "~0", "/~", "~/", "a~/b/~c~~//"]:
        assert unescape(escape(s)) == s


# --- path / str / repr / equality ------------------------------------------

@pytest.mark.parametrize(
    "pointer",
    ["", "/", "//", "/foo", "/foo/0", "/a~1b", "/m~0n", "/~0/0/~1", "/~01", "/ "],
)
def test_path_roundtrip(pointer):
    p = JsonPointer(pointer)
    assert p.path == pointer
    assert str(p) == pointer
    assert JsonPointer(p.path) == p


def test_repr():
    assert repr(JsonPointer("/a~1b")) == "JsonPointer('/a~1b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_repr_subclass():
    class MyPointer(JsonPointer):
        pass

    assert repr(MyPointer("/x")) == "MyPointer('/x')"


def test_equality_and_hash():
    a = JsonPointer("/a/b")
    b = JsonPointer("/a/b")
    c = JsonPointer("/a/c")
    assert a == b
    assert not (a != b)
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2
    assert {a: 1}[b] == 1


def test_equality_with_other_types():
    p = JsonPointer("/a")
    assert p != "/a"
    assert p != ["a"]
    assert not (p == "/a")
    assert p is not None and p != None  # noqa: E711


def test_root_vs_empty_key_differ():
    assert JsonPointer("") != JsonPointer("/")


# --- from_parts / join -------------------------------------------------------

def test_from_parts():
    p = JsonPointer.from_parts(["a", "~", "/", 0])
    assert p.path == "/a/~0/~1/0"
    assert p.parts == ["a", "~", "/", "0"]


def test_from_parts_empty():
    assert JsonPointer.from_parts([]) == JsonPointer("")


def test_from_parts_returns_subclass():
    class MyPointer(JsonPointer):
        pass

    assert type(MyPointer.from_parts(["a"])) is MyPointer


def test_join_variants():
    base = JsonPointer("/a")
    expected = JsonPointer("/a/b/c")
    assert base.join("/b/c") == expected
    assert base.join(JsonPointer("/b/c")) == expected
    assert base.join(["b", "c"]) == expected
    assert base / "/b/c" == expected
    assert base / ["b", "c"] == expected
    assert (base / "/b") / "/c" == expected


def test_join_does_not_mutate():
    base = JsonPointer("/a")
    base.join("/b")
    assert base.parts == ["a"]


def test_join_escapes_parts():
    p = JsonPointer("/a").join(["b/c", "d~e"])
    assert p.parts == ["a", "b/c", "d~e"]
    assert p.path == "/a/b~1c/d~0e"


def test_join_empty():
    assert JsonPointer("/a").join("") == JsonPointer("/a")
    assert JsonPointer("").join("/x") == JsonPointer("/x")


def test_join_invalid_suffix():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("noslash")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


# --- contains -----------------------------------------------------------------

def test_contains():
    p = JsonPointer("/a/b/c")
    assert p.contains(JsonPointer("/a"))
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer("/a/b/c"))
    assert p.contains(JsonPointer(""))
    assert not p.contains(JsonPointer("/a/b/c/d"))
    assert not p.contains(JsonPointer("/a/c"))
    assert not p.contains(JsonPointer("/b"))
    # prefix must be part-wise, not string-wise
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))


def test_in_operator():
    p = JsonPointer("/a/b")
    assert JsonPointer("/a") in p
    assert JsonPointer("/b") not in p


# --- set ----------------------------------------------------------------------

def test_set_inplace(doc):
    result = set_pointer(doc, "/foo/anArray/0/prop", 55)
    assert result is doc
    assert doc["foo"]["anArray"][0]["prop"] == 55


def test_set_new_key(doc):
    set_pointer(doc, "/foo/yet another prop", "added")
    assert doc["foo"]["yet another prop"] == "added"


def test_set_no_url_decoding():
    d = {"foo": {}}
    set_pointer(d, "/foo/a%20b", "x")
    assert d == {"foo": {"a%20b": "x"}}


def test_set_escaped_key():
    d = {}
    set_pointer(d, "/a~1b", 1)
    set_pointer(d, "/m~0n", 2)
    assert d == {"a/b": 1, "m~n": 2}


def test_set_not_inplace(doc):
    original = copy.deepcopy(doc)
    result = set_pointer(doc, "/foo/anArray/0/prop", 99, inplace=False)
    assert doc == original
    assert result is not doc
    assert result["foo"]["anArray"][0]["prop"] == 99
    # deep copy: nested objects not shared
    assert result["foo"] is not doc["foo"]


def test_set_root():
    d = {"a": 1}
    with pytest.raises(JsonPointerException, match="root"):
        set_pointer(d, "", 5)
    with pytest.raises(JsonPointerException):
        JsonPointer("").set(d, 5)
    assert set_pointer(d, "", 5, inplace=False) == 5
    assert d == {"a": 1}


def test_set_list_index():
    d = {"a": [1, 2, 3]}
    set_pointer(d, "/a/1", "x")
    assert d == {"a": [1, "x", 3]}


def test_set_list_append():
    d = {"a": [1, 2]}
    set_pointer(d, "/a/-", 3)
    assert d == {"a": [1, 2, 3]}


def test_set_top_level_list_append():
    lst = [1]
    assert set_pointer(lst, "/-", 2) is lst
    assert lst == [1, 2]


def test_set_list_out_of_bounds():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": [1, 2]}, "/a/2", 3)


def test_set_list_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/x", 3)
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/-1", 3)


def test_set_into_string():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_into_tuple():
    with pytest.raises(JsonPointerException, match="Invalid assignment"):
        set_pointer({"a": (1, 2)}, "/a/0", "x")


def test_set_into_scalar():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


def test_set_missing_intermediate():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_dash_on_dict_is_key():
    d = {}
    set_pointer(d, "/-", 1)
    assert d == {"-": 1}


def test_set_numeric_key_on_dict_is_string():
    d = {}
    set_pointer(d, "/0", 1)
    assert d == {"0": 1}


def test_set_method():
    d = {"a": {"b": 1}}
    assert JsonPointer("/a/b").set(d, 2) is d
    assert d["a"]["b"] == 2


# --- to_last / get_part --------------------------------------------------------

def test_to_last():
    d = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(d) == (d, None)
    parent, part = JsonPointer("/a/b/1").to_last(d)
    assert parent is d["a"]["b"]
    assert part == 1 and isinstance(part, int)
    parent, part = JsonPointer("/a/b").to_last(d)
    assert parent is d["a"]
    assert part == "b"
    parent, part = JsonPointer("/a/b/-").to_last(d)
    assert part == "-"


def test_get_part():
    assert JsonPointer.get_part({}, "x") == "x"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "12") == 12
    assert JsonPointer.get_part([1], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([1], "a")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(42, "0")


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([1, 2], "1") == 2
    assert isinstance(p.walk([1, 2], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk({"a": 1}, "b")
    with pytest.raises(JsonPointerException):
        p.walk([1], "1")


# --- pairwise ------------------------------------------------------------------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]
