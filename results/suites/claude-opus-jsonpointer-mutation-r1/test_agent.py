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


@pytest.fixture
def doc():
    return {
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
        "nested": {"list": [{"x": 10}, {"x": 20}], "obj": {"deep": "v"}},
    }


@pytest.fixture(autouse=True)
def restore_verbose():
    old = jsonpointer.VERBOSE_EXCEPTIONS
    yield
    jsonpointer.VERBOSE_EXCEPTIONS = old


# --- module metadata -------------------------------------------------------

def test_verbose_default_true():
    assert jsonpointer.VERBOSE_EXCEPTIONS is True


# --- RFC 6901 examples -----------------------------------------------------

@pytest.mark.parametrize("ptr,expected_key", [
    ("/foo", "foo"),
    ("/", ""),
    ("/a~1b", "a/b"),
    ("/c%d", "c%d"),
    ("/e^f", "e^f"),
    ("/g|h", "g|h"),
    ("/i\\j", "i\\j"),
    ('/k"l', 'k"l'),
    ("/ ", " "),
    ("/m~0n", "m~n"),
])
def test_rfc_examples(doc, ptr, expected_key):
    assert resolve_pointer(doc, ptr) == doc[expected_key]
    assert JsonPointer(ptr).resolve(doc) == doc[expected_key]
    assert JsonPointer(ptr).get(doc) == doc[expected_key]


def test_root(doc):
    assert resolve_pointer(doc, "") is doc
    assert JsonPointer("").parts == []


def test_list_indices(doc):
    assert resolve_pointer(doc, "/foo/0") == "bar"
    assert resolve_pointer(doc, "/foo/1") == "baz"
    assert resolve_pointer(doc, "/nested/list/1/x") == 20
    assert resolve_pointer(doc, "/nested/list/0/x") == 10
    assert resolve_pointer(doc, "/nested/obj/deep") == "v"


def test_multi_digit_index():
    lst = list(range(20))
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19
    assert resolve_pointer({"a": lst}, "/a/12") == 12


# --- parsing ---------------------------------------------------------------

def test_parts():
    assert JsonPointer("/a/b").parts == ["a", "b"]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a~1b/c~0d").parts == ["a/b", "c~d"]
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/~10").parts == ["/0"]
    assert JsonPointer("/~0~1").parts == ["~/"]


@pytest.mark.parametrize("ptr", ["a", "a/b", "foo", " /a", "0"])
def test_must_start_with_slash(ptr):
    with pytest.raises(JsonPointerException, match="Location must start with /"):
        JsonPointer(ptr)


@pytest.mark.parametrize("ptr,bad", [
    ("/~2", "~2"),
    ("/a~", "~"),
    ("/~", "~"),
    ("/~a/b", "~a"),
    ("/a/~~0", "~~"),
    ("/x~/y", "~/"),
])
def test_invalid_escape(ptr, bad):
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer(ptr)
    assert str(ei.value) == "Found invalid escape {}".format(bad)


@pytest.mark.parametrize("ptr", ["/~0", "/~1", "/a~0b~1c", "/~0~0", "/~1~1"])
def test_valid_escape(ptr):
    JsonPointer(ptr)


def test_invalid_escape_checked_before_slash():
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer("~2")


# --- escape / unescape -----------------------------------------------------

def test_escape():
    assert escape("a/b") == "a~1b"
    assert escape("a~b") == "a~0b"
    assert escape("~/") == "~0~1"
    assert escape("/~") == "~1~0"
    assert escape("~1") == "~01"
    assert escape("plain") == "plain"


def test_unescape():
    assert unescape("a~1b") == "a/b"
    assert unescape("a~0b") == "a~b"
    assert unescape("~01") == "~1"
    assert unescape("~0~1") == "~/"
    assert unescape("plain") == "plain"


@pytest.mark.parametrize("s", ["", "a", "~", "/", "~1", "~0", "/~/~", "~01/"])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


# --- pairwise --------------------------------------------------------------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# --- array index validation ------------------------------------------------

@pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", " 1", "1 ", "1.0",
                                 "1a", "a", "", "0x1", "1e2", "10a", "a10"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([0, 1, 2], "/" + idx)
    assert str(ei.value) == "'%s' is not a valid sequence index" % idx


def test_invalid_array_index_default():
    assert resolve_pointer([1, 2], "/01", "dflt") == "dflt"


def test_get_part():
    assert JsonPointer.get_part({}, "x") == "x"
    assert JsonPointer.get_part({}, "0") == "0"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "15") == 15
    assert type(JsonPointer.get_part([1], "3")) is int
    assert JsonPointer.get_part([1], "-") == "-"
    assert JsonPointer.get_part([1], 7) == 7
    assert JsonPointer.get_part((1,), "0") == 0


def test_get_part_string_errors():
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer.get_part("abc", "0")
    assert str(ei.value) == "Cannot apply token '0' to non-container type %s" % str


def test_get_part_non_indexable():
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer.get_part(5, "0")
    assert str(ei.value) == (
        "Document '%s' does not support indexing, "
        "must be mapping/sequence or support __getitem__" % int)
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(None, "a")


# --- walking / resolving ---------------------------------------------------

def test_index_out_of_bounds():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([1, 2], "/2")
    assert str(ei.value) == "index '2' is out of bounds"
    assert resolve_pointer([1, 2], "/1") == 2
    assert resolve_pointer([1, 2], "/5", None) is None


def test_member_not_found_verbose():
    d = {"a": 1}
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(d, "/b")
    assert str(ei.value) == "member 'b' not found in %s" % (d,)


def test_member_not_found_not_verbose():
    jsonpointer.VERBOSE_EXCEPTIONS = False
    d = {"a": 1}
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(d, "/b")
    assert str(ei.value) == "member 'b' not found"


def test_resolve_default():
    d = {"a": {"b": 1}}
    assert resolve_pointer(d, "/x/y", None) is None
    assert resolve_pointer(d, "/a/c", 42) == 42
    assert resolve_pointer(d, "/a/b", 42) == 1
    assert JsonPointer("/x").resolve(d, default=0) == 0
    assert JsonPointer("/x").resolve(d, False) is False
    with pytest.raises(JsonPointerException):
        JsonPointer("/x").resolve(d)


def test_resolve_through_scalar():
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        resolve_pointer({"a": 1}, "/a/b")
    assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"
    with pytest.raises(JsonPointerException, match="non-container"):
        resolve_pointer({"a": "str"}, "/a/0")


def test_end_of_list():
    lst = [1, 2]
    res = resolve_pointer(lst, "/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst
    assert repr(res) == "EndOfList([1, 2])"
    res2 = JsonPointer("/a/-").resolve({"a": []})
    assert isinstance(res2, EndOfList)
    assert repr(res2) == "EndOfList([])"


def test_dash_on_mapping_is_key():
    assert resolve_pointer({"-": 3}, "/-") == 3


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)


class Duck:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


def test_duck_typing():
    d = Duck({"a": {"b": 2}})
    assert resolve_pointer(d, "/a/b") == 2
    assert JsonPointer.get_part(d, "a") == "a"
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer(d, "/missing")
    assert "member 'missing' not found" in str(ei.value)
    assert resolve_pointer(d, "/missing", 9) == 9


def test_duck_typing_string_key_not_converted():
    d = Duck({"0": "zero", 0: "int"})
    assert resolve_pointer(d, "/0") == "zero"


def test_mapping_numeric_key_stays_string():
    assert resolve_pointer({"0": "s", 0: "i"}, "/0") == "s"


def test_to_last(doc):
    assert JsonPointer("").to_last(doc) == (doc, None)
    parent, part = JsonPointer("/nested/list/1").to_last(doc)
    assert parent is doc["nested"]["list"]
    assert part == 1 and type(part) is int
    parent, part = JsonPointer("/foo").to_last(doc)
    assert parent is doc
    assert part == "foo"
    parent, part = JsonPointer("/foo/-").to_last(doc)
    assert part == "-"
    parent, part = JsonPointer("/nested/obj/deep").to_last(doc)
    assert parent is doc["nested"]["obj"]
    assert part == "deep"


# --- set -------------------------------------------------------------------

def test_set_inplace():
    d = {"a": {"b": [1, 2]}}
    res = set_pointer(d, "/a/b/0", 9)
    assert res is d
    assert d == {"a": {"b": [9, 2]}}
    set_pointer(d, "/a/c", "new")
    assert d["a"]["c"] == "new"
    JsonPointer("/a/b/1").set(d, 7)
    assert d["a"]["b"] == [9, 7]


def test_set_not_inplace():
    d = {"a": {"b": [1, 2]}}
    orig = copy.deepcopy(d)
    res = set_pointer(d, "/a/b/0", 9, inplace=False)
    assert d == orig
    assert res == {"a": {"b": [9, 2]}}
    assert res is not d
    res2 = JsonPointer("/a/x").set(d, 1, False)
    assert res2 == {"a": {"b": [1, 2], "x": 1}}
    assert d == orig


def test_set_root():
    d = {"a": 1}
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer(d, "", 5)
    assert set_pointer(d, "", 5, inplace=False) == 5
    assert JsonPointer("").set(d, [1], inplace=False) == [1]
    assert d == {"a": 1}


def test_set_append():
    d = {"a": [1]}
    set_pointer(d, "/a/-", 2)
    assert d == {"a": [1, 2]}
    lst = []
    assert set_pointer(lst, "/-", "x") == ["x"]
    assert lst == ["x"]


def test_set_dash_on_mapping():
    d = {}
    set_pointer(d, "/-", 1)
    assert d == {"-": 1}


def test_set_in_string():
    with pytest.raises(JsonPointerException) as ei:
        set_pointer({"a": "str"}, "/a/0", 1)
    assert "non-container" in str(ei.value) or "string" in str(ei.value)


def test_set_in_string_parent_direct():
    class P(JsonPointer):
        def to_last(self, doc):
            return doc, "0"
    with pytest.raises(JsonPointerException) as ei:
        P("/0").set("abc", 1)
    assert str(ei.value) == "Cannot set value in a string"


def test_set_out_of_bounds():
    lst = [1, 2]
    with pytest.raises(JsonPointerException) as ei:
        set_pointer(lst, "/5", 3)
    assert str(ei.value).startswith("Invalid assignment target: ")
    assert lst == [1, 2]


def test_set_type_error():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer({"a": (1, 2)}, "/a/0", 3)


def test_set_existing_list_index_replaces():
    lst = [1, 2, 3]
    set_pointer(lst, "/1", "x")
    assert lst == [1, "x", 3]


# --- contains / join / path ------------------------------------------------

def test_contains():
    p1 = JsonPointer("/a/b/c")
    p2 = JsonPointer("/a/b")
    p3 = JsonPointer("/a/c")
    root = JsonPointer("")
    assert p1.contains(p2)
    assert p2 in p1
    assert p1.contains(p1)
    assert p1 in p1
    assert not p2.contains(p1)
    assert p1 not in p2
    assert not p1.contains(p3)
    assert p3 not in p1
    assert p1.contains(root)
    assert root in p2
    assert not root.contains(p2)
    assert JsonPointer("/a/bc").contains(JsonPointer("/a/b")) is False


def test_join():
    p = JsonPointer("/a")
    assert p.join("/b").path == "/a/b"
    assert p.join(JsonPointer("/b/c")).path == "/a/b/c"
    assert p.join(["b", "c/d"]).path == "/a/b/c~1d"
    assert p.join([1, 2]).parts == ["a", "1", "2"]
    assert (p / "/x").path == "/a/x"
    assert (p / ["y"]).path == "/a/y"
    assert (p / JsonPointer("/z")).parts == ["a", "z"]
    assert p.join("").path == "/a"
    assert JsonPointer("").join("/q").path == "/q"
    assert isinstance(p.join("/b"), JsonPointer)
    assert p.path == "/a"


def test_join_subclass():
    class Sub(JsonPointer):
        pass
    assert type(Sub("/a").join("/b")) is Sub


def test_join_invalid():
    p = JsonPointer("/a")
    with pytest.raises(JsonPointerException) as ei:
        p.join(5)
    assert str(ei.value) == "Invalid suffix"
    with pytest.raises(JsonPointerException):
        p / None
    with pytest.raises(JsonPointerException, match="Location must start"):
        p.join("b")


def test_path_and_str_repr():
    p = JsonPointer("/~0/0/~1")
    assert p.path == "/~0/0/~1"
    assert str(p) == "/~0/0/~1"
    assert repr(p) == "JsonPointer('/~0/0/~1')"
    assert JsonPointer("").path == ""
    assert str(JsonPointer("")) == ""
    assert JsonPointer("/").path == "/"
    assert JsonPointer("//a").path == "//a"


def test_repr_subclass():
    class Sub(JsonPointer):
        pass
    assert repr(Sub("/a")) == "Sub('/a')"


def test_from_parts():
    p = JsonPointer.from_parts(["a", "~", "/", 0])
    assert p.path == "/a/~0/~1/0"
    assert p.parts == ["a", "~", "/", "0"]
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    assert isinstance(JsonPointer.from_parts(["x"]), JsonPointer)


def test_eq_hash():
    a = JsonPointer("/a/b")
    b = JsonPointer.from_parts(["a", "b"])
    c = JsonPointer("/a/c")
    assert a == b
    assert not (a != b)
    assert a != c
    assert not (a == c)
    assert hash(a) == hash(b)
    assert hash(a) == hash(("a", "b"))
    assert a != "/a/b"
    assert (a == "/a/b") is False
    assert (a == ["a", "b"]) is False
    assert len({a, b, c}) == 2
    assert {a: 1}[b] == 1
