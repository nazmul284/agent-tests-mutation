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
        "nested": {"list": [{"x": 1}, {"x": 2}], "obj": {"y": None}},
    }


# ---------------------------------------------------------------- module consts

def test_module_metadata():
    assert jsonpointer.__version__ == "3.2.1"
    assert jsonpointer.VERBOSE_EXCEPTIONS is True


# ---------------------------------------------------------------- RFC examples

@pytest.mark.parametrize(
    "ptr, expected",
    [
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
        ("/nested/list/1/x", 2),
        ("/nested/obj/y", None),
    ],
)
def test_resolve_rfc_examples(doc, ptr, expected):
    assert resolve_pointer(doc, ptr) == expected
    assert JsonPointer(ptr).resolve(doc) == expected
    assert JsonPointer(ptr).get(doc) == expected


def test_resolve_root_returns_doc(doc):
    assert resolve_pointer(doc, "") is doc
    assert JsonPointer("").parts == []


def test_resolve_none_value_not_treated_as_missing(doc):
    assert resolve_pointer(doc, "/nested/obj/y", "dflt") is None


# ---------------------------------------------------------------- resolve errors / defaults

@pytest.mark.parametrize(
    "ptr",
    ["/missing", "/foo/2", "/foo/01", "/foo/-1", "/foo/+1", "/foo/ 1", "/foo/1.0",
     "/foo/a", "/foo/0/x", "/nested/obj/y/z", "/foo/1 ", "/foo/00", "/foo/10"],
)
def test_resolve_errors_raise_and_default(doc, ptr):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, ptr)
    sentinel = object()
    assert resolve_pointer(doc, ptr, sentinel) is sentinel
    assert resolve_pointer(doc, ptr, None) is None


def test_resolve_default_not_used_on_success(doc):
    assert resolve_pointer(doc, "/foo/0", "dflt") == "bar"


def test_resolve_multi_digit_index():
    lst = list(range(20))
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19
    assert resolve_pointer(lst, "/0") == 0
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/20")


def test_resolve_dash_returns_end_of_list(doc):
    res = resolve_pointer(doc, "/foo/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is doc["foo"]
    assert repr(res) == "EndOfList(['bar', 'baz'])"


def test_resolve_through_end_of_list_fails(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/foo/-/x")


def test_resolve_dash_key_in_mapping():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_resolve_numeric_key_in_mapping():
    assert resolve_pointer({"0": "a", "01": "b"}, "/0") == "a"
    assert resolve_pointer({"0": "a", "01": "b"}, "/01") == "b"


def test_resolve_on_string_raises():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"s": "abc"}, "/s/0")
    assert "Cannot apply token '0' to non-container type" in str(ei.value)


def test_resolve_on_scalar_raises():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"n": 5}, "/n/0")
    assert "does not support indexing" in str(ei.value)
    assert "int" in str(ei.value)
    assert resolve_pointer({"n": 5}, "/n/0", "d") == "d"


def test_resolve_tuple_sequence():
    assert resolve_pointer((1, (2, 3)), "/1/1") == 3


def test_resolve_ducktyped_getitem():
    class Thing:
        def __getitem__(self, key):
            if key == "ok":
                return 42
            raise KeyError(key)

    assert resolve_pointer({"t": Thing()}, "/t/ok") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer({"t": Thing()}, "/t/nope")


def test_missing_member_message_verbose():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"a": 1}, "/b")
    assert str(ei.value) == "member 'b' not found in {'a': 1}"


def test_missing_member_message_not_verbose(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"a": 1}, "/b")
    assert str(ei.value) == "member 'b' not found"


def test_out_of_bounds_message():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([1], "/5")
    assert str(ei.value) == "index '5' is out of bounds"


def test_invalid_index_message():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer([1], "/01")
    assert str(ei.value) == "'01' is not a valid sequence index"


# ---------------------------------------------------------------- constructor validation

@pytest.mark.parametrize("ptr", ["a", "foo/bar", " /a", "~0"])
def test_must_start_with_slash(ptr):
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer(ptr)
    assert "Location must start with /" in str(ei.value)


@pytest.mark.parametrize("ptr, bad", [("/~", "~"), ("/a~2", "~2"), ("/~a/b", "~a"), ("/a/~", "~")])
def test_invalid_escape(ptr, bad):
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer(ptr)
    assert str(ei.value) == "Found invalid escape {}".format(bad)


@pytest.mark.parametrize("ptr", ["/~0", "/~1", "/~01", "/~10", "/a~0b~1c"])
def test_valid_escapes(ptr):
    JsonPointer(ptr)


def test_parts_unescaped():
    assert JsonPointer("/a~1b/~0/~01/~10").parts == ["a/b", "~", "~1", "/0"]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]


# ---------------------------------------------------------------- escape / unescape

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
    assert unescape("~10") == "/0"
    assert unescape("plain") == "plain"


@pytest.mark.parametrize("s", ["", "~", "/", "~1", "~0", "/~", "~/", "a~1/b~0"])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


# ---------------------------------------------------------------- path / str / repr / eq / hash

@pytest.mark.parametrize("ptr", ["", "/", "/a", "/a/b", "/~0/0/~1", "/a~01", "//x"])
def test_path_roundtrip(ptr):
    p = JsonPointer(ptr)
    assert p.path == ptr
    assert str(p) == ptr


def test_repr():
    assert repr(JsonPointer("/a/~1")) == "JsonPointer('/a/~1')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_repr_subclass():
    class MyPtr(JsonPointer):
        pass

    assert repr(MyPtr("/x")) == "MyPtr('/x')"


def test_eq_and_hash():
    a = JsonPointer("/a/b")
    b = JsonPointer("/a/b")
    c = JsonPointer("/a/c")
    assert a == b
    assert not (a != b)
    assert a != c
    assert not (a == c)
    assert hash(a) == hash(b)
    assert hash(a) == hash(("a", "b"))
    assert len({a, b, c}) == 2
    assert a != "/a/b"
    assert not (a == "/a/b")
    assert a != ["a", "b"]


def test_from_parts():
    p = JsonPointer.from_parts(["a", "~", "/", 0])
    assert p.path == "/a/~0/~1/0"
    assert p.parts == ["a", "~", "/", "0"]
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([]).parts == []


def test_from_parts_subclass():
    class MyPtr(JsonPointer):
        pass

    p = MyPtr.from_parts(["a"])
    assert type(p) is MyPtr


# ---------------------------------------------------------------- contains / join

def test_contains():
    ab = JsonPointer("/a/b")
    a = JsonPointer("/a")
    b = JsonPointer("/b")
    root = JsonPointer("")
    assert ab.contains(a)
    assert ab.contains(ab)
    assert ab.contains(root)
    assert not ab.contains(b)
    assert not a.contains(ab)
    assert not JsonPointer("/a/c").contains(ab)
    assert a in ab
    assert b not in ab
    assert ab not in a


def test_join():
    a = JsonPointer("/a")
    assert a.join("/b/c") == JsonPointer("/a/b/c")
    assert a.join(JsonPointer("/b")) == JsonPointer("/a/b")
    assert a.join(["b", "c/d"]) == JsonPointer("/a/b/c~1d")
    assert a.join(["b", 1]).parts == ["a", "b", "1"]
    assert a.join("") == a
    assert (a / "/x") == JsonPointer("/a/x")
    assert (a / ["~"]).path == "/a/~0"
    assert a.path == "/a"  # unchanged


def test_join_subclass():
    class MyPtr(JsonPointer):
        pass

    assert type(MyPtr("/a").join("/b")) is MyPtr


def test_join_invalid():
    a = JsonPointer("/a")
    with pytest.raises(JsonPointerException) as ei:
        a.join(5)
    assert str(ei.value) == "Invalid suffix"
    with pytest.raises(JsonPointerException):
        a.join("no-slash")
    with pytest.raises(JsonPointerException):
        a / None


# ---------------------------------------------------------------- to_last / get_part / walk

def test_to_last():
    d = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(d) == (d, None)
    parent, part = JsonPointer("/a/b/1").to_last(d)
    assert parent is d["a"]["b"]
    assert part == 1 and isinstance(part, int)
    parent, part = JsonPointer("/a/b").to_last(d)
    assert parent is d["a"]
    assert part == "b"
    parent, part = JsonPointer("/a").to_last(d)
    assert parent is d and part == "a"


def test_get_part():
    assert JsonPointer.get_part({}, "x") == "x"
    assert JsonPointer.get_part({}, "-") == "-"
    assert JsonPointer.get_part([], "-") == "-"
    assert JsonPointer.get_part([], "3") == 3
    assert JsonPointer.get_part([], "0") == 0
    assert JsonPointer.get_part([], "12") == 12
    assert JsonPointer.get_part([], 7) == 7
    for bad in ("01", "-1", "a", "", "1a", "a1", " 1"):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([], bad)
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(3.5, "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(None, "0")

    class G:
        def __getitem__(self, k):
            return k

    assert JsonPointer.get_part(G(), "zz") == "zz"


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    e = p.walk([5, 6], "-")
    assert isinstance(e, EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk([5, 6], "2")
    with pytest.raises(JsonPointerException):
        p.walk({"a": 1}, "b")


# ---------------------------------------------------------------- set

def test_set_inplace_dict():
    d = {"a": {"b": 1}}
    res = set_pointer(d, "/a/b", 2)
    assert res is d
    assert d == {"a": {"b": 2}}


def test_set_add_key():
    d = {"a": {}}
    set_pointer(d, "/a/new key", "v")
    assert d == {"a": {"new key": "v"}}
    set_pointer(d, "/a/a%20b", "x")
    assert d["a"]["a%20b"] == "x"
    set_pointer(d, "/a~1b", 3)
    assert d["a/b"] == 3


def test_set_list_index():
    d = {"l": [1, 2, 3]}
    set_pointer(d, "/l/1", "x")
    assert d == {"l": [1, "x", 3]}
    set_pointer(d, "/l/0", "y")
    assert d == {"l": ["y", "x", 3]}


def test_set_list_append_dash():
    d = {"l": [1]}
    set_pointer(d, "/l/-", 2)
    assert d == {"l": [1, 2]}
    root = []
    set_pointer(root, "/-", "a")
    assert root == ["a"]


def test_set_dash_on_dict_sets_key():
    d = {}
    set_pointer(d, "/-", 1)
    assert d == {"-": 1}


def test_set_not_inplace():
    d = {"a": {"b": [1, 2]}}
    orig = copy.deepcopy(d)
    res = set_pointer(d, "/a/b/0", 9, inplace=False)
    assert d == orig
    assert res == {"a": {"b": [9, 2]}}
    assert res is not d
    res2 = JsonPointer("/a/b/-").set(d, 3, inplace=False)
    assert res2 == {"a": {"b": [1, 2, 3]}}
    assert d == orig


def test_set_inplace_default_true():
    d = {"a": 1}
    JsonPointer("/a").set(d, 2)
    assert d == {"a": 2}


def test_set_root():
    d = {"a": 1}
    with pytest.raises(JsonPointerException) as ei:
        set_pointer(d, "", 5)
    assert str(ei.value) == "Cannot set root in place"
    with pytest.raises(JsonPointerException):
        JsonPointer("").set(d, 5)
    assert set_pointer(d, "", 5, inplace=False) == 5
    assert d == {"a": 1}


def test_set_in_string_raises():
    d = {"s": "abc"}
    with pytest.raises(JsonPointerException):
        set_pointer(d, "/s/0", "x")
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer("/s").set("abc", 1)
    # parent is a string at top level
    assert "string" in str(ei.value) or "non-container" in str(ei.value)


def test_set_out_of_range_raises():
    d = {"l": [1]}
    with pytest.raises(JsonPointerException) as ei:
        set_pointer(d, "/l/5", 0)
    assert str(ei.value).startswith("Invalid assignment target: ")
    assert d == {"l": [1]}


def test_set_tuple_raises():
    with pytest.raises(JsonPointerException) as ei:
        set_pointer({"t": (1, 2)}, "/t/0", 0)
    assert str(ei.value).startswith("Invalid assignment target: ")


def test_set_tuple_dash_raises():
    with pytest.raises((JsonPointerException, AttributeError)):
        set_pointer({"t": (1, 2)}, "/t/-", 0)


def test_set_missing_intermediate_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": {}}, "/x/y", 1)


def test_set_invalid_index_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"l": [1, 2]}, "/l/01", 0)


def test_set_ducktyped():
    class Box:
        def __init__(self):
            self.d = {}

        def __getitem__(self, k):
            return self.d[k]

        def __setitem__(self, k, v):
            self.d[k] = v

    b = Box()
    set_pointer({"b": b}, "/b/k", 1)
    assert b.d == {"k": 1}


# ---------------------------------------------------------------- pairwise

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


# ---------------------------------------------------------------- exact error messages

def test_exact_message_location():
    with pytest.raises(JsonPointerException) as ei:
        JsonPointer("a/b")
    assert str(ei.value) == "Location must start with /"


def test_exact_message_non_container():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"s": "abc"}, "/s/0")
    assert str(ei.value) == "Cannot apply token '0' to non-container type <class 'str'>"


def test_exact_message_no_indexing():
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"n": 5}, "/n/0")
    assert str(ei.value) == (
        "Document '<class 'int'>' does not support indexing, "
        "must be mapping/sequence or support __getitem__"
    )
