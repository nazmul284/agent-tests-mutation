import copy
import pytest

import jsonpointer
from jsonpointer import (
    JsonPointer, JsonPointerException, EndOfList, resolve_pointer,
    set_pointer, pairwise, escape, unescape,
)

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


@pytest.mark.parametrize("ptr,expected", [
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
])
def test_rfc_examples(ptr, expected):
    assert resolve_pointer(RFC_DOC, ptr) == expected
    assert JsonPointer(ptr).resolve(RFC_DOC) == expected
    assert JsonPointer(ptr).get(RFC_DOC) == expected


def test_escape_unescape():
    assert escape("a~b/c") == "a~0b~1c"
    assert escape("~/") == "~0~1"
    assert escape("/") == "~1"
    assert escape("~") == "~0"
    assert escape("abc") == "abc"
    assert unescape("~0") == "~"
    assert unescape("~1") == "/"
    assert unescape("a~1b~0c") == "a/b~c"
    # order matters: ~01 -> ~1 (not /)
    assert unescape("~01") == "~1"
    assert unescape("~10") == "/0"
    assert escape("~1") == "~01"
    assert unescape(escape("~1/~0")) == "~1/~0"


def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


def test_init_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/b").parts == ["a", "b"]
    assert JsonPointer("/a~1b/~0c").parts == ["a/b", "~c"]
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/a/b").get_parts() == ["a", "b"]


@pytest.mark.parametrize("bad", ["a", "a/b", "foo", " /a", "0"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException, match="start with /"):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~/b", "/~/", "/a/b~", "/~ ", "/~-"])
def test_invalid_escape(bad):
    with pytest.raises(JsonPointerException, match="invalid escape"):
        JsonPointer(bad)


def test_invalid_escape_message():
    with pytest.raises(JsonPointerException) as e:
        JsonPointer("/a~2")
    assert str(e.value) == "Found invalid escape ~2"
    with pytest.raises(JsonPointerException) as e:
        JsonPointer("/a~")
    assert str(e.value) == "Found invalid escape ~"


def test_valid_escapes_accepted():
    JsonPointer("/~0")
    JsonPointer("/~1")
    JsonPointer("/~0~1~0")


def test_resolve_errors():
    doc = {"a": [1, 2], "s": "str", "n": 5}
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/b")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/a/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/a/x")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/s/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/n/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/a/0/1")


def test_resolve_default():
    doc = {"a": [1, 2], "s": "str", "n": 5, "z": None}
    for p in ["/b", "/a/2", "/a/x", "/s/0", "/n/0", "/a/01", "/a/-1"]:
        assert resolve_pointer(doc, p, "D") == "D"
        assert resolve_pointer(doc, p, None) is None
        assert JsonPointer(p).resolve(doc, default=7) == 7
        assert JsonPointer(p).get(doc, 7) == 7
    assert resolve_pointer(doc, "/b/c/d", 0) == 0
    # default not used when found
    assert resolve_pointer(doc, "/a/0", "D") == 1
    assert resolve_pointer(doc, "/z", "D") is None
    assert resolve_pointer(doc, "", "D") is doc
    # falsy default is returned
    assert resolve_pointer(doc, "/q", 0) == 0
    assert resolve_pointer(doc, "/q", "") == ""


def test_resolve_error_messages():
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": 1}, "/b")
    assert "member 'b' not found in {'a': 1}" == str(e.value)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer([1], "/3")
    assert str(e.value) == "index '3' is out of bounds"
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer([1], "/x")
    assert str(e.value) == "'x' is not a valid sequence index"


def test_verbose_exceptions(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": 1}, "/b")
    assert str(e.value) == "member 'b' not found"
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": 1}, "/b")
    assert str(e.value) == "member 'b' not found in {'a': 1}"


@pytest.mark.parametrize("idx", ["01", "00", "+1", "-1", " 1", "1 ", "1.0", "1e0", "a", "", "0x1", "1a", "١"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([10, 20, 30], "/" + idx)


def test_valid_array_index():
    lst = list(range(100, 120))
    assert resolve_pointer(lst, "/0") == 100
    assert resolve_pointer(lst, "/1") == 101
    assert resolve_pointer(lst, "/9") == 109
    assert resolve_pointer(lst, "/10") == 110
    assert resolve_pointer(lst, "/19") == 119
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/20")
    with pytest.raises(JsonPointerException):
        resolve_pointer(lst, "/100")


def test_index_regex_exact():
    # "10x" must not partially match; "0" prefixed numbers rejected
    for bad in ["0a", "10x", "1 0", "00", "007", "0 "]:
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([1] * 20, bad)
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "10") == 10
    assert JsonPointer.get_part([1], "5") == 5


def test_tuple_sequence():
    assert resolve_pointer((1, 2, 3), "/2") == 3
    assert resolve_pointer({"a": (1, 2)}, "/a/1") == 2


def test_end_of_list():
    lst = [1, 2]
    r = resolve_pointer(lst, "/-")
    assert isinstance(r, EndOfList)
    assert r.list_ is lst
    assert repr(r) == "EndOfList([1, 2])"
    r = resolve_pointer({"a": [3]}, "/a/-")
    assert isinstance(r, EndOfList)
    assert r.list_ == [3]
    assert resolve_pointer([], "/-").list_ == []
    # '-' on a dict is a normal key
    assert resolve_pointer({"-": 1}, "/-") == 1
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/-")


def test_get_part():
    assert JsonPointer.get_part({}, "x") == "x"
    assert JsonPointer.get_part({}, "0") == "0"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "-") == "-"
    with pytest.raises(JsonPointerException, match="non-container"):
        JsonPointer.get_part("abc", "0")
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        JsonPointer.get_part(5, "0")
    with pytest.raises(JsonPointerException, match="does not support indexing"):
        JsonPointer.get_part(None, "0")


def test_get_part_message():
    with pytest.raises(JsonPointerException) as e:
        JsonPointer.get_part("abc", "0")
    assert str(e.value) == "Cannot apply token '0' to non-container type %s" % str
    with pytest.raises(JsonPointerException) as e:
        JsonPointer.get_part(5, "0")
    assert "Document '%s' does not support indexing" % int in str(e.value)


def test_duck_typed_getitem():
    class Duck:
        def __getitem__(self, k):
            if k == "a":
                return 42
            raise KeyError(k)

    assert JsonPointer.get_part(Duck(), "a") == "a"
    assert resolve_pointer(Duck(), "/a") == 42
    with pytest.raises(JsonPointerException):
        resolve_pointer(Duck(), "/b")
    assert resolve_pointer(Duck(), "/b", "d") == "d"
    d = {"x": Duck()}
    assert resolve_pointer(d, "/x/a") == 42


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        p.walk([5, 6], "2")
    with pytest.raises(JsonPointerException):
        p.walk({}, "a")


def test_walk_str_doc():
    with pytest.raises(JsonPointerException):
        JsonPointer("").walk("abc", "0")


def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    assert JsonPointer("").to_last(doc) == (doc, None)
    sub, last = JsonPointer("/a").to_last(doc)
    assert sub is doc and last == "a"
    sub, last = JsonPointer("/a/b/1").to_last(doc)
    assert sub is doc["a"]["b"] and last == 1
    sub, last = JsonPointer("/a/b/-").to_last(doc)
    assert sub is doc["a"]["b"] and last == "-"
    sub, last = JsonPointer("/a/b").to_last(doc)
    assert sub is doc["a"] and last == "b"
    with pytest.raises(JsonPointerException):
        JsonPointer("/x/y").to_last(doc)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a/b/x").to_last(doc)


def test_set_dict():
    doc = {"a": {"b": 1}}
    res = set_pointer(doc, "/a/b", 2)
    assert res is doc
    assert doc == {"a": {"b": 2}}
    set_pointer(doc, "/a/c", 3)
    assert doc == {"a": {"b": 2, "c": 3}}
    set_pointer(doc, "/n", None)
    assert doc["n"] is None
    set_pointer(doc, "/x~1y", 1)
    assert doc["x/y"] == 1
    set_pointer(doc, "/x~0y", 1)
    assert doc["x~y"] == 1


def test_set_list():
    doc = {"l": [1, 2, 3]}
    set_pointer(doc, "/l/0", 9)
    assert doc["l"] == [9, 2, 3]
    set_pointer(doc, "/l/2", 8)
    assert doc["l"] == [9, 2, 8]
    set_pointer(doc, "/l/-", 7)
    assert doc["l"] == [9, 2, 8, 7]
    set_pointer(doc, "/l/-", [1])
    assert doc["l"] == [9, 2, 8, 7, [1]]


def test_set_top_level_list_append():
    doc = []
    assert set_pointer(doc, "/-", 1) is doc
    assert doc == [1]


def test_set_list_errors():
    doc = {"l": [1, 2, 3]}
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer(doc, "/l/3", 0)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/l/x", 0)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/l/01", 0)
    assert doc == {"l": [1, 2, 3]}


def test_set_tuple_error():
    with pytest.raises(JsonPointerException, match="Invalid assignment target"):
        set_pointer({"t": (1, 2)}, "/t/0", 5)
    # append to tuple: no append method
    with pytest.raises(AttributeError):
        set_pointer({"t": (1, 2)}, "/t/-", 5)


def test_set_string_error():
    with pytest.raises(JsonPointerException, match="non-container"):
        set_pointer({"s": "abc"}, "/s/0", "x")
    with pytest.raises(JsonPointerException):
        set_pointer({"s": "abc"}, "/s/-", "x")


def test_set_missing_parent():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "/a/b", 1)


def test_set_root():
    with pytest.raises(JsonPointerException, match="Cannot set root in place"):
        set_pointer({"a": 1}, "", 5)
    assert set_pointer({"a": 1}, "", 5, inplace=False) == 5
    assert set_pointer({"a": 1}, "", 5, False) == 5
    doc = {"a": 1}
    assert set_pointer(doc, "", None, inplace=False) is None
    assert doc == {"a": 1}


def test_set_not_inplace():
    doc = {"a": {"b": [1, 2]}}
    orig = copy.deepcopy(doc)
    res = set_pointer(doc, "/a/b/0", 5, inplace=False)
    assert doc == orig
    assert res == {"a": {"b": [5, 2]}}
    assert res is not doc
    assert res["a"] is not doc["a"]
    assert res["a"]["b"] is not doc["a"]["b"]
    res = set_pointer(doc, "/a/b/-", 5, inplace=False)
    assert res["a"]["b"] == [1, 2, 5]
    assert doc == orig
    res = set_pointer(doc, "/a/c", 5, False)
    assert doc == orig and res["a"]["c"] == 5


def test_set_inplace_default_true():
    doc = {"a": 1}
    assert JsonPointer("/a").set(doc, 2) is doc
    assert doc["a"] == 2
    assert JsonPointer("/a").set(doc, 3, True) is doc
    assert doc["a"] == 3


def test_set_str_percent_not_decoded():
    doc = {"foo": {}}
    set_pointer(doc, "/foo/a%20b", "x")
    assert doc == {"foo": {"a%20b": "x"}}


def test_set_invalid_assignment_message():
    with pytest.raises(JsonPointerException) as e:
        set_pointer([1], "/5", 0)
    assert str(e.value).startswith("Invalid assignment target: ")


def test_set_non_container_parent():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 5}, "/a/b", 1)


def test_path_and_str():
    for p in ["", "/", "//", "/a/b", "/a~1b/~0", "/~01", "/foo/0", "/ "]:
        ptr = JsonPointer(p)
        assert ptr.path == p
        assert str(ptr) == p
    assert JsonPointer("/a/b/c").path == "/a/b/c"


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"
    assert repr(JsonPointer("/a~1b")) == "JsonPointer('/a~1b')"

    class Sub(JsonPointer):
        pass
    assert repr(Sub("/a")) == "Sub('/a')"


def test_eq_hash():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a")
    assert not (JsonPointer("/a") == "/a")
    assert JsonPointer("/a") != "/a"
    assert JsonPointer("/a") != None  # noqa
    assert JsonPointer("/a") != ["a"]
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert hash(JsonPointer("/a/b")) == hash(("a", "b"))
    assert hash(JsonPointer("")) == hash(())
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert {JsonPointer("/a/b"): 1}[JsonPointer("/a/b")] == 1
    assert (JsonPointer("/a/b") == JsonPointer("/a/b")) is True
    assert (JsonPointer("/a") == JsonPointer("/b")) is False
    assert (JsonPointer("/a") == 5) is False


def test_contains():
    a = JsonPointer("/a")
    ab = JsonPointer("/a/b")
    abc = JsonPointer("/a/b/c")
    b = JsonPointer("/b")
    root = JsonPointer("")
    assert a.contains(ab) is False
    assert ab.contains(a) is True
    assert ab.contains(ab) is True
    assert abc.contains(ab) is True
    assert ab.contains(abc) is False
    assert ab.contains(b) is False
    assert b.contains(a) is False
    assert ab.contains(root) is True
    assert root.contains(root) is True
    assert root.contains(a) is False
    assert (a in ab) is True
    assert (ab in a) is False
    assert (b in ab) is False
    assert (root in a) is True
    assert JsonPointer("/a/x").contains(JsonPointer("/a/b")) is False
    assert JsonPointer("/x/b").contains(JsonPointer("/a/b")) is False


def test_join():
    a = JsonPointer("/a")
    assert a.join(JsonPointer("/b/c")).path == "/a/b/c"
    assert a.join("/b/c").path == "/a/b/c"
    assert a.join(["b", "c"]).path == "/a/b/c"
    assert a.join(["b", 0]).path == "/a/b/0"
    assert a.join(["x/y", "~"]).parts == ["a", "x/y", "~"]
    assert a.join(["x/y"]).path == "/a/x~1y"
    assert a.join("").path == "/a"
    assert JsonPointer("").join("/a").path == "/a"
    assert a.join("/b") == JsonPointer("/a/b")
    assert a.path == "/a"  # unchanged
    assert isinstance(a.join("/b"), JsonPointer)
    assert a.join(("b",)).path == "/a/b"


def test_join_invalid():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException, match="Invalid suffix"):
        JsonPointer("/a").join(5)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(None)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("/~2")


def test_join_subclass():
    class Sub(JsonPointer):
        pass
    assert isinstance(Sub("/a").join("/b"), Sub)
    assert isinstance(Sub("/a") / "/b", Sub)
    assert isinstance(Sub.from_parts(["a"]), Sub)


def test_truediv():
    a = JsonPointer("/a")
    assert (a / "/b").path == "/a/b"
    assert (a / JsonPointer("/b")).path == "/a/b"
    assert (a / ["b", "c"]).path == "/a/b/c"
    assert (a / "/b" / "/c").path == "/a/b/c"


def test_from_parts():
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([]).parts == []
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts(["a", "b"]).parts == ["a", "b"]
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["a/b", "~c"]).parts == ["a/b", "~c"]
    assert JsonPointer.from_parts(iter(["x", "y"])).path == "/x/y"
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]
    assert JsonPointer.from_parts([1, 2]).parts == ["1", "2"]


def test_roundtrip():
    for parts in [[], [""], ["a", "b"], ["~", "/", "~/", "~1", "~0"], ["", "", "x"]]:
        assert JsonPointer.from_parts(parts).parts == parts
        assert JsonPointer(JsonPointer.from_parts(parts).path).parts == parts


def test_empty_key():
    doc = {"": {"": 1}}
    assert resolve_pointer(doc, "/") == {"": 1}
    assert resolve_pointer(doc, "//") == 1


def test_nested_resolution():
    doc = {"a": [{"b": [10, {"c": "deep"}]}]}
    assert resolve_pointer(doc, "/a/0/b/1/c") == "deep"
    assert resolve_pointer(doc, "/a/0/b/0") == 10
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/a/0/b/1/d")


def test_numeric_dict_keys_are_strings():
    assert resolve_pointer({"0": "x"}, "/0") == "x"
    with pytest.raises(JsonPointerException):
        resolve_pointer({0: "x"}, "/0")


def test_resolve_default_nothing_sentinel():
    assert jsonpointer._nothing is not None
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").resolve({})
    # Exception propagates only JsonPointerException, not KeyError
    try:
        JsonPointer("/a").resolve({})
    except JsonPointerException:
        pass


def test_doctests():
    import doctest
    res = doctest.testmod(jsonpointer)
    assert res.failed == 0
    assert res.attempted > 0


def test_pointer_class_attrs():
    assert JsonPointer.get is JsonPointer.resolve
    assert jsonpointer.VERBOSE_EXCEPTIONS is True
    assert jsonpointer.__version__ == "3.2.1"
