import copy
from collections import OrderedDict

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


# The example document from RFC 6901, section 5
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
        "list": [10, 20, 30],
        "nested": [[1, 2], [3, [4, 5]]],
        "str": "hello",
        "num": 42,
        "none": None,
    }


# --- RFC 6901 examples -----------------------------------------------------

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


def test_root_resolves_to_same_object(doc):
    assert resolve_pointer(doc, "") is doc


# --- Parsing ---------------------------------------------------------------

@pytest.mark.parametrize(
    "pointer, parts",
    [
        ("", []),
        ("/", [""]),
        ("//", ["", ""]),
        ("/a", ["a"]),
        ("/a/b", ["a", "b"]),
        ("/a/", ["a", ""]),
        ("/~0", ["~"]),
        ("/~1", ["/"]),
        ("/~01", ["~1"]),
        ("/~10", ["/0"]),
        ("/~0~1", ["~/"]),
        ("/~1~0", ["/~"]),
        ("/~00", ["~0"]),
        ("/a~1b/c~0d", ["a/b", "c~d"]),
        ("/0", ["0"]),
        ("/-", ["-"]),
        ("/ü☃", ["ü☃"]),
    ],
)
def test_parts(pointer, parts):
    ptr = JsonPointer(pointer)
    assert ptr.parts == parts
    assert ptr.get_parts() == parts


@pytest.mark.parametrize("pointer", ["a", "a/b", " /a", "0", "~1/a"])
def test_must_start_with_slash(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


@pytest.mark.parametrize(
    "pointer", ["/~", "/a~", "/~2", "/~a", "/a~/b", "/~~", "/foo/~-1", "~"]
)
def test_invalid_escapes(pointer):
    with pytest.raises(JsonPointerException):
        JsonPointer(pointer)


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


# --- escape / unescape -----------------------------------------------------

@pytest.mark.parametrize(
    "raw, escaped",
    [
        ("", ""),
        ("abc", "abc"),
        ("~", "~0"),
        ("/", "~1"),
        ("~1", "~01"),
        ("/0", "~10"),
        ("a/b~c", "a~1b~0c"),
        ("~/~/", "~0~1~0~1"),
    ],
)
def test_escape_unescape(raw, escaped):
    assert escape(raw) == escaped
    assert unescape(escaped) == raw
    assert unescape(escape(raw)) == raw


# --- path / str / repr round trips ----------------------------------------

@pytest.mark.parametrize(
    "pointer",
    ["", "/", "//", "/a", "/a/b", "/~0", "/~1", "/~01", "/~10", "/a~1b/c~0d/0/-"],
)
def test_path_roundtrip(pointer):
    ptr = JsonPointer(pointer)
    assert ptr.path == pointer
    assert str(ptr) == pointer
    assert JsonPointer(ptr.path) == ptr


def test_repr():
    assert repr(JsonPointer("/a/~1")) == "JsonPointer('/a/~1')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_repr_of_subclass_uses_subclass_name():
    class MyPointer(JsonPointer):
        pass

    assert repr(MyPointer("/x")) == "MyPointer('/x')"


# --- from_parts ------------------------------------------------------------

def test_from_parts_escapes():
    ptr = JsonPointer.from_parts(["a", "~", "/", 0])
    assert ptr.path == "/a/~0/~1/0"
    assert ptr.parts == ["a", "~", "/", "0"]


def test_from_parts_empty():
    ptr = JsonPointer.from_parts([])
    assert ptr.parts == []
    assert ptr.path == ""


def test_from_parts_empty_string_part():
    assert JsonPointer.from_parts([""]).path == "/"


def test_from_parts_accepts_iterator():
    assert JsonPointer.from_parts(iter(["a", "b"])).path == "/a/b"


def test_from_parts_returns_subclass():
    class MyPointer(JsonPointer):
        pass

    assert type(MyPointer.from_parts(["a"])) is MyPointer


@pytest.mark.parametrize("parts", [[], ["a"], ["~0", "/1"], ["", "", "x"]])
def test_from_parts_roundtrip(parts):
    assert JsonPointer.from_parts(parts).parts == parts


# --- equality / hashing ----------------------------------------------------

def test_equality():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("") != JsonPointer("/")
    assert JsonPointer("/~1") == JsonPointer.from_parts(["/"])


def test_not_equal_to_non_pointer():
    assert JsonPointer("/a") != "/a"
    assert JsonPointer("/a") != ["a"]
    assert not (JsonPointer("/a") == "/a")
    assert JsonPointer("") != None  # noqa: E711


def test_hash():
    a = JsonPointer("/a/b")
    b = JsonPointer("/a/b")
    assert hash(a) == hash(b)
    assert len({a, b, JsonPointer("/a")}) == 2
    d = {a: 1}
    assert d[b] == 1


# --- resolve ---------------------------------------------------------------

def test_resolve_nested(doc):
    assert resolve_pointer(doc, "/foo") is doc["foo"]
    assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"
    assert resolve_pointer(doc, "/foo/anArray/0") == {"prop": 44}
    assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44
    assert resolve_pointer(doc, "/nested/1/1/0") == 4
    assert resolve_pointer(doc, "/list/2") == 30
    assert resolve_pointer(doc, "/none") is None


def test_resolve_does_not_url_decode():
    d = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(d, "/a%20b") == 1
    assert resolve_pointer(d, "/c d") == 2
    assert resolve_pointer(d, "/a b", None) is None
    assert resolve_pointer(d, "/c%20d", None) is None


def test_resolve_escaped_keys():
    d = {"~1": "tilde-one", "/": "slash", "~": "tilde", "/0": "slash-zero"}
    assert resolve_pointer(d, "/~01") == "tilde-one"
    assert resolve_pointer(d, "/~1") == "slash"
    assert resolve_pointer(d, "/~0") == "tilde"
    assert resolve_pointer(d, "/~10") == "slash-zero"


def test_resolve_missing_key_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/missing")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/foo/missing/deeper")


def test_resolve_default(doc):
    assert resolve_pointer(doc, "/missing", None) is None
    assert resolve_pointer(doc, "/missing", 5) == 5
    sentinel = object()
    assert resolve_pointer(doc, "/foo/missing/deeper", sentinel) is sentinel
    assert resolve_pointer(doc, "/list/99", sentinel) is sentinel
    assert resolve_pointer(doc, "/list/x", sentinel) is sentinel
    assert resolve_pointer(doc, "/num/x", sentinel) is sentinel
    assert resolve_pointer(doc, "/str/0", sentinel) is sentinel
    assert JsonPointer("/missing").resolve(doc, "d") == "d"
    assert JsonPointer("/missing").get(doc, "d") == "d"


def test_resolve_default_not_used_when_found(doc):
    assert resolve_pointer(doc, "/num", "default") == 42
    assert resolve_pointer(doc, "/none", "default") is None


def test_resolve_default_falsy_values(doc):
    assert resolve_pointer(doc, "/missing", 0) == 0
    assert resolve_pointer(doc, "/missing", False) is False
    assert resolve_pointer(doc, "/missing", "") == ""


@pytest.mark.parametrize("index", ["01", "00", "-1", "+1", "1.0", " 1", "1 ", "a", "1e2", "0x1", ""])
def test_invalid_list_index(index):
    d = {"l": list(range(20))}
    with pytest.raises(JsonPointerException):
        resolve_pointer(d, "/l/" + index)


@pytest.mark.parametrize("index", ["١", "１"])
def test_non_ascii_digits_rejected(index):
    d = {"l": list(range(20))}
    with pytest.raises(JsonPointerException):
        resolve_pointer(d, "/l/" + index)


def test_valid_list_indices():
    lst = list(range(20))
    assert resolve_pointer(lst, "/0") == 0
    assert resolve_pointer(lst, "/9") == 9
    assert resolve_pointer(lst, "/10") == 10
    assert resolve_pointer(lst, "/19") == 19


def test_list_index_out_of_bounds(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/list/3")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")


def test_resolve_end_of_list(doc):
    result = resolve_pointer(doc, "/list/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is doc["list"]


def test_resolve_past_end_of_list_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/list/-/0")
    assert resolve_pointer(doc, "/list/-/0", None) is None


def test_dash_is_regular_key_in_mapping():
    assert resolve_pointer({"-": 1}, "/-") == 1


def test_resolve_into_string_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/str/0")


def test_resolve_into_scalar_raises(doc):
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/num/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer(doc, "/none/x")
    with pytest.raises(JsonPointerException):
        resolve_pointer(5, "/0")


def test_mapping_keys_are_strings():
    d = {1: "int-key", "1": "str-key"}
    assert resolve_pointer(d, "/1") == "str-key"
    with pytest.raises(JsonPointerException):
        resolve_pointer({1: "x"}, "/1")


def test_resolve_tuple_sequence():
    t = ("a", ("b", "c"))
    assert resolve_pointer(t, "/1/0") == "b"
    with pytest.raises(JsonPointerException):
        resolve_pointer(t, "/2")


def test_resolve_ordered_dict():
    d = OrderedDict([("a", OrderedDict([("b", 1)]))])
    assert resolve_pointer(d, "/a/b") == 1


def test_resolve_custom_getitem():
    class Thing:
        def __getitem__(self, key):
            if key == "known":
                return "value"
            raise KeyError(key)

    assert resolve_pointer({"t": Thing()}, "/t/known") == "value"
    with pytest.raises(JsonPointerException):
        resolve_pointer({"t": Thing()}, "/t/unknown")
    assert resolve_pointer({"t": Thing()}, "/t/unknown", "dflt") == "dflt"


def test_resolve_custom_getitem_receives_string():
    seen = []

    class Thing:
        def __getitem__(self, key):
            seen.append(key)
            return key

    assert resolve_pointer(Thing(), "/0") == "0"
    assert seen == ["0"]


def test_resolve_does_not_modify_doc(doc):
    before = copy.deepcopy(doc)
    resolve_pointer(doc, "/foo/anArray/0/prop")
    resolve_pointer(doc, "/missing", None)
    resolve_pointer(doc, "/list/-")
    assert doc == before


# --- verbose exceptions ----------------------------------------------------

def test_verbose_exceptions(monkeypatch):
    d = {"secretcontent": 1}
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer(d, "/missing")
    assert "missing" in str(exc.value)
    assert "secretcontent" in str(exc.value)

    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer(d, "/missing")
    assert "missing" in str(exc.value)
    assert "secretcontent" not in str(exc.value)


def test_out_of_bounds_message_mentions_index():
    with pytest.raises(JsonPointerException) as exc:
        resolve_pointer([1, 2], "/5")
    assert "5" in str(exc.value)


def test_invalid_escape_message():
    with pytest.raises(JsonPointerException) as exc:
        JsonPointer("/a~2")
    assert "~2" in str(exc.value)


# --- get_part / walk / to_last ---------------------------------------------

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
        JsonPointer.get_part(42, "0")


def test_walk():
    ptr = JsonPointer("")
    assert ptr.walk({"a": 1}, "a") == 1
    assert ptr.walk([5, 6], "1") == 6
    eol = ptr.walk([5, 6], "-")
    assert isinstance(eol, EndOfList)
    with pytest.raises(JsonPointerException):
        ptr.walk({"a": 1}, "b")
    with pytest.raises(JsonPointerException):
        ptr.walk([5, 6], "2")


def test_to_last(doc):
    parent, last = JsonPointer("/foo/anArray/0").to_last(doc)
    assert parent is doc["foo"]["anArray"]
    assert last == 0

    parent, last = JsonPointer("/foo/another prop").to_last(doc)
    assert parent is doc["foo"]
    assert last == "another prop"

    parent, last = JsonPointer("/list/-").to_last(doc)
    assert parent is doc["list"]
    assert last == "-"

    parent, last = JsonPointer("").to_last(doc)
    assert parent is doc
    assert last is None


def test_to_last_invalid_index_raises(doc):
    with pytest.raises(JsonPointerException):
        JsonPointer("/list/01").to_last(doc)


# --- set -------------------------------------------------------------------

def test_set_pointer_inplace(doc):
    result = set_pointer(doc, "/foo/anArray/0/prop", 55)
    assert result is doc
    assert doc["foo"]["anArray"][0]["prop"] == 55


def test_set_new_key(doc):
    set_pointer(doc, "/foo/yet another prop", "added")
    assert doc["foo"]["yet another prop"] == "added"
    assert doc["foo"]["another prop"] == {"baz": "A string"}


def test_set_escaped_key():
    d = {}
    set_pointer(d, "/a~1b", 1)
    set_pointer(d, "/c~0d", 2)
    set_pointer(d, "/~01", 3)
    set_pointer(d, "/a%20b", 4)
    assert d == {"a/b": 1, "c~d": 2, "~1": 3, "a%20b": 4}


def test_set_empty_key():
    d = {}
    set_pointer(d, "/", "x")
    assert d == {"": "x"}


def test_set_list_element(doc):
    set_pointer(doc, "/list/1", 99)
    assert doc["list"] == [10, 99, 30]


def test_set_list_element_type_is_int_index(doc):
    set_pointer(doc, "/list/0", "z")
    assert doc["list"] == ["z", 20, 30]
    assert len(doc["list"]) == 3


def test_set_append_with_dash(doc):
    set_pointer(doc, "/list/-", 40)
    assert doc["list"] == [10, 20, 30, 40]
    set_pointer(doc, "/nested/1/1/-", 6)
    assert doc["nested"] == [[1, 2], [3, [4, 5, 6]]]


def test_set_append_to_empty_list():
    d = {"l": []}
    set_pointer(d, "/l/-", 1)
    assert d == {"l": [1]}


def test_set_dash_on_mapping_sets_key():
    d = {}
    set_pointer(d, "/-", 1)
    assert d == {"-": 1}


def test_set_out_of_bounds_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/list/3", 1)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/list/100", 1)
    assert doc["list"] == [10, 20, 30]


def test_set_invalid_index_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/list/01", 1)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/list/-1", 1)
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/list/a", 1)
    assert doc["list"] == [10, 20, 30]


def test_set_in_string_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/str/0", "x")
    assert doc["str"] == "hello"


def test_set_in_scalar_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/num/0", "x")


def test_set_in_tuple_raises():
    d = {"t": (1, 2)}
    with pytest.raises(JsonPointerException):
        set_pointer(d, "/t/0", 5)


def test_set_missing_intermediate_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/missing/key", 1)
    assert "missing" not in doc


def test_set_root_inplace_raises(doc):
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "", 1)
    with pytest.raises(JsonPointerException):
        JsonPointer("").set(doc, 1)


def test_set_root_not_inplace_returns_value(doc):
    before = copy.deepcopy(doc)
    assert set_pointer(doc, "", 1, inplace=False) == 1
    new = {"x": 1}
    assert JsonPointer("").set(doc, new, inplace=False) is new
    assert doc == before


def test_set_not_inplace_does_not_modify(doc):
    before = copy.deepcopy(doc)
    result = set_pointer(doc, "/foo/anArray/0/prop", 55, inplace=False)
    assert doc == before
    assert result is not doc
    assert result["foo"]["anArray"][0]["prop"] == 55
    assert result["foo"] is not doc["foo"]
    assert result["foo"]["anArray"] is not doc["foo"]["anArray"]

    result = set_pointer(doc, "/list/-", 40, inplace=False)
    assert doc == before
    assert result["list"] == [10, 20, 30, 40]


def test_set_method(doc):
    ptr = JsonPointer("/foo/another prop/baz")
    assert ptr.set(doc, "new") is doc
    assert doc["foo"]["another prop"]["baz"] == "new"


def test_set_then_resolve(doc):
    set_pointer(doc, "/foo/a~1b", [1])
    assert resolve_pointer(doc, "/foo/a~1b/0") == 1


def test_set_custom_setitem():
    class Store:
        def __init__(self):
            self.data = {}

        def __getitem__(self, k):
            return self.data[k]

        def __setitem__(self, k, v):
            self.data[k] = v

    s = Store()
    set_pointer({"s": s}, "/s/k", 1)
    assert s.data == {"k": 1}


# --- contains --------------------------------------------------------------

def test_contains():
    p1 = JsonPointer("/a/b/c")
    p2 = JsonPointer("/a/b")
    p3 = JsonPointer("/a/c")
    root = JsonPointer("")
    assert p1.contains(p2)
    assert p1.contains(p1)
    assert p1.contains(root)
    assert not p2.contains(p1)
    assert not p1.contains(p3)
    assert not root.contains(p2)
    assert p2 in p1
    assert p1 not in p2
    assert p3 not in p1


def test_contains_is_part_based_not_string_prefix():
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))
    assert not JsonPointer("/a/bc").contains(JsonPointer("/a/b"))
    assert JsonPointer("/a/b").contains(JsonPointer("/a"))


# --- join ------------------------------------------------------------------

def test_join_string():
    assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")
    assert JsonPointer("/a").join("") == JsonPointer("/a")
    assert JsonPointer("").join("/a") == JsonPointer("/a")
    assert JsonPointer("/a").join("/~1").parts == ["a", "/"]


def test_join_pointer():
    assert JsonPointer("/a").join(JsonPointer("/b")) == JsonPointer("/a/b")
    assert JsonPointer("/a").join(JsonPointer("/~0")).parts == ["a", "~"]


def test_join_list():
    assert JsonPointer("/a").join(["b", "c"]) == JsonPointer("/a/b/c")
    assert JsonPointer("/a").join(["b/c"]).parts == ["a", "b/c"]
    assert JsonPointer("/a").join(["~"]).path == "/a/~0"
    assert JsonPointer("/a").join([0, 1]).parts == ["a", "0", "1"]
    assert JsonPointer("/a").join([]) == JsonPointer("/a")


def test_join_preserves_escaping_of_self():
    ptr = JsonPointer("/a~1b/c~0d").join("/e")
    assert ptr.parts == ["a/b", "c~d", "e"]
    assert ptr.path == "/a~1b/c~0d/e"


def test_join_does_not_mutate():
    p = JsonPointer("/a")
    p.join("/b")
    p / ["c"]
    assert p.parts == ["a"]


def test_truediv():
    assert JsonPointer("/a") / "/b" == JsonPointer("/a/b")
    assert JsonPointer("/a") / ["b", "c"] == JsonPointer("/a/b/c")
    assert JsonPointer("") / JsonPointer("/x") / "/y" == JsonPointer("/x/y")


def test_join_invalid_suffix_raises():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("/~2")
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)
    with pytest.raises(JsonPointerException):
        JsonPointer("/a") / "b"


def test_join_returns_subclass():
    class MyPointer(JsonPointer):
        pass

    assert type(MyPointer("/a").join("/b")) is MyPointer


# --- pairwise / EndOfList --------------------------------------------------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


def test_end_of_list_repr():
    lst = [1, 2]
    eol = EndOfList(lst)
    assert eol.list_ is lst
    assert repr(eol) == "EndOfList([1, 2])"
