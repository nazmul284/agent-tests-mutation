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


@pytest.fixture(autouse=True)
def restore_verbose():
    saved = jsonpointer.VERBOSE_EXCEPTIONS
    yield
    jsonpointer.VERBOSE_EXCEPTIONS = saved


# --- module metadata / defaults ---------------------------------------------

def test_verbose_exceptions_default_true():
    assert jsonpointer.VERBOSE_EXCEPTIONS is True


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


# --- escape / unescape ------------------------------------------------------

@pytest.mark.parametrize("raw,escaped", [
    ("", ""),
    ("abc", "abc"),
    ("~", "~0"),
    ("/", "~1"),
    ("~/", "~0~1"),
    ("/~", "~1~0"),
    ("~1", "~01"),
    ("~0", "~00"),
    ("a/b~c", "a~1b~0c"),
])
def test_escape(raw, escaped):
    assert escape(raw) == escaped
    assert unescape(escaped) == raw


def test_unescape_order_matters():
    # "~01" must become "~1", not "/"
    assert unescape("~01") == "~1"
    assert unescape("~10") == "/0"


# --- pairwise ---------------------------------------------------------------

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("ptr,parts", [
    ("", []),
    ("/", [""]),
    ("//", ["", ""]),
    ("/foo", ["foo"]),
    ("/foo/0", ["foo", "0"]),
    ("/a~1b", ["a/b"]),
    ("/m~0n", ["m~n"]),
    ("/~01", ["~1"]),
    ("/~10", ["/0"]),
    ("/foo/", ["foo", ""]),
    ("/ ", [" "]),
])
def test_parse_parts(ptr, parts):
    p = JsonPointer(ptr)
    assert p.parts == parts
    assert p.get_parts() == parts


@pytest.mark.parametrize("ptr", ["foo", "a/b", " /a", "0"])
def test_must_start_with_slash(ptr):
    with pytest.raises(JsonPointerException) as e:
        JsonPointer(ptr)
    assert str(e.value) == "Location must start with /"


@pytest.mark.parametrize("ptr,bad", [
    ("/~", "~"),
    ("/a~", "~"),
    ("/~2", "~2"),
    ("/~a/b", "~a"),
    ("/ok/~~0", "~~"),
    ("/~0/~x", "~x"),
])
def test_invalid_escape(ptr, bad):
    with pytest.raises(JsonPointerException) as e:
        JsonPointer(ptr)
    assert str(e.value) == "Found invalid escape {}".format(bad)


def test_invalid_escape_checked_before_slash():
    with pytest.raises(JsonPointerException) as e:
        JsonPointer("a~")
    assert "invalid escape" in str(e.value)


# --- resolve: RFC 6901 examples --------------------------------------------

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


def test_resolve_root_returns_same_object():
    doc = {"a": 1}
    assert resolve_pointer(doc, "") is doc


def test_resolve_returns_identity_of_subobject():
    inner = {"x": [1]}
    doc = {"a": inner}
    assert resolve_pointer(doc, "/a") is inner


def test_resolve_nested():
    doc = {"foo": {"anArray": [{"prop": 44}], "another prop": {"baz": "A string"}}}
    assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44
    assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"


def test_no_percent_decoding():
    doc = {"a%20b": 1, "c d": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/c d") == 2
    assert resolve_pointer(doc, "/c%20d", None) is None


def test_resolve_list_root():
    doc = [10, 20, 30]
    assert resolve_pointer(doc, "/0") == 10
    assert resolve_pointer(doc, "/2") == 30


def test_resolve_large_index():
    doc = list(range(150))
    assert resolve_pointer(doc, "/10") == 10
    assert resolve_pointer(doc, "/149") == 149


def test_resolve_tuple():
    assert resolve_pointer((1, (2, 3)), "/1/0") == 2


def test_resolve_mapping_with_numeric_string_key():
    doc = {"0": "zero", "01": "leading"}
    assert resolve_pointer(doc, "/0") == "zero"
    assert resolve_pointer(doc, "/01") == "leading"


def test_resolve_mapping_with_dash_key():
    assert resolve_pointer({"-": 5}, "/-") == 5


def test_resolve_empty_key_in_nested():
    assert resolve_pointer({"a": {"": {"": 3}}}, "/a//") == 3


def test_resolve_none_value():
    assert resolve_pointer({"a": None}, "/a") is None
    assert resolve_pointer({"a": None}, "/a", "dflt") is None


def test_resolve_falsy_default_values():
    assert resolve_pointer({}, "/a", None) is None
    assert resolve_pointer({}, "/a", 0) == 0
    assert resolve_pointer({}, "/a", False) is False
    assert resolve_pointer({}, "/a", "") == ""


# --- resolve: errors -------------------------------------------------------

def test_missing_member_verbose():
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": 1}, "/b")
    assert str(e.value) == "member 'b' not found in {'a': 1}"


def test_missing_member_not_verbose():
    jsonpointer.VERBOSE_EXCEPTIONS = False
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": 1}, "/b")
    assert str(e.value) == "member 'b' not found"


def test_missing_member_default():
    assert resolve_pointer({"a": 1}, "/b", "d") == "d"
    assert resolve_pointer({"a": 1}, "/b/c/d", "d") == "d"


def test_missing_nested_member():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": {"b": 1}}, "/a/c")


def test_index_out_of_bounds():
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer([1, 2], "/2")
    assert str(e.value) == "index '2' is out of bounds"


def test_index_out_of_bounds_default():
    assert resolve_pointer([1, 2], "/5", "x") == "x"


@pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", "1.0", " 1", "1 ", "a", "", "1a", "0x1", "１"])
def test_invalid_array_index(idx):
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer([0, 1, 2], "/" + idx)
    assert str(e.value) == "'%s' is not a valid sequence index" % idx


@pytest.mark.parametrize("idx", ["01", "-1", "x"])
def test_invalid_array_index_default(idx):
    assert resolve_pointer([0, 1, 2], "/" + idx, "d") == "d"


def test_negative_index_does_not_wrap():
    assert resolve_pointer([1, 2, 3], "/-1", None) is None


def test_resolve_through_string_fails():
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": "hello"}, "/a/0")
    assert str(e.value) == "Cannot apply token '0' to non-container type %s" % str
    assert resolve_pointer({"a": "hello"}, "/a/0", "d") == "d"


@pytest.mark.parametrize("value", [1, 1.5, None, True])
def test_resolve_through_scalar_fails(value):
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer({"a": value}, "/a/b")
    assert str(e.value) == (
        "Document '%s' does not support indexing, "
        "must be mapping/sequence or support __getitem__" % type(value))
    assert resolve_pointer({"a": value}, "/a/b", "d") == "d"


def test_dash_resolves_to_end_of_list():
    lst = [1, 2]
    result = resolve_pointer({"a": lst}, "/a/-")
    assert isinstance(result, EndOfList)
    assert result.list_ is lst


def test_dash_on_list_root():
    lst = []
    result = JsonPointer("/-").resolve(lst)
    assert isinstance(result, EndOfList)
    assert result.list_ is lst


def test_end_of_list_repr():
    assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"

    class Sub(EndOfList):
        pass

    assert repr(Sub([])) == "Sub([])"


def test_walk_past_dash_fails():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": [1]}, "/a/-/0")


# --- duck-typed __getitem__ ------------------------------------------------

class Getter:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


def test_ducktyped_getitem():
    obj = Getter({"x": Getter({"y": 5})})
    assert resolve_pointer(obj, "/x/y") == 5


def test_ducktyped_getitem_keyerror():
    obj = Getter({"x": 1})
    with pytest.raises(JsonPointerException) as e:
        resolve_pointer(obj, "/nope")
    assert str(e.value).startswith("member 'nope' not found")
    assert resolve_pointer(obj, "/nope", 7) == 7


def test_ducktyped_part_is_string():
    seen = []

    class Rec:
        def __getitem__(self, key):
            seen.append(key)
            return key

    assert resolve_pointer(Rec(), "/0") == "0"
    assert seen == ["0"]


def test_mapping_subclass():
    ud = UserDict({"a": {"b": 2}})
    assert resolve_pointer(ud, "/a/b") == 2
    with pytest.raises(JsonPointerException):
        resolve_pointer(ud, "/zz")


# --- get_part ---------------------------------------------------------------

def test_get_part():
    assert JsonPointer.get_part({}, "0") == "0"
    assert JsonPointer.get_part([1], "0") == 0
    assert JsonPointer.get_part([1], "12") == 12
    assert isinstance(JsonPointer.get_part([1], "12"), int)
    assert JsonPointer.get_part([1], "-") == "-"
    assert JsonPointer.get_part([1], 3) == 3
    assert JsonPointer.get_part(Getter({}), "k") == "k"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("abc", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(5, "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([1], "01")


def test_walk():
    p = JsonPointer("")
    assert p.walk({"a": 1}, "a") == 1
    assert p.walk([5, 6], "1") == 6
    assert isinstance(p.walk([5, 6], "-"), EndOfList)


# --- to_last ---------------------------------------------------------------

def test_to_last():
    doc = {"a": {"b": [1, 2]}}
    parent, part = JsonPointer("/a/b/1").to_last(doc)
    assert parent is doc["a"]["b"]
    assert part == 1
    parent, part = JsonPointer("/a/b").to_last(doc)
    assert parent is doc["a"]
    assert part == "b"


def test_to_last_root():
    doc = {"a": 1}
    parent, part = JsonPointer("").to_last(doc)
    assert parent is doc
    assert part is None


def test_to_last_dash():
    doc = [1]
    parent, part = JsonPointer("/-").to_last(doc)
    assert parent is doc
    assert part == "-"


# --- set -------------------------------------------------------------------

def test_set_inplace_existing():
    doc = {"foo": {"anArray": [{"prop": 44}]}}
    result = set_pointer(doc, "/foo/anArray/0/prop", 55)
    assert result is doc
    assert doc == {"foo": {"anArray": [{"prop": 55}]}}


def test_set_inplace_default():
    doc = {"a": 1}
    JsonPointer("/a").set(doc, 2)
    assert doc == {"a": 2}


def test_set_new_key():
    doc = {"foo": {}}
    set_pointer(doc, "/foo/a%20b", "x")
    assert doc == {"foo": {"a%20b": "x"}}


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b~0c", 1)
    assert doc == {"a/b~c": 1}


def test_set_not_inplace():
    doc = {"a": {"b": [1, 2]}}
    original = copy.deepcopy(doc)
    result = set_pointer(doc, "/a/b/0", 9, inplace=False)
    assert doc == original
    assert result == {"a": {"b": [9, 2]}}
    assert result is not doc
    assert result["a"] is not doc["a"]


def test_set_not_inplace_method():
    doc = {"a": 1}
    result = JsonPointer("/a").set(doc, 2, inplace=False)
    assert doc == {"a": 1}
    assert result == {"a": 2}


def test_set_root_inplace_raises():
    with pytest.raises(JsonPointerException) as e:
        set_pointer({"a": 1}, "", 5)
    assert str(e.value) == "Cannot set root in place"
    with pytest.raises(JsonPointerException):
        JsonPointer("").set({}, 5)


def test_set_root_not_inplace_returns_value():
    doc = {"a": 1}
    value = {"b": 2}
    assert set_pointer(doc, "", value, inplace=False) is value
    assert doc == {"a": 1}


def test_set_list_index():
    doc = [1, 2, 3]
    set_pointer(doc, "/1", "x")
    assert doc == [1, "x", 3]


def test_set_list_append_dash():
    doc = {"a": [1, 2]}
    result = set_pointer(doc, "/a/-", 3)
    assert result is doc
    assert doc == {"a": [1, 2, 3]}


def test_set_list_append_dash_not_inplace():
    doc = {"a": [1]}
    result = set_pointer(doc, "/a/-", 2, inplace=False)
    assert doc == {"a": [1]}
    assert result == {"a": [1, 2]}


def test_set_dash_on_mapping_is_key():
    doc = {}
    set_pointer(doc, "/-", 1)
    assert doc == {"-": 1}


def test_set_list_out_of_bounds():
    with pytest.raises(JsonPointerException) as e:
        set_pointer([1, 2], "/5", 0)
    assert str(e.value).startswith("Invalid assignment target: ")
    assert "index out of range" in str(e.value)


def test_set_list_end_index_is_error():
    doc = [1, 2]
    with pytest.raises(JsonPointerException):
        set_pointer(doc, "/2", 3)
    assert doc == [1, 2]


def test_set_tuple_raises():
    with pytest.raises(JsonPointerException) as e:
        set_pointer({"a": (1, 2)}, "/a/0", 5)
    assert str(e.value).startswith("Invalid assignment target: ")


def test_set_in_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "str"}, "/a/0", "x")


def test_set_in_string_direct_parent():
    # to_last on a str parent raises via get_part
    with pytest.raises(JsonPointerException):
        set_pointer("abc", "/0", "x")


def test_set_invalid_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 0)


def test_set_missing_intermediate():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": {}}, "/a/b/c", 1)


def test_set_ducktyped_setitem():
    class Store:
        def __init__(self):
            self.d = {}

        def __getitem__(self, k):
            return self.d[k]

        def __setitem__(self, k, v):
            self.d[k] = v

    s = Store()
    set_pointer(s, "/k", 3)
    assert s.d == {"k": 3}


def test_set_ducktyped_no_setitem():
    with pytest.raises(JsonPointerException) as e:
        set_pointer(Getter({}), "/k", 3)
    assert str(e.value).startswith("Invalid assignment target: ")


def test_set_returns_doc_after_nested_set():
    doc = {"a": [{"b": 1}]}
    assert set_pointer(doc, "/a/0/b", 2) is doc


# --- contains --------------------------------------------------------------

def test_contains():
    p = JsonPointer("/a/b/c")
    assert p.contains(JsonPointer("/a/b"))
    assert p.contains(JsonPointer("/a/b/c"))
    assert p.contains(JsonPointer(""))
    assert not p.contains(JsonPointer("/a/x"))
    assert not p.contains(JsonPointer("/a/b/c/d"))
    assert not p.contains(JsonPointer("/b"))
    assert JsonPointer("/a/b") in p
    assert JsonPointer("/a/c") not in p
    assert JsonPointer("/a/b/c/d") not in p


def test_contains_partial_token_not_prefix():
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))


# --- join / truediv --------------------------------------------------------

def test_join_string():
    p = JsonPointer("/a").join("/b/c")
    assert isinstance(p, JsonPointer)
    assert p.parts == ["a", "b", "c"]
    assert p.path == "/a/b/c"


def test_join_pointer():
    assert JsonPointer("/a").join(JsonPointer("/b~1c")).parts == ["a", "b/c"]


def test_join_list():
    assert JsonPointer("/a").join(["b/c", "~"]).path == "/a/b~1c/~0"
    assert JsonPointer("/a").join([0, 1]).parts == ["a", "0", "1"]


def test_join_empty():
    assert JsonPointer("/a").join("").parts == ["a"]
    assert JsonPointer("").join("/x").parts == ["x"]
    assert JsonPointer("/a").join([]).parts == ["a"]


def test_join_does_not_mutate():
    p = JsonPointer("/a")
    p.join("/b")
    assert p.parts == ["a"]


def test_join_invalid_string():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")


def test_join_invalid_suffix_type():
    with pytest.raises(JsonPointerException) as e:
        JsonPointer("/a").join(5)
    assert str(e.value) == "Invalid suffix"


def test_truediv():
    p = JsonPointer("/a") / "/b"
    assert p == JsonPointer("/a/b")
    assert (JsonPointer("/a") / ["c"]).path == "/a/c"
    assert (JsonPointer("/a") / JsonPointer("/d")).path == "/a/d"
    with pytest.raises(JsonPointerException):
        JsonPointer("/a") / 1


def test_join_preserves_subclass():
    class MyPointer(JsonPointer):
        pass

    p = MyPointer("/a").join("/b")
    assert type(p) is MyPointer
    assert type(MyPointer.from_parts(["x"])) is MyPointer


# --- path / str / repr ----------------------------------------------------

@pytest.mark.parametrize("ptr", ["", "/", "//", "/a", "/~0/0/~1", "/a~1b/m~0n", "/~01", "/foo/"])
def test_path_roundtrip(ptr):
    p = JsonPointer(ptr)
    assert p.path == ptr
    assert str(p) == ptr


def test_repr():
    assert repr(JsonPointer("/a/~1")) == "JsonPointer('/a/~1')"
    assert repr(JsonPointer("")) == "JsonPointer('')"

    class Sub(JsonPointer):
        pass

    assert repr(Sub("/x")) == "Sub('/x')"


# --- from_parts -----------------------------------------------------------

def test_from_parts():
    p = JsonPointer.from_parts(["a", "~", "/", 0])
    assert p.path == "/a/~0/~1/0"
    assert p.parts == ["a", "~", "/", "0"]


def test_from_parts_empty():
    p = JsonPointer.from_parts([])
    assert p.path == ""
    assert p.parts == []


def test_from_parts_empty_string_part():
    assert JsonPointer.from_parts([""]).path == "/"


def test_from_parts_generator():
    assert JsonPointer.from_parts(x for x in ["a", "b"]).parts == ["a", "b"]


def test_from_parts_roundtrip_resolve():
    doc = {"a/b": {"~c": [0, {"x": 1}]}}
    p = JsonPointer.from_parts(["a/b", "~c", 1, "x"])
    assert p.resolve(doc) == 1


# --- equality / hash ----------------------------------------------------

def test_equality():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a~1b") == JsonPointer.from_parts(["a/b"])
    assert JsonPointer("/a") != JsonPointer("/b")
    assert JsonPointer("") != JsonPointer("/")
    assert not (JsonPointer("/a") == JsonPointer("/a/b"))


def test_equality_with_other_types():
    assert JsonPointer("/a") != "/a"
    assert not (JsonPointer("/a") == "/a")
    assert not (JsonPointer("") == [])
    assert JsonPointer("/a") != ["a"]


def test_hash():
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert hash(JsonPointer("/a/b")) == hash(("a", "b"))
    s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
    assert len(s) == 2
    d = {JsonPointer("/x"): 1}
    assert d[JsonPointer.from_parts(["x"])] == 1


# --- functional wrappers -------------------------------------------------

def test_resolve_pointer_raises_without_default():
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, "/a")


def test_resolve_pointer_invalid_pointer_not_swallowed_by_default():
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, "a", "default")


def test_set_pointer_default_is_inplace():
    doc = {"a": 1}
    set_pointer(doc, "/a", 2)
    assert doc["a"] == 2
