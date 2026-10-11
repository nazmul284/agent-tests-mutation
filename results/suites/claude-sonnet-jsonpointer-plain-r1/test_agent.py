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
    assert JsonPointer(ptr).resolve(RFC_DOC) == expected
    assert JsonPointer(ptr).get(RFC_DOC) == expected


# ---- escape / unescape ----

def test_escape_basic():
    assert escape("a/b") == "a~1b"
    assert escape("a~b") == "a~0b"
    assert escape("~/") == "~0~1"
    assert escape("~1") == "~01"
    assert escape("plain") == "plain"
    assert escape("") == ""


def test_unescape_basic():
    assert unescape("a~1b") == "a/b"
    assert unescape("a~0b") == "a~b"
    assert unescape("~01") == "~1"
    assert unescape("~0~1") == "~/"
    assert unescape("~10") == "/0"


@pytest.mark.parametrize("s", ["", "a", "~", "/", "~1", "~0", "~01", "~/~", "//~~", "a~1/b~0"])
def test_escape_roundtrip(s):
    assert unescape(escape(s)) == s


# ---- construction ----

def test_empty_pointer_has_no_parts():
    assert JsonPointer("").parts == []
    assert JsonPointer("").path == ""


def test_parts_parsed():
    assert JsonPointer("/a/b/c").parts == ["a", "b", "c"]
    assert JsonPointer("/").parts == [""]
    assert JsonPointer("//").parts == ["", ""]
    assert JsonPointer("/a/").parts == ["a", ""]
    assert JsonPointer("/a~1b/c~0d").parts == ["a/b", "c~d"]
    assert JsonPointer("/~01").parts == ["~1"]
    assert JsonPointer("/~00").parts == ["~0"]


def test_get_parts():
    assert JsonPointer("/a/0/b").get_parts() == ["a", "0", "b"]


@pytest.mark.parametrize("bad", ["a", "a/b", "foo", " /a", "#/a", "0"])
def test_must_start_with_slash(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


@pytest.mark.parametrize("bad", ["/~", "/a~", "/~2", "/~a", "/a~b/c", "/a/~", "/~ ", "/a~/b", "/~-1", "/x/~9"])
def test_invalid_escapes(bad):
    with pytest.raises(JsonPointerException):
        JsonPointer(bad)


def test_invalid_escape_at_end_of_string_only_tilde():
    with pytest.raises(JsonPointerException):
        JsonPointer("/abc~")


def test_valid_escape_not_rejected():
    JsonPointer("/~0")
    JsonPointer("/~1")
    JsonPointer("/~0~1~0")


def test_invalid_escape_error_message_mentions_escape():
    with pytest.raises(JsonPointerException, match="~x"):
        JsonPointer("/a~x")


# ---- path / str / repr ----

@pytest.mark.parametrize("p", ["", "/", "//", "/a", "/a/b", "/a~1b", "/a~0b", "/~01", "/~00/~10", "/0/1", "/ "])
def test_path_roundtrip(p):
    ptr = JsonPointer(p)
    assert ptr.path == p
    assert str(ptr) == p


def test_repr():
    assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
    assert repr(JsonPointer("")) == "JsonPointer('')"


def test_repr_subclass_name():
    class Sub(JsonPointer):
        pass
    assert repr(Sub("/a")) == "Sub('/a')"


# ---- resolve: dicts ----

def test_resolve_nested_dict():
    doc = {"a": {"b": {"c": 5}}}
    assert resolve_pointer(doc, "/a/b/c") == 5
    assert resolve_pointer(doc, "/a/b") == {"c": 5}
    assert resolve_pointer(doc, "") is doc


def test_resolve_returns_same_object():
    inner = {"x": 1}
    doc = {"a": inner}
    assert resolve_pointer(doc, "/a") is inner


def test_resolve_missing_key_raises():
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": 1}, "/b")


def test_resolve_empty_key():
    assert resolve_pointer({"": {"": 3}}, "//") == 3


def test_resolve_none_value_present():
    assert resolve_pointer({"a": None}, "/a", "dflt") is None


def test_resolve_falsey_values():
    doc = {"a": 0, "b": "", "c": [], "d": False}
    assert resolve_pointer(doc, "/a", "d") == 0
    assert resolve_pointer(doc, "/b", "d") == ""
    assert resolve_pointer(doc, "/c", "d") == []
    assert resolve_pointer(doc, "/d", "d") is False


def test_dict_numeric_string_keys():
    doc = {"0": "zero", "01": "oh-one", "-": "dash"}
    assert resolve_pointer(doc, "/0") == "zero"
    assert resolve_pointer(doc, "/01") == "oh-one"
    assert resolve_pointer(doc, "/-") == "dash"


def test_percent_not_decoded():
    doc = {"a%20b": 1, "a b": 2}
    assert resolve_pointer(doc, "/a%20b") == 1
    assert resolve_pointer(doc, "/a b") == 2


def test_resolve_escaped_keys():
    doc = {"a/b": 1, "m~n": 2, "~1": 3, "/": 4, "~": 5}
    assert resolve_pointer(doc, "/a~1b") == 1
    assert resolve_pointer(doc, "/m~0n") == 2
    assert resolve_pointer(doc, "/~01") == 3
    assert resolve_pointer(doc, "/~1") == 4
    assert resolve_pointer(doc, "/~0") == 5


# ---- resolve: lists ----

def test_resolve_list_index():
    doc = {"a": [10, 20, 30]}
    assert resolve_pointer(doc, "/a/0") == 10
    assert resolve_pointer(doc, "/a/1") == 20
    assert resolve_pointer(doc, "/a/2") == 30


def test_resolve_list_multi_digit_index():
    doc = list(range(100))
    assert resolve_pointer(doc, "/10") == 10
    assert resolve_pointer(doc, "/99") == 99
    assert resolve_pointer(doc, "/20") == 20


def test_resolve_root_list():
    assert resolve_pointer([1, 2], "/1") == 2


def test_resolve_list_out_of_bounds():
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2], "/2")
    with pytest.raises(JsonPointerException):
        resolve_pointer([], "/0")


@pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", "1.0", " 1", "1 ", "a", "", "1e0", "0x1", "١"])
def test_resolve_list_invalid_index(idx):
    with pytest.raises(JsonPointerException):
        resolve_pointer([1, 2, 3], "/" + idx)


def test_list_index_with_trailing_garbage():
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/1a")
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/10x")
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/0a")


def test_list_index_leading_garbage():
    with pytest.raises(JsonPointerException):
        resolve_pointer(list(range(20)), "/a1")


def test_resolve_tuple():
    assert resolve_pointer(((1, 2), (3, 4)), "/1/0") == 3


def test_resolve_dash_gives_end_of_list():
    lst = [1, 2]
    res = resolve_pointer({"a": lst}, "/a/-")
    assert isinstance(res, EndOfList)
    assert res.list_ is lst


def test_end_of_list_repr():
    assert repr(EndOfList([1])) == "EndOfList([1])"


def test_resolve_through_dash_fails():
    with pytest.raises((JsonPointerException, TypeError)):
        resolve_pointer({"a": [1]}, "/a/-/b")


def test_resolve_mixed():
    doc = {"a": [{"b": [None, {"c": "deep"}]}]}
    assert resolve_pointer(doc, "/a/0/b/1/c") == "deep"


# ---- default ----

def test_default_on_missing_key():
    assert resolve_pointer({"a": 1}, "/b", "d") == "d"
    assert resolve_pointer({"a": 1}, "/b", None) is None


def test_default_on_bad_index():
    assert resolve_pointer([1], "/5", "d") == "d"
    assert resolve_pointer([1], "/x", "d") == "d"
    assert resolve_pointer([1], "/01", "d") == "d"


def test_default_on_deep_missing():
    assert resolve_pointer({"a": {}}, "/a/b/c", "d") == "d"


def test_default_on_non_container():
    assert resolve_pointer({"a": "str"}, "/a/0", "d") == "d"
    assert resolve_pointer({"a": 5}, "/a/b", "d") == "d"
    assert resolve_pointer({"a": None}, "/a/b", "d") == "d"


def test_default_not_used_when_found():
    assert resolve_pointer({"a": 1}, "/a", "d") == 1


def test_default_on_root_returns_doc():
    assert resolve_pointer({"a": 1}, "", "d") == {"a": 1}


def test_default_method():
    assert JsonPointer("/x").resolve({}, 7) == 7
    assert JsonPointer("/x").get({}, 7) == 7


def test_invalid_pointer_raises_even_with_default():
    with pytest.raises(JsonPointerException):
        resolve_pointer({}, "a", "d")


# ---- non-container docs ----

def test_cannot_index_string():
    with pytest.raises(JsonPointerException):
        resolve_pointer("abc", "/0")
    with pytest.raises(JsonPointerException):
        resolve_pointer({"a": "abc"}, "/a/0")


def test_cannot_index_int_or_none():
    with pytest.raises(JsonPointerException):
        resolve_pointer(5, "/a")
    with pytest.raises(JsonPointerException):
        resolve_pointer(None, "/a")


def test_root_pointer_on_scalar():
    assert resolve_pointer(5, "") == 5
    assert resolve_pointer("s", "") == "s"
    assert resolve_pointer(None, "") is None


def test_duck_typed_getitem():
    class Duck:
        def __getitem__(self, key):
            if key == "k":
                return "v"
            raise KeyError(key)

    assert resolve_pointer(Duck(), "/k") == "v"
    with pytest.raises(JsonPointerException):
        resolve_pointer(Duck(), "/nope")


def test_duck_typed_keeps_string_key():
    seen = []

    class Duck:
        def __getitem__(self, key):
            seen.append(key)
            return 1

    resolve_pointer(Duck(), "/0")
    assert seen == ["0"]


def test_custom_mapping():
    from collections import OrderedDict, UserDict
    assert resolve_pointer(OrderedDict(a=1), "/a") == 1
    assert resolve_pointer(UserDict({"a": 2}), "/a") == 2


def test_custom_sequence():
    from collections import UserList
    assert resolve_pointer(UserList([5, 6]), "/1") == 6


# ---- verbose exceptions ----

def test_verbose_exceptions_include_doc(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"secret": 1}, "/missing")
    assert "missing" in str(ei.value)
    assert "secret" in str(ei.value)


def test_non_verbose_exceptions_hide_doc(monkeypatch):
    monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
    with pytest.raises(JsonPointerException) as ei:
        resolve_pointer({"secret": 1}, "/missing")
    assert "missing" in str(ei.value)
    assert "secret" not in str(ei.value)


# ---- to_last ----

def test_to_last():
    doc = {"a": {"b": 1}}
    parent, last = JsonPointer("/a/b").to_last(doc)
    assert parent is doc["a"]
    assert last == "b"


def test_to_last_list_converts_index():
    doc = {"a": [1, 2, 3]}
    parent, last = JsonPointer("/a/2").to_last(doc)
    assert parent is doc["a"]
    assert last == 2
    assert isinstance(last, int)


def test_to_last_dash():
    doc = [1]
    parent, last = JsonPointer("/-").to_last(doc)
    assert parent is doc
    assert last == "-"


def test_to_last_root():
    doc = {"a": 1}
    assert JsonPointer("").to_last(doc) == (doc, None)


def test_to_last_single_part():
    doc = {"a": 1}
    parent, last = JsonPointer("/a").to_last(doc)
    assert parent is doc and last == "a"


def test_to_last_does_not_require_last_to_exist():
    parent, last = JsonPointer("/a/new").to_last({"a": {}})
    assert parent == {} and last == "new"


def test_to_last_bad_list_index():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a/x").to_last({"a": [1]})


def test_to_last_missing_intermediate():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a/b/c").to_last({"a": {}})


# ---- get_part ----

def test_get_part():
    assert JsonPointer.get_part({}, "a") == "a"
    assert JsonPointer.get_part([], "3") == 3
    assert JsonPointer.get_part([], "0") == 0
    assert JsonPointer.get_part([], "-") == "-"
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part([], "03")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part("str", "0")
    with pytest.raises(JsonPointerException):
        JsonPointer.get_part(3, "0")


# ---- walk ----

def test_walk():
    ptr = JsonPointer("")
    assert ptr.walk({"a": 1}, "a") == 1
    assert ptr.walk([5, 6], "1") == 6
    assert isinstance(ptr.walk([5], "-"), EndOfList)
    with pytest.raises(JsonPointerException):
        ptr.walk({}, "a")
    with pytest.raises(JsonPointerException):
        ptr.walk([], "0")


# ---- set ----

def test_set_existing_key_inplace():
    doc = {"a": 1}
    res = set_pointer(doc, "/a", 2)
    assert doc == {"a": 2}
    assert res is doc


def test_set_new_key():
    doc = {"a": {}}
    set_pointer(doc, "/a/b", 1)
    assert doc == {"a": {"b": 1}}


def test_set_list_element():
    doc = {"a": [1, 2, 3]}
    set_pointer(doc, "/a/1", 99)
    assert doc == {"a": [1, 99, 3]}


def test_set_first_and_last_list_element():
    doc = [1, 2, 3]
    set_pointer(doc, "/0", "x")
    set_pointer(doc, "/2", "z")
    assert doc == ["x", 2, "z"]


def test_set_list_append_dash():
    doc = {"a": [1]}
    set_pointer(doc, "/a/-", 2)
    assert doc == {"a": [1, 2]}


def test_set_list_append_dash_root_list():
    doc = []
    set_pointer(doc, "/-", "x")
    set_pointer(doc, "/-", "y")
    assert doc == ["x", "y"]


def test_set_list_out_of_range():
    with pytest.raises(JsonPointerException):
        set_pointer([1], "/5", 0)
    with pytest.raises(JsonPointerException):
        set_pointer([], "/0", 0)


def test_set_list_bad_index():
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/01", 0)
    with pytest.raises(JsonPointerException):
        set_pointer([1, 2], "/x", 0)


def test_set_missing_intermediate():
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
    res = set_pointer(doc, "/a/b/0", 9, inplace=False)
    assert doc == orig
    assert res == {"a": {"b": [9, 2]}}
    assert res is not doc
    assert res["a"] is not doc["a"]


def test_set_not_inplace_append():
    doc = [1]
    res = set_pointer(doc, "/-", 2, inplace=False)
    assert doc == [1]
    assert res == [1, 2]


def test_set_inplace_default_true():
    doc = {}
    set_pointer(doc, "/a", 1)
    assert doc == {"a": 1}


def test_set_in_string_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": "abc"}, "/a/0", "x")


def test_set_in_string_root_raises():
    with pytest.raises(JsonPointerException):
        set_pointer("abc", "/0", "x")


def test_set_in_tuple_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": (1, 2)}, "/a/0", 5)


def test_set_in_tuple_dash_raises():
    with pytest.raises((JsonPointerException, AttributeError)):
        set_pointer({"a": (1, 2)}, "/a/-", 5)


def test_set_on_scalar_parent_raises():
    with pytest.raises(JsonPointerException):
        set_pointer({"a": 1}, "/a/b", 5)


def test_set_escaped_key():
    doc = {}
    set_pointer(doc, "/a~1b", 1)
    set_pointer(doc, "/m~0n", 2)
    assert doc == {"a/b": 1, "m~n": 2}


def test_set_empty_key():
    doc = {}
    set_pointer(doc, "/", 1)
    assert doc == {"": 1}


def test_set_value_can_be_none():
    doc = {"a": 1}
    set_pointer(doc, "/a", None)
    assert doc == {"a": None}


def test_set_then_resolve():
    doc = {"a": [{}]}
    set_pointer(doc, "/a/0/k", "v")
    assert resolve_pointer(doc, "/a/0/k") == "v"


def test_set_method():
    doc = {"a": 1}
    ptr = JsonPointer("/a")
    assert ptr.set(doc, 3) is doc
    assert doc["a"] == 3
    res = ptr.set(doc, 4, inplace=False)
    assert res == {"a": 4} and doc["a"] == 3


def test_set_invalid_pointer():
    with pytest.raises(JsonPointerException):
        set_pointer({}, "a", 1)


# ---- contains / in ----

def test_contains():
    assert JsonPointer("/a/b/c").contains(JsonPointer("/a/b"))
    assert JsonPointer("/a/b/c").contains(JsonPointer("/a"))
    assert JsonPointer("/a/b/c").contains(JsonPointer(""))
    assert JsonPointer("/a/b").contains(JsonPointer("/a/b"))


def test_not_contains():
    assert not JsonPointer("/a/b").contains(JsonPointer("/a/b/c"))
    assert not JsonPointer("/a/b").contains(JsonPointer("/b"))
    assert not JsonPointer("/a/b").contains(JsonPointer("/a/c"))
    assert not JsonPointer("").contains(JsonPointer("/a"))


def test_in_operator():
    assert (JsonPointer("/a") in JsonPointer("/a/b")) is True
    assert (JsonPointer("/a/b") in JsonPointer("/a")) is False


def test_contains_is_part_wise_not_prefix_string():
    assert not JsonPointer("/ab").contains(JsonPointer("/a"))
    assert not JsonPointer("/a/bc").contains(JsonPointer("/a/b"))


# ---- join / truediv ----

def test_join_str():
    assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")
    assert JsonPointer("").join("/b") == JsonPointer("/b")
    assert JsonPointer("/a").join("") == JsonPointer("/a")


def test_join_pointer():
    assert JsonPointer("/a").join(JsonPointer("/b")).path == "/a/b"


def test_join_list_of_parts_escapes():
    assert JsonPointer("/a").join(["b/c", "d~e"]).path == "/a/b~1c/d~0e"


def test_join_list_with_int():
    assert JsonPointer("/a").join([0]).path == "/a/0"


def test_join_does_not_mutate():
    base = JsonPointer("/a")
    base.join("/b")
    assert base.parts == ["a"]


def test_join_invalid_suffix_string():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join("b")


def test_join_invalid_suffix_type():
    with pytest.raises(JsonPointerException):
        JsonPointer("/a").join(5)


def test_truediv():
    assert (JsonPointer("/a") / "/b").path == "/a/b"
    assert (JsonPointer("/a") / JsonPointer("/b") / ["c"]).path == "/a/b/c"


def test_join_returns_subclass():
    class Sub(JsonPointer):
        pass
    assert isinstance(Sub("/a").join("/b"), Sub)


# ---- from_parts ----

def test_from_parts():
    assert JsonPointer.from_parts(["a", "~", "/", 0]).path == "/a/~0/~1/0"
    assert JsonPointer.from_parts([]).path == ""
    assert JsonPointer.from_parts([""]).path == "/"
    assert JsonPointer.from_parts(["a", "b"]).parts == ["a", "b"]
    assert JsonPointer.from_parts(["~1"]).parts == ["~1"]


def test_from_parts_accepts_iterator():
    assert JsonPointer.from_parts(iter(["a", "b"])).path == "/a/b"


def test_from_parts_roundtrip_parts():
    parts = ["a/b", "~", "~0", "~1", "", "x y", "%20"]
    assert JsonPointer.from_parts(parts).parts == parts


# ---- equality / hash ----

def test_equality():
    assert JsonPointer("/a/b") == JsonPointer("/a/b")
    assert JsonPointer("/a/b") != JsonPointer("/a/c")
    assert JsonPointer("/a") != JsonPointer("/a/b")
    assert JsonPointer("") == JsonPointer("")
    assert JsonPointer("/a~1b") == JsonPointer.from_parts(["a/b"])


def test_equality_other_types():
    assert JsonPointer("/a") != "/a"
    assert not (JsonPointer("/a") == "/a")
    assert JsonPointer("/a") != ["a"]
    assert JsonPointer("/a") != None  # noqa: E711


def test_hash():
    assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
    assert len({JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}) == 2
    assert {JsonPointer("/a/b"): 1}[JsonPointer("/a/b")] == 1


def test_hash_distinguishes_order():
    # not strictly required, but equal-parts-different-order must not be equal
    assert JsonPointer("/a/b") != JsonPointer("/b/a")


# ---- pairwise ----

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---- misc ----

def test_resolve_does_not_mutate_doc():
    doc = {"a": [1, {"b": 2}]}
    orig = copy.deepcopy(doc)
    resolve_pointer(doc, "/a/1/b")
    resolve_pointer(doc, "/a/-")
    assert doc == orig


def test_pointer_reusable():
    ptr = JsonPointer("/a")
    assert ptr.resolve({"a": 1}) == 1
    assert ptr.resolve({"a": 2}) == 2


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)


def test_unicode_keys():
    doc = {"é": 1, "日本": [2]}
    assert resolve_pointer(doc, "/é") == 1
    assert resolve_pointer(doc, "/日本/0") == 2
