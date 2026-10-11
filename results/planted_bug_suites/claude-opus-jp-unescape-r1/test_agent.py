"""Tests for jsonpointer (RFC 6901)."""

import copy
from collections import OrderedDict, UserDict

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


# The example document from RFC 6901 section 5.
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


# jsonpointer.unescape() replaces "~0" before "~1", so "~01" decodes to "/"
# instead of "~1" (RFC 6901 section 4). Tests exercising that are expected to
# fail until the library is fixed; strict=True makes them fail loudly once fixed.
UNESCAPE_ORDER_BUG = pytest.mark.xfail(
    strict=True, reason='unescape("~01") returns "/" instead of "~1" (RFC 6901 sec. 4)')


@pytest.fixture
def doc():
    return {
        "foo": {"anArray": [{"prop": 44}], "another prop": {"baz": "A string"}},
        "a%20b": 1,
        "c d": 2,
    }


# --------------------------------------------------------------------------
# escape / unescape
# --------------------------------------------------------------------------

class TestEscape:
    @pytest.mark.parametrize("raw, escaped", [
        ("", ""),
        ("abc", "abc"),
        ("~", "~0"),
        ("/", "~1"),
        ("a/b", "a~1b"),
        ("m~n", "m~0n"),
        ("~/", "~0~1"),
        ("/~", "~1~0"),
        ("~1", "~01"),
        ("~0", "~00"),
        ("//~~", "~1~1~0~0"),
    ])
    def test_escape(self, raw, escaped):
        assert escape(raw) == escaped

    @pytest.mark.parametrize("escaped, raw", [
        ("", ""),
        ("abc", "abc"),
        ("~0", "~"),
        ("~1", "/"),
        ("a~1b", "a/b"),
        ("m~0n", "m~n"),
        ("~0~1", "~/"),
        ("~1~0", "/~"),
        ("~00", "~0"),
    ])
    def test_unescape(self, escaped, raw):
        assert unescape(escaped) == raw

    @UNESCAPE_ORDER_BUG
    def test_unescape_tilde_zero_one_is_literal_tilde_one(self):
        # RFC 6901 section 4: "~01" must become "~1", not "/". Transforming
        # "~1" before "~0" is required to get this right.
        assert unescape("~01") == "~1"

    @pytest.mark.parametrize("raw", [
        "", "a", "~", "/", "~0", "~/~/",
        pytest.param("~1", marks=UNESCAPE_ORDER_BUG),
        pytest.param("a~1b/c~0", marks=UNESCAPE_ORDER_BUG),
    ])
    def test_round_trip(self, raw):
        assert unescape(escape(raw)) == raw


# --------------------------------------------------------------------------
# pairwise
# --------------------------------------------------------------------------

class TestPairwise:
    def test_empty(self):
        assert list(pairwise([])) == []

    def test_single(self):
        assert list(pairwise([1])) == []

    def test_pairs(self):
        assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_iterator_input(self):
        assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class TestParsing:
    @pytest.mark.parametrize("pointer, parts", [
        ("", []),
        ("/", [""]),
        ("//", ["", ""]),
        ("/foo", ["foo"]),
        ("/foo/0", ["foo", "0"]),
        ("/a~1b", ["a/b"]),
        ("/m~0n", ["m~n"]),
        ("/ ", [" "]),
        ("/foo/", ["foo", ""]),
        ("/~0~1", ["~/"]),
    ])
    def test_parts(self, pointer, parts):
        p = JsonPointer(pointer)
        assert p.parts == parts
        assert p.get_parts() == parts

    @pytest.mark.parametrize("pointer", ["foo", "a/b", " /x", "0"])
    def test_must_start_with_slash(self, pointer):
        with pytest.raises(JsonPointerException, match="must start with /"):
            JsonPointer(pointer)

    @pytest.mark.parametrize("pointer", ["/~", "/foo~", "/~2", "/~a", "/a~/b", "/~~0", "/x/~-"])
    def test_invalid_escape(self, pointer):
        with pytest.raises(JsonPointerException, match="invalid escape"):
            JsonPointer(pointer)

    def test_invalid_escape_message_contains_sequence(self):
        with pytest.raises(JsonPointerException) as ei:
            JsonPointer("/a~2")
        assert "~2" in str(ei.value)

    def test_invalid_escape_checked_before_leading_slash(self):
        with pytest.raises(JsonPointerException, match="invalid escape"):
            JsonPointer("foo~")

    @pytest.mark.parametrize("pointer", ["/~0", "/~1", "/~01", "/~10", "/a~0b~1c"])
    def test_valid_escape_accepted(self, pointer):
        JsonPointer(pointer)


# --------------------------------------------------------------------------
# Resolving
# --------------------------------------------------------------------------

class TestResolve:
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
    def test_rfc_examples(self, pointer, expected):
        assert resolve_pointer(RFC_DOC, pointer) == expected
        assert JsonPointer(pointer).resolve(RFC_DOC) == expected
        assert JsonPointer(pointer).get(RFC_DOC) == expected

    def test_root_is_same_object(self, doc):
        assert resolve_pointer(doc, "") is doc

    def test_nested(self, doc):
        assert resolve_pointer(doc, "/foo") is doc["foo"]
        assert resolve_pointer(doc, "/foo/another prop") == {"baz": "A string"}
        assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"
        assert resolve_pointer(doc, "/foo/anArray/0") is doc["foo"]["anArray"][0]
        assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44

    def test_no_percent_decoding(self, doc):
        assert resolve_pointer(doc, "/a%20b") == 1
        assert resolve_pointer(doc, "/c d") == 2
        assert resolve_pointer(doc, "/a b", None) is None
        assert resolve_pointer(doc, "/c%20d", None) is None

    @UNESCAPE_ORDER_BUG
    def test_tilde_zero_one_key(self):
        d = {"~1": "tilde-one", "/": "slash"}
        assert resolve_pointer(d, "/~01") == "tilde-one"
        assert resolve_pointer(d, "/~1") == "slash"

    def test_missing_key_raises(self, doc):
        with pytest.raises(JsonPointerException, match="member 'nope' not found"):
            resolve_pointer(doc, "/nope")

    def test_missing_nested_key_raises(self, doc):
        with pytest.raises(JsonPointerException):
            resolve_pointer(doc, "/foo/nope/deeper")

    def test_default_returned_on_missing(self, doc):
        assert resolve_pointer(doc, "/some/path", None) is None
        sentinel = object()
        assert resolve_pointer(doc, "/nope", sentinel) is sentinel
        assert JsonPointer("/nope").resolve(doc, 42) == 42

    @pytest.mark.parametrize("default", [None, 0, False, "", []])
    def test_falsy_defaults(self, doc, default):
        assert resolve_pointer(doc, "/nope", default) is default

    def test_default_not_used_when_found(self, doc):
        assert resolve_pointer(doc, "/c d", "default") == 2

    def test_default_on_index_out_of_bounds(self):
        assert resolve_pointer([1, 2], "/5", "d") == "d"

    def test_default_on_invalid_index(self):
        assert resolve_pointer([1, 2], "/x", "d") == "d"

    def test_default_on_string_traversal(self):
        assert resolve_pointer({"a": "str"}, "/a/b", "d") == "d"

    def test_list_index(self):
        assert resolve_pointer([10, 20, 30], "/0") == 10
        assert resolve_pointer([10, 20, 30], "/2") == 30
        assert resolve_pointer([[1, [2, 3]]], "/0/1/1") == 3

    def test_list_large_index(self):
        lst = list(range(20))
        assert resolve_pointer(lst, "/10") == 10
        assert resolve_pointer(lst, "/19") == 19

    def test_tuple_is_sequence(self):
        assert resolve_pointer((5, 6), "/1") == 6

    def test_index_out_of_bounds(self):
        with pytest.raises(JsonPointerException, match="out of bounds"):
            resolve_pointer([1, 2], "/2")

    def test_index_out_of_bounds_empty(self):
        with pytest.raises(JsonPointerException):
            resolve_pointer([], "/0")

    @pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", " 1", "1 ", "1.0", "a", "", "1a", "0x1", "١"])
    def test_invalid_index(self, idx):
        with pytest.raises(JsonPointerException, match="not a valid sequence index"):
            resolve_pointer([1, 2, 3], "/" + idx)

    def test_dash_resolves_to_end_of_list(self):
        lst = [1, 2]
        result = resolve_pointer(lst, "/-")
        assert isinstance(result, EndOfList)
        assert result.list_ is lst

    def test_dash_nested(self):
        d = {"a": [1]}
        result = resolve_pointer(d, "/a/-")
        assert isinstance(result, EndOfList)
        assert result.list_ is d["a"]

    def test_dash_is_ordinary_key_in_mapping(self):
        assert resolve_pointer({"-": 1}, "/-") == 1

    def test_digit_keys_in_mapping_are_strings(self):
        d = {"0": "zero", 0: "int-zero"}
        assert resolve_pointer(d, "/0") == "zero"
        assert resolve_pointer({"01": "x"}, "/01") == "x"

    def test_string_cannot_be_traversed(self):
        with pytest.raises(JsonPointerException, match="non-container"):
            resolve_pointer({"a": "hello"}, "/a/0")

    @pytest.mark.parametrize("value", [1, 1.5, None, True])
    def test_scalar_cannot_be_traversed(self, value):
        with pytest.raises(JsonPointerException, match="does not support indexing"):
            resolve_pointer({"a": value}, "/a/b")

    def test_scalar_default(self):
        assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"

    def test_ordered_dict_and_userdict(self):
        assert resolve_pointer(OrderedDict(a=1), "/a") == 1
        assert resolve_pointer(UserDict(a={"b": 2}), "/a/b") == 2

    def test_duck_typed_getitem(self):
        class Thing:
            def __getitem__(self, key):
                if key == "x":
                    return "got-x"
                raise KeyError(key)

        assert resolve_pointer(Thing(), "/x") == "got-x"
        with pytest.raises(JsonPointerException):
            resolve_pointer(Thing(), "/y")
        assert resolve_pointer(Thing(), "/y", None) is None

    def test_empty_key(self):
        assert resolve_pointer({"": {"": 5}}, "//") == 5

    def test_none_value_is_returned(self):
        assert resolve_pointer({"a": None}, "/a", "default") is None

    def test_verbose_exceptions(self, monkeypatch):
        monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
        with pytest.raises(JsonPointerException) as ei:
            resolve_pointer({"secret": 1}, "/x")
        assert "secret" in str(ei.value)

    def test_non_verbose_exceptions(self, monkeypatch):
        monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
        with pytest.raises(JsonPointerException) as ei:
            resolve_pointer({"secret": 1}, "/x")
        assert "secret" not in str(ei.value)
        assert "member 'x' not found" in str(ei.value)

    def test_invalid_pointer_raises_even_with_default(self):
        with pytest.raises(JsonPointerException):
            resolve_pointer({}, "nope", None)


# --------------------------------------------------------------------------
# get_part / walk / to_last
# --------------------------------------------------------------------------

class TestLowLevel:
    def test_get_part_mapping(self):
        assert JsonPointer.get_part({}, "0") == "0"

    def test_get_part_sequence(self):
        assert JsonPointer.get_part([1], "0") == 0
        assert JsonPointer.get_part([1], "12") == 12
        assert isinstance(JsonPointer.get_part([1], "3"), int)
        assert JsonPointer.get_part([1], "-") == "-"

    def test_get_part_sequence_int_part(self):
        assert JsonPointer.get_part([1], 3) == 3

    def test_get_part_invalid(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([1], "01")
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part("abc", "0")
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part(5, "0")

    def test_get_part_duck_typed(self):
        class G:
            def __getitem__(self, k):
                return k
        assert JsonPointer.get_part(G(), "x") == "x"

    def test_walk(self):
        p = JsonPointer("")
        assert p.walk({"a": 1}, "a") == 1
        assert p.walk([5, 6], "1") == 6
        assert isinstance(p.walk([5, 6], "-"), EndOfList)

    def test_to_last_root(self):
        d = {"a": 1}
        assert JsonPointer("").to_last(d) == (d, None)

    def test_to_last(self):
        d = {"a": {"b": [1, 2]}}
        parent, part = JsonPointer("/a/b/1").to_last(d)
        assert parent is d["a"]["b"]
        assert part == 1

        parent, part = JsonPointer("/a/b").to_last(d)
        assert parent is d["a"]
        assert part == "b"

    def test_to_last_single(self):
        d = {"a": 1}
        parent, part = JsonPointer("/a").to_last(d)
        assert parent is d
        assert part == "a"

    def test_to_last_missing_intermediate(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/x/y").to_last({})


# --------------------------------------------------------------------------
# Setting
# --------------------------------------------------------------------------

class TestSet:
    def test_set_existing_inplace(self, doc):
        result = set_pointer(doc, "/foo/anArray/0/prop", 55)
        assert result is doc
        assert doc["foo"]["anArray"][0]["prop"] == 55

    def test_set_new_key(self, doc):
        set_pointer(doc, "/foo/yet another prop", "added prop")
        assert doc["foo"]["yet another prop"] == "added prop"
        assert doc["foo"]["another prop"] == {"baz": "A string"}

    def test_set_no_percent_decoding(self):
        d = {"foo": {}}
        assert set_pointer(d, "/foo/a%20b", "x") == {"foo": {"a%20b": "x"}}

    def test_set_escaped_key(self):
        d = {}
        set_pointer(d, "/a~1b", 1)
        set_pointer(d, "/m~0n", 2)
        assert d == {"a/b": 1, "m~n": 2}

    def test_set_not_inplace(self, doc):
        original = copy.deepcopy(doc)
        result = set_pointer(doc, "/foo/anArray/0/prop", 99, inplace=False)
        assert doc == original
        assert result is not doc
        assert result["foo"]["anArray"][0]["prop"] == 99
        assert result["foo"] is not doc["foo"]

    def test_set_root_inplace_raises(self, doc):
        with pytest.raises(JsonPointerException, match="root"):
            set_pointer(doc, "", 1)

    def test_set_root_not_inplace(self, doc):
        new = {"x": 1}
        assert set_pointer(doc, "", new, inplace=False) is new
        assert JsonPointer("").set(doc, 5, inplace=False) == 5

    def test_set_list_index(self):
        lst = [1, 2, 3]
        set_pointer(lst, "/1", "b")
        assert lst == [1, "b", 3]

    def test_set_list_dash_appends(self):
        lst = [1, 2]
        result = set_pointer(lst, "/-", 3)
        assert result is lst
        assert lst == [1, 2, 3]

    def test_set_nested_list_dash(self):
        d = {"a": []}
        set_pointer(d, "/a/-", "x")
        set_pointer(d, "/a/-", "y")
        assert d == {"a": ["x", "y"]}

    def test_set_dash_not_inplace(self):
        d = {"a": [1]}
        result = set_pointer(d, "/a/-", 2, inplace=False)
        assert d == {"a": [1]}
        assert result == {"a": [1, 2]}

    def test_set_dash_on_mapping_is_key(self):
        d = {}
        set_pointer(d, "/-", 1)
        assert d == {"-": 1}

    def test_set_list_out_of_bounds(self):
        with pytest.raises(JsonPointerException, match="Invalid assignment target"):
            set_pointer([1], "/5", 0)

    def test_set_list_invalid_index(self):
        with pytest.raises(JsonPointerException):
            set_pointer([1], "/01", 0)

    def test_set_tuple_raises(self):
        with pytest.raises(JsonPointerException, match="Invalid assignment target"):
            set_pointer((1, 2), "/0", 0)

    def test_set_in_string_raises(self):
        with pytest.raises(JsonPointerException):
            set_pointer({"a": "str"}, "/a/0", "x")

    def test_set_in_string_parent_message(self):
        # Bypass get_part to reach the explicit string check in set().
        class P(JsonPointer):
            def to_last(self, doc):
                return doc["a"], "0"
        with pytest.raises(JsonPointerException, match="Cannot set value in a string"):
            P("/a/0").set({"a": "str"}, "x")

    def test_set_missing_intermediate_raises(self):
        with pytest.raises(JsonPointerException):
            set_pointer({}, "/a/b", 1)

    def test_method_set(self):
        d = {"a": {"b": 1}}
        assert JsonPointer("/a/b").set(d, 2) is d
        assert d == {"a": {"b": 2}}

    def test_set_default_is_inplace(self):
        d = {"a": 1}
        JsonPointer("/a").set(d, 2)
        assert d["a"] == 2


# --------------------------------------------------------------------------
# Containment / joining
# --------------------------------------------------------------------------

class TestContainsJoin:
    def test_contains(self):
        a = JsonPointer("/a/b/c")
        assert a.contains(JsonPointer("/a/b"))
        assert a.contains(JsonPointer("/a/b/c"))
        assert a.contains(JsonPointer(""))
        assert not a.contains(JsonPointer("/a/x"))
        assert not a.contains(JsonPointer("/a/b/c/d"))
        assert not JsonPointer("/a").contains(JsonPointer("/a/b"))

    def test_contains_is_part_based_not_prefix(self):
        assert not JsonPointer("/ab").contains(JsonPointer("/a"))

    def test_in_operator(self):
        assert JsonPointer("/a") in JsonPointer("/a/b")
        assert JsonPointer("/a/b") not in JsonPointer("/a")

    def test_join_string(self):
        assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")

    def test_join_pointer(self):
        assert JsonPointer("/a").join(JsonPointer("/b")) == JsonPointer("/a/b")

    def test_join_list(self):
        assert JsonPointer("/a").join(["b", "c/d"]).path == "/a/b/c~1d"

    def test_join_list_with_ints(self):
        assert JsonPointer("/a").join(["b", 0]).parts == ["a", "b", "0"]

    def test_join_does_not_mutate(self):
        a = JsonPointer("/a")
        a.join("/b")
        assert a.parts == ["a"]

    def test_join_root(self):
        assert JsonPointer("").join("/x") == JsonPointer("/x")
        assert JsonPointer("/x").join("") == JsonPointer("/x")

    def test_truediv(self):
        assert (JsonPointer("/a") / "/b").path == "/a/b"
        assert (JsonPointer("/a") / ["b", "~"]).path == "/a/b/~0"
        assert (JsonPointer("/a") / JsonPointer("/c")).path == "/a/c"

    def test_join_invalid_string_suffix(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/a").join("b")

    def test_join_invalid_suffix_type(self):
        with pytest.raises(JsonPointerException, match="Invalid suffix"):
            JsonPointer("/a").join(5)

    def test_join_preserves_subclass(self):
        class MyPointer(JsonPointer):
            pass
        assert type(MyPointer("/a").join("/b")) is MyPointer


# --------------------------------------------------------------------------
# path / str / repr / eq / hash / from_parts
# --------------------------------------------------------------------------

class TestRepresentation:
    @pytest.mark.parametrize("pointer", [
        "", "/", "//", "/foo", "/foo/0", "/a~1b", "/m~0n", "/~0/0/~1", "/ ",
        pytest.param("/~01", marks=UNESCAPE_ORDER_BUG),
    ])
    def test_path_round_trip(self, pointer):
        assert JsonPointer(pointer).path == pointer

    def test_str(self):
        assert str(JsonPointer("/a~1b/c")) == "/a~1b/c"

    def test_repr(self):
        assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
        assert repr(JsonPointer("")) == "JsonPointer('')"

    def test_repr_subclass(self):
        class Sub(JsonPointer):
            pass
        assert repr(Sub("/x")) == "Sub('/x')"

    def test_eq(self):
        assert JsonPointer("/a/b") == JsonPointer("/a/b")
        assert JsonPointer("") == JsonPointer("")
        assert JsonPointer("/a/b") != JsonPointer("/a/c")
        assert JsonPointer("/a") != JsonPointer("/a/b")

    def test_eq_other_types(self):
        assert JsonPointer("/a") != "/a"
        assert not (JsonPointer("/a") == "/a")
        assert JsonPointer("/a") != ["a"]
        assert JsonPointer("") != None  # noqa: E711

    def test_hash(self):
        assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
        s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
        assert len(s) == 2
        assert {JsonPointer("/x"): 1}[JsonPointer("/x")] == 1

    def test_from_parts(self):
        p = JsonPointer.from_parts(["a", "~", "/", 0])
        assert p.path == "/a/~0/~1/0"
        assert p.parts == ["a", "~", "/", "0"]

    def test_from_parts_empty(self):
        assert JsonPointer.from_parts([]) == JsonPointer("")

    @UNESCAPE_ORDER_BUG
    def test_from_parts_tricky(self):
        p = JsonPointer.from_parts(["~1", "~0", ""])
        assert p.path == "/~01/~00/"
        assert p.parts == ["~1", "~0", ""]

    def test_from_parts_generator(self):
        assert JsonPointer.from_parts(x for x in ["a", "b"]).path == "/a/b"

    def test_from_parts_subclass(self):
        class Sub(JsonPointer):
            pass
        assert type(Sub.from_parts(["a"])) is Sub

    def test_from_parts_resolves(self):
        d = {"a/b": {"c~d": [0, "hit"]}}
        assert JsonPointer.from_parts(["a/b", "c~d", 1]).resolve(d) == "hit"


# --------------------------------------------------------------------------
# EndOfList
# --------------------------------------------------------------------------

class TestEndOfList:
    def test_repr(self):
        assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"

    def test_holds_list(self):
        lst = []
        assert EndOfList(lst).list_ is lst


def test_exception_is_exception_subclass():
    assert issubclass(JsonPointerException, Exception)
