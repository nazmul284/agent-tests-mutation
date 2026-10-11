import copy
from collections import UserDict, UserList

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
        "a%20b": 1,
        "c d": 2,
    }


# ---------------------------------------------------------------------------
# escape / unescape
# ---------------------------------------------------------------------------

class TestEscaping:
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
        ("//", "~1~1"),
    ])
    def test_escape(self, raw, escaped):
        assert escape(raw) == escaped

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
    ])
    def test_unescape(self, raw, escaped):
        assert unescape(escaped) == raw

    def test_unescape_order_01_is_tilde_one(self):
        # RFC 6901: "~01" must become "~1", not "/"
        assert unescape("~01") == "~1"

    @pytest.mark.parametrize("s", ["", "a", "~", "/", "~0", "~1", "~01", "a/~b/~/"])
    def test_roundtrip(self, s):
        assert unescape(escape(s)) == s


# ---------------------------------------------------------------------------
# pairwise
# ---------------------------------------------------------------------------

class TestPairwise:
    def test_empty(self):
        assert list(pairwise([])) == []

    def test_single(self):
        assert list(pairwise([1])) == []

    def test_two(self):
        assert list(pairwise([1, 2])) == [(1, 2)]

    def test_many(self):
        assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_generator(self):
        assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_root(self):
        assert JsonPointer("").parts == []

    def test_slash_only_is_empty_key(self):
        assert JsonPointer("/").parts == [""]

    def test_simple(self):
        assert JsonPointer("/a/b").parts == ["a", "b"]

    def test_empty_parts(self):
        assert JsonPointer("//").parts == ["", ""]
        assert JsonPointer("/a//b/").parts == ["a", "", "b", ""]

    def test_unescapes_parts(self):
        assert JsonPointer("/a~1b/m~0n/~01").parts == ["a/b", "m~n", "~1"]

    def test_get_parts(self):
        assert JsonPointer("/x/0/~1").get_parts() == ["x", "0", "/"]

    def test_parts_are_strings(self):
        for p in JsonPointer("/0/1/-").parts:
            assert isinstance(p, str)

    @pytest.mark.parametrize("bad", ["a", "a/b", "0", " /a", "~0/a"])
    def test_must_start_with_slash(self, bad):
        with pytest.raises(JsonPointerException) as exc:
            JsonPointer(bad)
        assert str(exc.value) == "Location must start with /"

    @pytest.mark.parametrize("bad, found", [
        ("/~", "~"),
        ("/a~", "~"),
        ("/~2", "~2"),
        ("/a~xb", "~x"),
        ("/~a", "~a"),
        ("/ok/~/x", "~/"),
        ("/~0~", "~"),
    ])
    def test_invalid_escape(self, bad, found):
        with pytest.raises(JsonPointerException) as exc:
            JsonPointer(bad)
        assert str(exc.value) == "Found invalid escape {}".format(found)

    def test_invalid_escape_checked_before_leading_slash(self):
        with pytest.raises(JsonPointerException) as exc:
            JsonPointer("a~")
        assert "invalid escape" in str(exc.value)

    @pytest.mark.parametrize("good", ["/~0", "/~1", "/~0~1", "/a~0b~1c", "/~00", "/~11"])
    def test_valid_escapes(self, good):
        JsonPointer(good)  # does not raise

    def test_exception_is_exception_subclass(self):
        assert issubclass(JsonPointerException, Exception)


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

class TestResolveRFC:
    @pytest.mark.parametrize("ptr, expected", [
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
    def test_rfc_examples(self, ptr, expected):
        assert resolve_pointer(RFC_DOC, ptr) == expected
        assert JsonPointer(ptr).resolve(RFC_DOC) == expected
        assert JsonPointer(ptr).get(RFC_DOC) == expected

    def test_root_returns_same_object(self):
        d = {"a": 1}
        assert resolve_pointer(d, "") is d

    def test_returns_identity(self):
        inner = {"x": 1}
        d = {"a": [inner]}
        assert resolve_pointer(d, "/a/0") is inner


class TestResolve:
    def test_examples(self, doc):
        assert resolve_pointer(doc, "/foo") == doc["foo"]
        assert resolve_pointer(doc, "/foo/another prop") == {"baz": "A string"}
        assert resolve_pointer(doc, "/foo/another prop/baz") == "A string"
        assert resolve_pointer(doc, "/foo/anArray/0") == {"prop": 44}
        assert resolve_pointer(doc, "/foo/anArray/0/prop") == 44
        assert resolve_pointer(doc, "/a%20b") == 1
        assert resolve_pointer(doc, "/c d") == 2

    def test_no_percent_decoding(self, doc):
        assert resolve_pointer(doc, "/c%20d", None) is None
        assert resolve_pointer(doc, "/a b", None) is None

    def test_missing_key_raises(self, doc):
        with pytest.raises(JsonPointerException):
            resolve_pointer(doc, "/some/path")

    def test_missing_key_with_default(self, doc):
        assert resolve_pointer(doc, "/some/path", None) is None
        assert resolve_pointer(doc, "/some/path", 42) == 42
        sentinel = object()
        assert JsonPointer("/nope").resolve(doc, sentinel) is sentinel

    def test_default_on_deep_failure(self, doc):
        assert resolve_pointer(doc, "/foo/anArray/5/prop", "d") == "d"

    def test_default_not_used_when_found(self, doc):
        assert resolve_pointer(doc, "/a%20b", "d") == 1

    def test_default_none_is_honored(self):
        # None is a legitimate default (distinct from "no default")
        assert resolve_pointer({}, "/x", None) is None

    def test_default_false_like_values(self):
        assert resolve_pointer({}, "/x", 0) == 0
        assert resolve_pointer({}, "/x", "") == ""
        assert resolve_pointer({}, "/x", False) is False

    def test_get_alias(self):
        assert JsonPointer.get is JsonPointer.resolve

    def test_list_index(self):
        assert resolve_pointer([10, 20, 30], "/0") == 10
        assert resolve_pointer([10, 20, 30], "/2") == 30

    def test_list_index_multidigit(self):
        lst = list(range(20))
        assert resolve_pointer(lst, "/10") == 10
        assert resolve_pointer(lst, "/19") == 19

    def test_list_index_out_of_bounds(self):
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer([1, 2], "/2")
        assert str(exc.value) == "index '2' is out of bounds"

    def test_list_index_out_of_bounds_default(self):
        assert resolve_pointer([1, 2], "/5", "d") == "d"

    @pytest.mark.parametrize("bad", ["01", "00", "-1", "+1", " 1", "1 ", "1.0", "a", "", "1e2", "0x1", "１"])
    def test_invalid_list_index(self, bad):
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer([1, 2, 3], "/" + bad)
        assert str(exc.value) == "'%s' is not a valid sequence index" % bad

    def test_invalid_list_index_default(self):
        assert resolve_pointer([1], "/01", "d") == "d"

    def test_trailing_newline_index_invalid(self):
        with pytest.raises(JsonPointerException):
            resolve_pointer([1, 2], "/1\n")

    def test_dash_on_list_returns_end_of_list(self):
        lst = [1, 2]
        result = resolve_pointer(lst, "/-")
        assert isinstance(result, EndOfList)
        assert result.list_ is lst

    def test_dash_on_tuple(self):
        t = (1, 2)
        result = resolve_pointer(t, "/-")
        assert isinstance(result, EndOfList)
        assert result.list_ is t

    def test_dash_on_dict_is_key(self):
        assert resolve_pointer({"-": 5}, "/-") == 5

    def test_tuple_index(self):
        assert resolve_pointer((5, 6), "/1") == 6

    def test_numeric_key_in_dict_is_string(self):
        assert resolve_pointer({"0": "s"}, "/0") == "s"
        with pytest.raises(JsonPointerException):
            resolve_pointer({0: "int"}, "/0")

    def test_dict_leading_zero_key_allowed(self):
        assert resolve_pointer({"01": 1}, "/01") == 1

    def test_into_string_raises(self):
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer({"a": "hello"}, "/a/0")
        assert str(exc.value) == (
            "Cannot apply token '0' to non-container type %s" % type("")
        )

    def test_into_string_default(self):
        assert resolve_pointer({"a": "hello"}, "/a/0", "d") == "d"

    @pytest.mark.parametrize("value", [1, 1.5, None, True])
    def test_into_scalar_raises(self, value):
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer({"a": value}, "/a/b")
        assert str(exc.value) == (
            "Document '%s' does not support indexing, "
            "must be mapping/sequence or support __getitem__" % type(value)
        )

    def test_into_scalar_default(self):
        assert resolve_pointer({"a": 1}, "/a/b", "d") == "d"

    def test_empty_key(self):
        assert resolve_pointer({"": {"": 3}}, "//") == 3

    def test_key_with_slash_and_tilde(self):
        d = {"a/b": {"c~d": {"~1": "ok"}}}
        assert resolve_pointer(d, "/a~1b/c~0d/~01") == "ok"

    def test_duck_typed_getitem(self):
        class Thing:
            def __getitem__(self, key):
                if key == "x":
                    return "X"
                raise KeyError(key)

        assert resolve_pointer(Thing(), "/x") == "X"
        with pytest.raises(JsonPointerException):
            resolve_pointer(Thing(), "/y")
        assert resolve_pointer(Thing(), "/y", "d") == "d"

    def test_userdict_and_userlist(self):
        d = UserDict({"a": UserList([1, 2])})
        assert resolve_pointer(d, "/a/1") == 2

    def test_unicode(self):
        assert resolve_pointer({"ü": {"日本": 1}}, "/ü/日本") == 1


class TestVerboseExceptions:
    def test_verbose(self, monkeypatch):
        monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", True)
        d = {"a": 1}
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer(d, "/b")
        assert str(exc.value) == "member 'b' not found in %s" % (d,)

    def test_not_verbose(self, monkeypatch):
        monkeypatch.setattr(jsonpointer, "VERBOSE_EXCEPTIONS", False)
        with pytest.raises(JsonPointerException) as exc:
            resolve_pointer({"a": 1}, "/b")
        assert str(exc.value) == "member 'b' not found"

    def test_default_is_verbose(self):
        assert jsonpointer.VERBOSE_EXCEPTIONS is True


# ---------------------------------------------------------------------------
# walk / get_part / to_last
# ---------------------------------------------------------------------------

class TestWalkAndGetPart:
    def test_get_part_mapping(self):
        assert JsonPointer.get_part({}, "0") == "0"
        assert JsonPointer.get_part({}, "-") == "-"

    def test_get_part_sequence(self):
        assert JsonPointer.get_part([], "0") == 0
        assert JsonPointer.get_part([], "12") == 12
        assert isinstance(JsonPointer.get_part([], "3"), int)

    def test_get_part_sequence_int_input(self):
        assert JsonPointer.get_part([1], 0) == 0
        assert JsonPointer.get_part([1], 7) == 7

    def test_get_part_sequence_dash(self):
        assert JsonPointer.get_part([], "-") == "-"

    def test_get_part_sequence_invalid(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([], "x")
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([], "007")

    def test_get_part_string(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part("abc", "0")

    def test_get_part_duck(self):
        class G:
            def __getitem__(self, k):
                return k
        assert JsonPointer.get_part(G(), "007") == "007"

    def test_get_part_unindexable(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part(object(), "a")

    def test_walk(self):
        p = JsonPointer("")
        assert p.walk({"a": 1}, "a") == 1
        assert p.walk([5, 6], "1") == 6
        assert isinstance(p.walk([5, 6], "-"), EndOfList)

    def test_walk_out_of_bounds(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("").walk([1], "1")

    def test_walk_missing_key(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("").walk({}, "a")

    def test_to_last_root(self):
        d = {"a": 1}
        parent, part = JsonPointer("").to_last(d)
        assert parent is d
        assert part is None

    def test_to_last_mapping(self):
        d = {"a": {"b": 1}}
        parent, part = JsonPointer("/a/b").to_last(d)
        assert parent is d["a"]
        assert part == "b"

    def test_to_last_list(self):
        d = {"a": [1, 2]}
        parent, part = JsonPointer("/a/1").to_last(d)
        assert parent is d["a"]
        assert part == 1 and isinstance(part, int)

    def test_to_last_single(self):
        d = {"a": 1}
        parent, part = JsonPointer("/a").to_last(d)
        assert parent is d
        assert part == "a"

    def test_to_last_dash(self):
        d = [1]
        parent, part = JsonPointer("/-").to_last(d)
        assert parent is d
        assert part == "-"

    def test_to_last_missing_intermediate(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/x/y").to_last({})

    def test_to_last_does_not_require_last_to_exist(self):
        d = {"a": {}}
        parent, part = JsonPointer("/a/new").to_last(d)
        assert parent is d["a"]
        assert part == "new"


# ---------------------------------------------------------------------------
# EndOfList
# ---------------------------------------------------------------------------

class TestEndOfList:
    def test_repr(self):
        assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"
        assert repr(EndOfList([])) == "EndOfList([])"
        assert repr(EndOfList(["a"])) == "EndOfList(['a'])"

    def test_attr(self):
        lst = [1]
        assert EndOfList(lst).list_ is lst


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

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

    def test_set_default_is_inplace(self):
        d = {"a": 1}
        JsonPointer("/a").set(d, 2)
        assert d == {"a": 2}

    def test_set_not_inplace(self, doc):
        original = copy.deepcopy(doc)
        result = set_pointer(doc, "/foo/anArray/0/prop", 55, inplace=False)
        assert doc == original
        assert result is not doc
        assert result["foo"]["anArray"][0]["prop"] == 55
        # deep copy: nested objects are not shared
        assert result["foo"] is not doc["foo"]
        assert result["foo"]["another prop"] is not doc["foo"]["another prop"]

    def test_set_not_inplace_method(self):
        d = {"a": [1]}
        result = JsonPointer("/a/0").set(d, 9, inplace=False)
        assert d == {"a": [1]}
        assert result == {"a": [9]}

    def test_set_root_inplace_raises(self):
        with pytest.raises(JsonPointerException) as exc:
            set_pointer({}, "", 1)
        assert str(exc.value) == "Cannot set root in place"

    def test_set_root_inplace_raises_explicit(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("").set({}, 1, inplace=True)

    def test_set_root_not_inplace(self):
        d = {"a": 1}
        new = {"b": 2}
        assert set_pointer(d, "", new, inplace=False) is new
        assert d == {"a": 1}

    def test_set_list_index(self):
        d = [1, 2, 3]
        set_pointer(d, "/1", "x")
        assert d == [1, "x", 3]

    def test_set_list_append_dash(self):
        d = {"a": [1, 2]}
        set_pointer(d, "/a/-", 3)
        assert d == {"a": [1, 2, 3]}

    def test_set_list_append_dash_not_inplace(self):
        d = {"a": [1]}
        r = set_pointer(d, "/a/-", 2, inplace=False)
        assert d == {"a": [1]}
        assert r == {"a": [1, 2]}

    def test_set_dash_on_dict_is_key(self):
        d = {}
        set_pointer(d, "/-", 1)
        assert d == {"-": 1}

    def test_set_list_out_of_bounds(self):
        with pytest.raises(JsonPointerException) as exc:
            set_pointer([1], "/5", 0)
        assert str(exc.value).startswith("Invalid assignment target: ")

    def test_set_list_one_past_end_raises(self):
        with pytest.raises(JsonPointerException):
            set_pointer([1, 2], "/2", 0)

    def test_set_invalid_list_index(self):
        with pytest.raises(JsonPointerException):
            set_pointer([1, 2], "/01", 0)

    def test_set_on_tuple_raises(self):
        with pytest.raises(JsonPointerException) as exc:
            set_pointer({"t": (1, 2)}, "/t/0", 5)
        assert str(exc.value).startswith("Invalid assignment target: ")

    def test_set_in_string_raises(self):
        with pytest.raises(JsonPointerException) as exc:
            set_pointer({"s": "abc"}, "/s/0", "x")
        # The string check is either in get_part or set; both raise
        assert isinstance(exc.value, JsonPointerException)

    def test_set_in_string_inside_list(self):
        with pytest.raises(JsonPointerException):
            set_pointer(["abc"], "/0/1", "x")

    def test_set_missing_intermediate_raises(self):
        with pytest.raises(JsonPointerException):
            set_pointer({}, "/a/b", 1)

    def test_set_into_scalar_raises(self):
        with pytest.raises(JsonPointerException):
            set_pointer({"a": 1}, "/a/b", 1)

    def test_set_returns_doc(self):
        d = {"a": 1}
        assert set_pointer(d, "/a", 2) is d

    def test_set_empty_key(self):
        d = {}
        set_pointer(d, "/", 1)
        assert d == {"": 1}

    def test_set_duck_type(self):
        class Box:
            def __init__(self):
                self.data = {}

            def __getitem__(self, k):
                return self.data[k]

            def __setitem__(self, k, v):
                self.data[k] = v

        b = Box()
        set_pointer(b, "/k", 1)
        assert b.data == {"k": 1}

    def test_set_type_error_wrapped(self):
        class NoSet:
            def __getitem__(self, k):
                return 1

        with pytest.raises(JsonPointerException) as exc:
            set_pointer(NoSet(), "/k", 1)
        assert str(exc.value).startswith("Invalid assignment target: ")


# ---------------------------------------------------------------------------
# path / str / repr / equality / hashing
# ---------------------------------------------------------------------------

class TestRepresentation:
    @pytest.mark.parametrize("ptr", [
        "", "/", "//", "/a", "/a/b", "/~0/0/~1", "/a~1b/m~0n", "/~01", "/ ", "/-",
    ])
    def test_path_roundtrip(self, ptr):
        assert JsonPointer(ptr).path == ptr
        assert str(JsonPointer(ptr)) == ptr

    def test_repr(self):
        assert repr(JsonPointer("/a/b")) == "JsonPointer('/a/b')"
        assert repr(JsonPointer("")) == "JsonPointer('')"
        assert repr(JsonPointer("/~0~1")) == "JsonPointer('/~0~1')"

    def test_repr_subclass(self):
        class MyPointer(JsonPointer):
            pass
        assert repr(MyPointer("/x")) == "MyPointer('/x')"

    def test_eq(self):
        assert JsonPointer("/a/b") == JsonPointer("/a/b")
        assert JsonPointer("") == JsonPointer("")
        assert not (JsonPointer("/a") == JsonPointer("/b"))
        assert JsonPointer("/a") != JsonPointer("/a/b")
        assert JsonPointer("/") != JsonPointer("")

    def test_eq_other_types(self):
        assert JsonPointer("/a") != "/a"
        assert not (JsonPointer("/a") == "/a")
        assert not (JsonPointer("/a") == ["a"])
        assert JsonPointer("") != None  # noqa: E711

    def test_hash(self):
        assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
        assert hash(JsonPointer("/a/b")) == hash(("a", "b"))
        s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
        assert len(s) == 2
        d = {JsonPointer("/x"): 1}
        assert d[JsonPointer("/x")] == 1


# ---------------------------------------------------------------------------
# from_parts / join / truediv / contains
# ---------------------------------------------------------------------------

class TestFromParts:
    def test_basic(self):
        p = JsonPointer.from_parts(["a", "~", "/", 0])
        assert p.path == "/a/~0/~1/0"
        assert p.parts == ["a", "~", "/", "0"]

    def test_empty(self):
        p = JsonPointer.from_parts([])
        assert p.path == ""
        assert p.parts == []

    def test_empty_string_part(self):
        assert JsonPointer.from_parts([""]).path == "/"

    def test_generator(self):
        assert JsonPointer.from_parts(x for x in ["a", "b"]).parts == ["a", "b"]

    def test_tilde_one_literal(self):
        p = JsonPointer.from_parts(["~1"])
        assert p.path == "/~01"
        assert p.parts == ["~1"]

    def test_subclass(self):
        class MyPointer(JsonPointer):
            pass
        p = MyPointer.from_parts(["a"])
        assert type(p) is MyPointer

    def test_roundtrip(self):
        for ptr in ["/a/b", "/~0/~1", "/", "", "/a~1b/c~0d"]:
            p = JsonPointer(ptr)
            assert JsonPointer.from_parts(p.parts) == p


class TestJoin:
    def test_join_string(self):
        assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")

    def test_join_pointer(self):
        assert JsonPointer("/a").join(JsonPointer("/b")) == JsonPointer("/a/b")

    def test_join_list(self):
        assert JsonPointer("/a").join(["b", "c"]).path == "/a/b/c"

    def test_join_list_escapes(self):
        assert JsonPointer("/a").join(["b/c", "~"]).path == "/a/b~1c/~0"

    def test_join_list_ints(self):
        assert JsonPointer("/a").join([0, 1]).parts == ["a", "0", "1"]

    def test_join_empty(self):
        assert JsonPointer("/a").join("") == JsonPointer("/a")
        assert JsonPointer("/a").join([]) == JsonPointer("/a")
        assert JsonPointer("").join("/a") == JsonPointer("/a")

    def test_join_escaped_string(self):
        assert JsonPointer("/x").join("/a~1b").parts == ["x", "a/b"]

    def test_join_does_not_mutate(self):
        p = JsonPointer("/a")
        p.join("/b")
        assert p.parts == ["a"]

    def test_join_returns_new_instance(self):
        p = JsonPointer("/a")
        assert p.join([]) is not p

    def test_join_invalid_string(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/a").join("b")

    def test_join_invalid_type(self):
        with pytest.raises(JsonPointerException) as exc:
            JsonPointer("/a").join(5)
        assert str(exc.value) == "Invalid suffix"

    def test_join_subclass(self):
        class MyPointer(JsonPointer):
            pass
        assert type(MyPointer("/a").join("/b")) is MyPointer

    def test_truediv(self):
        assert (JsonPointer("/a") / "/b") == JsonPointer("/a/b")
        assert (JsonPointer("/a") / ["b", "c"]) == JsonPointer("/a/b/c")
        assert (JsonPointer("/a") / JsonPointer("/b")).path == "/a/b"

    def test_truediv_chain(self):
        p = JsonPointer("") / ["a"] / ["b"]
        assert p.path == "/a/b"

    def test_truediv_invalid(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/a") / 5


class TestContains:
    def test_contains_prefix(self):
        assert JsonPointer("/a/b/c").contains(JsonPointer("/a/b"))
        assert JsonPointer("/a/b/c").contains(JsonPointer("/a"))

    def test_contains_self(self):
        assert JsonPointer("/a/b").contains(JsonPointer("/a/b"))

    def test_contains_root(self):
        assert JsonPointer("/a").contains(JsonPointer(""))
        assert JsonPointer("").contains(JsonPointer(""))

    def test_not_contains_longer(self):
        assert not JsonPointer("/a").contains(JsonPointer("/a/b"))
        assert not JsonPointer("").contains(JsonPointer("/a"))

    def test_not_contains_different(self):
        assert not JsonPointer("/a/b").contains(JsonPointer("/a/c"))
        assert not JsonPointer("/a/b").contains(JsonPointer("/b"))

    def test_not_contains_string_prefix(self):
        # part-wise, not character-wise
        assert not JsonPointer("/abc").contains(JsonPointer("/ab"))

    def test_in_operator(self):
        assert JsonPointer("/a") in JsonPointer("/a/b")
        assert JsonPointer("/a/b") not in JsonPointer("/a")
        assert JsonPointer("/x") not in JsonPointer("/a/b")


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

def test_version():
    assert jsonpointer.__version__ == "3.2.1"
