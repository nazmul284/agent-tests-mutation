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


@pytest.fixture(autouse=True)
def _restore_verbose():
    old = jsonpointer.VERBOSE_EXCEPTIONS
    yield
    jsonpointer.VERBOSE_EXCEPTIONS = old


# ---------------------------------------------------------------- escaping

class TestEscaping:
    @pytest.mark.parametrize("raw, escaped", [
        ("", ""),
        ("abc", "abc"),
        ("~", "~0"),
        ("/", "~1"),
        ("a/b~c", "a~1b~0c"),
        ("~1", "~01"),
        ("~0", "~00"),
        ("/~", "~1~0"),
        ("~/", "~0~1"),
    ])
    def test_escape(self, raw, escaped):
        assert escape(raw) == escaped

    @pytest.mark.parametrize("escaped, raw", [
        ("", ""),
        ("abc", "abc"),
        ("~0", "~"),
        ("~1", "/"),
        ("a~1b~0c", "a/b~c"),
        # RFC 6901 sec. 4: ~01 must become ~1, not /
        ("~01", "~1"),
        ("~00", "~0"),
        ("~1~0", "/~"),
        ("~0~1", "~/"),
    ])
    def test_unescape(self, escaped, raw):
        assert unescape(escaped) == raw

    @pytest.mark.parametrize("raw", ["", "a", "~", "/", "~1", "~0", "/~0~1/", "~~//", "a~01b"])
    def test_roundtrip(self, raw):
        assert unescape(escape(raw)) == raw


# ---------------------------------------------------------------- pairwise

def test_pairwise():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []
    assert list(pairwise([1, 2])) == [(1, 2)]
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise(iter("abc"))) == [("a", "b"), ("b", "c")]


# ---------------------------------------------------------------- parsing

class TestParsing:
    def test_root(self):
        assert JsonPointer("").parts == []

    def test_single_slash(self):
        assert JsonPointer("/").parts == [""]

    def test_simple(self):
        assert JsonPointer("/a/b/0").parts == ["a", "b", "0"]

    def test_empty_segments(self):
        assert JsonPointer("//").parts == ["", ""]
        assert JsonPointer("/a/").parts == ["a", ""]

    def test_escapes_unescaped_in_parts(self):
        assert JsonPointer("/a~1b/m~0n").parts == ["a/b", "m~n"]

    def test_tilde01_part(self):
        assert JsonPointer("/~01").parts == ["~1"]

    def test_get_parts(self):
        p = JsonPointer("/x/y")
        assert p.get_parts() == ["x", "y"]
        assert p.get_parts() is p.parts

    @pytest.mark.parametrize("bad", ["a", "a/b", "foo", " /a", "0"])
    def test_must_start_with_slash(self, bad):
        with pytest.raises(JsonPointerException, match="must start with /"):
            JsonPointer(bad)

    @pytest.mark.parametrize("bad", ["/~", "/~2", "/a~", "/~a", "/foo~/bar", "/~~", "/~-1"])
    def test_invalid_escape(self, bad):
        with pytest.raises(JsonPointerException, match="invalid escape"):
            JsonPointer(bad)

    def test_invalid_escape_message_shows_escape(self):
        with pytest.raises(JsonPointerException) as ei:
            JsonPointer("/a~2")
        assert "~2" in str(ei.value)

    def test_invalid_escape_at_end_message(self):
        with pytest.raises(JsonPointerException) as ei:
            JsonPointer("/a~")
        assert str(ei.value) == "Found invalid escape ~"

    def test_valid_escapes_accepted(self):
        JsonPointer("/~0/~1/~01/~10")

    def test_exception_is_exception_subclass(self):
        assert issubclass(JsonPointerException, Exception)


# ---------------------------------------------------------------- resolve

class TestResolve:
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
        doc = {"a": 1}
        assert resolve_pointer(doc, "") is doc

    def test_returns_same_object(self):
        inner = {"x": 1}
        doc = {"a": [inner]}
        assert resolve_pointer(doc, "/a/0") is inner

    def test_tilde01_key(self):
        doc = {"~1": "tilde-one", "/": "slash"}
        assert resolve_pointer(doc, "/~01") == "tilde-one"
        assert resolve_pointer(doc, "/~1") == "slash"

    def test_tilde_key_variants(self):
        doc = {"~0": "a", "~": "b"}
        assert resolve_pointer(doc, "/~00") == "a"
        assert resolve_pointer(doc, "/~0") == "b"

    def test_nested(self):
        doc = {"a": {"b": [{"c": [10, 20, 30]}]}}
        assert resolve_pointer(doc, "/a/b/0/c/2") == 30

    def test_percent_not_decoded(self):
        doc = {"a%20b": 1, "c d": 2}
        assert resolve_pointer(doc, "/a%20b") == 1
        assert resolve_pointer(doc, "/c d") == 2
        assert resolve_pointer(doc, "/c%20d", None) is None

    def test_missing_key_raises(self):
        with pytest.raises(JsonPointerException, match="member 'nope' not found"):
            resolve_pointer({"a": 1}, "/nope")

    def test_missing_key_default(self):
        assert resolve_pointer({"a": 1}, "/nope", "dflt") == "dflt"
        assert resolve_pointer({"a": 1}, "/a/b/c", None) is None

    def test_default_none_is_honoured(self):
        assert JsonPointer("/x").resolve({}, None) is None

    def test_default_falsy_values(self):
        for d in (0, "", [], False):
            assert resolve_pointer({}, "/x", d) is d

    def test_default_not_used_when_found(self):
        assert resolve_pointer({"a": 0}, "/a", 5) == 0

    def test_verbose_exception_includes_doc(self):
        jsonpointer.VERBOSE_EXCEPTIONS = True
        with pytest.raises(JsonPointerException) as ei:
            resolve_pointer({"secret": 1}, "/nope")
        assert "secret" in str(ei.value)

    def test_non_verbose_exception_hides_doc(self):
        jsonpointer.VERBOSE_EXCEPTIONS = False
        with pytest.raises(JsonPointerException) as ei:
            resolve_pointer({"secret": 1}, "/nope")
        assert "secret" not in str(ei.value)
        assert str(ei.value) == "member 'nope' not found"

    def test_index_out_of_bounds(self):
        with pytest.raises(JsonPointerException, match="out of bounds"):
            resolve_pointer([1, 2], "/2")
        assert resolve_pointer([1, 2], "/2", "d") == "d"

    @pytest.mark.parametrize("idx", ["01", "00", "-1", "+1", " 1", "1 ", "1.0", "a", "1a", "", "0x1", "١"])
    def test_invalid_index(self, idx):
        with pytest.raises(JsonPointerException, match="not a valid sequence index"):
            JsonPointer("/" + idx).resolve([1, 2, 3])

    def test_invalid_index_default(self):
        assert resolve_pointer([1], "/01", "d") == "d"

    @pytest.mark.parametrize("idx, val", [("0", "a"), ("1", "b"), ("10", "k")])
    def test_valid_indices(self, idx, val):
        lst = list("abcdefghijk")
        assert resolve_pointer(lst, "/" + idx) == val

    def test_dash_gives_end_of_list(self):
        lst = [1, 2]
        r = resolve_pointer(lst, "/-")
        assert isinstance(r, EndOfList)
        assert r.list_ is lst

    def test_dash_in_mapping_is_key(self):
        assert resolve_pointer({"-": 5}, "/-") == 5

    def test_tuple_is_sequence(self):
        assert resolve_pointer((1, 2, 3), "/1") == 2

    def test_numeric_key_in_dict_is_string(self):
        doc = {"0": "str", 0: "int"}
        assert resolve_pointer(doc, "/0") == "str"

    def test_string_is_not_container(self):
        with pytest.raises(JsonPointerException, match="non-container"):
            resolve_pointer({"a": "hello"}, "/a/0")
        assert resolve_pointer({"a": "hello"}, "/a/0", None) is None

    @pytest.mark.parametrize("leaf", [1, 1.5, None, True])
    def test_scalar_not_indexable(self, leaf):
        with pytest.raises(JsonPointerException, match="does not support indexing"):
            resolve_pointer({"a": leaf}, "/a/b")
        assert resolve_pointer({"a": leaf}, "/a/b", "d") == "d"

    def test_ordered_dict_and_mapping(self):
        assert resolve_pointer(OrderedDict(a=1), "/a") == 1
        assert resolve_pointer(UserDict(a=2), "/a") == 2

    def test_duck_typed_getitem(self):
        class Duck:
            def __getitem__(self, key):
                if key == "k":
                    return "v"
                raise KeyError(key)

        assert resolve_pointer({"d": Duck()}, "/d/k") == "v"
        with pytest.raises(JsonPointerException, match="member 'x' not found"):
            resolve_pointer(Duck(), "/x")

    def test_walk_directly(self):
        p = JsonPointer("")
        assert p.walk({"a": 1}, "a") == 1
        assert p.walk([5, 6], "1") == 6
        assert isinstance(p.walk([5, 6], "-"), EndOfList)


# ---------------------------------------------------------------- get_part / to_last

class TestGetPart:
    def test_mapping(self):
        assert JsonPointer.get_part({}, "01") == "01"

    def test_sequence(self):
        assert JsonPointer.get_part([], "12") == 12
        assert JsonPointer.get_part([], "0") == 0
        assert JsonPointer.get_part([], "-") == "-"

    def test_sequence_int_part(self):
        assert JsonPointer.get_part([1, 2], 1) == 1

    def test_sequence_invalid(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([], "01")
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part([], "x")

    def test_string(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part("abc", "0")

    def test_other(self):
        with pytest.raises(JsonPointerException):
            JsonPointer.get_part(5, "0")

    def test_to_last(self):
        doc = {"a": {"b": [1, 2]}}
        assert JsonPointer("").to_last(doc) == (doc, None)
        parent, part = JsonPointer("/a/b/1").to_last(doc)
        assert parent is doc["a"]["b"]
        assert part == 1
        parent, part = JsonPointer("/a/b").to_last(doc)
        assert parent is doc["a"]
        assert part == "b"
        parent, part = JsonPointer("/a").to_last(doc)
        assert parent is doc
        assert part == "a"


# ---------------------------------------------------------------- set

class TestSet:
    def test_set_existing_key(self):
        doc = {"a": {"b": 1}}
        res = set_pointer(doc, "/a/b", 2)
        assert res is doc
        assert doc == {"a": {"b": 2}}

    def test_set_new_key(self):
        doc = {"a": {}}
        set_pointer(doc, "/a/new", "x")
        assert doc == {"a": {"new": "x"}}

    def test_set_escaped_key(self):
        doc = {}
        set_pointer(doc, "/a~1b", 1)
        set_pointer(doc, "/m~0n", 2)
        set_pointer(doc, "/~01", 3)
        assert doc == {"a/b": 1, "m~n": 2, "~1": 3}

    def test_set_list_index(self):
        doc = {"l": [1, 2, 3]}
        set_pointer(doc, "/l/1", 20)
        assert doc == {"l": [1, 20, 3]}

    def test_set_list_append(self):
        doc = {"l": [1]}
        set_pointer(doc, "/l/-", 2)
        set_pointer(doc, "/l/-", 3)
        assert doc == {"l": [1, 2, 3]}

    def test_set_dash_in_dict(self):
        doc = {}
        set_pointer(doc, "/-", 1)
        assert doc == {"-": 1}

    def test_set_list_out_of_range(self):
        with pytest.raises(JsonPointerException, match="Invalid assignment target"):
            set_pointer([1, 2], "/5", 0)

    def test_set_list_index_equal_len(self):
        with pytest.raises(JsonPointerException):
            set_pointer([1, 2], "/2", 0)

    def test_set_list_invalid_index(self):
        with pytest.raises(JsonPointerException):
            set_pointer([1, 2], "/01", 0)

    def test_set_in_tuple(self):
        with pytest.raises(JsonPointerException, match="Invalid assignment target"):
            set_pointer({"t": (1, 2)}, "/t/0", 5)

    def test_set_in_string(self):
        with pytest.raises(JsonPointerException):
            set_pointer({"s": "abc"}, "/s/0", "x")

    def test_set_in_string_message(self):
        # parent is a string: get_part raises first via to_last
        with pytest.raises(JsonPointerException):
            JsonPointer("/0").set("abc", "x")

    def test_set_missing_parent(self):
        with pytest.raises(JsonPointerException):
            set_pointer({}, "/a/b", 1)

    def test_set_root_inplace_raises(self):
        with pytest.raises(JsonPointerException, match="Cannot set root"):
            set_pointer({"a": 1}, "", 5)

    def test_set_root_not_inplace(self):
        doc = {"a": 1}
        assert set_pointer(doc, "", 5, inplace=False) == 5
        assert doc == {"a": 1}

    def test_set_not_inplace(self):
        doc = {"a": {"b": [1, 2]}}
        orig = copy.deepcopy(doc)
        res = set_pointer(doc, "/a/b/0", 9, inplace=False)
        assert res == {"a": {"b": [9, 2]}}
        assert doc == orig
        assert res is not doc
        assert res["a"] is not doc["a"]

    def test_set_not_inplace_append(self):
        doc = [1]
        res = JsonPointer("/-").set(doc, 2, inplace=False)
        assert res == [1, 2]
        assert doc == [1]

    def test_set_inplace_default_true(self):
        doc = {"a": 1}
        JsonPointer("/a").set(doc, 2)
        assert doc == {"a": 2}

    def test_set_returns_doc(self):
        doc = [0]
        assert JsonPointer("/0").set(doc, 1) is doc

    def test_set_duck_typed(self):
        class Box:
            def __init__(self):
                self.d = {}

            def __getitem__(self, k):
                return self.d[k]

            def __setitem__(self, k, v):
                self.d[k] = v

        b = Box()
        JsonPointer("/x").set(b, 1)
        assert b.d == {"x": 1}

    def test_set_then_resolve(self):
        doc = {"foo": {"anArray": [{"prop": 44}]}}
        set_pointer(doc, "/foo/anArray/0/prop", 55)
        assert resolve_pointer(doc, "/foo/anArray/0/prop") == 55


# ---------------------------------------------------------------- path / repr / eq

class TestRepresentation:
    @pytest.mark.parametrize("ptr", ["", "/", "//", "/a", "/a/b/0", "/~0/0/~1", "/~01", "/a~1b/m~0n", "/ /%"])
    def test_path_roundtrip(self, ptr):
        p = JsonPointer(ptr)
        assert p.path == ptr
        assert str(p) == ptr
        assert JsonPointer(p.path) == p

    def test_repr(self):
        assert repr(JsonPointer("/a/~0")) == "JsonPointer('/a/~0')"
        assert repr(JsonPointer("")) == "JsonPointer('')"

    def test_repr_subclass(self):
        class Sub(JsonPointer):
            pass
        assert repr(Sub("/x")) == "Sub('/x')"

    def test_eq(self):
        assert JsonPointer("/a/b") == JsonPointer("/a/b")
        assert JsonPointer("/a/b") != JsonPointer("/a/c")
        assert JsonPointer("/a") != JsonPointer("/a/")
        assert JsonPointer("") != JsonPointer("/")

    def test_eq_other_types(self):
        assert JsonPointer("/a") != "/a"
        assert not (JsonPointer("/a") == "/a")
        assert JsonPointer("") != []
        assert JsonPointer("/a") != ["a"]

    def test_hash(self):
        assert hash(JsonPointer("/a/b")) == hash(JsonPointer("/a/b"))
        s = {JsonPointer("/a"), JsonPointer("/a"), JsonPointer("/b")}
        assert len(s) == 2
        assert {JsonPointer("/x"): 1}[JsonPointer("/x")] == 1

    def test_end_of_list_repr(self):
        assert repr(EndOfList([1, 2])) == "EndOfList([1, 2])"


# ---------------------------------------------------------------- from_parts / join

class TestConstruction:
    def test_from_parts(self):
        p = JsonPointer.from_parts(["a", "~", "/", 0])
        assert p.path == "/a/~0/~1/0"
        assert p.parts == ["a", "~", "/", "0"]

    def test_from_parts_empty(self):
        p = JsonPointer.from_parts([])
        assert p.path == ""
        assert p.parts == []

    def test_from_parts_tricky(self):
        p = JsonPointer.from_parts(["~1", "~0", ""])
        assert p.parts == ["~1", "~0", ""]
        assert p.path == "/~01/~00/"

    def test_from_parts_generator(self):
        p = JsonPointer.from_parts(x for x in ["a", "b"])
        assert p.parts == ["a", "b"]

    def test_from_parts_subclass(self):
        class Sub(JsonPointer):
            pass
        assert type(Sub.from_parts(["a"])) is Sub

    def test_join_str(self):
        assert JsonPointer("/a").join("/b/c") == JsonPointer("/a/b/c")

    def test_join_pointer(self):
        assert JsonPointer("/a").join(JsonPointer("/b")) == JsonPointer("/a/b")

    def test_join_list(self):
        assert JsonPointer("/a").join(["b", "c/d"]).path == "/a/b/c~1d"

    def test_join_escaped(self):
        assert JsonPointer("/a").join("/~01").parts == ["a", "~1"]

    def test_join_does_not_mutate(self):
        p = JsonPointer("/a")
        p.join("/b")
        assert p.parts == ["a"]

    def test_join_root(self):
        assert JsonPointer("").join("/x") == JsonPointer("/x")
        assert JsonPointer("/x").join("") == JsonPointer("/x")

    def test_truediv(self):
        assert JsonPointer("/a") / "/b" == JsonPointer("/a/b")
        assert JsonPointer("/a") / ["b", 1] == JsonPointer("/a/b/1")
        assert (JsonPointer("/a") / JsonPointer("/b")).path == "/a/b"

    def test_join_invalid_suffix(self):
        with pytest.raises(JsonPointerException):
            JsonPointer("/a").join("b")
        with pytest.raises(JsonPointerException, match="Invalid suffix"):
            JsonPointer("/a").join(5)

    def test_join_subclass(self):
        class Sub(JsonPointer):
            pass
        assert type(Sub("/a").join("/b")) is Sub


# ---------------------------------------------------------------- contains

class TestContains:
    def test_contains_prefix(self):
        assert JsonPointer("/a/b/c").contains(JsonPointer("/a/b"))
        assert JsonPointer("/a/b/c").contains(JsonPointer("/a"))
        assert JsonPointer("/a/b/c").contains(JsonPointer(""))

    def test_contains_self(self):
        assert JsonPointer("/a/b").contains(JsonPointer("/a/b"))

    def test_not_contains(self):
        assert not JsonPointer("/a/b").contains(JsonPointer("/a/b/c"))
        assert not JsonPointer("/a/b").contains(JsonPointer("/a/c"))
        assert not JsonPointer("/a/b").contains(JsonPointer("/b"))
        assert not JsonPointer("").contains(JsonPointer("/a"))

    def test_contains_is_part_based(self):
        # string prefix, but not part prefix
        assert not JsonPointer("/ab").contains(JsonPointer("/a"))

    def test_in_operator(self):
        assert JsonPointer("/a") in JsonPointer("/a/b")
        assert JsonPointer("/a/b") not in JsonPointer("/a")
