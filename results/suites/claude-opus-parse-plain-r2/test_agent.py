import pickle
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse
from parse import (
    FixedTzOffset,
    Match,
    Parser,
    RepeatedNameError,
    Result,
    ResultIterator,
    TooManyFields,
    compile,
    convert_first,
    date_convert,
    extract_format,
    findall,
    int_convert,
    percentage,
    search,
    strf_date_convert,
    with_pattern,
)


def tz(minutes, name):
    return FixedTzOffset(minutes, name)


# --------------------------------------------------------------------------
# Basic parsing
# --------------------------------------------------------------------------


class TestBasicParse:
    def test_no_fields_exact(self):
        r = parse.parse("hello", "hello")
        assert r is not None
        assert r.fixed == ()
        assert r.named == {}

    def test_no_fields_mismatch(self):
        assert parse.parse("hello", "world") is None

    def test_parse_requires_full_match(self):
        assert parse.parse("hello", "hello world") is None
        assert parse.parse("hello", "say hello") is None

    def test_fixed_field(self):
        r = parse.parse("hello {}", "hello world")
        assert r.fixed == ("world",)
        assert r[0] == "world"

    def test_two_fixed_fields(self):
        r = parse.parse("{} {}", "a b")
        assert r.fixed == ("a", "b")

    def test_fixed_fields_non_greedy(self):
        r = parse.parse("{}-{}", "a-b-c")
        assert r.fixed == ("a", "b-c")

    def test_named_field(self):
        r = parse.parse("hello {name}", "hello world")
        assert r.named == {"name": "world"}
        assert r["name"] == "world"
        assert r.fixed == ()

    def test_mixed_fields(self):
        r = parse.parse("{} {name} {}", "a b c")
        assert r.fixed == ("a", "c")
        assert r.named == {"name": "b"}

    def test_numbered_field_is_fixed(self):
        r = parse.parse("{0} {1}", "a b")
        assert r.fixed == ("a", "b")

    def test_escaped_braces(self):
        r = parse.parse("{{hello}} {}", "{hello} world")
        assert r.fixed == ("world",)
        assert parse.parse("{{}}", "{}").fixed == ()
        assert parse.parse("{{x}}", "x") is None

    def test_regex_special_chars_literal(self):
        r = parse.parse("a.b*c+d?(e)[f]^g$h|i\\j {}", "a.b*c+d?(e)[f]^g$h|i\\j x")
        assert r.fixed == ("x",)
        # The "." must be literal, not any char
        assert parse.parse("a.b {}", "axb x") is None
        assert parse.parse("a+ {}", "aa x") is None
        assert parse.parse("(a) {}", "a x") is None
        assert parse.parse("a! {}", "a! x").fixed == ("x",)

    def test_case_insensitive_by_default(self):
        assert parse.parse("Hello {}", "hELLO world").fixed == ("world",)

    def test_case_sensitive(self):
        assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
        assert parse.parse("Hello {}", "Hello world", case_sensitive=True).fixed == (
            "world",
        )

    def test_dotall(self):
        r = parse.parse("{}", "a\nb")
        assert r.fixed == ("a\nb",)
        r = parse.parse("{}", "a\nb", case_sensitive=True)
        assert r.fixed == ("a\nb",)

    def test_trailing_newline_not_accepted(self):
        # \Z should not allow trailing newline the way $ does
        assert parse.parse("hello", "hello\n") is None

    def test_empty_string_no_match_for_field(self):
        assert parse.parse("{}", "") is None

    def test_named_dotted(self):
        r = parse.parse("{a.b}", "x")
        assert r.named == {"a.b": "x"}

    def test_named_with_hyphen(self):
        r = parse.parse("{a-b} {c}", "x y")
        assert r.named == {"a-b": "x", "c": "y"}

    def test_named_dotted_collision(self):
        r = parse.parse("{a.b} {a_b}", "x y")
        assert r.named == {"a.b": "x", "a_b": "y"}
        r = parse.parse("{a_b} {a.b}", "x y")
        assert r.named == {"a_b": "x", "a.b": "y"}

    def test_hyphen_collision(self):
        r = parse.parse("{a_b} {a-b}", "x y")
        assert r.named == {"a_b": "x", "a-b": "y"}

    def test_underscore_collision(self):
        r = parse.parse("{a-b} {a_b}", "x y")
        assert r.named == {"a-b": "x", "a_b": "y"}

    def test_triple_collision(self):
        r = parse.parse("{a.b} {a_b} {a-b}", "x y z")
        assert r.named == {"a.b": "x", "a_b": "y", "a-b": "z"}

    def test_dict_style_fields(self):
        r = parse.parse("{a[b]} {a[c]}", "x y")
        assert r.named == {"a": {"b": "x", "c": "y"}}
        assert r["a"]["b"] == "x"

    def test_nested_dict_style_fields(self):
        r = parse.parse("{a[b][c]} {a[b][d]} {e}", "1 2 3")
        assert r.named == {"a": {"b": {"c": "1", "d": "2"}}, "e": "3"}

    def test_dict_style_with_type(self):
        r = parse.parse("{a[b]:d}", "12")
        assert r.named == {"a": {"b": 12}}

    def test_repeated_name(self):
        r = parse.parse("{a} {a}", "x x")
        assert r.named == {"a": "x"}
        assert parse.parse("{a} {a}", "x y") is None

    def test_repeated_name_typed(self):
        r = parse.parse("{a:d} {a:d}", "1 1")
        assert r.named == {"a": 1}
        assert parse.parse("{a:d} {a:d}", "1 2") is None

    def test_repeated_name_type_mismatch(self):
        with pytest.raises(RepeatedNameError):
            parse.compile("{a:d} {a:w}")
        with pytest.raises(ValueError):
            parse.compile("{a} {a:d}")

    def test_repeated_name_error_is_value_error(self):
        assert issubclass(RepeatedNameError, ValueError)
        assert issubclass(TooManyFields, ValueError)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            parse.compile("{:q}")
        with pytest.raises(ValueError):
            parse.parse("{:yyy}", "x")

    def test_unicode(self):
        r = parse.parse("{} ünïcödé {}", "α ünïcödé β")
        assert r.fixed == ("α", "β")


# --------------------------------------------------------------------------
# Result, Match objects
# --------------------------------------------------------------------------


class TestResult:
    def test_getitem_and_slice(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r[0] == "a"
        assert r[-1] == "c"
        assert r[1:] == ("b", "c")

    def test_getitem_named(self):
        r = parse.parse("{x} {y}", "1 2")
        assert r["x"] == "1"
        with pytest.raises(KeyError):
            r["z"]

    def test_getitem_index_error(self):
        r = parse.parse("{}", "a")
        with pytest.raises(IndexError):
            r[1]

    def test_contains(self):
        r = parse.parse("{x}", "1")
        assert "x" in r
        assert "y" not in r

    def test_repr(self):
        r = parse.parse("{} {x}", "a b")
        assert repr(r) == "<Result ('a',) {'x': 'b'}>"

    def test_spans(self):
        r = parse.parse("hello {} {name} {}", "hello a bb ccc")
        assert r.spans == {0: (6, 7), "name": (8, 10), 1: (11, 14)}

    def test_spans_with_typed_fields(self):
        # typed fields add extra inner regex groups; spans must still be right
        r = parse.parse("{:n} {:x} {:g} {val:d}", "1,000 ff 1.5e3 42")
        assert r.fixed == (1000, 255, 1500.0)
        assert r["val"] == 42
        assert r.spans[0] == (0, 5)
        assert r.spans[1] == (6, 8)
        assert r.spans[2] == (9, 14)
        assert r.spans["val"] == (15, 17)

    def test_spans_with_padding(self):
        r = parse.parse("[{:>}]", "[   ab]")
        assert r.fixed == ("ab",)
        assert r.spans[0] == (4, 6)

    def test_width_includes_padding_in_value(self):
        # width is a minimum length of the captured text, padding included
        r = parse.parse("[{:>5}]", "[   ab]")
        assert r.fixed == ("   ab",)

    def test_result_constructor(self):
        r = Result((1,), {"a": 2}, {})
        assert r[0] == 1
        assert r["a"] == 2
        assert "a" in r

    def test_evaluate_result_false(self):
        m = parse.parse("{:d} {x}", "12 y", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert isinstance(r, Result)
        assert r.fixed == (12,)
        assert r.named == {"x": "y"}

    def test_evaluate_result_false_no_match(self):
        assert parse.parse("{:d}", "x", evaluate_result=False) is None

    def test_match_object_attributes(self):
        p = compile("{:d}")
        m = p.parse("5", evaluate_result=False)
        assert m.parser is p
        assert m.match.group(0) == "5"


# --------------------------------------------------------------------------
# Parser object
# --------------------------------------------------------------------------


class TestParser:
    def test_compile_returns_parser(self):
        p = compile("{x:d}")
        assert isinstance(p, Parser)
        assert p.parse("3").named == {"x": 3}
        assert p.parse("a") is None

    def test_format_property(self):
        assert compile("a {} b").format == "a {} b"

    def test_named_and_fixed_fields(self):
        p = compile("{} {a} {} {b.c}")
        assert p.fixed_fields == [0, 2]
        assert p.named_fields == ["a", "b_c"]

    def test_fields_lists_are_copies(self):
        p = compile("{} {a}")
        p.fixed_fields.append(99)
        p.named_fields.append("zz")
        assert p.fixed_fields == [0]
        assert p.named_fields == ["a"]

    def test_repr_short(self):
        assert repr(compile("{a}")) == "<Parser '{a}'>"

    def test_repr_exactly_20(self):
        f = "a" * 20
        assert repr(compile(f)) == "<Parser %r>" % f

    def test_repr_long(self):
        f = "a" * 21
        assert repr(compile(f)) == "<Parser %r>" % ("a" * 17 + "...")

    def test_case_sensitive_compile(self):
        p = compile("A{}", case_sensitive=True)
        assert p.parse("a1") is None
        assert p.parse("A1").fixed == ("1",)
        p = compile("A{}")
        assert p.parse("a1").fixed == ("1",)

    def test_case_sensitive_search(self):
        p = compile("A{:d}", case_sensitive=True)
        assert p.search("xa1") is None
        assert p.search("xA1").fixed == (1,)

    def test_reuse(self):
        p = compile("{:d}")
        assert p.parse("1").fixed == (1,)
        assert p.parse("2").fixed == (2,)
        assert p.search("x 3 y").fixed == (3,)
        assert p.parse("4").fixed == (4,)

    def test_pickle_parser(self):
        p = compile("{a:d} {b:x} {c:f} {d:ti}")
        p2 = pickle.loads(pickle.dumps(p))
        r = p2.parse("1 ff 1.5 2020-01-02")
        assert r.named == {"a": 1, "b": 255, "c": 1.5, "d": datetime(2020, 1, 2)}

    def test_many_simple_fields_ok(self):
        fmt = " ".join("{}" for _ in range(50))
        s = " ".join(str(i) for i in range(50))
        r = parse.parse(fmt, s)
        assert r.fixed == tuple(str(i) for i in range(50))


# --------------------------------------------------------------------------
# search / findall
# --------------------------------------------------------------------------


class TestSearch:
    def test_basic(self):
        r = search("age: {:d}\n", "name: Rufus\nage: 42\ncolor: red\n")
        assert r.fixed == (42,)

    def test_no_match(self):
        assert search("{:d}", "abc") is None

    def test_pos(self):
        r = search("{:d}", "1 2 3", pos=1)
        assert r.fixed == (2,)

    def test_endpos(self):
        assert search("x{:d}", "ab x12", endpos=3) is None
        r = search("x{:d}", "ab x12", endpos=5)
        assert r.fixed == (1,)

    def test_pos_and_endpos(self):
        r = search("{:d}", "12345", pos=1, endpos=3)
        assert r.fixed == (23,)

    def test_evaluate_result_false(self):
        m = search("{:d}", "a 12 b", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.evaluate_result().fixed == (12,)

    def test_evaluate_result_false_no_match(self):
        assert search("{:d}", "abc", evaluate_result=False) is None

    def test_case_sensitive(self):
        assert search("X{:d}", "x1", case_sensitive=True) is None
        assert search("X{:d}", "x1").fixed == (1,)

    def test_extra_types(self):
        @with_pattern(r"[a-z]+")
        def word(s):
            return s.upper()

        r = search("<{:W}>", "xx <abc> yy", extra_types={"W": word})
        assert r.fixed == ("ABC",)

    def test_search_spans(self):
        r = search("{:d}", "abc:123 def")
        assert r.spans[0] == (4, 7)
        r = search("x{name}y", "aaxbcyd")
        assert r.spans["name"] == (3, 5)


class TestFindall:
    def test_basic(self):
        rs = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
        assert [r.fixed[0] for r in rs] == ["some ", "bold", " text"]

    def test_returns_iterator(self):
        it = findall("{:d}", "1 2")
        assert isinstance(it, ResultIterator)
        assert iter(it) is it
        assert next(it).fixed == (1,)
        assert it.next().fixed == (2,)
        with pytest.raises(StopIteration):
            next(it)

    def test_join(self):
        s = "".join(r.fixed[0] for r in findall(">{}<", "<p>some <b>bold</b> text</p>"))
        assert s == "some bold text"

    def test_typed(self):
        assert [r[0] for r in findall("{:d}", "a1b22c333")] == [1, 22, 333]

    def test_pos_endpos(self):
        assert [r[0] for r in findall("{:d}", "1 2 3 4", pos=2, endpos=5)] == [2, 3]

    def test_no_matches(self):
        assert list(findall("{:d}", "abc")) == []

    def test_evaluate_result_false(self):
        ms = list(findall("{:d}", "1 2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in ms)
        assert [m.evaluate_result().fixed for m in ms] == [(1,), (2,)]

    def test_case_sensitive(self):
        assert list(findall("A{:d}", "a1 A2", case_sensitive=True))[0].fixed == (2,)
        assert len(list(findall("A{:d}", "a1 A2", case_sensitive=True))) == 1
        assert len(list(findall("A{:d}", "a1 A2"))) == 2

    def test_extra_types(self):
        @with_pattern(r"\d")
        def digit(s):
            return int(s) * 10

        rs = findall("{:D1}", "1a2", extra_types={"D1": digit})
        assert [r[0] for r in rs] == [10, 20]

    def test_parser_findall(self):
        p = compile("{:d}")
        assert [r[0] for r in p.findall("5 6 7", pos=2)] == [6, 7]
        assert [r[0] for r in p.findall("5 6 7", endpos=3)] == [5, 6]
        ms = list(p.findall("5 6", evaluate_result=False))
        assert all(isinstance(m, Match) for m in ms)


# --------------------------------------------------------------------------
# Integer-ish types
# --------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("0", 0),
            ("123", 123),
            ("-123", -123),
            ("+123", 123),
            (" 123", 123),
            ("0x1F", 31),
            ("0X1f", 31),
            ("0b101", 5),
            ("0B101", 5),
            ("0o17", 15),
            ("0O17", 15),
            ("-0x10", -16),
            ("+0b11", 3),
            ("-0o10", -8),
            ("007", 7),
            ("00", 0),
        ],
    )
    def test_d(self, s, expected):
        r = parse.parse("{:d}", s)
        assert r is not None
        assert r[0] == expected
        assert type(r[0]) is int

    @pytest.mark.parametrize("s", ["", "abc", "1.5", "0x", "1 2", "0b2"])
    def test_d_rejects(self, s):
        assert parse.parse("{:d}", s) is None

    def test_d_grouping_comma(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
        assert parse.parse("{:d}", "1,234") is None

    def test_d_grouping_underscore(self):
        assert parse.parse("{:_d}", "1_234_567")[0] == 1234567
        assert parse.parse("{:_d}", "1,234") is None

    def test_d_grouping_hex(self):
        assert parse.parse("{:_d}", "0xff_ff")[0] == 0xFFFF

    def test_d_width(self):
        assert parse.parse("{:2d}{:2d}", "1234").fixed == (12, 34)
        assert parse.parse("{:3d}", "1234") is None
        assert parse.parse("{:3d}", "  4")[0] == 4

    def test_d_width_search(self):
        r = search("{:2d}", "abc 12345")
        assert r[0] == 12

    def test_d_zero_padded(self):
        assert parse.parse("{:03d}", "007")[0] == 7
        assert parse.parse("{:05d}", "-0042")[0] == -42

    def test_d_fill_align_eq(self):
        assert parse.parse("{:=5d}", "00042")[0] == 42
        assert parse.parse("{:x=5d}", "xxx42")[0] == 42
        assert parse.parse("{:=5d}", "-0042")[0] == -42

    def test_d_left_align(self):
        assert parse.parse("[{:<5d}]", "[42   ]")[0] == 42

    def test_d_center_align(self):
        assert parse.parse("[{:^6d}]", "[  42  ]")[0] == 42

    def test_d_right_align_default_with_width(self):
        assert parse.parse("[{:5d}]", "[   42]")[0] == 42

    def test_d_fill_char(self):
        assert parse.parse("[{:*>5d}]", "[***42]")[0] == 42
        assert parse.parse("[{:*<5d}]", "[42***]")[0] == 42
        assert parse.parse("[{:*^6d}]", "[**42**]")[0] == 42

    def test_d_sign_in_format(self):
        assert parse.parse("{:+d}", "+5")[0] == 5
        assert parse.parse("{:-d}", "-5")[0] == -5
        assert parse.parse("{: d}", " 5")[0] == 5

    def test_n(self):
        assert parse.parse("{:n}", "1,234,567")[0] == 1234567
        assert parse.parse("{:n}", "1.234.567")[0] == 1234567
        assert parse.parse("{:n}", "123")[0] == 123
        assert parse.parse("{:n}", "-1,000")[0] == -1000
        assert parse.parse("{:n}", "1,23") is None
        assert parse.parse("{:n}", "1234") is None

    def test_n_followed_by_field(self):
        r = parse.parse("{:n} {}", "1,000 x")
        assert r.fixed == (1000, "x")

    def test_b(self):
        assert parse.parse("{:b}", "1010")[0] == 10
        assert parse.parse("{:b}", "0b1010")[0] == 10
        assert parse.parse("{:b}", "0B11")[0] == 3
        assert parse.parse("{:b}", "-101")[0] == -5
        assert parse.parse("{:b}", "102") is None

    def test_o(self):
        assert parse.parse("{:o}", "17")[0] == 15
        assert parse.parse("{:o}", "0o17")[0] == 15
        assert parse.parse("{:o}", "0O17")[0] == 15
        assert parse.parse("{:o}", "-10")[0] == -8
        assert parse.parse("{:o}", "8") is None

    def test_x(self):
        assert parse.parse("{:x}", "ff")[0] == 255
        assert parse.parse("{:x}", "0xff")[0] == 255
        assert parse.parse("{:X}", "FF")[0] == 255
        assert parse.parse("{:X}", "0XAB")[0] == 171
        assert parse.parse("{:x}", "-1a")[0] == -26
        assert parse.parse("{:x}", "g") is None

    def test_x_with_digit_b(self):
        # hex digits 'b' must not be confused with binary prefix
        assert parse.parse("{:x}", "0b")[0] == 11
        assert parse.parse("{:x}", "b1")[0] == 177

    def test_multiple_typed_fields_group_counting(self):
        r = parse.parse("{:b} {:o} {:x} {:n} {:d}", "11 17 ff 1,000 9")
        assert r.fixed == (3, 15, 255, 1000, 9)

    def test_named_typed_group_counting(self):
        r = parse.parse(
            "{a:b} {b:o} {c:x} {d:n} {e:g} {f:%} {g:d}", "11 17 ff 1,000 2.5 50% 9"
        )
        assert r.named == {
            "a": 3,
            "b": 15,
            "c": 255,
            "d": 1000,
            "e": 2.5,
            "f": 0.5,
            "g": 9,
        }


class TestIntConvert:
    def test_auto_detect(self):
        ic = int_convert()
        assert ic("123", None) == 123
        assert ic("0x1f", None) == 31
        assert ic("0b11", None) == 3
        assert ic("0o17", None) == 15
        assert ic("-0x10", None) == -16
        assert ic("+0b11", None) == 3

    def test_short_strings_are_decimal(self):
        ic = int_convert()
        assert ic("0", None) == 0
        assert ic("00", None) == 0
        assert ic("07", None) == 7
        assert ic("-07", None) == -7

    def test_explicit_base(self):
        assert int_convert(16)("ff", None) == 255
        assert int_convert(2)("101", None) == 5
        assert int_convert(8)("17", None) == 15
        assert int_convert(10)("1,234", None) == 1234
        assert int_convert(16)("0xff", None) == 255

    def test_ignores_junk(self):
        assert int_convert(10)("1_000", None) == 1000
        assert int_convert()("1 000", None) == 1000

    def test_uppercase(self):
        assert int_convert(16)("FF", None) == 255
        assert int_convert()("0XFF", None) == 255

    def test_base_36(self):
        assert int_convert(36)("z", None) == 35

    def test_sign(self):
        assert int_convert()("-5", None) == -5
        assert int_convert()("+5", None) == 5


# --------------------------------------------------------------------------
# Float-ish types
# --------------------------------------------------------------------------


class TestFloats:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("1.5", 1.5),
            (".5", 0.5),
            ("-1.5", -1.5),
            ("+1.5", 1.5),
            (" 1.5", 1.5),
            ("123.456", 123.456),
        ],
    )
    def test_f(self, s, expected):
        r = parse.parse("{:f}", s)
        assert r[0] == expected
        assert type(r[0]) is float

    @pytest.mark.parametrize("s", ["1", "1.", "abc", "", "1.5.5"])
    def test_f_rejects(self, s):
        assert parse.parse("{:f}", s) is None

    def test_f_nan_inf(self):
        import math

        assert math.isnan(parse.parse("{:f}", "nan")[0])
        assert math.isnan(parse.parse("{:f}", "NAN")[0])
        assert parse.parse("{:f}", "inf")[0] == float("inf")
        assert parse.parse("{:f}", "INF")[0] == float("inf")
        assert parse.parse("{:f}", "-inf")[0] == float("-inf")
        assert parse.parse("{:f}", "+inf")[0] == float("inf")

    def test_f_precision_zero(self):
        assert parse.parse("{:.0f}", "20")[0] == 20.0
        assert parse.parse("{:.0f}", "-20")[0] == -20.0
        assert parse.parse("{:.0f}", "20.5") is None

    def test_f_precision_nonzero(self):
        assert parse.parse("{:.2f}", "3.14")[0] == 3.14
        assert parse.parse("{:.2f}", "3") is None

    def test_f_width(self):
        assert parse.parse("[{:8.2f}]", "[    3.14]")[0] == 3.14

    def test_F_decimal(self):
        r = parse.parse("{:F}", "1.1")
        assert r[0] == Decimal("1.1")
        assert type(r[0]) is Decimal
        assert parse.parse("{:F}", "-0.25")[0] == Decimal("-0.25")

    def test_F_precision_zero(self):
        r = parse.parse("{:.0F}", "7")
        assert r[0] == Decimal("7")
        assert type(r[0]) is Decimal

    def test_e(self):
        assert parse.parse("{:e}", "1.0e10")[0] == 1.0e10
        assert parse.parse("{:e}", "1.5E-3")[0] == 1.5e-3
        assert parse.parse("{:e}", "-1.5e+3")[0] == -1500.0
        assert parse.parse("{:E}", "2.5E2")[0] == 250.0
        assert parse.parse("{:e}", "1.5") is None
        assert parse.parse("{:e}", "1e10") is None
        assert parse.parse("{:e}", "inf")[0] == float("inf")
        assert parse.parse("{:e}", "-inf")[0] == float("-inf")

    def test_e_nan(self):
        import math

        assert math.isnan(parse.parse("{:e}", "nan")[0])

    def test_g(self):
        assert parse.parse("{:g}", "1")[0] == 1.0
        assert type(parse.parse("{:g}", "1")[0]) is float
        assert parse.parse("{:g}", "1.5")[0] == 1.5
        assert parse.parse("{:g}", "1e10")[0] == 1e10
        assert parse.parse("{:g}", "1.5e-3")[0] == 1.5e-3
        assert parse.parse("{:g}", "-2.5")[0] == -2.5
        assert parse.parse("{:G}", "1E5")[0] == 1e5
        assert parse.parse("{:g}", "inf")[0] == float("inf")
        assert parse.parse("{:g}", "abc") is None

    def test_g_followed_by_fields(self):
        r = parse.parse("{:g} {} {:d}", "1.5e3 x 4")
        assert r.fixed == (1500.0, "x", 4)

    def test_percentage(self):
        assert parse.parse("{:%}", "50%")[0] == 0.5
        assert parse.parse("{:%}", "12.5%")[0] == 0.125
        assert parse.parse("{:%}", "-10%")[0] == -0.1
        assert parse.parse("{:%}", "50") is None

    def test_percentage_function(self):
        assert percentage("25%", None) == 0.25
        assert percentage("100%", None) == 1.0

    def test_percentage_then_field(self):
        r = parse.parse("{:%} {:d}", "12.5% 3")
        assert r.fixed == (0.125, 3)

    def test_zero_pad_float(self):
        assert parse.parse("{:=8.2f}", "00003.14")[0] == 3.14
        assert parse.parse("{:=8.2f}", "-0003.14")[0] == -3.14


# --------------------------------------------------------------------------
# String-ish types
# --------------------------------------------------------------------------


class TestStrings:
    def test_w(self):
        assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
        assert parse.parse("{:w}", "abc def") is None

    def test_W(self):
        assert parse.parse("{:W}", "  !")[0] == "  !"
        assert parse.parse("{:W}", "a") is None

    def test_s(self):
        assert parse.parse("{:s}", " \t\n")[0] == " \t\n"
        assert parse.parse("{:s}", "a") is None

    def test_S(self):
        assert parse.parse("{:S}", "abc!")[0] == "abc!"
        assert parse.parse("{:S}", "a b") is None

    def test_D(self):
        assert parse.parse("{:D}", "abc")[0] == "abc"
        assert parse.parse("{:D}", "a1") is None

    def test_l(self):
        assert parse.parse("{:l}", "Hello")[0] == "Hello"
        assert parse.parse("{:l}", "abc1") is None
        assert parse.parse("{:l}", "a_b") is None

    def test_types_are_strings(self):
        for fmt, s in [("{:w}", "a"), ("{:S}", "a"), ("{:l}", "a"), ("{:D}", "a")]:
            assert parse.parse(fmt, s)[0] == "a"

    def test_width_min(self):
        r = parse.parse("{:4}{}", "abcdef")
        assert r.fixed == ("abcd", "ef")
        assert parse.parse("{:4}", "abc") is None

    def test_precision_max(self):
        r = parse.parse("{:.2}{}", "abcdef")
        assert r.fixed == ("a", "bcdef")
        r = parse.parse("{:.2}", "ab")
        assert r.fixed == ("ab",)
        assert parse.parse("{:.2}", "abc") is None

    def test_width_and_precision(self):
        r = parse.parse("{:2.4}{}", "abcdef")
        assert r.fixed == ("ab", "cdef")
        assert parse.parse("{:2.4}", "a") is None
        assert parse.parse("{:2.4}", "abcd").fixed == ("abcd",)
        assert parse.parse("{:2.4}", "abcde") is None

    def test_align_left(self):
        assert parse.parse("[{:<}]", "[ab   ]")[0] == "ab"
        assert parse.parse("[{:<5}]", "[ab   ]")[0] == "ab   "

    def test_align_right(self):
        assert parse.parse("[{:>}]", "[   ab]")[0] == "ab"

    def test_align_center(self):
        assert parse.parse("[{:^}]", "[  ab  ]")[0] == "ab"
        assert parse.parse("[{:^6}]", "[  ab  ]")[0] == "  ab  "

    def test_width_default_right_align(self):
        assert parse.parse("[{:5}]", "[   ab]")[0] == "   ab"
        assert parse.parse("[{:5}]", "[ ab]") is None

    def test_no_align_no_strip(self):
        assert parse.parse("[{}]", "[  ab  ]")[0] == "  ab  "

    def test_fill(self):
        assert parse.parse("[{:x<}]", "[abxxx]")[0] == "ab"
        assert parse.parse("[{:x>}]", "[xxxab]")[0] == "ab"
        assert parse.parse("[{:x^}]", "[xxabxx]")[0] == "ab"

    @pytest.mark.parametrize("fill", list(".\\+?*[]()$"))
    def test_special_fill(self, fill):
        r = parse.parse("|{:%s<}|" % fill, "|ab%s%s|" % (fill, fill))
        assert r is not None
        assert r[0] == "ab"

    def test_dot_fill_literal(self):
        # escaped "." fill must not match arbitrary characters
        assert parse.parse("{:.>}", "...ab")[0] == "ab"
        assert parse.parse("{:.>}", "xyzab")[0] == "xyzab"

    def test_fill_star_right(self):
        assert parse.parse("{:*>}", "***ab")[0] == "ab"

    def test_fill_with_type_s(self):
        assert parse.parse("[{:-<w}]", "[ab--]")[0] == "ab"

    def test_named_with_format(self):
        r = parse.parse("{name:>}", "     hello")
        assert r["name"] == "hello"


# --------------------------------------------------------------------------
# Date/time types
# --------------------------------------------------------------------------


class TestDateTime:
    def test_ti_date_only(self):
        assert parse.parse("{:ti}", "2011-11-21")[0] == datetime(2011, 11, 21)

    def test_ti_datetime(self):
        assert parse.parse("{:ti}", "2011-11-21T10:21")[0] == datetime(
            2011, 11, 21, 10, 21
        )
        assert parse.parse("{:ti}", "2011-11-21 10:21:36")[0] == datetime(
            2011, 11, 21, 10, 21, 36
        )

    def test_ti_microseconds(self):
        assert parse.parse("{:ti}", "2011-11-21T10:21:36.5")[0] == datetime(
            2011, 11, 21, 10, 21, 36, 500000
        )
        assert parse.parse("{:ti}", "2011-11-21T10:21:36.123456789")[0] == datetime(
            2011, 11, 21, 10, 21, 36, 123456
        )
        assert parse.parse("{:ti}", "2011-11-21T10:21:36.000001")[0] == datetime(
            2011, 11, 21, 10, 21, 36, 1
        )

    def test_ti_utc(self):
        r = parse.parse("{:ti}", "2011-11-21T10:21:36Z")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(0, "UTC"))
        assert r.utcoffset() == timedelta(0)
        assert r.tzinfo == tz(0, "UTC")

    def test_ti_tz_offset(self):
        r = parse.parse("{:ti}", "2011-11-21T10:21:36+10:30")[0]
        assert r.utcoffset() == timedelta(hours=10, minutes=30)
        assert r.tzinfo == tz(630, "+10:30")
        r = parse.parse("{:ti}", "2011-11-21T10:21:36-0130")[0]
        assert r.utcoffset() == -timedelta(hours=1, minutes=30)
        r = parse.parse("{:ti}", "2011-11-21T10:21:36 +1000")[0]
        assert r.utcoffset() == timedelta(hours=10)

    def test_ti_named_and_fields_after(self):
        r = parse.parse("{a:ti} {b:d}", "2011-11-21T10:21:36Z 5")
        assert r["b"] == 5
        assert r["a"] == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(0, "UTC"))

    def test_ti_fixed_and_fields_after(self):
        r = parse.parse("{:ti} {:d}", "2011-11-21 5")
        assert r.fixed == (datetime(2011, 11, 21), 5)

    def test_ti_rejects(self):
        assert parse.parse("{:ti}", "21-11-2011") is None

    def test_te(self):
        r = parse.parse("{:te}", "Mon, 21 Nov 2011 10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(600, "+1000"))
        r = parse.parse("{:te}", "21 Nov 2011 10:21:36 -0500")[0]
        assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 10, 21, 36)
        assert r.utcoffset() == timedelta(hours=-5)

    def test_te_then_field(self):
        r = parse.parse("{:te} {:d}", "21 Nov 2011 10:21:36 +1000 7")
        assert r[1] == 7

    def test_th(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(600, "+1000"))
        r = parse.parse("{:th}", "21-Nov-2011:00:07:01 -0300")[0]
        assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 0, 7, 1)
        assert r.utcoffset() == timedelta(hours=-3)

    def test_th_then_field(self):
        r = parse.parse("{:th} {:d}", "21/Nov/2011:10:21:36 +1000 7")
        assert r[1] == 7

    def test_tc(self):
        r = parse.parse("{:tc}", "Mon Nov 21 10:21:36 2011")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36)
        r = parse.parse("{:tc}", "Sun Jan  2 01:02:03 2000")[0]
        assert r == datetime(2000, 1, 2, 1, 2, 3)

    def test_tc_then_field(self):
        r = parse.parse("{:tc} {:d}", "Mon Nov 21 10:21:36 2011 9")
        assert r[1] == 9

    def test_tg(self):
        assert parse.parse("{:tg}", "21/11/2011")[0] == datetime(2011, 11, 21)
        assert parse.parse("{:tg}", "21-11-2011 10:21")[0] == datetime(
            2011, 11, 21, 10, 21
        )
        assert parse.parse("{:tg}", "21/11/2011 10:21:36 AM")[0] == datetime(
            2011, 11, 21, 10, 21, 36
        )
        assert parse.parse("{:tg}", "21/11/2011 10:21:36 PM")[0] == datetime(
            2011, 11, 21, 22, 21, 36
        )
        assert parse.parse("{:tg}", "21/Nov/2011")[0] == datetime(2011, 11, 21)
        assert parse.parse("{:tg}", "1/November/2011")[0] == datetime(2011, 11, 1)

    def test_tg_tz(self):
        r = parse.parse("{:tg}", "21/11/2011 10:21:36 PM +1000")[0]
        assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 22, 21, 36)
        assert r.utcoffset() == timedelta(hours=10)
        r = parse.parse("{:tg}", "21/11/2011 10:21 -05:30")[0]
        assert r.utcoffset() == -timedelta(hours=5, minutes=30)

    def test_tg_then_field(self):
        r = parse.parse("{:tg} {:d}", "21/11/2011 10:21 3")
        assert r.fixed == (datetime(2011, 11, 21, 10, 21), 3)

    def test_ta(self):
        assert parse.parse("{:ta}", "11/21/2011")[0] == datetime(2011, 11, 21)
        assert parse.parse("{:ta}", "11-21-2011 10:21:36 PM")[0] == datetime(
            2011, 11, 21, 22, 21, 36
        )
        assert parse.parse("{:ta}", "Nov/21/2011")[0] == datetime(2011, 11, 21)
        assert parse.parse("{:ta}", "November-21-2011 10:21")[0] == datetime(
            2011, 11, 21, 10, 21
        )

    def test_ta_tz(self):
        r = parse.parse("{:ta}", "11/21/2011 10:21:36 AM +0100")[0]
        assert r.utcoffset() == timedelta(hours=1)
        assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 10, 21, 36)

    def test_ta_then_field(self):
        r = parse.parse("{:ta} {:d}", "11/21/2011 10:21 3")
        assert r.fixed == (datetime(2011, 11, 21, 10, 21), 3)

    def test_am_pm_12(self):
        assert parse.parse("{:tg}", "1/1/2000 12:15 AM")[0] == datetime(
            2000, 1, 1, 0, 15
        )
        assert parse.parse("{:tg}", "1/1/2000 12:15 PM")[0] == datetime(
            2000, 1, 1, 12, 15
        )
        assert parse.parse("{:tg}", "1/1/2000 1:15 PM")[0] == datetime(
            2000, 1, 1, 13, 15
        )
        assert parse.parse("{:tg}", "1/1/2000 1:15 AM")[0] == datetime(
            2000, 1, 1, 1, 15
        )

    def test_tt(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
        assert parse.parse("{:tt}", "10:21:36.25")[0] == time(10, 21, 36, 250000)
        assert parse.parse("{:tt}", "10:21:36 PM")[0] == time(22, 21, 36)
        assert parse.parse("{:tt}", "12:00 AM")[0] == time(0, 0)
        assert parse.parse("{:tt}", "12:00 PM")[0] == time(12, 0)

    def test_tt_tz(self):
        r = parse.parse("{:tt}", "10:21:36 PM -5:30")[0]
        assert r == time(22, 21, 36, tzinfo=tz(-330, "-5:30"))
        assert r.utcoffset() == -timedelta(hours=5, minutes=30)
        r = parse.parse("{:tt}", "10:21 +0130")[0]
        assert r.utcoffset() == timedelta(hours=1, minutes=30)
        r = parse.parse("{:tt}", "10:21 -130")[0]
        assert r.utcoffset() == -timedelta(hours=1, minutes=30)

    def test_tt_then_field(self):
        r = parse.parse("{:tt} {:d}", "10:21 PM 4")
        assert r.fixed == (time(22, 21), 4)

    def test_ts(self):
        year = datetime.today().year
        r = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
        assert r == datetime(year, 11, 9, 3, 37, 44)
        r = parse.parse("{:ts}", "Feb 27 15:00:00")[0]
        assert r == datetime(year, 2, 27, 15, 0, 0)

    def test_ts_then_field(self):
        year = datetime.today().year
        r = parse.parse("{:ts} {:d}", "Nov 9 03:37:44 5")
        assert r.fixed == (datetime(year, 11, 9, 3, 37, 44), 5)

    def test_datetime_search_in_text(self):
        r = search("on {:ti} we", "Meeting on 2020-02-29 we met")
        assert r[0] == datetime(2020, 2, 29)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            parse.parse("{:ti}", "2011-13-45")


class TestDateConvertDirect:
    def _match(self, pattern, s):
        return re.match(pattern, s)

    def test_ymd(self):
        m = self._match(r"(\d+-\d+-\d+)", "2020-01-02")
        assert date_convert("x", m, ymd=0) == datetime(2020, 1, 2)

    def test_mdy(self):
        m = self._match(r"(\S+)", "01/02/2020")
        assert date_convert("x", m, mdy=0) == datetime(2020, 1, 2)

    def test_dmy(self):
        m = self._match(r"(\S+)", "01/02/2020")
        assert date_convert("x", m, dmy=0) == datetime(2020, 2, 1)

    def test_dmy_space(self):
        m = self._match(r"(.+)", "01 Feb 2020")
        assert date_convert("x", m, dmy=0) == datetime(2020, 2, 1)

    def test_d_m_y(self):
        m = self._match(r"(\d+) (\w+) (\d+)", "3 Mar 2021")
        assert date_convert("x", m, d_m_y=(0, 1, 2)) == datetime(2021, 3, 3)

    def test_mm_dd(self):
        m = self._match(r"(x)(\w+) (\d+)", "xDec 25")
        r = date_convert("x", m, mm=1, dd=2)
        assert r == datetime(datetime.today().year, 12, 25)

    def test_time_only(self):
        m = self._match(r"(.+)", "01:02:03")
        assert date_convert("x", m, hms=0) == time(1, 2, 3)

    def test_hms_none_group(self):
        m = self._match(r"(\d+-\d+-\d+)( x)?", "2020-01-02")
        assert date_convert("x", m, ymd=0, hms=1) == datetime(2020, 1, 2)

    def test_tz_z(self):
        m = self._match(r"(.+)\s(Z)", "01:02 Z")
        r = date_convert("x", m, hms=0, tz=1)
        assert r.tzinfo == tz(0, "UTC")

    def test_tz_formats(self):
        for s, minutes in [
            ("+05:30", 330),
            ("-05:30", -330),
            ("+0530", 330),
            ("-530", -330),
            ("+5:30", 330),
        ]:
            m = self._match(r"(\S+) (\S+)", "01:02 " + s)
            r = date_convert("x", m, hms=0, tz=1)
            assert r.utcoffset() == timedelta(minutes=minutes), s

    def test_am(self):
        m = self._match(r"(\S+)( [AP]M)", "12:30 AM")
        assert date_convert("x", m, hms=0, am=1) == time(0, 30)
        m = self._match(r"(\S+)( [AP]M)", "11:30 PM")
        assert date_convert("x", m, hms=0, am=1) == time(23, 30)
        m = self._match(r"(\S+)( [AP]M)?", "11:30")
        assert date_convert("x", m, hms=0, am=1) == time(11, 30)

    def test_full_month_name(self):
        m = self._match(r"(\S+)", "01/January/2020")
        assert date_convert("x", m, dmy=0) == datetime(2020, 1, 1)


# --------------------------------------------------------------------------
# strftime-style formats
# --------------------------------------------------------------------------


class TestStrftimeFormats:
    def test_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-11-21")[0]
        assert r == date(2023, 11, 21)
        assert type(r) is date

    def test_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-11-21 10:21:36")[0]
        assert r == datetime(2023, 11, 21, 10, 21, 36)
        assert type(r) is datetime

    def test_time(self):
        r = parse.parse("{:%H:%M}", "10:21")[0]
        assert r == time(10, 21)
        assert type(r) is time

    def test_time_with_microseconds(self):
        r = parse.parse("{:%H:%M:%S.%f}", "10:21:36.5")[0]
        assert r == time(10, 21, 36, 500000)

    def test_time_with_tz(self):
        r = parse.parse("{:%H:%M %z}", "10:21 +0100")[0]
        assert r.utcoffset() == timedelta(hours=1)

    def test_12h(self):
        r = parse.parse("{:%I:%M %p}", "10:21 PM")[0]
        assert r == time(22, 21)

    def test_names(self):
        r = parse.parse("{:%a %d %b %Y}", "Tue 21 Nov 2023")[0]
        assert r == date(2023, 11, 21)
        r = parse.parse("{:%A %d %B %Y}", "Tuesday 21 November 2023")[0]
        assert r == date(2023, 11, 21)

    def test_two_digit_year(self):
        assert parse.parse("{:%d/%m/%y}", "21/11/23")[0] == date(2023, 11, 21)

    def test_no_year_uses_current(self):
        r = parse.parse("{:%m-%d}", "11-21")[0]
        assert r == date(datetime.today().year, 11, 21)

    def test_no_year_datetime_uses_current(self):
        r = parse.parse("{:%m-%d %H:%M}", "02-03 04:05")[0]
        assert r == datetime(datetime.today().year, 2, 3, 4, 5)

    def test_julian(self):
        assert parse.parse("{:%Y %j}", "2023 032")[0] == date(2023, 2, 1)

    def test_weekday(self):
        r = parse.parse("{:%Y %U %w}", "2023 01 1")[0]
        assert isinstance(r, date)

    def test_week_W(self):
        r = parse.parse("{:%Y %W %w}", "2023 01 1")[0]
        assert isinstance(r, date)

    def test_rejects(self):
        assert parse.parse("{:%Y-%m-%d}", "23-11-21") is None
        assert parse.parse("{:%H:%M}", "10:2") is None

    def test_named_and_followed(self):
        r = parse.parse("{d:%Y-%m-%d} {n:d}", "2023-11-21 5")
        assert r.named == {"d": date(2023, 11, 21), "n": 5}

    def test_strf_date_convert_direct(self):
        assert strf_date_convert("2023-01-02", None, "%Y-%m-%d") == date(2023, 1, 2)
        assert strf_date_convert("10:21", None, "%H:%M") == time(10, 21)
        assert strf_date_convert("2023-01-02 03", None, "%Y-%m-%d %H") == datetime(
            2023, 1, 2, 3
        )

    def test_strf_date_convert_neither(self):
        with pytest.raises(ValueError):
            strf_date_convert("x", None, "x")

    def test_strf_each_date_directive(self):
        # each of these alone must be classified as a date (returns date)
        today_year = datetime.today().year
        assert strf_date_convert("21", None, "%d") == date(today_year, 1, 21)
        assert strf_date_convert("11", None, "%m") == date(today_year, 11, 1)
        assert strf_date_convert("Nov", None, "%b") == date(today_year, 11, 1)
        assert strf_date_convert("November", None, "%B") == date(today_year, 11, 1)
        assert strf_date_convert("2020", None, "%Y") == date(2020, 1, 1)
        assert strf_date_convert("20", None, "%y") == date(2020, 1, 1)
        assert strf_date_convert("032", None, "%j") == date(today_year, 2, 1)
        assert type(strf_date_convert("Mon", None, "%a")) is date
        assert type(strf_date_convert("Monday", None, "%A")) is date
        assert type(strf_date_convert("1", None, "%w")) is date
        assert type(strf_date_convert("01", None, "%U")) is date
        assert type(strf_date_convert("01", None, "%W")) is date

    def test_strf_each_time_directive(self):
        assert strf_date_convert("10", None, "%H") == time(10)
        assert strf_date_convert("10", None, "%I") == time(10)
        assert strf_date_convert("05", None, "%M") == time(0, 5)
        assert strf_date_convert("07", None, "%S") == time(0, 0, 7)
        assert strf_date_convert("5", None, "%f") == time(0, 0, 0, 500000)
        assert type(strf_date_convert("PM", None, "%p")) is time
        r = strf_date_convert("+0200", None, "%z")
        assert type(r) is time
        assert r.utcoffset() == timedelta(hours=2)

    def test_get_regex_for_datetime_format(self):
        rx = parse.get_regex_for_datetime_format("%Y-%m-%d")
        assert re.fullmatch(rx, "2023-1-02")
        assert not re.fullmatch(rx, "23-1-02")

    @pytest.mark.parametrize(
        "directive,good,bad",
        [
            ("%a", "Sun", "Sunday"),
            ("%A", "Saturday", "Sat"),
            ("%w", "6", "7"),
            ("%d", "31", "123"),
            ("%b", "Dec", "December"),
            ("%B", "December", "Dec"),
            ("%m", "1", "123"),
            ("%y", "99", "1999"),
            ("%Y", "1999", "99"),
            ("%H", "23", "123"),
            ("%I", "1", "123"),
            ("%p", "PM", "XM"),
            ("%M", "05", "5"),
            ("%S", "05", "5"),
            ("%f", "123456", "1234567"),
            ("%z", "+0100", "0100"),
            ("%j", "366", "3666"),
            ("%U", "52", "521"),
            ("%W", "52", "521"),
        ],
    )
    def test_directive_regexes(self, directive, good, bad):
        rx = parse.dt_format_to_regex[directive]
        assert re.fullmatch(rx, good)
        assert not re.fullmatch(rx, bad)

    def test_z_variants(self):
        rx = parse.dt_format_to_regex["%z"]
        for s in ["+01", "-0100", "+01:00", "+01:00:30", "-010030"]:
            assert re.fullmatch(rx, s), s


# --------------------------------------------------------------------------
# Custom types
# --------------------------------------------------------------------------


class TestCustomTypes:
    def test_simple_converter(self):
        def yesno(s):
            return s == "yes"

        r = parse.parse("{:YN}", "yes", extra_types={"YN": yesno})
        assert r[0] is True

    def test_converter_default_pattern_non_greedy(self):
        r = parse.parse("{:U} {}", "a b c", extra_types={"U": str.upper})
        assert r.fixed == ("A", "b c")

    def test_with_pattern(self):
        @with_pattern(r"\d+")
        def number(s):
            return int(s)

        assert number.pattern == r"\d+"
        assert number.regex_group_count is None
        r = parse.parse("{:Number} {}", "12 ab", extra_types={"Number": number})
        assert r.fixed == (12, "ab")
        assert parse.parse("{:Number}", "ab", extra_types={"Number": number}) is None

    def test_with_pattern_returns_same_function(self):
        def f(s):
            return s

        assert with_pattern("x")(f) is f

    def test_with_pattern_group_count(self):
        @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
        def rng(s):
            a, b = s.split("-")
            return (int(a), int(b))

        assert rng.regex_group_count == 3
        r = parse.parse(
            "{:Range} {:d} {name:Range}", "1-2 3 4-5", extra_types={"Range": rng}
        )
        assert r.fixed == ((1, 2), 3)
        assert r["name"] == (4, 5)

    def test_with_pattern_group_count_needed(self):
        @with_pattern(r"(a|b)+", regex_group_count=1)
        def ab(s):
            return s

        r = parse.parse("{:AB} {:d}", "abab 7", extra_types={"AB": ab})
        assert r.fixed == ("abab", 7)

    def test_with_pattern_group_count_zero(self):
        @with_pattern(r"[ab]+", regex_group_count=0)
        def ab(s):
            return s

        r = parse.parse("{:AB} {:d}", "abab 7", extra_types={"AB": ab})
        assert r.fixed == ("abab", 7)

    def test_custom_type_overrides_builtin(self):
        @with_pattern(r"\w+")
        def d(s):
            return "custom:" + s

        r = parse.parse("{:d}", "abc", extra_types={"d": d})
        assert r[0] == "custom:abc"

    def test_custom_type_with_alignment(self):
        @with_pattern(r"\d+")
        def num(s):
            return int(s)

        r = parse.parse("[{:>N}]", "[  12]", extra_types={"N": num})
        assert r[0] == 12

    def test_custom_type_with_width(self):
        @with_pattern(r"\d+")
        def num(s):
            return int(s)

        r = parse.parse("[{:4N}]", "[  12]", extra_types={"N": num})
        assert r[0] == 12

    def test_compile_with_extra_types(self):
        @with_pattern(r"[A-Z]")
        def letter(s):
            return ord(s)

        p = compile("{:L}{:L}", extra_types={"L": letter}, case_sensitive=True)
        assert p.parse("AB").fixed == (65, 66)
        assert p.parse("ab") is None

    def test_convert_first(self):
        cf = convert_first(int)
        assert cf("12", object()) == 12
        assert pickle.loads(pickle.dumps(cf))("3", None) == 3


# --------------------------------------------------------------------------
# extract_format
# --------------------------------------------------------------------------


class TestExtractFormat:
    def test_empty_type(self):
        f = extract_format(">10", {})
        assert f["align"] == ">"
        assert f["fill"] is None
        assert f["width"] == "10"
        assert f["type"] == ""
        assert f["zero"] is False

    def test_fill_align(self):
        f = extract_format("x<5d", {})
        assert f["fill"] == "x"
        assert f["align"] == "<"
        assert f["width"] == "5"
        assert f["type"] == "d"

    def test_align_only(self):
        for a in "<>=^":
            f = extract_format(a + "d", {})
            assert f["align"] == a
            assert f["fill"] is None
            assert f["type"] == "d"

    def test_single_char(self):
        f = extract_format("d", {})
        assert f["align"] is None
        assert f["fill"] is None
        assert f["type"] == "d"

    def test_sign(self):
        for s in "+- ":
            f = extract_format(s + "d", {})
            assert f["type"] == "d"

    def test_zero(self):
        f = extract_format("05d", {})
        assert f["zero"] is True
        assert f["width"] == "5"
        assert f["type"] == "d"

    def test_grouping(self):
        assert extract_format(",d", {})["grouping"] == ","
        assert extract_format("_d", {})["grouping"] == "_"
        assert "grouping" not in extract_format("d", {})

    def test_precision(self):
        f = extract_format("10.3f", {})
        assert f["width"] == "10"
        assert f["precision"] == "3"
        assert f["type"] == "f"
        f = extract_format(".12", {})
        assert f["precision"] == "12"
        assert f["type"] == ""

    def test_full(self):
        f = extract_format("*^+010,.2f", {})
        assert f["fill"] == "*"
        assert f["align"] == "^"
        assert f["zero"] is True
        assert f["width"] == "10"
        assert f["grouping"] == ","
        assert f["precision"] == "2"
        assert f["type"] == "f"

    def test_extra_type_allowed(self):
        assert extract_format("Foo", {"Foo": str})["type"] == "Foo"

    def test_datetime_type_allowed(self):
        assert extract_format("%Y-%m", {})["type"] == "%Y-%m"

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            extract_format("Foo", {})
        with pytest.raises(ValueError):
            extract_format("5q", {})

    @pytest.mark.parametrize("t", sorted(parse.ALLOWED_TYPES))
    def test_allowed_types(self, t):
        assert extract_format(t, {})["type"] == t

    def test_allowed_types_set(self):
        assert parse.ALLOWED_TYPES == set("nboxX%fFeEgGwWdDsSl") | {
            "ti",
            "te",
            "ta",
            "th",
            "tg",
            "tc",
            "tt",
            "ts",
        }


# --------------------------------------------------------------------------
# FixedTzOffset
# --------------------------------------------------------------------------


class TestFixedTzOffset:
    def test_basic(self):
        t = FixedTzOffset(90, "+01:30")
        assert t.utcoffset(None) == timedelta(minutes=90)
        assert t.tzname(None) == "+01:30"
        assert t.dst(None) == timedelta(0)

    def test_repr(self):
        t = FixedTzOffset(60, "X")
        assert repr(t) == "<FixedTzOffset X 1:00:00>"

    def test_eq(self):
        assert FixedTzOffset(60, "X") == FixedTzOffset(60, "X")
        assert FixedTzOffset(60, "X") != FixedTzOffset(60, "Y")
        assert FixedTzOffset(60, "X") != FixedTzOffset(30, "X")
        assert FixedTzOffset(60, "X") != "X"
        assert FixedTzOffset(60, "X").__eq__("X") is NotImplemented

    def test_negative(self):
        t = FixedTzOffset(-300, "-05:00")
        assert t.utcoffset(None) == timedelta(hours=-5)


# --------------------------------------------------------------------------
# Module surface
# --------------------------------------------------------------------------


def test_all():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}


def test_version():
    assert parse.__version__ == "1.22.3"


def test_months_map():
    assert parse.MONTHS_MAP["Jan"] == 1
    assert parse.MONTHS_MAP["December"] == 12
    for i, m in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1
    ):
        assert parse.MONTHS_MAP[m] == i
    for i, m in enumerate(
        (
            "January February March April May June July August September "
            "October November December"
        ).split(),
        start=1,
    ):
        assert parse.MONTHS_MAP[m] == i


@pytest.mark.parametrize(
    "month,num",
    list(
        zip(
            "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
            range(1, 13),
        )
    ),
)
def test_te_all_months(month, num):
    r = parse.parse("{:te}", "1 %s 2000 00:00:00 +0000" % month)[0]
    assert r.month == num
