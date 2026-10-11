import math
import pickle
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
    extract_format,
    findall,
    int_convert,
    search,
    with_pattern,
)


def tz(minutes, name):
    return FixedTzOffset(minutes, name)


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------


class TestBasic:
    def test_no_fields_match(self):
        r = parse.parse("hello", "hello")
        assert r is not None
        assert r.fixed == ()
        assert r.named == {}

    def test_no_fields_no_match(self):
        assert parse.parse("hello", "world") is None

    def test_parse_requires_full_match(self):
        assert parse.parse("hello", "hello world") is None
        assert parse.parse("hello", "say hello") is None

    def test_trailing_newline_not_matched(self):
        # \Z rather than $ - trailing newline must not be accepted
        assert parse.parse("hello", "hello\n") is None

    def test_single_fixed(self):
        r = parse.parse("hello {}", "hello world")
        assert r.fixed == ("world",)
        assert r[0] == "world"

    def test_multiple_fixed(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r.fixed == ("a", "b", "c")

    def test_fixed_slicing(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r[1:] == ("b", "c")
        assert r[-1] == "c"

    def test_numbered_fields_are_fixed(self):
        r = parse.parse("{0} {1}", "a b")
        assert r.fixed == ("a", "b")
        assert r.named == {}

    def test_named(self):
        r = parse.parse("hello {name}", "hello world")
        assert r.named == {"name": "world"}
        assert r["name"] == "world"
        assert r.fixed == ()

    def test_named_and_fixed(self):
        r = parse.parse("{} {name} {}", "a b c")
        assert r.fixed == ("a", "c")
        assert r.named == {"name": "b"}

    def test_contains(self):
        r = parse.parse("{a} {b}", "1 2")
        assert "a" in r
        assert "b" in r
        assert "c" not in r

    def test_missing_named_key_raises(self):
        r = parse.parse("{a}", "1")
        with pytest.raises(KeyError):
            r["b"]

    def test_missing_index_raises(self):
        r = parse.parse("{}", "1")
        with pytest.raises(IndexError):
            r[1]

    def test_lazy_matching(self):
        r = parse.parse("{}-{}", "a-b-c")
        assert r.fixed == ("a", "b-c")

    def test_empty_field_does_not_match_empty(self):
        assert parse.parse("x{}y", "xy") is None

    def test_regex_special_chars_in_literal(self):
        fmt = "a.b*c+d?e(f)g[h]i^j$k|l\\m!"
        assert parse.parse(fmt, fmt) is not None
        # "." must be literal
        assert parse.parse("a.b", "axb") is None
        assert parse.parse("a*", "aaa") is None
        assert parse.parse("(x)", "x") is None

    def test_regex_special_chars_around_field(self):
        r = parse.parse("({}) [{}] $ {}^", "(1) [2] $ 3^")
        assert r.fixed == ("1", "2", "3")

    def test_braces_escaped(self):
        r = parse.parse("{{hello}} {}", "{hello} world")
        assert r.fixed == ("world",)

    def test_braces_escaped_only(self):
        assert parse.parse("{{}}", "{}") is not None
        assert parse.parse("{{}}", "{{}}") is None

    def test_brace_around_field(self):
        r = parse.parse("{{{}}}", "{x}")
        assert r.fixed == ("x",)

    def test_case_insensitive_default(self):
        assert parse.parse("Hello {}", "HELLO world").fixed == ("world",)

    def test_case_sensitive(self):
        assert parse.parse("Hello {}", "HELLO world", case_sensitive=True) is None
        r = parse.parse("Hello {}", "Hello world", case_sensitive=True)
        assert r.fixed == ("world",)

    def test_dotall_newline_in_field(self):
        r = parse.parse("{}:{}", "a\nb:c\nd")
        assert r.fixed == ("a\nb", "c\nd")

    def test_unicode(self):
        r = parse.parse("héllo {}", "héllo wörld")
        assert r.fixed == ("wörld",)

    def test_result_repr(self):
        r = parse.parse("{} {a}", "x y")
        assert repr(r) == "<Result ('x',) {'a': 'y'}>"

    def test_result_constructed_directly(self):
        r = Result((1, 2), {"a": 3}, {})
        assert r[0] == 1
        assert r["a"] == 3
        assert "a" in r
        assert r[0:2] == (1, 2)


# ---------------------------------------------------------------------------
# Named field features
# ---------------------------------------------------------------------------


class TestNamed:
    def test_dotted_name(self):
        r = parse.parse("{hello.world}", "x")
        assert r.named == {"hello.world": "x"}

    def test_dotted_name_and_underscore_collision(self):
        r = parse.parse("{a.b} {a_b}", "1 2")
        assert r.named == {"a.b": "1", "a_b": "2"}

    def test_underscore_then_dot_collision(self):
        r = parse.parse("{a_b} {a.b}", "1 2")
        assert r.named == {"a_b": "1", "a.b": "2"}

    def test_hyphen_name(self):
        r = parse.parse("{hello-world}", "x")
        assert r.named == {"hello-world": "x"}

    def test_hyphen_and_underscore_collision(self):
        r = parse.parse("{a_b} {a-b}", "1 2")
        assert r.named == {"a_b": "1", "a-b": "2"}

    def test_hyphen_collision_other_order(self):
        r = parse.parse("{a-b} {a_b}", "1 2")
        assert r.named == {"a-b": "1", "a_b": "2"}

    def test_triple_collision(self):
        r = parse.parse("{a.b} {a_b} {a-b}", "1 2 3")
        assert r.named == {"a.b": "1", "a_b": "2", "a-b": "3"}

    def test_dict_style(self):
        r = parse.parse("{a[b]}", "x")
        assert r.named == {"a": {"b": "x"}}

    def test_nested_dict_style(self):
        r = parse.parse("{a[b][c]} {a[b][d]} {a[e]}", "1 2 3")
        assert r.named == {"a": {"b": {"c": "1", "d": "2"}, "e": "3"}}
        assert r["a"]["b"]["c"] == "1"

    def test_dict_style_with_types(self):
        r = parse.parse("{p[x]:d},{p[y]:d}", "3,4")
        assert r.named == {"p": {"x": 3, "y": 4}}

    def test_dict_style_spans_keyed_by_full_name(self):
        r = parse.parse("{a[b]}-{c}", "xy-z")
        assert r.spans["a[b]"] == (0, 2)
        assert r.spans["c"] == (3, 4)

    def test_dict_key_with_digits(self):
        r = parse.parse("{a[0]} {a[1]}", "x y")
        assert r.named == {"a": {"0": "x", "1": "y"}}

    def test_repeated_name_must_match(self):
        r = parse.parse("{a} {a}", "x x")
        assert r.named == {"a": "x"}
        assert parse.parse("{a} {a}", "x y") is None

    def test_repeated_name_typed(self):
        r = parse.parse("{n:d} {n:d}", "3 3")
        assert r.named == {"n": 3}
        assert parse.parse("{n:d} {n:d}", "3 4") is None

    def test_repeated_name_type_mismatch(self):
        with pytest.raises(RepeatedNameError):
            parse.compile("{a:d} {a:w}")
        with pytest.raises(RepeatedNameError):
            parse.compile("{a} {a:d}")

    def test_repeated_name_error_is_value_error(self):
        assert issubclass(RepeatedNameError, ValueError)

    def test_repeated_name_only_listed_once(self):
        p = parse.compile("{a} {b} {a}")
        assert p.named_fields == ["a", "b"]

    def test_name_starting_with_digit_is_fixed(self):
        r = parse.parse("{0} {1}", "x y")
        assert r.fixed == ("x", "y")


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class TestSpans:
    def test_spans_fixed_and_named(self):
        r = parse.parse("{} {name} {}", "aa bbb c")
        assert r.spans == {0: (0, 2), "name": (3, 6), 1: (7, 8)}

    def test_spans_with_type_groups(self):
        # types adding inner groups must not shift spans of later fields
        r = parse.parse("{:g} {:ti} {}", "1.5e3 2020-01-02 xyz")
        assert r.spans[0] == (0, 5)
        assert r.spans[1] == (6, 16)
        assert r.spans[2] == (17, 20)

    def test_spans_search(self):
        # a leading space is accepted as a sign for {:d}
        r = parse.search("{:d}", "abc 123 def")
        assert r.spans == {0: (3, 7)}
        r = parse.search("{:d}", "abc-123 def")
        assert r.spans == {0: (3, 7)}
        assert r[0] == -123

    def test_spans_padded(self):
        r = parse.parse("[{:>}]", "[   ab]")
        assert r.fixed == ("ab",)
        assert r.spans[0] == (4, 6)


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "text,value",
        [
            ("0", 0),
            ("123", 123),
            ("-123", -123),
            ("+123", 123),
            (" 123", 123),
            ("0x1f", 31),
            ("0X1F", 31),
            ("-0x1f", -31),
            ("0b101", 5),
            ("0B101", 5),
            ("0o17", 15),
            ("0O17", 15),
            ("-0o17", -15),
            ("007", 7),
            ("00", 0),
        ],
    )
    def test_d(self, text, value):
        r = parse.parse("{:d}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is int

    def test_d_shortest_prefixed_numbers(self):
        # a prefix followed by a single digit must still switch base
        assert parse.parse("{:d}", "0xa")[0] == 10
        assert parse.parse("{:d}", "0xf")[0] == 15
        assert parse.parse("{:d}", "-0xa")[0] == -10
        assert int_convert()("0xa", None) == 10

    def test_d_rejects_non_number(self):
        assert parse.parse("{:d}", "abc") is None
        assert parse.parse("{:d}", "1.5") is None
        assert parse.parse("{:d}", "") is None

    def test_d_rejects_grouping_without_flag(self):
        assert parse.parse("{:d}", "1,000") is None

    def test_d_grouping_comma(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567

    def test_d_grouping_underscore(self):
        assert parse.parse("{:_d}", "1_234_567")[0] == 1234567

    def test_d_grouping_hex_underscore(self):
        assert parse.parse("{:_d}", "0xdead_beef")[0] == 0xDEADBEEF

    def test_d_width_limits_digits(self):
        r = parse.parse("{:2d}{:2d}", "1234")
        assert r.fixed == (12, 34)

    def test_d_width_with_padding(self):
        r = parse.parse("{:4d}", "  12")
        assert r[0] == 12

    def test_d_zero_padded_width(self):
        r = parse.parse("{:02d}{:02d}", "0507")
        assert r.fixed == (5, 7)

    def test_d_too_long_for_width(self):
        assert parse.parse("{:2d}", "123") is None

    def test_d_sign_specs(self):
        assert parse.parse("{:+d}", "+5")[0] == 5
        assert parse.parse("{:-d}", "-5")[0] == -5
        assert parse.parse("{: d}", " 5")[0] == 5

    def test_d_in_context(self):
        r = parse.parse("x={:d}, y={:d}", "x=10, y=-20")
        assert r.fixed == (10, -20)

    @pytest.mark.parametrize(
        "text,value",
        [("101", 5), ("0b101", 5), ("0B11", 3), ("-101", -5), ("+1", 1)],
    )
    def test_b(self, text, value):
        assert parse.parse("{:b}", text)[0] == value

    def test_b_rejects_other_digits(self):
        assert parse.parse("{:b}", "102") is None

    @pytest.mark.parametrize(
        "text,value", [("17", 15), ("0o17", 15), ("0O7", 7), ("-10", -8)]
    )
    def test_o(self, text, value):
        assert parse.parse("{:o}", text)[0] == value

    def test_o_rejects_8(self):
        assert parse.parse("{:o}", "8") is None

    @pytest.mark.parametrize(
        "fmt,text,value",
        [
            ("{:x}", "ff", 255),
            ("{:x}", "FF", 255),
            ("{:x}", "0xff", 255),
            ("{:X}", "0XFF", 255),
            ("{:x}", "-1a", -26),
            ("{:x}", "10", 16),
        ],
    )
    def test_x(self, fmt, text, value):
        assert parse.parse(fmt, text)[0] == value

    def test_x_rejects_g(self):
        assert parse.parse("{:x}", "fg") is None

    def test_b_is_base2_even_for_ambiguous(self):
        # "0b1" in binary would be invalid as "0b" prefix if not handled
        assert parse.parse("{:b}", "0b1")[0] == 1
        # "10" interpreted as base 2
        assert parse.parse("{:b}", "10")[0] == 2

    def test_x_leading_zero_b(self):
        # with an explicit base 16, "0b1" is hex 0xb1, not binary
        assert parse.parse("{:x}", "0b1")[0] == 0xB1

    @pytest.mark.parametrize(
        "text,value",
        [
            ("1", 1),
            ("123", 123),
            ("1,234", 1234),
            ("1.234", 1234),
            ("1,234,567", 1234567),
            ("-1,000", -1000),
        ],
    )
    def test_n(self, text, value):
        assert parse.parse("{:n}", text)[0] == value

    def test_n_bad_grouping(self):
        assert parse.parse("{:n}", "12,34") is None
        assert parse.parse("{:n}", "1234") is None

    def test_n_followed_by_field(self):
        r = parse.parse("{:n} {:d}", "1,000 5")
        assert r.fixed == (1000, 5)

    def test_multiple_numbers_group_counting(self):
        # each of these types adds inner regex groups; field indexing must survive
        r = parse.parse("{:n} {:b} {:o} {:x} {:%} {}", "1,000 0b1 0o7 0xa 50% end")
        assert r.fixed == (1000, 1, 7, 10, 0.5, "end")

    def test_named_after_numeric_groups(self):
        r = parse.parse("{a:x} {b:n} {c}", "0xff 1,000 hi")
        assert r.named == {"a": 255, "b": 1000, "c": "hi"}


class TestIntConvert:
    def test_auto_base(self):
        c = int_convert()
        assert c("0x10", None) == 16
        assert c("0o10", None) == 8
        assert c("0b10", None) == 2
        assert c("10", None) == 10
        assert c("-0x10", None) == -16
        assert c("+0b11", None) == 3

    def test_short_strings_decimal(self):
        c = int_convert()
        assert c("0", None) == 0
        assert c("01", None) == 1
        assert c("-0", None) == 0

    def test_0x_alone_too_short_for_prefix(self):
        # "0b" with nothing after: length not > 2, stays decimal; non-digits stripped
        c = int_convert()
        assert c("0b", None) == 0

    def test_explicit_base(self):
        assert int_convert(16)("ff", None) == 255
        assert int_convert(2)("0b11", None) == 3
        assert int_convert(8)("0o777", None) == 511
        assert int_convert(10)("1,234", None) == 1234

    def test_strips_ignored_chars(self):
        assert int_convert(10)("1_000", None) == 1000
        assert int_convert(10)(" 42", None) == 42

    def test_uppercase(self):
        assert int_convert(16)("ABCDEF", None) == 0xABCDEF

    def test_large_base(self):
        assert int_convert(36)("zz", None) == 36 * 36 - 1


# ---------------------------------------------------------------------------
# Float types
# ---------------------------------------------------------------------------


class TestFloats:
    @pytest.mark.parametrize(
        "text,value",
        [
            ("1.5", 1.5),
            (".5", 0.5),
            ("-1.5", -1.5),
            ("+1.5", 1.5),
            (" 1.5", 1.5),
            ("123.456", 123.456),
            ("-.25", -0.25),
        ],
    )
    def test_f(self, text, value):
        r = parse.parse("{:f}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is float

    def test_f_requires_decimal_point(self):
        assert parse.parse("{:f}", "15") is None
        assert parse.parse("{:f}", "1.") is None

    def test_f_nan_inf(self):
        assert math.isnan(parse.parse("{:f}", "nan")[0])
        assert math.isnan(parse.parse("{:f}", "NAN")[0])
        assert parse.parse("{:f}", "inf")[0] == math.inf
        assert parse.parse("{:f}", "-inf")[0] == -math.inf
        assert parse.parse("{:f}", "+INF")[0] == math.inf

    def test_f_precision_zero(self):
        r = parse.parse("{:.0f}", "20")
        assert r[0] == 20.0
        assert type(r[0]) is float
        assert parse.parse("{:.0f}", "-20")[0] == -20.0

    def test_f_precision_zero_rejects_decimal(self):
        assert parse.parse("{:.0f}", "20.5") is None

    def test_f_precision_nonzero(self):
        assert parse.parse("{:.2f}", "3.14")[0] == 3.14

    def test_f_width_precision(self):
        assert parse.parse("{:8.2f}", "    3.14")[0] == 3.14

    def test_F_is_decimal(self):
        r = parse.parse("{:F}", "1.10")
        assert r[0] == Decimal("1.10")
        assert isinstance(r[0], Decimal)
        assert parse.parse("{:F}", "-0.5")[0] == Decimal("-0.5")

    def test_F_precision_zero(self):
        r = parse.parse("{:.0F}", "7")
        assert r[0] == Decimal("7")

    @pytest.mark.parametrize(
        "text,value",
        [
            ("1.0e10", 1.0e10),
            ("1.5E-3", 1.5e-3),
            ("-1.5e+3", -1500.0),
            (".5e2", 50.0),
        ],
    )
    def test_e(self, text, value):
        assert parse.parse("{:e}", text)[0] == value
        assert parse.parse("{:E}", text)[0] == value

    def test_e_requires_exponent(self):
        assert parse.parse("{:e}", "1.5") is None
        assert parse.parse("{:e}", "15e3") is None

    def test_e_nan_inf(self):
        assert math.isnan(parse.parse("{:e}", "nan")[0])
        assert parse.parse("{:e}", "inf")[0] == math.inf
        assert parse.parse("{:e}", "-inf")[0] == -math.inf

    @pytest.mark.parametrize(
        "text,value",
        [
            ("1", 1.0),
            ("1.5", 1.5),
            ("-1.5", -1.5),
            ("1e10", 1e10),
            ("1.5e-3", 1.5e-3),
            ("15E3", 15000.0),
        ],
    )
    def test_g(self, text, value):
        r = parse.parse("{:g}", text)
        assert r[0] == value
        assert type(r[0]) is float
        assert parse.parse("{:G}", text)[0] == value

    def test_g_nan_inf(self):
        assert math.isnan(parse.parse("{:g}", "nan")[0])
        assert parse.parse("{:g}", "INF")[0] == math.inf
        assert parse.parse("{:g}", "-inf")[0] == -math.inf

    def test_g_rejects_text(self):
        assert parse.parse("{:g}", "abc") is None

    def test_g_followed_by_fields(self):
        r = parse.parse("{:g} {} {name}", "1.5e3 x y")
        assert r.fixed == (1500.0, "x")
        assert r.named == {"name": "y"}

    def test_e_followed_by_field(self):
        r = parse.parse("{:e} {}", "1.0e1 x")
        assert r.fixed == (10.0, "x")

    def test_f_followed_by_field(self):
        r = parse.parse("{:f} {}", "1.5 x")
        assert r.fixed == (1.5, "x")

    @pytest.mark.parametrize(
        "text,value",
        [("50%", 0.5), ("100%", 1.0), ("12.5%", 0.125), ("0%", 0.0), ("-10%", -0.1)],
    )
    def test_percentage(self, text, value):
        assert parse.parse("{:%}", text)[0] == pytest.approx(value)

    def test_percentage_requires_sign(self):
        assert parse.parse("{:%}", "50") is None

    def test_percentage_function(self):
        assert parse.percentage("25%", None) == 0.25


# ---------------------------------------------------------------------------
# String types and alignment
# ---------------------------------------------------------------------------


class TestStringTypes:
    def test_w(self):
        assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
        assert parse.parse("{:w}", "ab c") is None
        assert parse.parse("{:w}", "a-b") is None

    def test_W(self):
        assert parse.parse("{:W}", "!@ #")[0] == "!@ #"
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
        assert parse.parse("{:l}", "hello1") is None
        assert parse.parse("{:l}", "héllo") is None

    def test_w_greedy_in_context(self):
        r = parse.parse("{:w} {:w}", "ab cd")
        assert r.fixed == ("ab", "cd")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            parse.compile("{:z}")
        with pytest.raises(ValueError):
            parse.compile("{:q}")
        with pytest.raises(ValueError):
            parse.compile("{:dd}")

    def test_width_string(self):
        r = parse.parse("{:3}{}", "abcdef")
        assert r.fixed == ("abc", "def")

    def test_width_requires_minimum(self):
        assert parse.parse("{:5}", "abc") is None

    def test_precision_string(self):
        r = parse.parse("{:.2}{}", "abcdef")
        assert r.fixed == ("a", "bcdef")
        assert parse.parse("{:.2}", "abc") is None

    def test_width_and_precision(self):
        assert parse.parse("{:2.4}", "abcd")[0] == "abcd"
        assert parse.parse("{:2.4}", "abcde") is None
        assert parse.parse("{:2.4}", "a") is None

    def test_left_align(self):
        r = parse.parse("[{:<}]", "[abc   ]")
        assert r[0] == "abc"

    def test_right_align(self):
        r = parse.parse("[{:>}]", "[   abc]")
        assert r[0] == "abc"

    def test_center_align(self):
        r = parse.parse("[{:^}]", "[  abc  ]")
        assert r[0] == "abc"

    def test_string_width_is_minimum_value_length(self):
        # for untyped fields the width is a minimum length of the captured value,
        # so padding that is needed to reach the width stays in the value
        assert parse.parse("[{:10}]", "[       abc]")[0] == "       abc"
        assert parse.parse("[{:<10}]", "[abc       ]")[0] == "abc       "
        assert parse.parse("[{:<3}]", "[abc   ]")[0] == "abc"
        assert parse.parse("[{:>3}]", "[   abc]")[0] == "abc"
        assert parse.parse("[{:^3}]", "[  abc  ]")[0] == "abc"

    def test_custom_fill(self):
        assert parse.parse("[{:x<}]", "[abcxxx]")[0] == "abc"
        assert parse.parse("[{:x>}]", "[xxxabc]")[0] == "abc"
        assert parse.parse("[{:x^}]", "[xxabcxx]")[0] == "abc"

    @pytest.mark.parametrize("fill", list(".\\+?*[]()$"))
    def test_regex_special_fill(self, fill):
        fmt = "{:%s<}|" % fill
        text = "abc" + fill * 3 + "|"
        r = parse.parse(fmt, text)
        assert r is not None, (fmt, text)
        assert r[0] == "abc"

    def test_dot_fill_is_literal(self):
        # if "." fill weren't escaped it would match anything and eat into value
        r = parse.parse("{:.>}{:d}", "...ab5")
        assert r is not None
        assert r.fixed == ("ab", 5)

    def test_numeric_with_align(self):
        assert parse.parse("[{:>5d}]", "[   42]")[0] == 42
        assert parse.parse("[{:<5d}]", "[42   ]")[0] == 42
        assert parse.parse("[{:^6d}]", "[  42  ]")[0] == 42

    def test_numeric_equals_align_default_zero_fill(self):
        assert parse.parse("{:=5d}", "00042")[0] == 42
        assert parse.parse("{:=5d}", "-0042")[0] == -42

    def test_numeric_equals_align_default_fill_is_not_space(self):
        assert parse.parse("{:=6d}", "-  42") is None
        assert parse.parse("{: =6d}", "-  42")[0] == -42

    def test_numeric_equals_align_custom_fill(self):
        assert parse.parse("{:x=5d}", "xxx42")[0] == 42
        assert parse.parse("{:=8.2f}", "-0003.14")[0] == -3.14

    def test_zero_flag_width(self):
        assert parse.parse("{:05d}", "00042")[0] == 42

    def test_float_with_fill(self):
        assert parse.parse("{:*>8.2f}", "****3.14")[0] == 3.14


# ---------------------------------------------------------------------------
# extract_format
# ---------------------------------------------------------------------------


class TestExtractFormat:
    def test_empty_type(self):
        f = extract_format("5", {})
        assert f["width"] == "5"
        assert f["type"] == ""
        assert f["align"] is None
        assert f["fill"] is None
        assert f["zero"] is False

    def test_align_only(self):
        f = extract_format("<", {})
        assert f["align"] == "<"
        assert f["fill"] is None

    def test_fill_and_align(self):
        f = extract_format("*^10d", {})
        assert f["fill"] == "*"
        assert f["align"] == "^"
        assert f["width"] == "10"
        assert f["type"] == "d"

    def test_zero(self):
        f = extract_format("010d", {})
        assert f["zero"] is True
        assert f["width"] == "10"
        assert f["type"] == "d"

    def test_sign_dropped(self):
        f = extract_format("+5d", {})
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

    def test_no_precision_key_when_absent(self):
        assert "precision" not in extract_format("10f", {})

    def test_extra_type(self):
        f = extract_format("foo", {"foo": None})
        assert f["type"] == "foo"

    def test_datetime_type(self):
        f = extract_format("%Y-%m-%d", {})
        assert f["type"] == "%Y-%m-%d"

    def test_time_types_allowed(self):
        for t in ["ti", "te", "ta", "tg", "th", "tc", "tt", "ts"]:
            assert extract_format(t, {})["type"] == t

    def test_bad_type(self):
        with pytest.raises(ValueError):
            extract_format("tx", {})
        with pytest.raises(ValueError):
            extract_format("5j", {})


# ---------------------------------------------------------------------------
# Date / time types
# ---------------------------------------------------------------------------


class TestDateTime:
    def test_ti_date_only(self):
        assert parse.parse("{:ti}", "2011-03-15")[0] == datetime(2011, 3, 15)

    def test_ti_datetime_T(self):
        assert parse.parse("{:ti}", "2011-03-15T12:34")[0] == datetime(
            2011, 3, 15, 12, 34
        )

    def test_ti_datetime_space(self):
        assert parse.parse("{:ti}", "2011-03-15 12:34:56")[0] == datetime(
            2011, 3, 15, 12, 34, 56
        )

    def test_ti_microseconds(self):
        assert parse.parse("{:ti}", "2011-03-15T12:34:56.123")[0] == datetime(
            2011, 3, 15, 12, 34, 56, 123000
        )
        assert parse.parse("{:ti}", "2011-03-15T12:34:56.1234567")[0] == datetime(
            2011, 3, 15, 12, 34, 56, 123456
        )
        assert parse.parse("{:ti}", "2011-03-15T12:34:56.000001")[0] == datetime(
            2011, 3, 15, 12, 34, 56, 1
        )

    def test_ti_z(self):
        r = parse.parse("{:ti}", "2011-03-15T12:34:56Z")[0]
        assert r == datetime(2011, 3, 15, 12, 34, 56, tzinfo=tz(0, "UTC"))
        assert r.utcoffset() == timedelta(0)
        assert r.tzname() == "UTC"

    def test_ti_tz_colon(self):
        r = parse.parse("{:ti}", "2011-03-15T12:34:56+05:30")[0]
        assert r.utcoffset() == timedelta(hours=5, minutes=30)
        assert r.tzinfo == tz(330, "+05:30")

    def test_ti_tz_no_colon(self):
        r = parse.parse("{:ti}", "2011-03-15T12:34:56-0130")[0]
        assert r.utcoffset() == -timedelta(hours=1, minutes=30)

    def test_ti_tz_space(self):
        r = parse.parse("{:ti}", "2011-03-15 12:34 +0100")[0]
        assert r == datetime(2011, 3, 15, 12, 34, tzinfo=tz(60, "+0100"))

    def test_ti_followed_by_field(self):
        r = parse.parse("{:ti} {}", "2011-03-15T12:34:56Z rest")
        assert r[1] == "rest"

    def test_ti_invalid(self):
        assert parse.parse("{:ti}", "2011-3-15") is None

    def test_te(self):
        r = parse.parse("{:te}", "Mon, 21 Nov 2011 10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(600, "+1000"))

    def test_te_without_day(self):
        r = parse.parse("{:te}", "21 Nov 2011 10:21:36 -0500")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(-300, "-0500"))

    def test_te_three_digit_tz(self):
        r = parse.parse("{:te}", "21 Nov 2011 10:21:36 +100")[0]
        assert r.utcoffset() == timedelta(hours=1)

    def test_te_followed_by_field(self):
        r = parse.parse("{:te} {}", "21 Nov 2011 10:21:36 +1000 x")
        assert r[1] == "x"

    def test_tg(self):
        assert parse.parse("{:tg}", "20/1/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:tg}", "20-1-1972 10:21")[0] == datetime(
            1972, 1, 20, 10, 21
        )

    def test_tg_month_name(self):
        assert parse.parse("{:tg}", "20/Jan/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:tg}", "20/January/1972")[0] == datetime(1972, 1, 20)

    def test_tg_ampm(self):
        assert parse.parse("{:tg}", "20/1/1972 10:21 PM")[0] == datetime(
            1972, 1, 20, 22, 21
        )
        assert parse.parse("{:tg}", "20/1/1972 10:21 AM")[0] == datetime(
            1972, 1, 20, 10, 21
        )
        assert parse.parse("{:tg}", "20/1/1972 12:21 AM")[0] == datetime(
            1972, 1, 20, 0, 21
        )
        assert parse.parse("{:tg}", "20/1/1972 12:21 PM")[0] == datetime(
            1972, 1, 20, 12, 21
        )

    def test_tg_tz(self):
        r = parse.parse("{:tg}", "20/1/1972 10:21:36 PM +01:00")[0]
        assert r == datetime(1972, 1, 20, 22, 21, 36, tzinfo=tz(60, "+01:00"))

    def test_tg_followed_by_field(self):
        r = parse.parse("{:tg} {}", "20/1/1972 10:21 x")
        assert r.fixed == (datetime(1972, 1, 20, 10, 21), "x")

    def test_ta(self):
        assert parse.parse("{:ta}", "1/20/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:ta}", "1-20-1972 10:21:36 PM")[0] == datetime(
            1972, 1, 20, 22, 21, 36
        )

    def test_ta_month_name(self):
        assert parse.parse("{:ta}", "Feb/20/1972")[0] == datetime(1972, 2, 20)

    def test_ta_tz(self):
        r = parse.parse("{:ta}", "1/20/1972 10:21 -05:00")[0]
        assert r.utcoffset() == -timedelta(hours=5)

    def test_ta_followed_by_field(self):
        r = parse.parse("{:ta} {}", "1/20/1972 10:21 AM x")
        assert r.fixed == (datetime(1972, 1, 20, 10, 21), "x")

    def test_th(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=tz(600, "+1000"))

    def test_th_dash(self):
        r = parse.parse("{:th}", "21-Nov-2011:10:21:36 -0030")[0]
        assert r.utcoffset() == -timedelta(minutes=30)

    def test_th_followed_by_field(self):
        r = parse.parse("{:th} {}", "21/Nov/2011:10:21:36 +1000 x")
        assert r[1] == "x"

    def test_tc(self):
        r = parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0]
        assert r == datetime(1973, 9, 16, 1, 3, 52)

    def test_tc_followed_by_field(self):
        r = parse.parse("{:tc} {}", "Sun Sep 16 01:03:52 1973 x")
        assert r.fixed == (datetime(1973, 9, 16, 1, 3, 52), "x")

    def test_tt(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
        assert parse.parse("{:tt}", "10:21:36.5")[0] == time(10, 21, 36, 500000)

    def test_tt_ampm(self):
        assert parse.parse("{:tt}", "10:21 PM")[0] == time(22, 21)
        assert parse.parse("{:tt}", "12:00 AM")[0] == time(0, 0)
        assert parse.parse("{:tt}", "12:00 PM")[0] == time(12, 0)
        assert parse.parse("{:tt}", "1:05 am")[0] == time(1, 5)

    def test_tt_tz(self):
        r = parse.parse("{:tt}", "10:21:36 +0530")[0]
        assert r == time(10, 21, 36, tzinfo=tz(330, "+0530"))

    def test_tt_ampm_tz(self):
        r = parse.parse("{:tt}", "10:21:36 PM -07:00")[0]
        assert r == time(22, 21, 36, tzinfo=tz(-420, "-07:00"))

    def test_tt_followed_by_field(self):
        r = parse.parse("{:tt} {}", "10:21 PM x")
        assert r.fixed == (time(22, 21), "x")

    def test_ts(self):
        r = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
        assert r == datetime(datetime.today().year, 11, 9, 3, 37, 44)

    def test_ts_followed_by_field(self):
        r = parse.parse("{:ts} {}", "Nov 21 10:21:36 host")
        assert r.fixed == (
            datetime(datetime.today().year, 11, 21, 10, 21, 36),
            "host",
        )

    def test_named_datetime(self):
        r = parse.parse("{when:ti} {who}", "2020-02-29 bob")
        assert r.named == {"when": datetime(2020, 2, 29), "who": "bob"}

    def test_multiple_datetimes(self):
        r = parse.parse(
            "{:ti} {:tg} {:tt} {}",
            "2020-01-02T03:04:05 6/7/2008 09:10 PM end",
        )
        assert r.fixed == (
            datetime(2020, 1, 2, 3, 4, 5),
            datetime(2008, 7, 6),
            time(21, 10),
            "end",
        )


class TestFixedTzOffset:
    def test_basics(self):
        t = FixedTzOffset(90, "+01:30")
        assert t.utcoffset(None) == timedelta(minutes=90)
        assert t.tzname(None) == "+01:30"
        assert t.dst(None) == timedelta(0)

    def test_repr(self):
        assert repr(FixedTzOffset(60, "+01:00")) == "<FixedTzOffset +01:00 1:00:00>"

    def test_eq(self):
        assert FixedTzOffset(60, "a") == FixedTzOffset(60, "a")
        assert FixedTzOffset(60, "a") != FixedTzOffset(60, "b")
        assert FixedTzOffset(60, "a") != FixedTzOffset(30, "a")
        assert FixedTzOffset(60, "a") != "a"

    def test_negative(self):
        assert FixedTzOffset(-90, "x").utcoffset(None) == -timedelta(minutes=90)


class TestDateConvert:
    def test_month_name_mapping(self):
        for name, num in parse.MONTHS_MAP.items():
            r = parse.parse("{:tg}", "1/%s/2000" % name)
            assert r[0] == datetime(2000, num, 1), name


# ---------------------------------------------------------------------------
# strftime-style formats
# ---------------------------------------------------------------------------


class TestStrftime:
    def test_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-04-05")[0]
        assert r == date(2023, 4, 5)
        assert type(r) is date

    def test_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 06:07:08")[0]
        assert r == datetime(2023, 4, 5, 6, 7, 8)
        assert type(r) is datetime

    def test_time(self):
        r = parse.parse("{:%H:%M:%S}", "06:07:08")[0]
        assert r == time(6, 7, 8)
        assert type(r) is time

    def test_time_with_microseconds(self):
        assert parse.parse("{:%H:%M:%S.%f}", "06:07:08.123")[0] == time(
            6, 7, 8, 123000
        )

    def test_time_with_tz(self):
        r = parse.parse("{:%H:%M %z}", "06:07 +0100")[0]
        assert r.utcoffset() == timedelta(hours=1)

    def test_ampm(self):
        assert parse.parse("{:%I:%M %p}", "03:30 PM")[0] == time(15, 30)

    def test_names(self):
        r = parse.parse("{:%A, %B %d %Y}", "Wednesday, April 05 2023")[0]
        assert r == date(2023, 4, 5)
        r = parse.parse("{:%a %b %d %Y}", "Wed Apr 05 2023")[0]
        assert r == date(2023, 4, 5)

    def test_two_digit_year(self):
        assert parse.parse("{:%d/%m/%y}", "05/04/23")[0] == date(2023, 4, 5)

    def test_no_year_uses_current(self):
        r = parse.parse("{:%m-%d}", "04-05")[0]
        assert r == date(datetime.today().year, 4, 5)

    def test_day_of_year(self):
        assert parse.parse("{:%Y %j}", "2023 095")[0] == date(2023, 4, 5)

    def test_in_context(self):
        r = parse.parse("on {d:%Y/%m/%d} at {t:%H:%M}", "on 2023/04/05 at 13:45")
        assert r["d"] == date(2023, 4, 5)
        assert r["t"] == time(13, 45)

    def test_no_match(self):
        assert parse.parse("{:%Y-%m-%d}", "23-04-05") is None

    def test_strf_date_convert_neither(self):
        with pytest.raises(ValueError):
            parse.strf_date_convert("x", None, "x")

    def test_regex_for_format(self):
        assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
        assert parse.get_regex_for_datetime_format("abc") == "abc"


# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------


class TestCustomTypes:
    def test_simple_converter(self):
        def upper(s):
            return s.upper()

        r = parse.parse("{:upper}", "abc", extra_types={"upper": upper})
        assert r[0] == "ABC"

    def test_converter_with_pattern(self):
        @with_pattern(r"\d+")
        def number(s):
            return int(s)

        r = parse.parse("{:Number}{}", "123abc", extra_types={"Number": number})
        assert r.fixed == (123, "abc")

    def test_pattern_restricts_match(self):
        @with_pattern(r"[ab]+")
        def ab(s):
            return s

        assert parse.parse("{:ab}", "abc", extra_types={"ab": ab}) is None

    def test_with_pattern_attributes(self):
        @with_pattern(r"x", regex_group_count=2)
        def f(s):
            return s

        assert f.pattern == "x"
        assert f.regex_group_count == 2
        assert f("y") == "y"

    def test_with_pattern_default_group_count(self):
        @with_pattern(r"x")
        def f(s):
            return s

        assert f.regex_group_count is None

    def test_pattern_with_groups(self):
        @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
        def rng(s):
            a, b = s.split("-")
            return (int(a), int(b))

        r = parse.parse("{:Range} {} {name}", "1-5 x y", extra_types={"Range": rng})
        assert r.fixed == ((1, 5), "x")
        assert r.named == {"name": "y"}

    def test_pattern_with_groups_named(self):
        @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
        def ab(s):
            return s

        r = parse.parse("{x:AB} {}", "ac z", extra_types={"AB": ab})
        assert r.named == {"x": "ac"}
        assert r.fixed == ("z",)

    def test_override_builtin_type(self):
        def custom_d(s):
            return "custom:" + s

        r = parse.parse("{:d}", "5", extra_types={"d": custom_d})
        assert r[0] == "custom:5"

    def test_custom_type_named(self):
        def yesno(s):
            return s.lower() == "yes"

        yesno.pattern = r"yes|no"
        r = parse.parse("{a:YesNo},{b:YesNo}", "yes,NO", extra_types={"YesNo": yesno})
        assert r.named == {"a": True, "b": False}

    def test_extra_types_with_search_and_findall(self):
        @with_pattern(r"\d+")
        def num(s):
            return int(s)

        et = {"N": num}
        assert parse.search("{:N}", "ab 12 cd", extra_types=et)[0] == 12
        assert [r[0] for r in parse.findall("{:N}", "1 22 333", extra_types=et)] == [
            1,
            22,
            333,
        ]

    def test_convert_first_pickleable(self):
        c = parse.convert_first(int)
        c2 = pickle.loads(pickle.dumps(c))
        assert c2("5", None) == 5


# ---------------------------------------------------------------------------
# search / findall
# ---------------------------------------------------------------------------


class TestSearch:
    def test_basic(self):
        r = search("age: {:d}", "name: bob, age: 42, height: 180")
        assert r[0] == 42

    def test_no_match(self):
        assert search("age: {:d}", "nothing here") is None

    def test_pos(self):
        r = search("{:d}", "1 2 3", pos=1)
        assert r[0] == 2

    def test_endpos(self):
        assert search("{:d}", "abc 123", endpos=4) is None
        assert search("{:d}", "abc 123", endpos=5)[0] == 1

    def test_case_sensitive(self):
        assert search("X{:d}", "x5", case_sensitive=True) is None
        assert search("X{:d}", "x5")[0] == 5

    def test_evaluate_result_false(self):
        m = search("{:d}", "a 5", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert r[0] == 5

    def test_lazy_field_minimal(self):
        r = search("a{}b", "xxaYYbZZb")
        assert r[0] == "YY"


class TestFindall:
    def test_basic(self):
        results = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
        assert [r[0] for r in results] == ["some ", "bold", " text"]

    def test_returns_iterator(self):
        it = findall("{:d}", "1 2")
        assert isinstance(it, ResultIterator)
        assert iter(it) is it
        assert next(it)[0] == 1
        assert next(it)[0] == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_numbers(self):
        assert [r[0] for r in findall("{:d}", "a1b22c333")] == [1, 22, 333]

    def test_no_matches(self):
        assert list(findall("{:d}", "abc")) == []

    def test_pos_endpos(self):
        assert [r[0] for r in findall("{:d}", "1 2 3 4", pos=2, endpos=5)] == [2, 3]

    def test_non_overlapping(self):
        assert [r[0] for r in findall("{:w}", "ab cd ef")] == ["ab", "cd", "ef"]

    def test_evaluate_result_false(self):
        ms = list(findall("{:d}", "1 2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in ms)
        assert [m.evaluate_result()[0] for m in ms] == [1, 2]

    def test_case_sensitive(self):
        assert list(findall("a{:d}", "A1 a2", case_sensitive=True))[0][0] == 2
        assert len(list(findall("a{:d}", "A1 a2"))) == 2

    def test_named(self):
        res = list(findall("{k}={v:d};", "a=1;b=2;"))
        assert [r.named for r in res] == [{"k": "a", "v": 1}, {"k": "b", "v": 2}]

    def test_parser_findall(self):
        p = compile("{:d}")
        assert [r[0] for r in p.findall("9 8 7")] == [9, 8, 7]

    def test_spans(self):
        res = list(findall("{:d}", "aa_12_bb_345"))
        assert [r.spans[0] for r in res] == [(3, 5), (9, 12)]


# ---------------------------------------------------------------------------
# Parser / compile
# ---------------------------------------------------------------------------


class TestParser:
    def test_compile_returns_parser(self):
        p = compile("{} {}")
        assert isinstance(p, Parser)

    def test_reuse(self):
        p = compile("{:d}+{:d}")
        assert p.parse("1+2").fixed == (1, 2)
        assert p.parse("3+4").fixed == (3, 4)
        assert p.parse("a+b") is None

    def test_parse_then_search_same_parser(self):
        p = compile("{:d}")
        assert p.search("a 5 b")[0] == 5
        assert p.parse("5")[0] == 5
        assert p.parse("a 5") is None

    def test_format_property(self):
        assert compile("{a} {b}").format == "{a} {b}"

    def test_named_fields(self):
        p = compile("{a} {} {b.c}")
        assert p.named_fields == ["a", "b_c"]

    def test_named_fields_returns_copy(self):
        p = compile("{a}")
        p.named_fields.append("x")
        assert p.named_fields == ["a"]

    def test_fixed_fields(self):
        p = compile("{} {a} {} {:g} {}")
        assert len(p.fixed_fields) == 4

    def test_fixed_fields_returns_copy(self):
        p = compile("{}")
        p.fixed_fields.append(99)
        assert len(p.fixed_fields) == 1

    def test_repr_short(self):
        assert repr(compile("{}")) == "<Parser '{}'>"

    def test_repr_long(self):
        fmt = "a" * 25
        assert repr(compile(fmt)) == "<Parser %r>" % ("a" * 17 + "...")

    def test_repr_boundary(self):
        fmt = "a" * 20
        assert repr(compile(fmt)) == "<Parser %r>" % fmt

    def test_parse_evaluate_result_false(self):
        p = compile("{:d}")
        m = p.parse("5", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.parser is p
        assert m.match.group(0) == "5"
        assert m.evaluate_result()[0] == 5

    def test_module_parse_evaluate_result_false(self):
        m = parse.parse("{a:d}", "7", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.evaluate_result()["a"] == 7

    def test_parse_evaluate_result_false_no_match(self):
        assert parse.parse("{:d}", "x", evaluate_result=False) is None

    def test_evaluate_result_method(self):
        p = compile("{x:d}")
        m = p._match_re.match("12")
        assert p.evaluate_result(m)["x"] == 12

    def test_compile_case_sensitive(self):
        p = compile("A{}", case_sensitive=True)
        assert p.parse("a1") is None
        assert p.parse("A1")[0] == "1"

    def test_compile_extra_types(self):
        p = compile("{:Up}", extra_types={"Up": lambda s: s.upper()})
        assert p.parse("x")[0] == "X"

    def test_too_many_fields(self):
        assert issubclass(TooManyFields, ValueError)
        # Modern Python supports many named groups, so this should simply work
        fmt = " ".join("{f%d}" % i for i in range(150))
        text = " ".join(str(i) for i in range(150))
        r = parse.parse(fmt, text)
        assert r["f0"] == "0"
        assert r["f149"] == "149"

    def test_many_fixed_fields(self):
        fmt = ",".join("{:d}" for _ in range(50))
        text = ",".join(str(i) for i in range(50))
        assert parse.parse(fmt, text).fixed == tuple(range(50))

    def test_all(self):
        assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}

    def test_version(self):
        assert isinstance(parse.__version__, str)


# ---------------------------------------------------------------------------
# Round-trip with str.format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,values",
    [
        ("{} {}", ("a", "b")),
        ("{:d}/{:d}", (12, -34)),
        ("{:x}", (255,)),
        ("{:o}", (8,)),
        ("{:b}", (5,)),
        ("{:5d}", (42,)),
        ("{:<5d}|", (42,)),
        ("{:^7d}|", (42,)),
        ("{:05d}", (42,)),
        ("{:,d}", (1234567,)),
        ("{:_d}", (1234567,)),
        ("{:.2f}", (3.14,)),
        ("{:.0f}", (20.0,)),
        ("{:10.3f}", (-2.5,)),
        ("{:e}", (12345.678,)),
        ("{:g}", (0.5,)),
        ("{:%}", (0.25,)),
    ],
)
def test_round_trip(fmt, values):
    text = fmt.format(*values)
    r = parse.parse(fmt, text)
    assert r is not None, (fmt, text)
    for got, expected in zip(r.fixed, values):
        if isinstance(expected, float):
            assert got == pytest.approx(expected)
        else:
            assert got == expected


def test_round_trip_named():
    fmt = "{name:>}: {age:d} years, {ratio:.1%}"
    text = "{name:>10}: {age:d} years, {ratio:.1%}".format(
        name="Alice", age=30, ratio=0.5
    )
    r = parse.parse(fmt, text)
    assert r["name"] == "Alice"
    assert r["age"] == 30
    assert r["ratio"] == pytest.approx(0.5)
