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
    extract_format,
    findall,
    int_convert,
    search,
    with_pattern,
)


THIS_YEAR = datetime.today().year


def utc_offset(minutes):
    return timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------


class TestBasic:
    def test_no_fields_exact_match(self):
        r = parse.parse("hello", "hello")
        assert isinstance(r, Result)
        assert r.fixed == ()
        assert r.named == {}

    def test_no_fields_no_match(self):
        assert parse.parse("hello", "hello!") is None
        assert parse.parse("hello", "hell") is None

    def test_empty_format_and_string(self):
        r = parse.parse("", "")
        assert r.fixed == ()
        assert parse.parse("", "x") is None

    def test_fixed_field(self):
        r = parse.parse("hello {}", "hello world")
        assert r.fixed == ("world",)
        assert r[0] == "world"

    def test_multiple_fixed(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r.fixed == ("a", "b", "c")
        assert r[1:] == ("b", "c")
        assert r[-1] == "c"

    def test_fixed_is_tuple(self):
        r = parse.parse("{}-{}", "x-y")
        assert type(r.fixed) is tuple

    def test_named_field(self):
        r = parse.parse("hello {name}", "hello world")
        assert r.named == {"name": "world"}
        assert r["name"] == "world"
        assert "name" in r
        assert "other" not in r
        assert r.fixed == ()

    def test_mixed_named_and_fixed(self):
        r = parse.parse("{} {name} {}", "a b c")
        assert r.fixed == ("a", "c")
        assert r.named == {"name": "b"}

    def test_numeric_index_fields_are_fixed(self):
        r = parse.parse("{0} {1}", "a b")
        assert r.fixed == ("a", "b")
        assert r.named == {}

    def test_lazy_matching(self):
        r = parse.parse("{}x{}", "axbxc")
        assert r.fixed == ("a", "bxc")

    def test_braces_escaped(self):
        r = parse.parse("{{}} {}", "{} x")
        assert r.fixed == ("x",)
        r = parse.parse("{{hello}}", "{hello}")
        assert r is not None
        assert r.fixed == ()
        assert parse.parse("{{hello}}", "hello") is None

    def test_regex_special_chars_escaped(self):
        for ch in "?\\.[]()*+^$!|":
            fmt = "a%sb {}" % ch
            r = parse.parse(fmt, "a%sb x" % ch)
            assert r is not None, ch
            assert r.fixed == ("x",)
        # "." must not act as a wildcard
        assert parse.parse("a.b", "axb") is None
        assert parse.parse("a*", "aaa") is None
        assert parse.parse("a+", "aa") is None
        assert parse.parse("(a)", "a") is None
        assert parse.parse("a|b", "a") is None
        assert parse.parse("^a$", "^a$") is not None
        assert parse.parse("[ab]", "a") is None
        assert parse.parse("a?", "a") is None

    def test_dotall(self):
        r = parse.parse("{}", "a\nb")
        assert r.fixed == ("a\nb",)

    def test_case_insensitive_default(self):
        assert parse.parse("Hello {}", "hELLO x").fixed == ("x",)

    def test_case_sensitive(self):
        assert parse.parse("Hello {}", "hello x", case_sensitive=True) is None
        assert parse.parse("Hello {}", "Hello x", case_sensitive=True).fixed == ("x",)
        assert search("Hello", "hello", case_sensitive=True) is None
        assert search("Hello", "hello") is not None
        assert list(findall("A", "aAa", case_sensitive=True)).__len__() == 1
        assert len(list(findall("A", "aAa"))) == 3

    def test_compile_case_sensitive(self):
        p = compile("A{}", case_sensitive=True)
        assert p.parse("ax") is None
        assert p.parse("Ax").fixed == ("x",)

    def test_parse_must_match_entire_string(self):
        assert parse.parse("{:d}", "12 ") is None
        assert parse.parse("{:d}", " 12x") is None
        assert parse.parse("a{}", "a") is None

    def test_parse_no_trailing_newline_match(self):
        # \Z must be used rather than $
        assert parse.parse("abc", "abc\n") is None


# ---------------------------------------------------------------------------
# Named field handling
# ---------------------------------------------------------------------------


class TestNamed:
    def test_dotted_name(self):
        r = parse.parse("{hello.world}", "x")
        assert r.named == {"hello.world": "x"}

    def test_dotted_and_underscore_no_collision(self):
        r = parse.parse("{a.b} {a_b}", "1 2")
        assert r.named == {"a.b": "1", "a_b": "2"}
        r = parse.parse("{a_b} {a.b}", "1 2")
        assert r.named == {"a_b": "1", "a.b": "2"}

    def test_hyphen_name(self):
        r = parse.parse("{a-b}", "x")
        assert r.named == {"a-b": "x"}

    def test_hyphen_collision(self):
        r = parse.parse("{a_b} {a-b}", "1 2")
        assert r.named == {"a_b": "1", "a-b": "2"}

    def test_underscore_collisions_triple(self):
        r = parse.parse("{a.b} {a_b} {a__b}", "1 2 3")
        assert r["a.b"] == "1"
        assert r["a_b"] == "2"
        assert r["a__b"] == "3"

    def test_dict_style(self):
        r = parse.parse("{hello[world]}", "x")
        assert r.named == {"hello": {"world": "x"}}

    def test_nested_dict_style(self):
        r = parse.parse("{a[b][c]} {a[d]}", "1 2")
        assert r.named == {"a": {"b": {"c": "1"}, "d": "2"}}
        assert r["a"]["b"]["c"] == "1"

    def test_dict_style_with_types(self):
        r = parse.parse("{p[x]:d},{p[y]:d}", "3,4")
        assert r.named == {"p": {"x": 3, "y": 4}}

    def test_dict_style_spans(self):
        r = parse.parse("{a[b]} {c}", "xx yyy")
        assert r.spans["a[b]"] == (0, 2)
        assert r.spans["c"] == (3, 6)

    def test_repeated_name_must_match(self):
        r = parse.parse("{x} {x}", "a a")
        assert r.named == {"x": "a"}
        assert parse.parse("{x} {x}", "a b") is None

    def test_repeated_name_typed(self):
        r = parse.parse("{x:d} {x:d}", "1 1")
        assert r.named == {"x": 1}
        assert parse.parse("{x:d} {x:d}", "1 2") is None

    def test_repeated_name_type_mismatch(self):
        with pytest.raises(RepeatedNameError):
            Parser("{x:d} {x:w}")
        with pytest.raises(ValueError):
            Parser("{x} {x:d}")

    def test_repeated_name_error_is_value_error(self):
        assert issubclass(RepeatedNameError, ValueError)
        assert issubclass(TooManyFields, ValueError)

    def test_named_fields_property(self):
        p = compile("{a} {b.c} {}")
        assert p.named_fields == ["a", "b_c"]
        # fixed fields are recorded by regex group index
        assert p.fixed_fields == [2]
        # returns a copy
        p.named_fields.append("z")
        p.fixed_fields.append(9)
        assert p.named_fields == ["a", "b_c"]
        assert p.fixed_fields == [2]

    def test_fixed_fields_skip_type_groups(self):
        p = compile("{:g} {}")
        # {:g} uses 3 regex groups
        assert p.fixed_fields == [0, 3]

    def test_repeated_name_not_duplicated_in_named_fields(self):
        p = compile("{a} {a}")
        assert p.named_fields == ["a"]

    def test_duplicate_group_name_keyerror(self):
        p = Parser("")
        p._to_group_name("abc")
        with pytest.raises(KeyError):
            p._to_group_name("abc")

    def test_group_name_mapping(self):
        p = Parser("{a.b}")
        assert p._name_to_group_map == {"a.b": "a_b"}
        assert p._group_to_name_map == {"a_b": "a.b"}

    def test_to_group_name_collision_variants(self):
        p = Parser("")
        assert p._to_group_name("a_b") == "a_b"
        assert p._to_group_name("a.b") == "a__b"
        p = Parser("")
        assert p._to_group_name("a_b") == "a_b"
        assert p._to_group_name("a-b") == "a__b"
        p = Parser("")
        assert p._to_group_name("a__b") == "a__b"
        assert p._to_group_name("a_b") == "a_b"
        p = Parser("")
        assert p._to_group_name("a_b") == "a_b"
        assert p._to_group_name("a__b") == "a__b"
        assert p._to_group_name("a.b") == "a___b"
        p = Parser("")
        assert p._to_group_name("a[b]") == "a_b_"

    def test_name_starting_with_digit_is_fixed(self):
        r = parse.parse("{0:d}", "5")
        assert r.fixed == (5,)

    def test_name_starting_with_underscore_is_fixed(self):
        r = parse.parse("{_x}", "v")
        assert r.fixed == ("v",)
        assert r.named == {}


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class TestSpans:
    def test_spans_fixed_and_named(self):
        r = parse.parse("{} {name} {}", "ab cde f")
        assert r.spans == {0: (0, 2), "name": (3, 6), 1: (7, 8)}

    def test_spans_with_typed_groups(self):
        # types adding extra regex groups must not shift the spans
        r = parse.parse("{:g} {:ti} {}", "1.5e3 2020-01-02 xyz")
        assert r.spans[0] == (0, 5)
        assert r.spans[1] == (6, 16)
        assert r.spans[2] == (17, 20)
        assert r.fixed[2] == "xyz"

    def test_spans_search(self):
        r = search("{:d}", "abc 123 def")
        # the leading space is accepted as a sign position
        assert r.spans == {0: (3, 7)}
        assert r[0] == 123
        r = search("{:d}", "abc:123 def")
        assert r.spans == {0: (4, 7)}

    def test_spans_padded(self):
        r = parse.parse("[{:>}]", "[  abc]")
        assert r.fixed == ("abc",)
        assert r.spans[0] == (3, 6)


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "text, value",
        [
            ("0", 0),
            ("12", 12),
            ("-12", -12),
            ("+12", 12),
            (" 12", 12),
            ("0x1f", 31),
            ("0X1F", 31),
            ("-0x1f", -31),
            ("0b101", 5),
            ("0B101", 5),
            ("0o17", 15),
            ("0O17", 15),
            ("+0o17", 15),
            ("007", 7),
            ("10", 10),
        ],
    )
    def test_d(self, text, value):
        r = parse.parse("{:d}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is int

    def test_d_rejects(self):
        assert parse.parse("{:d}", "abc") is None
        assert parse.parse("{:d}", "1.5") is None
        assert parse.parse("{:d}", "") is None
        assert parse.parse("{:d}", "0b2") is None
        assert parse.parse("{:d}", "0o8") is None
        assert parse.parse("{:d}", "0xg") is None
        assert parse.parse("{:d}", "1-") is None

    def test_d_grouping(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
        assert parse.parse("{:_d}", "1_234")[0] == 1234
        assert parse.parse("{:d}", "1,234") is None
        assert parse.parse("{:d}", "1_234") is None
        assert parse.parse("{:,d}", "1_234") is None
        assert parse.parse("{:_d}", "1,234") is None

    def test_d_width(self):
        r = parse.parse("{:2d}{:2d}", "1234")
        assert r.fixed == (12, 34)
        r = parse.parse("{:3d}", "  5")
        assert r[0] == 5
        assert parse.parse("{:2d}", "123") is None

    def test_d_zero_padding(self):
        assert parse.parse("{:03d}", "007")[0] == 7
        assert parse.parse("{:03d}", "-07")[0] == -7

    def test_d_left_align(self):
        assert parse.parse("{:<5d}|", "42   |")[0] == 42

    def test_d_center_align(self):
        assert parse.parse("|{:^6d}|", "|  42  |")[0] == 42

    def test_d_eq_align(self):
        r = parse.parse("{:=5d}", "-0042")
        assert r[0] == -42
        r = parse.parse("{:x=5d}", "xxx42")
        assert r[0] == 42

    def test_d_two_fields_adjacent(self):
        r = parse.parse("{:d}{:w}", "12abc")
        assert r.fixed == (12, "abc")

    def test_d_named(self):
        r = parse.parse("{n:d}", "-5")
        assert r["n"] == -5

    @pytest.mark.parametrize(
        "fmt, text, value",
        [
            ("{:b}", "1010", 10),
            ("{:b}", "0b1010", 10),
            ("{:b}", "-0b11", -3),
            ("{:o}", "17", 15),
            ("{:o}", "0o17", 15),
            ("{:o}", "-17", -15),
            ("{:x}", "ff", 255),
            ("{:x}", "0xff", 255),
            ("{:x}", "FF", 255),
            ("{:X}", "0XFF", 255),
            ("{:x}", "-1a", -26),
            ("{:x}", "10", 16),
            ("{:o}", "10", 8),
            ("{:b}", "10", 2),
        ],
    )
    def test_bases(self, fmt, text, value):
        r = parse.parse(fmt, text)
        assert r is not None
        assert r[0] == value

    def test_bases_reject(self):
        assert parse.parse("{:b}", "12") is None
        assert parse.parse("{:o}", "8") is None
        assert parse.parse("{:x}", "g") is None

    def test_base_fields_followed_by_fields(self):
        # fixed field group indices must stay correct after typed groups
        r = parse.parse("{:x} {:b} {:o} {}", "0xa 0b1 0o7 end")
        assert r.fixed == (10, 1, 7, "end")

    @pytest.mark.parametrize(
        "text, value",
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
        r = parse.parse("{:n}", text)
        assert r is not None
        assert r[0] == value

    def test_n_reject(self):
        assert parse.parse("{:n}", "1234") is None
        assert parse.parse("{:n}", "1,23") is None

    def test_n_followed_by_field(self):
        r = parse.parse("{:n} {}", "1,234 x")
        assert r.fixed == (1234, "x")


class TestIntConvert:
    def test_auto_base(self):
        c = int_convert()
        assert c("0x10", None) == 16
        assert c("0b10", None) == 2
        assert c("0o10", None) == 8
        assert c("010", None) == 10
        assert c("10", None) == 10
        assert c("-0x10", None) == -16
        assert c("+0b10", None) == 2

    def test_short_strings(self):
        c = int_convert()
        assert c("0", None) == 0
        assert c("00", None) == 0
        assert c("-0", None) == 0
        assert c("5", None) == 5
        assert c("-5", None) == -5
        assert c("+5", None) == 5

    def test_three_char_prefixed(self):
        c = int_convert()
        assert c("0x1", None) == 1
        assert c("0b1", None) == 1
        assert c("0o7", None) == 7
        assert c("-0x1", None) == -1

    def test_explicit_base(self):
        assert int_convert(16)("ff", None) == 255
        assert int_convert(16)("0xff", None) == 255
        assert int_convert(2)("0b11", None) == 3
        assert int_convert(8)("0o11", None) == 9
        assert int_convert(10)("1,234", None) == 1234
        assert int_convert(36)("z", None) == 35

    def test_strips_garbage(self):
        assert int_convert(10)("1_000", None) == 1000
        assert int_convert()("1,000", None) == 1000

    def test_uppercase(self):
        assert int_convert(16)("FF", None) == 255
        assert int_convert()("0XFF", None) == 255

    def test_chars(self):
        assert int_convert.CHARS == "0123456789abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Float / decimal types
# ---------------------------------------------------------------------------


class TestFloats:
    @pytest.mark.parametrize(
        "text, value",
        [
            ("1.5", 1.5),
            (".5", 0.5),
            ("-1.5", -1.5),
            ("+1.5", 1.5),
            ("0.25", 0.25),
        ],
    )
    def test_f(self, text, value):
        r = parse.parse("{:f}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is float

    def test_f_requires_decimal_point(self):
        assert parse.parse("{:f}", "1") is None
        assert parse.parse("{:f}", "1.") is None

    def test_f_nan_inf(self):
        import math

        assert math.isnan(parse.parse("{:f}", "nan")[0])
        assert math.isnan(parse.parse("{:f}", "NAN")[0])
        assert parse.parse("{:f}", "inf")[0] == float("inf")
        assert parse.parse("{:f}", "INF")[0] == float("inf")
        assert parse.parse("{:f}", "-inf")[0] == float("-inf")

    def test_f_negative_followed(self):
        r = parse.parse("{:f} {:f}", "-1.5 -2.5")
        assert r.fixed == (-1.5, -2.5)

    def test_f_precision_zero(self):
        r = parse.parse("{:.0f}", "20")
        assert r[0] == 20.0
        assert type(r[0]) is float
        assert parse.parse("{:.0f}", "-20")[0] == -20.0
        assert parse.parse("{:.0f}", "20.5") is None

    def test_f_precision(self):
        assert parse.parse("{:.2f}", "3.14")[0] == 3.14
        assert parse.parse("{:.2f}", "3") is None

    def test_f_width(self):
        assert parse.parse("{:8.2f}", "    3.14")[0] == 3.14

    def test_F_decimal(self):
        r = parse.parse("{:F}", "1.25")
        assert r[0] == Decimal("1.25")
        assert type(r[0]) is Decimal
        assert parse.parse("{:F}", "-1.25")[0] == Decimal("-1.25")
        assert parse.parse("{:.0F}", "7")[0] == Decimal("7")

    @pytest.mark.parametrize(
        "text, value",
        [
            ("1.0e10", 1.0e10),
            ("1.5E-3", 1.5e-3),
            ("-1.5e+3", -1500.0),
            (".5e1", 5.0),
        ],
    )
    def test_e(self, text, value):
        r = parse.parse("{:e}", text)
        assert r is not None
        assert r[0] == value

    def test_e_reject(self):
        assert parse.parse("{:e}", "1.5") is None
        assert parse.parse("{:e}", "1e5") is None

    def test_e_inf_nan(self):
        import math

        assert parse.parse("{:e}", "inf")[0] == float("inf")
        assert parse.parse("{:e}", "-inf")[0] == float("-inf")
        assert parse.parse("{:E}", "INF")[0] == float("inf")
        assert math.isnan(parse.parse("{:e}", "nan")[0])

    @pytest.mark.parametrize(
        "text, value",
        [
            ("1", 1.0),
            ("1.5", 1.5),
            ("-1.5", -1.5),
            ("1e5", 1e5),
            ("1.5e-5", 1.5e-5),
            ("1.5E+5", 1.5e5),
            ("+3", 3.0),
        ],
    )
    def test_g(self, text, value):
        r = parse.parse("{:g}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is float

    def test_g_inf_nan(self):
        import math

        assert parse.parse("{:g}", "inf")[0] == float("inf")
        assert parse.parse("{:G}", "-INF")[0] == float("-inf")
        assert math.isnan(parse.parse("{:g}", "nan")[0])

    def test_g_followed_by_field(self):
        r = parse.parse("{:g} {} {name}", "1.5e3 x y")
        assert r.fixed == (1500.0, "x")
        assert r["name"] == "y"

    def test_percentage(self):
        assert parse.parse("{:%}", "50%")[0] == 0.5
        assert parse.parse("{:%}", "12.5%")[0] == 0.125
        assert parse.parse("{:%}", "-50%")[0] == -0.5
        assert parse.parse("{:%}", "50") is None

    def test_percentage_followed_by_field(self):
        r = parse.parse("{:%} {}", "12.5% x")
        assert r.fixed == (0.125, "x")

    def test_percentage_function(self):
        assert parse.percentage("25%", None) == 0.25
        assert parse.percentage("100%", None) == 1.0

    def test_numeric_sign_with_space(self):
        assert parse.parse("{:f}", " 1.5")[0] == 1.5

    def test_numeric_padding_fill(self):
        assert parse.parse("{:*>8.2f}", "****3.14")[0] == 3.14
        assert parse.parse("{:*<8.2f}", "3.14****")[0] == 3.14
        assert parse.parse("{:*^8.2f}", "**3.14**")[0] == 3.14


# ---------------------------------------------------------------------------
# String types
# ---------------------------------------------------------------------------


class TestStringTypes:
    def test_w(self):
        assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
        assert parse.parse("{:w}", "abc def") is None
        assert parse.parse("{:w}", "") is None

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
        assert parse.parse("{:l}", "abcDEF")[0] == "abcDEF"
        assert parse.parse("{:l}", "abc1") is None
        assert parse.parse("{:l}", "abc_") is None

    def test_w_greedy_with_following(self):
        r = parse.parse("{:w} {:w}", "ab cd")
        assert r.fixed == ("ab", "cd")

    def test_string_width(self):
        # width is the minimum length of the captured text
        r = parse.parse("{:4}|", "ab  |")
        assert r[0] == "ab  "
        r = parse.parse("{:>4}|", "  ab|")
        assert r[0] == "  ab"
        # extra leading padding beyond the width is stripped (implicit ">")
        r = parse.parse("{:4}|", "   abcd|")
        assert r[0] == "abcd"
        r = parse.parse("{:<4}|", "abcd   |")
        assert r[0] == "abcd"

    def test_string_min_width(self):
        assert parse.parse("{:3}", "ab") is None
        assert parse.parse("{:3}", "abc")[0] == "abc"
        assert parse.parse("{:3}", "abcd")[0] == "abcd"

    def test_string_precision(self):
        assert parse.parse("{:.2}", "ab")[0] == "ab"
        assert parse.parse("{:.2}", "abc") is None
        assert parse.parse("{:.2}", "a")[0] == "a"
        r = parse.parse("{:.2}{}", "abcd")
        assert r.fixed == ("a", "bcd")

    def test_string_width_and_precision(self):
        assert parse.parse("{:2.4}", "a") is None
        assert parse.parse("{:2.4}", "ab")[0] == "ab"
        assert parse.parse("{:2.4}", "abcd")[0] == "abcd"
        assert parse.parse("{:2.4}", "abcde") is None

    def test_left_align(self):
        r = parse.parse("{:<}|", "ab   |")
        assert r[0] == "ab"
        r = parse.parse("{:<2}|", "ab   |")
        assert r[0] == "ab"

    def test_right_align(self):
        r = parse.parse("|{:>}", "|   ab")
        assert r[0] == "ab"

    def test_center_align(self):
        r = parse.parse("|{:^}|", "|  ab  |")
        assert r[0] == "ab"
        r = parse.parse("|{:^2}|", "|  ab   |")
        assert r[0] == "ab"

    def test_fill_char(self):
        assert parse.parse("{:x<}|", "abxxx|")[0] == "ab"
        assert parse.parse("|{:x>}", "|xxxab")[0] == "ab"
        assert parse.parse("|{:x^}|", "|xxabxx|")[0] == "ab"

    @pytest.mark.parametrize("fill", list(".\\+?*[](){}^$"))
    def test_regex_special_fill(self, fill):
        fmt = "|{:%s^}|" % fill
        if fill in "{}^":
            # braces can't be expressed through PARSE_RE as a fill, and "^^"
            # is read as align "^" with type "^"
            return
        r = parse.parse(fmt, "|%sab%s|" % (fill * 2, fill * 3))
        assert r is not None, fill
        assert r[0] == "ab"

    def test_fill_dot_is_literal(self):
        r = parse.parse("|{:.<}|", "|ab..|")
        assert r[0] == "ab"
        r = parse.parse("|{:.<}|", "|abxx|")
        assert r[0] == "abxx"


# ---------------------------------------------------------------------------
# Format spec extraction
# ---------------------------------------------------------------------------


class TestExtractFormat:
    def test_simple_type(self):
        f = extract_format("d", {})
        assert f["type"] == "d"
        assert f["fill"] is None
        assert f["align"] is None
        assert f["zero"] is False
        assert f["width"] == ""

    def test_align(self):
        for a in "<>=^":
            f = extract_format(a + "d", {})
            assert f["align"] == a
            assert f["fill"] is None
            assert f["type"] == "d"

    def test_fill_align(self):
        f = extract_format("*>10d", {})
        assert f["fill"] == "*"
        assert f["align"] == ">"
        assert f["width"] == "10"
        assert f["type"] == "d"

    def test_align_char_alone_is_align(self):
        f = extract_format("<", {})
        assert f["align"] == "<"
        assert f["fill"] is None
        assert f["type"] == ""

    def test_sign(self):
        for s in "+- ":
            f = extract_format(s + "5d", {})
            assert f["width"] == "5"
            assert f["type"] == "d"

    def test_zero(self):
        f = extract_format("010d", {})
        assert f["zero"] is True
        assert f["width"] == "10"
        f = extract_format("0d", {})
        assert f["zero"] is True
        assert f["width"] == ""

    def test_grouping(self):
        assert extract_format(",d", {})["grouping"] == ","
        assert extract_format("_d", {})["grouping"] == "_"
        assert "grouping" not in extract_format("d", {})

    def test_precision(self):
        f = extract_format("10.3f", {})
        assert f["width"] == "10"
        assert f["precision"] == "3"
        assert f["type"] == "f"
        f = extract_format(".f", {})
        assert f["precision"] == ""
        assert "precision" not in extract_format("f", {})

    def test_full(self):
        f = extract_format("x^+010,.2f", {})
        assert f["fill"] == "x"
        assert f["align"] == "^"
        assert f["zero"] is True
        assert f["width"] == "10"
        assert f["grouping"] == ","
        assert f["precision"] == "2"
        assert f["type"] == "f"

    def test_bad_type(self):
        with pytest.raises(ValueError):
            extract_format("q", {})
        with pytest.raises(ValueError):
            Parser("{:q}")
        with pytest.raises(ValueError):
            parse.parse("{:zz}", "a")

    def test_extra_type_allowed(self):
        f = extract_format("custom", {"custom": str})
        assert f["type"] == "custom"

    def test_datetime_type_allowed(self):
        assert extract_format("%Y-%m-%d", {})["type"] == "%Y-%m-%d"

    def test_all_allowed_types(self):
        for t in "nboxX%fFeEgGwWdDsSl":
            assert extract_format(t, {})["type"] == t
        for t in ["ti", "te", "ta", "th", "tg", "tc", "tt", "ts"]:
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


# ---------------------------------------------------------------------------
# Datetime types
# ---------------------------------------------------------------------------


class TestDatetimeISO:
    def test_date_only(self):
        assert parse.parse("{:ti}", "2011-02-03")[0] == datetime(2011, 2, 3)

    def test_with_time(self):
        assert parse.parse("{:ti}", "2011-02-03 04:05")[0] == datetime(2011, 2, 3, 4, 5)
        assert parse.parse("{:ti}", "2011-02-03T04:05:06")[0] == datetime(
            2011, 2, 3, 4, 5, 6
        )

    def test_microseconds(self):
        r = parse.parse("{:ti}", "2011-02-03T04:05:06.123")[0]
        assert r == datetime(2011, 2, 3, 4, 5, 6, 123000)
        r = parse.parse("{:ti}", "2011-02-03T04:05:06.1234567")[0]
        assert r.microsecond == 123456

    def test_z(self):
        r = parse.parse("{:ti}", "2011-02-03T04:05:06Z")[0]
        assert r == datetime(2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(0, "UTC"))
        assert r.utcoffset() == timedelta(0)
        assert r.tzname() == "UTC"

    def test_offsets(self):
        r = parse.parse("{:ti}", "2011-02-03T04:05:06+01:30")[0]
        assert r.utcoffset() == timedelta(minutes=90)
        r = parse.parse("{:ti}", "2011-02-03T04:05:06-0130")[0]
        assert r.utcoffset() == timedelta(minutes=-90)
        r = parse.parse("{:ti}", "2011-02-03 04:05 +0000")[0]
        assert r.utcoffset() == timedelta(0)

    def test_followed_by_field(self):
        r = parse.parse("{:ti} {}", "2011-02-03T04:05:06Z end")
        assert r.fixed[1] == "end"

    def test_reject(self):
        assert parse.parse("{:ti}", "2011/02/03") is None


class TestDatetimeOther:
    def test_te_email(self):
        r = parse.parse("{:te}", "Mon, 20 Jan 2016 17:51:05 -0500")[0]
        assert r == datetime(
            2016, 1, 20, 17, 51, 5, tzinfo=FixedTzOffset(-300, "-0500")
        )
        r = parse.parse("{:te}", "20 Jan 2016 17:51:05 +0100")[0]
        assert r.utcoffset() == timedelta(hours=1)
        assert r.day == 20
        assert r.month == 1

    def test_te_followed(self):
        r = parse.parse("{:te} {}", "20 Jan 2016 17:51:05 +0100 end")
        assert r.fixed[1] == "end"

    def test_tg_global(self):
        assert parse.parse("{:tg}", "20/1/2016")[0] == datetime(2016, 1, 20)
        assert parse.parse("{:tg}", "20-Jan-2016")[0] == datetime(2016, 1, 20)
        assert parse.parse("{:tg}", "20/January/2016")[0] == datetime(2016, 1, 20)
        assert parse.parse("{:tg}", "20/1/2016 10:11")[0] == datetime(
            2016, 1, 20, 10, 11
        )
        assert parse.parse("{:tg}", "20/1/2016 10:11:12")[0] == datetime(
            2016, 1, 20, 10, 11, 12
        )

    def test_tg_am_pm(self):
        assert parse.parse("{:tg}", "20/1/2016 10:11 PM")[0] == datetime(
            2016, 1, 20, 22, 11
        )
        assert parse.parse("{:tg}", "20/1/2016 10:11 AM")[0] == datetime(
            2016, 1, 20, 10, 11
        )
        assert parse.parse("{:tg}", "20/1/2016 12:11 AM")[0] == datetime(
            2016, 1, 20, 0, 11
        )
        assert parse.parse("{:tg}", "20/1/2016 12:11 PM")[0] == datetime(
            2016, 1, 20, 12, 11
        )

    def test_tg_tz(self):
        r = parse.parse("{:tg}", "20/1/2016 10:11 +05:00")[0]
        assert r.utcoffset() == timedelta(hours=5)
        r = parse.parse("{:tg}", "20/1/2016 10:11 PM -0230")[0]
        assert r.hour == 22
        assert r.utcoffset() == timedelta(hours=-2, minutes=-30)

    def test_tg_followed(self):
        r = parse.parse("{:tg} {}", "20/1/2016 10:11 PM +0100 end")
        assert r.fixed[1] == "end"
        assert r.fixed[0].hour == 22

    def test_ta_american(self):
        assert parse.parse("{:ta}", "1/20/2016")[0] == datetime(2016, 1, 20)
        assert parse.parse("{:ta}", "Jan-20-2016")[0] == datetime(2016, 1, 20)
        assert parse.parse("{:ta}", "February/3/2016")[0] == datetime(2016, 2, 3)
        assert parse.parse("{:ta}", "1/20/2016 10:11 PM")[0] == datetime(
            2016, 1, 20, 22, 11
        )
        r = parse.parse("{:ta}", "1/20/2016 10:11:12.5 +01:00")[0]
        assert r == datetime(
            2016, 1, 20, 10, 11, 12, 500000, tzinfo=FixedTzOffset(60, "+01:00")
        )

    def test_ta_followed(self):
        r = parse.parse("{:ta} {}", "1/20/2016 10:11 PM end")
        assert r.fixed[1] == "end"

    def test_th_http(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
        assert r == datetime(
            2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000")
        )
        r = parse.parse("{:th}", "21/Nov/2011:00:07:11 -0100")[0]
        assert r.utcoffset() == timedelta(hours=-1)

    def test_th_followed(self):
        r = parse.parse("[{:th}] {}", "[21/Nov/2011:10:21:36 +1000] end")
        assert r.fixed[1] == "end"

    def test_tc_ctime(self):
        r = parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0]
        assert r == datetime(1973, 9, 16, 1, 3, 52)
        r = parse.parse("{:tc}", "Mon Dec  5 10:00:00 2005")[0]
        assert r == datetime(2005, 12, 5, 10)

    def test_tc_followed(self):
        r = parse.parse("{:tc} {}", "Sun Sep 16 01:03:52 1973 end")
        assert r.fixed[1] == "end"

    def test_tt_time(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
        assert parse.parse("{:tt}", "10:21:36.25")[0] == time(10, 21, 36, 250000)
        assert parse.parse("{:tt}", "10:21 PM")[0] == time(22, 21)
        assert parse.parse("{:tt}", "12:21 AM")[0] == time(0, 21)
        assert parse.parse("{:tt}", "12:21 PM")[0] == time(12, 21)
        assert parse.parse("{:tt}", "1:21 am")[0] == time(1, 21)

    def test_tt_tz(self):
        r = parse.parse("{:tt}", "10:21:36 +0530")[0]
        assert r == time(10, 21, 36, tzinfo=FixedTzOffset(330, "+0530"))
        assert r.utcoffset() == timedelta(hours=5, minutes=30)
        r = parse.parse("{:tt}", "10:21 PM -05:00")[0]
        assert r.hour == 22
        assert r.utcoffset() == timedelta(hours=-5)

    def test_tt_short_tz(self):
        # 'snnn' style offsets: one hour digit + two minute digits
        r = parse.parse("{:tt}", "10:21 +530")[0]
        assert r.utcoffset() == timedelta(hours=5, minutes=30)
        r = parse.parse("{:tt}", "10:21 -100")[0]
        assert r.utcoffset() == timedelta(hours=-1)

    def test_tt_followed(self):
        r = parse.parse("{:tt} {}", "10:21 PM +0100 end")
        assert r.fixed[1] == "end"
        assert r.fixed[0].hour == 22

    def test_ts_syslog(self):
        r = parse.parse("{:ts}", "Nov 21 10:21:36")[0]
        assert r == datetime(THIS_YEAR, 11, 21, 10, 21, 36)
        r = parse.parse("{:ts}", "Jan  5 01:02:03")[0]
        assert r == datetime(THIS_YEAR, 1, 5, 1, 2, 3)

    def test_ts_followed(self):
        r = parse.parse("{:ts} {}", "Nov 21 10:21:36 end")
        assert r.fixed[1] == "end"

    def test_datetime_named(self):
        r = parse.parse("{when:ti}", "2011-02-03")
        assert r["when"] == datetime(2011, 2, 3)


class TestDateConvertDirect:
    def _match(self, pattern, text):
        return re.match(pattern, text)

    def test_time_only(self):
        m = self._match(r"(\d+:\d+)", "10:20")
        assert parse.date_convert("10:20", m, hms=0) == time(10, 20)

    def test_ymd(self):
        m = self._match(r"(\d+-\d+-\d+)", "2001-02-03")
        assert parse.date_convert("", m, ymd=0) == datetime(2001, 2, 3)

    def test_mdy_month_name(self):
        m = self._match(r"(\w+/\d+/\d+)", "Feb/03/2001")
        assert parse.date_convert("", m, mdy=0) == datetime(2001, 2, 3)

    def test_dmy(self):
        m = self._match(r"(\d+ \w+ \d+)", "03 March 2001")
        assert parse.date_convert("", m, dmy=0) == datetime(2001, 3, 3)

    def test_d_m_y(self):
        m = self._match(r"(\d+) (\w+) (\d+)", "3 Apr 2001")
        assert parse.date_convert("", m, d_m_y=(0, 1, 2)) == datetime(2001, 4, 3)

    def test_mm_dd(self):
        m = self._match(r"(x)(\w+) (\d+)", "xDec 25")
        r = parse.date_convert("", m, mm=1, dd=2)
        assert r == datetime(THIS_YEAR, 12, 25)

    def test_am_none_group(self):
        m = self._match(r"(\d+:\d+)( [AP]M)?", "10:20")
        assert parse.date_convert("", m, hms=0, am=1) == time(10, 20)

    def test_hms_missing_group(self):
        m = self._match(r"(\d+-\d+-\d+)( \d+:\d+)?", "2001-02-03")
        assert parse.date_convert("", m, ymd=0, hms=1) == datetime(2001, 2, 3)

    def test_tz_uppercase_name_passthrough(self):
        # uppercase timezone names aren't resolved; tz stays a string, which
        # datetime rejects
        m = self._match(r"(\d+:\d+) (\w+)", "10:20 EST")
        with pytest.raises(TypeError):
            parse.date_convert("", m, hms=0, tz=1)

    def test_tz_long_no_colon(self):
        m = self._match(r"(\d+:\d+) (\S+)", "10:20 +1145")
        r = parse.date_convert("", m, hms=0, tz=1)
        assert r.utcoffset() == timedelta(hours=11, minutes=45)

    def test_all_months(self):
        for name, num in parse.MONTHS_MAP.items():
            m = self._match(r"(\d+) (\w+) (\d+)", "1 %s 2000" % name)
            assert parse.date_convert("", m, d_m_y=(0, 1, 2)).month == num

    def test_months_map(self):
        assert len(parse.MONTHS_MAP) == 23  # May has no abbreviation
        assert parse.MONTHS_MAP["May"] == 5
        assert parse.MONTHS_MAP["Sep"] == 9
        assert parse.MONTHS_MAP["December"] == 12


class TestStrftimeFormats:
    def test_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-11-04")[0]
        assert r == date(2023, 11, 4)
        assert type(r) is date

    def test_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-11-04 13:14:15")[0]
        assert r == datetime(2023, 11, 4, 13, 14, 15)
        assert type(r) is datetime

    def test_time(self):
        r = parse.parse("{:%H:%M}", "13:14")[0]
        assert r == time(13, 14)
        assert type(r) is time

    def test_time_with_microseconds(self):
        r = parse.parse("{:%H:%M:%S.%f}", "13:14:15.5")[0]
        assert r == time(13, 14, 15, 500000)

    def test_12hour(self):
        r = parse.parse("{:%I:%M %p}", "01:30 PM")[0]
        assert r == time(13, 30)

    def test_tz(self):
        r = parse.parse("{:%H:%M %z}", "13:14 +0100")[0]
        assert r.utcoffset() == timedelta(hours=1)

    def test_names(self):
        r = parse.parse("{:%A, %B %d, %Y}", "Saturday, November 04, 2023")[0]
        assert r == date(2023, 11, 4)
        r = parse.parse("{:%a %b %d %Y}", "Sat Nov 04 2023")[0]
        assert r == date(2023, 11, 4)

    def test_two_digit_year(self):
        assert parse.parse("{:%d/%m/%y}", "04/11/23")[0] == date(2023, 11, 4)

    def test_no_year_uses_current(self):
        r = parse.parse("{:%m-%d}", "11-04")[0]
        assert r == date(THIS_YEAR, 11, 4)

    def test_day_of_year(self):
        r = parse.parse("{:%Y %j}", "2023 032")[0]
        assert r == date(2023, 2, 1)

    def test_weekday_number(self):
        r = parse.parse("{:%Y %U %w}", "2023 01 1")[0]
        assert isinstance(r, date)

    def test_reject(self):
        assert parse.parse("{:%Y-%m-%d}", "23-11-04") is None
        assert parse.parse("{:%H:%M}", "13:4") is None

    def test_followed(self):
        r = parse.parse("{:%Y-%m-%d} {}", "2023-11-04 end")
        assert r.fixed == (date(2023, 11, 4), "end")

    def test_strf_date_convert_neither(self):
        with pytest.raises(ValueError):
            parse.strf_date_convert("x", None, type="x")

    def test_strf_date_convert_direct(self):
        assert parse.strf_date_convert("2020", None, type="%Y") == date(2020, 1, 1)
        assert parse.strf_date_convert("05", None, type="%M") == time(0, 5)

    def test_regex_for_format(self):
        assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
        assert parse.get_regex_for_datetime_format("abc") == "abc"


# ---------------------------------------------------------------------------
# FixedTzOffset
# ---------------------------------------------------------------------------


class TestFixedTzOffset:
    def test_basic(self):
        tz = FixedTzOffset(90, "+01:30")
        assert tz.utcoffset(None) == timedelta(minutes=90)
        assert tz.tzname(None) == "+01:30"
        assert tz.dst(None) == timedelta(0)

    def test_repr(self):
        tz = FixedTzOffset(60, "X")
        assert repr(tz) == "<FixedTzOffset X 1:00:00>"

    def test_eq(self):
        assert FixedTzOffset(60, "a") == FixedTzOffset(60, "a")
        assert FixedTzOffset(60, "a") != FixedTzOffset(60, "b")
        assert FixedTzOffset(60, "a") != FixedTzOffset(30, "a")
        assert FixedTzOffset(60, "a").__eq__(5) is NotImplemented
        assert FixedTzOffset(60, "a") != 5

    def test_negative(self):
        assert FixedTzOffset(-30, "n").utcoffset(None) == timedelta(minutes=-30)


# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------


class TestCustomTypes:
    def test_plain_function(self):
        def shouty(s):
            return s.upper()

        r = parse.parse("{:shouty}", "hello", extra_types={"shouty": shouty})
        assert r[0] == "HELLO"

    def test_with_pattern(self):
        @with_pattern(r"\d+")
        def number(text):
            return int(text)

        assert number.pattern == r"\d+"
        assert number.regex_group_count is None
        r = parse.parse("{:num}{:w}", "123abc", extra_types={"num": number})
        assert r.fixed == (123, "abc")
        assert parse.parse("{:num}", "abc", extra_types={"num": number}) is None

    def test_with_pattern_returns_same_function(self):
        def f(s):
            return s

        assert with_pattern("x")(f) is f

    def test_with_pattern_groups(self):
        @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
        def rng(text):
            a, b = text.split("-")
            return (int(a), int(b))

        assert rng.regex_group_count == 3
        r = parse.parse(
            "{:rng} {} {name}", "1-5 x y", extra_types={"rng": rng}
        )
        assert r.fixed == ((1, 5), "x")
        assert r["name"] == "y"

    def test_custom_type_named(self):
        @with_pattern(r"[ab]+")
        def ab(text):
            return len(text)

        r = parse.parse("{x:ab}", "abab", extra_types={"ab": ab})
        assert r["x"] == 4

    def test_custom_overrides_builtin(self):
        def myint(s):
            return "custom:" + s

        r = parse.parse("{:d}", "12", extra_types={"d": myint})
        assert r[0] == "custom:12"

    def test_custom_type_in_compile(self):
        p = compile("{:yesno}", extra_types={"yesno": with_pattern("yes|no")(lambda s: s == "yes")})
        assert p.parse("yes")[0] is True
        assert p.parse("no")[0] is False
        assert p.parse("maybe") is None

    def test_convert_first_pickle(self):
        c = parse.convert_first(int)
        assert c("5", None) == 5
        c2 = pickle.loads(pickle.dumps(c))
        assert c2("7", None) == 7


# ---------------------------------------------------------------------------
# search / findall
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search(self):
        r = search("age: {:d}", "name: x, age: 42, end")
        assert r[0] == 42

    def test_search_none(self):
        assert search("age: {:d}", "nothing") is None

    def test_search_pos(self):
        r = search("{:d}", "1 2 3", pos=1)
        assert r[0] == 2
        r = search("{:d}", "1 2 3", 3)
        assert r[0] == 3

    def test_search_endpos(self):
        assert search("{:d}", "abc 123", endpos=4) is None
        assert search("{:d}", "abc 123", endpos=5)[0] == 1

    def test_search_evaluate_false(self):
        m = search("{:d}", "x 12", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert r[0] == 12

    def test_parser_search(self):
        p = compile("{:d}")
        assert p.search("a 7 b")[0] == 7
        assert p.search("a 7 b 8", pos=4)[0] == 8
        assert p.search("a 7 b 8", 0, 2) is None
        assert p.search("none") is None
        assert isinstance(p.search("a 7", evaluate_result=False), Match)


class TestFindall:
    def test_findall(self):
        res = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
        assert [r[0] for r in res] == ["some ", "bold", " text"]

    def test_findall_typed(self):
        assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]

    def test_findall_iterator(self):
        it = findall("{:d}", "1 2")
        assert isinstance(it, ResultIterator)
        assert iter(it) is it
        assert next(it)[0] == 1
        assert it.next()[0] == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_findall_pos_endpos(self):
        assert [r[0] for r in findall("{:d}", "1 2 3 4", pos=2)] == [2, 3, 4]
        assert [r[0] for r in findall("{:d}", "1 2 3 4", endpos=3)] == [1, 2]
        assert [r[0] for r in findall("{:d}", "1 2 3 4", 2, 5)] == [2, 3]

    def test_findall_evaluate_false(self):
        res = list(findall("{:d}", "1 2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in res)
        assert [m.evaluate_result()[0] for m in res] == [1, 2]

    def test_findall_none(self):
        assert list(findall("{:d}", "abc")) == []

    def test_parser_findall(self):
        p = compile("{:w}")
        assert [r[0] for r in p.findall("ab cd ef")] == ["ab", "cd", "ef"]
        assert [r[0] for r in p.findall("ab cd ef", 3)] == ["cd", "ef"]
        assert [r[0] for r in p.findall("ab cd ef", 0, 5)] == ["ab", "cd"]
        res = list(p.findall("ab", evaluate_result=False))
        assert isinstance(res[0], Match)

    def test_findall_named(self):
        res = list(findall("{k}={v:d};", "a=1;b=2;"))
        assert [(r["k"], r["v"]) for r in res] == [("a", 1), ("b", 2)]


# ---------------------------------------------------------------------------
# Parser / Match / Result objects
# ---------------------------------------------------------------------------


class TestObjects:
    def test_compile_returns_parser(self):
        p = compile("{:d}")
        assert isinstance(p, Parser)
        assert p.format == "{:d}"
        assert p.parse("12")[0] == 12
        assert p.parse("x") is None

    def test_parser_reuse(self):
        p = compile("{a:d}-{b:d}")
        assert p.parse("1-2").named == {"a": 1, "b": 2}
        assert p.parse("3-4").named == {"a": 3, "b": 4}

    def test_parse_evaluate_false(self):
        m = parse.parse("{:d}", "12", evaluate_result=False)
        assert isinstance(m, Match)
        assert isinstance(m.parser, Parser)
        assert isinstance(m.match, re.Match)
        assert m.evaluate_result()[0] == 12
        assert parse.parse("{:d}", "x", evaluate_result=False) is None
        p = compile("{:d}")
        m = p.parse("3", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.parser is p
        assert p.evaluate_result(m.match)[0] == 3

    def test_repr_parser(self):
        assert repr(compile("{}")) == "<Parser '{}'>"
        assert repr(compile("a" * 20)) == "<Parser %r>" % ("a" * 20)
        long = "abcdefghijklmnopqrstuvwxyz"
        assert repr(compile(long)) == "<Parser %r>" % (long[:17] + "...")

    def test_repr_result(self):
        r = parse.parse("{} {n}", "a b")
        assert repr(r) == "<Result ('a',) {'n': 'b'}>"

    def test_result_constructor(self):
        r = Result((1, 2), {"a": 3}, {0: (0, 1)})
        assert r.fixed == (1, 2)
        assert r.named == {"a": 3}
        assert r.spans == {0: (0, 1)}
        assert r[0] == 1
        assert r[0:2] == (1, 2)
        assert r["a"] == 3
        assert "a" in r
        assert 0 not in r
        with pytest.raises(KeyError):
            r["b"]
        with pytest.raises(IndexError):
            r[5]

    def test_too_many_fields(self):
        fmt = "".join("{:ti}" for _ in range(15))
        p = compile(fmt)
        # Python's re no longer has the 100-group limit; ensure either it
        # works or raises the documented TooManyFields error
        try:
            p.parse("x")
        except TooManyFields:
            pass

    def test_expression_attribute(self):
        p = compile("a{}b")
        assert p._expression == "a(.+?)b"
        p = compile("{name}")
        assert p._expression == "(?P<name>.+?)"
        p = compile("{{}}")
        assert p._expression == r"\{\}"

    def test_re_flags(self):
        assert compile("x")._re_flags == re.IGNORECASE | re.DOTALL
        assert compile("x", case_sensitive=True)._re_flags == re.DOTALL

    def test_extra_types_default(self):
        assert compile("x")._extra_types == {}

    def test_regex_compiled_lazily_and_cached(self):
        p = compile("{:d}")
        assert p._match_re is p._match_re
        assert p._search_re is p._search_re

    def test_all(self):
        assert parse.__all__ == ["parse", "search", "findall", "with_pattern"]


# ---------------------------------------------------------------------------
# PARSE_RE field detection
# ---------------------------------------------------------------------------


class TestFieldDetection:
    def test_parse_re_split(self):
        parts = [p for p in parse.PARSE_RE.split("a{b}c{{d}}{e.f[g]:>3}") if p]
        assert parts == ["a", "{b}", "c", "{{", "d", "}}", "{e.f[g]:>3}"]

    def test_braces_with_spaces_not_field(self):
        # "{a b}" is not a valid field and is matched literally
        assert parse.parse("x{a b}", "x{a b}") is not None
        assert parse.parse("x{a b}", "xa b") is None

    def test_bad_group_name_not_implemented(self):
        with pytest.raises(NotImplementedError):
            parse.parse("{a b}", "x")

    def test_format_spec_with_colon(self):
        r = parse.parse("{:%H:%M}", "10:30")
        assert r[0] == time(10, 30)


# ---------------------------------------------------------------------------
# Round-trip with str.format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, values",
    [
        ("{:d}", (42,)),
        ("{:d}", (-42,)),
        ("{:5d}", (42,)),
        ("{:<5d}|", (42,)),
        ("{:^7d}", (42,)),
        ("{:05d}", (42,)),
        ("{:,d}", (1234567,)),
        ("{:b}", (10,)),
        ("{:o}", (10,)),
        ("{:x}", (255,)),
        ("{:X}", (255,)),
        ("{:.2f}", (3.14,)),
        ("{:10.3f}", (-2.5,)),
        ("{:.0f}", (20.0,)),
        ("{:e}", (12345.678,)),
        ("{:E}", (0.001,)),
        ("{:g}", (1.5,)),
        ("{:g}", (1e20,)),
        ("{:%}", (0.25,)),
        ("{:.1%}", (0.125,)),
        ("{} and {}", ("cats", "dogs")),
        ("{:>}|{:<}|{:^}", ("a", "b", "c")),
        ("{:*^}", ("mid",)),
        ("{}-{:d}", ("abc", 3)),
    ],
)
def test_roundtrip(fmt, values):
    text = fmt.format(*values)
    r = parse.parse(fmt, text)
    assert r is not None, (fmt, text)
    for got, want in zip(r.fixed, values):
        if isinstance(want, float):
            assert got == pytest.approx(want, rel=1e-3)
        else:
            assert got == want
