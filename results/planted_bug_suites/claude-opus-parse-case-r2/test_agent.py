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
    compile,
    extract_format,
    findall,
    int_convert,
    search,
    with_pattern,
)


def utc_offset(minutes):
    return timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------


class TestBasicParse:
    def test_no_fields_exact_match(self):
        r = parse.parse("hello", "hello")
        assert r is not None
        assert r.fixed == ()
        assert r.named == {}

    def test_no_fields_mismatch(self):
        assert parse.parse("hello", "goodbye") is None

    def test_parse_requires_full_match(self):
        assert parse.parse("hello {}", "hello world!") is not None
        assert parse.parse("hello", "hello world") is None
        assert parse.parse("hello", "say hello") is None

    def test_trailing_newline_not_matched(self):
        # \Z rather than $ – a trailing newline must not be accepted
        assert parse.parse("hello", "hello\n") is None

    def test_single_fixed_field(self):
        r = parse.parse("hello {}", "hello world")
        assert r.fixed == ("world",)
        assert r[0] == "world"

    def test_multiple_fixed_fields(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r.fixed == ("a", "b", "c")

    def test_fixed_slicing(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r[1:] == ("b", "c")
        assert r[-1] == "c"

    def test_named_field(self):
        r = parse.parse("hello {name}", "hello world")
        assert r.named == {"name": "world"}
        assert r["name"] == "world"
        assert "name" in r
        assert "other" not in r

    def test_mixed_fixed_and_named(self):
        r = parse.parse("{} {name} {}", "a b c")
        assert r.fixed == ("a", "c")
        assert r.named == {"name": "b"}

    def test_numeric_positional_field_treated_as_fixed(self):
        r = parse.parse("{0} {1}", "x y")
        assert r.fixed == ("x", "y")
        assert r.named == {}

    def test_untyped_field_matches_nonempty(self):
        assert parse.parse("a{}b", "ab") is None
        assert parse.parse("a{}b", "a b")[0] == " "

    def test_untyped_field_is_lazy(self):
        r = parse.parse("{}-{}", "a-b-c")
        assert r.fixed == ("a", "b-c")

    def test_multiline_dotall(self):
        r = parse.parse("{} end", "line1\nline2 end")
        assert r[0] == "line1\nline2"

    def test_missing_named_key_raises(self):
        r = parse.parse("{a}", "x")
        with pytest.raises(KeyError):
            r["b"]

    def test_missing_index_raises(self):
        r = parse.parse("{}", "x")
        with pytest.raises(IndexError):
            r[1]


class TestEscaping:
    def test_double_braces(self):
        r = parse.parse("{{hello}} {}", "{hello} world")
        assert r.fixed == ("world",)

    def test_double_braces_only(self):
        assert parse.parse("{{}}", "{}") is not None
        assert parse.parse("{{}}", "x") is None

    @pytest.mark.parametrize(
        "literal",
        ["a.b", "a?", "a\\b", "[x]", "(x)", "a*", "a+", "^a", "a$", "a|b", "a!"],
    )
    def test_regex_metacharacters_are_literal(self, literal):
        fmt = literal + " {}"
        r = parse.parse(fmt, literal + " val")
        assert r is not None
        assert r[0] == "val"

    def test_dot_not_wildcard(self):
        assert parse.parse("a.b {}", "axb x") is None

    def test_pipe_not_alternation(self):
        assert parse.parse("a|b", "a") is None
        assert parse.parse("a|b", "b") is None


# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------


class TestCaseSensitivity:
    def test_parse_case_insensitive_by_default(self):
        # documented: "The default behaviour is to match strings case insensitively"
        r = parse.parse("Hello {}", "hELLO world")
        assert r is not None
        assert r[0] == "world"

    def test_parse_case_sensitive_explicit(self):
        assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
        assert parse.parse("Hello {}", "Hello world", case_sensitive=True)[0] == "world"

    def test_parse_case_insensitive_explicit(self):
        assert parse.parse("Hello {}", "HELLO x", case_sensitive=False)[0] == "x"

    def test_parse_default_matches_compile_default(self):
        for s in ["HELLO x", "hello x", "Hello x"]:
            a = parse.parse("hello {}", s)
            b = compile("hello {}").parse(s)
            assert (a is None) == (b is None)

    def test_search_case_insensitive_by_default(self):
        assert search("Hello {:d}", "xx HELLO 42")[0] == 42

    def test_search_case_sensitive(self):
        assert search("Hello {:d}", "xx HELLO 42", case_sensitive=True) is None
        assert search("Hello {:d}", "xx Hello 42", case_sensitive=True)[0] == 42

    def test_findall_case_insensitive_by_default(self):
        res = [r[0] for r in findall("x{:d}", "X1 x2 X3")]
        assert res == [1, 2, 3]

    def test_findall_case_sensitive(self):
        res = [r[0] for r in findall("x{:d}", "X1 x2 X3", case_sensitive=True)]
        assert res == [2]

    def test_compile_case_default_and_explicit(self):
        assert compile("AB{}").parse("ab1")[0] == "1"
        assert compile("AB{}", case_sensitive=True).parse("ab1") is None

    def test_parser_class_default(self):
        assert Parser("AB{}").parse("ab1")[0] == "1"
        assert Parser("AB{}", case_sensitive=True).parse("ab1") is None


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0", 0),
            ("42", 42),
            ("-42", -42),
            ("+42", 42),
            (" 42", 42),
            ("0x1F", 31),
            ("0X1f", 31),
            ("0b101", 5),
            ("0B101", 5),
            ("0o17", 15),
            ("0O17", 15),
            ("-0x10", -16),
            ("007", 7),
        ],
    )
    def test_d(self, text, expected):
        r = parse.parse("{:d}", text)
        assert r is not None
        assert r[0] == expected
        assert type(r[0]) is int

    def test_d_rejects_non_number(self):
        assert parse.parse("{:d}", "abc") is None
        assert parse.parse("{:d}", "1.5") is None
        assert parse.parse("{:d}", "") is None

    def test_d_width_limits_digits(self):
        r = parse.parse("{:2d}{:2d}", "1234")
        assert r.fixed == (12, 34)

    def test_d_width_with_padding(self):
        r = parse.parse("[{:4d}]", "[  42]")
        assert r[0] == 42

    def test_d_grouping_comma(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567

    def test_d_grouping_underscore(self):
        assert parse.parse("{:_d}", "1_000")[0] == 1000

    def test_d_no_grouping_rejects_comma(self):
        assert parse.parse("{:d}", "1,000") is None

    def test_d_zero_padded(self):
        assert parse.parse("{:03d}", "007")[0] == 7

    def test_n_with_thousands(self):
        assert parse.parse("{:n}", "1,234,567")[0] == 1234567
        assert parse.parse("{:n}", "1.234.567")[0] == 1234567
        assert parse.parse("{:n}", "999")[0] == 999
        assert parse.parse("{:n}", "-1,000")[0] == -1000

    def test_n_rejects_bad_grouping(self):
        assert parse.parse("{:n}", "12,34") is None

    @pytest.mark.parametrize(
        "text,expected",
        [("1010", 10), ("0b1010", 10), ("0B11", 3), ("-101", -5)],
    )
    def test_b(self, text, expected):
        assert parse.parse("{:b}", text)[0] == expected

    def test_b_rejects_non_binary(self):
        assert parse.parse("{:b}", "102") is None

    @pytest.mark.parametrize(
        "text,expected",
        [("17", 15), ("0o17", 15), ("0O777", 511), ("-7", -7)],
    )
    def test_o(self, text, expected):
        assert parse.parse("{:o}", text)[0] == expected

    def test_o_rejects_8(self):
        assert parse.parse("{:o}", "8") is None

    @pytest.mark.parametrize(
        "fmt,text,expected",
        [
            ("{:x}", "ff", 255),
            ("{:x}", "FF", 255),
            ("{:x}", "0xff", 255),
            ("{:X}", "0XAB", 171),
            ("{:x}", "-1a", -26),
            ("{:x}", "10", 16),
        ],
    )
    def test_x(self, fmt, text, expected):
        assert parse.parse(fmt, text)[0] == expected

    def test_x_rejects_g(self):
        assert parse.parse("{:x}", "fg") is None

    def test_numeric_equals_align_fill(self):
        assert parse.parse("{:=5d}", "-0042")[0] == -42
        assert parse.parse("{:x=5d}", "xxx42")[0] == 42


class TestIntConvert:
    def test_auto_base_detection(self):
        c = int_convert()
        assert c("0x10", None) == 16
        assert c("0b10", None) == 2
        assert c("0o10", None) == 8
        assert c("10", None) == 10
        assert c("-0x10", None) == -16
        assert c("+0b11", None) == 3

    def test_short_strings_default_decimal(self):
        c = int_convert()
        assert c("0", None) == 0
        assert c("00", None) == 0
        assert c("-5", None) == -5

    def test_explicit_base_ignores_prefix_chars(self):
        assert int_convert(16)("0xff", None) == 255
        assert int_convert(2)("0b11", None) == 3
        assert int_convert(8)("0o17", None) == 15
        assert int_convert(10)("1,234", None) == 1234

    def test_strips_garbage(self):
        assert int_convert(10)("1_000_000", None) == 1000000

    def test_uppercase_handled(self):
        assert int_convert(16)("FF", None) == 255
        assert int_convert()("0XFF", None) == 255


# ---------------------------------------------------------------------------
# Floats / decimals / percentages
# ---------------------------------------------------------------------------


class TestFloats:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("3.14", 3.14),
            (".5", 0.5),
            ("-3.14", -3.14),
            ("+2.5", 2.5),
            ("0.0", 0.0),
        ],
    )
    def test_f(self, text, expected):
        r = parse.parse("{:f}", text)
        assert r[0] == expected
        assert type(r[0]) is float

    def test_f_requires_decimal_point(self):
        assert parse.parse("{:f}", "3") is None

    def test_f_nan_inf(self):
        v = parse.parse("{:f}", "nan")[0]
        assert v != v
        assert parse.parse("{:f}", "inf")[0] == float("inf")
        assert parse.parse("{:f}", "-inf")[0] == float("-inf")

    def test_f_precision_zero(self):
        r = parse.parse("{:.0f}", "20")
        assert r[0] == 20.0
        assert parse.parse("{:.0f}", "-20")[0] == -20.0

    def test_f_precision(self):
        assert parse.parse("{:.2f}", "3.14")[0] == 3.14

    def test_f_width(self):
        assert parse.parse("[{:8.2f}]", "[    3.14]")[0] == 3.14

    def test_F_returns_decimal(self):
        r = parse.parse("{:F}", "3.14")
        assert r[0] == Decimal("3.14")
        assert isinstance(r[0], Decimal)
        assert parse.parse("{:F}", "-1.5")[0] == Decimal("-1.5")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1.5e10", 1.5e10),
            ("1.5E10", 1.5e10),
            ("1.5e-3", 1.5e-3),
            ("-2.0e+2", -200.0),
            ("inf", float("inf")),
            ("-inf", float("-inf")),
        ],
    )
    def test_e(self, text, expected):
        assert parse.parse("{:e}", text)[0] == expected

    def test_e_requires_exponent(self):
        assert parse.parse("{:e}", "1.5") is None

    def test_e_nan(self):
        v = parse.parse("{:e}", "nan")[0]
        assert v != v

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1", 1.0),
            ("1.5", 1.5),
            ("1e10", 1e10),
            ("1.5E-2", 0.015),
            ("-3", -3.0),
            ("inf", float("inf")),
        ],
    )
    def test_g(self, text, expected):
        r = parse.parse("{:g}", text)
        assert r[0] == expected
        assert type(r[0]) is float

    def test_g_followed_by_other_fields(self):
        # g uses internal groups; subsequent fixed fields must be indexed correctly
        r = parse.parse("{:g} {} {:d}", "1.5e3 abc 7")
        assert r.fixed == (1500.0, "abc", 7)

    def test_g_nan(self):
        v = parse.parse("{:g}", "nan")[0]
        assert v != v

    def test_percentage(self):
        assert parse.parse("{:%}", "50%")[0] == 0.5
        assert parse.parse("{:%}", "12.5%")[0] == 0.125
        assert parse.parse("{:%}", "50") is None

    def test_percentage_function(self):
        assert parse.percentage("25%", None) == 0.25


# ---------------------------------------------------------------------------
# Strings, letters, width/precision/alignment
# ---------------------------------------------------------------------------


class TestStringTypes:
    def test_s_whitespace(self):
        assert parse.parse("a{:s}b", "a \t b")[0] == " \t "
        assert parse.parse("a{:s}b", "axb") is None

    def test_w_word(self):
        assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
        assert parse.parse("{:w}", "ab c") is None

    def test_W_non_word(self):
        assert parse.parse("{:W}", "!@#")[0] == "!@#"
        assert parse.parse("{:W}", "a!") is None

    def test_S_non_whitespace(self):
        assert parse.parse("{:S}", "a-b!")[0] == "a-b!"
        assert parse.parse("{:S}", "a b") is None

    def test_D_non_digit(self):
        assert parse.parse("{:D}", "abc")[0] == "abc"
        assert parse.parse("{:D}", "a1") is None

    def test_l_letters(self):
        assert parse.parse("{:l}", "Hello")[0] == "Hello"
        assert parse.parse("{:l}", "abc1") is None
        assert parse.parse("{:l}", "ab_c") is None

    def test_width_minimum(self):
        assert parse.parse("{:5}", "abc") is None
        assert parse.parse("{:3}", "abc")[0] == "abc"

    def test_precision_maximum(self):
        assert parse.parse("{:.2}", "abc") is None
        assert parse.parse("{:.3}", "abc")[0] == "abc"
        assert parse.parse("{:.2}{}", "abcd").fixed == ("a", "bcd")

    def test_width_and_precision(self):
        assert parse.parse("{:2.4}", "a") is None
        assert parse.parse("{:2.4}", "abcde") is None
        assert parse.parse("{:2.4}", "abc")[0] == "abc"

    def test_left_align(self):
        assert parse.parse("{:<}|", "abc    |")[0] == "abc"

    def test_right_align(self):
        assert parse.parse("|{:>}", "|    abc")[0] == "abc"

    def test_center_align(self):
        assert parse.parse("|{:^}|", "|   abc   |")[0] == "abc"

    def test_string_width_counts_padding(self):
        # for untyped strings the width is a minimum on the captured text
        assert parse.parse("|{:6}", "|   abc")[0] == "   abc"
        assert parse.parse("|{:7}", "|   abc") is None

    def test_width_implies_right_align(self):
        assert parse.parse("|{:6d}", "|    42")[0] == 42
        assert parse.parse("{:6d}|", "42    |") is None

    def test_custom_fill(self):
        assert parse.parse("{:x<}|", "abcxxx|")[0] == "abc"
        assert parse.parse("{:*>}", "***abc")[0] == "abc"
        assert parse.parse("{:.^}", "..abc..")[0] == "abc"

    @pytest.mark.parametrize("fill", list(".\\+?*[](){}^$"[:3]) + ["?", "*", "(", ")", "$"])
    def test_special_fill_chars_escaped(self, fill):
        fmt = "{:" + fill + "<}|"
        if fill in "{}":
            pytest.skip("braces cannot be used as fill in format")
        r = parse.parse(fmt, "abc" + fill * 3 + "|")
        assert r is not None
        assert r[0] == "abc"

    def test_number_with_alignment(self):
        assert parse.parse("{:>5d}", "   42")[0] == 42
        assert parse.parse("{:<5d}|", "42   |")[0] == 42
        assert parse.parse("{:^7d}|", "  42   |")[0] == 42

    def test_sign_in_spec(self):
        assert parse.parse("{:+d}", "+5")[0] == 5
        assert parse.parse("{: d}", " 5")[0] == 5
        assert parse.parse("{:-d}", "-5")[0] == -5


class TestInvalidFormats:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            parse.parse("{:q}", "x")

    def test_unknown_type_compile_raises(self):
        with pytest.raises(ValueError):
            compile("{:zz}")


# ---------------------------------------------------------------------------
# extract_format
# ---------------------------------------------------------------------------


class TestExtractFormat:
    def test_simple_type(self):
        f = extract_format("d", {})
        assert f["type"] == "d"
        assert f["align"] is None
        assert f["fill"] is None
        assert f["width"] == ""
        assert f["zero"] is False

    def test_align_only(self):
        f = extract_format("<", {})
        assert f["align"] == "<"
        assert f["fill"] is None
        assert f["type"] == ""

    def test_fill_and_align(self):
        f = extract_format("*^10s", {})
        assert f["fill"] == "*"
        assert f["align"] == "^"
        assert f["width"] == "10"
        assert f["type"] == "s"

    def test_zero_and_width(self):
        f = extract_format("010d", {})
        assert f["zero"] is True
        assert f["width"] == "10"
        assert f["type"] == "d"

    def test_precision(self):
        f = extract_format("8.3f", {})
        assert f["width"] == "8"
        assert f["precision"] == "3"
        assert f["type"] == "f"

    def test_grouping(self):
        assert extract_format(",d", {})["grouping"] == ","
        assert extract_format("_d", {})["grouping"] == "_"

    def test_sign_stripped(self):
        assert extract_format("+d", {})["type"] == "d"
        assert extract_format("-d", {})["type"] == "d"
        assert extract_format(" d", {})["type"] == "d"

    def test_extra_type_accepted(self):
        assert extract_format("custom", {"custom": lambda s: s})["type"] == "custom"

    def test_datetime_type_accepted(self):
        assert extract_format("%Y-%m-%d", {})["type"] == "%Y-%m-%d"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_format("abc", {})

    @pytest.mark.parametrize("t", sorted(parse.ALLOWED_TYPES))
    def test_all_allowed_types_accepted(self, t):
        assert extract_format(t, {})["type"] == t


# ---------------------------------------------------------------------------
# Named fields: dotted, indexed, repeated
# ---------------------------------------------------------------------------


class TestNamedFields:
    def test_dotted_name(self):
        r = parse.parse("{a.b}", "x")
        assert r.named == {"a.b": "x"}
        assert r["a.b"] == "x"

    def test_dotted_and_underscore_dont_collide(self):
        r = parse.parse("{a.b} {a_b}", "1 2")
        assert r.named == {"a.b": "1", "a_b": "2"}

    def test_underscore_collisions(self):
        r = parse.parse("{a_b} {a.b}", "1 2")
        assert r.named == {"a_b": "1", "a.b": "2"}

    def test_hyphenated_name(self):
        r = parse.parse("{a-b}", "x")
        assert r.named == {"a-b": "x"}

    def test_hyphen_and_underscore_collide(self):
        r = parse.parse("{a_b} {a-b}", "1 2")
        assert r.named == {"a_b": "1", "a-b": "2"}

    def test_indexed_names_become_nested_dict(self):
        r = parse.parse("{a[b]} {a[c]}", "1 2")
        assert r.named == {"a": {"b": "1", "c": "2"}}

    def test_deeply_indexed(self):
        r = parse.parse("{a[b][c]}", "x")
        assert r.named == {"a": {"b": {"c": "x"}}}

    def test_indexed_with_type(self):
        r = parse.parse("{p[x]:d},{p[y]:d}", "3,4")
        assert r.named == {"p": {"x": 3, "y": 4}}

    def test_repeated_name_must_match_same_value(self):
        r = parse.parse("{x} {x}", "a a")
        assert r.named == {"x": "a"}
        assert parse.parse("{x} {x}", "a b") is None

    def test_repeated_name_with_type(self):
        r = parse.parse("{x:d}-{x:d}", "5-5")
        assert r["x"] == 5
        assert parse.parse("{x:d}-{x:d}", "5-6") is None

    def test_repeated_name_type_mismatch(self):
        with pytest.raises(RepeatedNameError):
            compile("{x:d} {x:w}")
        with pytest.raises(RepeatedNameError):
            compile("{x} {x:d}")

    def test_repeated_name_error_is_value_error(self):
        assert issubclass(RepeatedNameError, ValueError)

    def test_named_fields_property(self):
        p = compile("{a} {} {b.c}")
        assert p.named_fields == ["a", "b_c"]
        # fixed_fields holds the regex group index of each positional field
        assert p.fixed_fields == [1]

    def test_named_fields_property_is_copy(self):
        p = compile("{a}")
        p.named_fields.append("zzz")
        p.fixed_fields.append(99)
        assert p.named_fields == ["a"]
        assert p.fixed_fields == []

    def test_format_property(self):
        assert compile("{a} b").format == "{a} b"


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class TestSpans:
    def test_fixed_spans(self):
        r = parse.parse("{} {}", "ab cde")
        assert r.spans == {0: (0, 2), 1: (3, 6)}

    def test_named_spans(self):
        r = parse.parse("x{a}y{b}", "x12y345")
        assert r.spans == {"a": (1, 3), "b": (4, 7)}

    def test_spans_after_multigroup_types(self):
        s = "1.5 hello 7"
        r = parse.parse("{:g} {} {:d}", s)
        assert r.spans[0] == (0, 3)
        assert r.spans[1] == (4, 9)
        assert r.spans[2] == (10, 11)
        for i, val in enumerate(["1.5", "hello", "7"]):
            a, b = r.spans[i]
            assert s[a:b] == val

    def test_search_spans_absolute(self):
        r = search("val={:d}", "foo val=42 bar")
        assert r.spans[0] == (8, 10)


# ---------------------------------------------------------------------------
# search / findall
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_substring(self):
        r = search("age: {:d}", "name: Bob, age: 32, x")
        assert r[0] == 32

    def test_search_none(self):
        assert search("age: {:d}", "nothing here") is None

    def test_search_pos(self):
        s = "a1 a2 a3"
        assert search("a{:d}", s)[0] == 1
        assert search("a{:d}", s, pos=1)[0] == 2
        assert search("a{:d}", s, 4)[0] == 3

    def test_search_endpos(self):
        assert search("a{:d}", "xx a12", endpos=4) is None
        assert search("a{:d}", "xx a12", endpos=5)[0] == 1
        assert search("a{:d}", "xx a12", 0, 6)[0] == 12

    def test_search_evaluate_result_false(self):
        m = search("a{:d}", "xa5", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert isinstance(r, Result)
        assert r[0] == 5

    def test_parser_search(self):
        p = compile("<{}>")
        assert p.search("xx <a> yy")[0] == "a"
        assert p.search("no") is None


class TestFindall:
    def test_findall_basic(self):
        res = [r[0] for r in findall(">{}<", "<p>the <b>bold</b> text</p>")]
        assert res == ["the ", "bold", " text"]

    def test_findall_ints(self):
        assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]

    def test_findall_named(self):
        res = [r["k"] for r in findall("{k:w}=", "a=1 b=2 c=3")]
        assert res == ["a", "b", "c"]

    def test_findall_none(self):
        assert list(findall("x{:d}", "nothing")) == []

    def test_findall_pos_endpos(self):
        s = "x1 x2 x3 x4"
        assert [r[0] for r in findall("x{:d}", s, pos=3)] == [2, 3, 4]
        assert [r[0] for r in findall("x{:d}", s, endpos=5)] == [1, 2]
        assert [r[0] for r in findall("x{:d}", s, 3, 8)] == [2, 3]

    def test_findall_evaluate_result_false(self):
        res = list(findall("x{:d}", "x1 x2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in res)
        assert [m.evaluate_result()[0] for m in res] == [1, 2]

    def test_findall_is_iterator(self):
        it = findall("{:d}", "1 2")
        assert iter(it) is it
        assert next(it)[0] == 1
        assert it.next()[0] == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_parser_findall(self):
        p = compile("[{:d}]")
        assert [r[0] for r in p.findall("[1][2] [3]")] == [1, 2, 3]


# ---------------------------------------------------------------------------
# evaluate_result / Match / Result
# ---------------------------------------------------------------------------


class TestResultObjects:
    def test_parse_evaluate_result_false(self):
        m = parse.parse("{:d} {name}", "3 bob", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert r.fixed == (3,)
        assert r.named == {"name": "bob"}

    def test_parse_evaluate_result_false_no_match(self):
        assert parse.parse("{:d}", "x", evaluate_result=False) is None

    def test_result_repr(self):
        r = parse.parse("{} {a}", "1 2")
        assert repr(r) == "<Result ('1',) {'a': '2'}>"

    def test_result_direct(self):
        r = Result((1, 2), {"a": 3}, {})
        assert r[0] == 1
        assert r["a"] == 3
        assert "a" in r
        assert r[0:2] == (1, 2)

    def test_parser_repr_short(self):
        assert repr(compile("{a}")) == "<Parser '{a}'>"

    def test_parser_repr_truncated(self):
        fmt = "a" * 25
        assert repr(compile(fmt)) == "<Parser %r>" % ("a" * 17 + "...")

    def test_parser_repr_boundary(self):
        fmt = "a" * 20
        assert repr(compile(fmt)) == "<Parser %r>" % fmt

    def test_compile_returns_parser_reusable(self):
        p = compile("{:d}+{:d}")
        assert isinstance(p, Parser)
        assert p.parse("1+2").fixed == (1, 2)
        assert p.parse("3+4").fixed == (3, 4)
        assert p.parse("x") is None


# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------


class TestCustomTypes:
    def test_extra_type_without_pattern(self):
        def shout(s):
            return s.upper()

        r = parse.parse("{:shout}!", "hi!", extra_types={"shout": shout})
        assert r[0] == "HI"

    def test_with_pattern(self):
        @with_pattern(r"\d+")
        def number(text):
            return int(text)

        assert number.pattern == r"\d+"
        assert number.regex_group_count is None
        r = parse.parse("{:Number}abc", "42abc", {"Number": number})
        assert r[0] == 42
        assert parse.parse("{:Number}", "x", {"Number": number}) is None

    def test_with_pattern_groups(self):
        @with_pattern(r"(yes|no)", regex_group_count=1)
        def yesno(text):
            return text == "yes"

        r = parse.parse("{:YN} {:d} {}", "yes 5 tail", {"YN": yesno})
        assert r.fixed == (True, 5, "tail")

    def test_with_pattern_groups_named(self):
        @with_pattern(r"((a)|(b))", regex_group_count=3)
        def ab(text):
            return text * 2

        r = parse.parse("{x:AB} {} {}", "a 1 2", {"AB": ab})
        assert r["x"] == "aa"
        assert r.fixed == ("1", "2")

    def test_custom_type_overrides_builtin(self):
        def myint(s):
            return "custom:" + s

        r = parse.parse("{:d}", "12", extra_types={"d": myint})
        assert r[0] == "custom:12"

    def test_custom_type_in_search_and_findall(self):
        @with_pattern(r"[A-Z]{3}")
        def code(text):
            return text.lower()

        assert search("{:C}", "xx ABC yy", extra_types={"C": code}, case_sensitive=True)[0] == "abc"
        res = [r[0] for r in findall("{:C}", "ABC DEF", extra_types={"C": code})]
        assert res == ["abc", "def"]

    def test_compile_with_extra_types(self):
        p = compile("{:Up}", extra_types={"Up": lambda s: s.upper()})
        assert p.parse("ab")[0] == "AB"


# ---------------------------------------------------------------------------
# Date/time types
# ---------------------------------------------------------------------------


class TestDateTimeIso:
    def test_ti_date_only(self):
        assert parse.parse("{:ti}", "2023-04-05")[0] == datetime(2023, 4, 5)

    def test_ti_datetime_T(self):
        assert parse.parse("{:ti}", "2023-04-05T10:20:30")[0] == datetime(2023, 4, 5, 10, 20, 30)

    def test_ti_datetime_space(self):
        assert parse.parse("{:ti}", "2023-04-05 10:20")[0] == datetime(2023, 4, 5, 10, 20)

    def test_ti_microseconds(self):
        r = parse.parse("{:ti}", "2023-04-05T10:20:30.5")[0]
        assert r.microsecond == 500000
        r = parse.parse("{:ti}", "2023-04-05T10:20:30.123456789")[0]
        assert r.microsecond == 123456

    def test_ti_z(self):
        r = parse.parse("{:ti}", "2023-04-05T10:20:30Z")[0]
        assert r.utcoffset() == timedelta(0)
        assert r.tzinfo == FixedTzOffset(0, "UTC")

    def test_ti_offset(self):
        r = parse.parse("{:ti}", "2023-04-05T10:20:30+05:30")[0]
        assert r.utcoffset() == utc_offset(330)
        r = parse.parse("{:ti}", "2023-04-05T10:20:30-0130")[0]
        assert r.utcoffset() == utc_offset(-90)

    def test_ti_followed_by_field(self):
        r = parse.parse("{:ti} {}", "2023-04-05 rest")
        assert r.fixed == (datetime(2023, 4, 5), "rest")


class TestDateTimeOther:
    def test_tg_dmy(self):
        assert parse.parse("{:tg}", "20/1/1972")[0] == datetime(1972, 1, 20)

    def test_tg_month_name(self):
        assert parse.parse("{:tg}", "20/Jan/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:tg}", "20-January-1972")[0] == datetime(1972, 1, 20)

    def test_tg_time_ampm(self):
        assert parse.parse("{:tg}", "20/1/1972 10:21:36 PM")[0] == datetime(1972, 1, 20, 22, 21, 36)
        assert parse.parse("{:tg}", "20/1/1972 10:21 AM")[0] == datetime(1972, 1, 20, 10, 21)

    def test_tg_tz(self):
        r = parse.parse("{:tg}", "20/1/1972 10:21:36 +1:00")[0]
        assert r.utcoffset() == utc_offset(60)

    def test_ta_mdy(self):
        assert parse.parse("{:ta}", "1/20/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:ta}", "Jan/20/1972 10:21 PM")[0] == datetime(1972, 1, 20, 22, 21)

    def test_te_email(self):
        r = parse.parse("{:te}", "Mon, 20 Jan 1972 10:21:36 +1000")[0]
        assert r.replace(tzinfo=None) == datetime(1972, 1, 20, 10, 21, 36)
        assert r.utcoffset() == utc_offset(600)

    def test_te_without_day(self):
        r = parse.parse("{:te}", "20 Jan 1972 10:21:36 -0500")[0]
        assert r.replace(tzinfo=None) == datetime(1972, 1, 20, 10, 21, 36)
        assert r.utcoffset() == utc_offset(-300)

    def test_th_http(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
        assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 10, 21, 36)
        assert r.utcoffset() == utc_offset(600)

    def test_tc_ctime(self):
        assert parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0] == datetime(1973, 9, 16, 1, 3, 52)

    def test_tt_time(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
        assert parse.parse("{:tt}", "10:21:36 PM")[0] == time(22, 21, 36)

    def test_tt_12_am_pm(self):
        assert parse.parse("{:tt}", "12:15 AM")[0] == time(0, 15)
        assert parse.parse("{:tt}", "12:15 PM")[0] == time(12, 15)
        assert parse.parse("{:tt}", "1:15 AM")[0] == time(1, 15)

    def test_tt_tz(self):
        r = parse.parse("{:tt}", "10:21:36 +5:30")[0]
        assert r.utcoffset() == utc_offset(330)
        assert r.replace(tzinfo=None) == time(10, 21, 36)

    def test_tt_negative_tz(self):
        r = parse.parse("{:tt}", "10:21 -0800")[0]
        assert r.utcoffset() == utc_offset(-480)

    def test_ts_syslog(self):
        r = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
        assert r == datetime(datetime.today().year, 11, 9, 3, 37, 44)

    def test_datetime_fields_followed_by_fixed(self):
        r = parse.parse("{:tg} {} {:d}", "20/1/1972 x 3")
        assert r.fixed == (datetime(1972, 1, 20), "x", 3)
        r = parse.parse("{:tt} {}", "10:21 rest")
        assert r.fixed == (time(10, 21), "rest")

    def test_named_datetime(self):
        r = parse.parse("on {d:ti} by {who}", "on 2020-01-02 by me")
        assert r["d"] == datetime(2020, 1, 2)
        assert r["who"] == "me"


class TestStrftimeFormats:
    def test_ymd_returns_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-04-05")[0]
        assert r == date(2023, 4, 5)
        assert type(r) is date

    def test_full_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 10:20:30")[0]
        assert r == datetime(2023, 4, 5, 10, 20, 30)
        assert type(r) is datetime

    def test_time_only(self):
        r = parse.parse("{:%H:%M}", "10:20")[0]
        assert r == time(10, 20)
        assert type(r) is time

    def test_microseconds(self):
        r = parse.parse("{:%H:%M:%S.%f}", "10:20:30.123")[0]
        assert r == time(10, 20, 30, 123000)

    def test_month_names(self):
        assert parse.parse("{:%d %b %Y}", "05 Apr 2023")[0] == date(2023, 4, 5)
        assert parse.parse("{:%d %B %Y}", "05 April 2023")[0] == date(2023, 4, 5)

    def test_weekday(self):
        assert parse.parse("{:%a %Y-%m-%d}", "Wed 2023-04-05")[0] == date(2023, 4, 5)

    def test_two_digit_year(self):
        assert parse.parse("{:%y/%m/%d}", "23/04/05")[0] == date(2023, 4, 5)

    def test_year_not_specified_uses_current_year(self):
        r = parse.parse("{:%m-%d}", "04-05")[0]
        assert r == date(datetime.today().year, 4, 5)

    def test_ampm(self):
        r = parse.parse("{:%I:%M %p}", "10:20 PM")[0]
        assert r == time(22, 20)

    def test_tz(self):
        r = parse.parse("{:%Y-%m-%d %H:%M %z}", "2023-04-05 10:20 +0530")[0]
        assert r.utcoffset() == utc_offset(330)

    def test_day_of_year(self):
        assert parse.parse("{:%Y %j}", "2023 100")[0] == date(2023, 4, 10)

    def test_no_match(self):
        assert parse.parse("{:%Y-%m-%d}", "20-04-05") is None

    def test_with_other_fields(self):
        r = parse.parse("{} on {:%Y-%m-%d}", "event on 2023-04-05")
        assert r.fixed == ("event", date(2023, 4, 5))

    def test_regex_for_datetime_format(self):
        rx = parse.get_regex_for_datetime_format("%Y-%m-%d")
        assert re.fullmatch(rx, "2023-04-05")
        assert not re.fullmatch(rx, "23-04-05")

    def test_strf_date_convert_neither(self):
        with pytest.raises(ValueError):
            parse.strf_date_convert("x", None, "x")


# ---------------------------------------------------------------------------
# date_convert & FixedTzOffset directly
# ---------------------------------------------------------------------------


class TestDateConvertDirect:
    def _match(self, pattern, s):
        return re.match(pattern, s)

    def test_ymd(self):
        m = self._match(r"(\S+)", "2020-03-04")
        assert parse.date_convert(None, m, ymd=0) == datetime(2020, 3, 4)

    def test_mdy(self):
        m = self._match(r"(\S+)", "03/04/2020")
        assert parse.date_convert(None, m, mdy=0) == datetime(2020, 3, 4)

    def test_dmy(self):
        m = self._match(r"(\S+)", "03/04/2020")
        assert parse.date_convert(None, m, dmy=0) == datetime(2020, 4, 3)

    def test_dmy_month_name(self):
        m = self._match(r"(\S+)", "03-Apr-2020")
        assert parse.date_convert(None, m, dmy=0) == datetime(2020, 4, 3)

    def test_d_m_y(self):
        m = self._match(r"(\d+) (\w+) (\d+)", "3 Apr 2020")
        assert parse.date_convert(None, m, d_m_y=(0, 1, 2)) == datetime(2020, 4, 3)

    def test_time_only(self):
        m = self._match(r"(\S+)", "01:02:03")
        assert parse.date_convert(None, m, hms=0) == time(1, 2, 3)

    def test_tz_formats(self):
        for tzs, minutes in [("+0100", 60), ("-0230", -150), ("+1:00", 60), ("+100", 60), ("-12:45", -765)]:
            m = self._match(r"(\S+) (\S+)", "01:02 " + tzs)
            t = parse.date_convert(None, m, hms=0, tz=1)
            assert t.utcoffset() == utc_offset(minutes), tzs

    def test_uppercase_tz_name_kept_as_string_ignored(self):
        m = self._match(r"(\S+) (\S+)", "01:02 Z")
        assert parse.date_convert(None, m, hms=0, tz=1).utcoffset() == timedelta(0)


class TestFixedTzOffset:
    def test_basic(self):
        tz = FixedTzOffset(90, "+01:30")
        assert tz.utcoffset(None) == timedelta(minutes=90)
        assert tz.tzname(None) == "+01:30"
        assert tz.dst(None) == timedelta(0)

    def test_eq(self):
        assert FixedTzOffset(60, "a") == FixedTzOffset(60, "a")
        assert FixedTzOffset(60, "a") != FixedTzOffset(60, "b")
        assert FixedTzOffset(60, "a") != FixedTzOffset(30, "a")
        assert FixedTzOffset(60, "a") != "a"

    def test_repr(self):
        assert repr(FixedTzOffset(60, "X")) == "<FixedTzOffset X 1:00:00>"

    def test_negative(self):
        assert FixedTzOffset(-60, "x").utcoffset(None) == timedelta(minutes=-60)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_all_exports(self):
        assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
        for name in parse.__all__:
            assert callable(getattr(parse, name))

    def test_parser_picklable(self):
        p = compile("{:d} {name:w} {:ti}")
        p2 = pickle.loads(pickle.dumps(p))
        r = p2.parse("5 bob 2020-01-02")
        assert r.fixed == (5, datetime(2020, 1, 2))
        assert r["name"] == "bob"

    def test_many_fields(self):
        fmt = " ".join("{:d}" for _ in range(50))
        s = " ".join(str(i) for i in range(50))
        assert parse.parse(fmt, s).fixed == tuple(range(50))

    def test_many_named_fields(self):
        fmt = " ".join("{f%d}" % i for i in range(60))
        s = " ".join(str(i) for i in range(60))
        r = parse.parse(fmt, s)
        assert r["f59"] == "59"
        assert len(r.named) == 60

    def test_unicode(self):
        r = parse.parse("héllo {}", "héllo wörld")
        assert r[0] == "wörld"

    def test_convert_first(self):
        c = parse.convert_first(int)
        assert c("12", None) == 12

    def test_empty_format_and_string(self):
        r = parse.parse("", "")
        assert r is not None and r.fixed == ()
        assert parse.parse("", "x") is None
