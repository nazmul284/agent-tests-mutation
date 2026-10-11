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
    compile,
    extract_format,
    findall,
    int_convert,
    search,
    with_pattern,
)


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------


class TestBasic:
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

    def test_multiple_fixed(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r.fixed == ("a", "b", "c")
        assert r[1:] == ("b", "c")
        assert r[-1] == "c"

    def test_named_field(self):
        r = parse.parse("hello {name}", "hello world")
        assert r.named == {"name": "world"}
        assert r["name"] == "world"
        assert "name" in r
        assert "other" not in r

    def test_mixed_fields(self):
        r = parse.parse("{} {name} {}", "a b c")
        assert r.fixed == ("a", "c")
        assert r.named == {"name": "b"}

    def test_numbered_fields_are_fixed(self):
        r = parse.parse("{0} {1}", "a b")
        assert r.fixed == ("a", "b")

    def test_escaped_braces(self):
        r = parse.parse("{{hello}} {}", "{hello} world")
        assert r.fixed == ("world",)
        assert parse.parse("{{}}", "{}").fixed == ()
        assert parse.parse("{{x}}", "x") is None

    def test_regex_special_chars_in_literal(self):
        literal = "a.b*c+d?e(f)[g]^h$i|j\\k!"
        r = parse.parse(literal + "{}", literal + "Z")
        assert r.fixed == ("Z",)
        # a "." in the format must not behave as a wildcard
        assert parse.parse("a.b", "axb") is None
        assert parse.parse("a*", "aaa") is None

    def test_case_insensitive_default(self):
        assert parse.parse("Hello {}", "hELLO world").fixed == ("world",)

    def test_case_sensitive(self):
        assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
        assert parse.parse("Hello {}", "Hello world", case_sensitive=True).fixed == (
            "world",
        )

    def test_dotall_newlines(self):
        r = parse.parse("{}:{}", "a\nb:c\nd")
        assert r.fixed == ("a\nb", "c\nd")

    def test_non_greedy_field(self):
        r = parse.parse("{} {}", "a b c")
        assert r.fixed == ("a", "b c")

    def test_empty_field_does_not_match_empty(self):
        assert parse.parse("x{}y", "xy") is None

    def test_unicode(self):
        r = parse.parse("{} ☃ {}", "ä ☃ ö")
        assert r.fixed == ("ä", "ö")

    def test_result_repr(self):
        r = parse.parse("{} {x}", "a b")
        assert repr(r) == "<Result ('a',) {'x': 'b'}>"


# ---------------------------------------------------------------------------
# Named fields: dotted, indexed, hyphens, repeats
# ---------------------------------------------------------------------------


class TestNames:
    def test_dotted_name(self):
        r = parse.parse("{hello.world}", "x")
        assert r.named == {"hello.world": "x"}

    def test_dotted_and_underscore_no_collision(self):
        r = parse.parse("{a.b} {a_b}", "1 2")
        assert r.named == {"a.b": "1", "a_b": "2"}

    def test_underscore_then_dotted_no_collision(self):
        r = parse.parse("{a_b} {a.b}", "1 2")
        assert r.named == {"a_b": "1", "a.b": "2"}

    def test_hyphen_name(self):
        r = parse.parse("{a-b} {c}", "1 2")
        assert r.named == {"a-b": "1", "c": "2"}

    def test_hyphen_and_underscore_no_collision(self):
        r = parse.parse("{a_b} {a-b}", "1 2")
        assert r.named == {"a_b": "1", "a-b": "2"}

    def test_hyphen_collision_hyphen_first(self):
        r = parse.parse("{a-b} {a_b}", "1 2")
        assert r.named == {"a-b": "1", "a_b": "2"}

    def test_three_way_collision(self):
        r = parse.parse("{a.b} {a_b} {a-b}", "1 2 3")
        assert r.named == {"a.b": "1", "a_b": "2", "a-b": "3"}

    def test_indexed_name_builds_dict(self):
        r = parse.parse("{a[b]} {a[c]}", "1 2")
        assert r.named == {"a": {"b": "1", "c": "2"}}
        assert r["a"]["b"] == "1"

    def test_nested_index(self):
        r = parse.parse("{a[b][c]} {a[b][d]} {a[e]}", "1 2 3")
        assert r.named == {"a": {"b": {"c": "1", "d": "2"}, "e": "3"}}

    def test_index_span(self):
        r = parse.parse("{a[b]} {a[c]}", "1 22")
        assert r.spans["a[b]"] == (0, 1)
        assert r.spans["a[c]"] == (2, 4)

    def test_repeated_name_must_match_same_value(self):
        r = parse.parse("{x} {x}", "a a")
        assert r.named == {"x": "a"}
        assert parse.parse("{x} {x}", "a b") is None

    def test_repeated_name_with_type(self):
        r = parse.parse("{x:d}-{x:d}", "5-5")
        assert r.named == {"x": 5}
        assert parse.parse("{x:d}-{x:d}", "5-6") is None

    def test_repeated_name_type_mismatch(self):
        with pytest.raises(RepeatedNameError):
            parse.compile("{x:d} {x:w}")
        with pytest.raises(ValueError):
            parse.compile("{x} {x:d}")

    def test_repeated_dotted_name(self):
        r = parse.parse("{a.b} {a.b}", "z z")
        assert r.named == {"a.b": "z"}

    def test_named_fields_property(self):
        p = compile("{a} {b.c} {}")
        assert p.named_fields == ["a", "b_c"]
        # fixed fields are recorded as regex group indices
        assert p.fixed_fields == [2]
        # copies returned
        p.named_fields.append("zz")
        p.fixed_fields.append(99)
        assert p.named_fields == ["a", "b_c"]
        assert p.fixed_fields == [2]

    def test_format_property_and_repr(self):
        p = compile("{a} {b}")
        assert p.format == "{a} {b}"
        assert repr(p) == "<Parser '{a} {b}'>"
        long = "x" * 25
        assert repr(compile(long)) == "<Parser %r>" % ("x" * 17 + "...")
        assert repr(compile("x" * 20)) == "<Parser %r>" % ("x" * 20)


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "text,value",
        [
            ("0", 0),
            ("42", 42),
            ("-42", -42),
            ("+42", 42),
            (" 42", 42),
            ("0x1F", 31),
            ("0X1f", 31),
            ("-0x10", -16),
            ("0b101", 5),
            ("0B11", 3),
            ("0o17", 15),
            ("0O17", 15),
            ("+0o7", 7),
            ("007", 7),
            ("10", 10),
        ],
    )
    def test_d(self, text, value):
        r = parse.parse("{:d}", text)
        assert r is not None
        assert r[0] == value
        assert type(r[0]) is int

    def test_d_rejects_non_number(self):
        assert parse.parse("{:d}", "abc") is None
        assert parse.parse("{:d}", "1.5") is None
        assert parse.parse("{:d}", "") is None

    def test_d_width_limits_digits(self):
        r = parse.parse("{:2d}{:2d}", "1234")
        assert r.fixed == (12, 34)
        assert parse.parse("{:2d}", "123") is None

    def test_d_width_with_padding(self):
        assert parse.parse("{:4d}", "  12")[0] == 12

    def test_d_zero_width(self):
        r = parse.parse("{:02d}:{:02d}", "07:05")
        assert r.fixed == (7, 5)

    def test_d_grouping_comma(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567

    def test_d_grouping_underscore(self):
        assert parse.parse("{:_d}", "1_000")[0] == 1000

    def test_d_no_grouping_rejects_comma(self):
        assert parse.parse("{:d}", "1,000") is None

    def test_d_sign_spec(self):
        assert parse.parse("{:+d}", "+5")[0] == 5
        assert parse.parse("{:-d}", "-5")[0] == -5
        assert parse.parse("{: d}", " 5")[0] == 5

    def test_n_grouping(self):
        assert parse.parse("{:n}", "1,234,567")[0] == 1234567
        assert parse.parse("{:n}", "1.234.567")[0] == 1234567
        assert parse.parse("{:n}", "123")[0] == 123
        assert parse.parse("{:n}", "-1,000")[0] == -1000
        assert parse.parse("{:n}", "1,23") is None

    @pytest.mark.parametrize(
        "fmt,text,value",
        [
            ("{:b}", "1010", 10),
            ("{:b}", "0b1010", 10),
            ("{:b}", "-0b11", -3),
            ("{:o}", "17", 15),
            ("{:o}", "0o17", 15),
            ("{:o}", "-17", -15),
            ("{:x}", "ff", 255),
            ("{:x}", "FF", 255),
            ("{:x}", "0xff", 255),
            ("{:X}", "0XAB", 171),
            ("{:x}", "-a", -10),
            ("{:x}", "10", 16),
            ("{:o}", "10", 8),
            ("{:b}", "10", 2),
        ],
    )
    def test_bases(self, fmt, text, value):
        r = parse.parse(fmt, text)
        assert r is not None
        assert r[0] == value

    def test_base_rejects_invalid_digits(self):
        assert parse.parse("{:b}", "102") is None
        assert parse.parse("{:o}", "8") is None
        assert parse.parse("{:x}", "g") is None

    def test_x_value_starting_with_b(self):
        # "0b1" as hex is 0xb1, not binary
        assert parse.parse("{:x}", "0b1")[0] == 0xB1

    def test_numeric_align_equals(self):
        assert parse.parse("{:=5d}", "00042")[0] == 42
        assert parse.parse("{:x=5d}", "xxx42")[0] == 42
        assert parse.parse("{:=5d}", "-0042")[0] == -42

    def test_int_followed_by_text(self):
        r = parse.parse("{:d}px", "12px")
        assert r[0] == 12

    def test_two_ints_separated(self):
        r = parse.parse("{:d},{:d}", "-3,+4")
        assert r.fixed == (-3, 4)


class TestIntConvert:
    def test_auto_base(self):
        c = int_convert()
        assert c("0x10", None) == 16
        assert c("0b10", None) == 2
        assert c("0o10", None) == 8
        assert c("10", None) == 10
        assert c("-0x10", None) == -16
        assert c("+0b11", None) == 3

    def test_short_strings_stay_decimal(self):
        c = int_convert()
        assert c("0", None) == 0
        assert c("00", None) == 0
        assert c("-0", None) == 0
        assert c("05", None) == 5

    def test_explicit_base(self):
        assert int_convert(16)("ff", None) == 255
        assert int_convert(16)("0xff", None) == 255
        assert int_convert(2)("0b11", None) == 3
        assert int_convert(8)("0o11", None) == 9
        assert int_convert(10)("1,000", None) == 1000
        assert int_convert(10)("-1.000", None) == -1000

    def test_strips_grouping(self):
        assert int_convert()("1_000", None) == 1000
        assert int_convert()("1,000", None) == 1000
        assert int_convert()("0x_ff", None) == 255


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
            ("0.0", 0.0),
            ("123.456", 123.456),
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
        assert parse.parse("{:f}", "inf")[0] == float("inf")
        assert parse.parse("{:f}", "-inf")[0] == float("-inf")
        assert parse.parse("{:f}", "INF")[0] == float("inf")

    def test_f_precision_zero(self):
        assert parse.parse("{:.0f}", "20")[0] == 20.0
        assert parse.parse("{:.0f}", "-20")[0] == -20.0
        assert parse.parse("{:.0f}", "20.5") is None

    def test_f_precision(self):
        assert parse.parse("{:.2f}", "3.14")[0] == 3.14
        assert parse.parse("{:8.2f}", "    3.14")[0] == 3.14

    def test_F_decimal(self):
        r = parse.parse("{:F}", "1.10")
        assert r[0] == Decimal("1.10")
        assert isinstance(r[0], Decimal)
        assert parse.parse("{:F}", "-0.5")[0] == Decimal("-0.5")

    @pytest.mark.parametrize(
        "text,value",
        [
            ("1.0e10", 1.0e10),
            ("1.5E-3", 1.5e-3),
            ("-2.5e+2", -250.0),
            (".5e1", 5.0),
        ],
    )
    def test_e(self, text, value):
        r = parse.parse("{:e}", text)
        assert r is not None
        assert r[0] == pytest.approx(value)

    def test_e_requires_exponent(self):
        assert parse.parse("{:e}", "1.5") is None

    def test_e_inf(self):
        assert parse.parse("{:e}", "inf")[0] == float("inf")
        assert parse.parse("{:E}", "-INF")[0] == float("-inf")

    @pytest.mark.parametrize(
        "text,value",
        [
            ("1", 1.0),
            ("1.5", 1.5),
            ("-1.5", -1.5),
            ("1e5", 1e5),
            ("1.5E-2", 0.015),
            ("+3", 3.0),
        ],
    )
    def test_g(self, text, value):
        r = parse.parse("{:g}", text)
        assert r is not None
        assert r[0] == pytest.approx(value)
        assert type(r[0]) is float

    def test_g_nan(self):
        import math

        assert math.isnan(parse.parse("{:G}", "NAN")[0])

    def test_g_group_offsets_followed_by_other_field(self):
        # g uses internal regex groups; subsequent fields must still line up
        r = parse.parse("{:g} {} {:d}", "1.5e3 x 7")
        assert r.fixed == (1500.0, "x", 7)

    def test_percentage(self):
        assert parse.parse("{:%}", "50%")[0] == 0.5
        assert parse.parse("{:%}", "12.5%")[0] == pytest.approx(0.125)
        assert parse.parse("{:%}", "-10%")[0] == pytest.approx(-0.1)
        assert parse.parse("{:%}", "50") is None

    def test_percentage_then_field(self):
        r = parse.parse("{:%} {:d}", "12.5% 3")
        assert r.fixed == (pytest.approx(0.125), 3)


# ---------------------------------------------------------------------------
# String types
# ---------------------------------------------------------------------------


class TestStringTypes:
    def test_w(self):
        assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
        assert parse.parse("{:w}", "abc def") is None

    def test_W(self):
        assert parse.parse("{:W}", "!@ #")[0] == "!@ #"
        assert parse.parse("{:W}", "a") is None

    def test_s(self):
        assert parse.parse("{:s}", " \t\n")[0] == " \t\n"
        assert parse.parse("{:s}", "a") is None

    def test_S(self):
        assert parse.parse("{:S}", "abc!")[0] == "abc!"
        assert parse.parse("{:S}", "a b") is None

    def test_d_vs_D(self):
        assert parse.parse("{:D}", "abc")[0] == "abc"
        assert parse.parse("{:D}", "a1") is None

    def test_l_letters(self):
        assert parse.parse("{:l}", "Hello")[0] == "Hello"
        assert parse.parse("{:l}", "Hello1") is None
        assert parse.parse("{:l}", "a_b") is None

    def test_w_split(self):
        r = parse.parse("{:w} {:w}", "foo bar")
        assert r.fixed == ("foo", "bar")

    def test_width_string(self):
        r = parse.parse("{:3}{}", "abcdef")
        assert r.fixed == ("abc", "def")

    def test_precision_string(self):
        r = parse.parse("{:.2}{}", "abcdef")
        assert r.fixed == ("a", "bcdef")
        assert parse.parse("{:.2}", "abc") is None
        assert parse.parse("{:.2}", "ab")[0] == "ab"

    def test_width_and_precision_string(self):
        r = parse.parse("{:2.4}", "abc")
        assert r[0] == "abc"
        assert parse.parse("{:2.4}", "a") is None
        assert parse.parse("{:2.4}", "abcde") is None

    def test_width_is_minimum(self):
        assert parse.parse("{:3}", "ab") is None
        assert parse.parse("{:3}", "abcd")[0] == "abcd"


# ---------------------------------------------------------------------------
# Alignment and fill
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_left(self):
        r = parse.parse("{:<}|", "abc   |")
        assert r[0] == "abc"

    def test_right(self):
        r = parse.parse("|{:>}", "|   abc")
        assert r[0] == "abc"

    def test_center(self):
        r = parse.parse("|{:^}|", "|  abc  |")
        assert r[0] == "abc"

    def test_fill_char(self):
        assert parse.parse("{:x<}|", "abcxxx|")[0] == "abc"
        assert parse.parse("|{:x>}", "|xxxabc")[0] == "abc"
        assert parse.parse("|{:x^}|", "|xxabcxx|")[0] == "abc"

    @pytest.mark.parametrize("fill", list(".\\+?*[]()$"))
    def test_regex_special_fill(self, fill):
        r = parse.parse("|{:%s^}|" % fill, "|%sabc%s|" % (fill * 2, fill * 2))
        assert r is not None
        assert r[0] == "abc"

    def test_width_is_minimum_including_padding(self):
        # the width is a minimum length for the captured text, so untyped
        # fields keep their padding
        assert parse.parse("{:5}", "  abc")[0] == "  abc"
        assert parse.parse("{:5}", "abc") is None

    def test_width_with_d_padding(self):
        assert parse.parse("{:>5d}", "   42")[0] == 42
        assert parse.parse("{:<5d}|", "42   |")[0] == 42
        assert parse.parse("{:^6d}|", "  42  |")[0] == 42

    def test_numeric_fill(self):
        assert parse.parse("{:*>6d}", "****42")[0] == 42
        assert parse.parse("{:0>4d}", "0042")[0] == 42

    def test_center_float(self):
        assert parse.parse("{:^9.2f}", "  3.14   ")[0] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# extract_format
# ---------------------------------------------------------------------------


class TestExtractFormat:
    def test_full_spec(self):
        f = extract_format("x^+010,.3f", {})
        assert f["fill"] == "x"
        assert f["align"] == "^"
        assert f["zero"] is True
        assert f["width"] == "10"
        assert f["grouping"] == ","
        assert f["precision"] == "3"
        assert f["type"] == "f"

    def test_align_only(self):
        f = extract_format("<", {})
        assert f["align"] == "<"
        assert f["fill"] is None
        assert f["type"] == ""

    def test_fill_align(self):
        f = extract_format("*>8", {})
        assert f["fill"] == "*"
        assert f["align"] == ">"
        assert f["width"] == "8"

    def test_zero_and_width(self):
        f = extract_format("05d", {})
        assert f["zero"] is True
        assert f["width"] == "5"
        assert f["type"] == "d"

    def test_no_zero(self):
        f = extract_format("5d", {})
        assert f["zero"] is False
        assert f["width"] == "5"

    def test_underscore_grouping(self):
        assert extract_format("_d", {})["grouping"] == "_"

    def test_no_grouping_key(self):
        assert "grouping" not in extract_format("d", {})
        assert "precision" not in extract_format("d", {})

    def test_type_only(self):
        f = extract_format("ti", {})
        assert f["type"] == "ti"
        assert f["width"] == ""

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            extract_format("q", {})
        with pytest.raises(ValueError):
            parse.compile("{:q}")
        with pytest.raises(ValueError):
            parse.parse("{:zz}", "x")

    def test_extra_type_accepted(self):
        f = extract_format("custom", {"custom": str})
        assert f["type"] == "custom"

    def test_datetime_format_accepted(self):
        assert extract_format("%Y-%m-%d", {})["type"] == "%Y-%m-%d"

    @pytest.mark.parametrize("t", sorted(parse.ALLOWED_TYPES))
    def test_all_allowed_types_compile(self, t):
        parse.compile("{:%s}" % t)


# ---------------------------------------------------------------------------
# Date/time types
# ---------------------------------------------------------------------------


UTC = FixedTzOffset(0, "UTC")


class TestDateTimes:
    def test_ti_date_only(self):
        assert parse.parse("{:ti}", "2011-03-04")[0] == datetime(2011, 3, 4)

    def test_ti_full(self):
        r = parse.parse("{:ti}", "2011-03-04T12:34:56")
        assert r[0] == datetime(2011, 3, 4, 12, 34, 56)

    def test_ti_space_sep(self):
        assert parse.parse("{:ti}", "2011-03-04 01:02")[0] == datetime(
            2011, 3, 4, 1, 2
        )

    def test_ti_microseconds(self):
        r = parse.parse("{:ti}", "2011-03-04T12:34:56.123")
        assert r[0] == datetime(2011, 3, 4, 12, 34, 56, 123000)
        r = parse.parse("{:ti}", "2011-03-04T12:34:56.1234567")
        assert r[0].microsecond == 123456

    def test_ti_z(self):
        r = parse.parse("{:ti}", "2011-03-04T12:34:56Z")
        assert r[0] == datetime(2011, 3, 4, 12, 34, 56, tzinfo=UTC)
        assert r[0].utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        "suffix,minutes",
        [
            ("+10:30", 630),
            ("-05:00", -300),
            ("+0130", 90),
            ("-0130", -90),
            (" +01:00", 60),
        ],
    )
    def test_ti_offsets(self, suffix, minutes):
        r = parse.parse("{:ti}", "2011-03-04T12:34:56" + suffix)
        assert r is not None
        assert r[0].utcoffset() == timedelta(minutes=minutes)
        assert r[0].replace(tzinfo=None) == datetime(2011, 3, 4, 12, 34, 56)

    def test_ti_then_field(self):
        r = parse.parse("{:ti} {:d}", "2011-03-04T01:02:03 9")
        assert r.fixed == (datetime(2011, 3, 4, 1, 2, 3), 9)

    def test_tg(self):
        assert parse.parse("{:tg}", "20/1/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:tg}", "20/1/1972 10:21:36")[0] == datetime(
            1972, 1, 20, 10, 21, 36
        )
        assert parse.parse("{:tg}", "20-Jan-1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:tg}", "20/January/1972")[0] == datetime(1972, 1, 20)

    def test_tg_am_pm(self):
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
        r = parse.parse("{:tg}", "20/1/1972 10:21:36 PM +1:00")
        assert r[0].replace(tzinfo=None) == datetime(1972, 1, 20, 22, 21, 36)
        assert r[0].utcoffset() == timedelta(hours=1)
        r = parse.parse("{:tg}", "20/1/1972 10:21:36 -0530")
        assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)
        r = parse.parse("{:tg}", "20/1/1972 10:21:36 +130")
        assert r[0].utcoffset() == timedelta(hours=1, minutes=30)

    def test_ta(self):
        assert parse.parse("{:ta}", "1/20/1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:ta}", "Jan-20-1972")[0] == datetime(1972, 1, 20)
        assert parse.parse("{:ta}", "1/20/1972 10:21:36 PM")[0] == datetime(
            1972, 1, 20, 22, 21, 36
        )
        r = parse.parse("{:ta}", "1/20/1972 10:21:36 AM +01:00")
        assert r[0].utcoffset() == timedelta(hours=1)
        assert r[0].hour == 10

    def test_te(self):
        r = parse.parse("{:te}", "Mon, 20 Jan 1972 10:21:36 +1000")
        assert r[0].replace(tzinfo=None) == datetime(1972, 1, 20, 10, 21, 36)
        assert r[0].utcoffset() == timedelta(hours=10)
        r = parse.parse("{:te}", "20 Jan 1972 10:21:36 -0100")
        assert r[0].utcoffset() == -timedelta(hours=1)
        assert r[0].day == 20 and r[0].month == 1

    def test_th(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")
        assert r[0].replace(tzinfo=None) == datetime(2011, 11, 21, 10, 21, 36)
        assert r[0].utcoffset() == timedelta(hours=10)

    def test_tc(self):
        r = parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")
        assert r[0] == datetime(1973, 9, 16, 1, 3, 52)

    def test_tt(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
        assert parse.parse("{:tt}", "10:21:36 PM")[0] == time(22, 21, 36)
        assert parse.parse("{:tt}", "12:00 AM")[0] == time(0, 0)
        r = parse.parse("{:tt}", "10:21:36 PM -5:30")
        assert r[0].replace(tzinfo=None) == time(22, 21, 36)
        assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)

    def test_tt_microseconds(self):
        assert parse.parse("{:tt}", "10:21:36.5")[0] == time(10, 21, 36, 500000)

    def test_ts(self):
        r = parse.parse("{:ts}", "Nov  9 03:37:44")
        assert r[0] == datetime(datetime.today().year, 11, 9, 3, 37, 44)
        r = parse.parse("{:ts}", "Feb 1 00:00:01")
        assert r[0] == datetime(datetime.today().year, 2, 1, 0, 0, 1)

    def test_datetime_field_offsets(self):
        r = parse.parse("{:tg} {} {:tt} {:d}", "20/1/1972 x 10:21 5")
        assert r.fixed == (datetime(1972, 1, 20), "x", time(10, 21), 5)

    def test_named_datetime(self):
        r = parse.parse("on {when:ti} by {who}", "on 2020-01-02 by bob")
        assert r["when"] == datetime(2020, 1, 2)
        assert r["who"] == "bob"

    def test_invalid_date_no_match(self):
        assert parse.parse("{:ti}", "not a date") is None


class TestStrftime:
    def test_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-11-21")
        assert r[0] == date(2023, 11, 21)
        assert type(r[0]) is date

    def test_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-11-21 13:23:27")
        assert r[0] == datetime(2023, 11, 21, 13, 23, 27)
        assert type(r[0]) is datetime

    def test_time(self):
        r = parse.parse("{:%H:%M:%S}", "13:23:27")
        assert r[0] == time(13, 23, 27)
        assert type(r[0]) is time

    def test_time_microseconds(self):
        r = parse.parse("{:%H:%M:%S.%f}", "10:11:12.5")
        assert r[0] == time(10, 11, 12, 500000)

    def test_month_names_and_weekday(self):
        r = parse.parse("{:%A, %d %B %Y}", "Tuesday, 21 November 2023")
        assert r[0] == date(2023, 11, 21)
        r = parse.parse("{:%a %b %d %Y}", "Tue Nov 21 2023")
        assert r[0] == date(2023, 11, 21)

    def test_two_digit_year(self):
        assert parse.parse("{:%d/%m/%y}", "21/11/23")[0] == date(2023, 11, 21)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_no_year_uses_current(self):
        r = parse.parse("{:%m-%d}", "11-21")
        assert r[0] == date(datetime.today().year, 11, 21)

    def test_12h(self):
        r = parse.parse("{:%I:%M %p}", "01:30 PM")
        assert r[0] == time(13, 30)

    def test_tz(self):
        r = parse.parse("{:%Y-%m-%d %H:%M %z}", "2023-11-21 10:00 +0530")
        assert r[0].utcoffset() == timedelta(hours=5, minutes=30)

    def test_day_of_year(self):
        r = parse.parse("{:%Y %j}", "2023 032")
        assert r[0] == date(2023, 2, 1)

    def test_no_match(self):
        assert parse.parse("{:%Y-%m-%d}", "2023/11/21") is None

    def test_regex_for_format(self):
        assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"

    def test_strf_neither(self):
        with pytest.raises(ValueError):
            parse.strf_date_convert("x", None, "x")


class TestDateConvertDirect:
    def test_fixed_tz_offset(self):
        tz = FixedTzOffset(90, "+01:30")
        assert tz.utcoffset(None) == timedelta(minutes=90)
        assert tz.tzname(None) == "+01:30"
        assert tz.dst(None) == timedelta(0)
        assert repr(tz) == "<FixedTzOffset +01:30 1:30:00>"
        assert tz == FixedTzOffset(90, "+01:30")
        assert tz != FixedTzOffset(60, "+01:30")
        assert tz != FixedTzOffset(90, "other")
        assert (tz == 5) is False

    def test_negative_offset(self):
        assert FixedTzOffset(-60, "x").utcoffset(None) == timedelta(hours=-1)


# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------


class TestCustomTypes:
    def test_plain_function(self):
        r = parse.parse("{:shout}", "hi", extra_types={"shout": lambda s: s.upper()})
        assert r[0] == "HI"

    def test_with_pattern(self):
        @with_pattern(r"\d+")
        def num(text):
            return int(text)

        assert num.pattern == r"\d+"
        assert num.regex_group_count is None
        r = parse.parse("{:Num}{:w}", "12ab", extra_types={"Num": num})
        assert r.fixed == (12, "ab")
        assert parse.parse("{:Num}", "x", extra_types={"Num": num}) is None

    def test_with_pattern_groups(self):
        @with_pattern(r"((a)|(b))", regex_group_count=3)
        def ab(text):
            return text.upper()

        r = parse.parse("{:ab} {:d} {name:ab}", "a 5 b", extra_types={"ab": ab})
        assert r.fixed == ("A", 5)
        assert r["name"] == "B"
        assert r.spans[1] == (2, 3)

    def test_with_pattern_groups_named_followed_by_fixed(self):
        @with_pattern(r"(\d)(\d)", regex_group_count=2)
        def two(text):
            return text

        r = parse.parse("{:two}-{}", "12-x", extra_types={"two": two})
        assert r.fixed == ("12", "x")

    def test_custom_type_with_width(self):
        @with_pattern(r"[a-z]+")
        def low(text):
            return text

        r = parse.parse("{:>6low}", "   abc", extra_types={"low": low})
        assert r[0] == "abc"

    def test_custom_overrides_builtin(self):
        r = parse.parse("{:d}", "abc", extra_types={"d": lambda s: s * 2})
        assert r[0] == "abcabc"

    def test_named_custom(self):
        @with_pattern(r"yes|no")
        def yn(text):
            return text == "yes"

        r = parse.parse("{a:yn} {b:yn}", "yes no", extra_types={"yn": yn})
        assert r.named == {"a": True, "b": False}


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class TestSpans:
    def test_spans(self):
        r = parse.parse("{} {name} {:d}", "ab cd 123")
        assert r.spans == {0: (0, 2), "name": (3, 5), 1: (6, 9)}

    def test_spans_padding_excluded(self):
        r = parse.parse("[{:^}]", "[  abc  ]")
        assert r.spans[0] == (3, 6)

    def test_spans_after_complex_type(self):
        r = parse.parse("{:ti} {}", "2020-01-02T03:04:05 tail")
        assert r.spans[0] == (0, 19)
        assert r.spans[1] == (20, 24)

    def test_search_spans_absolute(self):
        r = search("<{}>", "xx <abc> yy")
        assert r.spans[0] == (4, 7)


# ---------------------------------------------------------------------------
# search / findall
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search(self):
        r = search("age: {:d}", "name: x age: 42 more")
        assert r[0] == 42

    def test_search_none(self):
        assert search("age: {:d}", "nothing") is None

    def test_search_pos(self):
        assert search("{:d}", "1 2 3", pos=1)[0] == 2
        assert search("{:d}", "1 2 3", pos=2)[0] == 2
        assert search("{:d}", "1 2 3", pos=3)[0] == 3
        assert search("{:d}", "1 2 3", pos=5) is None

    def test_search_endpos(self):
        assert search("{:d}", "abc 12345", endpos=6)[0] == 12
        assert search("{:d}", "abc 12345", endpos=3) is None

    def test_search_case(self):
        assert search("X{:d}", "x5")[0] == 5
        assert search("X{:d}", "x5", case_sensitive=True) is None

    def test_search_evaluate_false(self):
        m = search("{:d}", "a 7", evaluate_result=False)
        assert isinstance(m, Match)
        r = m.evaluate_result()
        assert isinstance(r, Result)
        assert r[0] == 7

    def test_search_extra_types(self):
        r = search("<{:up}>", "x<ab>", extra_types={"up": str.upper})
        assert r[0] == "AB"

    def test_parser_search(self):
        p = compile("{:d}")
        assert p.search("a 5 b")[0] == 5
        assert p.search("a 5 b 6", pos=4)[0] == 6
        assert p.search("a b") is None


class TestFindall:
    def test_findall(self):
        res = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
        assert [r[0] for r in res] == ["some ", "bold", " text"]

    def test_findall_typed(self):
        assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]

    def test_findall_empty(self):
        assert list(findall("{:d}", "abc")) == []

    def test_findall_pos_endpos(self):
        assert [r[0] for r in findall("{:d}", "1 2 3 4", pos=2, endpos=5)] == [2, 3]

    def test_findall_named(self):
        res = findall("{k}={v:d};", "a=1;b=2;")
        assert [(r["k"], r["v"]) for r in res] == [("a", 1), ("b", 2)]

    def test_findall_evaluate_false(self):
        res = list(findall("{:d}", "1 2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in res)
        assert [m.evaluate_result()[0] for m in res] == [1, 2]

    def test_findall_is_iterator(self):
        it = findall("{:d}", "1 2")
        assert iter(it) is it
        assert next(it)[0] == 1
        assert it.next()[0] == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_findall_case(self):
        assert len(list(findall("a{:d}", "A1 a2"))) == 2
        assert len(list(findall("a{:d}", "A1 a2", case_sensitive=True))) == 1

    def test_findall_extra_types(self):
        res = findall("[{:up}]", "[a][b]", extra_types={"up": str.upper})
        assert [r[0] for r in res] == ["A", "B"]

    def test_parser_findall(self):
        p = compile("{:w}")
        assert [r[0] for r in p.findall("ab cd ef", pos=3)] == ["cd", "ef"]


# ---------------------------------------------------------------------------
# Parser / compile / evaluate_result / misc
# ---------------------------------------------------------------------------


class TestParser:
    def test_compile_reuse(self):
        p = compile("{:d}+{:d}")
        assert isinstance(p, Parser)
        assert p.parse("1+2").fixed == (1, 2)
        assert p.parse("3+4").fixed == (3, 4)
        assert p.parse("x") is None

    def test_compile_case_sensitive(self):
        p = compile("A{}", case_sensitive=True)
        assert p.parse("a1") is None
        assert p.parse("A1")[0] == "1"

    def test_compile_extra_types(self):
        p = compile("{:up}", extra_types={"up": str.upper})
        assert p.parse("x")[0] == "X"

    def test_parse_evaluate_false(self):
        m = parse.parse("{a:d}", "5", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.parser.format == "{a:d}"
        assert m.evaluate_result().named == {"a": 5}

    def test_parse_evaluate_false_no_match(self):
        assert parse.parse("{:d}", "x", evaluate_result=False) is None

    def test_parser_parse_evaluate_false(self):
        p = compile("{}")
        m = p.parse("x", evaluate_result=False)
        assert p.evaluate_result(m.match)[0] == "x"

    def test_result_object(self):
        r = Result(("a", "b"), {"x": 1}, {})
        assert r[0] == "a"
        assert r[0:2] == ("a", "b")
        assert r["x"] == 1
        assert "x" in r
        with pytest.raises(KeyError):
            r["y"]
        with pytest.raises(IndexError):
            r[5]

    def test_many_fields(self):
        n = 150
        fmt = " ".join("{}" for _ in range(n))
        s = " ".join(str(i) for i in range(n))
        r = parse.parse(fmt, s)
        assert r.fixed == tuple(str(i) for i in range(n))

    def test_many_named_fields(self):
        n = 120
        fmt = " ".join("{f%d:d}" % i for i in range(n))
        s = " ".join(str(i) for i in range(n))
        r = parse.parse(fmt, s)
        assert r.named == {"f%d" % i: i for i in range(n)}

    def test_pickle_parser_components(self):
        assert pickle.loads(pickle.dumps(int_convert(16)))("ff", None) == 255
        c = pickle.loads(pickle.dumps(parse.convert_first(int)))
        assert c("12", None) == 12

    def test_convert_first(self):
        assert parse.convert_first(str.upper)("ab", object()) == "AB"

    def test_percentage_func(self):
        assert parse.percentage("25%", None) == 0.25

    def test_all(self):
        assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}


# ---------------------------------------------------------------------------
# Realistic examples
# ---------------------------------------------------------------------------


class TestRealistic:
    def test_log_line(self):
        fmt = "{ip} - - [{when:th}] \"{method:w} {path} HTTP/{ver:f}\" {status:d} {size:d}"
        line = (
            '127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /a.gif HTTP/1.0" 200 2326'
        )
        r = parse.parse(fmt, line)
        assert r["ip"] == "127.0.0.1"
        assert r["when"].replace(tzinfo=None) == datetime(2000, 10, 10, 13, 55, 36)
        assert r["when"].utcoffset() == -timedelta(hours=7)
        assert r["method"] == "GET"
        assert r["path"] == "/a.gif"
        assert r["ver"] == 1.0
        assert r["status"] == 200
        assert r["size"] == 2326

    def test_roundtrip_with_format(self):
        fmt = "{:>8d}|{:<}|{:^9.3f}|{:x}|{:o}|{:b}"
        values = (42, "ab", 3.142, 255, 8, 5)
        s = "{:>8d}|{:<6}|{:^9.3f}|{:x}|{:o}|{:b}".format(*values)
        r = parse.parse(fmt, s)
        assert r.fixed == (42, "ab", pytest.approx(3.142), 255, 8, 5)

    def test_roundtrip_negative_float_padded(self):
        s = "{:>10.2f}".format(-3.5)
        assert parse.parse("{:>10.2f}", s)[0] == -3.5

    def test_roundtrip_grouping(self):
        s = "{:,d}".format(-1234567)
        assert parse.parse("{:,d}", s)[0] == -1234567
