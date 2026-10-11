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


class TestBasicParse:
    def test_no_fields_exact_match(self):
        r = parse.parse("hello", "hello")
        assert r is not None
        assert r.fixed == ()
        assert r.named == {}

    def test_no_fields_mismatch(self):
        assert parse.parse("hello", "goodbye") is None

    def test_parse_requires_full_match(self):
        assert parse.parse("hello {}", "say hello world") is None
        assert parse.parse("{} world", "hello world!") is None

    def test_fixed_fields(self):
        r = parse.parse("{} {}", "a b")
        assert r.fixed == ("a", "b")
        assert r[0] == "a"
        assert r[1] == "b"

    def test_fixed_slice(self):
        r = parse.parse("{} {} {}", "a b c")
        assert r[1:] == ("b", "c")

    def test_named_fields(self):
        r = parse.parse("{greeting} {name}", "hi bob")
        assert r.named == {"greeting": "hi", "name": "bob"}
        assert r["name"] == "bob"
        assert "name" in r
        assert "missing" not in r

    def test_mixed_fixed_and_named(self):
        r = parse.parse("{} {x} {}", "1 2 3")
        assert r.fixed == ("1", "3")
        assert r.named == {"x": "2"}

    def test_empty_field_needs_at_least_one_char(self):
        assert parse.parse("a{}b", "ab") is None

    def test_non_greedy_untyped(self):
        r = parse.parse("{}-{}", "a-b-c")
        assert r.fixed == ("a", "b-c")

    def test_escaped_braces(self):
        r = parse.parse("{{{}}}", "{x}")
        assert r.fixed == ("x",)
        assert parse.parse("{{}}", "{}").fixed == ()

    def test_regex_metacharacters_are_literal(self):
        r = parse.parse("a.b*c+d?(e)[f]^$|\\ {}", "a.b*c+d?(e)[f]^$|\\ z")
        assert r.fixed == ("z",)
        assert parse.parse("a.b {}", "aXb z") is None

    def test_multiline_value(self):
        r = parse.parse("<{}>", "<a\nb>")
        assert r.fixed == ("a\nb",)

    def test_newline_at_end_is_not_ignored(self):
        # \Z anchors at the very end of the string, unlike $
        assert parse.parse("{:d}", "12\n") is None


class TestCaseSensitivity:
    def test_parse_default_is_case_insensitive(self):
        # documented: "The default behaviour is to match strings case insensitively"
        r = parse.parse("Hello {}", "hello world")
        assert r is not None
        assert r.fixed == ("world",)

    def test_parse_case_sensitive_explicit(self):
        assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
        assert parse.parse("Hello {}", "Hello world", case_sensitive=True) is not None

    def test_parse_case_insensitive_explicit(self):
        assert parse.parse("Hello {}", "HELLO world", case_sensitive=False) is not None

    def test_search_default_case_insensitive(self):
        assert search("Hello {:w}", "xx hello world").fixed == ("world",)

    def test_search_case_sensitive(self):
        assert search("Hello {}", "xx hello world", case_sensitive=True) is None

    def test_findall_default_case_insensitive(self):
        assert len(list(findall("x{:d}", "X1 x2"))) == 2

    def test_findall_case_sensitive(self):
        assert [r[0] for r in findall("x{:d}", "X1 x2", case_sensitive=True)] == [2]

    def test_compile_default_case_insensitive(self):
        p = compile("Hello {}")
        assert p.parse("HELLO x") is not None

    def test_compile_case_sensitive(self):
        p = compile("Hello {}", case_sensitive=True)
        assert p.parse("HELLO x") is None
        assert p.parse("Hello x") is not None

    def test_parser_default_case_insensitive(self):
        assert Parser("ABC").parse("abc") is not None


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


class TestNames:
    def test_dotted_name(self):
        r = parse.parse("{a.b}", "x")
        assert r.named == {"a.b": "x"}

    def test_dotted_and_underscore_no_collision(self):
        r = parse.parse("{a.b} {a_b}", "1 2")
        assert r.named == {"a.b": "1", "a_b": "2"}

    def test_underscore_then_dotted_no_collision(self):
        r = parse.parse("{a_b} {a.b}", "1 2")
        assert r.named == {"a_b": "1", "a.b": "2"}

    def test_hyphenated_name(self):
        r = parse.parse("{a-b} {a_b}", "1 2")
        assert r.named == {"a-b": "1", "a_b": "2"}

    def test_indexed_name_builds_nested_dict(self):
        r = parse.parse("{a[b]} {a[c]}", "1 2")
        assert r.named == {"a": {"b": "1", "c": "2"}}
        assert r["a"]["b"] == "1"

    def test_deep_index(self):
        r = parse.parse("{a[b][c]}", "z")
        assert r.named == {"a": {"b": {"c": "z"}}}

    def test_repeated_name_must_match_same_value(self):
        r = parse.parse("{x} {x}", "a a")
        assert r.named == {"x": "a"}
        assert parse.parse("{x} {x}", "a b") is None

    def test_repeated_name_typed(self):
        r = parse.parse("{x:d}-{x:d}", "5-5")
        assert r["x"] == 5
        assert parse.parse("{x:d}-{x:d}", "5-6") is None

    def test_repeated_name_mismatched_type_raises(self):
        with pytest.raises(RepeatedNameError):
            Parser("{x:d} {x:w}")
        assert issubclass(RepeatedNameError, ValueError)

    def test_numeric_name_is_fixed(self):
        r = parse.parse("{0} {1}", "a b")
        assert r.fixed == ("a", "b")
        assert r.named == {}

    def test_named_fields_property(self):
        p = compile("{a} {} {b.c}")
        assert p.named_fields == ["a", "b_c"]
        assert p.fixed_fields == [1]
        # returns a copy
        p.named_fields.append("zzz")
        assert p.named_fields == ["a", "b_c"]
        p.fixed_fields.append(9)
        assert p.fixed_fields == [1]

    def test_format_property(self):
        assert compile("{a}x").format == "{a}x"


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestIntegers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0", 0),
            ("123", 123),
            ("-123", -123),
            ("+123", 123),
            (" 12", 12),
            ("0x1F", 31),
            ("0X1f", 31),
            ("-0x10", -16),
            ("0b101", 5),
            ("0B11", 3),
            ("0o17", 15),
            ("0O7", 7),
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

    def test_d_with_grouping(self):
        assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
        assert parse.parse("{:_d}", "1_000")[0] == 1000

    def test_d_without_grouping_rejects_comma(self):
        assert parse.parse("{:d}", "1,000") is None

    def test_d_width_limits_digits(self):
        r = parse.parse("{:2d}{:2d}", "1234")
        assert r.fixed == (12, 34)

    def test_d_width_with_padding(self):
        assert parse.parse("{:4d}", "  42")[0] == 42

    def test_zero_padded_width(self):
        assert parse.parse("{:03d}", "007")[0] == 7

    def test_n(self):
        assert parse.parse("{:n}", "1,234,567")[0] == 1234567
        assert parse.parse("{:n}", "1.234")[0] == 1234
        assert parse.parse("{:n}", "12")[0] == 12
        assert parse.parse("{:n}", "-1,000")[0] == -1000
        assert parse.parse("{:n}", "1,23") is None

    def test_b(self):
        assert parse.parse("{:b}", "1010")[0] == 10
        assert parse.parse("{:b}", "0b1010")[0] == 10
        assert parse.parse("{:b}", "102") is None

    def test_o(self):
        assert parse.parse("{:o}", "17")[0] == 15
        assert parse.parse("{:o}", "0o17")[0] == 15
        assert parse.parse("{:o}", "8") is None

    def test_x(self):
        assert parse.parse("{:x}", "ff")[0] == 255
        assert parse.parse("{:x}", "0xFF")[0] == 255
        assert parse.parse("{:X}", "1A")[0] == 26
        assert parse.parse("{:x}", "-a")[0] == -10
        assert parse.parse("{:x}", "g") is None

    def test_b_ints_are_not_misdetected_as_other_base(self):
        # "0b1" in base 16 would otherwise be 0xb1
        assert parse.parse("{:x}", "0b1")[0] == 0xB1

    def test_equals_align_fill(self):
        assert parse.parse("{:=5d}", "-0042")[0] == -42
        assert parse.parse("{:x=5d}", "xxx42")[0] == 42

    def test_sign_in_format_accepted(self):
        assert parse.parse("{:+d}", "+5")[0] == 5
        assert parse.parse("{:-d}", "-5")[0] == -5
        assert parse.parse("{: d}", " 5")[0] == 5


class TestIntConvert:
    def test_auto_base(self):
        c = int_convert()
        assert c("0x10", None) == 16
        assert c("0b10", None) == 2
        assert c("0o10", None) == 8
        assert c("10", None) == 10
        assert c("-0x10", None) == -16
        assert c("+0b11", None) == 3

    def test_short_zero_prefixed_not_base(self):
        c = int_convert()
        # "0x" alone is too short to be base-prefixed; "00" is decimal
        assert c("00", None) == 0
        assert c("09", None) == 9

    def test_explicit_base_strips_junk(self):
        assert int_convert(10)("1,234", None) == 1234
        assert int_convert(16)("0xff", None) == 255
        assert int_convert(2)("0b11", None) == 3
        assert int_convert(8)("0o17", None) == 15
        assert int_convert(16)("FF", None) == 255


# ---------------------------------------------------------------------------
# Floats & friends
# ---------------------------------------------------------------------------


class TestFloats:
    def test_f(self):
        r = parse.parse("{:f}", "3.25")
        assert r[0] == 3.25
        assert type(r[0]) is float

    def test_f_negative_and_positive(self):
        assert parse.parse("{:f}", "-3.5")[0] == -3.5
        assert parse.parse("{:f}", "+3.5")[0] == 3.5
        assert parse.parse("{:f}", ".5")[0] == 0.5

    def test_f_requires_decimal_point(self):
        assert parse.parse("{:f}", "3") is None

    def test_f_precision_zero(self):
        assert parse.parse("{:.0f}", "20")[0] == 20.0
        assert parse.parse("{:.0f}", "-20")[0] == -20.0

    def test_f_nan_inf(self):
        import math

        assert math.isnan(parse.parse("{:f}", "nan")[0])
        assert parse.parse("{:f}", "inf")[0] == float("inf")
        assert parse.parse("{:f}", "-inf")[0] == float("-inf")

    def test_F_is_decimal(self):
        r = parse.parse("{:F}", "1.10")
        assert r[0] == Decimal("1.10")
        assert isinstance(r[0], Decimal)

    def test_f_with_width_and_precision(self):
        assert parse.parse("{:8.2f}", "   12.34")[0] == 12.34

    def test_f_fixed_fields_after_float(self):
        r = parse.parse("{:f} {:d}", "1.5 7")
        assert r.fixed == (1.5, 7)

    def test_e(self):
        assert parse.parse("{:e}", "1.5e10")[0] == 1.5e10
        assert parse.parse("{:e}", "1.5E-3")[0] == 1.5e-3
        assert parse.parse("{:e}", "-1.0e+2")[0] == -100.0
        assert parse.parse("{:E}", "2.0E2")[0] == 200.0
        assert parse.parse("{:e}", "15") is None

    def test_g(self):
        assert parse.parse("{:g}", "1")[0] == 1.0
        assert parse.parse("{:g}", "1.5")[0] == 1.5
        assert parse.parse("{:g}", "1e5")[0] == 1e5
        assert parse.parse("{:g}", "-1.5e-2")[0] == -0.015
        assert parse.parse("{:G}", "2E3")[0] == 2000.0

    def test_g_followed_by_more_fields(self):
        # g consumes extra regex groups; subsequent fixed fields must still align
        r = parse.parse("{:g} {} {:d}", "1.5e3 hi 4")
        assert r.fixed == (1500.0, "hi", 4)

    def test_percentage(self):
        assert parse.parse("{:%}", "50%")[0] == 0.5
        assert parse.parse("{:%}", "12.5%")[0] == 0.125
        assert parse.parse("{:%}", "50") is None

    def test_percentage_function(self):
        assert parse.percentage("25%", None) == 0.25


# ---------------------------------------------------------------------------
# String-ish types
# ---------------------------------------------------------------------------


class TestStringTypes:
    def test_w(self):
        assert parse.parse("{:w} x", "abc_1 x")[0] == "abc_1"
        assert parse.parse("{:w}", "a b") is None

    def test_W(self):
        assert parse.parse("{:W}", " .!")[0] == " .!"
        assert parse.parse("{:W}", "a") is None

    def test_s(self):
        assert parse.parse("{:s}", " \t\n")[0] == " \t\n"
        assert parse.parse("{:s}", "a") is None

    def test_S(self):
        assert parse.parse("{:S}", "abc")[0] == "abc"
        assert parse.parse("{:S}", "a b") is None

    def test_D(self):
        assert parse.parse("{:D}", "abc")[0] == "abc"
        assert parse.parse("{:D}", "a1") is None

    def test_l(self):
        assert parse.parse("{:l}", "Hello")[0] == "Hello"
        assert parse.parse("{:l}", "abc1") is None

    def test_precision_limits_length(self):
        r = parse.parse("{:.2}{}", "abcd")
        assert r.fixed == ("a", "bcd")
        assert parse.parse("{:.2}", "abc") is None

    def test_width_minimum_length(self):
        r = parse.parse("{:3}{}", "abcd")
        assert r.fixed == ("abc", "d")
        assert parse.parse("{:3}", "ab") is None

    def test_width_and_precision(self):
        r = parse.parse("{:2.3}", "abc")
        assert r[0] == "abc"
        assert parse.parse("{:2.3}", "abcd") is None
        assert parse.parse("{:2.3}", "a") is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            Parser("{:q}")
        with pytest.raises(ValueError):
            parse.parse("{:zz}", "a")


# ---------------------------------------------------------------------------
# Alignment & fill
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_left(self):
        assert parse.parse("{:<}|", "ab   |")[0] == "ab"

    def test_right(self):
        assert parse.parse("|{:>}", "|   ab")[0] == "ab"

    def test_center(self):
        assert parse.parse("|{:^}|", "|  ab  |")[0] == "ab"

    def test_width_implies_right_align(self):
        assert parse.parse("{:2}", "   ab")[0] == "ab"

    def test_custom_fill(self):
        assert parse.parse("{:x<}|", "abxxx|")[0] == "ab"
        assert parse.parse("|{:*^}|", "|**ab**|")[0] == "ab"

    def test_regex_special_fill(self):
        assert parse.parse("|{:.^}|", "|..ab..|")[0] == "ab"
        assert parse.parse("{:+>d}", "++5")[0] == 5

    def test_numeric_center(self):
        assert parse.parse("|{:^d}|", "| 42 |")[0] == 42

    def test_numeric_left(self):
        assert parse.parse("{:<d}|", "42  |")[0] == 42


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
        assert f["width"] == ""
        assert f["zero"] is False

    def test_underscore_grouping(self):
        assert extract_format("_d", {})["grouping"] == "_"

    def test_extra_type_allowed(self):
        assert extract_format("custom", {"custom": lambda s: s})["type"] == "custom"

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            extract_format("Q", {})


# ---------------------------------------------------------------------------
# Datetime types
# ---------------------------------------------------------------------------


class TestDatetimes:
    def test_ti_date_only(self):
        assert parse.parse("{:ti}", "2011-05-04")[0] == datetime(2011, 5, 4)

    def test_ti_full(self):
        r = parse.parse("{:ti}", "2011-05-04T12:34:56.123Z")[0]
        assert r == datetime(2011, 5, 4, 12, 34, 56, 123000, tzinfo=FixedTzOffset(0, "UTC"))
        assert r.utcoffset() == timedelta(0)

    def test_ti_offset(self):
        r = parse.parse("{:ti}", "2011-05-04 10:00:00+05:30")[0]
        assert r.utcoffset() == timedelta(hours=5, minutes=30)
        assert (r.hour, r.minute) == (10, 0)

    def test_ti_negative_offset_no_colon(self):
        r = parse.parse("{:ti}", "2011-05-04 10:00-0130")[0]
        assert r.utcoffset() == -timedelta(hours=1, minutes=30)

    def test_ti_hour_minute_only(self):
        r = parse.parse("{:ti}", "2011-05-04 7:8")[0]
        assert r == datetime(2011, 5, 4, 7, 8)

    def test_ti_followed_by_field(self):
        r = parse.parse("{:ti} {:d}", "2011-05-04T01:02:03 9")
        assert r.fixed == (datetime(2011, 5, 4, 1, 2, 3), 9)

    def test_te(self):
        r = parse.parse("{:te}", "Mon, 21 Nov 2011 10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
        r = parse.parse("{:te}", "21 Nov 2011 10:21:36 -0500")[0]
        assert r.utcoffset() == -timedelta(hours=5)
        assert r.day == 21 and r.month == 11

    def test_tg(self):
        r = parse.parse("{:tg}", "20/11/2011 10:21:36 PM")[0]
        assert r == datetime(2011, 11, 20, 22, 21, 36)
        r = parse.parse("{:tg}", "20-Nov-2011")[0]
        assert r == datetime(2011, 11, 20)
        r = parse.parse("{:tg}", "1/2/2011 12:00 AM +0100")[0]
        assert r.hour == 0
        assert r.month == 2 and r.day == 1
        assert r.utcoffset() == timedelta(hours=1)

    def test_ta(self):
        r = parse.parse("{:ta}", "11/20/2011 10:21:36 AM")[0]
        assert r == datetime(2011, 11, 20, 10, 21, 36)
        r = parse.parse("{:ta}", "January-21-2011 12:30 PM")[0]
        assert r == datetime(2011, 1, 21, 12, 30)

    def test_th(self):
        r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
        assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))

    def test_tc(self):
        r = parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0]
        assert r == datetime(1973, 9, 16, 1, 3, 52)

    def test_tt(self):
        assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
        assert parse.parse("{:tt}", "10:21:36 PM")[0] == time(22, 21, 36)
        assert parse.parse("{:tt}", "12:05 AM")[0] == time(0, 5)
        assert parse.parse("{:tt}", "12:05 PM")[0] == time(12, 5)
        r = parse.parse("{:tt}", "10:21:36.5 -05:00")[0]
        assert r.microsecond == 500000
        assert r.utcoffset() == -timedelta(hours=5)

    def test_ts(self):
        r = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
        assert (r.month, r.day, r.hour, r.minute, r.second) == (11, 9, 3, 37, 44)
        assert r.year == datetime.today().year

    def test_invalid_date_rejected_by_regex(self):
        assert parse.parse("{:ti}", "not a date") is None

    def test_strftime_date(self):
        r = parse.parse("{:%Y-%m-%d}", "2023-01-15")[0]
        assert r == date(2023, 1, 15)
        assert type(r) is date

    def test_strftime_datetime(self):
        r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-01-15 08:09:10")[0]
        assert r == datetime(2023, 1, 15, 8, 9, 10)

    def test_strftime_time(self):
        r = parse.parse("{:%H:%M}", "08:09")[0]
        assert r == time(8, 9)
        assert type(r) is time

    def test_strftime_no_year_uses_current_year(self):
        r = parse.parse("{:%b %d}", "Feb 03")[0]
        assert r == date(datetime.today().year, 2, 3)

    def test_strftime_month_names(self):
        r = parse.parse("{:%A %B %d %Y}", "Monday January 02 2023")[0]
        assert r == date(2023, 1, 2)

    def test_strftime_pm_and_tz(self):
        r = parse.parse("{:%I:%M %p %z}", "01:30 PM +0200")[0]
        assert (r.hour, r.minute) == (13, 30)
        assert r.utcoffset() == timedelta(hours=2)

    def test_strf_date_convert_neither(self):
        with pytest.raises(ValueError):
            parse.strf_date_convert("x", None, "x")

    def test_get_regex_for_datetime_format(self):
        assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"


class TestDateConvert:
    def test_fixed_tz_offset(self):
        tz = FixedTzOffset(90, "+0130")
        assert tz.utcoffset(None) == timedelta(minutes=90)
        assert tz.tzname(None) == "+0130"
        assert tz.dst(None) == timedelta(0)
        assert tz == FixedTzOffset(90, "+0130")
        assert tz != FixedTzOffset(60, "+0130")
        assert tz != FixedTzOffset(90, "other")
        assert (tz == "x") is False
        assert "FixedTzOffset" in repr(tz)
        assert "+0130" in repr(tz)

    def test_tz_three_digit_hour(self):
        # 'snnn' form e.g. "+130" -> 1h30m
        r = parse.parse("{:tt}", "10:00 +130")[0]
        assert r.utcoffset() == timedelta(hours=1, minutes=30)


# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------


class TestCustomTypes:
    def test_simple_converter(self):
        r = parse.parse("{:upper}", "abc", extra_types={"upper": str.upper})
        assert r[0] == "ABC"

    def test_with_pattern(self):
        @with_pattern(r"\d+")
        def num(text):
            return int(text)

        assert num.pattern == r"\d+"
        assert num.regex_group_count is None
        r = parse.parse("{:Num}{}", "123abc", extra_types={"Num": num})
        assert r.fixed == (123, "abc")
        assert parse.parse("{:Num}", "x", extra_types={"Num": num}) is None

    def test_with_pattern_groups(self):
        @with_pattern(r"(yes|no)", regex_group_count=1)
        def yesno(text):
            return text == "yes"

        r = parse.parse("{:YN} {:d}", "yes 3", extra_types={"YN": yesno})
        assert r.fixed == (True, 3)

    def test_with_pattern_named(self):
        @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
        def ab(text):
            return text

        r = parse.parse("{x:AB} {y:d}", "ac 4", extra_types={"AB": ab})
        assert r.named == {"x": "ac", "y": 4}

    def test_custom_overrides_builtin(self):
        r = parse.parse("{:d}", "abc", extra_types={"d": lambda s: s * 2})
        assert r[0] == "abcabc"

    def test_compile_with_extra_types(self):
        p = compile("{:my}", extra_types={"my": len})
        assert p.parse("hello")[0] == 5


# ---------------------------------------------------------------------------
# search / findall / spans / Match objects
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_substring(self):
        r = search("age: {:d}", "name: x age: 42 more")
        assert r[0] == 42

    def test_search_none(self):
        assert search("age: {:d}", "nothing") is None

    def test_search_pos(self):
        r = search("{:d}", "1 2 3", pos=1)
        assert r[0] == 2

    def test_search_endpos(self):
        assert search("{:d}x", "12x", endpos=2) is None
        assert search("{:d}", "12345", endpos=3)[0] == 123

    def test_parser_search(self):
        p = compile("{:d}")
        assert p.search("ab 7 cd")[0] == 7
        assert p.search("ab 7 cd 9", pos=4)[0] == 9

    def test_search_evaluate_result_false(self):
        m = search("{:d}", "a 5", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.evaluate_result()[0] == 5


class TestFindall:
    def test_findall(self):
        rs = list(findall(">{}<", "<p>a<b>bc<i>d<"))
        assert [r[0] for r in rs] == ["a", "bc", "d"]

    def test_findall_typed(self):
        assert [r[0] for r in findall("{:d}", "a1b22c333")] == [1, 22, 333]

    def test_findall_returns_iterator(self):
        it = findall("{:d}", "1 2")
        assert isinstance(it, ResultIterator)
        assert iter(it) is it
        assert next(it)[0] == 1
        assert it.next()[0] == 2
        with pytest.raises(StopIteration):
            next(it)

    def test_findall_pos_endpos(self):
        assert [r[0] for r in findall("{:d}", "1 2 3 4", pos=2, endpos=5)] == [2, 3]

    def test_findall_evaluate_result_false(self):
        ms = list(findall("{:d}", "1 2", evaluate_result=False))
        assert all(isinstance(m, Match) for m in ms)
        assert [m.evaluate_result()[0] for m in ms] == [1, 2]

    def test_findall_empty(self):
        assert list(findall("{:d}", "abc")) == []

    def test_parser_findall(self):
        p = compile("<{}>")
        assert [r[0] for r in p.findall("<a> <b>")] == ["a", "b"]
        assert [r[0] for r in p.findall("<a> <b> <c>", pos=1, endpos=7)] == ["b"]


class TestSpansAndResult:
    def test_spans(self):
        r = parse.parse("{} {name} {}", "ab cde f")
        assert r.spans[0] == (0, 2)
        assert r.spans["name"] == (3, 6)
        assert r.spans[1] == (7, 8)

    def test_spans_with_typed_fields(self):
        r = parse.parse("{:g} {:d}", "1.5 22")
        assert r.spans[0] == (0, 3)
        assert r.spans[1] == (4, 6)

    def test_spans_search(self):
        r = search("x{:d}", "aaax12")
        assert r.spans[0] == (4, 6)

    def test_result_repr(self):
        r = Result(("a",), {"b": 1}, {})
        assert repr(r) == "<Result ('a',) {'b': 1}>"

    def test_parse_evaluate_result_false(self):
        m = parse.parse("{:d}", "5", evaluate_result=False)
        assert isinstance(m, Match)
        assert m.parser.format == "{:d}"
        assert m.match.group(0) == "5"
        r = m.evaluate_result()
        assert r[0] == 5

    def test_parse_evaluate_result_false_no_match(self):
        assert parse.parse("{:d}", "x", evaluate_result=False) is None


class TestParserMisc:
    def test_repr_short(self):
        assert repr(compile("{a}")) == "<Parser '{a}'>"

    def test_repr_long_truncated(self):
        fmt = "a" * 25
        assert repr(compile(fmt)) == "<Parser %r>" % ("a" * 17 + "...")

    def test_repr_exactly_20(self):
        fmt = "b" * 20
        assert repr(compile(fmt)) == "<Parser %r>" % fmt

    def test_parser_reuse(self):
        p = compile("{:d}+{:d}")
        assert p.parse("1+2").fixed == (1, 2)
        assert p.parse("3+4").fixed == (3, 4)
        assert p.parse("x") is None

    def test_pickle_parser_result(self):
        p = compile("{a:d} {b:x} {c:f}")
        p2 = pickle.loads(pickle.dumps(p))
        assert p2.parse("1 ff 1.5").named == {"a": 1, "b": 255, "c": 1.5}

    def test_many_fields(self):
        fmt = " ".join("{}" for _ in range(50))
        s = " ".join(str(i) for i in range(50))
        assert parse.parse(fmt, s).fixed == tuple(str(i) for i in range(50))

    def test_all_exports(self):
        assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
