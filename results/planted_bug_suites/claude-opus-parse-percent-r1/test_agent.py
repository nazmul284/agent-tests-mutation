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


# ---------------------------------------------------------------- basics


def test_no_fields_exact_match():
    r = parse.parse("hello", "hello")
    assert r.fixed == ()
    assert r.named == {}


def test_no_fields_mismatch():
    assert parse.parse("hello", "goodbye") is None


def test_parse_must_match_whole_string():
    assert parse.parse("hello {}", "say hello world") is None
    assert parse.parse("{} world", "hello world!") is None


def test_fixed_fields():
    r = parse.parse("{} {}", "a b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a"
    assert r[1] == "b"
    assert r[0:2] == ("a", "b")
    assert r[-1] == "b"


def test_named_fields():
    r = parse.parse("{greeting} {name}", "hello world")
    assert r.named == {"greeting": "hello", "name": "world"}
    assert r["name"] == "world"
    assert "name" in r
    assert "missing" not in r


def test_mixed_fixed_and_named():
    r = parse.parse("{} {name} {}", "a b c")
    assert r.fixed == ("a", "c")
    assert r.named == {"name": "b"}


def test_escaped_braces():
    r = parse.parse("{{hello}} {}", "{hello} world")
    assert r.fixed == ("world",)
    assert parse.parse("{{}}", "{}").fixed == ()
    assert parse.parse("{{}}", "x") is None


def test_regex_special_chars_in_literal_text():
    r = parse.parse("a.b*c+d?(e)[f]^$|\\ {}", "a.b*c+d?(e)[f]^$|\\ x")
    assert r.fixed == ("x",)
    # the "." must be literal, not wildcard
    assert parse.parse("a.b {}", "axb x") is None
    assert parse.parse("(x) {}", "x y") is None


def test_case_insensitive_by_default():
    assert parse.parse("Hello {}", "hELLO world").fixed == ("world",)


def test_case_sensitive():
    assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello world", case_sensitive=True).fixed == (
        "world",
    )
    p = compile("Hello {}", case_sensitive=True)
    assert p.parse("hello x") is None


def test_dotall_newlines():
    r = parse.parse("{}:{}", "a\nb:c\nd")
    assert r.fixed == ("a\nb", "c\nd")


def test_non_greedy_default_field():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_unicode():
    r = parse.parse("{} {}", "héllo wörld")
    assert r.fixed == ("héllo", "wörld")


# ---------------------------------------------------------------- dotted / indexed names


def test_dotted_name():
    r = parse.parse("{hello.world}", "x")
    assert r.named == {"hello.world": "x"}
    assert r["hello.world"] == "x"


def test_dotted_and_underscore_no_collision():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}


def test_underscore_collision_variants():
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_hyphenated_name():
    r = parse.parse("{a-b}", "x")
    assert r.named == {"a-b": "x"}


def test_hyphen_and_underscore_no_collision():
    r = parse.parse("{a_b} {a-b}", "1 2")
    assert r.named == {"a_b": "1", "a-b": "2"}
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_indexed_names_become_nested_dict():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    assert r["a"]["b"] == "1"


def test_deeply_nested_names():
    r = parse.parse("{a[b][c]} {a[b][d]} {x}", "1 2 3")
    assert r.named == {"a": {"b": {"c": "1", "d": "2"}}, "x": "3"}


def test_indexed_name_spans_use_original_name():
    r = parse.parse("{a[b]}", "xyz")
    assert r.spans == {"a[b]": (0, 3)}


def test_numeric_name_is_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r.named == {}


# ---------------------------------------------------------------- repeated names


def test_repeated_name_must_match_same_text():
    r = parse.parse("{x} {x}", "a a")
    assert r.named == {"x": "a"}
    assert parse.parse("{x} {x}", "a b") is None


def test_repeated_name_typed():
    r = parse.parse("{x:d}-{x:d}", "3-3")
    assert r["x"] == 3
    assert parse.parse("{x:d}-{x:d}", "3-4") is None


def test_repeated_name_type_mismatch_raises():
    with pytest.raises(RepeatedNameError):
        parse.compile("{x:d} {x:w}")
    with pytest.raises(ValueError):
        parse.compile("{x} {x:d}")


# ---------------------------------------------------------------- spans


def test_spans():
    r = parse.parse("{} {name} {}", "aa bbb c")
    assert r.spans[0] == (0, 2)
    assert r.spans["name"] == (3, 6)
    assert r.spans[1] == (7, 8)


def test_spans_with_typed_groups():
    # extra regex groups from type patterns must not shift fixed-field spans
    r = parse.parse("{:g} {:x} {}", "1.5e3 ff end")
    assert r.fixed == (1500.0, 255, "end")
    assert r.spans[0] == (0, 5)
    assert r.spans[1] == (6, 8)
    assert r.spans[2] == (9, 12)


def test_search_spans_are_absolute():
    r = search("age: {:d}", "name: x age: 42 more")
    assert r.spans[0] == (13, 15)


# ---------------------------------------------------------------- integer types


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("42", 42),
        ("-42", -42),
        ("+42", 42),
        ("007", 7),
        ("0x1F", 31),
        ("0X1f", 31),
        ("0b101", 5),
        ("0B11", 3),
        ("0o17", 15),
        ("0O17", 15),
        ("-0x10", -16),
        ("+0b11", 3),
    ],
)
def test_d_type(text, expected):
    r = parse.parse("{:d}", text)
    assert r[0] == expected
    assert type(r[0]) is int


def test_d_type_space_sign():
    assert parse.parse("{:d}", " 5")[0] == 5


def test_d_rejects_non_numbers():
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "1.5") is None
    assert parse.parse("{:d}", "") is None


def test_d_with_grouping():
    assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
    assert parse.parse("{:_d}", "1_000")[0] == 1000
    assert parse.parse("{:d}", "1,000") is None


def test_d_with_width_limits_digits():
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)


def test_d_width_padding():
    assert parse.parse("{:5d}", "   42")[0] == 42
    assert parse.parse("{:<5d}", "42   ")[0] == 42
    assert parse.parse("{:^6d}", "  42  ")[0] == 42


def test_d_zero_padding():
    assert parse.parse("{:05d}", "00042")[0] == 42


def test_d_equals_alignment():
    assert parse.parse("{:=5d}", "00042")[0] == 42
    assert parse.parse("{:x=5d}", "xxx42")[0] == 42
    assert parse.parse("{:x=5d}", "-xx42") is not None


def test_n_type():
    assert parse.parse("{:n}", "1,234,567")[0] == 1234567
    assert parse.parse("{:n}", "1.234.567")[0] == 1234567
    assert parse.parse("{:n}", "123")[0] == 123
    assert parse.parse("{:n}", "-1,000")[0] == -1000
    assert parse.parse("{:n}", "1,23") is None


def test_b_type():
    assert parse.parse("{:b}", "1010")[0] == 10
    assert parse.parse("{:b}", "0b1010")[0] == 10
    assert parse.parse("{:b}", "-0b11")[0] == -3
    assert parse.parse("{:b}", "102") is None


def test_o_type():
    assert parse.parse("{:o}", "17")[0] == 15
    assert parse.parse("{:o}", "0o17")[0] == 15
    assert parse.parse("{:o}", "8") is None


def test_x_type():
    assert parse.parse("{:x}", "ff")[0] == 255
    assert parse.parse("{:x}", "0xFF")[0] == 255
    assert parse.parse("{:X}", "DEADBEEF")[0] == 0xDEADBEEF
    assert parse.parse("{:x}", "-a")[0] == -10
    assert parse.parse("{:x}", "g") is None


def test_x_type_explicit_base_ignores_0b_prefix_lookalike():
    # "0b1" in hex is 0xb1, not binary
    assert parse.parse("{:x}", "0b1")[0] == 0xB1


# ---------------------------------------------------------------- float types


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.5", 1.5),
        (".5", 0.5),
        ("-1.5", -1.5),
        ("+1.5", 1.5),
        ("0.000", 0.0),
    ],
)
def test_f_type(text, expected):
    r = parse.parse("{:f}", text)
    assert r[0] == expected
    assert type(r[0]) is float


def test_f_requires_decimal_point():
    assert parse.parse("{:f}", "15") is None


def test_f_precision_zero_accepts_integers():
    assert parse.parse("{:.0f}", "20")[0] == 20.0
    assert parse.parse("{:.0f}", "-20")[0] == -20.0


def test_f_nan_inf():
    import math

    assert math.isnan(parse.parse("{:f}", "nan")[0])
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert parse.parse("{:f}", "-inf")[0] == float("-inf")
    assert parse.parse("{:f}", "-INF")[0] == float("-inf")


def test_F_type_is_decimal():
    r = parse.parse("{:F}", "1.25")
    assert r[0] == Decimal("1.25")
    assert isinstance(r[0], Decimal)
    assert parse.parse("{:F}", "-0.1")[0] == Decimal("-0.1")


def test_e_type():
    assert parse.parse("{:e}", "1.5e3")[0] == 1500.0
    assert parse.parse("{:e}", "1.5E-3")[0] == 0.0015
    assert parse.parse("{:e}", "-1.0e+2")[0] == -100.0
    assert parse.parse("{:E}", "2.0E2")[0] == 200.0
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "inf")[0] == float("inf")


def test_g_type():
    assert parse.parse("{:g}", "1")[0] == 1.0
    assert parse.parse("{:g}", "1.5")[0] == 1.5
    assert parse.parse("{:g}", "1.5e3")[0] == 1500.0
    assert parse.parse("{:g}", "-2e-2")[0] == -0.02
    assert parse.parse("{:G}", "1E2")[0] == 100.0
    assert type(parse.parse("{:g}", "1")[0]) is float
    assert parse.parse("{:g}", "abc") is None


def test_g_followed_by_fields_keeps_group_alignment():
    r = parse.parse("{:g} {:d} {name:g} {}", "1.5 2 3e1 tail")
    assert r.fixed == (1.5, 2, "tail")
    assert r["name"] == 30.0


# ---------------------------------------------------------------- percentage


@pytest.mark.parametrize(
    "text,expected",
    [
        ("50%", 0.5),
        ("100%", 1.0),
        ("0%", 0.0),
        ("12.5%", 0.125),
        ("200%", 2.0),
        ("-25%", -0.25),
    ],
)
def test_percentage_type(text, expected):
    assert parse.parse("{:%}", text)[0] == pytest.approx(expected)


def test_percentage_requires_percent_sign():
    assert parse.parse("{:%}", "50") is None


def test_percentage_function_directly():
    assert parse.percentage("50%", None) == pytest.approx(0.5)


def test_percentage_with_following_field():
    r = parse.parse("{:%} {:d}", "10.5% 3")
    assert r.fixed == (pytest.approx(0.105), 3)


# ---------------------------------------------------------------- string types


def test_w_type():
    assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
    assert parse.parse("{:w}", "abc def") is None


def test_W_type():
    assert parse.parse("{:W}", " -!")[0] == " -!"
    assert parse.parse("{:W}", "a") is None


def test_s_and_S_type():
    assert parse.parse("{:s}", " \t ")[0] == " \t "
    assert parse.parse("{:s}", "x") is None
    assert parse.parse("{:S}", "abc")[0] == "abc"
    assert parse.parse("{:S}", "a b") is None


def test_D_type():
    assert parse.parse("{:D}", "abc")[0] == "abc"
    assert parse.parse("{:D}", "a1") is None


def test_l_type():
    assert parse.parse("{:l}", "Hello")[0] == "Hello"
    assert parse.parse("{:l}", "Hello1") is None


def test_string_alignment_strips_padding():
    assert parse.parse("{:>}", "   ab")[0] == "ab"
    assert parse.parse("{:<}|", "ab   |")[0] == "ab"
    assert parse.parse("{:^}|", "  ab   |")[0] == "ab"
    assert parse.parse("{:*^}|", "**ab***|")[0] == "ab"
    assert parse.parse("{:.<}|", "ab...|")[0] == "ab"
    assert parse.parse("{:x>}", "xxab")[0] == "ab"


def test_width_counts_padding():
    # the width is a minimum field length, padding included
    assert parse.parse("{:>5}", "   ab")[0] == "   ab"
    assert parse.parse("{:4}", "  ab")[0] == "  ab"
    assert parse.parse("{:>5}", "ab") is None


def test_string_min_width():
    assert parse.parse("[{:4}]", "[abc]") is None
    assert parse.parse("[{:4}]", "[abcd]")[0] == "abcd"


def test_string_precision_max_width():
    assert parse.parse("{:.2}{}", "abcd").fixed == ("a", "bcd")
    assert parse.parse("[{:.2}]", "[abc]") is None
    assert parse.parse("[{:.2}]", "[ab]")[0] == "ab"


def test_string_width_and_precision():
    assert parse.parse("[{:2.4}]", "[a]") is None
    assert parse.parse("[{:2.4}]", "[abcd]")[0] == "abcd"
    assert parse.parse("[{:2.4}]", "[abcde]") is None


def test_regex_meta_fill_chars():
    # "^" is excluded: a leading "^" is read as the align char, not fill
    for fill in ".+?*[](){}$\\":
        if fill in "{}":
            continue
        fmt = "{:%s<}|" % fill
        assert parse.parse(fmt, "ab" + fill * 2 + "|")[0] == "ab", fill


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        parse.compile("{:zz}")
    with pytest.raises(ValueError):
        parse.compile("{:q}")


# ---------------------------------------------------------------- extract_format


def test_extract_format_parts():
    f = extract_format("x<+010,.3d", {})
    assert f["fill"] == "x"
    assert f["align"] == "<"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "d"


def test_extract_format_minimal():
    f = extract_format("d", {})
    assert f["fill"] is None
    assert f["align"] is None
    assert f["zero"] is False
    assert f["width"] == ""
    assert f["type"] == "d"


def test_extract_format_align_only():
    f = extract_format(">", {})
    assert f["align"] == ">"
    assert f["fill"] is None
    assert f["type"] == ""


def test_extract_format_extra_type():
    f = extract_format("Number", {"Number": None})
    assert f["type"] == "Number"


# ---------------------------------------------------------------- datetime types


def test_ti_iso():
    assert parse.parse("{:ti}", "2023-04-05")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:ti}", "2023-04-05T10:11")[0] == datetime(
        2023, 4, 5, 10, 11
    )
    assert parse.parse("{:ti}", "2023-04-05 10:11:12")[0] == datetime(
        2023, 4, 5, 10, 11, 12
    )
    assert parse.parse("{:ti}", "2023-04-05T10:11:12.5")[0] == datetime(
        2023, 4, 5, 10, 11, 12, 500000
    )
    assert parse.parse("{:ti}", "2023-04-05T10:11:12.123456789")[0] == datetime(
        2023, 4, 5, 10, 11, 12, 123456
    )


def test_ti_timezones():
    r = parse.parse("{:ti}", "2023-04-05T10:11:12Z")[0]
    assert r.utcoffset() == timedelta(0)
    r = parse.parse("{:ti}", "2023-04-05T10:11:12+05:30")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2023-04-05T10:11:12-0800")[0]
    assert r.utcoffset() == timedelta(hours=-8)
    assert r == datetime(2023, 4, 5, 18, 11, 12, tzinfo=FixedTzOffset(0, "UTC"))


def test_ti_followed_by_field():
    r = parse.parse("{:ti} {}", "2023-04-05T10:11:12Z rest")
    assert r[1] == "rest"


def test_tg_global():
    assert parse.parse("{:tg}", "20/1/1972 10:21:36 AM +1:00")[0] == datetime(
        1972, 1, 20, 10, 21, 36, tzinfo=FixedTzOffset(60, "+1:00")
    )
    assert parse.parse("{:tg}", "20/1/1972")[0] == datetime(1972, 1, 20)
    assert parse.parse("{:tg}", "20/1/1972 10:21 PM")[0] == datetime(
        1972, 1, 20, 22, 21
    )
    assert parse.parse("{:tg}", "20-Jan-1972")[0] == datetime(1972, 1, 20)
    assert parse.parse("{:tg}", "20/January/1972")[0] == datetime(1972, 1, 20)


def test_ta_american():
    assert parse.parse("{:ta}", "1/20/1972 10:21:36 PM")[0] == datetime(
        1972, 1, 20, 22, 21, 36
    )
    assert parse.parse("{:ta}", "Jan-20-1972")[0] == datetime(1972, 1, 20)


def test_am_pm_noon_midnight():
    assert parse.parse("{:ta}", "1/20/1972 12:15 AM")[0] == datetime(
        1972, 1, 20, 0, 15
    )
    assert parse.parse("{:ta}", "1/20/1972 12:15 PM")[0] == datetime(
        1972, 1, 20, 12, 15
    )
    assert parse.parse("{:ta}", "1/20/1972 1:15 PM")[0] == datetime(
        1972, 1, 20, 13, 15
    )
    assert parse.parse("{:ta}", "1/20/1972 1:15 AM")[0] == datetime(
        1972, 1, 20, 1, 15
    )


def test_te_email():
    r = parse.parse("{:te}", "Mon, 20 Jan 1972 10:21:36 +1000")[0]
    assert r == datetime(1972, 1, 20, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
    r = parse.parse("{:te}", "20 Jan 1972 10:21:36 -0130")[0]
    assert r.utcoffset() == -timedelta(hours=1, minutes=30)


def test_th_http():
    r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
    assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))


def test_tc_ctime():
    assert parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0] == datetime(
        1973, 9, 16, 1, 3, 52
    )


def test_tt_time():
    assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
    assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
    assert parse.parse("{:tt}", "10:21:36 PM")[0] == time(22, 21, 36)
    r = parse.parse("{:tt}", "10:21:36 PM -5:30")[0]
    assert r == time(22, 21, 36, tzinfo=FixedTzOffset(-330, "-5:30"))


def test_ts_syslog():
    year = datetime.today().year
    assert parse.parse("{:ts}", "Nov 21 10:21:36")[0] == datetime(
        year, 11, 21, 10, 21, 36
    )


def test_tz_without_colon_three_digits():
    r = parse.parse("{:tt}", "10:00 +130")[0]
    assert r.utcoffset() == timedelta(hours=1, minutes=30)


def test_datetime_bad_date_raises():
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2023-13-45")


# ---------------------------------------------------------------- strftime formats


def test_strftime_date():
    r = parse.parse("{:%Y-%m-%d}", "2023-04-05")[0]
    assert r == date(2023, 4, 5)
    assert type(r) is date


def test_strftime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 10:11:12")[0]
    assert r == datetime(2023, 4, 5, 10, 11, 12)


def test_strftime_time_only():
    r = parse.parse("{:%H:%M}", "10:11")[0]
    assert r == time(10, 11)
    assert type(r) is time


def test_strftime_no_year_uses_current_year():
    r = parse.parse("{:%b %d}", "Apr 05")[0]
    assert r == date(datetime.today().year, 4, 5)


def test_strftime_names_and_fraction():
    r = parse.parse("{:%A %B %d %Y %I:%M:%S.%f %p}", "Wednesday April 05 2023 01:02:03.25 PM")[0]
    assert r == datetime(2023, 4, 5, 13, 2, 3, 250000)


def test_strftime_tz():
    r = parse.parse("{:%Y-%m-%d %H:%M %z}", "2023-04-05 10:11 +0200")[0]
    assert r.utcoffset() == timedelta(hours=2)


def test_strftime_mismatch():
    assert parse.parse("{:%Y-%m-%d}", "2023/04/05") is None


def test_strf_date_convert_neither_date_nor_time():
    with pytest.raises(ValueError):
        parse.strf_date_convert("x", None, "x")


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"


# ---------------------------------------------------------------- FixedTzOffset


def test_fixed_tz_offset():
    tz = FixedTzOffset(-90, "-01:30")
    assert tz.utcoffset(None) == timedelta(minutes=-90)
    assert tz.tzname(None) == "-01:30"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedTzOffset -01:30 %s>" % timedelta(minutes=-90)


def test_fixed_tz_offset_equality():
    assert FixedTzOffset(60, "a") == FixedTzOffset(60, "a")
    assert FixedTzOffset(60, "a") != FixedTzOffset(60, "b")
    assert FixedTzOffset(60, "a") != FixedTzOffset(30, "a")
    assert FixedTzOffset(60, "a").__eq__(5) is NotImplemented


# ---------------------------------------------------------------- int_convert


def test_int_convert_auto_base():
    c = int_convert()
    assert c("123", None) == 123
    assert c("-0x1a", None) == -26
    assert c("0b11", None) == 3
    assert c("0o10", None) == 8
    assert c("0", None) == 0
    assert c("00", None) == 0
    assert c("1,234", None) == 1234


def test_int_convert_fixed_base():
    assert int_convert(16)("ff", None) == 255
    assert int_convert(2)("+101", None) == 5
    assert int_convert(8)("-17", None) == -15
    assert int_convert(36)("z", None) == 35


# ---------------------------------------------------------------- custom types


def test_extra_types_simple():
    def shout(s):
        return s.upper()

    r = parse.parse("{:Shout} {}", "hi there", extra_types={"Shout": shout})
    assert r.fixed == ("HI", "there")


def test_extra_types_with_pattern():
    @with_pattern(r"\d+")
    def number(s):
        return int(s)

    assert number.pattern == r"\d+"
    assert number.regex_group_count is None
    r = parse.parse("{:Num}{:Num}", "12ab", extra_types={"Num": number})
    assert r is None
    r = parse.parse("{n:Num}x", "12x", extra_types={"Num": number})
    assert r["n"] == 12


def test_extra_types_with_regex_groups():
    @with_pattern(r"((a)|(b))", regex_group_count=3)
    def ab(s):
        return s * 2

    r = parse.parse("{:AB} {:d} {:AB}", "a 5 b", extra_types={"AB": ab})
    assert r.fixed == ("aa", 5, "bb")
    assert r.spans[2] == (4, 5)


def test_extra_types_override_builtin():
    def custom_d(s):
        return "custom:" + s

    r = parse.parse("{:d}", "abc", extra_types={"d": custom_d})
    assert r[0] == "custom:abc"


def test_extra_types_named():
    @with_pattern(r"[a-z]+")
    def word(s):
        return s[::-1]

    r = parse.parse("{x:Word} {y:Word}", "abc def", extra_types={"Word": word})
    assert r.named == {"x": "cba", "y": "fed"}


# ---------------------------------------------------------------- search / findall


def test_search():
    r = search("age: {:d}", "name: x, age: 42, x")
    assert r[0] == 42
    assert search("age: {:d}", "no match here") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert search("a{:d}", s, pos=3)[0] == 2
    assert search("a{:d}", s, pos=3, endpos=4) is None
    assert search("a{:d}", s, endpos=2)[0] == 1


def test_search_evaluate_result_false():
    m = search("a{:d}", "xa5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.evaluate_result()[0] == 5


def test_findall():
    results = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
    assert [r[0] for r in results] == ["some ", "bold", " text"]


def test_findall_typed():
    assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]


def test_findall_pos_endpos():
    assert [r[0] for r in findall("x{:d}", "x1 x2 x3 x4", pos=3, endpos=8)] == [2, 3]


def test_findall_evaluate_result_false():
    ms = list(findall("x{:d}", "x1 x2", evaluate_result=False))
    assert all(isinstance(m, Match) for m in ms)
    assert [m.evaluate_result()[0] for m in ms] == [1, 2]


def test_findall_case_sensitive():
    assert [r[0] for r in findall("X{:d}", "x1 X2", case_sensitive=True)] == [2]
    assert [r[0] for r in findall("X{:d}", "x1 X2")] == [1, 2]


def test_findall_iterator_protocol():
    it = findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it)[0] == 1
    assert it.next()[0] == 2
    with pytest.raises(StopIteration):
        next(it)


# ---------------------------------------------------------------- Parser / compile


def test_compile_reuse():
    p = compile("{:d}+{:d}")
    assert isinstance(p, Parser)
    assert p.parse("1+2").fixed == (1, 2)
    assert p.parse("3+4").fixed == (3, 4)
    assert p.parse("x") is None
    assert p.search("sum 5+6 !").fixed == (5, 6)
    assert [r.fixed for r in p.findall("1+2 3+4")] == [(1, 2), (3, 4)]


def test_parser_properties():
    p = compile("{a} {} {b.c}")
    assert p.format == "{a} {} {b.c}"
    # fixed fields are recorded as regex group indices
    assert p.fixed_fields == [1]
    assert p.named_fields == ["a", "b_c"]
    # returned lists are copies
    p.named_fields.append("zzz")
    p.fixed_fields.append(99)
    assert p.named_fields == ["a", "b_c"]
    assert p.fixed_fields == [1]


def test_parser_repr():
    assert repr(compile("{}")) == "<Parser '{}'>"
    long_fmt = "x" * 25
    assert repr(compile(long_fmt)) == "<Parser %r>" % ("x" * 17 + "...")
    assert repr(compile("x" * 20)) == "<Parser %r>" % ("x" * 20)


def test_parse_evaluate_result_false():
    m = parse.parse("{:d}", "5", evaluate_result=False)
    assert isinstance(m, Match)
    r = m.evaluate_result()
    assert isinstance(r, Result)
    assert r[0] == 5
    assert parse.parse("{:d}", "x", evaluate_result=False) is None


def test_result_repr():
    r = parse.parse("{} {n}", "a b")
    assert repr(r) == "<Result ('a',) {'n': 'b'}>"


def test_result_getitem_missing_name():
    r = parse.parse("{n}", "a")
    with pytest.raises(KeyError):
        r["missing"]
    with pytest.raises(IndexError):
        r[0]


def test_too_many_fields_handled():
    # Python 3 supports many named groups; just ensure a big format works
    fmt = " ".join("{f%d}" % i for i in range(150))
    s = " ".join(str(i) for i in range(150))
    r = parse.parse(fmt, s)
    assert r["f0"] == "0"
    assert r["f149"] == "149"


def test_convert_first_is_picklable():
    c = parse.convert_first(int)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2("5", None) == 5


def test_parser_picklable_and_works():
    p = compile("{:d} {name:f}")
    p2 = pickle.loads(pickle.dumps(p))
    r = p2.parse("1 2.5")
    assert r[0] == 1
    assert r["name"] == 2.5


def test_all_exports():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
