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
    get_regex_for_datetime_format,
    int_convert,
    percentage,
    search,
    strf_date_convert,
    with_pattern,
)


def utc_tz(minutes, name):
    return FixedTzOffset(minutes, name)


# ---------------------------------------------------------------- basics


def test_module_metadata():
    assert parse.__all__ == ["parse", "search", "findall", "with_pattern"]
    assert parse.__version__ == "1.22.3"


def test_no_fields_exact_match():
    r = parse.parse("hello", "hello")
    assert r.fixed == ()
    assert r.named == {}
    assert r.spans == {}


def test_no_match_returns_none():
    assert parse.parse("hello", "goodbye") is None
    assert parse.parse("hello", "hello world") is None
    assert parse.parse("hello", "say hello") is None


def test_fixed_fields():
    r = parse.parse("{} {}", "a b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a"
    assert r[1] == "b"
    assert r[0:2] == ("a", "b")
    assert r[-1] == "b"


def test_fixed_spans():
    r = parse.parse("x{}y{}z", "xaaybbbz")
    assert r.spans == {0: (1, 3), 1: (4, 7)}


def test_named_fields_and_spans():
    r = parse.parse("{a} and {b}", "foo and barbaz")
    assert r.named == {"a": "foo", "b": "barbaz"}
    assert r["a"] == "foo"
    assert "a" in r
    assert "c" not in r
    assert r.spans == {"a": (0, 3), "b": (8, 14)}


def test_mixed_fixed_and_named():
    r = parse.parse("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}
    assert r.spans == {0: (0, 1), 1: (4, 5), "name": (2, 3)}


def test_braces_escape():
    r = parse.parse("{{x}} {}", "{x} 5")
    assert r.fixed == ("5",)
    assert parse.parse("{{}}", "{}").fixed == ()
    assert parse.parse("{{}}", "x") is None


def test_regex_special_chars_escaped():
    fmt = "a.b?c*d+e^f$g|h(i)j[k]l\\m!n"
    assert parse.parse(fmt, fmt) is not None
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("(x)", "x") is None
    assert parse.parse("x+", "xx") is None
    assert parse.parse("a|b", "a") is None
    assert parse.parse("[ab]", "a") is None
    assert parse.parse("a\\d", "a5") is None
    assert parse.parse("a\\d", "a\\d") is not None


def test_case_insensitive_default():
    assert parse.parse("Hello {}", "hELLO x").fixed == ("x",)
    assert parse.parse("Hello {}", "hELLO x", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello x", case_sensitive=True).fixed == ("x",)


def test_dotall():
    r = parse.parse("{}-{}", "a\nb-c\nd")
    assert r.fixed == ("a\nb", "c\nd")
    r = parse.parse("{}-{}", "a\nb-c\nd", case_sensitive=True)
    assert r.fixed == ("a\nb", "c\nd")


def test_trailing_newline_not_allowed():
    # \Z rather than $
    assert parse.parse("{:d}", "12\n") is None


def test_dotted_names():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}
    assert r.spans == {"a.b": (0, 1), "a_b": (2, 3)}


def test_dotted_collision_other_order():
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_underscore_collision():
    r = parse.parse("{a_b} {a__b} {a___b}", "1 2 3")
    assert r.named == {"a_b": "1", "a__b": "2", "a___b": "3"}


def test_hyphen_names():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}
    r = parse.parse("{a_b} {a-b}", "1 2")
    assert r.named == {"a_b": "1", "a-b": "2"}


def test_hyphen_collision_only_hyphen():
    p = Parser("{ab} {a-b}")
    # "a-b" -> "a_b" doesn't collide with "ab"
    assert p.parse("1 2").named == {"ab": "1", "a-b": "2"}


def test_to_group_name_triple_collision():
    p = Parser("{a.b} {a_b} {a__b}")
    r = p.parse("1 2 3")
    assert r.named == {"a.b": "1", "a_b": "2", "a__b": "3"}


def test_to_group_name_mappings():
    p = Parser("{a.b} {a_b}")
    assert p._group_to_name_map == {"a_b": "a.b", "a__b": "a_b"}
    assert p._name_to_group_map == {"a.b": "a_b", "a_b": "a__b"}
    assert p.named_fields == ["a_b", "a__b"]


def test_to_group_name_hyphen_collision_mapping():
    p = Parser("{a_b} {a-b}")
    assert p._name_to_group_map == {"a_b": "a_b", "a-b": "a__b"}


def test_to_group_name_dot_collision_with_n3():
    p = Parser("{a_b} {a__b}")
    p._to_group_name("a.b")
    # a.b -> a_b collides, a__b collides, a___b ok
    assert p._name_to_group_map["a.b"] == "a___b"


def test_to_group_name_duplicate_raises():
    p = Parser("{ab}")
    with pytest.raises(KeyError):
        p._to_group_name("ab")


def test_bracket_names():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{x[y][z]}", "v")
    assert r.named == {"x": {"y": {"z": "v"}}}
    assert r.spans == {"x[y][z]": (0, 1)}


def test_bracket_names_with_dict():
    r = parse.parse("{a[b]} {c}", "1 2")
    assert r.named == {"a": {"b": "1"}, "c": "2"}


def test_expand_named_fields_direct():
    p = Parser("")
    assert p._expand_named_fields({"a": 1, "b[c]": 2, "b[d][e]": 3}) == {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
    }


def test_repeated_name():
    r = parse.parse("{n} {n}", "x x")
    assert r.named == {"n": "x"}
    assert parse.parse("{n} {n}", "x y") is None
    r = parse.parse("{n:d} {n:d}", "3 3")
    assert r.named == {"n": 3}


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.compile("{n:d} {n:w}")
    with pytest.raises(RepeatedNameError):
        parse.compile("{n} {n:d}")
    assert issubclass(RepeatedNameError, ValueError)
    assert issubclass(TooManyFields, ValueError)


def test_repeated_name_error_message():
    with pytest.raises(RepeatedNameError) as ei:
        parse.compile("{n:d} {n:w}")
    msg = str(ei.value)
    assert "'w'" in msg and '"n"' in msg and "'d'" in msg


def test_numeric_name_is_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r.named == {}


def test_unknown_format_spec():
    with pytest.raises(ValueError) as ei:
        parse.compile("{:zz}")
    assert "'zz'" in str(ei.value)
    with pytest.raises(ValueError):
        parse.compile("{:q}")


# ---------------------------------------------------------------- types


def test_type_d():
    assert parse.parse("{:d}", "123").fixed == (123,)
    assert parse.parse("{:d}", "-123").fixed == (-123,)
    assert parse.parse("{:d}", "+123").fixed == (123,)
    assert parse.parse("{:d}", " 12").fixed == (12,)
    assert parse.parse("{:d}", "0x1F").fixed == (31,)
    assert parse.parse("{:d}", "0X1f").fixed == (31,)
    assert parse.parse("{:d}", "0b101").fixed == (5,)
    assert parse.parse("{:d}", "0B101").fixed == (5,)
    assert parse.parse("{:d}", "0o17").fixed == (15,)
    assert parse.parse("{:d}", "0O17").fixed == (15,)
    assert parse.parse("{:d}", "-0x10").fixed == (-16,)
    assert parse.parse("{:d}", "+0b11").fixed == (3,)
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "1.5") is None
    assert parse.parse("{:d}", "0") .fixed == (0,)
    assert parse.parse("{:d}", "00").fixed == (0,)
    assert parse.parse("{:d}", "007").fixed == (7,)
    assert parse.parse("{:d}", "010").fixed == (10,)


def test_type_d_types():
    v = parse.parse("{:d}", "5").fixed[0]
    assert type(v) is int


def test_type_d_width():
    assert parse.parse("{:2d}{:2d}", "1234").fixed == (12, 34)
    assert parse.parse("{:2d}", "123") is None
    assert parse.parse("{:3d}", " 12").fixed == (12,)
    assert parse.parse("{:3d}", "  1").fixed == (1,)
    assert parse.parse("{:3d}", "   1").fixed == (1,)


def test_type_d_grouping():
    assert parse.parse("{:,d}", "1,234,567").fixed == (1234567,)
    assert parse.parse("{:_d}", "1_234").fixed == (1234,)
    assert parse.parse("{:d}", "1,234") is None
    assert parse.parse("{:_d}", "1,234") is None
    assert parse.parse("{:,d}", "1_234") is None


def test_type_d_zero_pad():
    assert parse.parse("{:05d}", "00042").fixed == (42,)
    assert parse.parse("{:04d}", "-042").fixed == (-42,)


def test_type_n():
    assert parse.parse("{:n}", "1,234,567").fixed == (1234567,)
    assert parse.parse("{:n}", "1.234.567").fixed == (1234567,)
    assert parse.parse("{:n}", "123").fixed == (123,)
    assert parse.parse("{:n}", "-1,000").fixed == (-1000,)
    assert parse.parse("{:n}", "1234") is None
    assert parse.parse("{:n}", "1,23") is None


def test_type_n_with_following_fields():
    r = parse.parse("{:n} {:d}", "1,000 5")
    assert r.fixed == (1000, 5)


def test_type_b():
    assert parse.parse("{:b}", "1010").fixed == (10,)
    assert parse.parse("{:b}", "0b1010").fixed == (10,)
    assert parse.parse("{:b}", "0B11").fixed == (3,)
    assert parse.parse("{:b}", "-0b11").fixed == (-3,)
    assert parse.parse("{:b}", "102") is None
    assert parse.parse("{:b} {:d}", "0b11 7").fixed == (3, 7)


def test_type_o():
    assert parse.parse("{:o}", "17").fixed == (15,)
    assert parse.parse("{:o}", "0o17").fixed == (15,)
    assert parse.parse("{:o}", "0O17").fixed == (15,)
    assert parse.parse("{:o}", "18") is None
    assert parse.parse("{:o} {:d}", "0o17 7").fixed == (15, 7)


def test_type_x():
    assert parse.parse("{:x}", "ff").fixed == (255,)
    assert parse.parse("{:x}", "0xFF").fixed == (255,)
    assert parse.parse("{:X}", "0XfF").fixed == (255,)
    assert parse.parse("{:x}", "fg") is None
    assert parse.parse("{:x} {:d}", "0x1f 7").fixed == (31, 7)
    assert parse.parse("{:X} {:d}", "1f 7").fixed == (31, 7)


def test_type_x_case_sensitive():
    assert parse.parse("{:x}", "FF", case_sensitive=True).fixed == (255,)
    assert parse.parse("{:x}", "0XFF", case_sensitive=True).fixed == (255,)


def test_type_percent():
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "-10%").fixed == (-0.1,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%} {:d}", "12.5% 7").fixed == (0.125, 7)


def test_percentage_func():
    assert percentage("25%", None) == 0.25
    assert percentage("100%", None) == 1.0


def test_type_f():
    assert parse.parse("{:f}", "3.14").fixed == (3.14,)
    assert parse.parse("{:f}", ".5").fixed == (0.5,)
    assert parse.parse("{:f}", "-3.5").fixed == (-3.5,)
    assert parse.parse("{:f}", "+3.5").fixed == (3.5,)
    assert parse.parse("{:f}", "3") is None
    assert parse.parse("{:f}", "3.") is None
    v = parse.parse("{:f}", "1.5").fixed[0]
    assert type(v) is float
    r = parse.parse("{:f}", "nan").fixed[0]
    assert r != r
    assert parse.parse("{:f}", "inf").fixed == (float("inf"),)
    assert parse.parse("{:f}", "-inf").fixed == (float("-inf"),)
    assert parse.parse("{:f}", "INF").fixed == (float("inf"),)
    assert parse.parse("{:f} {:d}", "1.5 7").fixed == (1.5, 7)


def test_type_f_precision_zero():
    assert parse.parse("{:.0f}", "20").fixed == (20.0,)
    assert parse.parse("{:.0f}", "-20").fixed == (-20.0,)
    assert parse.parse("{:.0f}", "20.5") is None
    assert parse.parse("{:.2f}", "20") is None
    assert parse.parse("{:.2f}", "20.25").fixed == (20.25,)
    assert parse.parse("{:.10f}", "1.5").fixed == (1.5,)


def test_type_F_decimal():
    v = parse.parse("{:F}", "3.14").fixed[0]
    assert v == Decimal("3.14")
    assert type(v) is Decimal
    assert parse.parse("{:F}", "-1.25").fixed == (Decimal("-1.25"),)


def test_type_e():
    assert parse.parse("{:e}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5E-3").fixed == (0.0015,)
    assert parse.parse("{:e}", "-1.5e+3").fixed == (-1500.0,)
    assert parse.parse("{:E}", "1.5E3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "inf").fixed == (float("inf"),)
    assert parse.parse("{:e}", "-inf").fixed == (float("-inf"),)
    assert parse.parse("{:e} {:d}", "1.0e1 7").fixed == (10.0, 7)
    v = parse.parse("{:e}", "nan").fixed[0]
    assert v != v


def test_type_g():
    assert parse.parse("{:g}", "1.5").fixed == (1.5,)
    assert parse.parse("{:g}", "15").fixed == (15.0,)
    assert parse.parse("{:g}", "1e10").fixed == (1e10,)
    assert parse.parse("{:g}", "1.5e-3").fixed == (0.0015,)
    assert parse.parse("{:g}", "-1.5").fixed == (-1.5,)
    assert parse.parse("{:G}", "1.5E3").fixed == (1500.0,)
    assert parse.parse("{:g}", "inf").fixed == (float("inf"),)
    assert type(parse.parse("{:g}", "15").fixed[0]) is float
    assert parse.parse("{:g}", "abc") is None


def test_type_g_group_counting():
    # g regex has 2 internal groups; following fields must still line up
    r = parse.parse("{:g} {:d} {}", "1.5e3 7 x")
    assert r.fixed == (1500.0, 7, "x")
    assert r.spans == {0: (0, 5), 1: (6, 7), 2: (8, 9)}


def test_group_counting_all_numeric():
    r = parse.parse(
        "{:n} {:b} {:o} {:x} {:%} {:f} {:e} {:g} {:d} {}",
        "1,000 0b1 0o7 0xf 5% 1.5 1.0e1 2.5 9 end",
    )
    assert r.fixed == (1000, 1, 7, 15, 0.05, 1.5, 10.0, 2.5, 9, "end")


def test_group_counting_named_after_numeric():
    r = parse.parse("{:n} {:b} {:o} {:x} {:%} {:g} {a}", "1,000 0b1 0o7 0xf 5% 2.5 z")
    assert r.named == {"a": "z"}
    assert r.fixed == (1000, 1, 7, 15, 0.05, 2.5)


def test_type_w_W():
    assert parse.parse("{:w}", "abc_12").fixed == ("abc_12",)
    assert parse.parse("{:w}", "ab c") is None
    assert parse.parse("{:W}", "  !!").fixed == ("  !!",)
    assert parse.parse("{:W}", "a") is None


def test_type_s_S():
    assert parse.parse("{:s}", " \t ").fixed == (" \t ",)
    assert parse.parse("{:s}", "x") is None
    assert parse.parse("{:S}", "abc").fixed == ("abc",)
    assert parse.parse("{:S}", "a c") is None


def test_type_D():
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None


def test_type_l():
    assert parse.parse("{:l}", "abcXYZ").fixed == ("abcXYZ",)
    assert parse.parse("{:l}", "abc1") is None
    assert parse.parse("{:l}", "") is None


def test_type_l_not_numeric():
    assert parse.parse("{:l}", "-abc") is None


def test_string_types_no_sign():
    assert parse.parse("{:w}", "-abc") is None
    assert parse.parse("{:w}", "+abc") is None


def test_numeric_types_sign_prefix():
    for spec in "nbox%fegdEGX":
        assert parse.compile("{:%s}" % spec)._expression.startswith(r"([-+ ]?")


# ---------------------------------------------------------------- width / precision / alignment


def test_width_string():
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "abcd").fixed == ("abcd",)


def test_width_string_right_align_default():
    # width implies '>' alignment with spaces
    assert parse.parse("{:2}", "   ab").fixed == ("ab",)


def test_precision_string():
    assert parse.parse("{:.2}", "abc") is None
    assert parse.parse("{:.2}", "ab").fixed == ("ab",)
    assert parse.parse("{:.2}", "a").fixed == ("a",)
    assert parse.parse("{:.2}{}", "abc").fixed == ("a", "bc")


def test_width_and_precision_string():
    p = parse.compile("{:2.4}")
    assert p.parse("a") is None
    assert p.parse("ab").fixed == ("ab",)
    assert p.parse("abcd").fixed == ("abcd",)
    assert p.parse("abcde") is None
    assert "{2,4}" in p._expression


def test_width_regex_exprs():
    assert parse.compile("{:5}")._expression == r" *(.{5,}?)"
    assert parse.compile("{:.3}")._expression == r"(.{1,3}?)"
    assert parse.compile("{}")._expression == r"(.+?)"
    assert parse.compile("{a}")._expression == r"(?P<a>.+?)"
    assert parse.compile("{:<}")._expression == r"(.+?) *"
    assert parse.compile("{:>}")._expression == r" *(.+?)"
    assert parse.compile("{:^}")._expression == r" *(.+?) *"


def test_alignment():
    assert parse.parse("{:<} |", "ab   |").fixed == ("ab",)
    assert parse.parse("{:>}|", "   ab|").fixed == ("ab",)
    assert parse.parse("{:^}|", "  ab  |").fixed == ("ab",)
    assert parse.parse("|{:<}|", "|ab   |").fixed == ("ab",)


def test_alignment_with_fill():
    assert parse.parse("{:x<}|", "abxxx|") is not None
    assert parse.parse("{:*<}|", "ab***|").fixed == ("ab",)
    assert parse.parse("{:*>}|", "***ab|").fixed == ("ab",)
    assert parse.parse("{:*^}|", "**ab**|").fixed == ("ab",)
    assert parse.parse("{:.^}|", "..ab..|").fixed == ("ab",)
    assert parse.parse("{:-^}|", "--ab--|").fixed == ("ab",)


def test_fill_escaping():
    for ch in ".\\+?*[](){}^$":
        if ch in "{}^":
            continue
        fmt = "{:" + ch + "^}|"
        p = parse.compile(fmt)
        assert p.parse(ch * 2 + "ab" + ch * 2 + "|").fixed == ("ab",), ch
    # unescaped "." fill would match anything
    assert parse.parse("{:.>}|", "xxab|").fixed == ("xxab",)


def test_alignment_with_numbers():
    assert parse.parse("{:>5d}", "   42").fixed == (42,)
    assert parse.parse("{:<5d}|", "42   |").fixed == (42,)
    assert parse.parse("{:^5d}|", " 42  |").fixed == (42,)
    assert parse.parse("{:5d}", "   42").fixed == (42,)
    assert parse.parse("{:x>5d}", "xxx42").fixed == (42,)


def test_align_equals_numeric():
    assert parse.parse("{:=5d}", "00042").fixed == (42,)
    assert parse.parse("{:=d}", "-00042").fixed == (-42,)
    assert parse.parse("{:x=5d}", "xxx42").fixed == (42,)
    assert parse.compile("{:=d}")._expression.startswith(r"([-+ ]?0*")
    assert parse.compile("{:x=d}")._expression.startswith(r"([-+ ]?x*")


def test_align_equals_f():
    assert parse.parse("{:=8.2f}", "00003.14").fixed == (3.14,)


def test_sign_in_spec():
    assert parse.parse("{:+d}", "+5").fixed == (5,)
    assert parse.parse("{:-d}", "-5").fixed == (-5,)
    assert parse.parse("{: d}", " 5").fixed == (5,)


def test_zero_and_width_numeric():
    assert parse.parse("{:08.3f}", "0003.142").fixed == (3.142,)
    assert parse.parse("{:3d}|", "  5|").fixed == (5,)


def test_named_with_format():
    r = parse.parse("{x:d}+{y:f}", "3+1.5")
    assert r.named == {"x": 3, "y": 1.5}
    assert r.spans == {"x": (0, 1), "y": (2, 5)}


# ---------------------------------------------------------------- extract_format


def test_extract_format_full():
    f = extract_format("*^+010,.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "^"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "f"


def test_extract_format_align_only():
    f = extract_format("<5", {})
    assert f["align"] == "<"
    assert f["fill"] is None
    assert f["width"] == "5"
    assert f["type"] == ""
    assert f["zero"] is False


def test_extract_format_each_align():
    for a in "<>=^":
        f = extract_format(a + "d", {})
        assert f["align"] == a
        assert f["fill"] is None
        assert f["type"] == "d"
        f = extract_format("x" + a + "d", {})
        assert f["align"] == a
        assert f["fill"] == "x"
        assert f["type"] == "d"


def test_extract_format_single_char():
    f = extract_format("d", {})
    assert f["type"] == "d"
    assert f["align"] is None
    assert f["fill"] is None
    assert f["width"] == ""
    assert "precision" not in f
    assert "grouping" not in f


def test_extract_format_grouping_underscore():
    f = extract_format("_d", {})
    assert f["grouping"] == "_"
    assert f["type"] == "d"


def test_extract_format_zero():
    f = extract_format("05d", {})
    assert f["zero"] is True
    assert f["width"] == "5"
    f = extract_format("0", {})
    assert f["zero"] is True
    assert f["width"] == ""
    assert f["type"] == ""


def test_extract_format_signs():
    for s in "+- ":
        f = extract_format(s + "5d", {})
        assert f["width"] == "5"
        assert f["type"] == "d"


def test_extract_format_precision_then_type():
    f = extract_format(".12e", {})
    assert f["precision"] == "12"
    assert f["type"] == "e"
    f = extract_format(".", {})
    assert f["precision"] == ""
    assert f["type"] == ""


def test_extract_format_extra_types():
    f = extract_format("custom", {"custom": None})
    assert f["type"] == "custom"
    with pytest.raises(ValueError):
        extract_format("custom", {})


def test_extract_format_datetime_type():
    f = extract_format("%Y-%m-%d", {})
    assert f["type"] == "%Y-%m-%d"


def test_extract_format_t_types():
    for c in "ieahgcts":
        assert extract_format("t" + c, {})["type"] == "t" + c
    with pytest.raises(ValueError):
        extract_format("tz", {})


def test_allowed_types():
    assert parse.ALLOWED_TYPES == set(
        list("nboxX%fFeEgGwWdDsSl") + ["t" + c for c in "ieahgcts"]
    )


# ---------------------------------------------------------------- int_convert


def test_int_convert_auto_base():
    ic = int_convert()
    assert ic("123", None) == 123
    assert ic("-123", None) == -123
    assert ic("+123", None) == 123
    assert ic("0x1f", None) == 31
    assert ic("0X1F", None) == 31
    assert ic("0b11", None) == 3
    assert ic("0B11", None) == 3
    assert ic("0o17", None) == 15
    assert ic("0O17", None) == 15
    assert ic("-0x10", None) == -16
    assert ic("+0b10", None) == 2
    assert ic("1,234", None) == 1234
    assert ic("0", None) == 0
    assert ic("00", None) == 0
    assert ic("09", None) == 9
    assert ic("-09", None) == -9


def test_int_convert_short_prefix_strings():
    ic = int_convert()
    assert ic("01", None) == 1
    assert ic("-01", None) == -1
    # exactly 3 chars after start triggers prefix detection
    assert ic("0x1", None) == 1
    assert ic("-0x1", None) == -1
    assert ic("0b1", None) == 1
    assert ic("0o7", None) == 7


def test_int_convert_explicit_base():
    assert int_convert(16)("ff", None) == 255
    assert int_convert(16)("0xff", None) == 255
    assert int_convert(2)("0b101", None) == 5
    assert int_convert(8)("0o17", None) == 15
    assert int_convert(10)("1.234.567", None) == 1234567
    assert int_convert(10)("-1,000", None) == -1000
    assert int_convert(16)("-FF", None) == -255


def test_int_convert_base_attr():
    assert int_convert().base is None
    assert int_convert(8).base == 8
    assert int_convert.CHARS == "0123456789abcdefghijklmnopqrstuvwxyz"


def test_convert_first():
    c = convert_first(int)
    assert c("12", object()) == 12
    assert c.converter is int
    assert pickle.loads(pickle.dumps(c))("5", None) == 5


# ---------------------------------------------------------------- with_pattern / extra types


def test_with_pattern():
    @with_pattern(r"\d+")
    def num(text):
        return int(text)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    r = parse.parse("{:num} {:num}", "12 34", {"num": num})
    assert r.fixed == (12, 34)
    assert parse.parse("{:num}", "ab", {"num": num}) is None


def test_with_pattern_group_count():
    @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
    def rng(text):
        a, b = text.split("-")
        return (int(a), int(b))

    assert rng.regex_group_count == 3
    r = parse.parse("{:rng} {:d} {name}", "1-2 3 x", {"rng": rng})
    assert r.fixed == ((1, 2), 3)
    assert r.named == {"name": "x"}
    assert r.spans[1] == (4, 5)


def test_extra_type_without_pattern():
    def upper(text):
        return text.upper()

    r = parse.parse("{:up}-{}", "abc-d", {"up": upper})
    assert r.fixed == ("ABC", "d")


def test_extra_type_overrides_builtin():
    def myd(text):
        return "D" + text

    r = parse.parse("{:d}", "12", {"d": myd})
    assert r.fixed == ("D12",)


def test_extra_type_named():
    @with_pattern(r"[a-z]+")
    def word(t):
        return t[::-1]

    r = parse.parse("{w:word} {n:d}", "abc 5", extra_types={"word": word})
    assert r.named == {"w": "cba", "n": 5}


def test_extra_type_regex_group_count_none_then_fields():
    ab2 = with_pattern(r"(a|b)+", regex_group_count=1)(lambda t: t)
    r = parse.parse("{:ab} {:d}", "abab 4", {"ab": ab2})
    assert r.fixed == ("abab", 4)


def test_extra_type_with_width_alignment():
    @with_pattern(r"\d+")
    def num(text):
        return int(text)

    r = parse.parse("{:>5num}|", "   42|", {"num": num})
    assert r.fixed == (42,)


# ---------------------------------------------------------------- datetime types


def test_ti():
    r = parse.parse("{:ti}", "2023-04-05")
    assert r.fixed == (datetime(2023, 4, 5),)
    r = parse.parse("{:ti}", "2023-04-05T10:20")
    assert r.fixed == (datetime(2023, 4, 5, 10, 20),)
    r = parse.parse("{:ti}", "2023-04-05 10:20:30")
    assert r.fixed == (datetime(2023, 4, 5, 10, 20, 30),)
    r = parse.parse("{:ti}", "2023-04-05T10:20:30.5")
    assert r.fixed == (datetime(2023, 4, 5, 10, 20, 30, 500000),)
    r = parse.parse("{:ti}", "2023-04-05T10:20:30.1234567")
    assert r.fixed == (datetime(2023, 4, 5, 10, 20, 30, 123456),)


def test_ti_timezones():
    r = parse.parse("{:ti}", "2023-04-05T10:20:30Z")[0]
    assert r.tzinfo == FixedTzOffset(0, "UTC")
    assert r.utcoffset() == timedelta(0)
    r = parse.parse("{:ti}", "2023-04-05T10:20:30+05:30")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2023-04-05T10:20:30-0130")[0]
    assert r.utcoffset() == -timedelta(hours=1, minutes=30)
    assert r.tzinfo == FixedTzOffset(-90, "-0130")


def test_ti_followed_by_field():
    r = parse.parse("{:ti} {}", "2023-04-05T10:20:30Z rest")
    assert r.fixed[1] == "rest"
    assert r.spans[1] == (21, 25)
    r = parse.parse("{:ti} {name}", "2023-04-05 x")
    assert r.named == {"name": "x"}


def test_tg():
    r = parse.parse("{:tg}", "5/4/2023")[0]
    assert r == datetime(2023, 4, 5)
    r = parse.parse("{:tg}", "5/4/2023 10:20")[0]
    assert r == datetime(2023, 4, 5, 10, 20)
    r = parse.parse("{:tg}", "5-Apr-2023 10:20:30 PM")[0]
    assert r == datetime(2023, 4, 5, 22, 20, 30)
    r = parse.parse("{:tg}", "5/April/2023 12:20 AM")[0]
    assert r == datetime(2023, 4, 5, 0, 20)
    r = parse.parse("{:tg}", "5/4/2023 12:20 PM")[0]
    assert r == datetime(2023, 4, 5, 12, 20)
    r = parse.parse("{:tg}", "5/4/2023 11:20 AM")[0]
    assert r == datetime(2023, 4, 5, 11, 20)
    r = parse.parse("{:tg}", "5/4/2023 10:20 +0100")[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = parse.parse("{:tg} {}", "5/4/2023 10:20 -1:00 x")
    assert r[0].utcoffset() == -timedelta(hours=1)
    assert r[1] == "x"


def test_ta():
    r = parse.parse("{:ta}", "4/5/2023")[0]
    assert r == datetime(2023, 4, 5)
    r = parse.parse("{:ta}", "Apr/5/2023 3:04:05 PM")[0]
    assert r == datetime(2023, 4, 5, 15, 4, 5)
    r = parse.parse("{:ta} {}", "4-5-2023 3:04 AM x")
    assert r[0] == datetime(2023, 4, 5, 3, 4)
    assert r[1] == "x"


def test_te():
    r = parse.parse("{:te}", "Mon, 20 Jan 2020 10:20:30 +0000")[0]
    assert r == datetime(2020, 1, 20, 10, 20, 30, tzinfo=FixedTzOffset(0, "+0000"))
    r = parse.parse("{:te}", "20 Jan 2020 10:20:30 -0500")[0]
    assert r.utcoffset() == -timedelta(hours=5)
    assert r.replace(tzinfo=None) == datetime(2020, 1, 20, 10, 20, 30)
    r = parse.parse("{:te} {}", "20 Jan 2020 10:20 +0130 x")
    assert r[1] == "x"
    assert r[0].utcoffset() == timedelta(hours=1, minutes=30)


def test_th():
    r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
    assert r.replace(tzinfo=None) == datetime(2011, 11, 21, 10, 21, 36)
    assert r.utcoffset() == timedelta(hours=10)
    r = parse.parse("{:th} {}", "21-Nov-2011:10:21:36 -1000 x")
    assert r[0].utcoffset() == -timedelta(hours=10)
    assert r[1] == "x"


def test_tc():
    r = parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0]
    assert r == datetime(1973, 9, 16, 1, 3, 52)
    r = parse.parse("{:tc} {}", "Sun Sep 16 01:03:52 1973 x")
    assert r[1] == "x"


def test_tt():
    assert parse.parse("{:tt}", "10:20")[0] == time(10, 20)
    assert parse.parse("{:tt}", "10:20:30")[0] == time(10, 20, 30)
    assert parse.parse("{:tt}", "10:20:30.25")[0] == time(10, 20, 30, 250000)
    assert parse.parse("{:tt}", "1:20 PM")[0] == time(13, 20)
    assert parse.parse("{:tt}", "12:20 PM")[0] == time(12, 20)
    assert parse.parse("{:tt}", "12:20 AM")[0] == time(0, 20)
    assert parse.parse("{:tt}", "1:20 AM")[0] == time(1, 20)
    t = parse.parse("{:tt}", "10:20 +05:00")[0]
    assert t.utcoffset() == timedelta(hours=5)
    t = parse.parse("{:tt}", "10:20 -530")[0]
    assert t.utcoffset() == -timedelta(hours=5, minutes=30)
    r = parse.parse("{:tt} {}", "10:20 PM x")
    assert r[0] == time(22, 20)
    assert r[1] == "x"


def test_ts():
    year = datetime.today().year
    r = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
    assert r == datetime(year, 11, 9, 3, 37, 44)
    r = parse.parse("{:ts} {}", "Feb 21 03:37:44 x")
    assert r[0] == datetime(year, 2, 21, 3, 37, 44)
    assert r[1] == "x"


def test_datetime_named():
    r = parse.parse("{when:ti} {n:d}", "2023-04-05 7")
    assert r.named == {"when": datetime(2023, 4, 5), "n": 7}


# ---------------------------------------------------------------- date_convert direct


def test_date_convert_ymd_slash_space():
    m = re.match(r"(.*)", "2020/01/02")
    # groups() is 0-indexed for date_convert
    assert date_convert(None, m, ymd=0) == datetime(2020, 1, 2)
    m = re.match(r"(.*)", "2020 01 02")
    assert date_convert(None, m, ymd=0) == datetime(2020, 1, 2)


def test_date_convert_mdy_dmy():
    m = re.match(r"(.*)", "01-02-2020")
    assert date_convert(None, m, mdy=0) == datetime(2020, 1, 2)
    assert date_convert(None, m, dmy=0) == datetime(2020, 2, 1)
    m = re.match(r"(.*)", "01-Feb-2020")
    assert date_convert(None, m, dmy=0) == datetime(2020, 2, 1)


def test_date_convert_d_m_y():
    m = re.match(r"(\d+) (\w+) (\d+)", "3 March 2021")
    assert date_convert(None, m, d_m_y=(0, 1, 2)) == datetime(2021, 3, 3)


def test_date_convert_mm_dd():
    # mm/dd are only used if both truthy, so avoid group index 0
    m2 = re.match(r"(x)(\w+) (\d+)", "xMar 7")
    r = date_convert(None, m2, mm=1, dd=2)
    assert r == datetime(datetime.today().year, 3, 7)


def test_date_convert_time_only():
    m = re.match(r"(.*)", "10:20:30")
    assert date_convert(None, m, hms=0) == time(10, 20, 30)
    m = re.match(r"(.*)", "10:20:30.000001")
    assert date_convert(None, m, hms=0) == time(10, 20, 30, 1)


def test_date_convert_tz_variants():
    m = re.match(r"(.*)\|(.*)", "10:20|Z")
    t = date_convert(None, m, hms=0, tz=1)
    assert t.tzinfo == FixedTzOffset(0, "UTC")
    m = re.match(r"(.*)\|(.*)", "10:20| +0530")
    t = date_convert(None, m, hms=0, tz=1)
    assert t.tzinfo == FixedTzOffset(330, "+0530")
    m = re.match(r"(.*)\|(.*)", "10:20|+530")
    t = date_convert(None, m, hms=0, tz=1)
    assert t.tzinfo == FixedTzOffset(330, "+530")
    m = re.match(r"(.*)\|(.*)", "10:20|-05:30")
    t = date_convert(None, m, hms=0, tz=1)
    assert t.tzinfo == FixedTzOffset(-330, "-05:30")
    m = re.match(r"(.*)\|(.*)", "10:20|EST")
    with pytest.raises(TypeError):
        date_convert(None, m, hms=0, tz=1)


def test_date_convert_no_tz():
    m = re.match(r"(.*)\|(x)?", "10:20|")
    t = date_convert(None, m, hms=0, tz=1)
    assert t.tzinfo is None
    assert t == time(10, 20)


def test_date_convert_am_none():
    m = re.match(r"(.*)\|(.*)", "10:20|")
    t = date_convert(None, m, hms=0, am=1)
    assert t == time(10, 20)


def test_date_convert_am_pm():
    m = re.match(r"(.*)\|(.*)", "12:05| AM")
    assert date_convert(None, m, hms=0, am=1) == time(0, 5)
    m = re.match(r"(.*)\|(.*)", "12:05|PM")
    assert date_convert(None, m, hms=0, am=1) == time(12, 5)
    m = re.match(r"(.*)\|(.*)", "11:05|PM")
    assert date_convert(None, m, hms=0, am=1) == time(23, 5)
    m = re.match(r"(.*)\|(.*)", "11:05|AM")
    assert date_convert(None, m, hms=0, am=1) == time(11, 5)


def test_date_convert_no_hms_group():
    m = re.match(r"(.*)\|(.*)", "2020-01-02|")
    assert date_convert(None, m, ymd=0, hms=1) == datetime(2020, 1, 2)


# ---------------------------------------------------------------- FixedTzOffset


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert FixedTzOffset.ZERO == timedelta(0)
    assert repr(tz) == "<FixedTzOffset X 1:30:00>"
    assert tz == FixedTzOffset(90, "X")
    assert tz != FixedTzOffset(91, "X")
    assert tz != FixedTzOffset(90, "Y")
    assert tz.__eq__(5) is NotImplemented
    assert (tz == 5) is False


# ---------------------------------------------------------------- strftime-style


def test_strftime_date():
    r = parse.parse("{:%Y-%m-%d}", "2023-04-05")[0]
    assert r == date(2023, 4, 5)
    assert type(r) is date


def test_strftime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 10:20:30")[0]
    assert r == datetime(2023, 4, 5, 10, 20, 30)
    assert type(r) is datetime


def test_strftime_time():
    r = parse.parse("{:%H:%M}", "10:20")[0]
    assert r == time(10, 20)
    assert type(r) is time
    r = parse.parse("{:%I:%M %p}", "10:20 PM")[0]
    assert r == time(22, 20)


def test_strftime_time_with_tz():
    r = parse.parse("{:%H:%M %z}", "10:20 +0100")[0]
    assert r.utcoffset() == timedelta(hours=1)


def test_strftime_no_year_uses_current():
    r = parse.parse("{:%m-%d}", "04-05")[0]
    assert r == date(datetime.today().year, 4, 5)


def test_strftime_two_digit_year():
    r = parse.parse("{:%y-%m-%d}", "23-04-05")[0]
    assert r == date(2023, 4, 5)


def test_strftime_names():
    r = parse.parse("{:%a %b %d %Y}", "Wed Apr 05 2023")[0]
    assert r == date(2023, 4, 5)
    r = parse.parse("{:%A %B %d %Y}", "Wednesday April 05 2023")[0]
    assert r == date(2023, 4, 5)


def test_strftime_microseconds_and_doy():
    r = parse.parse("{:%Y %j}", "2023 095")[0]
    assert r == date(2023, 4, 5)
    r = parse.parse("{:%H:%M:%S.%f}", "10:20:30.5")[0]
    assert r == time(10, 20, 30, 500000)


def test_strftime_followed_by_field():
    r = parse.parse("{:%Y-%m-%d} {:d}", "2023-04-05 7")
    assert r.fixed == (date(2023, 4, 5), 7)


def test_strftime_no_match():
    assert parse.parse("{:%Y-%m-%d}", "23-04-05") is None
    assert parse.parse("{:%H:%M}", "10:2") is None


def test_strf_date_convert_each_flag():
    year = datetime.today().year
    assert strf_date_convert("3", None, "%w") == date(1900, 1, 1).replace(year=year)
    assert strf_date_convert("Apr", None, "%b") == date(year, 4, 1)
    assert strf_date_convert("April", None, "%B") == date(year, 4, 1)
    assert strf_date_convert("Wed", None, "%a") == date(year, 1, 1)
    assert strf_date_convert("Wednesday", None, "%A") == date(year, 1, 1)
    assert strf_date_convert("05", None, "%d") == date(year, 1, 5)
    assert strf_date_convert("04", None, "%m") == date(year, 4, 1)
    assert strf_date_convert("2020", None, "%Y") == date(2020, 1, 1)
    assert strf_date_convert("20", None, "%y") == date(2020, 1, 1)
    assert strf_date_convert("032", None, "%j") == date(year, 2, 1)
    assert strf_date_convert("1", None, "%U") == date(year, 1, 1)
    assert strf_date_convert("1", None, "%W") == date(year, 1, 1)
    assert strf_date_convert("10", None, "%H") == time(10)
    assert strf_date_convert("10", None, "%I") == time(10)
    assert strf_date_convert("PM", None, "%p") == time(0)
    assert strf_date_convert("05", None, "%M") == time(0, 5)
    assert strf_date_convert("05", None, "%S") == time(0, 0, 5)
    assert strf_date_convert("5", None, "%f") == time(0, 0, 0, 500000)
    t = strf_date_convert("+0100", None, "%z")
    assert type(t) is time
    assert t.utcoffset() == timedelta(hours=1)


def test_strf_date_convert_neither():
    with pytest.raises(ValueError):
        strf_date_convert("x", None, "x")


def test_get_regex_for_datetime_format():
    assert get_regex_for_datetime_format("%Y-%m-%d") == "[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"
    assert get_regex_for_datetime_format("abc") == "abc"
    assert get_regex_for_datetime_format("%H:%M") == "[0-9]{1,2}:[0-9]{2}"


def test_dt_regexes_individually():
    cases = {
        "%a": ("Sun", "Xyz"),
        "%A": ("Sunday", "Sun"),
        "%w": ("6", "7"),
        "%d": ("31", "123"),
        "%b": ("Dec", "December"),
        "%B": ("December", "Dec"),
        "%m": ("12", "123"),
        "%y": ("99", "999"),
        "%Y": ("1999", "99"),
        "%H": ("23", "234"),
        "%I": ("11", "111"),
        "%p": ("AM", "XM"),
        "%M": ("59", "5"),
        "%S": ("59", "5"),
        "%f": ("123456", "1234567"),
        "%z": ("+01:00", "01:00"),
        "%j": ("366", "3666"),
        "%U": ("53", "533"),
        "%W": ("53", "533"),
    }
    for k, (good, bad) in cases.items():
        rx = re.compile(r"\A" + parse.dt_format_to_regex[k] + r"\Z")
        assert rx.match(good), k
        assert not rx.match(bad), k


# ---------------------------------------------------------------- search / findall


def test_search():
    r = search("age: {:d}", "name: x age: 42 years")
    assert r.fixed == (42,)
    assert r.spans == {0: (13, 15)}
    assert search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert search("a{:d}", s).fixed == (1,)
    assert search("a{:d}", s, pos=1).fixed == (2,)
    assert search("a{:d}", s, pos=3).fixed == (2,)
    assert search("a{:d}", s, pos=4).fixed == (3,)
    assert search("a{:d}", s, endpos=1) is None
    assert search("a{:d}", s, endpos=2).fixed == (1,)
    assert search("a{:d}", s, pos=7) is None


def test_search_case_sensitive():
    assert search("X{:d}", "x1") .fixed == (1,)
    assert search("X{:d}", "x1", case_sensitive=True) is None


def test_search_evaluate_false():
    m = search("a{:d}", "xa5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.evaluate_result().fixed == (5,)
    assert m.match.group(0) == "a5"


def test_search_extra_types():
    r = search("{:up}!", "xx abc!", extra_types={"up": with_pattern("[a-z]+")(lambda t: t.upper())})
    assert r.fixed == ("ABC",)


def test_findall():
    results = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
    assert [r.fixed[0] for r in results] == ["some ", "bold", " text"]


def test_findall_numbers():
    r = [x[0] for x in findall("{:d}", "1 22 333")]
    assert r == [1, 22, 333]


def test_findall_pos_endpos():
    s = "a1 a2 a3 a4"
    assert [x[0] for x in findall("a{:d}", s, pos=3)] == [2, 3, 4]
    assert [x[0] for x in findall("a{:d}", s, endpos=8)] == [1, 2, 3]
    assert [x[0] for x in findall("a{:d}", s, pos=3, endpos=5)] == [2]


def test_findall_case_sensitive():
    assert len(list(findall("A{:d}", "a1 A2"))) == 2
    assert [x[0] for x in findall("A{:d}", "a1 A2", case_sensitive=True)] == [2]


def test_findall_evaluate_false():
    it = findall("a{:d}", "a1 a2", evaluate_result=False)
    assert isinstance(it, ResultIterator)
    ms = list(it)
    assert all(isinstance(m, Match) for m in ms)
    assert [m.evaluate_result()[0] for m in ms] == [1, 2]


def test_findall_extra_types():
    num = with_pattern(r"\d")(lambda t: int(t))
    assert [r[0] for r in findall("{:num}", "a1b2", extra_types={"num": num})] == [1, 2]


def test_result_iterator_protocol():
    it = findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it)[0] == 1
    assert it.next()[0] == 2
    with pytest.raises(StopIteration):
        next(it)


def test_result_iterator_attrs():
    p = compile("{:d}")
    it = p.findall("1 2 3", pos=1, endpos=4)
    assert it.parser is p
    assert it.string == "1 2 3"
    assert it.pos == 1
    assert it.endpos == 4
    assert it.evaluate_result is True
    assert [x[0] for x in it] == [2]
    it = p.findall("1 2 3")
    assert it.endpos == 5
    assert it.pos == 0


def test_findall_advances_pos():
    it = findall("{:d}", "12 34")
    next(it)
    assert it.pos == 2


# ---------------------------------------------------------------- Parser / compile


def test_compile():
    p = compile("{:d}x")
    assert isinstance(p, Parser)
    assert p.parse("12x").fixed == (12,)
    assert p.parse("12y") is None
    assert p.format == "{:d}x"


def test_compile_case_sensitive():
    p = compile("A{}", case_sensitive=True)
    assert p.parse("a1") is None
    assert p.parse("A1").fixed == ("1",)
    assert p._re_flags == re.DOTALL
    assert compile("A{}")._re_flags == re.IGNORECASE | re.DOTALL


def test_compile_extra_types():
    num = with_pattern(r"\d+")(lambda t: int(t))
    p = compile("{:num}", extra_types={"num": num})
    assert p.parse("12").fixed == (12,)
    assert p._extra_types == {"num": num}
    assert Parser("x")._extra_types == {}


def test_parser_parse_evaluate_false():
    p = compile("{:d}")
    m = p.parse("5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.parser is p
    assert m.evaluate_result().fixed == (5,)
    assert p.parse("x", evaluate_result=False) is None


def test_parse_function_evaluate_false():
    m = parse.parse("{:d}", "5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.evaluate_result()[0] == 5


def test_parser_search_method():
    p = compile("{:d}")
    assert p.search("ab 12 cd").fixed == (12,)
    assert p.search("ab 12 cd", pos=4).fixed == (2,)
    assert p.search("ab 12 cd", endpos=4).fixed == (1,)
    assert p.search("ab") is None
    m = p.search("a 3", evaluate_result=False)
    assert isinstance(m, Match)


def test_parser_repr():
    assert repr(compile("{}")) == "<Parser '{}'>"
    short = "a" * 20
    assert repr(compile(short)) == "<Parser %r>" % short
    long = "b" * 21
    assert repr(compile(long)) == "<Parser %r>" % ("b" * 17 + "...")


def test_named_fixed_fields_copies():
    p = compile("{a} {} {b}")
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]
    p.named_fields.append("zzz")
    p.fixed_fields.append(9)
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]


def test_fixed_fields_indices_with_groups():
    p = compile("{:g} {}")
    assert p.fixed_fields == [0, 3]
    p = compile("{:ti} {}")
    assert p.fixed_fields == [0, 8]
    p = compile("{:tg} {}")
    assert p.fixed_fields == [0, 10]
    p = compile("{:ta} {}")
    assert p.fixed_fields == [0, 10]
    p = compile("{:te} {}")
    assert p.fixed_fields == [0, 9]
    p = compile("{:th} {}")
    assert p.fixed_fields == [0, 7]
    p = compile("{:tc} {}")
    assert p.fixed_fields == [0, 9]
    p = compile("{:tt} {}")
    assert p.fixed_fields == [0, 6]
    p = compile("{:ts} {}")
    assert p.fixed_fields == [0, 6]
    p = compile("{:n} {}")
    assert p.fixed_fields == [0, 2]
    p = compile("{:b} {:o} {:x} {:%} {}")
    assert p.fixed_fields == [0, 2, 4, 6, 8]
    p = compile("{:f} {:e} {:d} {:w} {:l} {}")
    assert p.fixed_fields == [0, 1, 2, 3, 4, 5]
    p = compile("{} {}")
    assert p.fixed_fields == [0, 1]


def test_regex_cached():
    p = compile("{:d}")
    assert p._search_re is p._search_re
    assert p._match_re is p._match_re
    assert p._match_re.pattern == r"\A%s\Z" % p._expression
    assert p._search_re.pattern == p._expression


def test_match_and_search_re_flags():
    p = compile("a", case_sensitive=True)
    assert p._match_re.flags & re.IGNORECASE == 0
    assert p._search_re.flags & re.IGNORECASE == 0
    assert p._search_re.flags & re.DOTALL
    p = compile("a")
    assert p._search_re.flags & re.IGNORECASE
    assert p._match_re.flags & re.IGNORECASE


def test_many_fields():
    # Python 3.13 allows lots of named groups
    fmt = " ".join("{a%d}" % i for i in range(150))
    s = " ".join(str(i) for i in range(150))
    r = parse.parse(fmt, s)
    assert r["a0"] == "0"
    assert r["a149"] == "149"


def test_bad_group_name_raises_not_implemented():
    p = Parser("x")
    p._expression = "(?P<1bad>x)"
    with pytest.raises(NotImplementedError):
        p._match_re


def test_evaluate_result_direct():
    p = compile("{a:d}-{}")
    m = p._match_re.match("3-x")
    r = p.evaluate_result(m)
    assert isinstance(r, Result)
    assert r.fixed == ("x",)
    assert r.named == {"a": 3}


def test_type_conversions_recorded():
    p = compile("{:d} {} {a:f}")
    assert set(p._type_conversions) == {0, "a"}


# ---------------------------------------------------------------- Result / Match


def test_result_object():
    r = Result(("a", "b"), {"x": 1}, {0: (0, 1)})
    assert r[0] == "a"
    assert r[1:] == ("b",)
    assert r["x"] == 1
    assert "x" in r
    assert "y" not in r
    assert repr(r) == "<Result ('a', 'b') {'x': 1}>"
    with pytest.raises(KeyError):
        r["y"]
    with pytest.raises(IndexError):
        r[5]


def test_result_repr_from_parse():
    r = parse.parse("{} {n}", "a b")
    assert repr(r) == "<Result ('a',) {'n': 'b'}>"


def test_match_object():
    p = compile("{:d}")
    mo = p._match_re.match("4")
    m = Match(p, mo)
    assert m.parser is p
    assert m.match is mo
    assert m.evaluate_result().fixed == (4,)


# ---------------------------------------------------------------- misc regex details


def test_parse_re_splitting():
    assert parse.PARSE_RE.split("a{b}c{{d}}") == ["a", "{b}", "c", "{{", "d", "}}", ""]
    assert parse.parse("{a.b[c]}", "v").named == {"a.b": {"c": "v"}}


def test_field_with_colon_in_format():
    r = parse.parse("{:%H:%M}", "10:20")
    assert r[0] == time(10, 20)


def test_empty_format_and_string():
    r = parse.parse("", "")
    assert r.fixed == ()
    assert parse.parse("", "x") is None


def test_regex_safety_pattern():
    assert parse.REGEX_SAFETY.sub(lambda m: "\\" + m.group(1), "a.b") == "a\\.b"
    p = Parser("")
    m = re.match("(.)", "*")
    assert p._regex_replace(m) == "\\*"


def test_lazy_fields():
    r = parse.parse("{}{}", "abc")
    assert r.fixed == ("a", "bc")


def test_with_pattern_returns_same_function():
    def f(t):
        return t

    assert with_pattern("x")(f) is f


def test_parse_named_numeric_with_alignment_and_spans():
    r = parse.parse("[{n:>4d}]", "[  12]")
    assert r.named == {"n": 12}
    assert r.spans == {"n": (3, 5)}


def test_d_width_regex():
    assert "{1,3}" in compile("{:3d}")._expression
    assert "{1,3}" not in compile("{:d}")._expression


def test_precision_zero_regex():
    assert r"(?:\d+|nan" in compile("{:.0f}")._expression
    assert r"(?:\d*\.\d+|nan" in compile("{:.1f}")._expression
    assert r"(?:\d*\.\d+|nan" in compile("{:f}")._expression


# ---------------------------------------------------------------- targeted edge cases


def test_int_convert_prefix_with_letter_digit():
    ic = int_convert()
    assert ic("0xa", None) == 10
    assert ic("-0xa", None) == -10
    assert ic("0Xb", None) == 11


ALL_MONTHS = [
    ("Jan", "January", 1),
    ("Feb", "February", 2),
    ("Mar", "March", 3),
    ("Apr", "April", 4),
    ("May", "May", 5),
    ("Jun", "June", 6),
    ("Jul", "July", 7),
    ("Aug", "August", 8),
    ("Sep", "September", 9),
    ("Oct", "October", 10),
    ("Nov", "November", 11),
    ("Dec", "December", 12),
]


@pytest.mark.parametrize("short,long,num", ALL_MONTHS)
def test_month_names(short, long, num):
    assert parse.MONTHS_MAP[short] == num
    assert parse.MONTHS_MAP[long] == num
    r = parse.parse("{:tg} {}", "5/%s/2023 10:20 PM x" % short)
    assert r.fixed == (datetime(2023, num, 5, 22, 20), "x")
    r = parse.parse("{:tg} {}", "5/%s/2023 x" % long)
    assert r.fixed == (datetime(2023, num, 5), "x")
    r = parse.parse("{:ta}", "%s-5-2023" % long)
    assert r[0] == datetime(2023, num, 5)
    r = parse.parse("{:te}", "5 %s 2023 01:02:03 +0000" % short)
    assert r[0].replace(tzinfo=None) == datetime(2023, num, 5, 1, 2, 3)
    r = parse.parse("{:th}", "5/%s/2023:01:02:03 +0000" % short)
    assert r[0].replace(tzinfo=None) == datetime(2023, num, 5, 1, 2, 3)


def test_months_map_size():
    assert len(parse.MONTHS_MAP) == 23


def test_date_convert_only_mm_falls_through():
    m = re.match(r"(.*)\|(.*)", "2020-01-02|Mar")
    assert date_convert(None, m, mm=1, ymd=0) == datetime(2020, 1, 2)
    m = re.match(r"(.*)\|(.*)", "2020-01-02|7")
    assert date_convert(None, m, dd=1, ymd=0) == datetime(2020, 1, 2)


def test_tz_offset_two_digit_hours():
    m = re.match(r"(.*)\|(.*)", "10:20|+1030")
    assert date_convert(None, m, hms=0, tz=1).utcoffset() == timedelta(hours=10, minutes=30)
    m = re.match(r"(.*)\|(.*)", "10:20|-1145")
    assert date_convert(None, m, hms=0, tz=1).utcoffset() == -timedelta(hours=11, minutes=45)
    m = re.match(r"(.*)\|(.*)", "10:20|+945")
    assert date_convert(None, m, hms=0, tz=1).utcoffset() == timedelta(hours=9, minutes=45)


def test_strf_date_convert_unknown_directive_message():
    with pytest.raises(ValueError, match="Datetime not a date nor a time"):
        strf_date_convert("10:20:30", None, "%X")
    with pytest.raises(ValueError, match="Datetime not a date nor a time"):
        strf_date_convert("x", None, "x")


def test_extract_format_X_is_not_align():
    with pytest.raises(ValueError, match="not recognised"):
        extract_format("aX", {})
    with pytest.raises(ValueError, match=r"format spec 'zz' not recognised"):
        extract_format("zz", {})


def test_debug_logging(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="parse"):
        compile("{:d}")
    assert any(
        rec.getMessage() == "format '{:d}' -> %r" % compile("{:d}")._expression
        for rec in caplog.records
    )


def test_not_implemented_message():
    p = Parser("x")
    p._expression = "(?P<1bad>x)"
    with pytest.raises(NotImplementedError) as ei:
        p._match_re
    assert str(ei.value) == (
        "Group names (e.g. (?P<name>) can cause failure, as they are not "
        "escaped properly: '\\A(?P<1bad>x)\\Z'"
    )


def test_literal_partial_braces():
    assert parse.parse("{a", "{a").fixed == ()
    assert parse.parse("{a", "xy") is None
    assert parse.parse("a}", "a}").named == {}
    assert parse.parse("a}", "xy") is None


def test_bracket_group_name():
    p = Parser("{a[b]}")
    assert p._name_to_group_map == {"a[b]": "a_b_"}
    p = Parser("{a.b}")
    assert p._name_to_group_map == {"a.b": "a_b"}


def test_error_messages():
    p = Parser("{ab}")
    with pytest.raises(KeyError, match="duplicated group name 'ab'"):
        p._to_group_name("ab")
    with pytest.raises(RepeatedNameError) as ei:
        compile("{n:d} {n:w}")
    assert str(ei.value) == (
        "field type 'w' for field \"n\" does not match previous seen type 'd'"
    )


def test_fill_X_letter():
    assert parse.parse("{:X<}|", "abXX|").fixed == ("ab",)
    assert parse.parse("{:Z>}|", "ZZab|").fixed == ("ab",)


@pytest.mark.parametrize(
    "fmt,text,expected",
    [
        ("{:ti}", "2023-04-05T10:20:30+01:00", datetime(2023, 4, 5, 10, 20, 30)),
        ("{:tg}", "5/4/2023 10:20:30 PM +0100", datetime(2023, 4, 5, 22, 20, 30)),
        ("{:ta}", "4/5/2023 10:20:30 PM +0100", datetime(2023, 4, 5, 22, 20, 30)),
        ("{:te}", "Wed, 5 Apr 2023 10:20:30 +0100", datetime(2023, 4, 5, 10, 20, 30)),
        ("{:th}", "05/Apr/2023:10:20:30 +0100", datetime(2023, 4, 5, 10, 20, 30)),
        ("{:tc}", "Wed Apr 5 10:20:30 2023", datetime(2023, 4, 5, 10, 20, 30)),
        ("{:tt}", "10:20:30 PM +0100", time(22, 20, 30)),
    ],
)
def test_datetime_types_after_other_groups(fmt, text, expected):
    # preceding fields shift the group indices used by date_convert
    full = "{:g} {:n} {a} " + fmt + " {:d} {b}"
    r = parse.parse(full, "1.5 1,000 q " + text + " 7 z")
    assert r.fixed[0] == 1.5
    assert r.fixed[1] == 1000
    v = r.fixed[2]
    if fmt not in ("{:tc}",):
        assert v.utcoffset() == timedelta(hours=1)
    assert v.replace(tzinfo=None) == expected
    assert r.fixed[3] == 7
    assert r.named == {"a": "q", "b": "z"}


def test_ts_after_other_groups():
    year = datetime.today().year
    r = parse.parse("{:g} {a} {:ts} {:d}", "1.5 q Nov  9 03:37:44 7")
    assert r.fixed == (1.5, datetime(year, 11, 9, 3, 37, 44), 7)
    assert r.named == {"a": "q"}


def test_exact_error_messages():
    with pytest.raises(ValueError) as ei:
        strf_date_convert("x", None, "x")
    assert str(ei.value) == "Datetime not a date nor a time?"
    with pytest.raises(ValueError) as ei:
        extract_format("zz", {})
    assert str(ei.value) == "format spec 'zz' not recognised"
    with pytest.raises(KeyError) as ei:
        Parser("{ab}")._to_group_name("ab")
    assert ei.value.args == ("duplicated group name 'ab'",)


def test_parser_defaults():
    p = Parser("A{}")
    assert p.parse("a1").fixed == ("1",)
    p = Parser("{:d}")
    assert p.search("12").fixed == (12,)
    assert isinstance(p.search("12"), Result)
    it = p.findall("1 2")
    assert it.pos == 0
    assert it.evaluate_result is True
    assert [r[0] for r in it] == [1, 2]
    it = ResultIterator(p, "3 4", 0, 3)
    assert it.evaluate_result is True
    assert [r.fixed for r in it] == [(3,), (4,)]


def test_module_functions_default_extra_types():
    assert search("{:5}|", "abcde|").fixed == ("abcde",)
    assert [r[0] for r in findall("{:2}|", "ab|cd|")] == ["ab", "cd"]
    assert parse.parse("{:5}", "abcde").fixed == ("abcde",)
    assert compile("{:5}").parse("abcde").fixed == ("abcde",)
    assert search("{:d}", "x12").fixed == (12,)
    assert isinstance(search("{:d}", "x12"), Result)
    assert isinstance(next(findall("{:d}", "x12")), Result)


class _FakeRe:
    """Stands in for the re module to simulate old 100-named-group limit."""

    error = re.error
    DOTALL = re.DOTALL
    IGNORECASE = re.IGNORECASE

    def __init__(self, message):
        self.message = message

    def compile(self, *args):
        raise AssertionError(self.message)


@pytest.mark.parametrize("attr", ["_search_re", "_match_re"])
def test_too_many_fields(monkeypatch, attr):
    p = Parser("{a}")
    monkeypatch.setattr(
        parse, "re", _FakeRe("sorry, but this version only supports 100 named groups")
    )
    with pytest.raises(TooManyFields) as ei:
        getattr(p, attr)
    assert str(ei.value) == "sorry, you are attempting to parse too many complex fields"


@pytest.mark.parametrize("attr", ["_search_re", "_match_re"])
def test_other_assertion_not_too_many_fields(monkeypatch, attr):
    p = Parser("{a}")
    monkeypatch.setattr(parse, "re", _FakeRe("something else"))
    assert getattr(p, attr) is None
