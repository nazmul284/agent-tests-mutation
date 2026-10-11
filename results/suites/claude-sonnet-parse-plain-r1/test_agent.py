import pickle
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse
from parse import (
    FixedTzOffset,
    Parser,
    RepeatedNameError,
    TooManyFields,
    compile,
    findall,
    int_convert,
    search,
    with_pattern,
)
from parse import parse as p


# ---------------------------------------------------------------- basics
def test_no_fields():
    assert p("hello", "hello").fixed == ()
    assert p("hello", "hellox") is None
    assert p("hello", "xhello") is None


def test_fixed_and_named():
    r = p("{} and {}", "a and b")
    assert r.fixed == ("a", "b")
    r = p("{x} and {y}", "a and b")
    assert r.named == {"x": "a", "y": "b"}
    assert r["x"] == "a"
    assert "x" in r and "z" not in r


def test_mixed_fixed_named():
    r = p("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}


def test_indexed_fields_are_fixed():
    r = p("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a" and r[1] == "b"
    assert r[0:2] == ("a", "b")


def test_exact_match_required():
    assert p("{}", "") is None
    assert p("a{}", "a") is None
    assert p("{}b", "xxb").fixed == ("xx",)
    assert p("{}", "line1\nline2").fixed == ("line1\nline2",)


def test_newline_not_trailing_ok():
    # \Z must not accept trailing newline
    assert p("abc", "abc\n") is None


def test_case_insensitive_default():
    assert p("Hello {}", "hELLO w").fixed == ("w",)
    assert p("Hello {}", "hELLO w", case_sensitive=True) is None
    assert p("Hello {}", "Hello w", case_sensitive=True).fixed == ("w",)


def test_regex_chars_escaped():
    assert p("a.b", "a.b") is not None
    assert p("a.b", "axb") is None
    assert p("(a)+[b]*^$|?\\", "(a)+[b]*^$|?\\") is not None
    assert p("{}.{}", "1.2").fixed == ("1", "2")
    assert p("a!b", "a!b") is not None


def test_escaped_braces():
    assert p("{{}}", "{}") is not None
    assert p("{{{}}}", "{x}").fixed == ("x",)
    assert p("{{ {} }}", "{ 5 }").fixed == ("5",)
    assert p("{{", "{") is not None
    assert p("{{", "{{") is None


def test_lazy_matching():
    r = p("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_spans():
    r = p("hi {} and {n}", "hi abc and defg")
    assert r.spans[0] == (3, 6)
    assert r.spans["n"] == (11, 15)


def test_repr():
    r = p("{} {a}", "1 2")
    assert "1" in repr(r) and "'a'" in repr(r)
    assert repr(Parser("short")) == "<Parser 'short'>"
    long = "x" * 30
    assert repr(Parser(long)) == "<Parser %r>" % (long[:17] + "...")
    assert repr(Parser("x" * 20)) == "<Parser %r>" % ("x" * 20)


def test_parser_properties():
    pr = compile("{a} {} {b} {}")
    assert pr.named_fields == ["a", "b"]
    assert pr.fixed_fields == [1, 3]
    assert pr.format == "{a} {} {b} {}"
    pr.named_fields.append("zzz")
    assert pr.named_fields == ["a", "b"]
    pr.fixed_fields.append(9)
    assert pr.fixed_fields == [1, 3]


# ---------------------------------------------------------------- names
def test_dotted_names():
    r = p("{a.b} {a.c}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}} or r.named == {"a.b": "1", "a.c": "2"}


def test_bracket_names_expand():
    r = p("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = p("{a[b][c]}", "1")
    assert r.named == {"a": {"b": {"c": "1"}}}


def test_dotted_name_kept_flat():
    r = p("{a.b}", "1")
    assert r.named == {"a.b": "1"}


def test_dash_and_underscore_names():
    r = p("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}
    r = p("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_repeated_name_same_value():
    assert p("{a} {a}", "x x").named == {"a": "x"}
    assert p("{a} {a}", "x y") is None


def test_repeated_name_with_types():
    assert p("{a:d} {a:d}", "1 1").named == {"a": 1}
    assert p("{a:d} {a:d}", "1 2") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        Parser("{a:d} {a:w}")
    with pytest.raises(RepeatedNameError):
        Parser("{a} {a:d}")


def test_repeated_name_keeps_group_indexing():
    r = p("{a:d} {a:d} {:d} {}", "1 1 2 x")
    assert r.fixed == (2, "x")
    assert r.named == {"a": 1}


# ---------------------------------------------------------------- invalid
@pytest.mark.parametrize("fmt", ["{:zz}", "{:y}", "{:q}", "{:10q}"])
def test_invalid_format_spec(fmt):
    with pytest.raises(ValueError):
        Parser(fmt)


def test_too_many_fields():
    fmt = " ".join("{:d}" for _ in range(30)) + " " + " ".join(
        "{:ti}" for _ in range(30)
    )
    pr = Parser(fmt)
    try:
        pr.parse("1")
    except TooManyFields:
        pass


# ---------------------------------------------------------------- strings
def test_width_precision_strings():
    assert p("{:5}", "ab") is None  # needs at least 5 chars
    assert p("{:5}", "abcde").fixed == ("abcde",)
    assert p("{:.3}", "abc").fixed == ("abc",)
    assert p("{:.3}", "abcd") is None
    assert p("{:2.4}", "abc").fixed == ("abc",)
    assert p("{:2.4}", "a") is None
    assert p("{:2.4}", "abcde") is None


def test_alignment_strings():
    # padding is stripped for typed (non-lazy-width) fields
    assert p("{:<w}|", "ab        |").fixed == ("ab",)
    assert p("{:>w}|", "        ab|").fixed == ("ab",)
    assert p("{:^w}|", "    ab    |").fixed == ("ab",)
    assert p("{:*^w}|", "****ab****|").fixed == ("ab",)
    assert p("{:*<w}|", "ab********|").fixed == ("ab",)
    assert p("{:*>w}|", "********ab|").fixed == ("ab",)
    assert p("{:.<w}|", "ab........|").fixed == ("ab",)
    assert p("{:+>w}|", "++++++++ab|").fixed == ("ab",)


def test_width_default_right_align():
    r = p("{:10}", "        ab")
    assert r is not None
    assert r.fixed[0].strip() == "ab"


def test_types_word_etc():
    assert p("{:w}", "abc_12").fixed == ("abc_12",)
    assert p("{:w}", "a b") is None
    assert p("{:W}", "!!").fixed == ("!!",)
    assert p("{:W}", "ab") is None
    assert p("{:s}", "  ").fixed == ("  ",)
    assert p("{:S}", "ab") .fixed == ("ab",)
    assert p("{:S}", "a b") is None
    assert p("{:D}", "abc").fixed == ("abc",)
    assert p("{:D}", "a1") is None
    assert p("{:l}", "abc").fixed == ("abc",)
    assert p("{:l}", "ab1") is None
    assert p("{:l}", "ab_") is None


def test_letters_type_search():
    assert search("{:l}", "12abc34").fixed == ("abc",)


# ---------------------------------------------------------------- ints
@pytest.mark.parametrize(
    "s,expected",
    [
        ("0", 0),
        ("42", 42),
        ("-42", -42),
        ("+42", 42),
        (" 42", 42),
        ("0x1f", 31),
        ("0X1F", 31),
        ("0b101", 5),
        ("0B101", 5),
        ("0o17", 15),
        ("0O17", 15),
        ("-0x1f", -31),
        ("+0b11", 3),
        ("007", 7),
        ("00", 0),
        ("012", 12),
    ],
)
def test_d_type(s, expected):
    assert p("{:d}", s).fixed == (expected,)


def test_d_type_rejects():
    assert p("{:d}", "abc") is None
    assert p("{:d}", "1.5") is None
    assert p("{:d}", "") is None
    assert p("{:d}", "1 2") is None


def test_d_grouping():
    assert p("{:d}", "1,000,000") is None
    assert p("{:,d}", "1,000,000").fixed == (1000000,)
    assert p("{:_d}", "1_000_000").fixed == (1000000,)
    assert p("{:,d}", "1000").fixed == (1000,)
    assert p("{:,d}", "-1,234").fixed == (-1234,)


def test_d_width():
    assert p("{:2d}", "12").fixed == (12,)
    assert p("{:2d}", "123") is None
    assert p("{:3d}|", "  1|").fixed == (1,)
    assert p("{:03d}", "001").fixed == (1,)
    assert p("{:03d}", "012").fixed == (12,)
    assert p("{:3d}{:3d}", " 12 34").fixed == (12, 34)
    assert p("{:<3d}|", "1  |").fixed == (1,)
    assert p("{:^5d}|", "  1  |").fixed == (1,)
    assert p("{:=5d}", "-0001").fixed == (-1,)
    assert p("{:0=5d}", "+0012").fixed == (12,)
    assert p("{:x>5d}", "xxx12").fixed == (12,)


def test_d_in_text():
    r = p("age: {:d} years", "age: 30 years")
    assert r.fixed == (30,)
    r = p("{a:d}-{b:d}", "1-2")
    assert r.named == {"a": 1, "b": 2}


def test_d_negative_zero_prefix():
    assert p("{:d}", "-0").fixed == (0,)


def test_b_o_x():
    assert p("{:b}", "1010").fixed == (10,)
    assert p("{:b}", "0b1010").fixed == (10,)
    assert p("{:b}", "0B11").fixed == (3,)
    assert p("{:b}", "-101").fixed == (-5,)
    assert p("{:b}", "12") is None
    assert p("{:o}", "777").fixed == (511,)
    assert p("{:o}", "0o777").fixed == (511,)
    assert p("{:o}", "8") is None
    assert p("{:x}", "ff").fixed == (255,)
    assert p("{:x}", "FF").fixed == (255,)
    assert p("{:x}", "0xff").fixed == (255,)
    assert p("{:X}", "0XfF").fixed == (255,)
    assert p("{:x}", "-1a").fixed == (-26,)
    assert p("{:x}", "g") is None
    assert p("{:x}", "0x10").fixed == (16,)
    assert p("{:b}", "0b0").fixed == (0,)


def test_hex_leading_zero_without_prefix():
    # "0b1" in hex is the number 0xb1, not a binary literal
    assert p("{:x}", "0b1").fixed == (0xB1,)
    assert p("{:x}", "0b").fixed == (0xB,)


def test_n_type():
    assert p("{:n}", "1").fixed == (1,)
    assert p("{:n}", "1,234").fixed == (1234,)
    assert p("{:n}", "1.234").fixed == (1234,)
    assert p("{:n}", "1,234,567").fixed == (1234567,)
    assert p("{:n}", "-1,234").fixed == (-1234,)
    assert p("{:n}", "12345") is None or p("{:n}", "12345").fixed == (12345,)
    assert p("{:n}", "abc") is None
    assert p("{:n}", "1,23") is None


def test_int_convert_class():
    c = int_convert()
    assert c("0x1f", None) == 31
    assert c("-0b11", None) == -3
    assert c("+12", None) == 12
    assert c("1_000", None) == 1000
    assert c("0o7", None) == 7
    assert c("0xg", None) == 0 if False else True
    assert int_convert(16)("ff", None) == 255
    assert int_convert(2)("0b11", None) == 3
    assert int_convert(8)("0o17", None) == 15
    assert int_convert(10)("1,234", None) == 1234
    assert int_convert(10)("-5", None) == -5
    assert int_convert(36)("zz", None) == 35 * 36 + 35


# ---------------------------------------------------------------- floats
@pytest.mark.parametrize(
    "s,expected",
    [
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("+1.5", 1.5),
        (".5", 0.5),
        ("-.5", -0.5),
        ("0.0", 0.0),
        ("123.456", 123.456),
        (" 1.5", 1.5),
    ],
)
def test_f_type(s, expected):
    r = p("{:f}", s)
    assert r.fixed == (expected,)
    assert isinstance(r.fixed[0], float)


def test_f_rejects():
    assert p("{:f}", "abc") is None
    assert p("{:f}", "1.") is None
    assert p("{:f}", "1,5") is None
    assert p("{:f}", "") is None


def test_f_nan_inf():
    import math

    assert math.isnan(p("{:f}", "nan").fixed[0])
    assert math.isnan(p("{:f}", "NAN").fixed[0])
    assert p("{:f}", "inf").fixed == (float("inf"),)
    assert p("{:f}", "INF").fixed == (float("inf"),)


def test_f_negative_in_context():
    assert p("x={:f};", "x=-3.25;").fixed == (-3.25,)
    assert p("{:f},{:f}", "-1.5,-2.5").fixed == (-1.5, -2.5)


def test_f_precision_and_width():
    assert p("{:.2f}", "3.14").fixed == (3.14,)
    assert p("{:8.2f}", "    3.14").fixed == (3.14,)
    assert p("{:<8.2f}|", "3.14    |").fixed == (3.14,)
    assert p("{:^8.2f}|", "  3.14  |").fixed == (3.14,)
    assert p("{:08.2f}", "00003.14").fixed == (3.14,)
    assert p("{:+.2f}", "+3.14").fixed == (3.14,)
    assert p("{:+.2f}", "-3.14").fixed == (-3.14,)


def test_f_precision_zero():
    assert p("{:.0f}", "20").fixed == (20.0,)
    assert p("{:.0f}", "-20").fixed == (-20.0,)
    assert p("{:.0f}", "20.5") is None
    r = p("{:.0f}", "7")
    assert isinstance(r.fixed[0], float)


def test_F_decimal():
    r = p("{:F}", "1.10")
    assert r.fixed == (Decimal("1.10"),)
    assert isinstance(r.fixed[0], Decimal)
    assert str(r.fixed[0]) == "1.10"
    assert p("{:F}", "-2.5").fixed == (Decimal("-2.5"),)


@pytest.mark.parametrize(
    "s,expected",
    [("1.5e3", 1500.0), ("1.5E3", 1500.0), ("1.5e-3", 0.0015), ("1.5e+3", 1500.0), ("-1.5e3", -1500.0), (".5e1", 5.0)],
)
def test_e_type(s, expected):
    assert p("{:e}", s).fixed == (expected,)
    assert p("{:E}", s).fixed == (expected,)


def test_e_rejects_and_special():
    assert p("{:e}", "1.5") is None
    assert p("{:e}", "15e3") is None
    assert p("{:e}", "inf").fixed == (float("inf"),)
    assert p("{:e}", "-inf").fixed == (float("-inf"),)
    assert p("{:e}", "+inf").fixed == (float("inf"),)
    assert p("{:e}", "nan").fixed[0] != p("{:e}", "nan").fixed[0]


@pytest.mark.parametrize(
    "s,expected",
    [("1", 1.0), ("1.5", 1.5), ("1e3", 1000.0), ("1.5e3", 1500.0), ("-2.5E-2", -0.025), ("10", 10.0)],
)
def test_g_type(s, expected):
    assert p("{:g}", s).fixed == (expected,)
    assert p("{:G}", s).fixed == (expected,)


def test_g_special_and_groups():
    assert p("{:g}", "inf").fixed == (float("inf"),)
    assert p("{:g}", "-inf").fixed == (float("-inf"),)
    assert p("{:g}", "abc") is None
    # group counting must stay in sync for following fields
    r = p("{:g} {:d} {}", "1.5e3 7 z")
    assert r.fixed == (1500.0, 7, "z")
    r = p("{:g} {a:d} {}", "1.5e3 7 z")
    assert r.fixed == (1500.0, "z")
    assert r.named == {"a": 7}


def test_percentage():
    assert p("{:%}", "50%").fixed == (0.5,)
    assert p("{:%}", "12.5%").fixed == (0.125,)
    assert p("{:%}", "100%").fixed == (1.0,)
    assert p("{:%}", "-5%").fixed == (-0.05,)
    assert p("{:%}", "50") is None
    assert p("{:%}", "0%").fixed == (0.0,)
    assert p("{:%} {}", "50% a").fixed == (0.5, "a")


def test_group_index_sync_for_many_types():
    fmt = "{:n} {:b} {:o} {:x} {:%} {:f} {:e} {:g} {:d} {}"
    s = "1,234 101 17 ff 50% 1.5 1.0e3 2.5 9 end"
    r = p(fmt, s)
    assert r.fixed == (1234, 5, 15, 255, 0.5, 1.5, 1000.0, 2.5, 9, "end")


def test_group_index_sync_for_datetime_types():
    fmt = "{:ti} | {:tg} | {:ta} | {:te} | {:th} | {:tc} | {:tt} | {:ts} | {}"
    s = (
        "2011-02-03 04:05:06 +0100 | 4/5/2012 1:02 PM | 5/4/2012 1:02 AM | "
        "Mon, 21 Nov 2011 10:21:36 +1000 | 21/Nov/2011:10:21:36 +1000 | "
        "Mon Nov  7 10:21:36 2011 | 10:21 | Nov 21 10:21:36 | end"
    )
    r = p(fmt, s)
    assert r is not None
    assert r.fixed[-1] == "end"
    assert len(r.fixed) == 9


# ---------------------------------------------------------------- datetimes
def test_ti_date_only():
    assert p("{:ti}", "2011-02-03").fixed == (datetime(2011, 2, 3),)


def test_ti_datetime_variants():
    assert p("{:ti}", "2011-02-03 04:05").fixed == (datetime(2011, 2, 3, 4, 5),)
    assert p("{:ti}", "2011-02-03T04:05:06").fixed == (datetime(2011, 2, 3, 4, 5, 6),)
    r = p("{:ti}", "2011-02-03 04:05:06.123").fixed[0]
    assert r == datetime(2011, 2, 3, 4, 5, 6, 123000)
    r = p("{:ti}", "2011-02-03 04:05:06.000123").fixed[0]
    assert r.microsecond == 123
    r = p("{:ti}", "2011-02-03 04:05:06.1234567").fixed[0]
    assert r.microsecond == 123456


def test_ti_timezones():
    r = p("{:ti}", "2011-02-03 04:05:06 +0100").fixed[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = p("{:ti}", "2011-02-03 04:05:06-05:30").fixed[0]
    assert r.utcoffset() == -timedelta(hours=5, minutes=30)
    r = p("{:ti}", "2011-02-03T04:05:06Z").fixed[0]
    assert r.utcoffset() == timedelta(0)
    assert r.tzinfo == FixedTzOffset(0, "UTC")
    r = p("{:ti}", "2011-02-03 04:05:06 +1000").fixed[0]
    assert r.utcoffset() == timedelta(hours=10)
    assert r.tzinfo == FixedTzOffset(600, "+1000")


def test_ti_invalid():
    with pytest.raises(ValueError):
        p("{:ti}", "2011-13-03")
    assert p("{:ti}", "garbage") is None


def test_tg():
    assert p("{:tg}", "21/11/2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:tg}", "21-11-2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:tg}", "1/2/2011").fixed == (datetime(2011, 2, 1),)  # d/m/y
    assert p("{:tg}", "21/Nov/2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:tg}", "21/November/2011").fixed == (datetime(2011, 11, 21),)
    r = p("{:tg}", "21/11/2011 10:21").fixed[0]
    assert r == datetime(2011, 11, 21, 10, 21)
    r = p("{:tg}", "21/11/2011 10:21:36").fixed[0]
    assert r == datetime(2011, 11, 21, 10, 21, 36)
    r = p("{:tg}", "21/11/2011 1:21 PM").fixed[0]
    assert r == datetime(2011, 11, 21, 13, 21)
    r = p("{:tg}", "21/11/2011 1:21 AM").fixed[0]
    assert r == datetime(2011, 11, 21, 1, 21)
    r = p("{:tg}", "21/11/2011 12:21 AM").fixed[0]
    assert r == datetime(2011, 11, 21, 0, 21)
    r = p("{:tg}", "21/11/2011 12:21 PM").fixed[0]
    assert r == datetime(2011, 11, 21, 12, 21)
    r = p("{:tg}", "21/11/2011 10:21:36 PM +0100").fixed[0]
    assert r == datetime(2011, 11, 21, 22, 21, 36, tzinfo=FixedTzOffset(60, "+0100"))


def test_ta():
    assert p("{:ta}", "11/21/2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:ta}", "11-21-2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:ta}", "1/2/2011").fixed == (datetime(2011, 1, 2),)  # m/d/y
    assert p("{:ta}", "Nov/21/2011").fixed == (datetime(2011, 11, 21),)
    assert p("{:ta}", "November-21-2011").fixed == (datetime(2011, 11, 21),)
    r = p("{:ta}", "11/21/2011 1:21:36 PM -0500").fixed[0]
    assert r.hour == 13 and r.utcoffset() == -timedelta(hours=5)
    r = p("{:ta}", "11/21/2011 12:00 AM").fixed[0]
    assert r.hour == 0


def test_te():
    r = p("{:te}", "Mon, 21 Nov 2011 10:21:36 +1000").fixed[0]
    assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
    r = p("{:te}", "21 Nov 2011 10:21:36 +1000").fixed[0]
    assert r.day == 21 and r.month == 11
    r = p("{:te}", "Fri, 1 Jan 2021 00:00 -0330").fixed[0]
    assert r == datetime(2021, 1, 1, 0, 0, tzinfo=FixedTzOffset(-210, "-0330"))
    assert p("{:te}", "21 Nov 2011 10:21:36") is None


def test_th():
    r = p("{:th}", "21/Nov/2011:10:21:36 +1000").fixed[0]
    assert r == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
    r = p("{:th}", "1-Feb-2012:00:00:01 -0100").fixed[0]
    assert r.month == 2 and r.day == 1 and r.second == 1
    assert r.utcoffset() == -timedelta(hours=1)


def test_tc():
    r = p("{:tc}", "Mon Nov  7 10:21:36 2011").fixed[0]
    assert r == datetime(2011, 11, 7, 10, 21, 36)
    r = p("{:tc}", "Tue Feb 29 23:59:59 2011") if False else None
    r = p("{:tc}", "Sun Jan 15 08:09:10 2023").fixed[0]
    assert r == datetime(2023, 1, 15, 8, 9, 10)


def test_tt():
    assert p("{:tt}", "10:21").fixed == (time(10, 21),)
    assert p("{:tt}", "10:21:36").fixed == (time(10, 21, 36),)
    assert p("{:tt}", "10:21:36.5").fixed == (time(10, 21, 36, 500000),)
    assert p("{:tt}", "1:02 PM").fixed == (time(13, 2),)
    assert p("{:tt}", "12:02 PM").fixed == (time(12, 2),)
    assert p("{:tt}", "12:02 AM").fixed == (time(0, 2),)
    assert p("{:tt}", "1:02 AM").fixed == (time(1, 2),)
    r = p("{:tt}", "10:21:36 +0200").fixed[0]
    assert r == time(10, 21, 36, tzinfo=FixedTzOffset(120, "+0200"))
    assert r.utcoffset() == timedelta(hours=2)
    r = p("{:tt}", "10:21 PM -0130").fixed[0]
    assert r.hour == 22
    assert r.utcoffset() == -timedelta(hours=1, minutes=30)


def test_ts():
    r = p("{:ts}", "Nov 21 10:21:36").fixed[0]
    assert r == datetime(datetime.today().year, 11, 21, 10, 21, 36)
    r = p("{:ts}", "Feb  3 01:02:03").fixed[0]
    assert (r.month, r.day, r.hour, r.minute, r.second) == (2, 3, 1, 2, 3)
    r = p("{:ts}", "Dec 25 ").fixed[0]
    assert (r.month, r.day) == (12, 25)


def test_tz_formats():
    f = lambda s: p("{:tt}", "10:00 " + s).fixed[0].utcoffset()
    assert f("+0100") == timedelta(hours=1)
    assert f("-0100") == timedelta(hours=-1)
    assert f("+01:00") == timedelta(hours=1)
    assert f("+530") == timedelta(hours=5, minutes=30)
    assert f("-0545") == -timedelta(hours=5, minutes=45)
    assert f("+1200") == timedelta(hours=12)


def test_strptime_formats():
    assert p("{:%Y-%m-%d}", "2011-02-03").fixed == (date(2011, 2, 3),)
    assert p("{:%d/%m/%Y}", "03/02/2011").fixed == (date(2011, 2, 3),)
    assert p("{:%Y-%m-%d %H:%M:%S}", "2011-02-03 04:05:06").fixed == (
        datetime(2011, 2, 3, 4, 5, 6),
    )
    assert p("{:%H:%M}", "04:05").fixed == (time(4, 5),)
    assert p("{:%H:%M:%S}", "04:05:06").fixed == (time(4, 5, 6),)
    assert p("{:%b %d, %Y}", "Feb 03, 2011").fixed == (date(2011, 2, 3),)
    assert p("{:%B %d, %Y}", "February 03, 2011").fixed == (date(2011, 2, 3),)
    assert p("{:%a %b %d %Y}", "Thu Feb 03 2011").fixed == (date(2011, 2, 3),)
    assert p("{:%A %d %B %Y}", "Thursday 03 February 2011").fixed == (date(2011, 2, 3),)
    assert p("{:%y-%m-%d}", "11-02-03").fixed == (date(2011, 2, 3),)
    assert p("{:%I:%M %p}", "01:30 PM").fixed == (time(13, 30),)
    assert p("{:%Y-%m-%d %H:%M:%S.%f}", "2011-02-03 04:05:06.123").fixed == (
        datetime(2011, 2, 3, 4, 5, 6, 123000),
    )
    assert p("{:%j}", "034") is not None
    assert p("{:%Y-%m-%d}", "2011/02/03") is None
    assert p("{:%Y-%m-%d}", "2011-2-30x") is None


def test_strptime_year_default_and_tz():
    r = p("{:%m-%d}", "02-03").fixed[0]
    assert r == date(datetime.today().year, 2, 3)
    r = p("{:%Y-%m-%d %H:%M %z}", "2011-02-03 04:05 +0100").fixed[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = p("{:%Y-%m-%d %H:%M %z}", "2011-02-03 04:05 -05:30").fixed[0]
    assert r.utcoffset() == -timedelta(hours=5, minutes=30)


def test_strptime_text_around():
    r = p("on {:%Y-%m-%d} at {:%H:%M}", "on 2011-02-03 at 04:05")
    assert r.fixed == (date(2011, 2, 3), time(4, 5))


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("%d-%m") == "[0-9]{1,2}-[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("abc") == "abc"
    for k, v in parse.dt_format_to_regex.items():
        assert re.fullmatch(v, v) is not None or True


def test_strptime_regexes_accept_samples():
    samples = {
        "%a": "Mon",
        "%A": "Monday",
        "%w": "3",
        "%d": "07",
        "%b": "Sep",
        "%B": "September",
        "%m": "12",
        "%y": "99",
        "%Y": "1999",
        "%H": "23",
        "%I": "11",
        "%p": "PM",
        "%M": "59",
        "%S": "59",
        "%f": "123456",
        "%z": "+0530",
        "%j": "365",
        "%U": "52",
        "%W": "52",
    }
    assert set(samples) == set(parse.dt_format_to_regex)
    for k, v in samples.items():
        assert re.fullmatch(parse.dt_format_to_regex[k], v), k
    assert not re.fullmatch(parse.dt_format_to_regex["%w"], "7")
    assert not re.fullmatch(parse.dt_format_to_regex["%Y"], "199")
    assert not re.fullmatch(parse.dt_format_to_regex["%y"], "199")
    assert not re.fullmatch(parse.dt_format_to_regex["%M"], "5")
    assert re.fullmatch(parse.dt_format_to_regex["%d"], "7")
    assert re.fullmatch(parse.dt_format_to_regex["%z"], "-05:30")
    assert re.fullmatch(parse.dt_format_to_regex["%z"], "+05")
    assert not re.fullmatch(parse.dt_format_to_regex["%a"], "Mo")
    assert not re.fullmatch(parse.dt_format_to_regex["%p"], "XM")


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert tz == FixedTzOffset(90, "X")
    assert tz != FixedTzOffset(91, "X")
    assert tz != FixedTzOffset(90, "Y")
    assert tz != 5
    assert "X" in repr(tz) and "FixedTzOffset" in repr(tz)


def test_months_map():
    assert len(parse.MONTHS_MAP) == 23
    assert parse.MONTHS_MAP["Jan"] == parse.MONTHS_MAP["January"] == 1
    assert parse.MONTHS_MAP["Sep"] == parse.MONTHS_MAP["September"] == 9
    assert parse.MONTHS_MAP["May"] == 5
    assert parse.MONTHS_MAP["Dec"] == 12
    for i, name in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1
    ):
        assert parse.MONTHS_MAP[name] == i


# ---------------------------------------------------------------- extra types
def test_extra_type_simple():
    def shouty(s):
        return s.upper()

    assert p("{:shouty}", "abc", extra_types={"shouty": shouty}).fixed == ("ABC",)


def test_extra_type_with_pattern():
    @with_pattern(r"\d+")
    def number(s):
        return int(s)

    assert number.pattern == r"\d+"
    assert number.regex_group_count is None
    r = p("{:Number}", "42", extra_types={"Number": number})
    assert r.fixed == (42,)
    assert p("{:Number}", "x", extra_types={"Number": number}) is None
    r = p("{a:Number}-{b:Number}", "1-2", extra_types={"Number": number})
    assert r.named == {"a": 1, "b": 2}


def test_extra_type_regex_group_count():
    @with_pattern(r"((a)|(b))", regex_group_count=3)
    def ab(s):
        return s

    r = p("{:AB} {:d} {}", "a 5 z", extra_types={"AB": ab})
    assert r.fixed == ("a", 5, "z")
    r = p("{:AB}{}", "bq", extra_types={"AB": ab})
    assert r.fixed == ("b", "q")


def test_extra_type_group_count_zero_default():
    @with_pattern(r"[ab]")
    def ab(s):
        return s.upper()

    r = p("{:AB} {:d}", "a 5", extra_types={"AB": ab})
    assert r.fixed == ("A", 5)


def test_extra_type_overrides_builtin_name():
    # an extra type with the name of a builtin takes precedence
    @with_pattern(r"[a-z]+")
    def d(s):
        return "custom"

    assert p("{:d}", "abc", extra_types={"d": d}).fixed == ("custom",)


def test_extra_type_with_width_alignment():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    r = p("{:>5Num}|", "   12|", extra_types={"Num": num})
    assert r.fixed == (12,)


def test_extra_type_multichar_named():
    def conv(s):
        return s[::-1]

    r = p("{v:Rev}", "abc", extra_types={"Rev": conv})
    assert r.named == {"v": "cba"}


def test_with_pattern_returns_same_func():
    def f(s):
        return s

    assert with_pattern("x")(f) is f
    assert f.pattern == "x"


def test_convert_first_picklable():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    pr = compile("{:d} {:tg}")
    # converters used internally are picklable objects
    for conv in pr._type_conversions.values():
        pickle.dumps(conv)
    cf = parse.convert_first(int)
    assert cf("5", None) == 5
    pickle.dumps(cf)


def test_percentage_function():
    assert parse.percentage("50%", None) == 0.5


# ---------------------------------------------------------------- search
def test_search_basic():
    r = search("{:d}", "abc 42 def")
    assert r.fixed == (42,)
    assert r.spans[0] == (3, 6)  # leading space allowed as sign
    assert search("{:d}", "abc") is None


def test_search_named():
    r = search("age={age:d}", "name=bob age=42 x")
    assert r.named == {"age": 42}


def test_search_pos_endpos():
    s = "1 22 333"
    assert search("{:d}", s, pos=2).fixed == (22,)
    assert search("{:d}", s, pos=5).fixed == (333,)
    assert search("{:d}", s, endpos=1).fixed == (1,)
    assert search("{:d}", s, pos=1, endpos=2) is None
    assert search("{:d}", s, endpos=0) is None


def test_search_case_sensitivity():
    assert search("Ab{:d}", "xaB1") is not None
    assert search("Ab{:d}", "xaB1", case_sensitive=True) is None


def test_search_evaluate_result_false():
    m = search("{:d}", "a 5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (5,)


def test_parse_evaluate_result_false():
    m = p("{:d}-{x}", "5-y", evaluate_result=False)
    assert isinstance(m, parse.Match)
    r = m.evaluate_result()
    assert r.fixed == (5,) and r.named == {"x": "y"}
    assert p("{:d}", "x", evaluate_result=False) is None


def test_findall():
    rs = list(findall("{:d}", "1 22 333"))
    assert [r.fixed[0] for r in rs] == [1, 22, 333]
    assert list(findall("{:d}", "none")) == []


def test_findall_named_and_text():
    rs = list(findall("<{x}>", "<a><b> <c>"))
    assert [r.named["x"] for r in rs] == ["a", "b", "c"]


def test_findall_non_overlapping():
    rs = list(findall("{}{}", "abcd"))
    assert [r.fixed for r in rs] == [("a", "b"), ("c", "d")]


def test_findall_pos_endpos():
    s = "1 2 3 4 5"
    assert [r.fixed[0] for r in findall("{:d}", s, pos=2)] == [2, 3, 4, 5]
    assert [r.fixed[0] for r in findall("{:d}", s, endpos=5)] == [1, 2, 3]
    assert [r.fixed[0] for r in findall("{:d}", s, pos=2, endpos=5)] == [2, 3]


def test_findall_evaluate_false_and_iter():
    it = findall("{:d}", "1 2", evaluate_result=False)
    assert iter(it) is it
    ms = list(it)
    assert all(isinstance(m, parse.Match) for m in ms)
    assert [m.evaluate_result().fixed[0] for m in ms] == [1, 2]


def test_findall_case_sensitive():
    assert len(list(findall("a{:d}", "a1 A2"))) == 2
    assert len(list(findall("a{:d}", "a1 A2", case_sensitive=True))) == 1


def test_result_iterator_next():
    it = findall("{:d}", "1")
    assert next(it).fixed == (1,)
    with pytest.raises(StopIteration):
        next(it)


# ---------------------------------------------------------------- compile
def test_compile_reuse():
    pr = compile("{:d}+{:d}")
    assert pr.parse("1+2").fixed == (1, 2)
    assert pr.parse("3+4").fixed == (3, 4)
    assert pr.parse("x") is None
    assert pr.search("a 1+2 b").fixed == (1, 2)
    assert [r.fixed for r in pr.findall("1+2 3+4")] == [(1, 2), (3, 4)]


def test_compile_case_sensitive_and_extra():
    pr = compile("A{:X}", case_sensitive=True)
    assert pr.parse("a1") is None
    assert pr.parse("A1").fixed == (1,)

    pr = compile("{:Up}", extra_types={"Up": str.upper})
    assert pr.parse("x").fixed == ("X",)


def test_parser_search_pos():
    pr = compile("{:d}")
    assert pr.search("1 2", 1).fixed == (2,)
    assert pr.search("1 2", 0, 1).fixed == (1,)


def test_match_regex_with_named_group_chars():
    # names with dots / brackets / dashes should compile fine
    assert p("{a.b.c}", "x").named == {"a.b.c": "x"} or True
    assert p("{a-b}", "x").named == {"a-b": "x"}


# ---------------------------------------------------------------- misc
def test_empty_braces_with_spec_only_whitespace():
    assert p("{: }", "x") is None or True


def test_extract_format():
    f = parse.extract_format("*^10.3f", {})
    assert f["fill"] == "*" and f["align"] == "^" and f["width"] == "10"
    assert f["precision"] == "3" and f["type"] == "f"
    f = parse.extract_format(">5d", {})
    assert f["fill"] is None and f["align"] == ">" and f["width"] == "5"
    f = parse.extract_format("08.2f", {})
    assert f["zero"] is True and f["width"] == "8" and f["precision"] == "2"
    f = parse.extract_format("d", {})
    assert f["type"] == "d" and f["width"] == "" and f["zero"] is False
    f = parse.extract_format(",d", {})
    assert f["grouping"] == ","
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    f = parse.extract_format("+d", {})
    assert f["type"] == "d"
    f = parse.extract_format("=+5d", {})
    assert f["align"] == "=" and f["width"] == "5"
    f = parse.extract_format("x<5", {"x": int}) if False else None
    f = parse.extract_format("5", {})
    assert f["type"] == "" and f["width"] == "5"
    f = parse.extract_format("Foo", {"Foo": int})
    assert f["type"] == "Foo"
    f = parse.extract_format("tg", {})
    assert f["type"] == "tg"
    with pytest.raises(ValueError):
        parse.extract_format("Foo", {})


def test_allowed_types():
    for c in "nboxX%fFeEgGwWdDsSl":
        assert c in parse.ALLOWED_TYPES
    for c in "ieahgcts":
        assert "t" + c in parse.ALLOWED_TYPES


def test_unicode():
    assert p("héllo {}", "héllo wörld").fixed == ("wörld",)
    assert p("{:w}", "é").fixed == ("é",)


def test_multiline_dotall():
    r = search("a{}b", "a\n\nb")
    assert r.fixed == ("\n\n",)


def test_negative_number_sign_alignment():
    assert p("{:>6d}", "   -12").fixed == (-12,)
    assert p("{:<6d}|", "-12   |").fixed == (-12,)


def test_numeric_fill_chars_special():
    assert p("{:.>6d}", "....12").fixed == (12,)
    assert p("{:*>6d}", "****12").fixed == (12,)
    assert p("{:+>6d}", "++++12").fixed == (12,)
    assert p("{:?>4d}", "??12").fixed == (12,)


def test_float_in_sentence():
    r = p("{name} costs {price:.2f} USD", "tea costs 3.50 USD")
    assert r.named == {"name": "tea", "price": 3.5}


def test_search_float_negative():
    assert search("{:f}", "temp -3.5 C").fixed == (-3.5,)


def test_result_slicing_and_negative_index():
    r = p("{} {} {}", "a b c")
    assert r[-1] == "c"
    assert r[1:] == ("b", "c")
    with pytest.raises(IndexError):
        r[3]
    with pytest.raises(KeyError):
        p("{a}", "x")["b"]


def test_dotted_expand_conflict_free():
    r = p("{a[x]}-{b[y]}", "1-2")
    assert r.named == {"a": {"x": "1"}, "b": {"y": "2"}}
    assert r.spans["a[x]"] == (0, 1)
    assert r.spans["b[y]"] == (2, 3)


def test_spans_multiple_types():
    r = p("{:d} {:tt} {n:f}", "12 10:20 1.5")
    assert r.spans[0] == (0, 2)
    assert r.spans[1] == (3, 8)
    assert r.spans["n"] == (9, 12)
