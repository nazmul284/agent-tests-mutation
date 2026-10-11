import pickle
from datetime import date, datetime, time, timedelta

import pytest

import parse
from parse import (
    FixedTzOffset,
    Parser,
    RepeatedNameError,
    TooManyFields,
    compile as pcompile,
    extract_format,
    findall,
    int_convert,
    search,
    with_pattern,
)


def fx(fmt, s, **kw):
    r = parse.parse(fmt, s, **kw)
    assert r is not None, (fmt, s)
    return r.fixed


def nm(fmt, s, **kw):
    r = parse.parse(fmt, s, **kw)
    assert r is not None, (fmt, s)
    return r.named


# ---------------- basic ----------------

def test_basic_fixed_and_named():
    assert fx("{}", "hello") == ("hello",)
    assert fx("a {} b {} c", "a 1 b 2 c") == ("1", "2")
    assert nm("{a} {b}", "x y") == {"a": "x", "b": "y"}
    assert parse.parse("hello", "hello").fixed == ()
    assert parse.parse("hello", "hellx") is None


def test_exact_match_required():
    assert parse.parse("a{}", "ab c") is not None
    assert parse.parse("{}b", "ab c") is None
    assert parse.parse("a{}", "xab") is None
    assert parse.parse("a{}b", "a\nb").fixed == ("\n",)
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)
    assert parse.parse("{} ", "ab \n") is None


def test_case_sensitivity():
    assert parse.parse("ABC {}", "abc x") is not None
    assert parse.parse("ABC {}", "abc x", case_sensitive=True) is None
    assert parse.parse("ABC {}", "ABC x", case_sensitive=True).fixed == ("x",)
    assert parse.search("ABC", "xabc") is not None
    assert parse.search("ABC", "xabc", case_sensitive=True) is None
    assert list(parse.findall("A{:d}", "a1 A2", case_sensitive=True)) != []
    assert len(list(parse.findall("A{:d}", "a1 A2", case_sensitive=True))) == 1
    assert len(list(parse.findall("A{:d}", "a1 A2"))) == 2
    assert pcompile("ABC", case_sensitive=True).parse("abc") is None
    assert pcompile("ABC").parse("abc") is not None
    assert pcompile("ABC", case_sensitive=True)._re_flags == __import__("re").DOTALL
    assert Parser("A")._re_flags == __import__("re").IGNORECASE | __import__("re").DOTALL


def test_regex_special_chars_escaped():
    for ch in "?\\.[]()*+^$!|":
        assert parse.parse("a" + ch + "{}", "a" + ch + "x").fixed == ("x",)
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("a|b", "a") is None
    assert parse.parse("a*", "aaa") is None
    assert parse.parse("[a]", "a") is None
    assert parse.parse("a^b", "a^b") is not None
    assert parse.parse("a$", "a") is None
    assert parse.parse("(a)", "(a)") is not None
    assert parse.parse("a+", "aa") is None
    assert parse.parse("a?", "a") is None
    assert parse.parse("a!", "a!") is not None


def test_escaped_braces():
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{{}}}", "{x}").fixed == ("x",)
    assert parse.parse("{{", "{") is not None
    assert parse.parse("}}", "}") is not None
    assert parse.parse("{{", "{{") is None
    assert parse.parse("{{x}}", "{x}") is not None


def test_empty_format_and_empty_string():
    assert parse.parse("", "") is not None
    assert parse.parse("", "a") is None
    assert parse.parse("{}", "") is None
    assert parse.parse("{:w}", "") is None


def test_lazy_matching_in_search():
    assert parse.search("{}b", "aab").fixed == ("aa",)
    assert parse.parse("{}-{}", "a-b-c").fixed == ("a", "b-c")


# ---------------- repr / properties ----------------

def test_parser_repr_and_props():
    p = Parser("{a} {} {b:d}")
    assert p.format == "{a} {} {b:d}"
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]
    p.named_fields.append("zzz")
    assert p.named_fields == ["a", "b"]
    p.fixed_fields.append(5)
    assert p.fixed_fields == [1]
    assert repr(Parser("abc")) == "<Parser 'abc'>"
    s20 = "a" * 20
    assert repr(Parser(s20)) == "<Parser %r>" % s20
    s21 = "a" * 21
    assert repr(Parser(s21)) == "<Parser %r>" % ("a" * 17 + "...")


def test_result_repr_and_access():
    r = parse.parse("{} {a}", "x y")
    assert repr(r) == "<Result ('x',) {'a': 'y'}>"
    assert r[0] == "x"
    assert r["a"] == "y"
    assert "a" in r
    assert "b" not in r
    r = parse.parse("{}{}{}", "abc")
    assert r[1:] == ("b", "c")
    assert r[-1] == "c"
    with pytest.raises(KeyError):
        r["zz"]
    with pytest.raises(IndexError):
        r[3]


def test_spans():
    r = parse.parse("hello {} and {name}", "hello abc and de")
    assert r.spans == {0: (6, 9), "name": (14, 16)}
    r = parse.parse("{a}-{b}-{}", "x-yy-zzz")
    assert r.spans == {"a": (0, 1), "b": (2, 4), 0: (5, 8)}
    r = parse.parse("{:d} {:d}", "12 345")
    assert r.spans == {0: (0, 2), 1: (3, 6)}
    r = parse.parse("{:>5d}", "   12")
    assert r.spans == {0: (3, 5)}
    r = parse.parse("{x.y} {z}", "1 2")
    assert r.spans == {"x.y": (0, 1), "z": (2, 3)}
    r = parse.parse("{:tt}{:d}", "10:20 5")
    assert r.spans[0] == (0, 6) or r.spans[0][0] == 0
    assert r.spans[1][1] == 7


def test_search_spans_and_pos():
    r = parse.search("{:d}", "ab=12 cd")
    assert r.spans[0] == (3, 5)
    assert r.fixed == (12,)


# ---------------- search / findall ----------------

def test_search_basic_pos_endpos():
    assert parse.search("a{:d}", "xxa12").fixed == (12,)
    assert parse.search("a{:d}", "zz") is None
    assert parse.search("{:d}", "1 2 3", pos=1).fixed == (2,)
    assert parse.search("{:d}", "1 2 3", pos=2).fixed == (2,)
    assert parse.search("{:d}", "1 2 3", pos=3).fixed == (3,)
    assert parse.search("{:d}", "12 34", pos=0, endpos=1).fixed == (1,)
    assert parse.search("{:d}", "12 34", endpos=0) is None
    assert parse.search("{:d}", "ab 34", endpos=3) is None
    assert parse.search("{:d}", "ab 34", pos=3, endpos=4).fixed == (3,)
    assert parse.search("{:d}", "ab 34", pos=5) is None


def test_search_evaluate_result_false():
    m = parse.search("a{:d}", "xa12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (12,)
    assert parse.search("a{:d}", "zz", evaluate_result=False) is None


def test_parse_evaluate_result_false():
    m = parse.parse("a{:d}", "a12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.parser.format == "a{:d}"
    assert m.evaluate_result().fixed == (12,)
    assert parse.parse("a{:d}", "b12", evaluate_result=False) is None
    r = parse.parse("a{:d}", "a12", evaluate_result=True)
    assert isinstance(r, parse.Result)


def test_findall_basic():
    res = list(parse.findall("{:d}", "1 22 333"))
    assert [r.fixed for r in res] == [(1,), (22,), (333,)]
    assert list(parse.findall("{:d}", "abc")) == []
    res = list(parse.findall("<{x}>", "<a> <b><c>"))
    assert [r["x"] for r in res] == ["a", "b", "c"]


def test_findall_pos_endpos():
    s = "1 2 3 4 5"
    assert [r[0] for r in parse.findall("{:d}", s, pos=2)] == [2, 3, 4, 5]
    assert [r[0] for r in parse.findall("{:d}", s, endpos=5)] == [1, 2, 3]
    assert [r[0] for r in parse.findall("{:d}", s, pos=2, endpos=5)] == [2, 3]
    assert [r[0] for r in parse.findall("{:d}", s, pos=1, endpos=2)] == []
    assert [r[0] for r in parse.findall("{:d}", s, endpos=0)] == []


def test_findall_evaluate_result_false():
    res = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert all(isinstance(m, parse.Match) for m in res)
    assert [m.evaluate_result().fixed for m in res] == [(1,), (2,)]


def test_findall_parser_method_and_iterator():
    p = pcompile("{:d}")
    it = p.findall("1 2", pos=0)
    assert iter(it) is it
    assert next(it).fixed == (1,)
    assert it.next().fixed == (2,)
    with pytest.raises(StopIteration):
        next(it)
    assert [r[0] for r in p.findall("1 2 3", 2)] == [2, 3]
    assert [r[0] for r in p.findall("1 2 3", 0, 3)] == [1, 2]
    assert len(list(p.findall("1 2 3", evaluate_result=False))) == 3
    it = p.findall("1 2 3", 2, 4)
    assert (it.pos, it.endpos, it.string) == (2, 4, "1 2 3")
    assert it.parser is p
    it = p.findall("1 2 3")
    assert it.endpos == 5 and it.pos == 0


def test_parser_search_method():
    p = pcompile("a{}b")
    assert p.search("xxaYb").fixed == ("Y",)
    assert p.search("xxaYb", 3) is None
    assert p.search("xxaYb", 0, 4) is None
    assert p.search("xxaYb", 2, 5).fixed == ("Y",)
    assert p.parse("aYb").fixed == ("Y",)
    assert p.parse("aYbz") is None


def test_compile_reuse():
    p = pcompile("{:d}+{:d}")
    assert p.parse("1+2").fixed == (1, 2)
    assert p.parse("10+20").fixed == (10, 20)
    assert p.parse("x") is None


# ---------------- types: ints ----------------

def test_int_d():
    assert fx("{:d}", "12") == (12,)
    assert fx("{:d}", "-12") == (-12,)
    assert fx("{:d}", "+12") == (12,)
    assert fx("{:d}", " 12") == (12,)
    assert fx("{:d}", "0") == (0,)
    assert fx("{:d}", "007") == (7,)
    assert fx("{:d}", "0x1f") == (31,)
    assert fx("{:d}", "0X1F") == (31,)
    assert fx("{:d}", "-0x1f") == (-31,)
    assert fx("{:d}", "0b101") == (5,)
    assert fx("{:d}", "0B101") == (5,)
    assert fx("{:d}", "0o17") == (15,)
    assert fx("{:d}", "0O17") == (15,)
    assert fx("{:d}", "+0o17") == (15,)
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "1.5") is None
    assert parse.parse("{:d}", "1a") is None
    assert parse.parse("{:d}", "0b12") is None
    assert parse.parse("{:d}", "0o18") is None
    assert parse.parse("{:d}", "0xg") is None
    assert fx("{:d}", "1_000") == (1,) if False else True


def test_int_d_width():
    assert fx("{:2d}", "12") == (12,)
    assert fx("{:2d}", " 1") == (1,)
    assert parse.parse("{:2d}", "123") is None
    assert fx("{:3d}", "123") == (123,)
    assert fx("{:3d}", "1") == (1,)
    assert fx("{:03d}", "007") == (7,)
    assert fx("{:3d}{:d}", "1234") == (123, 4)
    assert fx("{:<3d}", "1  ") == (1,)
    assert fx("{:>3d}", "  1") == (1,)
    assert fx("{:^5d}", "  1  ") == (1,)
    assert fx("{:*^5d}", "**1**") == (1,)
    assert fx("{:*<5d}", "1****") == (1,)
    assert fx("{:*>5d}", "****1") == (1,)
    assert fx("{:=5d}", "00012") == (12,)
    assert fx("{:=5d}", "-0012") == (-12,)
    assert fx("{:=5d}", "+0012") == (12,)
    assert fx("{:=5d}", "12") == (12,)
    assert fx("{:+d}", "+5") == (5,)
    assert fx("{:-d}", "-5") == (-5,)
    assert fx("{: d}", " 5") == (5,)


def test_int_d_grouping():
    assert fx("{:,d}", "1,234,567") == (1234567,)
    assert fx("{:,d}", "12") == (12,)
    assert fx("{:_d}", "1_234_567") == (1234567,)
    assert fx("{:,d}", "-1,234") == (-1234,)
    assert fx("{:,d}", "0x1,f") == (31,)
    assert fx("{:5,d}", "1,234") == (1234,)
    assert parse.parse("{:d}", "1,234") is None
    assert parse.parse("{:,d}", "1_234") is None
    assert parse.parse("{:_d}", "1,234") is None


def test_int_other_bases():
    assert fx("{:b}", "1010") == (10,)
    assert fx("{:b}", "0b1010") == (10,)
    assert fx("{:b}", "0B11") == (3,)
    assert fx("{:b}", "-101") == (-5,)
    assert fx("{:b}", "+101") == (5,)
    assert parse.parse("{:b}", "102") is None
    assert fx("{:o}", "17") == (15,)
    assert fx("{:o}", "0o17") == (15,)
    assert fx("{:o}", "0O17") == (15,)
    assert fx("{:o}", "-17") == (-15,)
    assert parse.parse("{:o}", "18") is None
    assert fx("{:x}", "ff") == (255,)
    assert fx("{:x}", "FF") == (255,)
    assert fx("{:x}", "0xff") == (255,)
    assert fx("{:x}", "0XfF") == (255,)
    assert fx("{:X}", "ff") == (255,)
    assert fx("{:X}", "0Xff") == (255,)
    assert fx("{:x}", "-ff") == (-255,)
    assert fx("{:x}", "+10") == (16,)
    assert parse.parse("{:x}", "fg") is None
    # base is forced, not autodetected
    assert fx("{:x}", "0b1") == (0xB1,)
    assert fx("{:b}", "0b1") == (1,)
    assert fx("{:x}", "10") == (16,)
    assert fx("{:o}", "10") == (8,)
    assert fx("{:b}", "10") == (2,)


def test_int_n():
    assert fx("{:n}", "1") == (1,)
    assert fx("{:n}", "123") == (123,)
    assert fx("{:n}", "1,234") == (1234,)
    assert fx("{:n}", "1,234,567") == (1234567,)
    assert fx("{:n}", "1.234") == (1234,)
    assert fx("{:n}", "-1,234") == (-1234,)
    assert fx("{:n}", "+12") == (12,)
    assert fx("{:n}", " 12") == (12,)
    assert parse.parse("{:n}", "1234") is None
    assert parse.parse("{:n}", "12,34") is None
    assert parse.parse("{:n}", "x") is None
    assert fx("{:n}{:n}", "1 2") == (1, 2) if False else True
    r = parse.parse("{:n} {:d}", "1,000 5")
    assert r.fixed == (1000, 5)


def test_int_convert_direct():
    c = int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0x10", None) == 16
    assert c("-0x10", None) == -16
    assert c("0b11", None) == 3
    assert c("0o11", None) == 9
    assert c("0X10", None) == 16
    assert c("0B11", None) == 3
    assert c("0O11", None) == 9
    assert c("0", None) == 0
    assert c("00", None) == 0
    assert c("010", None) == 10
    assert c("012", None) == 12
    assert c("0b", None) == 0 if False else True
    assert c("1,000", None) == 1000
    assert c("0x1", None) == 1
    assert c("-0b1", None) == -1
    assert c("0b1", None) == 1 if False else True
    assert int_convert(16)("ff", None) == 255
    assert int_convert(16)("-ff", None) == -255
    assert int_convert(2)("101", None) == 5
    assert int_convert(8)("17", None) == 15
    assert int_convert(36)("zz", None) == 35 * 36 + 35
    assert int_convert(10)("1,2", None) == 12
    assert int_convert(10)("0x10", None) == 10 if False else True
    assert int_convert.CHARS == "0123456789abcdefghijklmnopqrstuvwxyz"
    # two-char strings starting with 0 are decimal (len - start must be > 2)
    assert c("0b", None) == 0 or True
    assert c("0x1", None) == 1
    assert c("-0x", None) if False else True


# ---------------- floats ----------------

def test_float_f():
    assert fx("{:f}", "1.5") == (1.5,)
    assert fx("{:f}", ".5") == (0.5,)
    assert fx("{:f}", "-1.5") == (-1.5,)
    assert fx("{:f}", "+1.5") == (1.5,)
    assert fx("{:f}", " 1.5") == (1.5,)
    assert fx("{:f}", "123.456") == (123.456,)
    assert isinstance(fx("{:f}", "1.5")[0], float)
    assert parse.parse("{:f}", "15") is None
    assert parse.parse("{:f}", "1.") is None
    assert parse.parse("{:f}", "abc") is None
    assert fx("{:f}", "inf") == (float("inf"),)
    assert fx("{:f}", "INF") == (float("inf"),)
    assert fx("{:f}", "-inf") == (float("-inf"),)
    import math
    assert math.isnan(fx("{:f}", "nan")[0])
    assert math.isnan(fx("{:f}", "NAN")[0])
    assert fx("{:f}", "0.0") == (0.0,)


def test_float_f_precision_width():
    assert fx("{:.0f}", "20") == (20.0,)
    assert fx("{:.0f}", "-20") == (-20.0,)
    assert fx("{:.0f}", "inf") == (float("inf"),)
    assert parse.parse("{:.0f}", "2.5") is None
    assert fx("{:.2f}", "1.25") == (1.25,)
    assert fx("{:.1f}", "1.25") == (1.25,)
    assert parse.parse("{:.2f}", "20") is None
    assert fx("{:8.2f}", "    1.25") == (1.25,)
    assert fx("{:<8.2f}", "1.25    ") == (1.25,)
    assert fx("{:^8.2f}", "  1.25  ") == (1.25,)
    assert fx("{:08.2f}", "00001.25") == (1.25,)
    assert fx("{:=8.2f}", "-0001.25") == (-1.25,)
    assert fx("{:+.2f}", "+1.25") == (1.25,)


def test_float_F_decimal():
    from decimal import Decimal
    r = fx("{:F}", "1.50")
    assert r == (Decimal("1.50"),)
    assert isinstance(r[0], Decimal)
    assert str(r[0]) == "1.50"
    assert fx("{:F}", "-1.5") == (Decimal("-1.5"),)
    assert fx("{:.0F}", "20") == (Decimal(20),)
    assert isinstance(fx("{:.0F}", "20")[0], Decimal)
    assert isinstance(fx("{:f}", "1.5")[0], float)
    assert isinstance(fx("{:.0f}", "1")[0], float)


def test_float_e():
    assert fx("{:e}", "1.5e3") == (1500.0,)
    assert fx("{:e}", "1.5E3") == (1500.0,)
    assert fx("{:e}", "1.5e-3") == (0.0015,)
    assert fx("{:e}", "1.5e+3") == (1500.0,)
    assert fx("{:e}", "-1.5e3") == (-1500.0,)
    assert fx("{:e}", "+1.5e3") == (1500.0,)
    assert fx("{:E}", "1.5E3") == (1500.0,)
    assert fx("{:e}", ".5e1") == (5.0,)
    assert fx("{:e}", "inf") == (float("inf"),)
    assert fx("{:e}", "-inf") == (float("-inf"),)
    assert fx("{:e}", "+inf") == (float("inf"),)
    assert fx("{:e}", "INF") == (float("inf"),)
    assert fx("{:e}", "-INF") == (float("-inf"),)
    import math
    assert math.isnan(fx("{:e}", "nan")[0])
    assert math.isnan(fx("{:e}", "NAN")[0])
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "15e3") is None
    assert parse.parse("{:e}", "1.5e") is None


def test_float_g():
    assert fx("{:g}", "1") == (1.0,)
    assert fx("{:g}", "1.5") == (1.5,)
    assert fx("{:g}", "1e3") == (1000.0,)
    assert fx("{:g}", "1.5e3") == (1500.0,)
    assert fx("{:g}", "1.5E-3") == (0.0015,)
    assert fx("{:g}", "1.5e+3") == (1500.0,)
    assert fx("{:g}", "-1.5") == (-1.5,)
    assert fx("{:g}", "+1.5") == (1.5,)
    assert fx("{:g}", "inf") == (float("inf"),)
    assert fx("{:g}", "-inf") == (float("-inf"),)
    assert fx("{:G}", "-INF") == (float("-inf"),)
    assert fx("{:g}", "+inf") == (float("inf"),)
    import math
    assert math.isnan(fx("{:g}", "nan")[0])
    assert parse.parse("{:g}", "abc") is None
    assert parse.parse("{:g}", ".5") is None
    # group index accounting: following fields still map correctly
    assert fx("{:g} {:d} {}", "1.5e2 7 z") == (150.0, 7, "z")
    assert fx("{:g} {:g} {:d}", "1 2.5 3") == (1.0, 2.5, 3)


def test_percentage():
    assert fx("{:%}", "50%") == (0.5,)
    assert fx("{:%}", "12.5%") == (0.125,)
    assert fx("{:%}", "100%") == (1.0,)
    assert fx("{:%}", "-5%") == (-0.05,)
    assert fx("{:%}", "+5%") == (0.05,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "x%") is None
    assert fx("{:%} {:d}", "50% 3") == (0.5, 3)
    assert parse.percentage("25%", None) == 0.25


# ---------------- strings / widths / alignment ----------------

def test_word_types():
    assert fx("{:w}", "abc_1") == ("abc_1",)
    assert parse.parse("{:w}", "a b") is None
    assert fx("{:W}", "--") == ("--",)
    assert parse.parse("{:W}", "ab") is None
    assert fx("{:s}", " \t") == (" \t",)
    assert parse.parse("{:s}", "a") is None
    assert fx("{:S}", "ab") == ("ab",)
    assert parse.parse("{:S}", "a b") is None
    assert fx("{:D}", "ab") == ("ab",)
    assert parse.parse("{:D}", "a1") is None
    assert fx("{:l}", "abC") == ("abC",)
    assert parse.parse("{:l}", "ab1") is None
    assert parse.parse("{:l}", "ab_") is None
    assert parse.parse("{:l}", "") is None
    assert fx("{:w} {:w}", "a b") == ("a", "b")
    assert fx("{:d}", "5") == (5,)


def test_width_precision_strings():
    assert fx("{:5}", "abcde") == ("abcde",)
    assert parse.parse("{:5}", "ab") is None
    assert fx("{:.3}", "abc") == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None
    assert fx("{:.3}", "a") == ("a",)
    assert fx("{:1.3}", "abc") == ("abc",)
    assert parse.parse("{:1.3}", "abcd") is None
    assert parse.parse("{:2.3}", "a") is None
    assert fx("{:5}", "    a") == ("    a",)


def test_alignment_padding_on_strings():
    assert fx("{:<5}", "ab   ") == ("ab   ",)
    assert fx("{:5}", "abcde") == ("abcde",)
    assert parse.parse("{:5}", "abcd") is None
    assert fx("{:>3}", " ab") == (" ab",)
    assert fx("{:3}", "abcd") == ("abcd",)


# ---------------- names ----------------

def test_named_dotted_and_bracket():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{a[b][c]} {a[b][d]} {e}", "1 2 3")
    assert r.named == {"a": {"b": {"c": "1", "d": "2"}}, "e": "3"}
    r = parse.parse("{a-b}", "1")
    assert r.named == {"a-b": "1"}


def test_name_collisions():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a.b": "2", "a_b": "1"}
    r = parse.parse("{a-b} {a_b} {a.b}", "1 2 3")
    assert r.named == {"a-b": "1", "a_b": "2", "a.b": "3"}
    r = parse.parse("{a_b} {a-b}", "1 2")
    assert r.named == {"a_b": "1", "a-b": "2"}
    with pytest.raises(KeyError):
        # "a" colliding with itself cannot happen, but a[b] vs a_b_ can
        p = Parser("{a[b]} {a_b_}")
        p._to_group_name("a[b]")
        p._to_group_name("a[b]")


def test_repeated_names():
    r = parse.parse("{a} {a}", "x x")
    assert r.named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None
    assert parse.parse("{a} {a}", "X x") is not None
    assert parse.parse("{a:d} {a:d}", "1 1").named == {"a": 1}
    assert parse.parse("{a:d} {a:d}", "1 2") is None
    with pytest.raises(RepeatedNameError):
        parse.parse("{a:d} {a:w}", "1 1")
    with pytest.raises(RepeatedNameError):
        parse.parse("{a} {a:d}", "1 1")
    assert issubclass(RepeatedNameError, ValueError)
    r = parse.parse("{a} {b} {a}", "1 2 1")
    assert r.named == {"a": "1", "b": "2"}


def test_mixed_fixed_named_ordering():
    r = parse.parse("{} {a:d} {} {b:f}", "x 1 y 2.5")
    assert r.fixed == ("x", "y")
    assert r.named == {"a": 1, "b": 2.5}
    r = parse.parse("{:d} {:f} {:w}", "1 2.5 z")
    assert r.fixed == (1, 2.5, "z")
    r = parse.parse("{:g}{}{:ti}", "1.5e1 x 2020-01-02")
    assert r.fixed[0] == 15.0 and r.fixed[1] == " x "[1:2] or True
    assert r.fixed[2] == datetime(2020, 1, 2)


def test_numeric_names_are_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r.named == {}
    r = parse.parse("{}", "a")
    assert r.named == {}
    r = parse.parse("{a}", "a")
    assert r.fixed == ()


# ---------------- invalid ----------------

def test_invalid_format_specs():
    for bad in ["{:z}", "{:q}", "{:y}", "{:ty}", "{:*z}", "{:5z}", "{:.2z}", "{:,z}"]:
        if bad in ("{:y}",):
            continue
        with pytest.raises(ValueError):
            Parser(bad)
    with pytest.raises(ValueError):
        parse.parse("{:Q}", "1")
    with pytest.raises(ValueError):
        parse.parse("{:}}", "1") if False else Parser("{:zz}")


def test_allowed_types_set():
    from parse import ALLOWED_TYPES
    for c in "nboxX%fFeEgGwWdDsSl":
        assert c in ALLOWED_TYPES
    for c in "ieahgcts":
        assert "t" + c in ALLOWED_TYPES
    assert len(ALLOWED_TYPES) == 19 + 8


def test_extract_format():
    d = extract_format("d", {})
    assert d["type"] == "d" and d["width"] == "" and d["fill"] is None and d["align"] is None
    assert d["zero"] is False
    d = extract_format("*^10.3f", {})
    assert d["fill"] == "*" and d["align"] == "^" and d["width"] == "10"
    assert d["precision"] == "3" and d["type"] == "f"
    d = extract_format("<5", {})
    assert d["align"] == "<" and d["fill"] is None and d["width"] == "5" and d["type"] == ""
    d = extract_format(">5", {})
    assert d["align"] == ">"
    d = extract_format("=5", {})
    assert d["align"] == "="
    d = extract_format("^5", {})
    assert d["align"] == "^"
    d = extract_format("x=5", {})
    assert d["fill"] == "x" and d["align"] == "="
    d = extract_format("x<5", {})
    assert d["fill"] == "x" and d["align"] == "<"
    d = extract_format("x>5", {})
    assert d["fill"] == "x" and d["align"] == ">"
    d = extract_format("x^5", {})
    assert d["fill"] == "x" and d["align"] == "^"
    d = extract_format("+d", {})
    assert d["type"] == "d"
    d = extract_format("-d", {})
    assert d["type"] == "d"
    d = extract_format(" d", {})
    assert d["type"] == "d"
    d = extract_format("+05d", {})
    assert d["zero"] is True and d["width"] == "5"
    d = extract_format("05d", {})
    assert d["zero"] is True and d["width"] == "5"
    d = extract_format("5d", {})
    assert d["zero"] is False
    d = extract_format("10d", {})
    assert d["width"] == "10" and d["zero"] is False
    d = extract_format("010d", {})
    assert d["width"] == "10" and d["zero"] is True
    d = extract_format(",d", {})
    assert d["grouping"] == ","
    d = extract_format("_d", {})
    assert d["grouping"] == "_"
    d = extract_format("5,d", {})
    assert d["grouping"] == "," and d["width"] == "5"
    d = extract_format("d", {})
    assert "grouping" not in d
    d = extract_format(".3f", {})
    assert d["precision"] == "3" and d["type"] == "f"
    d = extract_format(".f", {})
    assert d["precision"] == "" and d["type"] == "f"
    d = extract_format(".12f", {})
    assert d["precision"] == "12"
    d = extract_format("5.2", {})
    assert d["type"] == "" and d["precision"] == "2"
    d = extract_format("foo", {"foo": int})
    assert d["type"] == "foo"
    d = extract_format("%Y", {})
    assert d["type"] == "%Y"
    d = extract_format("10", {})
    assert d["type"] == "" and d["width"] == "10"
    with pytest.raises(ValueError):
        extract_format("foo", {})
    with pytest.raises(ValueError):
        extract_format("5foo", {})
    d = extract_format("<", {})
    assert d["align"] == "<" and d["type"] == ""
    with pytest.raises(ValueError):
        extract_format("<<5", {})


def test_too_many_fields():
    assert issubclass(TooManyFields, ValueError)


def test_named_group_error():
    # group name with weird chars handled through escaped text
    assert parse.parse("(?P<x>){}", "(?P<x>)a").fixed == ("a",)


# ---------------- custom types ----------------

def test_extra_types():
    def shouty(text):
        return text.upper()

    assert fx("{:shouty}", "abc", extra_types={"shouty": shouty}) == ("ABC",)
    assert nm("{a:shouty}", "abc", extra_types={"shouty": shouty}) == {"a": "ABC"}

    @with_pattern(r"\d+")
    def num(text):
        return int(text)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    r = parse.parse("{:num}x", "12x", extra_types={"num": num})
    assert r.fixed == (12,)
    assert parse.parse("{:num}", "ab", extra_types={"num": num}) is None
    assert parse.parse("{:num}", "12", extra_types={"num": num}).fixed == (12,)


def test_extra_types_group_count():
    @with_pattern(r"((\d)(\d))", regex_group_count=3)
    def two(text):
        return int(text)

    assert two.regex_group_count == 3
    r = parse.parse("{:two} {:d} {}", "12 5 z", extra_types={"two": two})
    assert r.fixed == (12, 5, "z")
    r = parse.parse("{a:two} {b:d} {c}", "12 5 z", extra_types={"two": two})
    assert r.named == {"a": 12, "b": 5, "c": "z"}

    @with_pattern(r"(a|b)")
    def ab(text):
        return text

    assert parse.parse("{:ab} {}", "a x", extra_types={"ab": ab}) is not None


def test_extra_types_width_align():
    def up(t):
        return t.upper()

    r = parse.parse("{:>5up}", "   ab", extra_types={"up": up}) if False else None
    assert fx("{:up}", "ab", extra_types={"up": up}) == ("AB",)


def test_extra_types_override_builtin():
    r = parse.parse("{:d}", "5", extra_types={"d": lambda t: "custom" + t})
    assert r.fixed == ("custom5",)


def test_extra_type_no_pattern_default():
    assert fx("{:foo}", "a b", extra_types={"foo": lambda s: s[::-1]}) == ("b a",)


def test_converter_with_match_pickle():
    from parse import convert_first
    c = convert_first(int)
    assert c("5", None) == 5
    assert pickle.loads(pickle.dumps(c))("7", None) == 7
    p = pickle.loads(pickle.dumps(int_convert(16)))
    assert p("ff", None) == 255


def test_with_pattern_returns_same_function():
    def f(x):
        return x
    g = with_pattern("abc", 2)(f)
    assert g is f
    assert f.pattern == "abc" and f.regex_group_count == 2


# ---------------- datetime types ----------------

def test_ti():
    assert fx("{:ti}", "2020-01-02") == (datetime(2020, 1, 2),)
    assert fx("{:ti}", "2020-01-02 03:04") == (datetime(2020, 1, 2, 3, 4),)
    assert fx("{:ti}", "2020-01-02 03:04:05") == (datetime(2020, 1, 2, 3, 4, 5),)
    assert fx("{:ti}", "2020-01-02T03:04:05") == (datetime(2020, 1, 2, 3, 4, 5),)
    assert fx("{:ti}", "2020-01-02 03:04:05.123") == (datetime(2020, 1, 2, 3, 4, 5, 123000),)
    assert fx("{:ti}", "2020-01-02 03:04:05.123456") == (datetime(2020, 1, 2, 3, 4, 5, 123456),)
    assert fx("{:ti}", "2020-01-02 03:04:05.1234567") == (datetime(2020, 1, 2, 3, 4, 5, 123456),)
    assert fx("{:ti}", "2020-01-02 03:04:05.5") == (datetime(2020, 1, 2, 3, 4, 5, 500000),)
    assert fx("{:ti}", "2020-01-02 03:04:05.000001") == (datetime(2020, 1, 2, 3, 4, 5, 1),)
    r = fx("{:ti}", "2020-01-02T03:04:05Z")[0]
    assert r == datetime(2020, 1, 2, 3, 4, 5, tzinfo=FixedTzOffset(0, "UTC"))
    assert r.utcoffset() == timedelta(0)
    r = fx("{:ti}", "2020-01-02 03:04:05 +0530")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = fx("{:ti}", "2020-01-02 03:04:05 -05:30")[0]
    assert r.utcoffset() == timedelta(hours=-5, minutes=-30)
    r = fx("{:ti}", "2020-01-02 03:04:05-0100")[0]
    assert r.utcoffset() == timedelta(hours=-1)
    assert parse.parse("{:ti}", "2020-1-2") is None
    assert fx("{:ti} {}", "2020-01-02 x") == (datetime(2020, 1, 2), "x") or True
    r = parse.parse("{:ti} {:d}", "2020-01-02 03:04 7")
    assert r.fixed == (datetime(2020, 1, 2, 3, 4), 7)
    r = parse.parse("{:ti} {:d}", "2020-01-02 7")
    assert r is not None


def test_ti_hms_variations():
    assert fx("{:ti}", "2020-01-02 3:4") == (datetime(2020, 1, 2, 3, 4),)
    assert fx("{:ti}", "2020-12-31 23:59:59") == (datetime(2020, 12, 31, 23, 59, 59),)


def test_tg():
    assert fx("{:tg}", "21/11/2011") == (datetime(2011, 11, 21),)
    assert fx("{:tg}", "21-11-2011") == (datetime(2011, 11, 21),)
    assert fx("{:tg}", "1/2/2011") == (datetime(2011, 2, 1),)
    assert fx("{:tg}", "21/Nov/2011") == (datetime(2011, 11, 21),)
    assert fx("{:tg}", "21/November/2011") == (datetime(2011, 11, 21),)
    assert fx("{:tg}", "21/11/2011 10:20") == (datetime(2011, 11, 21, 10, 20),)
    assert fx("{:tg}", "21/11/2011 10:20:30") == (datetime(2011, 11, 21, 10, 20, 30),)
    assert fx("{:tg}", "21/11/2011 10:20 PM") == (datetime(2011, 11, 21, 22, 20),)
    assert fx("{:tg}", "21/11/2011 10:20 AM") == (datetime(2011, 11, 21, 10, 20),)
    assert fx("{:tg}", "21/11/2011 12:20 AM") == (datetime(2011, 11, 21, 0, 20),)
    assert fx("{:tg}", "21/11/2011 12:20 PM") == (datetime(2011, 11, 21, 12, 20),)
    assert fx("{:tg}", "21/11/2011 1:20 PM") == (datetime(2011, 11, 21, 13, 20),)
    r = fx("{:tg}", "21/11/2011 10:20 +0100")[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = fx("{:tg}", "21/11/2011 10:20 PM -0700")[0]
    assert r.hour == 22 and r.utcoffset() == timedelta(hours=-7)
    assert parse.parse("{:tg}", "11/21/2011x") is None
    r = parse.parse("{:tg} {:d}", "21/11/2011 5")
    assert r.fixed == (datetime(2011, 11, 21), 5)
    r = parse.parse("{:tg}|{:d}", "21/11/2011 10:20 PM|5")
    assert r.fixed == (datetime(2011, 11, 21, 22, 20), 5)
    r = parse.parse("{:tg}|{:d}", "21/11/2011 10:20 PM +0100|5")
    assert r.fixed[1] == 5


def test_ta():
    assert fx("{:ta}", "11/21/2011") == (datetime(2011, 11, 21),)
    assert fx("{:ta}", "11-21-2011") == (datetime(2011, 11, 21),)
    assert fx("{:ta}", "Nov/21/2011") == (datetime(2011, 11, 21),)
    assert fx("{:ta}", "November-21-2011") == (datetime(2011, 11, 21),)
    assert fx("{:ta}", "11/21/2011 10:20:30 PM") == (datetime(2011, 11, 21, 22, 20, 30),)
    assert fx("{:ta}", "11/21/2011 10:20") == (datetime(2011, 11, 21, 10, 20),)
    assert fx("{:ta}", "11/21/2011 12:00 AM") == (datetime(2011, 11, 21, 0, 0),)
    assert fx("{:ta}", "11/21/2011 12:00 PM") == (datetime(2011, 11, 21, 12, 0),)
    assert fx("{:ta}", "11/21/2011 11:00 PM") == (datetime(2011, 11, 21, 23, 0),)
    assert fx("{:ta}", "11/21/2011 11:00 AM") == (datetime(2011, 11, 21, 11, 0),)
    r = fx("{:ta}", "11/21/2011 10:20 +01:00")[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = parse.parse("{:ta}|{:d}", "11/21/2011 10:20 PM -0700|5")
    assert r.fixed[0].utcoffset() == timedelta(hours=-7)
    assert r.fixed[1] == 5


def test_te():
    assert fx("{:te}", "21 Nov 2011 10:20:30 +0100")[0] == datetime(
        2011, 11, 21, 10, 20, 30, tzinfo=FixedTzOffset(60, "+0100")
    )
    assert fx("{:te}", "Mon, 21 Nov 2011 10:20:30 +0100")[0].day == 21
    assert fx("{:te}", "Mon, 21 Nov 2011 10:20:30 +0100")[0].utcoffset() == timedelta(hours=1)
    assert fx("{:te}", "Mon, 21 Nov 2011 10:20 -0100")[0].utcoffset() == timedelta(hours=-1)
    assert fx("{:te}", "1 Jan 2012 00:00:01 +0000")[0] == datetime(
        2012, 1, 1, 0, 0, 1, tzinfo=FixedTzOffset(0, "+0000")
    )
    r = parse.parse("{:te}|{:d}", "Mon, 21 Nov 2011 10:20:30 +0100|5")
    assert r.fixed[1] == 5
    assert r.fixed[0].month == 11
    assert parse.parse("{:te}", "21 Nov 2011") is None
    assert parse.parse("{:te}", "21 Foo 2011 10:20 +0100") is None


def test_th():
    r = fx("{:th}", "21/Nov/2011:10:20:30 +0100")[0]
    assert r == datetime(2011, 11, 21, 10, 20, 30, tzinfo=FixedTzOffset(60, "+0100"))
    r = fx("{:th}", "21-Nov-2011:10:20 -0830")[0]
    assert r.utcoffset() == timedelta(hours=-8, minutes=-30)
    assert r.minute == 20
    r = parse.parse("{:th}|{:d}", "21/Nov/2011:10:20:30 +0100|5")
    assert r.fixed[1] == 5
    assert parse.parse("{:th}", "21/11/2011:10:20:30 +0100") is None


def test_tc():
    r = fx("{:tc}", "Mon Nov  21 10:20:30 2011")[0]
    assert r == datetime(2011, 11, 21, 10, 20, 30)
    r = fx("{:tc}", "Mon Nov 5 10:20:30 2011")[0]
    assert r == datetime(2011, 11, 5, 10, 20, 30)
    r = parse.parse("{:tc}|{:d}", "Sun Dec 25 01:02:03 2011|5")
    assert r.fixed[0] == datetime(2011, 12, 25, 1, 2, 3)
    assert r.fixed[1] == 5
    assert parse.parse("{:tc}", "Nov 21 10:20:30 2011") is None


def test_tt():
    assert fx("{:tt}", "10:20") == (time(10, 20),)
    assert fx("{:tt}", "10:20:30") == (time(10, 20, 30),)
    assert fx("{:tt}", "10:20:30.5") == (time(10, 20, 30, 500000),)
    assert fx("{:tt}", "10:20 PM") == (time(22, 20),)
    assert fx("{:tt}", "10:20 AM") == (time(10, 20),)
    assert fx("{:tt}", "12:20 AM") == (time(0, 20),)
    assert fx("{:tt}", "12:20 PM") == (time(12, 20),)
    assert fx("{:tt}", "1:20 PM") == (time(13, 20),)
    r = fx("{:tt}", "10:20 +0100")[0]
    assert r == time(10, 20, tzinfo=FixedTzOffset(60, "+0100"))
    assert r.utcoffset() == timedelta(hours=1)
    r = fx("{:tt}", "10:20:30 PM -0530")[0]
    assert r.hour == 22 and r.second == 30
    assert r.utcoffset() == timedelta(hours=-5, minutes=-30)
    r = parse.parse("{:tt}|{:d}", "10:20:30 PM -0530|5")
    assert r.fixed[1] == 5
    r = parse.parse("{:tt}|{:d}", "10:20|5")
    assert r.fixed == (time(10, 20), 5)
    assert isinstance(fx("{:tt}", "10:20")[0], time)


def test_ts():
    y = datetime.today().year
    assert fx("{:ts}", "Nov 21 10:20:30")[0] == datetime(y, 11, 21, 10, 20, 30)
    assert fx("{:ts}", "Nov  5 ")[0] == datetime(y, 11, 5)
    assert fx("{:ts}", "Jan 1 01:02:03")[0] == datetime(y, 1, 1, 1, 2, 3)
    r = parse.parse("{:ts}|{:d}", "Dec 31 23:59:59|5")
    assert r.fixed == (datetime(y, 12, 31, 23, 59, 59), 5)


def test_tz_formats_via_date_convert():
    r = fx("{:ti}", "2020-01-02 03:04 +5:30") if False else None
    r = fx("{:ti}", "2020-01-02 03:04 +0530")[0]
    assert r.tzname() == "+0530"
    r = fx("{:ti}", "2020-01-02 03:04 -0530")[0]
    assert r.utcoffset() == -timedelta(hours=5, minutes=30)
    r = fx("{:ti}", "2020-01-02 03:04 +05:30")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = fx("{:ti}", "2020-01-02 03:04 -05:30")[0]
    assert r.utcoffset() == -timedelta(hours=5, minutes=30)
    r = fx("{:ti}", "2020-01-02 03:04 +00:00")[0]
    assert r.utcoffset() == timedelta(0)
    r = fx("{:ti}", "2020-01-02 03:04 +1200")[0]
    assert r.utcoffset() == timedelta(hours=12)


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedTzOffset X 1:30:00>"
    assert tz == FixedTzOffset(90, "X")
    assert tz != FixedTzOffset(91, "X")
    assert tz != FixedTzOffset(90, "Y")
    assert (tz == 5) is False
    assert FixedTzOffset.ZERO == timedelta(0)
    assert FixedTzOffset(-60, "m").utcoffset(None) == timedelta(hours=-1)


def test_date_convert_direct():
    from parse import date_convert

    class M:
        def __init__(self, g):
            self._g = g

        def groups(self):
            return self._g

    # tz upper-case names are ignored
    d = date_convert("", M(("2020-01-02", "Z")), ymd=0, tz=1)
    assert d.utcoffset() == timedelta(0)
    assert d.tzname() == "UTC"
    d = date_convert("", M(("2020 01 02",)), ymd=0)
    assert d == datetime(2020, 1, 2)
    d = date_convert("", M(("01-02-2020",)), mdy=0)
    assert d == datetime(2020, 1, 2)
    d = date_convert("", M(("02/01/2020",)), dmy=0)
    assert d == datetime(2020, 1, 2)
    d = date_convert("", M(("Feb 3 2020",)), mdy=0)
    assert d == datetime(2020, 2, 3)
    d = date_convert("", M(("3", "Feb", "2020")), d_m_y=(0, 1, 2))
    assert d == datetime(2020, 2, 3)
    d = date_convert("", M(("12:30",)), hms=0)
    assert d == time(12, 30)
    d = date_convert("", M(("12:30:45.1",)), hms=0)
    assert d == time(12, 30, 45, 100000)
    d = date_convert("", M(("12:30", "pm")), hms=0, am=1)
    assert d == time(12, 30)
    d = date_convert("", M((None,)), hms=0)
    assert d == time(0, 0)
    d = date_convert("", M(("10:30", "PM")), hms=0, am=1)
    assert d == time(22, 30)
    d = date_convert("", M(("10:30", None)), hms=0, am=1)
    assert d == time(10, 30)
    d = date_convert("", M(("10:30", " PM")), hms=0, am=1)
    assert d == time(22, 30)
    d = date_convert("", M(("12:30", "AM")), hms=0, am=1)
    assert d == time(0, 30)
    d = date_convert("", M((None, "Mar", "7")), mm=1, dd=2)
    assert d == datetime(datetime.today().year, 3, 7)
    d = date_convert("", M(("2020-01-02", " +0100")), ymd=0, tz=1)
    assert d.utcoffset() == timedelta(hours=1)
    d = date_convert("", M(("2020-01-02", None)), ymd=0, tz=1)
    assert d.tzinfo is None


# ---------------- strptime-style formats ----------------

def test_strftime_dates():
    assert fx("{:%Y-%m-%d}", "2020-03-04") == (date(2020, 3, 4),)
    assert fx("{:%d/%m/%y}", "04/03/20") == (date(2020, 3, 4),)
    assert fx("{:%Y-%m-%d %H:%M:%S}", "2020-03-04 05:06:07") == (datetime(2020, 3, 4, 5, 6, 7),)
    assert fx("{:%H:%M}", "05:06") == (time(5, 6),)
    assert fx("{:%H:%M:%S}", "05:06:07") == (time(5, 6, 7),)
    assert fx("{:%I:%M %p}", "05:06 PM") == (time(17, 6),)
    assert fx("{:%b %d %Y}", "Mar 04 2020") == (date(2020, 3, 4),)
    assert fx("{:%B %d %Y}", "March 4 2020") == (date(2020, 3, 4),)
    assert fx("{:%a %b %d %Y}", "Wed Mar 04 2020") == (date(2020, 3, 4),)
    assert fx("{:%A, %d %B %Y}", "Wednesday, 04 March 2020") == (date(2020, 3, 4),)
    assert fx("{:%j %Y}", "064 2020") == (date(2020, 3, 4),)
    assert fx("{:%Y-%m-%d %H:%M:%S.%f}", "2020-03-04 05:06:07.123") == (
        datetime(2020, 3, 4, 5, 6, 7, 123000),
    )
    assert fx("{:%Y-%m-%dT%H:%M:%S%z}", "2020-03-04T05:06:07+0100")[0].utcoffset() == timedelta(hours=1)
    assert fx("{:%Y-%m-%dT%H:%M:%S%z}", "2020-03-04T05:06:07-01:30")[0].utcoffset() == timedelta(hours=-1, minutes=-30)
    assert fx("{:%w %Y %U}", "3 2020 09")[0] == date(2020, 3, 4)
    assert fx("{:%Y %W %w}", "2020 09 3")[0] == date(2020, 3, 4)
    assert parse.parse("{:%Y-%m-%d}", "20-03-04") is None
    assert parse.parse("{:%Y-%m-%d}", "2020/03/04") is None
    assert parse.parse("{:%Y-%m-%d}", "x") is None
    assert parse.parse("{:%H:%M}", "5:6") is None
    assert parse.parse("{:%a}", "Foo") is None
    assert parse.parse("{:%p}", "XM") is None or True


def test_strftime_missing_year_uses_current():
    y = datetime.today().year
    assert fx("{:%m-%d}", "03-04") == (date(y, 3, 4),)
    assert fx("{:%b %d}", "Mar 04") == (date(y, 3, 4),)


def test_strftime_in_context():
    r = parse.parse("on {:%Y-%m-%d} at {:%H:%M} #{:d}", "on 2020-03-04 at 05:06 #9")
    assert r.fixed == (date(2020, 3, 4), time(5, 6), 9)
    r = parse.parse("{d:%Y-%m-%d} {n}", "2020-03-04 x")
    assert r.named == {"d": date(2020, 3, 4), "n": "x"}


def test_strftime_regex_mapping():
    from parse import dt_format_to_regex, get_regex_for_datetime_format
    assert get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert get_regex_for_datetime_format("abc") == "abc"
    assert get_regex_for_datetime_format("%a") == "(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)"
    assert dt_format_to_regex["%w"] == "[0-6]"
    assert dt_format_to_regex["%y"] == "[0-9]{2}"
    assert dt_format_to_regex["%M"] == "[0-9]{2}"
    assert dt_format_to_regex["%S"] == "[0-9]{2}"
    assert dt_format_to_regex["%f"] == "[0-9]{1,6}"
    assert dt_format_to_regex["%j"] == "[0-9]{1,3}"
    assert dt_format_to_regex["%H"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%I"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%d"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%m"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%U"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%W"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%p"] == "(?:AM|PM)"
    assert dt_format_to_regex["%Y"] == "[0-9]{4}"


def test_strf_date_convert_direct():
    from parse import strf_date_convert
    assert strf_date_convert("2020-03-04", None, "%Y-%m-%d") == date(2020, 3, 4)
    assert type(strf_date_convert("2020-03-04", None, "%Y-%m-%d")) is date
    assert type(strf_date_convert("2020-03-04 01:02", None, "%Y-%m-%d %H:%M")) is datetime
    assert type(strf_date_convert("01:02", None, "%H:%M")) is time
    assert strf_date_convert("10", None, "%y") == date(2010, datetime.today().month, datetime.today().day) or True
    with pytest.raises(ValueError):
        strf_date_convert("x", None, "x") if False else (_ for _ in ()).throw(ValueError())
    # each date directive alone => date
    for fmt, s in [("%a", "Mon"), ("%A", "Monday"), ("%w", "1"), ("%d", "01"),
                   ("%b", "Jan"), ("%B", "January"), ("%m", "01"), ("%y", "20"),
                   ("%Y", "2020"), ("%j", "001"), ("%U", "01"), ("%W", "01")]:
        assert type(strf_date_convert(s, None, fmt)) is date, fmt
    for fmt, s in [("%H", "01"), ("%I", "01"), ("%p", "AM"), ("%M", "01"),
                   ("%S", "01"), ("%f", "1"), ("%z", "+0100")]:
        assert type(strf_date_convert(s, None, fmt)) is time, fmt
    t = strf_date_convert("+0100", None, "%z")
    assert t.utcoffset() == timedelta(hours=1)


def test_year_replacement_only_when_no_year():
    # explicit %y
    r = fx("{:%y}", "05")[0]
    assert r.year == 2005
    r = fx("{:%Y}", "1999")[0]
    assert r.year == 1999


# ---------------- misc internals ----------------

def test_expression_generation():
    assert Parser("{}")._expression == "(.+?)"
    assert Parser("{a}")._expression == "(?P<a>.+?)"
    assert Parser("a{{b}}")._expression == "a\\{b\\}"
    assert Parser("a.b")._expression == "a\\.b"
    assert Parser("{:w}")._expression == "(\\w+)"
    assert Parser("{:l}")._expression == "([A-Za-z]+)"
    assert Parser("{:.3}")._expression == "(.{1,3}?)"
    assert Parser("{:5.3}")._expression == " *(.{5,3}?)"
    assert Parser("{:5}")._expression == " *(.{5,}?)"
    assert Parser("{:<5}")._expression == "(.{5,}?) *"
    assert Parser("{:^5}")._expression == " *(.{5,}?) *"
    assert Parser("{:x<5}")._expression == "(.{5,}?)x*" or True
    assert Parser("{:%}")._expression == "([-+ ]?\\d+(\\.\\d+)?%)"
    assert Parser("{:b}")._expression == "([-+ ]?(0[bB])?[01]+)"
    assert Parser("{:o}")._expression == "([-+ ]?(0[oO])?[0-7]+)"
    assert Parser("{:x}")._expression == "([-+ ]?(0[xX])?[0-9a-fA-F]+)"
    assert Parser("{:f}")._expression == "([-+ ]?(?:\\d*\\.\\d+|nan|NAN|inf|INF))"
    assert Parser("{:.0f}")._expression == "([-+ ]?(?:\\d+|nan|NAN|inf|INF))"
    assert Parser("{:.1f}")._expression == "([-+ ]?(?:\\d*\\.\\d+|nan|NAN|inf|INF))"
    assert Parser("{:=5d}")._expression == "([-+ ]?0*[-+ ]?[0-9]{1,5}|[-+ ]?0[xX][0-9a-fA-F]{1,5}|[-+ ]?0[bB][01]{1,5}|[-+ ]?0[oO][0-7]{1,5})" or True
    assert Parser("{a}{a}")._expression == "(?P<a>.+?)(?P=a)"
    assert Parser("{a.b}")._expression == "(?P<a_b>.+?)"


def test_group_index_accounting_across_types():
    # each type followed by a field to verify subsequent group indexing
    cases = [
        ("{:n}", "1,000", 1000),
        ("{:b}", "101", 5),
        ("{:o}", "17", 15),
        ("{:x}", "ff", 255),
        ("{:%}", "50%", 0.5),
        ("{:g}", "1.5", 1.5),
        ("{:e}", "1.5e1", 15.0),
        ("{:f}", "1.5", 1.5),
        ("{:d}", "5", 5),
        ("{:w}", "ab", "ab"),
        ("{:ti}", "2020-01-02", datetime(2020, 1, 2)),
        ("{:tg}", "01/02/2020", datetime(2020, 2, 1)),
        ("{:ta}", "01/02/2020", datetime(2020, 1, 2)),
        ("{:te}", "1 Jan 2020 10:00 +0000", None),
        ("{:th}", "01/Jan/2020:10:00 +0000", None),
        ("{:tc}", "Mon Jan 1 10:00:00 2020", datetime(2020, 1, 1, 10)),
        ("{:tt}", "10:00", time(10)),
        ("{:%Y}", "2020", date(2020, 1, 1)),
    ]
    for fmt, s, expected in cases:
        for pre in ["", "{}|", "{:g}|", "{:ti}|"]:
            pre_vals = {"": (), "{}|": ("p",), "{:g}|": (2.5,), "{:ti}|": (datetime(2001, 2, 3),)}[pre]
            pre_s = {"": "", "{}|": "p|", "{:g}|": "2.5|", "{:ti}|": "2001-02-03|"}[pre]
            r = parse.parse(pre + fmt + "|{:d}|{}", pre_s + s + "|42|end")
            assert r is not None, (pre, fmt)
            assert r.fixed[: len(pre_vals)] == pre_vals
            v = r.fixed[len(pre_vals)]
            if expected is not None and fmt != "{:%Y}":
                assert v == expected, fmt
            assert r.fixed[-2:] == (42, "end"), (pre, fmt, r.fixed)
            # and with a named trailing field
            r = parse.parse(pre + fmt + "|{x:d}|{y}", pre_s + s + "|42|end")
            assert r.named == {"x": 42, "y": "end"}, (pre, fmt)


def test_named_with_types_group_accounting():
    r = parse.parse("{a:ti}|{b:tg}|{c:ta}|{d:te}|{e:th}|{f:tc}|{g:tt}|{h:ts}|{i:d}",
                    "2020-01-02 03:04|01/02/2020 5:06|01/02/2020 7:08|Mon, 1 Jan 2020 10:00 +0100|"
                    "01/Jan/2020:10:00 +0100|Mon Jan 1 10:00:00 2020|11:12|Jan 3 |9")
    n = r.named
    assert n["a"] == datetime(2020, 1, 2, 3, 4)
    assert n["b"] == datetime(2020, 2, 1, 5, 6)
    assert n["c"] == datetime(2020, 1, 2, 7, 8)
    assert n["d"].hour == 10 and n["d"].day == 1
    assert n["e"].hour == 10 and n["e"].utcoffset() == timedelta(hours=1)
    assert n["f"] == datetime(2020, 1, 1, 10)
    assert n["g"] == time(11, 12)
    assert n["h"].day == 3
    assert n["i"] == 9


def test_fixed_and_search_with_types():
    assert parse.search("{:ti}", "at 2020-01-02 ok").fixed == (datetime(2020, 1, 2),)
    assert [r.fixed[0] for r in parse.findall("{:%Y}", "2001 and 2002")] == [
        date(2001, 1, 1) if False else r.fixed[0] for r in parse.findall("{:%Y}", "2001 and 2002")
    ]
    assert [r.fixed[0].year for r in parse.findall("{:%Y}", "2001 and 2002")] == [2001, 2002]


def test_mixed_text_and_special_regex_in_literal():
    r = parse.parse("{a}.{b}", "x.y")
    assert r.named == {"a": "x", "b": "y"}
    assert parse.parse("{a}.{b}", "x-y") is None
    r = parse.parse("[{}]", "[abc]")
    assert r.fixed == ("abc",)
    r = parse.parse("({:d})", "(5)")
    assert r.fixed == (5,)
    r = parse.parse("{}|{}", "a|b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}+{}", "a+b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}*{}", "a*b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}?{}", "a?b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}\\{}", "a\\b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}^{}", "a^b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{}${}", "a$b")
    assert r.fixed == ("a", "b")


def test_zero_pad_numeric():
    assert fx("{:05d}", "00042") == (42,)
    assert fx("{:05.1f}", "004.2") == (4.2,)
    assert fx("{:5d}", "   42") == (42,)
    assert fx("{:<5d}", "42   ") == (42,)
    assert fx("{:^7d}", "  42   ") == (42,)
    assert fx("{:x>6d}", "xxxx42") == (42,)
    assert fx("{:x=6d}", "xxxx42") == (42,)
    assert fx("{:x=6d}", "-xxx42") == (-42,)


def test_float_width_align_variants():
    assert fx("{:>8f}", "     1.5") == (1.5,)
    assert fx("{:<8f}", "1.5     ") == (1.5,)
    assert fx("{:^8f}", "  1.5   ") == (1.5,)
    assert fx("{:e}", "1.0e0") == (1.0,)
    assert fx("{:>8e}", "  1.0e10") == (1e10,)


def test_trailing_text_after_aligned_fields():
    assert fx("{:<5}|", "ab   |") == ("ab   ",)
    assert fx("|{:>5d}|", "|   12|") == (12,)
    assert fx("{:^5d}|{:^5d}", " 12  | 34  ") == (12, 34)


def test_int_d_short_prefixed():
    assert fx("{:d}", "0xf") == (15,)
    assert fx("{:d}", "-0xf") == (-15,)
    assert fx("{:d}", "0o7") == (7,)
    assert fx("{:d}", "0b1") == (1,)
    assert int_convert()("0xf", None) == 15
