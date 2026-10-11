import pickle
import sys
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


# ---------- basic fixed / named ----------

def test_no_fields_literal():
    assert parse.parse("hello", "hello").fixed == ()
    assert parse.parse("hello", "hello!") is None


def test_fixed_fields():
    r = parse.parse("{} and {}", "a and b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a" and r[1] == "b"
    assert r[0:2] == ("a", "b")
    assert r.named == {}


def test_named_fields():
    r = parse.parse("{a}-{b}", "x-y")
    assert r.named == {"a": "x", "b": "y"}
    assert r["a"] == "x"
    assert "a" in r and "c" not in r
    assert r.fixed == ()


def test_mixed_fixed_named():
    r = parse.parse("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}


def test_no_match_returns_none():
    assert parse.parse("{}-{}", "abc") is None


def test_exact_match_required():
    assert parse.parse("a{}", "ab\nc") is not None  # DOTALL
    assert parse.parse("{}b", "abc") is None
    assert parse.parse("a{}", "xab") is None


def test_trailing_newline_not_matched():
    assert parse.parse("{}", "abc\n").fixed == ("abc\n",)
    assert parse.parse("abc", "abc\n") is None


def test_case_insensitive_default():
    assert parse.parse("Hello {}", "hELLO x").fixed == ("x",)
    assert parse.parse("Hello {}", "hELLO x", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello x", case_sensitive=True).fixed == ("x",)


def test_empty_string():
    assert parse.parse("{}", "") is None
    assert parse.parse("", "").fixed == ()


def test_lazy_matching():
    r = parse.parse("{}.{}", "a.b.c")
    assert r.fixed == ("a", "b.c")


def test_escaped_braces():
    assert parse.parse("{{}}", "{}").fixed == ()
    assert parse.parse("{{{}}}", "{x}").fixed == ("x",)
    assert parse.parse("{{ {a} }}", "{ 1 }").named == {"a": "1"}


def test_regex_special_chars_escaped():
    for ch in "?\\.[]()*+^$!|":
        r = parse.parse("a" + ch + "{}", "a" + ch + "b")
        assert r is not None and r.fixed == ("b",), ch
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("a|b", "a") is None
    assert parse.parse("(x){}", "(x)y").fixed == ("y",)


def test_dotall_in_fields():
    assert parse.parse("{}|{}", "a\nb|c\nd").fixed == ("a\nb", "c\nd")


def test_spans():
    r = parse.parse("hi {} and {n}", "hi abc and de")
    assert r.spans[0] == (3, 6)
    assert r.spans["n"] == (11, 13)


def test_spans_with_typed_fixed():
    r = parse.parse("{:d} {:g} {}", "12 3.5 xyz")
    assert r.spans == {0: (0, 2), 1: (3, 6), 2: (7, 10)}


def test_repr():
    r = parse.parse("{} {a}", "1 2")
    assert "1" in repr(r) and "'a'" in repr(r)
    assert repr(Parser("abc")) == "<Parser 'abc'>"
    long = "x" * 30
    assert repr(Parser(long)) == "<Parser %r>" % ("x" * 17 + "...")
    assert repr(Parser("x" * 20)) == "<Parser %r>" % ("x" * 20)


# ---------- repeated / named structure ----------

def test_repeated_name_must_match():
    r = parse.parse("{a} {a}", "x x")
    assert r.named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.compile("{a:d} {a:w}")
    with pytest.raises(RepeatedNameError):
        parse.compile("{a} {a:d}")


def test_repeated_name_same_type():
    r = parse.parse("{a:d} {a:d}", "1 1")
    assert r.named == {"a": 1}
    assert parse.parse("{a:d} {a:d}", "1 2") is None


def test_dotted_names():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}
    assert r["a.b"] == "1"


def test_bracket_names():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{a[0][x]}", "q")
    assert r.named == {"a": {"0": {"x": "q"}}}


def test_name_collisions():
    r = parse.parse("{a.b} {a_b} {a__b}", "1 2 3")
    assert r.named == {"a.b": "1", "a_b": "2", "a__b": "3"}


def test_dash_names():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_underscore_name_collision():
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_named_and_fixed_fields_lists():
    p = Parser("{} {a} {:d} {b.c}")
    assert len(p.fixed_fields) == 2
    assert p.named_fields == ["a", "b_c"]
    # returned copies
    p.named_fields.append("zzz")
    assert p.named_fields == ["a", "b_c"]
    p.fixed_fields.append(99)
    assert 99 not in p.fixed_fields


def test_fixed_field_indices_with_groups():
    p = Parser("{:d} {:g} {:d}")
    assert len(p.fixed_fields) == 3
    r = p.parse("1 2.5 3")
    assert r.fixed == (1, 2.5, 3)


def test_digit_name_is_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_parser_format_property():
    assert Parser("{} x").format == "{} x"


# ---------- numeric types ----------

@pytest.mark.parametrize(
    "fmt,s,expected",
    [
        ("{:d}", "12", 12),
        ("{:d}", "-12", -12),
        ("{:d}", "+12", 12),
        ("{:d}", " 12", 12),
        ("{:d}", "0x1f", 31),
        ("{:d}", "0X1F", 31),
        ("{:d}", "0b101", 5),
        ("{:d}", "0o17", 15),
        ("{:d}", "-0x10", -16),
        ("{:d}", "007", 7),
        ("{:d}", "0", 0),
        ("{:,d}", "1,000", 1000),
        ("{:_d}", "1_000", 1000),
        ("{:n}", "1,234", 1234),
        ("{:n}", "1.234.567", 1234567),
        ("{:n}", "42", 42),
        ("{:n}", "-42", -42),
        ("{:b}", "101", 5),
        ("{:b}", "0b101", 5),
        ("{:b}", "-101", -5),
        ("{:o}", "17", 15),
        ("{:o}", "0o17", 15),
        ("{:x}", "ff", 255),
        ("{:x}", "0xff", 255),
        ("{:X}", "FF", 255),
        ("{:x}", "FF", 255),
        ("{:x}", "-0xff", -255),
    ],
)
def test_int_types(fmt, s, expected):
    r = parse.parse(fmt, s)
    assert r is not None, (fmt, s)
    assert r.fixed == (expected,)
    assert isinstance(r[0], int)


@pytest.mark.parametrize(
    "fmt,s",
    [
        ("{:d}", "abc"),
        ("{:d}", "1.5"),
        ("{:d}", ""),
        ("{:d}", "1 2"),
        ("{:b}", "102"),
        ("{:o}", "8"),
        ("{:x}", "xyz"),
        ("{:n}", "1234,5"),
        ("{:n}", "abc"),
    ],
)
def test_int_types_reject(fmt, s):
    assert parse.parse(fmt, s) is None


def test_int_width():
    assert parse.parse("{:2d}", "12").fixed == (12,)
    assert parse.parse("{:2d}", "123") is None
    assert parse.parse("{:3d}", " 12").fixed == (12,)
    assert parse.parse("{:3d}", "12 ") is None
    assert parse.parse("{:<3d}", "12 ").fixed == (12,)


def test_int_zero_padded():
    assert parse.parse("{:05d}", "00042").fixed == (42,)
    assert parse.parse("{:=5d}", "00042").fixed == (42,)
    assert parse.parse("{:x=5d}", "xxx42").fixed == (42,)
    assert parse.parse("{:x=5d}", "-xx42").fixed == (-42,)


def test_int_sign_after_fill_align():
    assert parse.parse("{:>5d}", "   42").fixed == (42,)
    assert parse.parse("{:<5d}", "42   ").fixed == (42,)
    assert parse.parse("{:^5d}", " 42  ").fixed == (42,)


def test_int_grouping_in_spec():
    assert parse.parse("{:,d}", "1,234").fixed == (1234,)
    assert parse.parse("{:_d}", "1_234").fixed == (1234,)


@pytest.mark.parametrize(
    "fmt,s,expected",
    [
        ("{:f}", "3.14", 3.14),
        ("{:f}", "-3.14", -3.14),
        ("{:f}", "+3.14", 3.14),
        ("{:f}", ".5", 0.5),
        ("{:f}", "0.5", 0.5),
        ("{:.2f}", "1.25", 1.25),
        ("{:.0f}", "20", 20.0),
        ("{:.0f}", "-20", -20.0),
        ("{:e}", "1.5e3", 1500.0),
        ("{:e}", "1.5E-3", 0.0015),
        ("{:e}", "-1.5e+3", -1500.0),
        ("{:g}", "1.5", 1.5),
        ("{:g}", "12", 12.0),
        ("{:g}", "1e5", 1e5),
        ("{:g}", "-1.5e5", -1.5e5),
        ("{:G}", "2.5E3", 2500.0),
        ("{:E}", "2.5E3", 2500.0),
    ],
)
def test_float_types(fmt, s, expected):
    r = parse.parse(fmt, s)
    assert r is not None, (fmt, s)
    assert r.fixed == (expected,)
    assert isinstance(r[0], float)


@pytest.mark.parametrize("fmt", ["{:f}", "{:e}", "{:g}"])
def test_float_inf_nan(fmt):
    assert parse.parse(fmt, "inf").fixed == (float("inf"),)
    assert parse.parse(fmt, "INF").fixed == (float("inf"),)
    r = parse.parse(fmt, "nan")
    assert r[0] != r[0]
    r = parse.parse(fmt, "NAN")
    assert r[0] != r[0]


@pytest.mark.parametrize("fmt", ["{:e}", "{:g}"])
def test_float_signed_inf(fmt):
    assert parse.parse(fmt, "-inf").fixed == (float("-inf"),)
    assert parse.parse(fmt, "+inf").fixed == (float("inf"),)


def test_f_rejects():
    assert parse.parse("{:f}", "12") is None
    assert parse.parse("{:f}", "abc") is None
    assert parse.parse("{:f}", "1.") is None
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:g}", "abc") is None
    assert parse.parse("{:.0f}", "2.5") is None


def test_f_decimal_F():
    r = parse.parse("{:F}", "3.14")
    assert r[0] == Decimal("3.14") and isinstance(r[0], Decimal)
    r = parse.parse("{:F}", "-3.14")
    assert r[0] == Decimal("-3.14")
    r = parse.parse("{:.0F}", "7")
    assert r[0] == Decimal("7") and isinstance(r[0], Decimal)


def test_float_in_context():
    r = parse.parse("x={:f};y={:f}", "x=-1.5;y=2.5")
    assert r.fixed == (-1.5, 2.5)
    r = parse.parse("{:f} {:f}", "1.5 -2.5")
    assert r.fixed == (1.5, -2.5)


def test_float_width_align():
    assert parse.parse("{:8.2f}", "    1.25").fixed == (1.25,)
    assert parse.parse("{:<8.2f}", "1.25    ").fixed == (1.25,)
    assert parse.parse("{:^8.2f}", "  1.25  ").fixed == (1.25,)
    assert parse.parse("{:08.2f}", "00001.25").fixed == (1.25,)
    assert parse.parse("{:*>8.2f}", "****1.25").fixed == (1.25,)


def test_percent():
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "100%").fixed == (1.0,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "-50%") is not None
    assert parse.parse("{:%}", "abc%") is None


def test_numeric_in_search_and_findall():
    assert search("{:d}", "abc 42 def").fixed == (42,)
    assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]


# ---------- strings / other types ----------

def test_word_types():
    assert parse.parse("{:w}", "abc_123").fixed == ("abc_123",)
    assert parse.parse("{:w}", "abc def") is None
    assert parse.parse("{:W}", "abc") is None
    assert parse.parse("{:W}", "!!!").fixed == ("!!!",)
    assert parse.parse("{:s}", " \t").fixed == (" \t",)
    assert parse.parse("{:s}", "a") is None
    assert parse.parse("{:S}", "ab").fixed == ("ab",)
    assert parse.parse("{:S}", "a b") is None
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None
    assert parse.parse("{:l}", "abc").fixed == ("abc",)
    assert parse.parse("{:l}", "ab1") is None
    assert parse.parse("{:l}", "ab c") is None


def test_string_width_precision():
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:2.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "abcd") is None


def test_string_alignment():
    assert parse.parse("{:>}", "  abc").fixed == ("abc",)
    assert parse.parse("{:<}", "abc  ").fixed == ("abc",)
    assert parse.parse("{:^}", "  abc  ").fixed == ("abc",)
    assert parse.parse("{:>5}", "  abc").fixed == ("  abc",)
    assert parse.parse("{:x^}", "xxabcxx").fixed == ("abc",)
    assert parse.parse("{:x<}", "abcxxx").fixed == ("abc",)
    assert parse.parse("{:x>}", "xxxabc").fixed == ("abc",)


def test_width_implies_right_align():
    # width with no explicit fill: the padding is part of the capture
    assert parse.parse("{:5}", "  abc").fixed == ("  abc",)
    assert parse.parse("{:5}", "abc") is None


def test_align_only():
    assert parse.parse("{:<}", "abc").fixed == ("abc",)


def test_named_with_spec():
    r = parse.parse("{n:d}-{s:w}", "5-ab")
    assert r.named == {"n": 5, "s": "ab"}


def test_invalid_specs():
    for fmt in ["{:q}", "{:z}", "{:.2q}", "{:td}", "{:tx}"]:
        with pytest.raises(ValueError):
            parse.parse(fmt, "x")


def test_all_allowed_types_compile():
    for t in "nboxX%fFeEgGwWdDsSl":
        Parser("{:%s}" % t)
    for t in ["ti", "ta", "te", "tg", "th", "tc", "tt", "ts"]:
        Parser("{:%s}" % t)


# ---------- extra types / with_pattern ----------

def test_with_pattern_attributes():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    assert num("5") == 5


def test_extra_types_basic():
    @with_pattern(r"\d+")
    def num(s):
        return int(s) * 2

    r = parse.parse("v{:Num}", "v21", extra_types={"Num": num})
    assert r.fixed == (42,)
    r = parse.parse("v{x:Num}", "v21", {"Num": num})
    assert r.named == {"x": 42}


def test_extra_types_no_pattern_attr():
    r = parse.parse("{:X1}", "abc", extra_types={"X1": lambda s: s.upper()})
    assert r.fixed == ("ABC",)


def test_extra_types_group_count():
    @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(s):
        return s[::-1]

    r = parse.parse("{:AB} {:d} {}", "ac 7 z", extra_types={"AB": ab})
    assert r.fixed == ("ca", 7, "z")
    r = parse.parse("{n:AB} {m:d} {}", "ac 7 z", extra_types={"AB": ab})
    assert r.named == {"n": "ca", "m": 7}
    assert r.fixed == ("z",)


def test_extra_types_group_count_fixed_after():
    @with_pattern(r"((a)|(b))", regex_group_count=3)
    def ab(s):
        return s

    r = parse.parse("{:AB}-{:d}", "a-5", extra_types={"AB": ab})
    assert r.fixed == ("a", 5)


def test_extra_type_overrides_builtin_name():
    r = parse.parse("{:w}", "a b", extra_types={"w": with_pattern(r".+")(lambda s: s)})
    assert r.fixed == ("a b",)


def test_extra_type_alignment_width():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    r = parse.parse("{:>5Num}", "   42", extra_types={"Num": num})
    assert r.fixed == (42,)


def test_extra_type_pickle_convert_first():
    from parse import convert_first

    c = convert_first(int)
    assert c("5", None) == 5
    assert pickle.loads(pickle.dumps(c))("7", None) == 7
    p = pickle.loads(pickle.dumps(int_convert(16)))
    assert p("ff", None) == 255


# ---------- int_convert ----------

def test_int_convert_unit():
    c = int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0x1f", None) == 31
    assert c("0b11", None) == 3
    assert c("0o11", None) == 9
    assert c("-0b11", None) == -3
    assert c("1,000", None) == 1000
    assert c("0", None) == 0
    assert c("00", None) == 0
    assert c("010", None) == 10
    assert int_convert(2)("101", None) == 5
    assert int_convert(8)("0o17", None) == 15
    assert int_convert(16)("0xFF", None) == 255
    assert int_convert(16)("-ff", None) == -255
    assert int_convert(10)("1.234", None) == 1234


# ---------- search / findall / pos / endpos ----------

def test_search_basic():
    r = search("<{}>", "xx <abc> yy <def>")
    assert r.fixed == ("abc",)
    assert search("<{}>", "nothing") is None


def test_search_pos_endpos():
    s = "<a> <b> <c>"
    assert search("<{}>", s, pos=1).fixed == ("b",)
    assert search("<{}>", s, pos=4).fixed == ("b",)
    assert search("<{}>", s, endpos=3).fixed == ("a",)
    assert search("<{}>", s, pos=8, endpos=11).fixed == ("c",)
    assert search("<{}>", s, endpos=2) is None
    assert search("<{}>", s, pos=11) is None


def test_search_spans_offset():
    r = search("<{}>", "xx <abc>")
    assert r.spans[0] == (4, 7)


def test_findall_basic():
    res = list(findall("<{}>", "<a> <b> <c>"))
    assert [r[0] for r in res] == ["a", "b", "c"]


def test_findall_named():
    res = list(findall("{k}={v};", "a=1;b=2;"))
    assert [r.named for r in res] == [{"k": "a", "v": "1"}, {"k": "b", "v": "2"}]


def test_findall_empty():
    assert list(findall("<{}>", "none")) == []


def test_findall_pos_endpos():
    s = "<a> <b> <c> <d>"
    assert [r[0] for r in findall("<{}>", s, pos=4)] == ["b", "c", "d"]
    assert [r[0] for r in findall("<{}>", s, endpos=11)] == ["a", "b", "c"]
    assert [r[0] for r in findall("<{}>", s, pos=4, endpos=11)] == ["b", "c"]
    assert [r[0] for r in findall("<{}>", s, endpos=0)] == []


def test_findall_non_overlapping():
    res = list(findall("{:d}{:d}", "1234"))
    assert [r.fixed for r in res] == [(123, 4)]
    res = list(findall("aa", "aaaa"))
    assert len(res) == 2


def test_findall_is_iterator():
    it = findall("<{}>", "<a><b>")
    assert iter(it) is it
    assert next(it)[0] == "a"
    assert next(it)[0] == "b"
    with pytest.raises(StopIteration):
        next(it)


def test_findall_evaluate_result_false():
    res = list(findall("<{}>", "<a><b>", evaluate_result=False))
    assert all(isinstance(m, parse.Match) for m in res)
    assert [m.evaluate_result()[0] for m in res] == ["a", "b"]


def test_findall_case_sensitive():
    assert len(list(findall("a{}", "A1 a2", case_sensitive=True))) == 1
    assert len(list(findall("a{}", "A1 a2"))) == 2


def test_search_case_sensitive():
    assert search("ab{}", "AB1", case_sensitive=True) is None
    assert search("ab{}", "AB1").fixed == ("1",)


def test_search_extra_types():
    num = with_pattern(r"\d+")(lambda s: int(s))
    assert search("v{:N}", "xx v12", extra_types={"N": num}).fixed == (12,)
    assert [r[0] for r in findall("v{:N}", "v1 v2", extra_types={"N": num})] == [1, 2]


# ---------- evaluate_result=False / Match ----------

def test_parse_evaluate_result_false():
    m = parse.parse("{:d}", "12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (12,)
    assert m.parser.format == "{:d}"
    assert parse.parse("{:d}", "x", evaluate_result=False) is None


def test_search_evaluate_result_false():
    m = search("{:d}", "a 12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result()[0] == 12
    assert search("{:d}", "a", evaluate_result=False) is None


def test_parser_methods():
    p = compile("{a:d}+{b:d}")
    assert isinstance(p, Parser)
    assert p.parse("1+2").named == {"a": 1, "b": 2}
    assert p.parse("1+x") is None
    assert p.search("zz 1+2 zz").named == {"a": 1, "b": 2}
    assert [r["a"] for r in p.findall("1+2 3+4")] == [1, 3]
    # reusable
    assert p.parse("5+6").named == {"a": 5, "b": 6}


def test_compile_case_sensitive():
    p = compile("AB{}", case_sensitive=True)
    assert p.parse("ab1") is None
    assert p.parse("AB1").fixed == ("1",)
    assert compile("AB{}").parse("ab1").fixed == ("1",)


def test_evaluate_result_via_parser():
    p = compile("{x:d}")
    m = p.parse("5", evaluate_result=False)
    assert p.evaluate_result(m.match).named == {"x": 5}


# ---------- date / time ----------

def test_ti():
    r = parse.parse("{:ti}", "2011-02-03 04:05:06")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6)
    r = parse.parse("{:ti}", "2011-02-03T04:05:06")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6)
    r = parse.parse("{:ti}", "2011-02-03")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:ti}", "2011-02-03 04:05")
    assert r[0] == datetime(2011, 2, 3, 4, 5)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.123")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, 123000)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.000007")
    assert r[0].microsecond == 7
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.1234567")
    assert r[0].microsecond == 123456


def test_ti_tz():
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    assert r[0].replace(tzinfo=None) == datetime(2011, 2, 3, 4, 5, 6)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 -01:30")
    assert r[0].utcoffset() == timedelta(hours=-1, minutes=-30)
    r = parse.parse("{:ti}", "2011-02-03T04:05:06Z")
    assert r[0].utcoffset() == timedelta(0)
    assert r[0].tzinfo == FixedTzOffset(0, "UTC")
    r = parse.parse("{:ti}", "2011-02-03T04:05:06+05:45")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=45)


def test_ti_rejects():
    assert parse.parse("{:ti}", "2011-2-3") is None
    assert parse.parse("{:ti}", "garbage") is None


def test_ti_in_context():
    r = parse.parse("at {:ti} ok {:d}", "at 2011-02-03 04:05:06 ok 5")
    assert r.fixed == (datetime(2011, 2, 3, 4, 5, 6), 5)
    r = parse.parse("{a:ti} {b:ti}", "2011-02-03 2012-03-04")
    assert r.named == {"a": datetime(2011, 2, 3), "b": datetime(2012, 3, 4)}


def test_tg():
    r = parse.parse("{:tg}", "3/2/2011")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:tg}", "03-02-2011 4:05 PM")
    assert r[0] == datetime(2011, 2, 3, 16, 5)
    r = parse.parse("{:tg}", "3/Feb/2011")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:tg}", "3/February/2011 12:00:00 AM")
    assert r[0] == datetime(2011, 2, 3, 0, 0, 0)
    r = parse.parse("{:tg}", "3/2/2011 12:15 PM")
    assert r[0] == datetime(2011, 2, 3, 12, 15)
    r = parse.parse("{:tg}", "3/2/2011 12:15 AM")
    assert r[0] == datetime(2011, 2, 3, 0, 15)
    r = parse.parse("{:tg}", "3/2/2011 1:15:30 PM +0200")
    assert r[0].replace(tzinfo=None) == datetime(2011, 2, 3, 13, 15, 30)
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:tg}", "3/2/2011 13:15")
    assert r[0] == datetime(2011, 2, 3, 13, 15)
    r = parse.parse("{:tg}", "3/2/2011 1:15 AM")
    assert r[0] == datetime(2011, 2, 3, 1, 15)


def test_ta():
    r = parse.parse("{:ta}", "2/3/2011")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:ta}", "Feb/3/2011")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:ta}", "2-3-2011 4:05:06 PM")
    assert r[0] == datetime(2011, 2, 3, 16, 5, 6)
    r = parse.parse("{:ta}", "2/3/2011 12:30 AM")
    assert r[0] == datetime(2011, 2, 3, 0, 30)
    r = parse.parse("{:ta}", "2/3/2011 12:30 PM")
    assert r[0] == datetime(2011, 2, 3, 12, 30)
    r = parse.parse("{:ta}", "December/25/2011")
    assert r[0] == datetime(2011, 12, 25)


def test_te():
    r = parse.parse("{:te}", "Thu, 03 Feb 2011 04:05:06 +0100")
    assert r[0].replace(tzinfo=None) == datetime(2011, 2, 3, 4, 5, 6)
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:te}", "3 Feb 2011 04:05:06 -0500")
    assert r[0].utcoffset() == timedelta(hours=-5)
    assert parse.parse("{:te}", "3 Feb 2011") is None


def test_th():
    r = parse.parse("{:th}", "03/Feb/2011:04:05:06 +0100")
    assert r[0].replace(tzinfo=None) == datetime(2011, 2, 3, 4, 5, 6)
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:th}", "03/Dec/2011:23:59:59 -0800")
    assert r[0].month == 12 and r[0].hour == 23
    assert r[0].utcoffset() == timedelta(hours=-8)
    assert parse.parse("{:th}", "03/Feb/2011") is None


def test_tc():
    r = parse.parse("{:tc}", "Thu Feb  3 04:05:06 2011")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6)
    r = parse.parse("{:tc}", "Mon Dec 25 23:59:01 1999")
    assert r[0] == datetime(1999, 12, 25, 23, 59, 1)


def test_tt():
    r = parse.parse("{:tt}", "04:05:06")
    assert r[0] == time(4, 5, 6)
    r = parse.parse("{:tt}", "04:05")
    assert r[0] == time(4, 5)
    r = parse.parse("{:tt}", "4:05 PM")
    assert r[0] == time(16, 5)
    r = parse.parse("{:tt}", "12:05 AM")
    assert r[0] == time(0, 5)
    r = parse.parse("{:tt}", "12:05 PM")
    assert r[0] == time(12, 5)
    r = parse.parse("{:tt}", "11:59:59 PM")
    assert r[0] == time(23, 59, 59)
    r = parse.parse("{:tt}", "04:05:06 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    assert r[0].replace(tzinfo=None) == time(4, 5, 6)
    r = parse.parse("{:tt}", "04:05:06.5")
    assert r[0] == time(4, 5, 6, 500000)


def test_tz_formats():
    for s, off in [
        ("+1", None),
        ("+0100", 60),
        ("-0100", -60),
        ("+01:00", 60),
        ("-05:30", -330),
        ("+530", 330),
        ("-930", -570),
    ]:
        r = parse.parse("{:tt}", "04:05 " + s)
        if off is None:
            continue
        assert r[0].utcoffset() == timedelta(minutes=off), s


def test_ts():
    r = parse.parse("{:ts}", "Feb 3 04:05:06")
    y = datetime.today().year
    assert r[0] == datetime(y, 2, 3, 4, 5, 6)
    r = parse.parse("{:ts}", "Dec 25 ")
    assert r[0] == datetime(y, 12, 25)
    r = parse.parse("{:ts}", "Jan  1 00:00:00")
    assert r[0] == datetime(y, 1, 1)


def test_strftime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2011-02-03 04:05:06")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6)


def test_strftime_date_only():
    r = parse.parse("{:%Y-%m-%d}", "2011-02-03")
    assert r[0] == date(2011, 2, 3)
    assert not isinstance(r[0], datetime)


def test_strftime_time_only():
    r = parse.parse("{:%H:%M}", "04:05")
    assert r[0] == time(4, 5)
    assert not isinstance(r[0], datetime)


def test_strftime_missing_year():
    r = parse.parse("{:%m-%d}", "02-03")
    assert r[0] == date(datetime.today().year, 2, 3)


def test_strftime_names_and_ampm():
    r = parse.parse("{:%A %d %B %Y}", "Thursday 03 February 2011")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%a %b %d %y}", "Thu Feb 03 11")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%I:%M %p}", "04:05 PM")
    assert r[0] == time(16, 5)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S.%f}", "2011-02-03 04:05:06.123")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, 123000)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2011-02-03 04:05:06+0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:%Y-%j}", "2011-034")
    assert r[0] == date(2011, 2, 3)


def test_strftime_rejects():
    assert parse.parse("{:%Y-%m-%d}", "2011/02/03") is None
    assert parse.parse("{:%Y}", "11") is None
    assert parse.parse("{:%y}", "2011") is None
    assert parse.parse("{:%H:%M}", "4:5") is None


def test_strftime_in_context():
    r = parse.parse("on {:%Y-%m-%d} at {:%H:%M}", "on 2011-02-03 at 04:05")
    assert r.fixed == (date(2011, 2, 3), time(4, 5))
    r = parse.parse("{d:%Y-%m-%d}|{n:d}", "2011-02-03|5")
    assert r.named == {"d": date(2011, 2, 3), "n": 5}


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("x%y-") == "x[0-9]{2}-"
    assert parse.get_regex_for_datetime_format("%d/%m") == "[0-9]{1,2}/[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("plain") == "plain"


def test_date_groups_consistent_with_following_fields():
    # group indexes of subsequent fields must not drift after complex types
    for t, text in [
        ("ti", "2011-02-03 04:05:06 +0100"),
        ("tg", "3/2/2011 4:05 PM +0100"),
        ("ta", "2/3/2011 4:05 PM +0100"),
        ("te", "Thu, 03 Feb 2011 04:05:06 +0100"),
        ("th", "03/Feb/2011:04:05:06 +0100"),
        ("tc", "Thu Feb  3 04:05:06 2011"),
        ("tt", "04:05:06 PM +0100"),
        ("ts", "Feb 3 04:05:06"),
        ("%Y-%m-%d", "2011-02-03"),
    ]:
        r = parse.parse("{:%s}|{:d}|{}|{:g}" % t, text + "|7|abc|2.5")
        assert r is not None, t
        assert r[1:] == (7, "abc", 2.5), t
        r = parse.parse("{a:%s}|{b:d}|{c}" % t, text + "|7|abc")
        assert r.named["b"] == 7 and r.named["c"] == "abc", t


def test_numeric_group_drift():
    # every numeric type consumes the right number of regex groups
    for t, text, val in [
        ("n", "1,234", 1234),
        ("b", "0b11", 3),
        ("o", "0o7", 7),
        ("x", "0xf", 15),
        ("%", "5%", 0.05),
        ("f", "1.5", 1.5),
        ("e", "1.5e1", 15.0),
        ("g", "1.5e1", 15.0),
        ("d", "15", 15),
        ("w", "ab", "ab"),
    ]:
        r = parse.parse("{:%s}|{}|{:d}" % t, "%s|abc|9" % text)
        assert r.fixed == (val, "abc", 9), t


def test_date_convert_tz_naming():
    r = parse.parse("{:tt}", "04:05 +0100")
    assert r[0].tzname() == "+0100"
    r = parse.parse("{:ti}", "2011-02-03T04:05:06Z")
    assert r[0].tzname() == "UTC"


# ---------- FixedTzOffset ----------

def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert tz == FixedTzOffset(90, "X")
    assert tz != FixedTzOffset(91, "X")
    assert tz != FixedTzOffset(90, "Y")
    assert tz != "X"
    assert "X" in repr(tz) and "FixedTzOffset" in repr(tz)


# ---------- errors ----------

def test_too_many_fields():
    import re
    if sys.version_info >= (3, 11):
        pytest.skip("named-group limit removed in re")
    fmt = " ".join("{:ti}" for _ in range(20))
    with pytest.raises(TooManyFields):
        Parser(fmt).parse("x")


def test_invalid_format_is_valueerror():
    with pytest.raises(ValueError):
        parse.parse("{:xyz}", "a")


def test_bad_group_name_chars():
    # names starting with non-alpha are fixed fields
    assert parse.parse("{_a}", "x").fixed == ("x",)


# ---------- misc ----------

def test_adjacent_fields():
    r = parse.parse("{:d}{:w}", "12ab")
    assert r is not None
    assert r.fixed == (12, "ab")


def test_literal_text_with_braces_and_spec_like():
    r = parse.parse("{a} : {b}", "x : y")
    assert r.named == {"a": "x", "b": "y"}


def test_unicode():
    assert parse.parse("héllo {}", "héllo wörld").fixed == ("wörld",)


def test_multiple_calls_independent():
    assert parse.parse("{:d}", "1").fixed == (1,)
    assert parse.parse("{:d}", "2").fixed == (2,)


def test_module_exports():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
    assert isinstance(parse.__version__, str)


def test_int_negative_with_alignment():
    assert parse.parse("{:>6d}", "   -42").fixed == (-42,)
    assert parse.parse("{:<6d}", "-42   ").fixed == (-42,)
    assert parse.parse("{:+d}", "+42").fixed == (42,)
    assert parse.parse("{:+d}", "-42").fixed == (-42,)


def test_float_negative_search():
    assert search("{:f}", "t=-2.5C").fixed == (-2.5,)


def test_findall_with_dates():
    res = list(findall("{:ti}", "a 2011-02-03 b 2012-04-05 c"))
    assert [r[0] for r in res] == [datetime(2011, 2, 3), datetime(2012, 4, 5)]
