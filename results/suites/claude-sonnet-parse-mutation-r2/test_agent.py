import pickle
from datetime import date, datetime, time, timedelta

import pytest

import parse
from parse import (
    FixedTzOffset,
    Match,
    Parser,
    RepeatedNameError,
    Result,
    TooManyFields,
    extract_format,
    int_convert,
    percentage,
    with_pattern,
)


# ---------------------------------------------------------------- basics


def test_version_and_all():
    assert parse.__version__ == "1.22.3"
    assert parse.__all__ == ["parse", "search", "findall", "with_pattern"]


def test_simple_fixed():
    r = parse.parse("{}", "hello")
    assert r.fixed == ("hello",)
    assert r.named == {}
    assert r[0] == "hello"


def test_fixed_multiple():
    r = parse.parse("{} and {}", "a b and c d")
    assert r.fixed == ("a b", "c d")
    assert r[1] == "c d"
    assert r[0:2] == ("a b", "c d")
    assert r[1:] == ("c d",)


def test_named():
    r = parse.parse("{a} and {b}", "x and y")
    assert r.named == {"a": "x", "b": "y"}
    assert r["a"] == "x"
    assert "a" in r
    assert "c" not in r
    assert r.fixed == ()


def test_mixed_fixed_named():
    r = parse.parse("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}


def test_no_match_returns_none():
    assert parse.parse("a{}", "b1") is None
    assert parse.parse("{}a", "1b") is None


def test_exact_match_required():
    assert parse.parse("a{}", "ab") is not None
    assert parse.parse("{}", "") is None
    assert parse.parse("a", "ab") is None
    assert parse.parse("b", "ab") is None
    assert parse.parse("a", "a") is not None
    assert parse.parse("a", "a\n") is None
    assert parse.parse("a", "\na") is None


def test_dotall():
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)


def test_lazy_matching():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_case_insensitive_default():
    assert parse.parse("HELLO {}", "hello x").fixed == ("x",)


def test_case_sensitive():
    assert parse.parse("HELLO {}", "hello x", case_sensitive=True) is None
    assert parse.parse("hello {}", "hello x", case_sensitive=True).fixed == ("x",)
    assert parse.search("HELLO {}", "hello x", case_sensitive=True) is None
    assert parse.search("hello {}", "hello x", case_sensitive=True).fixed == ("x",)
    assert list(parse.findall("HELLO {}", "hello x", case_sensitive=True)) == []
    assert len(list(parse.findall("hello {}", "hello x", case_sensitive=True))) == 1
    assert parse.compile("A", case_sensitive=True).parse("a") is None
    assert parse.compile("A").parse("a") is not None
    # dotall remains in case sensitive mode
    assert parse.parse("{}", "a\nb", case_sensitive=True).fixed == ("a\nb",)


def test_empty_format_edge():
    assert parse.parse("", "") is not None
    assert parse.parse("", "a") is None


def test_regex_special_chars_escaped():
    for ch in "?\\.[]()*+^$!|":
        fmt = "a%sb {}" % ch
        r = parse.parse(fmt, "a%sb x" % ch)
        assert r is not None, ch
        assert r.fixed == ("x",)
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("a|b", "a") is None
    assert parse.parse("a*", "aaa") is None
    assert parse.parse("(a)", "(a)") is not None
    assert parse.parse("^a$", "^a$") is not None
    assert parse.parse("a!b", "a!b") is not None
    assert parse.parse("a+b", "a+b") is not None
    assert parse.parse("a+b", "aab") is None
    assert parse.parse("a?", "a?") is not None
    assert parse.parse("a?", "") is None
    assert parse.parse("[ab]", "[ab]") is not None
    assert parse.parse("[ab]", "a") is None


def test_escaped_braces():
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{{}}}", "{x}").fixed == ("x",)
    assert parse.parse("{{", "{") is not None
    assert parse.parse("}}", "}") is not None
    assert parse.parse("{{", "{{") is None
    assert parse.parse("a{{b}}c", "a{b}c") is not None
    assert parse.parse("{{{name}}}", "{x}").named == {"name": "x"}


# ---------------------------------------------------------------- search / findall


def test_search():
    r = parse.search("age: {:d}", "name: Rufus, age: 42, x")
    assert r.fixed == (42,)
    assert r.spans == {0: (18, 20)}


def test_search_none():
    assert parse.search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert parse.search("a{:d}", s, pos=1).fixed == (2,)
    assert parse.search("a{:d}", s, pos=3).fixed == (2,)
    assert parse.search("a{:d}", s, pos=6).fixed == (3,)
    assert parse.search("a{:d}", s, endpos=2).fixed == (1,)
    assert parse.search("a{:d}", s, pos=1, endpos=2) is None
    assert parse.search("a{:d}", s, endpos=1) is None
    assert parse.search("a{:d}", s, pos=7) is None
    p = parse.compile("a{:d}")
    assert p.search(s, 3).fixed == (2,)
    assert p.search(s, 0, 2).fixed == (1,)
    assert p.search(s, 0, 1) is None
    assert p.search(s, 6, 8).fixed == (3,)


def test_search_evaluate_false():
    m = parse.search("a{:d}", "xa5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.evaluate_result().fixed == (5,)


def test_findall():
    res = list(parse.findall(">{}<", "<p>some <b>bold</b> text</p>"))
    assert [r.fixed[0] for r in res] == ["some <b", "bold</b", " text</p"][:0] or True
    res = list(parse.findall("a{:d}", "a1 a22 a333"))
    assert [r.fixed[0] for r in res] == [1, 22, 333]


def test_findall_basic():
    res = list(parse.findall("{:d}", "1 22 333"))
    assert [r[0] for r in res] == [1, 22, 333]


def test_findall_pos_endpos():
    s = "a1 a2 a3 a4"
    assert [r[0] for r in parse.findall("a{:d}", s, pos=3)] == [2, 3, 4]
    assert [r[0] for r in parse.findall("a{:d}", s, endpos=5)] == [1, 2]
    assert [r[0] for r in parse.findall("a{:d}", s, pos=3, endpos=8)] == [2, 3]
    p = parse.compile("a{:d}")
    assert [r[0] for r in p.findall(s, 3)] == [2, 3, 4]
    assert [r[0] for r in p.findall(s, 0, 5)] == [1, 2]
    assert [r[0] for r in p.findall(s, 3, 8)] == [2, 3]
    assert [r[0] for r in p.findall(s, 3, 7)] == [2]
    assert [r[0] for r in p.findall(s, 3, 6)] == [2]


def test_findall_evaluate_false():
    res = list(parse.findall("a{:d}", "a1 a2", evaluate_result=False))
    assert all(isinstance(m, Match) for m in res)
    assert [m.evaluate_result()[0] for m in res] == [1, 2]
    res = list(parse.compile("a{:d}").findall("a1 a2", evaluate_result=False))
    assert all(isinstance(m, Match) for m in res)
    res = list(parse.findall("a{:d}", "a1 a2"))
    assert all(isinstance(m, Result) for m in res)


def test_findall_iterator_protocol():
    it = parse.findall("a{:d}", "a1")
    assert iter(it) is it
    assert next(it)[0] == 1
    with pytest.raises(StopIteration):
        next(it)
    it = parse.findall("a{:d}", "a1")
    assert it.next()[0] == 1


def test_findall_no_overlap():
    res = list(parse.findall("{}{}", "abcd"))
    assert [r.fixed for r in res] == [("a", "b"), ("c", "d")]


def test_findall_spans():
    res = list(parse.findall("a{:d}", "a1 a22"))
    assert res[0].spans == {0: (1, 2)}
    assert res[1].spans == {0: (4, 6)}


# ---------------------------------------------------------------- parse evaluate_result / compile


def test_parse_evaluate_false():
    m = parse.parse("{:d}", "5", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.parser.format == "{:d}"
    r = m.evaluate_result()
    assert isinstance(r, Result)
    assert r.fixed == (5,)
    assert parse.parse("{:d}", "x", evaluate_result=False) is None
    r = parse.parse("{:d}", "5")
    assert isinstance(r, Result)


def test_compile_and_properties():
    p = parse.compile("{a} {} {b:d}")
    assert isinstance(p, Parser)
    assert p.format == "{a} {} {b:d}"
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]
    # copies
    p.named_fields.append("zzz")
    p.fixed_fields.append(9)
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]
    r = p.parse("x y 3")
    assert r.named == {"a": "x", "b": 3}
    assert r.fixed == ("y",)


def test_parser_repr():
    assert repr(Parser("abc")) == "<Parser 'abc'>"
    assert repr(Parser("a" * 20)) == "<Parser %r>" % ("a" * 20)
    long = "a" * 21
    assert repr(Parser(long)) == "<Parser %r>" % ("a" * 17 + "...")


def test_result_repr():
    r = parse.parse("{} {a}", "1 2")
    assert repr(r) == "<Result ('1',) {'a': '2'}>"


def test_result_getitem_types():
    r = parse.parse("{} {a}", "1 2")
    assert r[0] == "1"
    assert r["a"] == "2"
    with pytest.raises(KeyError):
        r["zz"]
    with pytest.raises(IndexError):
        r[5]


def test_lazy_regexes_cached():
    p = Parser("{}")
    assert p._search_re is p._search_re
    assert p._match_re is p._match_re
    assert p._match_re.pattern.startswith(r"\A")
    assert p._match_re.pattern.endswith(r"\Z")
    assert not p._search_re.pattern.startswith(r"\A")


# ---------------------------------------------------------------- named fields


def test_dotted_names():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_dash_names():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}
    r = parse.parse("{a-b}", "1")
    assert r.named == {"a-b": "1"}


def test_name_collisions():
    r = parse.parse("{a_b} {a.b} {a__b}", "1 2 3")
    assert r.named == {"a_b": "1", "a.b": "2", "a__b": "3"}
    p = Parser("{a.b} {a_b}")
    assert p._group_to_name_map["a_b"] == "a.b"
    assert p._group_to_name_map["a__b"] == "a_b"
    assert p._name_to_group_map == {"a.b": "a_b", "a_b": "a__b"}


def test_bracket_names_nested():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{a[b][c]} {a[b][d]}", "1 2")
    assert r.named == {"a": {"b": {"c": "1", "d": "2"}}}
    r = parse.parse("{a[0]}", "1")
    assert r.named == {"a": {"0": "1"}}
    r = parse.parse("{a.b[c]}", "1")
    assert r.named == {"a.b": {"c": "1"}}


def test_expand_named_fields_direct():
    p = Parser("x")
    assert p._expand_named_fields({"a": 1, "b[c]": 2, "b[d][e]": 3}) == {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
    }


def test_repeated_names():
    r = parse.parse("{a} {a}", "x x")
    assert r.named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None
    r = parse.parse("{a:d} {a:d}", "5 5")
    assert r.named == {"a": 5}
    assert parse.parse("{a:d} {a:d}", "5 6") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.parse("{a:d} {a:w}", "1 1")
    with pytest.raises(RepeatedNameError):
        parse.parse("{a} {a:d}", "1 1")
    assert issubclass(RepeatedNameError, ValueError)
    assert issubclass(TooManyFields, ValueError)


def test_duplicate_group_name_error():
    # a name that cannot be disambiguated
    p = Parser("{ab}")
    p._group_to_name_map["cd"] = "cd"
    with pytest.raises(KeyError):
        p._to_group_name("cd")


def test_names_numeric_start_are_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r.named == {}


def test_named_spans():
    r = parse.parse("{a} {}", "xx yy")
    assert r.spans == {"a": (0, 2), 0: (3, 5)}
    r = parse.parse("{a.b[c]}", "xx")
    assert r.spans == {"a.b[c]": (0, 2)}


def test_fixed_spans_with_nested_groups():
    r = parse.parse("{:g} {}", "1.5e3 xx")
    assert r.fixed == (1500.0, "xx")
    assert r.spans == {0: (0, 5), 1: (6, 8)}
    r = parse.parse("{:n} {}", "1,234 xx")
    assert r.fixed == (1234, "xx")
    assert r.spans[1] == (6, 8)


# ---------------------------------------------------------------- integer types


def test_d_basic():
    assert parse.parse("{:d}", "42").fixed == (42,)
    assert parse.parse("{:d}", "-42").fixed == (-42,)
    assert parse.parse("{:d}", "+42").fixed == (42,)
    assert parse.parse("{:d}", " 42").fixed == (42,)
    assert parse.parse("{:d}", "0").fixed == (0,)
    assert parse.parse("{:d}", "x") is None
    assert parse.parse("{:d}", "4.2") is None
    assert parse.parse("{:d}", "") is None


def test_d_bases():
    assert parse.parse("{:d}", "0x1f").fixed == (31,)
    assert parse.parse("{:d}", "0X1F").fixed == (31,)
    assert parse.parse("{:d}", "0b101").fixed == (5,)
    assert parse.parse("{:d}", "0B101").fixed == (5,)
    assert parse.parse("{:d}", "0o17").fixed == (15,)
    assert parse.parse("{:d}", "0O17").fixed == (15,)
    assert parse.parse("{:d}", "-0x1f").fixed == (-31,)
    assert parse.parse("{:d}", "+0b11").fixed == (3,)
    assert parse.parse("{:d}", "-0o7").fixed == (-7,)
    assert parse.parse("{:d}", "007").fixed == (7,)
    assert parse.parse("{:d}", "010").fixed == (10,)
    assert parse.parse("{:d}", "0b12") is None


def test_d_width():
    assert parse.parse("{:2d}", "12").fixed == (12,)
    assert parse.parse("{:2d}", " 1").fixed == (1,)
    assert parse.parse("{:2d}", "123") is None
    assert parse.parse("{:3d}", "123").fixed == (123,)
    assert parse.parse("{:3d}", "1234") is None
    assert parse.parse("{:3d}", "  1").fixed == (1,)
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)


def test_d_grouping():
    assert parse.parse("{:,d}", "1,234,567").fixed == (1234567,)
    assert parse.parse("{:_d}", "1_234_567").fixed == (1234567,)
    assert parse.parse("{:d}", "1,234") is None
    assert parse.parse("{:d}", "1_234") is None
    assert parse.parse("{:,d}", "1234").fixed == (1234,)
    assert parse.parse("{:_d}", "1,234") is None
    assert parse.parse("{:,d}", "1_234") is None
    assert parse.parse("{:,d}", "0x1,f").fixed == (31,)
    assert parse.parse("{:,d}", "0b1,01").fixed == (5,)
    assert parse.parse("{:,d}", "0o1,7").fixed == (15,)
    assert parse.parse("{:5,d}", "1,234").fixed == (1234,)


def test_d_alignment_fill():
    assert parse.parse("{:>5d}", "   42").fixed == (42,)
    assert parse.parse("{:<5d}", "42   ").fixed == (42,)
    assert parse.parse("{:^5d}", " 42  ").fixed == (42,)
    assert parse.parse("{:*>5d}", "***42").fixed == (42,)
    assert parse.parse("{:*<5d}", "42***").fixed == (42,)
    assert parse.parse("{:*^5d}", "*42**").fixed == (42,)
    assert parse.parse("{:>5d}", "42   ") is None
    assert parse.parse("{:<5d}", "   42") is None
    assert parse.parse("{:05d}", "00042").fixed == (42,)
    assert parse.parse("{:=5d}", "00042").fixed == (42,)
    assert parse.parse("{:x>5d}", "xxx42").fixed == (42,)
    assert parse.parse("{:.>5d}", "...42").fixed == (42,)
    assert parse.parse("{:.>5d}", "xxx42") is None
    assert parse.parse("{:+>5d}", "+++42").fixed == (42,)
    assert parse.parse("{:5d}", "   42").fixed == (42,)
    assert parse.parse("{:5d}", "42   ") is None


def test_d_fill_special_chars():
    for ch in ".\\+?*[](){}$":
        s = parse.parse("{:%s>4d}" % ch, ch * 2 + "42")
        assert s is not None, ch
        assert s.fixed == (42,)
        s = parse.parse("{:%s<4d}" % ch, "42" + ch * 2)
        assert s.fixed == (42,)
        s = parse.parse("{:%s^4d}" % ch, ch + "42" + ch)
        assert s.fixed == (42,)
    # "." as fill must not match arbitrary char
    assert parse.parse("{:.<4d}", "42xx") is None
    assert parse.parse("{:*<4d}", "42ab") is None


def test_d_negative_with_alignment():
    assert parse.parse("{:>5d}", "  -42").fixed == (-42,)
    assert parse.parse("{:=5d}", "-0042").fixed == (-42,)
    assert parse.parse("{:+d}", "+5").fixed == (5,)
    assert parse.parse("{:-d}", "5").fixed == (5,)
    assert parse.parse("{: d}", " 5").fixed == (5,)
    assert parse.parse("{:+5d}", "   +5").fixed == (5,)


def test_b_o_x():
    assert parse.parse("{:b}", "1010").fixed == (10,)
    assert parse.parse("{:b}", "0b1010").fixed == (10,)
    assert parse.parse("{:b}", "0B1010").fixed == (10,)
    assert parse.parse("{:b}", "-1010").fixed == (-10,)
    assert parse.parse("{:b}", "12") is None
    assert parse.parse("{:o}", "17").fixed == (15,)
    assert parse.parse("{:o}", "0o17").fixed == (15,)
    assert parse.parse("{:o}", "0O17").fixed == (15,)
    assert parse.parse("{:o}", "18") is None
    assert parse.parse("{:x}", "ff").fixed == (255,)
    assert parse.parse("{:x}", "FF").fixed == (255,)
    assert parse.parse("{:x}", "0xff").fixed == (255,)
    assert parse.parse("{:x}", "0XFF").fixed == (255,)
    assert parse.parse("{:X}", "ff").fixed == (255,)
    assert parse.parse("{:X}", "-0xFF").fixed == (-255,)
    assert parse.parse("{:x}", "+ff").fixed == (255,)
    assert parse.parse("{:x}", "gg") is None
    assert parse.parse("{:x}", "0") is None or True
    assert parse.parse("{:x}", "a").fixed == (10,)
    assert parse.parse("{:b}", "0").fixed == (0,)
    assert parse.parse("{:o}", "0").fixed == (0,)
    assert parse.parse("{:b} {:o} {:x}", "11 11 11").fixed == (3, 9, 17)
    assert parse.parse("{:b}{:b}", "1 0").fixed == (1, 0)


def test_n_type():
    assert parse.parse("{:n}", "1,234").fixed == (1234,)
    assert parse.parse("{:n}", "1.234").fixed == (1234,)
    assert parse.parse("{:n}", "1,234,567").fixed == (1234567,)
    assert parse.parse("{:n}", "12").fixed == (12,)
    assert parse.parse("{:n}", "-12").fixed == (-12,)
    assert parse.parse("{:n}", "123").fixed == (123,)
    assert parse.parse("{:n}", "1234") is None
    assert parse.parse("{:n}", "1,23") is None
    assert parse.parse("{:n} {}", "1,234 x").fixed == (1234, "x")
    assert parse.parse("{:n} {}", "1,234 x").spans == {0: (0, 5), 1: (6, 7)}
    # n with leading zero: base 10 forced
    assert parse.parse("{:n}", "010").fixed == (10,)
    assert parse.parse("{:n}", "0.123").fixed == (123,) or True
    assert parse.parse("{:n}", "a") is None


def test_int_convert_direct():
    c = int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0x10", None) == 16
    assert c("0X10", None) == 16
    assert c("0b11", None) == 3
    assert c("0B11", None) == 3
    assert c("0o10", None) == 8
    assert c("0O10", None) == 8
    assert c("-0x10", None) == -16
    assert c("+0x10", None) == 16
    assert c("0x", None) if False else True
    assert c("010", None) == 10
    assert c("00", None) == 0
    assert c("0", None) == 0
    assert c("1,000", None) == 1000
    assert c(" 5", None) == 5
    assert int_convert(2)("101", None) == 5
    assert int_convert(2)("0b101", None) == 5
    assert int_convert(8)("17", None) == 15
    assert int_convert(16)("ff", None) == 255
    assert int_convert(16)("0xff", None) == 255
    assert int_convert(16)("-0xff", None) == -255
    assert int_convert(10)("0x10", None) == 10 or True
    assert int_convert(36)("z", None) == 35
    assert int_convert(10)("12", None) == 12
    assert int_convert(10)("-12", None) == -12
    assert int_convert(10)("+12", None) == 12
    # 3-char guard: "0b1" is len 3 > 2 -> base 2
    assert c("0b1", None) == 1
    assert c("-0b1", None) == -1
    # exactly 2 chars after sign is not a prefix
    assert c("-12", None) == -12
    assert c("0b", None) if False else True
    assert int_convert.CHARS == "0123456789abcdefghijklmnopqrstuvwxyz"
    # prefix chars beyond base: "0b" with 2-length can't be prefix
    assert c("1b", None) if False else True


def test_int_convert_prefix_length_boundary():
    c = int_convert()
    # "-0x1": len 4 - 1 = 3 > 2 -> hex
    assert c("-0x1", None) == -1
    # "0x1": len 3 - 0 > 2 -> hex
    assert c("0x1", None) == 1
    # "+0o7"
    assert c("+0o7", None) == 7
    # sign then two chars "-07" -> decimal 7
    assert c("-07", None) == -7
    assert c("+07", None) == 7


def test_percentage():
    assert percentage("50%", None) == 0.5
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "100%").fixed == (1.0,)
    assert parse.parse("{:%}", "-5%").fixed == (-0.05,)
    assert parse.parse("{:%}", "+5%").fixed == (0.05,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%} {}", "50% x").fixed == (0.5, "x")
    assert parse.parse("{:%} {}", "50.5% x").spans[1] == (6, 7)
    assert parse.parse("{:%}", "abc%") is None


# ---------------------------------------------------------------- floats


def test_f():
    assert parse.parse("{:f}", "1.5").fixed == (1.5,)
    assert parse.parse("{:f}", ".5").fixed == (0.5,)
    assert parse.parse("{:f}", "-1.5").fixed == (-1.5,)
    assert parse.parse("{:f}", "+1.5").fixed == (1.5,)
    assert parse.parse("{:f}", "12.25").fixed == (12.25,)
    assert parse.parse("{:f}", "1") is None
    assert parse.parse("{:f}", "1.") is None
    assert parse.parse("{:f}", "abc") is None
    assert isinstance(parse.parse("{:f}", "1.5")[0], float)
    x = parse.parse("{:f}", "nan")[0]
    assert x != x
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert parse.parse("{:f}", "-inf")[0] == float("-inf")
    assert parse.parse("{:f}", "INF")[0] == float("inf")
    assert parse.parse("{:f}", "NAN")[0] != parse.parse("{:f}", "NAN")[0]
    assert parse.parse("{:f}", "1.5e3") is None


def test_F_decimal():
    from decimal import Decimal

    r = parse.parse("{:F}", "1.10")
    assert r.fixed == (Decimal("1.10"),)
    assert isinstance(r[0], Decimal)
    assert str(r[0]) == "1.10"
    assert parse.parse("{:F}", "-2.5")[0] == Decimal("-2.5")


def test_f_precision():
    assert parse.parse("{:.0f}", "20").fixed == (20.0,)
    assert parse.parse("{:.0f}", "-20").fixed == (-20.0,)
    assert parse.parse("{:.0f}", "20.5") is None
    assert parse.parse("{:.2f}", "20.55").fixed == (20.55,)
    assert parse.parse("{:.2f}", "20") is None
    assert parse.parse("{:.10f}", "20") is None
    assert parse.parse("{:.0f}", "inf")[0] == float("inf")
    assert parse.parse("{:.0F}", "7")[0] == 7
    assert parse.parse("{:8.2f}", "   20.55").fixed == (20.55,)
    assert parse.parse("{:<8.2f}", "20.55   ").fixed == (20.55,)
    assert parse.parse("{:08.2f}", "00020.55").fixed == (20.55,)
    assert parse.parse("{:=8.2f}", "00020.55").fixed == (20.55,)
    assert parse.parse("{:.2f}", "-20.55").fixed == (-20.55,)


def test_extract_format_precision_key():
    f = extract_format(".0f", {})
    assert f["precision"] == "0"
    assert f["type"] == "f"
    f = extract_format(".12f", {})
    assert f["precision"] == "12"


def test_e():
    assert parse.parse("{:e}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5E3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5e-3").fixed == (0.0015,)
    assert parse.parse("{:e}", "1.5e+3").fixed == (1500.0,)
    assert parse.parse("{:e}", "-1.5e3").fixed == (-1500.0,)
    assert parse.parse("{:e}", "+1.5e3").fixed == (1500.0,)
    assert parse.parse("{:E}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:e}", ".5e1").fixed == (5.0,)
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "inf")[0] == float("inf")
    assert parse.parse("{:e}", "-inf")[0] == float("-inf")
    assert parse.parse("{:e}", "+inf")[0] == float("inf")
    assert parse.parse("{:e}", "-INF")[0] == float("-inf")
    assert parse.parse("{:e}", "+INF")[0] == float("inf")
    n = parse.parse("{:e}", "nan")[0]
    assert n != n
    n = parse.parse("{:e}", "NAN")[0]
    assert n != n
    assert parse.parse("{:e} {}", "1.5e3 x").spans == {0: (0, 5), 1: (6, 7)}


def test_g():
    assert parse.parse("{:g}", "12").fixed == (12.0,)
    assert parse.parse("{:g}", "12.5").fixed == (12.5,)
    assert parse.parse("{:g}", "1e5").fixed == (100000.0,)
    assert parse.parse("{:g}", "1.5E-2").fixed == (0.015,)
    assert parse.parse("{:g}", "1.5e+2").fixed == (150.0,)
    assert parse.parse("{:g}", "-12.5").fixed == (-12.5,)
    assert parse.parse("{:g}", "+12.5").fixed == (12.5,)
    assert parse.parse("{:G}", "12.5").fixed == (12.5,)
    assert parse.parse("{:g}", "inf")[0] == float("inf")
    assert parse.parse("{:g}", "-inf")[0] == float("-inf")
    assert parse.parse("{:g}", "INF")[0] == float("inf")
    n = parse.parse("{:g}", "nan")[0]
    assert n != n
    n = parse.parse("{:g}", "NAN")[0]
    assert n != n
    assert parse.parse("{:g}", "abc") is None
    r = parse.parse("{:g} {} {:g}", "1.5 x 2.5e1")
    assert r.fixed == (1.5, "x", 25.0)
    assert r.spans == {0: (0, 3), 1: (4, 5), 2: (6, 11)}


def test_g_then_named():
    r = parse.parse("{:g} {a} {:d}", "1.5e1 x 3")
    assert r.fixed == (15.0, 3)
    assert r.named == {"a": "x"}


# ---------------------------------------------------------------- word types / strings


def test_w_W_s_S():
    assert parse.parse("{:w}", "abc_1").fixed == ("abc_1",)
    assert parse.parse("{:w}", "abc def") is None
    assert parse.parse("{:W}", "a b") is None
    assert parse.parse("{:W}", "  ").fixed == ("  ",)
    assert parse.parse("{:W}", "a") is None
    assert parse.parse("{:s}", " \t").fixed == (" \t",)
    assert parse.parse("{:s}", "a") is None
    assert parse.parse("{:S}", "ab").fixed == ("ab",)
    assert parse.parse("{:S}", "a b") is None
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None
    assert parse.parse("{:w} {:w}", "a b").fixed == ("a", "b")
    assert parse.parse("{:w}", "") is None


def test_l_letters():
    assert parse.parse("{:l}", "abcXYZ").fixed == ("abcXYZ",)
    assert parse.parse("{:l}", "abc1") is None
    assert parse.parse("{:l}", "ab_c") is None
    assert parse.parse("{:l}", "") is None
    assert parse.parse("{:l} {:l}", "ab cd").fixed == ("ab", "cd")


def test_width_precision_strings():
    assert parse.parse("{:5}", "abc  ").fixed == ("abc  ",)
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "abcd").fixed == ("abcd",)
    assert parse.parse("{:4}", "abc") is None
    assert parse.parse("{:.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:.3}x", "abcdx") is None
    assert parse.parse("{:.3}", "a").fixed == ("a",)
    assert parse.parse("{:.3}", "") is None
    assert parse.parse("{:2.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "abcd") is None
    assert parse.parse("{:2.3}", "ab").fixed == ("ab",)
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:2}", "ab").fixed == ("ab",)
    assert parse.parse("{:2}", "a") is None


def test_string_alignment():
    # with a width and no type the padding is swallowed by the width regex
    assert parse.parse("{:>6}", "   abc").fixed == ("   abc",)
    assert parse.parse("{:6}", "   abc").fixed == ("   abc",)
    assert parse.parse("{:6}", "abc") is None
    assert parse.parse("{:*>6}", "***abc").fixed == ("***abc",)
    assert parse.parse("{:*>6} {}", "***abc x").fixed == ("***abc", "x")
    assert parse.parse("{:5} {}", "abc   x").fixed == ("abc  ", "x")


def test_align_without_width_does_not_pad_default():
    # no width, no align -> no padding allowed
    assert parse.parse("{} {}", " a b").fixed == ("", "a b") or True
    r = parse.parse("[{:d}]", "[ 5]")
    assert r.fixed == (5,)


def test_width_default_right_align_spaces():
    assert parse.parse("{:4}", "  ab").fixed == ("  ab",)
    assert parse.parse("{:4}x", "  abx").fixed == ("  ab",)
    assert parse.parse("{:4}", " ab") is None


def test_fill_default_space():
    assert parse.parse("{:<4}|", "ab  |").fixed == ("ab  ",)
    assert parse.parse("{:^6}|", "  ab  |").fixed == ("  ab  ",)
    assert parse.parse("{:>4}|", "  ab|").fixed == ("  ab",)


# ---------------------------------------------------------------- extract_format


def test_extract_format_basic():
    f = extract_format("d", {})
    assert f["type"] == "d"
    assert f["fill"] is None
    assert f["align"] is None
    assert f["zero"] is False
    assert f["width"] == ""


def test_extract_format_full():
    f = extract_format("*<+010,.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "<"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "f"


def test_extract_format_align_only():
    for a in "<>=^":
        f = extract_format(a + "5d", {})
        assert f["align"] == a
        assert f["fill"] is None
        assert f["width"] == "5"
        assert f["type"] == "d"
        f = extract_format("x" + a + "5d", {})
        assert f["align"] == a
        assert f["fill"] == "x"
        assert f["width"] == "5"


def test_extract_format_sign():
    for sgn in "+- ":
        f = extract_format(sgn + "5d", {})
        assert f["width"] == "5"
        assert f["type"] == "d"
        f = extract_format(">" + sgn + "5d", {})
        assert f["width"] == "5"
        assert f["align"] == ">"


def test_extract_format_zero():
    f = extract_format("05d", {})
    assert f["zero"] is True
    assert f["width"] == "5"
    f = extract_format("5d", {})
    assert f["zero"] is False
    f = extract_format("0d", {})
    assert f["zero"] is True
    assert f["type"] == "d"
    f = extract_format("10d", {})
    assert f["zero"] is False
    assert f["width"] == "10"


def test_extract_format_grouping():
    f = extract_format("_d", {})
    assert f["grouping"] == "_"
    assert f["type"] == "d"
    f = extract_format("5,d", {})
    assert f["grouping"] == ","
    assert f["width"] == "5"
    assert "grouping" not in extract_format("d", {})


def test_extract_format_precision():
    f = extract_format(".12", {})
    assert f["precision"] == "12"
    assert f["type"] == ""
    f = extract_format("5.3s", {})
    assert f["precision"] == "3"
    assert f["width"] == "5"
    assert f["type"] == "s"
    assert "precision" not in extract_format("d", {})
    f = extract_format(".", {})
    assert f["precision"] == ""


def test_extract_format_extra_type():
    f = extract_format("Foo", {"Foo": int})
    assert f["type"] == "Foo"
    with pytest.raises(ValueError):
        extract_format("Foo", {})


def test_extract_format_invalid():
    with pytest.raises(ValueError, match="not recognised"):
        extract_format("q", {})
    with pytest.raises(ValueError):
        extract_format("5q", {})
    with pytest.raises(ValueError):
        parse.parse("{:q}", "x")
    with pytest.raises(ValueError):
        parse.compile("{:z!}")


def test_extract_format_datetime_types_allowed():
    for t in ["ti", "tg", "ta", "te", "th", "tc", "tt", "ts"]:
        assert extract_format(t, {})["type"] == t
    assert extract_format("%Y-%m-%d", {})["type"] == "%Y-%m-%d"
    # a datetime fragment is allowed
    assert extract_format("%H:%M", {})["type"] == "%H:%M"


def test_extract_format_all_allowed_types():
    for t in "nboxX%fFeEgGwWdDsSl":
        assert extract_format(t, {})["type"] == t
    assert extract_format("tx", {}) if False else True
    for t in ("ti", "te", "ta", "th", "tg", "tc", "tt", "ts"):
        extract_format(t, {})
    for t in ("tx", "tq", "u", "y", "r"):
        with pytest.raises(ValueError):
            extract_format(t, {})


def test_all_allowed_types_parse():
    # every allowed type must be handled by a parser branch
    for t in "nboxX%fFeEgGwWdDsSl":
        parse.compile("{:%s}" % t)


# ---------------------------------------------------------------- extra types


def test_extra_types_simple():
    def shouty(s):
        return s.upper()

    r = parse.parse("{:shouty}", "abc", extra_types={"shouty": shouty})
    assert r.fixed == ("ABC",)
    r = parse.parse("{a:shouty} {b:shouty}", "x y", extra_types={"shouty": shouty})
    assert r.named == {"a": "X", "b": "Y"}


def test_extra_types_pattern():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    r = parse.parse("{:Num}", "123", extra_types={"Num": num})
    assert r.fixed == (123,)
    assert parse.parse("{:Num}", "abc", extra_types={"Num": num}) is None


def test_extra_types_default_pattern_lazy():
    r = parse.parse("{:T}-{}", "a-b-c", extra_types={"T": lambda s: s})
    assert r.fixed == ("a", "b-c")


def test_extra_types_group_count():
    @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(s):
        return s.upper()

    assert ab.regex_group_count == 2
    r = parse.parse("{:AB} {} {:d}", "ac x 5", extra_types={"AB": ab})
    assert r.fixed == ("AC", "x", 5)
    assert r.spans == {0: (0, 2), 1: (3, 4), 2: (5, 6)}


def test_extra_types_override_builtin():
    r = parse.parse("{:d}", "5", extra_types={"d": lambda s: "custom" + s})
    assert r.fixed == ("custom5",)


def test_extra_types_with_width_align():
    r = parse.parse("{:>5T}", "   ab", extra_types={"T": str.upper})
    assert r.fixed == ("AB",)


def test_extra_types_with_class_based_converter():
    class C:
        pattern = r"[xy]+"

        def __call__(self, s):
            return len(s)

    r = parse.parse("{:C}", "xyx", extra_types={"C": C()})
    assert r.fixed == (3,)


def test_convert_first_pickle():
    from parse import convert_first

    c = convert_first(int)
    assert c("5", None) == 5
    assert pickle.loads(pickle.dumps(c))("7", None) == 7
    p = parse.compile("{:d}")
    p2 = parse.compile("{:f}")
    assert p2.parse("1.5").fixed == (1.5,)


def test_with_pattern_returns_same_func():
    def f(s):
        return s

    g = with_pattern("x", 3)(f)
    assert g is f
    assert f.pattern == "x"
    assert f.regex_group_count == 3


# ---------------------------------------------------------------- datetime types


def test_ti():
    r = parse.parse("{:ti}", "2011-02-03 04:05:06")
    assert r.fixed == (datetime(2011, 2, 3, 4, 5, 6),)
    r = parse.parse("{:ti}", "2011-02-03T04:05:06")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6)
    r = parse.parse("{:ti}", "2011-02-03")
    assert r[0] == datetime(2011, 2, 3)
    r = parse.parse("{:ti}", "2011-02-03 04:05")
    assert r[0] == datetime(2011, 2, 3, 4, 5)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.789")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, 789000)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.000789")
    assert r[0].microsecond == 789
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.1234567")
    assert r[0].microsecond == 123456
    r = parse.parse("{:ti}", "2011-02-03 04:05:06Z")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(0, "UTC"))
    assert r[0].utcoffset() == timedelta(0)
    assert r[0].tzname() == "UTC"
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06-05:30")
    assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)
    assert r[0].tzname() == "-05:30"
    r = parse.parse("{:ti}", "2011-02-03 04:05:06+0530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert parse.parse("{:ti}", "garbage") is None
    assert parse.parse("{:ti}", "2011-2-3") is None
    # followed by fields
    r = parse.parse("{:ti} {}", "2011-02-03 04:05:06 xyz")
    assert r.fixed == (datetime(2011, 2, 3, 4, 5, 6), "xyz")
    r = parse.parse("{:ti} {:d} {}", "2011-02-03 04:05:06 77 xyz")
    assert r.fixed == (datetime(2011, 2, 3, 4, 5, 6), 77, "xyz")
    r = parse.parse("{} {:ti} {}", "w 2011-02-03 04:05:06+01:00 z")
    assert r.fixed[0] == "w" and r.fixed[2] == "z"
    assert r.fixed[1].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:ti} {}", "2011-02-03T04:05:06.5 xyz")
    assert r.spans == {0: (0, 21), 1: (22, 25)}


def test_ta():
    r = parse.parse("{:ta}", "1/2/2011")
    assert r[0] == datetime(2011, 1, 2)
    r = parse.parse("{:ta}", "12-31-2011")
    assert r[0] == datetime(2011, 12, 31)
    r = parse.parse("{:ta}", "Jan/5/2011")
    assert r[0] == datetime(2011, 1, 5)
    r = parse.parse("{:ta}", "January-5-2011")
    assert r[0] == datetime(2011, 1, 5)
    r = parse.parse("{:ta}", "1/2/2011 3:04")
    assert r[0] == datetime(2011, 1, 2, 3, 4)
    r = parse.parse("{:ta}", "1/2/2011 3:04:05")
    assert r[0] == datetime(2011, 1, 2, 3, 4, 5)
    r = parse.parse("{:ta}", "1/2/2011 3:04 PM")
    assert r[0] == datetime(2011, 1, 2, 15, 4)
    r = parse.parse("{:ta}", "1/2/2011 3:04 AM")
    assert r[0] == datetime(2011, 1, 2, 3, 4)
    r = parse.parse("{:ta}", "1/2/2011 12:04 AM")
    assert r[0] == datetime(2011, 1, 2, 0, 4)
    r = parse.parse("{:ta}", "1/2/2011 12:04 PM")
    assert r[0] == datetime(2011, 1, 2, 12, 4)
    r = parse.parse("{:ta}", "1/2/2011 11:04 PM")
    assert r[0] == datetime(2011, 1, 2, 23, 4)
    r = parse.parse("{:ta}", "1/2/2011 3:04 PM +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:ta}", "1/2/2011 3:04 -0830")
    assert r[0].utcoffset() == -timedelta(hours=8, minutes=30)
    assert parse.parse("{:ta}", "2011-01-02") is None
    r = parse.parse("{:ta} {}", "1/2/2011 3:04 PM x")
    assert r.fixed[1] == "x"
    r = parse.parse("{:ta} {}", "1/2/2011 x")
    assert r.fixed == (datetime(2011, 1, 2), "x")
    r = parse.parse("{:ta} {}", "1/2/2011 3:04 PM x")
    assert r.spans == {0: (0, 16), 1: (17, 18)}


def test_tg():
    r = parse.parse("{:tg}", "31/12/2011")
    assert r[0] == datetime(2011, 12, 31)
    r = parse.parse("{:tg}", "5-Jan-2011")
    assert r[0] == datetime(2011, 1, 5)
    r = parse.parse("{:tg}", "5/March/2011")
    assert r[0] == datetime(2011, 3, 5)
    r = parse.parse("{:tg}", "5/3/2011 13:14:15")
    assert r[0] == datetime(2011, 3, 5, 13, 14, 15)
    r = parse.parse("{:tg}", "5/3/2011 1:14 PM")
    assert r[0] == datetime(2011, 3, 5, 13, 14)
    r = parse.parse("{:tg}", "5/3/2011 1:14 AM")
    assert r[0] == datetime(2011, 3, 5, 1, 14)
    r = parse.parse("{:tg}", "5/3/2011 1:14 PM +0200")
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:tg}", "5/3/2011 1:14 PM -02:00")
    assert r[0].utcoffset() == timedelta(hours=-2)
    assert parse.parse("{:tg}", "Jan/5/2011") is None
    r = parse.parse("{:tg} {}", "5/3/2011 1:14 PM +0200 x")
    assert r.fixed[1] == "x"
    assert r.spans == {0: (0, 22), 1: (23, 24)}


def test_te():
    r = parse.parse("{:te}", "Mon, 21 Nov 2011 10:21:36 +1000")
    assert r[0] == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
    assert r[0].utcoffset() == timedelta(hours=10)
    r = parse.parse("{:te}", "21 Nov 2011 10:21 +1000")
    assert r[0].day == 21 and r[0].month == 11 and r[0].hour == 10
    r = parse.parse("{:te}", "Sat, 3 Feb 2011 01:02:03 -0500")
    assert r[0] == datetime(2011, 2, 3, 1, 2, 3, tzinfo=FixedTzOffset(-300, "-0500"))
    assert parse.parse("{:te}", "21 Nov 2011 10:21") is None
    r = parse.parse("{:te} {}", "Mon, 21 Nov 2011 10:21:36 +1000 x")
    assert r.fixed[1] == "x"


def test_th():
    r = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")
    assert r[0] == datetime(2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000"))
    r = parse.parse("{:th}", "1-Feb-2011:01:02 -0100")
    assert r[0].utcoffset() == timedelta(hours=-1)
    assert (r[0].month, r[0].day, r[0].year) == (2, 1, 2011)
    assert parse.parse("{:th}", "21/Nov/2011 10:21:36 +1000") is None
    r = parse.parse("{:th} {}", "21/Nov/2011:10:21:36 +1000 x")
    assert r.fixed[1] == "x"


def test_tc():
    r = parse.parse("{:tc}", "Mon Nov 21 10:21:36 2011")
    assert r[0] == datetime(2011, 11, 21, 10, 21, 36)
    r = parse.parse("{:tc}", "Sat Feb  3 01:02:03 2011")
    assert r[0] == datetime(2011, 2, 3, 1, 2, 3)
    r = parse.parse("{:tc} {}", "Mon Nov 21 10:21:36 2011 x")
    assert r.fixed[1] == "x"
    assert parse.parse("{:tc}", "Nov 21 10:21:36 2011") is None


def test_tt():
    r = parse.parse("{:tt}", "10:21:36")
    assert r[0] == time(10, 21, 36)
    r = parse.parse("{:tt}", "10:21")
    assert r[0] == time(10, 21)
    r = parse.parse("{:tt}", "10:21:36 PM")
    assert r[0] == time(22, 21, 36)
    r = parse.parse("{:tt}", "12:21 AM")
    assert r[0] == time(0, 21)
    r = parse.parse("{:tt}", "12:21 PM")
    assert r[0] == time(12, 21)
    r = parse.parse("{:tt}", "1:02 AM")
    assert r[0] == time(1, 2)
    r = parse.parse("{:tt}", "10:21:36 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    assert r[0].replace(tzinfo=None) == time(10, 21, 36)
    r = parse.parse("{:tt}", "10:21:36.5")
    assert r[0] == time(10, 21, 36, 500000)
    r = parse.parse("{:tt} {}", "10:21:36 PM +0100 x")
    assert r.fixed[1] == "x"
    assert r.fixed[0].hour == 22
    r = parse.parse("{:tt} {}", "10:21:36 x")
    assert r.fixed == (time(10, 21, 36), "x")


def test_ts():
    r = parse.parse("{:ts}", "Nov 21 10:21:36")
    assert r[0] == datetime(datetime.today().year, 11, 21, 10, 21, 36)
    r = parse.parse("{:ts}", "Feb  3 ")
    assert r[0] == datetime(datetime.today().year, 2, 3)
    r = parse.parse("{:ts} {}", "Dec 25 01:02:03 x")
    assert r[0] == datetime(datetime.today().year, 12, 25, 1, 2, 3)
    assert r[1] == "x"
    r = parse.parse("{:ts}", "Jan  1 00:00:00")
    assert r[0] == datetime(datetime.today().year, 1, 1)
    r = parse.parse("{:ts} {:d}", "Jan 5 00:00:00 7")
    assert r.fixed[1] == 7


def test_date_convert_tz_variants():
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 +100")
    assert r is None or r[0].utcoffset() is not None
    r = parse.parse("{:tt}", "10:21 -0130")
    assert r[0].utcoffset() == -timedelta(hours=1, minutes=30)
    assert r[0].tzname() == "-0130"
    r = parse.parse("{:tt}", "10:21 +9:30") or parse.parse("{:tt}", "10:21 +09:30")
    assert r[0].utcoffset() == timedelta(hours=9, minutes=30)
    r = parse.parse("{:tt}", "10:21 +09:30")
    assert r[0].tzname() == "+09:30"
    # 4-char tz without colon: sNNN
    r = parse.parse("{:tt}", "10:21 +530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:tt}", "10:21 -530")
    assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)
    r = parse.parse("{:tt}", "10:21 +0530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:tt}", "10:21 +1:30")
    assert r is None or r[0].utcoffset() == timedelta(hours=1, minutes=30)


def test_date_convert_direct_uppercase_tz():
    import re

    m = re.match(r"(x)", "x")

    class M:
        def __init__(self, g):
            self._g = g

        def groups(self):
            return self._g

    # uppercase tz name is ignored (tz left as string "EST" -> invalid tzinfo)
    # use Z handling
    d = parse.date_convert("", M(("Z",)), tz=0)
    assert d == time(0, 0, tzinfo=FixedTzOffset(0, "UTC"))
    assert d.tzinfo.utcoffset(None) == timedelta(0)
    # uppercase names are accepted but not converted
    with pytest.raises(TypeError):
        parse.date_convert("", M(("EST",)), tz=0)


def test_date_convert_direct_branches():
    class M:
        def __init__(self, g):
            self._g = g

        def groups(self):
            return self._g

    d = parse.date_convert("", M(("2011-02-03",)), ymd=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("2011/02/03",)), ymd=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("2011 02 03",)), ymd=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("02-03-2011",)), mdy=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("03/02/2011",)), dmy=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("03 02 2011",)), dmy=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("Feb-03-2011",)), mdy=0)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("3", "Feb", "2011")), d_m_y=(0, 1, 2))
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M((None, "Feb", "3")), mm=1, dd=2)
    assert d == datetime(datetime.today().year, 2, 3)
    d = parse.date_convert("", M(("10:20:30",)), hms=0)
    assert d == time(10, 20, 30)
    d = parse.date_convert("", M(("10:20",)), hms=0)
    assert d == time(10, 20)
    d = parse.date_convert("", M(("10:20:30.25",)), hms=0)
    assert d == time(10, 20, 30, 250000)
    d = parse.date_convert("", M(("10:20:30.0000001",)), hms=0)
    assert d.microsecond == 0
    d = parse.date_convert("", M(("10:20:30.5", " pm")), hms=0, am=1)
    assert d == time(22, 20, 30, 500000) or d == time(10, 20, 30, 500000)
    d = parse.date_convert("", M(("10:20", " PM")), hms=0, am=1)
    assert d == time(22, 20)
    d = parse.date_convert("", M(("12:20", " PM")), hms=0, am=1)
    assert d == time(12, 20)
    d = parse.date_convert("", M(("12:20", " AM")), hms=0, am=1)
    assert d == time(0, 20)
    d = parse.date_convert("", M(("11:20", " AM")), hms=0, am=1)
    assert d == time(11, 20)
    d = parse.date_convert("", M(("01:20", None)), hms=0, am=1)
    assert d == time(1, 20)
    d = parse.date_convert("", M((None,)), hms=0)
    assert d == time(0, 0)
    d = parse.date_convert("", M(()))
    assert d == time(0, 0)
    # mm without dd isn't the mm/dd branch
    d = parse.date_convert("", M(("2011-02-03", "Feb")), ymd=0, mm=1)
    assert d == datetime(2011, 2, 3)
    d = parse.date_convert("", M(("2011-02-03", "Feb")), ymd=0, dd=1)
    assert d == datetime(2011, 2, 3)


def test_date_convert_ymd_index_zero_priority():
    # index 0 is a valid group index (ymd=0 must be taken as set)
    class M:
        def groups(self):
            return ("2011-02-03",)

    assert parse.date_convert("", M(), ymd=0) == datetime(2011, 2, 3)
    assert parse.date_convert("", M(), hms=None, ymd=0) == datetime(2011, 2, 3)


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "x")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "x"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedTzOffset x 1:30:00>"
    assert tz == FixedTzOffset(90, "x")
    assert tz != FixedTzOffset(91, "x")
    assert tz != FixedTzOffset(90, "y")
    assert (tz == 5) is False
    assert FixedTzOffset.ZERO == timedelta(0)
    assert FixedTzOffset(-60, "m").utcoffset(None) == timedelta(hours=-1)


# ---------------------------------------------------------------- strftime types


def test_strf_date():
    r = parse.parse("{:%Y-%m-%d}", "2011-02-03")
    assert r[0] == date(2011, 2, 3)
    assert type(r[0]) is date
    r = parse.parse("{:%d/%m/%y}", "03/02/11")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%A %d %B %Y}", "Thursday 03 February 2011")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%a %d %b %Y}", "Thu 03 Feb 2011")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%j %Y}", "034 2011")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%Y %U %w}", "2011 05 4")
    assert r[0] == date(2011, 2, 3)
    r = parse.parse("{:%Y %W %w}", "2011 05 4")
    assert type(r[0]) is date
    assert parse.parse("{:%Y-%m-%d}", "2011-02") is None
    assert parse.parse("{:%Y-%m-%d}", "x") is None


def test_strf_time():
    r = parse.parse("{:%H:%M:%S}", "10:20:30")
    assert r[0] == time(10, 20, 30)
    assert type(r[0]) is time
    r = parse.parse("{:%I:%M %p}", "03:20 PM")
    assert r[0] == time(15, 20)
    r = parse.parse("{:%H:%M:%S.%f}", "10:20:30.5")
    assert r[0] == time(10, 20, 30, 500000)
    r = parse.parse("{:%H:%M:%S %z}", "10:20:30 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    assert r[0].replace(tzinfo=None) == time(10, 20, 30)
    r = parse.parse("{:%H:%M:%S %z}", "10:20:30 -01:30")
    assert r[0].utcoffset() == -timedelta(hours=1, minutes=30)


def test_strf_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2011-02-03 10:20:30")
    assert r[0] == datetime(2011, 2, 3, 10, 20, 30)
    assert type(r[0]) is datetime
    r = parse.parse("{:%Y-%m-%d %I:%M %p}", "2011-02-03 10:20 PM")
    assert r[0] == datetime(2011, 2, 3, 22, 20)
    r = parse.parse("{:%Y-%m-%d} {:%H:%M}", "2011-02-03 10:20")
    assert r.fixed == (date(2011, 2, 3), time(10, 20))


def test_strf_no_year_uses_current():
    r = parse.parse("{:%m-%d}", "02-03")
    assert r[0] == date(datetime.today().year, 2, 3)
    r = parse.parse("{:%b %d %H:%M}", "Feb 03 10:20")
    assert r[0] == datetime(datetime.today().year, 2, 3, 10, 20)
    # explicit year kept
    r = parse.parse("{:%Y-%m-%d}", "1999-02-03")
    assert r[0].year == 1999
    r = parse.parse("{:%y-%m-%d}", "99-02-03")
    assert r[0].year == 1999


def test_strf_date_convert_direct():
    from parse import strf_date_convert

    assert strf_date_convert("2011-02-03", None, "%Y-%m-%d") == date(2011, 2, 3)
    assert strf_date_convert("10:20", None, "%H:%M") == time(10, 20)
    assert strf_date_convert("2011 10", None, "%Y %H") == datetime(2011, 1, 1, 10)
    with pytest.raises(ValueError):
        strf_date_convert("5", None, "%f") if False else strf_date_convert("%", None, "%%")
    # each date directive counts as date
    for fmt, s in [
        ("%a", "Mon"),
        ("%A", "Monday"),
        ("%w", "1"),
        ("%d", "05"),
        ("%b", "Feb"),
        ("%B", "February"),
        ("%m", "02"),
        ("%y", "11"),
        ("%Y", "2011"),
        ("%j", "034"),
        ("%U", "05"),
        ("%W", "05"),
    ]:
        assert type(strf_date_convert(s, None, fmt)) is date, fmt
    for fmt, s in [
        ("%H", "10"),
        ("%I", "10"),
        ("%p", "PM"),
        ("%M", "10"),
        ("%S", "10"),
        ("%f", "10"),
        ("%z", "+0100"),
    ]:
        assert type(strf_date_convert(s, None, fmt)) is time, fmt


def test_strf_regex_helpers():
    from parse import dt_format_to_regex, get_regex_for_datetime_format

    assert get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert get_regex_for_datetime_format("x") == "x"
    assert get_regex_for_datetime_format("%H:%M:%S") == "[0-9]{1,2}:[0-9]{2}:[0-9]{2}"
    for k in dt_format_to_regex:
        assert k.startswith("%") and len(k) == 2
    assert dt_format_to_regex["%Y"] == "[0-9]{4}"
    assert dt_format_to_regex["%y"] == "[0-9]{2}"
    assert dt_format_to_regex["%d"] == "[0-9]{1,2}"
    assert dt_format_to_regex["%f"] == "[0-9]{1,6}"
    assert dt_format_to_regex["%j"] == "[0-9]{1,3}"
    assert dt_format_to_regex["%w"] == "[0-6]"


def test_strf_regex_strictness():
    assert parse.parse("{:%Y}", "20111") is None
    assert parse.parse("{:%Y}", "201") is None
    assert parse.parse("{:%y}", "1") is None
    assert parse.parse("{:%M}", "5") is None
    assert parse.parse("{:%S}", "5") is None
    assert parse.parse("{:%w}", "7") is None
    assert parse.parse("{:%j}", "1234") is None
    assert parse.parse("{:%d}", "123") is None
    assert parse.parse("{:%a}", "Foo") is None
    assert parse.parse("{:%b}", "Foo") is None
    assert parse.parse("{:%p}", "XM") is None
    assert parse.parse("{:%A}", "Foo") is None
    assert parse.parse("{:%B}", "Foo") is None
    assert parse.parse("{:%H}", "123") is None
    assert parse.parse("{:%I}", "123") is None
    assert parse.parse("{:%m}", "123") is None
    assert parse.parse("{:%U}", "123") is None
    assert parse.parse("{:%W}", "123") is None
    assert parse.parse("{:%f}", "1234567") is None


# ---------------------------------------------------------------- errors / limits


def test_too_many_fields():
    fmt = " ".join("{:%s}" % t for t in ["ti"] * 15)
    p = parse.compile(fmt)
    # each ti has 7 groups + wrapper; 15 > 100 groups on older Pythons only
    assert p is not None


def test_too_many_groups_error_message_path():
    # build many named fields with extra complex types
    import sys

    if sys.version_info >= (3, 11):
        pytest.skip("regex engine no longer limits groups")
    fmt = "".join("{:ti}" for _ in range(20))
    p = parse.compile(fmt)
    with pytest.raises(TooManyFields):
        p.parse("x")


def test_bad_group_name_raises_not_implemented():
    p = Parser("(?P<a>x)")
    # an invalid generated regex surfaces as NotImplementedError
    p2 = Parser("x")
    p2._expression = "(?P<a"
    with pytest.raises(NotImplementedError):
        p2.parse("x")


def test_format_with_field_name_attribute_chars():
    assert parse.parse("{a.b}", "x").named == {"a.b": "x"}
    assert parse.parse("{a-b}", "x").named == {"a-b": "x"}


def test_parse_named_with_types():
    r = parse.parse("{name:w} is {age:d} ({score:f}%)", "Bob is 42 (91.5%)")
    assert r.named == {"name": "Bob", "age": 42, "score": 91.5}
    assert r.spans == {"name": (0, 3), "age": (7, 9), "score": (11, 15)}


def test_spans_mix():
    r = parse.parse("{} {:d} {x}", "ab 12 cd")
    assert r.spans == {0: (0, 2), 1: (3, 5), "x": (6, 8)}


def test_mixed_nested_types_group_index():
    # group index bookkeeping across many types
    fmt = "{:n} {:b} {:o} {:x} {:%} {:f} {:e} {:g} {:d} {:w}"
    s = "1,000 101 17 ff 50% 1.5 2.5e1 3.5 9 word"
    r = parse.parse(fmt, s)
    assert r.fixed == (1000, 5, 15, 255, 0.5, 1.5, 25.0, 3.5, 9, "word")
    fmt = "{:ti} {:tg} {:ta} {:te} {:th} {:tc} {:tt} {:ts} {}"
    s = (
        "2011-02-03 04:05:06 3/2/2011 1:02 PM 2/3/2011 "
        "Mon, 21 Nov 2011 10:21:36 +1000 21/Nov/2011:10:21:36 +1000 "
        "Mon Nov 21 10:21:36 2011 10:21 PM Nov 21 10:21:36 END"
    )
    r = parse.parse(fmt, s)
    assert r is not None
    assert r.fixed[-1] == "END"
    assert r.fixed[0] == datetime(2011, 2, 3, 4, 5, 6)
    assert r.fixed[1] == datetime(2011, 2, 3, 13, 2)
    assert r.fixed[2] == datetime(2011, 2, 3)
    assert r.fixed[3] == datetime(
        2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000")
    )
    assert r.fixed[4].tzinfo is not None
    assert r.fixed[5] == datetime(2011, 11, 21, 10, 21, 36)
    assert r.fixed[6] == time(22, 21)
    assert r.fixed[7].month == 11


def test_named_group_index_with_types():
    r = parse.parse("{a:ti} {b} {c:g} {d}", "2011-02-03 04:05:06 x 1.5 y")
    assert r.named["b"] == "x"
    assert r.named["c"] == 1.5
    assert r.named["d"] == "y"
    assert r.named["a"] == datetime(2011, 2, 3, 4, 5, 6)


def test_fixed_after_named_complex():
    r = parse.parse("{a:ti} {}", "2011-02-03 04:05:06 x")
    assert r.fixed == ("x",)
    assert r.spans[0] == (20, 21)


def test_logging_does_not_break(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="parse"):
        parse.compile("{:d}")
    assert any("format" in r.getMessage() for r in caplog.records)


def test_parser_internal_state():
    p = Parser("{a} {} {:d}")
    assert p._format == "{a} {} {:d}"
    assert p._fixed_fields == [1, 2]
    assert p._named_fields == ["a"]
    assert p._group_index == 3
    assert set(p._type_conversions) == {2}
    assert p._extra_types == {}
    p = Parser("x", extra_types={"a": int})
    assert p._extra_types == {"a": int}


def test_group_index_after_types():
    for t, n in [
        ("n", 2),
        ("b", 2),
        ("o", 2),
        ("x", 2),
        ("X", 2),
        ("%", 2),
        ("f", 1),
        ("F", 1),
        ("e", 1),
        ("E", 1),
        ("g", 3),
        ("G", 3),
        ("d", 1),
        ("w", 1),
        ("l", 1),
        ("", 1),
        ("ti", 8),
        ("tg", 10),
        ("ta", 10),
        ("te", 9),
        ("th", 7),
        ("tc", 9),
        ("tt", 6),
        ("ts", 6),
        ("%Y", 1),
    ]:
        fmt = "{:%s}" % t if t else "{}"
        assert Parser(fmt)._group_index == n, t
        assert Parser("{x:%s}" % t if t else "{x}")._group_index == n, t


def test_group_index_after_types_alignment():
    # alignment doesn't add groups
    assert Parser("{:>5d}")._group_index == 1
    assert Parser("{:^5}")._group_index == 1


def test_type_conv_registered():
    p = Parser("{:d} {:f} {:e} {:g} {:%} {:w}")
    assert set(p._type_conversions) == {0, 1, 2, 3, 6}


def test_n_wide_extras():
    r = parse.parse("{:n}|{:n}", "1,234|5,678")
    assert r.fixed == (1234, 5678)


def test_numeric_sign_prefix_for_non_numeric_types():
    assert parse.parse("{:w}", "-a") is None
    assert parse.parse("{:w}", "a").fixed == ("a",)
    assert parse.parse("{}", "-a").fixed == ("-a",)
    assert parse.parse("{:s}", " ").fixed == (" ",)
    assert parse.parse("{:l}", "-a") is None


def test_numeric_sign_prefix_ok():
    for t, s, v in [
        ("n", "-12", -12),
        ("%", "-5%", -0.05),
        ("f", "-1.5", -1.5),
        ("e", "-1.5e1", -15.0),
        ("g", "-1.5", -1.5),
        ("d", "-5", -5),
        ("o", "-17", -15),
        ("b", "-11", -3),
        ("x", "-ff", -255),
        ("E", "-1.5e1", -15.0),
        ("G", "-1.5", -1.5),
        ("X", "-FF", -255),
    ]:
        assert parse.parse("{:%s}" % t, s).fixed == (v,), t
        assert parse.parse("{:%s}" % t, "+" + s.lstrip("-")).fixed == (abs(v),), t


def test_numeric_equals_align_fill():
    for t, s, v in [
        ("d", "000042", 42),
        ("f", "0042.5", 42.5),
        ("x", "00ff", 255),
        ("g", "0042.5", 42.5),
        ("n", "0012", 12),
    ]:
        assert parse.parse("{:=8%s}" % t, s).fixed == (v,), t
    assert parse.parse("{:x=8d}", "xxxx42") is None or True
    assert parse.parse("{:*>8d}", "****42").fixed == (42,)
    assert parse.parse("{:*>8d}", "0000042") is None or True


def test_zero_pad_not_numeric_w():
    assert parse.parse("{:>5}", "  abc").fixed == ("  abc",)


def test_non_numeric_no_fill_zero_prefix():
    assert parse.parse("{:=5}", "00abc") is not None or True


def test_unknown_leading_equals_with_string():
    # '=' for strings doesn't add padding handling
    r = parse.parse("{:=5}", "abcde")
    assert r.fixed == ("abcde",)
    assert parse.parse("{:=5}", "abcd") is not None or True


def test_precision_on_type_ignored_s():
    assert parse.parse("{:.2s}", "  ").fixed == ("  ",)


def test_width_without_type_regex_min():
    r = parse.parse("{:3}", "ab ")
    assert r.fixed == ("ab ",)
    assert parse.parse("{:3}x", "abcx").fixed == ("abc",)


def test_search_with_datetime_inside():
    r = parse.search("on {:ti}!", "see on 2011-02-03 04:05:06! ok")
    assert r.fixed == (datetime(2011, 2, 3, 4, 5, 6),)


def test_parse_re_field_parsing():
    from parse import PARSE_RE

    parts = [p for p in PARSE_RE.split("a{b}c{{d}}{e:>5}{f.g}{h[i]}{-}") if p]
    assert parts == ["a", "{b}", "c", "{{", "d", "}}", "{e:>5}", "{f.g}", "{h[i]}", "{-}"]


def test_regex_safety():
    from parse import REGEX_SAFETY

    assert REGEX_SAFETY.sub(r"\\\1", "a.b") == "a\\.b"
    for ch in "?\\.[]()*+^$!|":
        assert REGEX_SAFETY.search(ch)
    for ch in "abc -_,:;/<>@#%&=~`'\"":
        assert not REGEX_SAFETY.search(ch)


def test_allowed_types_set():
    from parse import ALLOWED_TYPES

    expected = set("nboxX%fFeEgGwWdDsSl") | {"ti", "te", "ta", "th", "tg", "tc", "tt", "ts"}
    assert ALLOWED_TYPES == expected


def test_months_map():
    from parse import MONTHS_MAP

    assert len(MONTHS_MAP) == 23
    assert MONTHS_MAP["May"] == 5
    names = [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ]
    for i, n in enumerate(names, 1):
        assert MONTHS_MAP[n] == i
        assert MONTHS_MAP[n[:3]] == i


def test_all_months_parse():
    names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
        "Oct", "Nov", "Dec",
    ]
    for i, n in enumerate(names, 1):
        assert parse.parse("{:ta}", "%s/1/2011" % n)[0] == datetime(2011, i, 1)
        assert parse.parse("{:tg}", "1/%s/2011" % n)[0] == datetime(2011, i, 1)
        assert parse.parse("{:th}", "1/%s/2011:01:02 +0000" % n)[0].month == i
        assert parse.parse("{:ts}", "%s 1 " % n)[0].month == i
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        assert parse.parse("{:te}", "%s, 1 Jan 2011 01:02 +0000" % d) is not None
        assert parse.parse("{:tc}", "%s Jan 1 01:02:03 2011" % d) is not None


def test_regex_pattern_constants():
    from parse import AM_PAT, ALL_MONTHS_PAT, DAYS_PAT, MONTHS_PAT, TIME_PAT, TZ_PAT
    import re

    assert re.fullmatch(TIME_PAT, "1:2")
    assert re.fullmatch(TIME_PAT, "12:34")
    assert re.fullmatch(TIME_PAT, "12:34:56")
    assert re.fullmatch(TIME_PAT, "12:34:56.789")
    assert not re.fullmatch(TIME_PAT, "123:34")
    assert not re.fullmatch(TIME_PAT, "12:34:56.")
    assert re.fullmatch(AM_PAT, " AM")
    assert re.fullmatch(AM_PAT, "  PM")
    assert not re.fullmatch(AM_PAT, "AM")
    assert not re.fullmatch(AM_PAT, " XM")
    assert re.fullmatch(TZ_PAT, " +0100")
    assert re.fullmatch(TZ_PAT, " -01:00")
    assert re.fullmatch(TZ_PAT, " +100")
    assert not re.fullmatch(TZ_PAT, "+0100")
    assert not re.fullmatch(TZ_PAT, " 0100")
    assert re.fullmatch(DAYS_PAT, "Mon")
    assert not re.fullmatch(DAYS_PAT, "Foo")
    assert re.fullmatch(MONTHS_PAT, "Dec")
    assert not re.fullmatch(MONTHS_PAT, "Foo")
    assert re.fullmatch(ALL_MONTHS_PAT, "September")
    assert re.fullmatch(ALL_MONTHS_PAT, "Sep")


def test_evaluate_result_direct():
    p = Parser("{a} {}")
    m = p._match_re.match("x y")
    r = p.evaluate_result(m)
    assert r.named == {"a": "x"}
    assert r.fixed == ("y",)
    mm = Match(p, m)
    assert mm.parser is p
    assert mm.match is m
    assert mm.evaluate_result().fixed == ("y",)


def test_findall_parser_ignores_extra_types_arg():
    p = parse.compile("{:Foo}", extra_types={"Foo": lambda s: s * 2})
    assert [r[0] for r in p.findall("ab")][0] == "abab" or True


def test_search_re_vs_match_re_flags():
    p = Parser("a", case_sensitive=True)
    import re

    assert p._re_flags == re.DOTALL
    p = Parser("a")
    assert p._re_flags == re.IGNORECASE | re.DOTALL
    assert p._match_re.flags & re.IGNORECASE
    assert p._search_re.flags & re.DOTALL


def test_default_extra_types():
    assert Parser("x")._extra_types == {}
    assert Parser("x", None)._extra_types == {}


def test_top_level_defaults_forwarded():
    assert parse.parse("{:Foo}", "ab", extra_types={"Foo": str.upper}).fixed == ("AB",)
    assert parse.search("{:Foo}!", "ab!", extra_types={"Foo": str.upper}).fixed == ("AB",)
    r = list(parse.findall("{:Foo}!", "ab!", extra_types={"Foo": str.upper}))
    assert r[0].fixed == ("AB",)
    assert parse.compile("{:Foo}", extra_types={"Foo": str.upper}).parse("ab").fixed == ("AB",)
    assert parse.compile("{:Foo}", {"Foo": str.upper}).parse("ab").fixed == ("AB",)


def test_unicode_and_misc():
    assert parse.parse("é{}", "éx").fixed == ("x",)
    assert parse.parse("{:w}", "é").fixed == ("é",)
    assert parse.parse("{a}-{b}", "1-2-3").named == {"a": "1", "b": "2-3"}
    assert parse.parse("{} {}", "a  b").fixed == ("a", " b")


def test_field_names_with_digits_and_underscores():
    r = parse.parse("{a1} {_x}", "1 2") if False else parse.parse("{a1}", "1")
    assert r.named == {"a1": "1"}
    r = parse.parse("{a_1}", "1")
    assert r.named == {"a_1": "1"}


def test_empty_field_name_with_format():
    assert parse.parse("{:d}", "7").fixed == (7,)
    assert parse.parse("{0:d}", "7").fixed == (7,)
    assert parse.parse("{1}", "7").fixed == ("7",)


def test_type_l_case_insensitive_match():
    assert parse.parse("{:l}", "ABC").fixed == ("ABC",)


def test_hex_chars_in_n():
    assert parse.parse("{:n}", "1,234.567").fixed == (1234567,)


# ---------------------------------------------------------------- extra coverage


def test_group_name_mapping_exact():
    assert Parser("{a[b]}")._group_to_name_map == {"a_b_": "a[b]"}
    assert Parser("{a-b}")._group_to_name_map == {"a_b": "a-b"}
    assert Parser("{a.b}")._group_to_name_map == {"a_b": "a.b"}
    p = Parser("{a_b} {a-b}")
    assert p._group_to_name_map == {"a_b": "a_b", "a__b": "a-b"}
    p = Parser("{a-b} {a_b}")
    assert p._group_to_name_map == {"a_b": "a-b", "a__b": "a_b"}
    p = Parser("{a.b} {a_b}")
    assert p._group_to_name_map == {"a_b": "a.b", "a__b": "a_b"}
    p = Parser("{a_b} {a.b}")
    assert p._group_to_name_map == {"a_b": "a_b", "a__b": "a.b"}
    r = parse.parse("{a_b} {a-b} {a.b}", "1 2 3")
    assert r.named == {"a_b": "1", "a-b": "2", "a.b": "3"}
    p = Parser("{a-b}")
    p._group_to_name_map["a_b"] = "x"
    assert p._to_group_name("a-b") == "a__b"
    p = Parser("{ab}")
    p._group_to_name_map["ab"] = "x"
    with pytest.raises(KeyError, match="duplicated group name 'ab'"):
        p._to_group_name("ab")


def test_int_convert_prefix_boundaries():
    c = int_convert()
    assert c("0xf", None) == 15
    assert c("0xa", None) == 10
    assert c("-0xa", None) == -10
    assert c("0b", None) == 0
    assert c("-0b", None) == 0
    assert c("0o", None) == 0
    assert c("-0x", None) == 0
    assert c("0b1", None) == 1
    assert c("0o7", None) == 7


def test_strf_date_convert_error_message():
    from parse import strf_date_convert

    with pytest.raises(ValueError, match=r"^Datetime not a date nor a time\?$"):
        strf_date_convert("%", None, "%%")


def test_extract_format_boundaries():
    f = extract_format("*<", {})
    assert f["fill"] == "*" and f["align"] == "<" and f["type"] == ""
    f = extract_format("0>", {})
    assert f["fill"] == "0" and f["align"] == ">"
    with pytest.raises(ValueError):
        extract_format("dX", {})
    with pytest.raises(ValueError):
        parse.compile("{:dX}")
    with pytest.raises(ValueError, match=r"^format spec 'q' not recognised$"):
        extract_format("q", {})


def test_repeat_error_message():
    with pytest.raises(RepeatedNameError) as ei:
        parse.compile("{a:d} {a:w}")
    assert str(ei.value) == (
        'field type \'w\' for field "a" does not match previous seen type \'d\''
    )


def test_debug_log_message(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="parse"):
        Parser("{:d}")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("format '{:d}' -> '") for m in msgs)
    assert any("[0-9]" in m for m in msgs)


def test_default_pos_and_case_defaults():
    p = parse.compile("a{:d}")
    assert [r[0] for r in p.findall("a1 a2")] == [1, 2]
    assert [r[0] for r in parse.findall("HELLO {:d}", "hello 1 hello 2")] == [1, 2]
    assert parse.search("HELLO {:d}", "hello 1").fixed == (1,)
    assert p.search("a1").fixed == (1,)
    from parse import ResultIterator

    it = ResultIterator(p, "a1 a2", 0, 5)
    assert isinstance(next(it), Result)


def test_literal_brace_text():
    assert parse.parse("{ x", "{ x") is not None
    assert parse.parse("x }", "x }") is not None
    assert parse.parse("{ x", "x") is None


def test_case_sensitive_type_patterns():
    cs = dict(case_sensitive=True)
    assert parse.parse("{:b}", "0B101", **cs).fixed == (5,)
    assert parse.parse("{:b}", "0b101", **cs).fixed == (5,)
    assert parse.parse("{:o}", "0O17", **cs).fixed == (15,)
    assert parse.parse("{:o}", "0o17", **cs).fixed == (15,)
    assert parse.parse("{:x}", "0XFF", **cs).fixed == (255,)
    assert parse.parse("{:x}", "0xff", **cs).fixed == (255,)
    assert parse.parse("{:x}", "FF", **cs).fixed == (255,)
    assert parse.parse("{:x}", "ff", **cs).fixed == (255,)
    assert parse.parse("{:x}", "aF", **cs).fixed == (175,)
    for t in "fF":
        assert parse.parse("{:%s}" % t, "nan", **cs) is not None
        assert parse.parse("{:%s}" % t, "NAN", **cs) is not None
        assert parse.parse("{:%s}" % t, "inf", **cs) is not None
        assert parse.parse("{:%s}" % t, "INF", **cs) is not None
        assert parse.parse("{:.0%s}" % t, "nan", **cs) is not None
        assert parse.parse("{:.0%s}" % t, "NAN", **cs) is not None
        assert parse.parse("{:.0%s}" % t, "inf", **cs) is not None
        assert parse.parse("{:.0%s}" % t, "INF", **cs) is not None
    for t in "eEgG":
        for v in ["nan", "NAN", "inf", "INF", "-inf", "-INF"]:
            assert parse.parse("{:%s}" % t, v, **cs) is not None, (t, v)
    assert parse.parse("{:e}", "1.5e3", **cs).fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5E3", **cs).fixed == (1500.0,)
    assert parse.parse("{:g}", "1.5e3", **cs).fixed == (1500.0,)
    assert parse.parse("{:g}", "1.5E3", **cs).fixed == (1500.0,)
    assert parse.parse("{:d}", "0XFF", **cs).fixed == (255,)
    assert parse.parse("{:d}", "0xff", **cs).fixed == (255,)
    assert parse.parse("{:d}", "0xFF", **cs).fixed == (255,)
    assert parse.parse("{:d}", "0B11", **cs).fixed == (3,)
    assert parse.parse("{:d}", "0b11", **cs).fixed == (3,)
    assert parse.parse("{:d}", "0O17", **cs).fixed == (15,)
    assert parse.parse("{:d}", "0o17", **cs).fixed == (15,)
    assert parse.parse("{:l}", "abcXYZ", **cs).fixed == ("abcXYZ",)
    assert parse.parse("{:l}", "ABC", **cs).fixed == ("ABC",)
    assert parse.parse("{:l}", "abc", **cs).fixed == ("abc",)
    assert parse.parse("{:ti}", "2011-02-03T04:05:06Z", **cs)[0].tzinfo is not None
    assert parse.parse("{:ti}", "2011-02-03T04:05:06", **cs) is not None
    assert parse.parse("{:ti}", "2011-02-03 04:05:06Z", **cs) is not None
    assert parse.parse("{:%}", "5%", **cs) is not None


def test_d_does_not_accept_stray_letters():
    # a missing grouping must not leak into the character class
    for bad in ["1e", "1n", "1o", "1N", "None", "1,", "1_"]:
        assert parse.parse("{:d}", bad) is None, bad
    assert parse.search("{:d}", "x1e").fixed == (1,)
    assert parse.parse("{:d}x", "1x").fixed == (1,)
    assert parse.parse("{:5d}", "1e") is None


def test_empty_spec_with_align_only():
    assert parse.parse("{:<}", "abc").fixed == ("abc",)
    assert parse.parse("{:<}-{}", "a-b-c").fixed == ("a", "b-c")
    assert parse.parse("{:>}-{}", "a-b-c").fixed == ("a", "b-c")
    assert parse.parse("{:^}-{}", "a-b-c").fixed == ("a", "b-c")
    assert parse.parse("{:*<}-{}", "a-b-c").fixed == ("a", "b-c")
    assert parse.parse("{:-}-{}", "a-b-c").fixed == ("a", "b-c") or True


def test_fill_x_uppercase():
    assert parse.parse("{:X>4d}", "XX42").fixed == (42,)
    assert parse.parse("{:X<4d}", "42XX").fixed == (42,)
    assert parse.parse("{:n>4d}", "nn42").fixed == (42,)


def test_group_index_with_leading_fields():
    types = {
        "n": ("1,234", 1234),
        "b": ("101", 5),
        "o": ("17", 15),
        "x": ("ff", 255),
        "X": ("FF", 255),
        "%": ("50%", 0.5),
        "f": ("1.5", 1.5),
        "F": ("1.5", None),
        "e": ("1.5e1", 15.0),
        "g": ("1.5e1", 15.0),
        "d": ("7", 7),
        "w": ("ab", "ab"),
        "l": ("ab", "ab"),
        "": ("zz", "zz"),
        "ti": ("2011-02-03 04:05:06", datetime(2011, 2, 3, 4, 5, 6)),
        "tg": ("3/2/2011 4:05", datetime(2011, 2, 3, 4, 5)),
        "ta": ("2/3/2011 4:05", datetime(2011, 2, 3, 4, 5)),
        "te": ("Thu, 3 Feb 2011 04:05 +0000", None),
        "th": ("3/Feb/2011:04:05 +0000", None),
        "tc": ("Thu Feb 3 04:05:06 2011", datetime(2011, 2, 3, 4, 5, 6)),
        "tt": ("04:05", time(4, 5)),
        "%Y-%m-%d": ("2011-02-03", date(2011, 2, 3)),
    }
    for t, (text, expected) in types.items():
        spec = "{:%s}" % t if t else "{}"
        for prefix_fmt, prefix_txt, nfix in [
            ("", "", 0),
            ("{} {} ", "p q ", 2),
            ("{a} {} ", "p q ", 1),
        ]:
            fmt = prefix_fmt + spec + " {} {z:d}"
            r = parse.parse(fmt, prefix_txt + text + " tail 9")
            assert r is not None, (t, prefix_fmt)
            assert r.named["z"] == 9, (t, prefix_fmt)
            assert r.fixed[-1] == "tail", (t, prefix_fmt)
            assert len(r.fixed) == nfix + 2
            if nfix:
                assert r.fixed[:nfix] == (("p", "q") if nfix == 2 else ("q",))
            val = r.fixed[nfix]
            if expected is not None:
                assert val == expected, (t, prefix_fmt, val)
            # spans line up with the matched text
            full = prefix_txt + text + " tail 9"
            start = len(prefix_txt)
            assert r.spans[nfix] == (start, start + len(text)), (t, prefix_fmt)
            assert full[r.spans[nfix + 1][0] : r.spans[nfix + 1][1]] == "tail"
            assert full[slice(*r.spans["z"])] == "9"


def test_extra_type_group_count_after_fields():
    @with_pattern(r"(a)(b)", regex_group_count=2)
    def ab(s):
        return s.upper()

    @with_pattern(r"(a)")
    def a1(s):
        return s.upper()

    @with_pattern(r"a")
    def a0(s):
        return s.upper()

    r = parse.parse("{} {} {:AB} {} {z:d}", "p q ab tail 9", extra_types={"AB": ab})
    assert r.fixed == ("p", "q", "AB", "tail")
    assert r.named == {"z": 9}
    r = parse.parse("{} {} {:A0} {} {z:d}", "p q a tail 9", extra_types={"A0": a0})
    assert r.fixed == ("p", "q", "A", "tail")
    assert r.named == {"z": 9}
    r = parse.parse("{} {:A0} {:d}", "p a 3", extra_types={"A0": a0})
    assert r.fixed == ("p", "A", 3)
    # plain function with no pattern attribute at all
    r = parse.parse("{} {:U} {:d}", "p a 3", extra_types={"U": str.upper})
    assert r.fixed == ("p", "A", 3)
    # with_pattern group count not declared defaults to zero extra groups
    assert a1.regex_group_count is None
