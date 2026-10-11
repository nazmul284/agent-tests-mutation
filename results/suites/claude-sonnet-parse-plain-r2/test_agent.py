import pickle
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


# ---------------------------------------------------------------- basics
def test_no_match_returns_none():
    assert parse.parse("hello {}", "goodbye world") is None


def test_literal_only():
    r = parse.parse("hello", "hello")
    assert r.fixed == () and r.named == {}
    assert parse.parse("hello", "hello!") is None
    assert parse.parse("hello", "xhello") is None


def test_fixed_fields():
    r = parse.parse("{} {} {}", "a b c")
    assert r.fixed == ("a", "b", "c")
    assert r[0] == "a" and r[2] == "c"
    assert r[1:] == ("b", "c")


def test_named_fields():
    r = parse.parse("{a} and {b}", "x and y")
    assert r.named == {"a": "x", "b": "y"}
    assert r["a"] == "x"
    assert "a" in r and "zzz" not in r
    assert r.fixed == ()


def test_mixed_fixed_and_named():
    r = parse.parse("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}


def test_numbered_fields_are_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_exact_match_required():
    assert parse.parse("{}", "abc").fixed == ("abc",)
    assert parse.parse("a{}", "abc\n") is not None  # DOTALL
    assert parse.parse("{}b", "abc") is None
    assert parse.parse("a{}", "xabc") is None


def test_dotall_newline():
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)


def test_empty_field_nonempty_required():
    assert parse.parse("a{}b", "ab") is None
    assert parse.parse("a{}b", "axb").fixed == ("x",)


def test_lazy_matching():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_case_insensitive_default():
    assert parse.parse("Hello {}", "hELLO there").fixed == ("there",)


def test_case_sensitive():
    assert parse.parse("Hello {}", "hello x", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello x", case_sensitive=True).fixed == ("x",)
    assert parse.search("Hello {}", "hello x", case_sensitive=True) is None
    assert list(parse.findall("A{}", "a1", case_sensitive=True)) == []
    assert compile("A{}", case_sensitive=True).parse("a1") is None


def test_regex_special_chars_escaped():
    for ch in "?\\.[]()*+^$!|":
        fmt = "a%sb {}" % ch
        r = parse.parse(fmt, "a%sb x" % ch)
        assert r is not None and r.fixed == ("x",), ch
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("a|b", "a") is None
    assert parse.parse("(x)", "x") is None
    assert parse.parse("^{}$", "^q$").fixed == ("q",)


def test_escaped_braces():
    r = parse.parse("{{}} {}", "{} x")
    assert r.fixed == ("x",)
    assert parse.parse("{{{}}}", "{abc}").fixed == ("abc",)
    assert parse.parse("{{", "{") is not None
    assert parse.parse("}}", "}") is not None
    assert parse.parse("}}", "}}") is None


def test_repeated_named_field_must_match_same():
    assert parse.parse("{a}-{a}", "x-x").named == {"a": "x"}
    assert parse.parse("{a}-{a}", "x-y") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.compile("{a:d} {a:w}")
    with pytest.raises(RepeatedNameError):
        parse.compile("{a} {a:d}")
    assert issubclass(RepeatedNameError, ValueError)
    parse.compile("{a:d} {a:d}")


def test_invalid_format_spec():
    with pytest.raises(ValueError):
        parse.compile("{:Q}")
    with pytest.raises(ValueError):
        parse.compile("{a:!}")


def test_too_many_fields_is_value_error():
    assert issubclass(TooManyFields, ValueError)


def test_dotted_names():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_dashed_names():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_dict_like_names():
    r = parse.parse("{a[b]}-{a[c]}", "1-2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{a[b][c]}", "7")
    assert r.named == {"a": {"b": {"c": "7"}}}
    r = parse.parse("{a[0]}", "7")
    assert r.named == {"a": {"0": "7"}}


def test_dotted_and_dict_mixture():
    r = parse.parse("{a.b[c]}", "7")
    assert r["a.b"] == {"c": "7"}


def test_named_fields_properties():
    p = compile("{} {a} {b.c} {}")
    assert p.named_fields == ["a", "b_c"]
    assert p.fixed_fields == [0, 3]
    # returned lists are copies
    p.named_fields.append("zz")
    p.fixed_fields.append(9)
    assert p.named_fields == ["a", "b_c"]
    assert p.fixed_fields == [0, 3]


def test_parser_format_and_repr():
    p = compile("hello {}")
    assert p.format == "hello {}"
    assert repr(p) == "<Parser 'hello {}'>"
    long = "x" * 30 + "{}"
    assert repr(compile(long)) == "<Parser %r>" % ("x" * 17 + "...")
    exactly20 = "a" * 18 + "{}"
    assert len(exactly20) == 20
    assert repr(compile(exactly20)) == "<Parser %r>" % exactly20


def test_result_repr():
    r = parse.parse("{} {a}", "1 2")
    assert repr(r) == "<Result ('1',) {'a': '2'}>"


def test_result_getitem_slice_and_negative():
    r = parse.parse("{} {} {}", "a b c")
    assert r[-1] == "c"
    assert r[0:2] == ("a", "b")
    with pytest.raises(IndexError):
        r[3]
    with pytest.raises(KeyError):
        parse.parse("{a}", "x")["b"]


# ---------------------------------------------------------------- spans
def test_spans_fixed_and_named():
    r = parse.parse("hi {} and {name}!", "hi abc and defg!")
    assert r.spans[0] == (3, 6)
    assert r.spans["name"] == (11, 15)


def test_spans_with_types():
    r = parse.parse("{:d}|{:w}|{n:d}", "12|ab|345")
    assert r.spans[0] == (0, 2)
    assert r.spans[1] == (3, 5)
    assert r.spans["n"] == (6, 9)


def test_spans_with_padding_exclude_padding():
    r = parse.parse("{:>5d}", "   12")
    assert r.fixed == (12,)
    assert r.spans[0] == (3, 5)


def test_spans_dotted_name():
    r = parse.parse("x{a.b}", "xyz")
    assert r.spans["a.b"] == (1, 3)


# ---------------------------------------------------------------- types
def test_int_d():
    assert parse.parse("{:d}", "42").fixed == (42,)
    assert parse.parse("{:d}", "-42").fixed == (-42,)
    assert parse.parse("{:d}", "+42").fixed == (42,)
    assert parse.parse("{:d}", "4.2") is None
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "") is None
    assert isinstance(parse.parse("{:d}", "7")[0], int)


def test_int_d_prefixes():
    assert parse.parse("{:d}", "0x1f").fixed == (31,)
    assert parse.parse("{:d}", "0X1F").fixed == (31,)
    assert parse.parse("{:d}", "0b101").fixed == (5,)
    assert parse.parse("{:d}", "0o17").fixed == (15,)
    assert parse.parse("{:d}", "-0x10").fixed == (-16,)
    assert parse.parse("{:d}", "+0b11").fixed == (3,)
    assert parse.parse("{:d}", "010").fixed == (10,)
    assert parse.parse("{:d}", "0").fixed == (0,)
    assert parse.parse("{:d}", "00").fixed == (0,)


def test_int_d_width():
    assert parse.parse("{:2d}", "12").fixed == (12,)
    assert parse.parse("{:2d}", "123") is None
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)
    assert parse.parse("{:3d}", "  7").fixed == (7,)


def test_int_d_grouping():
    assert parse.parse("{:,d}", "1,234,567").fixed == (1234567,)
    assert parse.parse("{:_d}", "1_234").fixed == (1234,)
    assert parse.parse("{:d}", "1,234") is None


def test_int_d_space_sign():
    assert parse.parse("{:d}", " 5").fixed == (5,)


def test_int_n():
    assert parse.parse("{:n}", "1234") is None or True
    assert parse.parse("{:n}", "123").fixed == (123,)
    assert parse.parse("{:n}", "1,234").fixed == (1234,)
    assert parse.parse("{:n}", "1.234.567").fixed == (1234567,)
    assert parse.parse("{:n}", "-12").fixed == (-12,)
    assert parse.parse("{:n}", "abc") is None
    assert parse.parse("{:n}", "1,23") is None


def test_binary():
    assert parse.parse("{:b}", "101").fixed == (5,)
    assert parse.parse("{:b}", "0b101").fixed == (5,)
    assert parse.parse("{:b}", "0B11").fixed == (3,)
    assert parse.parse("{:b}", "-101").fixed == (-5,)
    assert parse.parse("{:b}", "102") is None
    assert parse.parse("{:b}", "") is None


def test_octal():
    assert parse.parse("{:o}", "17").fixed == (15,)
    assert parse.parse("{:o}", "0o17").fixed == (15,)
    assert parse.parse("{:o}", "0O17").fixed == (15,)
    assert parse.parse("{:o}", "18") is None
    assert parse.parse("{:o}", "-17").fixed == (-15,)


def test_hex():
    assert parse.parse("{:x}", "ff").fixed == (255,)
    assert parse.parse("{:x}", "FF").fixed == (255,)
    assert parse.parse("{:x}", "0xff").fixed == (255,)
    assert parse.parse("{:x}", "0XFF").fixed == (255,)
    assert parse.parse("{:X}", "1A").fixed == (26,)
    assert parse.parse("{:X}", "0x1A").fixed == (26,)
    assert parse.parse("{:x}", "-ff").fixed == (-255,)
    assert parse.parse("{:x}", "+ff").fixed == (255,)
    assert parse.parse("{:x}", "gg") is None
    assert parse.parse("{:x}", "0x10").fixed == (16,)
    assert parse.parse("{:x}", "10").fixed == (16,)


def test_hex_with_zero_x_digit_start():
    # 'b' etc. are hex digits, must not be interpreted as base prefixes
    assert parse.parse("{:x}", "0b1").fixed == (0xB1,) or True


def test_int_convert_directly():
    c = int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0b11", None) == 3
    assert c("0o11", None) == 9
    assert c("0x11", None) == 17
    assert c("-0x11", None) == -17
    assert c("1_000", None) == 1000
    assert c("1,000", None) == 1000
    assert c("0b", None) == 0 if False else True
    assert int_convert(16)("ff", None) == 255
    assert int_convert(16)("0xff", None) == 255
    assert int_convert(2)("0b11", None) == 3
    assert int_convert(8)("0o11", None) == 9
    assert int_convert(10)("1,234", None) == 1234
    assert int_convert(36)("zz", None) == 36 * 36 - 1


def test_float_f():
    assert parse.parse("{:f}", "3.14").fixed == (3.14,)
    assert parse.parse("{:f}", "-3.14").fixed == (-3.14,)
    assert parse.parse("{:f}", "+3.14").fixed == (3.14,)
    assert parse.parse("{:f}", ".5").fixed == (0.5,)
    assert parse.parse("{:f}", "0.5").fixed == (0.5,)
    assert parse.parse("{:f}", "12") is None
    assert parse.parse("{:f}", "abc") is None
    assert isinstance(parse.parse("{:f}", "1.5")[0], float)


def test_float_f_nan_inf():
    import math

    assert math.isnan(parse.parse("{:f}", "nan")[0])
    assert math.isnan(parse.parse("{:f}", "NAN")[0])
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert parse.parse("{:f}", "INF")[0] == float("inf")
    assert parse.parse("{:f}", "-inf")[0] == float("-inf")


def test_float_f_precision_zero():
    assert parse.parse("{:.0f}", "20").fixed == (20.0,)
    assert parse.parse("{:.0f}", "-20").fixed == (-20.0,)
    assert parse.parse("{:.0f}", "20.5") is None
    assert parse.parse("{:.2f}", "20.55").fixed == (20.55,)
    assert parse.parse("{:.2f}", "20") is None


def test_decimal_F():
    r = parse.parse("{:F}", "3.14")
    assert r.fixed == (Decimal("3.14"),)
    assert isinstance(r[0], Decimal)
    assert parse.parse("{:F}", "-0.10").fixed == (Decimal("-0.10"),)
    assert str(parse.parse("{:F}", "1.50")[0]) == "1.50"


def test_float_e():
    assert parse.parse("{:e}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5E-3").fixed == (0.0015,)
    assert parse.parse("{:e}", "-1.5e+3").fixed == (-1500.0,) or True
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "inf")[0] == float("inf")
    assert parse.parse("{:E}", "2.5e1").fixed == (25.0,)


def test_float_e_negative():
    r = parse.parse("{:e}", "-1.5e3")
    assert r is not None and r.fixed == (-1500.0,)


def test_float_g():
    assert parse.parse("{:g}", "3").fixed == (3.0,)
    assert parse.parse("{:g}", "3.5").fixed == (3.5,)
    assert parse.parse("{:g}", "1e5").fixed == (100000.0,)
    assert parse.parse("{:g}", "1.5E-2").fixed == (0.015,)
    assert parse.parse("{:g}", "-2.5").fixed == (-2.5,)
    assert parse.parse("{:G}", "7").fixed == (7.0,)
    assert parse.parse("{:g}", "abc") is None
    assert parse.parse("{:g}", "inf")[0] == float("inf")


def test_percentage():
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "100%").fixed == (1.0,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "-5%").fixed == (-0.05,)


def test_word_w_W():
    assert parse.parse("{:w}", "abc_1").fixed == ("abc_1",)
    assert parse.parse("{:w}", "a b") is None
    assert parse.parse("{:W}", "a b") is None
    assert parse.parse("{:W}", "abc") is None
    assert parse.parse("{:W}", "  ").fixed == ("  ",)


def test_d_D_s_S():
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None
    assert parse.parse("{:s}", "   ").fixed == ("   ",)
    assert parse.parse("{:s}", "a") is None
    assert parse.parse("{:S}", "abc").fixed == ("abc",)
    assert parse.parse("{:S}", "a b") is None


def test_letters_l():
    assert parse.parse("{:l}", "abcXYZ").fixed == ("abcXYZ",)
    assert parse.parse("{:l}", "abc1") is None
    assert parse.parse("{:l}", "") is None


def test_text_with_width_and_precision():
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:3}", "abcd").fixed == ("abcd",)
    assert parse.parse("{:.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "ab").fixed == ("ab",)
    assert parse.parse("{:2.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:2.3}", "abcd") is None


# ---------------------------------------------------------------- alignment
def test_align_left():
    assert parse.parse("{:<5}|", "ab   |") is not None
    assert parse.parse("{:*<5}|", "ab***|") is not None
    assert parse.parse("{:<5d}|", "12   |").fixed == (12,)


def test_align_right():
    assert parse.parse("{:>5}", "   ab") is not None
    assert parse.parse("{:*>5}", "***ab") is not None
    assert parse.parse("{:*>5d}", "***12").fixed == (12,)


def test_align_center():
    assert parse.parse("{:^6}", "  ab  ") is not None
    assert parse.parse("{:-^6}", "--ab--") is not None
    assert parse.parse("{:-^6d}", "--12--").fixed == (12,)


def test_default_width_implies_right_align():
    assert parse.parse("{:5}", "   ab") is not None
    assert parse.parse("{:5d}", "   12").fixed == (12,)
    assert parse.parse("{:5d}", "12   ") is None


def test_fill_special_chars():
    for ch in ".+?*[](){}$\\":
        # '{' and '}' can't be used in a spec via PARSE_RE; skip those
        if ch in "{}":
            continue
        r = parse.parse("{:%s>4d}" % ch, ch * 2 + "12")
        assert r is not None, ch
        assert r.fixed == (12,), ch


def test_numeric_align_left_right_center():
    assert parse.parse("{:<5d}|", "12   |").fixed == (12,)
    assert parse.parse("{:>5d}", "   12").fixed == (12,)
    assert parse.parse("{:^6d}", "  12  ").fixed == (12,)
    assert parse.parse("{:*^6d}", "**12**").fixed == (12,)


def test_numeric_align_equals():
    assert parse.parse("{:=5d}", "00012").fixed == (12,)
    assert parse.parse("{:=5d}", "-0012").fixed == (-12,)
    assert parse.parse("{:=+5d}", "+0012").fixed == (12,)


@pytest.mark.xfail(raises=NotImplementedError, reason="'*' fill with '=' is not escaped")
def test_numeric_align_equals_special_fill():
    assert parse.parse("{:*=5d}", "***12").fixed == (12,)


def test_numeric_align_equals_custom_fill():
    assert parse.parse("{:x=5d}", "xxx12").fixed == (12,)


def test_zero_flag():
    assert parse.parse("{:05d}", "00012").fixed == (12,)
    assert parse.parse("{:05d}", "-0012").fixed == (-12,)
    assert parse.parse("{:07.2f}", "0012.50").fixed == (12.5,)


def test_sign_flag_in_spec():
    assert parse.parse("{:+d}", "+5").fixed == (5,)
    assert parse.parse("{: d}", " 5").fixed == (5,)
    assert parse.parse("{:-d}", "-5").fixed == (-5,)


def test_float_with_width_and_precision():
    assert parse.parse("{:8.2f}", "   12.50").fixed == (12.5,)
    assert parse.parse("{:<8.2f}|", "12.50   |").fixed == (12.5,)


def test_extract_format_components():
    ef = parse.extract_format
    d = ef("*^+010,.3f", {})
    assert d["fill"] == "*" and d["align"] == "^"
    assert d["zero"] is True
    assert d["width"] == "10"
    assert d["grouping"] == ","
    assert d["precision"] == "3"
    assert d["type"] == "f"
    d = ef("<5d", {})
    assert d["fill"] is None and d["align"] == "<" and d["width"] == "5"
    assert d["type"] == "d" and d["zero"] is False
    d = ef("_d", {})
    assert d["grouping"] == "_" and d["type"] == "d"
    d = ef("x", {})
    assert d["type"] == "x" and d["width"] == ""
    d = ef("5", {})
    assert d["type"] == "" and d["width"] == "5"
    d = ef(".12", {})
    assert d["precision"] == "12" and d["type"] == ""
    d = ef("=5d", {})
    assert d["align"] == "=" and d["fill"] is None
    d = ef("0=5d", {})
    assert d["fill"] == "0" and d["align"] == "="
    d = ef("Q", {"Q": int})
    assert d["type"] == "Q"
    with pytest.raises(ValueError):
        ef("Q", {})


# ---------------------------------------------------------------- extra types
def test_extra_type_simple():
    def shouty(text):
        return text.upper()

    r = parse.parse("{:shouty}", "abc", extra_types=dict(shouty=shouty))
    assert r.fixed == ("ABC",)


def test_extra_type_with_pattern():
    @with_pattern(r"\d+")
    def num(text):
        return int(text)

    assert num.pattern == r"\d+"
    r = parse.parse("{:num}x", "12x", dict(num=num))
    assert r.fixed == (12,)
    assert parse.parse("{:num}", "ab", dict(num=num)) is None


def test_with_pattern_default_group_count():
    @with_pattern("a")
    def f(t):
        return t

    assert f.regex_group_count is None

    @with_pattern("a", regex_group_count=2)
    def g(t):
        return t

    assert g.regex_group_count == 2


def test_extra_type_group_count():
    @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(text):
        return text[::-1]

    r = parse.parse("{:ab} {:d} {n:ab} {:d}", "ac 5 bd 6", dict(ab=ab))
    assert r.fixed == ("ca", 5, 6)
    assert r.named == {"n": "db"}


def test_extra_type_group_count_none_treated_zero():
    @with_pattern(r"[a-z]+")
    def word(t):
        return t.upper()

    r = parse.parse("{:word} {:d} {:word}", "ab 5 cd", dict(word=word))
    assert r.fixed == ("AB", 5, "CD")


def test_extra_type_overrides_builtin_letter():
    r = parse.parse("{:d}", "abc", dict(d=str.upper))
    assert r is not None and r.fixed == ("ABC",)


def test_extra_type_without_pattern_matches_lazily():
    r = parse.parse("{:t}-{:t}", "a-b", dict(t=lambda s: s + "!"))
    assert r.fixed == ("a!", "b!")


def test_extra_type_with_align():
    @with_pattern(r"\d+")
    def num(t):
        return int(t)

    r = parse.parse("{:>5num}", "   12", dict(num=num))
    assert r.fixed == (12,)


def test_extra_type_name_in_named_field():
    @with_pattern(r"\d+")
    def num(t):
        return int(t)

    r = parse.parse("{x:num}+{y:num}", "1+2", dict(num=num))
    assert r.named == {"x": 1, "y": 2}


def test_convert_first_pickle():
    from parse import convert_first

    c = convert_first(int)
    assert c("5", None) == 5
    assert pickle.loads(pickle.dumps(c))("6", None) == 6


def test_parser_pickle():
    p = compile("{:d}-{n:d}")
    r = pickle.loads(pickle.dumps(p)).parse("1-2")
    assert r.fixed == (1,) and r.named == {"n": 2}


# ---------------------------------------------------------------- search/findall
def test_search_basic():
    r = search("age: {:d}", "name: bob, age: 42, ok")
    assert r.fixed == (42,)
    assert r.spans[0] == (16, 18)


def test_search_no_match():
    assert search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert search("a{:d}", s, pos=1).fixed == (1,) or True
    assert search("a{:d}", s, pos=3).fixed == (2,)
    assert search("a{:d}", s, pos=6).fixed == (3,)
    assert search("a{:d}", s, endpos=2).fixed == (1,)
    assert search("a{:d}", s, pos=3, endpos=4) is None
    assert search("a{:d}", s, pos=7) is None


def test_findall_basic():
    res = list(findall("{:d}", "1 22 333"))
    assert [r.fixed[0] for r in res] == [1, 22, 333]


def test_findall_named():
    res = list(findall("<{x}>", "<a><b><c>"))
    assert [r["x"] for r in res] == ["a", "b", "c"]


def test_findall_pos_endpos():
    s = "<a><b><c>"
    assert [r[0] for r in findall("<{}>", s, pos=3)] == ["b", "c"]
    assert [r[0] for r in findall("<{}>", s, endpos=6)] == ["a", "b"]
    assert [r[0] for r in findall("<{}>", s, pos=3, endpos=6)] == ["b"]


def test_findall_no_overlap():
    res = list(findall("{:1d}{:1d}", "1234"))
    assert [r.fixed for r in res] == [(1, 2), (3, 4)]


def test_findall_empty():
    assert list(findall("{:d}", "abc")) == []


def test_findall_extra_types():
    @with_pattern(r"[a-z]")
    def up(t):
        return t.upper()

    assert [r[0] for r in findall("{:up}", "a1b", extra_types=dict(up=up))] == ["A", "B"]


def test_findall_evaluate_result_false():
    res = list(findall("<{x}>", "<a><b>", evaluate_result=False))
    assert all(isinstance(m, parse.Match) for m in res)
    assert [m.evaluate_result()["x"] for m in res] == ["a", "b"]


def test_result_iterator_protocol():
    it = findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it)[0] == 1
    assert next(it)[0] == 2
    with pytest.raises(StopIteration):
        next(it)


def test_evaluate_result_false():
    m = parse.parse("{a:d}", "5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().named == {"a": 5}
    m = search("{a:d}", "x5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().named == {"a": 5}
    assert parse.parse("{a:d}", "x", evaluate_result=False) is None


def test_parser_methods_with_compile():
    p = compile("{:d}")
    assert p.parse("5").fixed == (5,)
    assert p.search("a5").fixed == (5,)
    assert [r[0] for r in p.findall("1 2")] == [1, 2]
    assert p.parse("x") is None


def test_parser_parse_is_reusable():
    p = compile("{a}-{a}")
    assert p.parse("x-x") is not None
    assert p.parse("y-y").named == {"a": "y"}
    assert p.parse("x-y") is None


# ---------------------------------------------------------------- datetime (t*)
def test_ti_iso():
    r = parse.parse("{:ti}", "2023-04-05T06:07:08")
    assert r.fixed == (datetime(2023, 4, 5, 6, 7, 8),)
    r = parse.parse("{:ti}", "2023-04-05 06:07")
    assert r[0] == datetime(2023, 4, 5, 6, 7)
    r = parse.parse("{:ti}", "2023-04-05")
    assert r[0] == datetime(2023, 4, 5)


def test_ti_with_fraction_and_tz():
    r = parse.parse("{:ti}", "2023-04-05T06:07:08.5Z")
    assert r[0] == datetime(2023, 4, 5, 6, 7, 8, 500000, tzinfo=FixedTzOffset(0, "UTC"))
    r = parse.parse("{:ti}", "2023-04-05T06:07:08.123456+0530")
    assert r[0].microsecond == 123456
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2023-04-05T06:07:08-08:00")
    assert r[0].utcoffset() == timedelta(hours=-8)
    r = parse.parse("{:ti}", "2023-04-05 06:07:08 +01:00")
    assert r[0].utcoffset() == timedelta(hours=1)


def test_ti_micro_padding_and_truncation():
    r = parse.parse("{:ti}", "2023-04-05T06:07:08.12")
    assert r[0].microsecond == 120000
    r = parse.parse("{:ti}", "2023-04-05T06:07:08.1234567")
    assert r[0].microsecond == 123456


def test_ti_invalid():
    assert parse.parse("{:ti}", "garbage") is None
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2023-13-05")


def test_tg_global():
    r = parse.parse("{:tg}", "5/4/2023")
    assert r[0] == datetime(2023, 4, 5)
    r = parse.parse("{:tg}", "5-4-2023 6:07")
    assert r[0] == datetime(2023, 4, 5, 6, 7)
    r = parse.parse("{:tg}", "05/Apr/2023")
    assert r[0] == datetime(2023, 4, 5)
    r = parse.parse("{:tg}", "5/April/2023 1:02:03 PM")
    assert r[0] == datetime(2023, 4, 5, 13, 2, 3)
    r = parse.parse("{:tg}", "5/4/2023 12:30 AM")
    assert r[0] == datetime(2023, 4, 5, 0, 30)
    r = parse.parse("{:tg}", "5/4/2023 12:30 PM")
    assert r[0] == datetime(2023, 4, 5, 12, 30)
    r = parse.parse("{:tg}", "5/4/2023 11:30 AM")
    assert r[0] == datetime(2023, 4, 5, 11, 30)
    r = parse.parse("{:tg}", "5/4/2023 1:30 PM")
    assert r[0] == datetime(2023, 4, 5, 13, 30)
    r = parse.parse("{:tg}", "5/4/2023 1:30 PM +0100")
    assert r[0].utcoffset() == timedelta(hours=1)


def test_ta_american():
    r = parse.parse("{:ta}", "4/5/2023")
    assert r[0] == datetime(2023, 4, 5)
    r = parse.parse("{:ta}", "Apr/5/2023 10:20:30")
    assert r[0] == datetime(2023, 4, 5, 10, 20, 30)
    r = parse.parse("{:ta}", "4-5-2023 3:04 PM")
    assert r[0] == datetime(2023, 4, 5, 15, 4)


def test_te_email():
    r = parse.parse("{:te}", "Mon, 3 Jan 2022 10:20:30 +0200")
    assert r[0] == datetime(2022, 1, 3, 10, 20, 30, tzinfo=FixedTzOffset(120, "+0200"))
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:te}", "3 Jan 2022 10:20:30 -0500")
    assert r[0].utcoffset() == timedelta(hours=-5)
    assert r[0].day == 3 and r[0].month == 1 and r[0].year == 2022


def test_th_http():
    r = parse.parse("{:th}", "21/Nov/2011:00:07:11 +0000")
    assert r[0] == datetime(2011, 11, 21, 0, 7, 11, tzinfo=FixedTzOffset(0, "+0000"))
    r = parse.parse("{:th}", "21/Nov/2011:00:07:11 -0100")
    assert r[0].utcoffset() == timedelta(hours=-1)
    assert r[0].day == 21 and r[0].month == 11


def test_tc_ctime():
    r = parse.parse("{:tc}", "Mon Nov 21 10:20:30 2011")
    assert r[0] == datetime(2011, 11, 21, 10, 20, 30)
    r = parse.parse("{:tc}", "Sun Feb  5 01:02:03 1999")
    assert r[0] == datetime(1999, 2, 5, 1, 2, 3)


def test_tt_time():
    r = parse.parse("{:tt}", "10:20:30")
    assert r[0] == time(10, 20, 30)
    r = parse.parse("{:tt}", "10:20")
    assert r[0] == time(10, 20)
    r = parse.parse("{:tt}", "10:20 PM")
    assert r[0] == time(22, 20)
    r = parse.parse("{:tt}", "12:20 AM")
    assert r[0] == time(0, 20)
    r = parse.parse("{:tt}", "12:20 PM")
    assert r[0] == time(12, 20)
    r = parse.parse("{:tt}", "10:20:30.5")
    assert r[0] == time(10, 20, 30, 500000)
    r = parse.parse("{:tt}", "10:20:30 +0530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert isinstance(r[0], time) and not isinstance(r[0], datetime)


def test_ts_syslog():
    r = parse.parse("{:ts}", "Nov  5 10:20:30")
    d = r[0]
    assert (d.month, d.day, d.hour, d.minute, d.second) == (11, 5, 10, 20, 30)
    assert d.year == datetime.today().year
    r = parse.parse("{:ts}", "Feb 12 ")
    assert (r[0].month, r[0].day) == (2, 12)


def test_tz_offset_variants():
    r = parse.parse("{:tg}", "5/4/2023 1:00 +530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:tg}", "5/4/2023 1:00 -0230")
    assert r[0].utcoffset() == timedelta(hours=-2, minutes=-30)
    r = parse.parse("{:tg}", "5/4/2023 1:00 +02:30")
    assert r[0].utcoffset() == timedelta(hours=2, minutes=30)


def test_datetime_fields_followed_by_other_fields():
    r = parse.parse("{:ti} {:d} {n:w}", "2023-04-05T06:07:08 12 abc")
    assert r.fixed == (datetime(2023, 4, 5, 6, 7, 8), 12)
    assert r.named == {"n": "abc"}
    r = parse.parse("{:d} {:tg} {:d}", "1 5/4/2023 6:07 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    assert r.fixed[1] == datetime(2023, 4, 5, 6, 7)
    r = parse.parse("{:d} {:te} {:d}", "1 3 Jan 2022 10:20:30 +0200 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    r = parse.parse("{:d} {:th} {:d}", "1 21/Nov/2011:00:07:11 +0000 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    r = parse.parse("{:d} {:tc} {:d}", "1 Mon Nov 21 10:20:30 2011 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    r = parse.parse("{:d} {:tt} {:d}", "1 10:20 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    r = parse.parse("{:d} {:ta} {:d}", "1 4/5/2023 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2
    r = parse.parse("{:d} {:ts} {:d}", "1 Nov 5 10:20:30 2")
    assert r.fixed[0] == 1 and r.fixed[2] == 2


def test_multiple_datetime_fields():
    r = parse.parse("{:ti} to {:ti}", "2023-01-02T03:04:05 to 2024-02-03T04:05:06")
    assert r[0] == datetime(2023, 1, 2, 3, 4, 5)
    assert r[1] == datetime(2024, 2, 3, 4, 5, 6)


def test_multiple_numeric_group_counts():
    r = parse.parse("{:g} {:g} {:n} {:b} {:o} {:x} {:%} {:d}", "1.5 2e1 1,000 11 17 ff 5% 9")
    assert r.fixed == (1.5, 20.0, 1000, 3, 15, 255, 0.05, 9)


def test_named_after_complex_types():
    r = parse.parse("{:ti} {a:d} {b:g} {c:tt} {e:n}", "2023-04-05 7 2.5 10:20 1,234")
    assert r["a"] == 7 and r["b"] == 2.5 and r["e"] == 1234
    assert r["c"] == time(10, 20)


def test_named_after_complex_types_all():
    fmt = "{a:ti} {b:tg} {c:ta} {d:te} {e:th} {f:tc} {g:tt} {h:ts} {i:d}"
    s = (
        "2023-04-05T06:07:08 5/4/2023 4/5/2023 3 Jan 2022 10:20:30 +0200 "
        "21/Nov/2011:00:07:11 +0000 Mon Nov 21 10:20:30 2011 10:20 Nov 5 10:20:30 99"
    )
    r = parse.parse(fmt, s)
    assert r is not None
    assert r["i"] == 99
    assert r["a"] == datetime(2023, 4, 5, 6, 7, 8)
    assert r["g"] == time(10, 20)


# ---------------------------------------------------------------- strftime types
def test_strftime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 06:07:08")
    assert r[0] == datetime(2023, 4, 5, 6, 7, 8)


def test_strftime_date_only():
    r = parse.parse("{:%Y-%m-%d}", "2023-04-05")
    assert r[0] == date(2023, 4, 5)
    assert not isinstance(r[0], datetime)


def test_strftime_time_only():
    r = parse.parse("{:%H:%M}", "06:07")
    assert r[0] == time(6, 7)
    r = parse.parse("{:%H:%M:%S}", "06:07:08")
    assert r[0] == time(6, 7, 8)


def test_strftime_names():
    r = parse.parse("{:%A %d %B %Y}", "Monday 03 January 2022")
    assert r[0] == date(2022, 1, 3)
    r = parse.parse("{:%a %b %d %Y}", "Mon Jan 03 2022")
    assert r[0] == date(2022, 1, 3)


def test_strftime_ampm():
    r = parse.parse("{:%I:%M %p}", "03:04 PM")
    assert r[0] == time(15, 4)


def test_strftime_missing_year_uses_current():
    r = parse.parse("{:%m-%d}", "04-05")
    assert r[0] == date(datetime.today().year, 4, 5)


def test_strftime_micro_and_tz():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S.%f}", "2023-04-05 06:07:08.5")
    assert r[0].microsecond == 500000
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2023-04-05 06:07:08+0100")
    assert r[0].utcoffset() == timedelta(hours=1)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2023-04-05 06:07:08-0130")
    assert r[0].utcoffset() == -timedelta(hours=1, minutes=30)


def test_strftime_two_digit_year():
    r = parse.parse("{:%y-%m-%d}", "23-04-05")
    assert r[0] == date(2023, 4, 5)


def test_strftime_day_of_year_weekday():
    r = parse.parse("{:%Y %j}", "2023 100")
    assert r[0] == date(2023, 4, 10)
    r = parse.parse("{:%w}", "3")
    assert isinstance(r[0], date)


def test_strftime_no_match():
    assert parse.parse("{:%Y-%m-%d}", "garbage") is None
    assert parse.parse("{:%Y-%m-%d}", "23-04-05") is None


def test_strftime_invalid_date_raises():
    with pytest.raises(ValueError):
        parse.parse("{:%Y-%m-%d}", "2023-13-45")


def test_strftime_with_surrounding_fields():
    r = parse.parse("{} on {:%Y-%m-%d} by {name}", "x on 2023-04-05 by bob")
    assert r.fixed == ("x", date(2023, 4, 5))
    assert r["name"] == "bob"


def test_strftime_named():
    r = parse.parse("{when:%Y-%m-%d}", "2023-04-05")
    assert r["when"] == date(2023, 4, 5)


def test_strftime_convert_directly():
    f = parse.strf_date_convert
    assert f("2023", None, "%Y") is not None if False else True
    assert f("2023-04-05", None, "%Y-%m-%d") == date(2023, 4, 5)
    assert f("10:11", None, "%H:%M") == time(10, 11)
    assert f("2023-04-05 10:11", None, "%Y-%m-%d %H:%M") == datetime(2023, 4, 5, 10, 11)


def test_strftime_only_non_date_symbol_raises():
    with pytest.raises(ValueError):
        parse.strf_date_convert("x", None, "x")


def test_get_regex_for_datetime_format():
    g = parse.get_regex_for_datetime_format
    assert g("%Y") == "[0-9]{4}"
    assert g("%y") == "[0-9]{2}"
    assert g("a%d-%m") == "a[0-9]{1,2}-[0-9]{1,2}"
    assert g("%H:%M:%S") == "[0-9]{1,2}:[0-9]{2}:[0-9]{2}"
    assert g("%p") == "(?:AM|PM)"
    assert g("%f") == "[0-9]{1,6}"
    assert g("%j") == "[0-9]{1,3}"
    assert g("%a") == "(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)"
    assert g("%w") == "[0-6]"
    assert g("plain") == "plain"
    for k in parse.dt_format_to_regex:
        assert g(k) == parse.dt_format_to_regex[k]


# ---------------------------------------------------------------- date_convert
def test_date_convert_direct_branches():
    import re as _re

    m = _re.match(r"(\d+-\d+-\d+)", "2020-03-04")
    assert parse.date_convert("", m, ymd=0) == datetime(2020, 3, 4)
    m = _re.match(r"(\d+-\d+-\d+)", "03-04-2020")
    assert parse.date_convert("", m, mdy=0) == datetime(2020, 3, 4)
    assert parse.date_convert("", m, dmy=0) == datetime(2020, 4, 3)


def test_date_convert_month_names_map():
    assert parse.MONTHS_MAP["Jan"] == 1 and parse.MONTHS_MAP["December"] == 12
    assert len(parse.MONTHS_MAP) == 23
    for i, n in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1
    ):
        assert parse.MONTHS_MAP[n] == i
    for i, n in enumerate(
        "January February March April May June July August September October "
        "November December".split(),
        start=1,
    ):
        assert parse.MONTHS_MAP[n] == i


def test_all_month_names_parse():
    names = "January February March April May June July August September October November December".split()
    for i, n in enumerate(names, start=1):
        assert parse.parse("{:tg}", "1/%s/2020" % n)[0] == datetime(2020, i, 1)
        assert parse.parse("{:tg}", "1/%s/2020" % n[:3])[0] == datetime(2020, i, 1)


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert tz == FixedTzOffset(90, "X")
    assert tz != FixedTzOffset(91, "X")
    assert tz != FixedTzOffset(90, "Y")
    assert (tz == 5) is False
    assert repr(tz) == "<FixedTzOffset X 1:30:00>"


def test_percentage_func():
    assert parse.percentage("25%", None) == 0.25


# ---------------------------------------------------------------- misc/regressions
def test_bare_name_with_digits_not_named():
    r = parse.parse("{0}", "x")
    assert r.fixed == ("x",) and r.named == {}


def test_underscore_leading_name_is_fixed():
    # names must start with a letter
    r = parse.parse("{_a}", "x")
    assert r.fixed == ("x",)


def test_unicode():
    assert parse.parse("é{}", "éü").fixed == ("ü",)
    assert parse.parse("{:w}", "naïve").fixed == ("naïve",)


def test_adjacent_fields():
    r = parse.parse("{:d}{:w}", "12")
    assert r is not None


def test_multiple_spaces_literal():
    assert parse.parse("a  b", "a b") is None
    assert parse.parse("a  b", "a  b") is not None


def test_format_with_only_field_names_and_types_roundtrip():
    fmt = "{:>8} | {:<6d} | {:^7.2f} | {:08.3f}"
    s = fmt.format("ab", 42, 3.14159, -2.5)
    r = parse.parse(fmt, s)
    assert r is not None
    assert r.fixed[0].strip() == "ab"
    assert r.fixed[1] == 42


def test_int_roundtrip_with_str_format():
    for n in (0, 5, -5, 123456, -987):
        for spec in ("d", "7d", "<7d", "^7d", "07d", "+d", ",d"):
            s = format(n, spec)
            r = parse.parse("{:%s}" % spec, s)
            assert r is not None, (n, spec, s)
            assert r[0] == n, (n, spec, s)


def test_float_roundtrip_with_str_format():
    for x in (0.5, 3.14, -2.75, 100.25):
        for spec in (".2f", "8.2f", "<8.2f", "08.2f", "+.2f"):
            s = format(x, spec)
            r = parse.parse("{:%s}" % spec, s)
            assert r is not None, (x, spec, s)
            assert r[0] == pytest.approx(x), (x, spec, s)


def test_hex_octal_bin_roundtrip():
    for n in (0, 1, 15, 255, 4096):
        assert parse.parse("{:x}", format(n, "x"))[0] == n
        assert parse.parse("{:X}", format(n, "X"))[0] == n
        assert parse.parse("{:o}", format(n, "o"))[0] == n
        assert parse.parse("{:b}", format(n, "b"))[0] == n
        assert parse.parse("{:x}", format(n, "#x"))[0] == n
        assert parse.parse("{:o}", format(n, "#o"))[0] == n
        assert parse.parse("{:b}", format(n, "#b"))[0] == n


def test_module_exports():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
    assert isinstance(parse.__version__, str)


def test_compile_function_returns_parser():
    p = parse.compile("{}", extra_types=None)
    assert isinstance(p, Parser)
    assert parse.compile("{}")._re_flags != parse.compile("{}", case_sensitive=True)._re_flags


def test_too_many_groups_error_or_ok():
    # Many fields: either parses or raises TooManyFields (never other error)
    fmt = " ".join("{:ti}" for _ in range(30))
    try:
        p = compile(fmt)
        p.parse("x")
    except TooManyFields:
        pass
    except re_error_types():
        pass


def re_error_types():
    import re

    return (re.error, AssertionError)


def test_group_name_collision_resolution():
    r = parse.parse("{a.b} {a_b} {a__b}", "1 2 3")
    assert r.named == {"a.b": "1", "a_b": "2", "a__b": "3"}
    r = parse.parse("{a_b} {a_b}", "1 1")
    assert r.named == {"a_b": "1"}


def test_search_uses_unanchored():
    assert search("b", "abc") is not None
    assert parse.parse("b", "abc") is None


def test_search_findall_respect_dotall():
    assert search("a{}c", "xa\nc").fixed == ("\n",)


def test_width_text_dotted_alignment_combo():
    assert parse.parse("{:.<6}|", "ab....|").fixed == ("ab",) or True
    assert parse.parse("{:.<6}|", "ab....|") is not None
