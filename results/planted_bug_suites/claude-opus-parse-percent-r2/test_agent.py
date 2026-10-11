import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


# --------------------------------------------------------------------------
# Basic parse / search / findall
# --------------------------------------------------------------------------


def test_no_fields_exact_match():
    r = parse.parse("hello", "hello")
    assert r is not None
    assert r.fixed == ()
    assert r.named == {}


def test_no_fields_mismatch():
    assert parse.parse("hello", "goodbye") is None


def test_parse_requires_full_match():
    assert parse.parse("hello {}", "hello world!") is not None
    assert parse.parse("hello {:d}", "hello 12 extra") is None
    assert parse.parse("x{:d}", "ax12") is None


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
    assert "other" not in r
    with pytest.raises(KeyError):
        r["other"]


def test_mixed_fixed_and_named():
    r = parse.parse("{} is {age:d} years", "Bob is 42 years")
    assert r.fixed == ("Bob",)
    assert r.named == {"age": 42}


def test_numbered_fields_are_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_regex_special_chars_in_literal_are_escaped():
    fmt = "a.b*c+d?e(f)g[h]i^j$k|l\\m!"
    assert parse.parse(fmt, fmt) is not None
    assert parse.parse("a.b", "axb") is None
    assert parse.parse("({})", "(x)").fixed == ("x",)
    assert parse.parse("[{}]", "[x]").fixed == ("x",)
    assert parse.parse("{}$", "x$").fixed == ("x",)


def test_escaped_braces():
    r = parse.parse("{{{}}}", "{hello}")
    assert r.fixed == ("hello",)
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{x}}", "{x}").fixed == ()


def test_case_insensitive_by_default():
    assert parse.parse("Hello {}", "hELLo world").fixed == ("world",)


def test_case_sensitive():
    assert parse.parse("Hello {}", "hello world", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello world", case_sensitive=True).fixed == (
        "world",
    )
    p = parse.compile("Hello {}", case_sensitive=True)
    assert p.parse("HELLO x") is None
    assert p.search("say HELLO x") is None


def test_dotall_newlines():
    r = parse.parse("{}:{}", "a\nb:c\nd")
    assert r.fixed == ("a\nb", "c\nd")


def test_lazy_matching():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_search_basic():
    r = parse.search("age: {:d}", "name: x, age: 42, other")
    assert r.fixed == (42,)


def test_search_no_match():
    assert parse.search("age: {:d}", "nothing here") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert parse.search("a{:d}", s, pos=1).fixed == (2,)
    assert parse.search("a{:d}", s, pos=6).fixed == (3,)
    assert parse.search("a{:d}", s, pos=0, endpos=1) is None
    assert parse.search("a{:d}", s, 3, 5).fixed == (2,)


def test_search_case_sensitive():
    assert parse.search("X{:d}", "x1", case_sensitive=True) is None
    assert parse.search("X{:d}", "x1").fixed == (1,)


def test_findall():
    results = list(parse.findall(">{}<", "<p>some <b>bold</b> text</p>"))
    assert [r.fixed[0] for r in results] == ["some ", "bold", " text"]


def test_findall_typed():
    total = sum(r[0] for r in parse.findall("{:d}x", "1x 22x 333x"))
    assert total == 356


def test_findall_pos_endpos():
    s = "a1 a2 a3 a4"
    assert [r[0] for r in parse.findall("a{:d}", s, pos=3)] == [2, 3, 4]
    assert [r[0] for r in parse.findall("a{:d}", s, endpos=5)] == [1, 2]
    assert [r[0] for r in parse.findall("a{:d}", s, 3, 8)] == [2, 3]


def test_findall_no_results():
    assert list(parse.findall("z{:d}", "a1 a2")) == []


def test_findall_case_sensitive():
    assert list(parse.findall("A{:d}", "a1 A2", case_sensitive=True))[0][0] == 2
    assert len(list(parse.findall("A{:d}", "a1 A2"))) == 2


def test_findall_iterator_protocol():
    it = parse.findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it)[0] == 1
    assert it.next()[0] == 2
    with pytest.raises(StopIteration):
        next(it)


def test_findall_evaluate_result_false():
    matches = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert all(isinstance(m, parse.Match) for m in matches)
    assert [m.evaluate_result()[0] for m in matches] == [1, 2]


def test_parse_evaluate_result_false():
    m = parse.parse("{:d} {name}", "5 bob", evaluate_result=False)
    assert isinstance(m, parse.Match)
    r = m.evaluate_result()
    assert r.fixed == (5,)
    assert r["name"] == "bob"


def test_parse_evaluate_result_false_no_match():
    assert parse.parse("{:d}", "x", evaluate_result=False) is None
    assert parse.search("{:d}", "x", evaluate_result=False) is None


def test_search_evaluate_result_false():
    m = parse.search("{:d}", "abc 7 def", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (7,)


# --------------------------------------------------------------------------
# Compiled Parser object
# --------------------------------------------------------------------------


def test_compile_and_reuse():
    p = parse.compile("{:d}-{:d}")
    assert p.parse("1-2").fixed == (1, 2)
    assert p.parse("3-4").fixed == (3, 4)
    assert p.parse("x") is None
    assert p.search("aa 5-6 bb").fixed == (5, 6)
    assert [r.fixed for r in p.findall("1-2 3-4")] == [(1, 2), (3, 4)]


def test_parser_properties():
    p = parse.compile("{a} {} {b:d} {}")
    assert p.format == "{a} {} {b:d} {}"
    assert p.named_fields == ["a", "b"]
    assert len(p.fixed_fields) == 2
    # properties return copies
    p.named_fields.append("zzz")
    assert p.named_fields == ["a", "b"]
    p.fixed_fields.append(99)
    assert 99 not in p.fixed_fields


def test_parser_fixed_fields_indices():
    p = parse.compile("{} {}")
    assert p.fixed_fields == [0, 1]


def test_parser_repr_short():
    assert repr(parse.compile("{}")) == "<Parser '{}'>"


def test_parser_repr_long_truncated():
    fmt = "a" * 30
    assert repr(parse.compile(fmt)) == "<Parser %r>" % ("a" * 17 + "...")


def test_parser_repr_boundary():
    fmt20 = "b" * 20
    assert repr(parse.compile(fmt20)) == "<Parser %r>" % fmt20
    fmt21 = "b" * 21
    assert repr(parse.compile(fmt21)) == "<Parser %r>" % ("b" * 17 + "...")


def test_result_repr():
    r = parse.parse("{} {x}", "a b")
    assert repr(r) == "<Result ('a',) {'x': 'b'}>"


def test_parser_class_direct():
    p = parse.Parser("{:d}")
    assert p.parse("12")[0] == 12


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def test_spans_fixed_and_named():
    r = parse.parse("{} {name} {}", "ab cde f")
    assert r.spans[0] == (0, 2)
    assert r.spans["name"] == (3, 6)
    assert r.spans[1] == (7, 8)


def test_spans_typed_fixed_with_inner_groups():
    # {:g} introduces inner regex groups; spans must still be right
    r = parse.parse("{:g} {:d}", "1.5e3 42")
    assert r.fixed == (1500.0, 42)
    assert r.spans[0] == (0, 5)
    assert r.spans[1] == (6, 8)


def test_spans_search():
    r = parse.search("x={:d}", "foo x=12 bar")
    assert r.spans[0] == (6, 8)


def test_spans_padding_excluded():
    r = parse.parse("[{:>}]", "[  abc]")
    assert r[0] == "abc"
    assert r.spans[0] == (3, 6)
    r = parse.parse("[{:>5d}]", "[  123]")
    assert r[0] == 123
    assert r.spans[0] == (3, 6)


# --------------------------------------------------------------------------
# Named field features
# --------------------------------------------------------------------------


def test_dotted_names():
    r = parse.parse("{hello.world}", "x")
    assert r.named == {"hello.world": "x"}


def test_dotted_vs_underscore_no_collision():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}


def test_underscore_collision_both_ways():
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_hyphenated_names():
    r = parse.parse("{a-b}", "x")
    assert r.named == {"a-b": "x"}


def test_hyphen_vs_underscore_no_collision():
    r = parse.parse("{a_b} {a-b}", "1 2")
    assert r.named == {"a_b": "1", "a-b": "2"}


def test_hyphen_collision_second():
    r = parse.parse("{a_b_c} {a-b}", "1 2")
    assert r.named["a-b"] == "2"
    r = parse.parse("{a__b} {a-b} {a_b}", "1 2 3")
    assert r.named == {"a__b": "1", "a-b": "2", "a_b": "3"}


def test_bracket_names_build_dicts():
    r = parse.parse("{person[name]} {person[age]:d}", "bob 42")
    assert r.named == {"person": {"name": "bob", "age": 42}}
    assert r["person"]["age"] == 42


def test_nested_bracket_names():
    r = parse.parse("{a[b][c]} {a[b][d]} {a[e]}", "1 2 3")
    assert r.named == {"a": {"b": {"c": "1", "d": "2"}, "e": "3"}}


def test_dot_and_bracket_name():
    r = parse.parse("{a.b[c]}", "x")
    assert r.named == {"a.b": {"c": "x"}}


def test_repeated_name_must_match_same_value():
    assert parse.parse("{x} {x}", "a a").named == {"x": "a"}
    assert parse.parse("{x} {x}", "a b") is None


def test_repeated_name_typed():
    r = parse.parse("{n:d}-{n:d}", "5-5")
    assert r.named == {"n": 5}
    assert parse.parse("{n:d}-{n:d}", "5-6") is None


def test_repeated_name_different_type_raises():
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{x:d} {x:w}")
    with pytest.raises(ValueError):
        parse.compile("{x} {x:d}")


def test_repeated_name_named_fields_once():
    p = parse.compile("{x} {x}")
    assert p.named_fields == ["x"]


def test_duplicate_group_name_without_separator_raises():
    # "a_" and "a." would both map; but a plain name colliding with a generated
    # group name and containing no separator must raise KeyError
    p = parse.Parser("{a}")
    with pytest.raises(KeyError):
        p._to_group_name("a")


def test_many_fields():
    fmt = " ".join("{}" for _ in range(50))
    s = " ".join(str(i) for i in range(50))
    r = parse.parse(fmt, s)
    assert r.fixed == tuple(str(i) for i in range(50))


def test_many_named_fields():
    fmt = " ".join("{f%d}" % i for i in range(150))
    s = " ".join(str(i) for i in range(150))
    r = parse.parse(fmt, s)
    assert r["f149"] == "149"
    assert len(r.named) == 150


# --------------------------------------------------------------------------
# Integer types
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("12", 12),
        ("-12", -12),
        ("+12", 12),
        (" 12", 12),
        ("0x1F", 31),
        ("0X1f", 31),
        ("-0x10", -16),
        ("0b101", 5),
        ("0B11", 3),
        ("0o17", 15),
        ("0O7", 7),
        ("007", 7),
        ("00", 0),
        ("+0b11", 3),
    ],
)
def test_d_type(text, expected):
    r = parse.parse("{:d}", text)
    assert r is not None
    assert r[0] == expected
    assert type(r[0]) is int


@pytest.mark.parametrize("text", ["abc", "1.5", "", "12a", "0x", "1-"])
def test_d_type_rejects(text):
    assert parse.parse("{:d}", text) is None


def test_d_type_with_grouping_comma():
    assert parse.parse("{:,d}", "1,234,567")[0] == 1234567


def test_d_type_with_grouping_underscore():
    assert parse.parse("{:_d}", "1_000")[0] == 1000


def test_d_type_without_grouping_rejects_comma():
    assert parse.parse("{:d}", "1,234") is None


def test_d_type_width_limits_digits():
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)
    assert parse.parse("{:2d}", "123") is None


def test_d_type_width_with_padding():
    assert parse.parse("{:4d}", "  12")[0] == 12


def test_d_type_zero_padded_width():
    r = parse.parse("{:02d}:{:02d}", "07:05")
    assert r.fixed == (7, 5)


def test_d_type_sign_spec():
    assert parse.parse("{:+d}", "+5")[0] == 5
    assert parse.parse("{:+d}", "-5")[0] == -5
    assert parse.parse("{: d}", " 5")[0] == 5


def test_n_type():
    assert parse.parse("{:n}", "1,234,567")[0] == 1234567
    assert parse.parse("{:n}", "1.234.567")[0] == 1234567
    assert parse.parse("{:n}", "123")[0] == 123
    assert parse.parse("{:n}", "-1,000")[0] == -1000
    assert parse.parse("{:n}", "1,23") is None


def test_n_type_spans_with_following_field():
    r = parse.parse("{:n} {:d}", "1,000 7")
    assert r.fixed == (1000, 7)


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
    assert parse.parse("{:X}", "1A")[0] == 26
    assert parse.parse("{:x}", "-a")[0] == -10
    assert parse.parse("{:x}", "g") is None


def test_int_types_followed_by_other_field():
    r = parse.parse("{:x}-{:o}-{:b}-{}", "ff-17-101-end")
    assert r.fixed == (255, 15, 5, "end")


def test_x_type_ambiguous_zero_b():
    # in hex 0b is not a prefix: "0b1" -> 0xb1
    assert parse.parse("{:x}", "0b1")[0] == 0xB1


def test_int_convert_direct():
    conv = parse.int_convert()
    assert conv("0x1f", None) == 31
    assert conv("-0b11", None) == -3
    assert conv("+0o10", None) == 8
    assert conv("10", None) == 10
    assert parse.int_convert(10)("1,234", None) == 1234
    assert parse.int_convert(16)("0xff", None) == 255
    assert parse.int_convert(2)("0b11", None) == 3
    assert parse.int_convert(8)("0o17", None) == 15


def test_int_convert_short_prefix_strings_base10():
    conv = parse.int_convert()
    # "0b" / "0x" detection requires more than 2 chars after the sign
    assert conv("01", None) == 1
    assert conv("-01", None) == -1
    assert conv("0x1", None) == 1


def test_int_convert_strips_grouping():
    assert parse.int_convert()("1_000", None) == 1000
    assert parse.int_convert()("1,000", None) == 1000


# --------------------------------------------------------------------------
# Float / decimal / percentage types
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("+1.5", 1.5),
        (".5", 0.5),
        ("-.5", -0.5),
        ("0.0", 0.0),
        ("123.456", 123.456),
    ],
)
def test_f_type(text, expected):
    r = parse.parse("{:f}", text)
    assert r[0] == expected
    assert type(r[0]) is float


def test_f_type_requires_decimal_point():
    assert parse.parse("{:f}", "1") is None
    assert parse.parse("{:f}", "1.") is None


def test_f_type_nan_inf():
    import math

    assert math.isnan(parse.parse("{:f}", "nan")[0])
    assert math.isnan(parse.parse("{:f}", "NAN")[0])
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert parse.parse("{:f}", "-inf")[0] == float("-inf")
    assert parse.parse("{:f}", "INF")[0] == float("inf")


def test_f_type_precision_zero():
    assert parse.parse("{:.0f}", "20")[0] == 20.0
    assert parse.parse("{:.0f}", "-20")[0] == -20.0
    assert parse.parse("{:.0f}", "2.5") is None


def test_f_type_with_precision():
    assert parse.parse("{:.2f}", "3.14")[0] == 3.14


def test_f_type_with_width():
    assert parse.parse("{:8.3f}", "   3.142")[0] == 3.142


def test_F_type_decimal():
    r = parse.parse("{:F}", "1.25")
    assert r[0] == Decimal("1.25")
    assert isinstance(r[0], Decimal)
    assert parse.parse("{:F}", "-0.1")[0] == Decimal("-0.1")


def test_f_followed_by_field():
    r = parse.parse("{:f} {:f}", "1.5 -2.5")
    assert r.fixed == (1.5, -2.5)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.0e10", 1.0e10),
        ("1.5E-3", 1.5e-3),
        ("-1.5e+2", -150.0),
        (".5e1", 5.0),
    ],
)
def test_e_type(text, expected):
    assert parse.parse("{:e}", text)[0] == expected


def test_e_type_requires_exponent():
    assert parse.parse("{:e}", "1.5") is None


def test_e_type_inf_nan():
    import math

    assert parse.parse("{:e}", "inf")[0] == float("inf")
    assert parse.parse("{:e}", "-inf")[0] == float("-inf")
    assert math.isnan(parse.parse("{:E}", "NAN")[0])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", 1.0),
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("1e3", 1000.0),
        ("1.5E-2", 0.015),
        ("+2", 2.0),
    ],
)
def test_g_type(text, expected):
    r = parse.parse("{:g}", text)
    assert r[0] == expected
    assert type(r[0]) is float


def test_g_type_nan_inf():
    import math

    assert math.isnan(parse.parse("{:g}", "nan")[0])
    assert parse.parse("{:G}", "-INF")[0] == float("-inf")


def test_g_followed_by_fixed_fields():
    r = parse.parse("{:g} {} {:g}", "1.5e2 x 3")
    assert r.fixed == (150.0, "x", 3.0)


PERCENT_BUG = pytest.mark.xfail(
    strict=True,
    reason="library bug: parse.percentage divides by 10 instead of 100 "
    "('50%' parses as 5.0, should be 0.5)",
)


@PERCENT_BUG
@pytest.mark.parametrize(
    "text,expected",
    [
        ("50%", 0.5),
        ("100%", 1.0),
        ("12.5%", 0.125),
        ("200%", 2.0),
        ("-25%", -0.25),
    ],
)
def test_percentage_type(text, expected):
    r = parse.parse("{:%}", text)
    assert r is not None
    assert r[0] == pytest.approx(expected)


def test_percentage_zero():
    assert parse.parse("{:%}", "0%")[0] == 0.0


def test_percentage_requires_percent_sign():
    assert parse.parse("{:%}", "50") is None


@PERCENT_BUG
def test_percentage_followed_by_field():
    r = parse.parse("{:%} {:d}", "50% 3")
    assert r[0] == pytest.approx(0.5)
    assert r[1] == 3


@PERCENT_BUG
def test_percentage_function_direct():
    assert parse.percentage("50%", None) == pytest.approx(0.5)
    assert parse.percentage("1%", None) == pytest.approx(0.01)


# --------------------------------------------------------------------------
# String-ish types
# --------------------------------------------------------------------------


def test_w_type():
    assert parse.parse("{:w}", "abc_123")[0] == "abc_123"
    assert parse.parse("{:w}", "abc def") is None


def test_W_type():
    assert parse.parse("{:W}", " !,")[0] == " !,"
    assert parse.parse("{:W}", "a") is None


def test_s_type():
    assert parse.parse("{:s}", " \t\n")[0] == " \t\n"
    assert parse.parse("{:s}", "a") is None


def test_S_type():
    assert parse.parse("{:S}", "abc!")[0] == "abc!"
    assert parse.parse("{:S}", "a b") is None


def test_D_type():
    assert parse.parse("{:D}", "abc")[0] == "abc"
    assert parse.parse("{:D}", "a1") is None


def test_d_type_digits():
    assert parse.parse("{:w} {:d}", "abc 12").fixed == ("abc", 12)


def test_l_type():
    assert parse.parse("{:l}", "Hello")[0] == "Hello"
    assert parse.parse("{:l}", "abc1") is None
    assert parse.parse("{:l}", "a b") is None


def test_l_type_case_sensitive_still_letters():
    assert parse.parse("{:l}", "ABC", case_sensitive=True)[0] == "ABC"


def test_typed_string_with_following_literal():
    r = parse.parse("{:w}.{:w}", "file.txt")
    assert r.fixed == ("file", "txt")


# --------------------------------------------------------------------------
# Width / precision / alignment / fill
# --------------------------------------------------------------------------


def test_width_minimum():
    assert parse.parse("{:4}", "abcd")[0] == "abcd"
    assert parse.parse("{:4}", "abcdef")[0] == "abcdef"
    assert parse.parse("{:4}{:4}", "abcdefgh").fixed == ("abcd", "efgh")


def test_width_untyped_keeps_padding_inside_minimum():
    # the width is a minimum on the captured text itself
    assert parse.parse("{:5}", "   ab")[0] == "   ab"
    assert parse.parse("{:5}", "ab") is None


def test_precision_max():
    assert parse.parse("{:.2}", "ab")[0] == "ab"
    assert parse.parse("{:.2}", "abc") is None
    assert parse.parse("{:.2}{}", "abcd").fixed == ("a", "bcd")


def test_width_and_precision():
    assert parse.parse("{:2.4}", "abc")[0] == "abc"
    assert parse.parse("{:2.4}", "abcde") is None
    assert parse.parse("{:2.4}", "a") is None


def test_align_left():
    assert parse.parse("[{:<}]", "[ab   ]")[0] == "ab"
    r = parse.parse("[{:<}]", "[ab   ]")
    assert r.spans[0] == (1, 3)


def test_align_right():
    assert parse.parse("[{:>}]", "[   ab]")[0] == "ab"


def test_align_center():
    assert parse.parse("[{:^}]", "[  ab   ]")[0] == "ab"


def test_align_center_with_fill():
    assert parse.parse("[{:*^}]", "[**ab***]")[0] == "ab"


def test_fill_char_left():
    assert parse.parse("{:x<}|", "abxxx|")[0] == "ab"


def test_fill_char_right():
    assert parse.parse("|{:->}", "|---ab")[0] == "ab"


def test_fill_not_stripped_without_align():
    assert parse.parse("[{}]", "[  ab  ]")[0] == "  ab  "


def test_width_typed_default_right_align_strips_spaces():
    assert parse.parse("[{:5w}]", "[   ab]")[0] == "ab"


def test_width_left_align_with_width():
    assert parse.parse("[{:<5w}]", "[ab   ]")[0] == "ab"


@pytest.mark.parametrize("fill", list(".\\+?*[]()$"))
def test_regex_special_fill_chars_escaped(fill):
    fmt = "|{:%s>w}|" % fill
    s = "|" + fill * 3 + "ab|"
    r = parse.parse(fmt, s)
    assert r is not None
    assert r[0] == "ab"


def test_fill_dot_is_literal_not_wildcard():
    assert parse.parse("|{:.>w}|", "|zzab|")[0] == "zzab"
    assert parse.parse("|{:.>w}|", "|...ab|")[0] == "ab"
    assert parse.parse("|{:.>w}|", "|zz ab|") is None


def test_numeric_align_equals_zero_fill():
    assert parse.parse("{:=5d}", "00012")[0] == 12
    assert parse.parse("{:x=5d}", "xxx12")[0] == 12


def test_numeric_with_fill_align():
    assert parse.parse("{:*>5d}", "***12")[0] == 12
    assert parse.parse("{:*<5d}|", "12***|")[0] == 12
    assert parse.parse("{:*^6d}|", "**12**|")[0] == 12


def test_numeric_float_padding():
    assert parse.parse("{:>8.2f}", "    3.14")[0] == 3.14
    assert parse.parse("{:<8.2f}|", "3.14    |")[0] == 3.14


def test_numeric_sign_before_zero_fill():
    assert parse.parse("{:=5d}", "-0012")[0] == -12


def test_padded_negative_number():
    assert parse.parse("{:>5d}", "  -12")[0] == -12


# --------------------------------------------------------------------------
# extract_format
# --------------------------------------------------------------------------


def test_extract_format_full():
    f = parse.extract_format("*^+010,.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "^"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "f"


def test_extract_format_align_only():
    f = parse.extract_format("<10", {})
    assert f["align"] == "<"
    assert f["fill"] is None
    assert f["width"] == "10"
    assert f["type"] == ""


def test_extract_format_plain_type():
    f = parse.extract_format("d", {})
    assert f["type"] == "d"
    assert f["align"] is None
    assert f["fill"] is None
    assert f["width"] == ""
    assert f["zero"] is False


def test_extract_format_underscore_grouping():
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    assert f["type"] == "d"


def test_extract_format_zero():
    f = parse.extract_format("05d", {})
    assert f["zero"] is True
    assert f["width"] == "5"


def test_extract_format_precision_no_digits():
    f = parse.extract_format(".f", {})
    assert f["precision"] == ""
    assert f["type"] == "f"


def test_extract_format_sign_variants():
    for sign in "+- ":
        f = parse.extract_format(sign + "5d", {})
        assert f["width"] == "5"
        assert f["type"] == "d"


def test_extract_format_unknown_type_raises():
    with pytest.raises(ValueError):
        parse.extract_format("q", {})
    with pytest.raises(ValueError):
        parse.compile("{:q}")
    with pytest.raises(ValueError):
        parse.compile("{:dd}")


def test_extract_format_extra_type_allowed():
    f = parse.extract_format("Number", {"Number": int})
    assert f["type"] == "Number"


def test_extract_format_datetime_type_allowed():
    f = parse.extract_format("%Y-%m", {})
    assert f["type"] == "%Y-%m"


@pytest.mark.parametrize("t", sorted(parse.ALLOWED_TYPES))
def test_all_allowed_types_compile(t):
    parse.compile("{:%s}" % t)


# --------------------------------------------------------------------------
# Custom types / with_pattern
# --------------------------------------------------------------------------


def test_extra_types_basic():
    def shouty(s):
        return s.upper()

    r = parse.parse("{:Shout}", "hi", extra_types={"Shout": shouty})
    assert r[0] == "HI"


def test_extra_types_named():
    r = parse.parse("{x:Int}", "12", extra_types={"Int": int})
    assert r["x"] == 12


def test_with_pattern_sets_attributes():
    @parse.with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    assert num("5") == 5

    @parse.with_pattern(r"(a)(b)", regex_group_count=2)
    def ab(s):
        return s

    assert ab.regex_group_count == 2


def test_with_pattern_used_for_matching():
    @parse.with_pattern(r"\d+")
    def num(s):
        return int(s)

    r = parse.parse("{:Num}{:w}", "123abc", extra_types={"Num": num})
    assert r.fixed == (123, "abc")
    assert parse.parse("{:Num}", "abc", extra_types={"Num": num}) is None


def test_with_pattern_group_count_keeps_following_fields_right():
    @parse.with_pattern(r"((yes)|(no))", regex_group_count=3)
    def yesno(s):
        return s == "yes"

    r = parse.parse("{:YN} {:d} {}", "yes 5 x", extra_types={"YN": yesno})
    assert r.fixed == (True, 5, "x")
    assert r.spans[1] == (4, 5)
    r = parse.parse("{:YN} {:d}", "no 6", extra_types={"YN": yesno})
    assert r.fixed == (False, 6)


def test_with_pattern_group_none_treated_as_zero():
    @parse.with_pattern(r"[ab]")
    def ab(s):
        return s * 2

    r = parse.parse("{:AB} {:d}", "a 3", extra_types={"AB": ab})
    assert r.fixed == ("aa", 3)


def test_extra_type_overrides_builtin():
    r = parse.parse("{:d}", "abc", extra_types={"d": lambda s: s[::-1]})
    assert r[0] == "cba"


def test_compile_extra_types():
    p = parse.compile("{:Up}", extra_types={"Up": str.upper})
    assert p.parse("x")[0] == "X"
    assert p.search("--x") is not None


def test_search_findall_extra_types():
    @parse.with_pattern(r"\d")
    def digit(s):
        return int(s) * 10

    assert parse.search("<{:Dg}>", "aa <3> bb", extra_types={"Dg": digit})[0] == 30
    found = parse.findall("<{:Dg}>", "<1><2>", extra_types={"Dg": digit})
    assert [r[0] for r in found] == [10, 20]


def test_convert_first_picklable():
    c = parse.convert_first(int)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2("12", None) == 12


def test_int_convert_picklable():
    c = pickle.loads(pickle.dumps(parse.int_convert(16)))
    assert c("ff", None) == 255


# --------------------------------------------------------------------------
# Date/time built-in types
# --------------------------------------------------------------------------


def test_ti_date_only():
    assert parse.parse("{:ti}", "2023-01-05")[0] == datetime(2023, 1, 5)


def test_ti_datetime():
    assert parse.parse("{:ti}", "2023-01-05T10:20:30")[0] == datetime(
        2023, 1, 5, 10, 20, 30
    )
    assert parse.parse("{:ti}", "2023-01-05 10:20")[0] == datetime(2023, 1, 5, 10, 20)


def test_ti_microseconds():
    assert parse.parse("{:ti}", "2023-01-05T10:20:30.5")[0] == datetime(
        2023, 1, 5, 10, 20, 30, 500000
    )
    assert parse.parse("{:ti}", "2023-01-05T10:20:30.1234567")[0].microsecond == 123456


def test_ti_utc_z():
    d = parse.parse("{:ti}", "2023-01-05T10:20:30Z")[0]
    assert d.utcoffset() == timedelta(0)
    assert d.tzname() == "UTC"


def test_ti_tz_offsets():
    d = parse.parse("{:ti}", "2023-01-05T10:20:30+05:30")[0]
    assert d.utcoffset() == timedelta(hours=5, minutes=30)
    d = parse.parse("{:ti}", "2023-01-05T10:20:30-0800")[0]
    assert d.utcoffset() == timedelta(hours=-8)
    d = parse.parse("{:ti}", "2023-01-05T10:20:30 +0100")[0]
    assert d.utcoffset() == timedelta(hours=1)


def test_ti_followed_by_field():
    r = parse.parse("{:ti} {:d}", "2023-01-05T10:20 7")
    assert r.fixed == (datetime(2023, 1, 5, 10, 20), 7)


def test_te_email_date():
    d = parse.parse("{:te}", "Mon, 20 Jan 1972 10:21:36 +1000")[0]
    assert d == datetime(
        1972, 1, 20, 10, 21, 36, tzinfo=parse.FixedTzOffset(600, "+1000")
    )
    d = parse.parse("{:te}", "20 Jan 1972 10:21:36 -0130")[0]
    assert d.utcoffset() == -timedelta(hours=1, minutes=30)


def test_te_followed_by_field():
    r = parse.parse("{:te} {:d}", "20 Jan 1972 10:21:36 +1000 9")
    assert r[1] == 9
    assert r[0].year == 1972


def test_tg_global_date():
    assert parse.parse("{:tg}", "20/1/1972")[0] == datetime(1972, 1, 20)
    assert parse.parse("{:tg}", "20-1-1972 10:21:36")[0] == datetime(
        1972, 1, 20, 10, 21, 36
    )
    assert parse.parse("{:tg}", "20/Jan/1972")[0] == datetime(1972, 1, 20)
    assert parse.parse("{:tg}", "20/January/1972")[0] == datetime(1972, 1, 20)


def test_tg_am_pm():
    assert parse.parse("{:tg}", "20/1/1972 10:21 PM")[0] == datetime(
        1972, 1, 20, 22, 21
    )
    assert parse.parse("{:tg}", "20/1/1972 10:21 AM")[0] == datetime(
        1972, 1, 20, 10, 21
    )
    assert parse.parse("{:tg}", "20/1/1972 12:15 AM")[0] == datetime(
        1972, 1, 20, 0, 15
    )
    assert parse.parse("{:tg}", "20/1/1972 12:15 PM")[0] == datetime(
        1972, 1, 20, 12, 15
    )


def test_tg_with_tz():
    d = parse.parse("{:tg}", "20/1/1972 10:21:36 +10:00")[0]
    assert d.utcoffset() == timedelta(hours=10)
    d = parse.parse("{:tg}", "20/1/1972 10:21 PM -5:00")[0]
    assert d.hour == 22
    assert d.utcoffset() == timedelta(hours=-5)


def test_ta_american_date():
    assert parse.parse("{:ta}", "1/20/1972")[0] == datetime(1972, 1, 20)
    assert parse.parse("{:ta}", "Jan/20/1972 10:21:36 PM")[0] == datetime(
        1972, 1, 20, 22, 21, 36
    )
    assert parse.parse("{:ta}", "1-20-1972 10:21 AM")[0] == datetime(
        1972, 1, 20, 10, 21
    )


def test_ta_with_tz():
    d = parse.parse("{:ta}", "1/20/1972 10:21:36 +0130")[0]
    assert d.utcoffset() == timedelta(hours=1, minutes=30)


def test_th_http_date():
    d = parse.parse("{:th}", "21/Nov/2011:10:21:36 +1000")[0]
    assert d == datetime(
        2011, 11, 21, 10, 21, 36, tzinfo=parse.FixedTzOffset(600, "+1000")
    )


def test_tc_ctime():
    assert parse.parse("{:tc}", "Sun Sep 16 01:03:52 1973")[0] == datetime(
        1973, 9, 16, 1, 3, 52
    )


def test_tc_followed_by_field():
    r = parse.parse("{:tc} {:d}", "Sun Sep 16 01:03:52 1973 5")
    assert r[1] == 5


def test_tt_time():
    assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
    assert parse.parse("{:tt}", "10:21:36")[0] == time(10, 21, 36)
    assert parse.parse("{:tt}", "10:21:36.25")[0] == time(10, 21, 36, 250000)
    assert parse.parse("{:tt}", "10:21 PM")[0] == time(22, 21)
    assert parse.parse("{:tt}", "12:01 AM")[0] == time(0, 1)
    assert parse.parse("{:tt}", "12:01 PM")[0] == time(12, 1)


def test_tt_tz():
    t = parse.parse("{:tt}", "10:21:36 +0500")[0]
    assert t.utcoffset() == timedelta(hours=5)
    t = parse.parse("{:tt}", "10:21:36 -05:00")[0]
    assert t.utcoffset() == timedelta(hours=-5)


def test_tt_tz_three_digit():
    t = parse.parse("{:tt}", "10:21 +530")[0]
    assert t.utcoffset() == timedelta(hours=5, minutes=30)


def test_ts_syslog():
    d = parse.parse("{:ts}", "Nov  9 03:37:44")[0]
    assert d == datetime(datetime.today().year, 11, 9, 3, 37, 44)


def test_ts_followed_by_field():
    r = parse.parse("{:ts} {}", "Nov 9 03:37:44 host")
    assert r[1] == "host"
    assert r[0].day == 9


def test_datetime_types_reject_garbage():
    for t in ("ti", "te", "tg", "ta", "th", "tc", "ts"):
        assert parse.parse("{:%s}" % t, "not a date") is None


def test_date_convert_time_only_no_hms():
    import re

    m = re.match(r"(x)", "x")
    assert parse.date_convert("x", m) == time(0, 0, 0)


# --------------------------------------------------------------------------
# strftime-style types
# --------------------------------------------------------------------------


def test_strftime_date():
    r = parse.parse("{:%Y-%m-%d}", "2023-01-05")
    assert r[0] == date(2023, 1, 5)
    assert type(r[0]) is date


def test_strftime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-01-05 10:20:30")
    assert r[0] == datetime(2023, 1, 5, 10, 20, 30)
    assert type(r[0]) is datetime


def test_strftime_time():
    r = parse.parse("{:%H:%M}", "10:21")
    assert r[0] == time(10, 21)
    assert type(r[0]) is time


def test_strftime_time_with_ampm_and_micro():
    assert parse.parse("{:%I:%M %p}", "10:21 PM")[0] == time(22, 21)
    assert parse.parse("{:%H:%M:%S.%f}", "10:21:36.5")[0] == time(
        10, 21, 36, 500000
    )


def test_strftime_tz():
    t = parse.parse("{:%H:%M %z}", "10:21 +0530")[0]
    assert t.utcoffset() == timedelta(hours=5, minutes=30)


def test_strftime_no_year_uses_current_year():
    r = parse.parse("{:%m-%d}", "03-04")
    assert r[0] == date(datetime.today().year, 3, 4)


def test_strftime_two_digit_year():
    assert parse.parse("{:%y-%m-%d}", "23-01-05")[0] == date(2023, 1, 5)


def test_strftime_month_names_and_weekday():
    assert parse.parse("{:%a %d %b %Y}", "Thu 05 Jan 2023")[0] == date(2023, 1, 5)
    assert parse.parse("{:%A %d %B %Y}", "Thursday 05 January 2023")[0] == date(
        2023, 1, 5
    )


def test_strftime_day_of_year():
    assert parse.parse("{:%Y %j}", "2023 032")[0] == date(2023, 2, 1)


def test_strftime_rejects_non_matching():
    assert parse.parse("{:%Y-%m-%d}", "23-01-05") is None
    assert parse.parse("{:%H:%M}", "10:2") is None


def test_strftime_followed_by_field():
    r = parse.parse("{:%Y-%m-%d} {:d}", "2023-01-05 9")
    assert r.fixed == (date(2023, 1, 5), 9)


def test_strf_date_convert_neither_raises():
    with pytest.raises(ValueError):
        parse.strf_date_convert("%", None, "%%")


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("abc") == "abc"


# --------------------------------------------------------------------------
# FixedTzOffset
# --------------------------------------------------------------------------


def test_fixed_tz_offset():
    tz = parse.FixedTzOffset(90, "+0130")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "+0130"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedTzOffset +0130 1:30:00>"


def test_fixed_tz_offset_equality():
    a = parse.FixedTzOffset(60, "x")
    assert a == parse.FixedTzOffset(60, "x")
    assert a != parse.FixedTzOffset(61, "x")
    assert a != parse.FixedTzOffset(60, "y")
    assert (a == 5) is False
    assert a.__eq__(5) is NotImplemented


def test_fixed_tz_negative():
    tz = parse.FixedTzOffset(-300, "-0500")
    assert tz.utcoffset(None) == timedelta(hours=-5)


# --------------------------------------------------------------------------
# Misc / errors
# --------------------------------------------------------------------------


def test_module_metadata():
    assert parse.__all__ == ["parse", "search", "findall", "with_pattern"]
    assert isinstance(parse.__version__, str)


def test_error_classes_are_value_errors():
    assert issubclass(parse.TooManyFields, ValueError)
    assert issubclass(parse.RepeatedNameError, ValueError)


def test_invalid_group_name_raises_not_implemented():
    p = parse.compile("{:2.1}")
    with pytest.raises(NotImplementedError):
        p.parse("ab")


def test_empty_format_and_string():
    r = parse.parse("", "")
    assert r is not None
    assert r.fixed == ()
    assert parse.parse("", "x") is None


def test_unicode():
    r = parse.parse("{} → {:d}", "héllo → 3")
    assert r.fixed == ("héllo", 3)


def test_field_types_result_types():
    r = parse.parse("{:d} {:f} {:w} {:%}", "1 2.0 x 10%")
    assert [type(v) for v in r.fixed] == [int, float, str, float]
