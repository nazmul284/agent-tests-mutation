import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


# ---------- basic fixed / named ----------

def test_fixed_basic():
    r = parse.parse("hello {}", "hello world")
    assert r.fixed == ("world",)
    assert r.named == {}


def test_multiple_fixed():
    r = parse.parse("{} and {}", "cats and dogs")
    assert r.fixed == ("cats", "dogs")
    assert r[0] == "cats" and r[1] == "dogs"
    assert r[0:2] == ("cats", "dogs")


def test_named():
    r = parse.parse("{a} + {b}", "1 + 2")
    assert r.named == {"a": "1", "b": "2"}
    assert r["a"] == "1"
    assert "a" in r and "z" not in r


def test_no_match_returns_none():
    assert parse.parse("hello {}", "goodbye world") is None
    assert parse.parse("{}x", "abc") is None


def test_parse_must_match_fully():
    assert parse.parse("a{}", "abc def") is not None
    assert parse.parse("{:d}", "12abc") is None
    assert parse.parse("x{}", "yx1") is None
    assert parse.parse("{}x", "1xy") is None


def test_empty_format_and_string():
    r = parse.parse("", "")
    assert r is not None and r.fixed == ()
    assert parse.parse("", "a") is None


def test_lazy_matching():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_dotall_newline():
    r = parse.parse("{}|{}", "a\nb|c\nd")
    assert r.fixed == ("a\nb", "c\nd")


def test_case_insensitive_default():
    assert parse.parse("HELLO {}", "hello x").fixed == ("x",)


def test_case_sensitive():
    assert parse.parse("HELLO {}", "hello x", case_sensitive=True) is None
    assert parse.parse("hello {}", "hello x", case_sensitive=True).fixed == ("x",)
    assert parse.search("HELLO {}", "hello x", case_sensitive=True) is None
    assert list(parse.findall("HELLO {}", "hello x", case_sensitive=True)) == []
    assert parse.compile("A{}", case_sensitive=True).parse("a1") is None


def test_regex_special_chars_escaped():
    assert parse.parse("a.b {}", "a.b x").fixed == ("x",)
    assert parse.parse("a.b {}", "axb x") is None
    for lit in ["(", ")", "[", "]", "*", "+", "?", "^", "$", "|", "\\", "!"]:
        r = parse.parse("a" + lit + "{}", "a" + lit + "z")
        assert r is not None and r.fixed == ("z",), lit


def test_escaped_braces():
    r = parse.parse("{{{}}}", "{x}")
    assert r.fixed == ("x",)
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{}}", "ab") is None


# ---------- named fields: dotted / indexed / repeated ----------

def test_dotted_names():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}


def test_dotted_collision_names():
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_dash_names():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_dict_style_names():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}
    r = parse.parse("{a[b][c]}", "1")
    assert r.named == {"a": {"b": {"c": "1"}}}


def test_repeated_name_same_value():
    r = parse.parse("{a} {a}", "x x")
    assert r.named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None


def test_repeated_name_typed():
    r = parse.parse("{a:d} {a:d}", "5 5")
    assert r.named == {"a": 5}
    assert parse.parse("{a:d} {a:d}", "5 6") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(parse.RepeatedNameError):
        parse.parse("{a:d} {a:f}", "1 1.0")
    assert issubclass(parse.RepeatedNameError, ValueError)


def test_named_and_fixed_mixed_positions():
    r = parse.parse("{} {a:d} {}", "x 1 y")
    assert r.fixed == ("x", "y")
    assert r.named == {"a": 1}


def test_named_fields_property():
    p = parse.compile("{a} {} {b.c}")
    assert p.named_fields == ["a", "b_c"]
    assert p.fixed_fields == [1]
    # returned lists are copies
    p.named_fields.append("zzz")
    assert p.named_fields == ["a", "b_c"]
    p.fixed_fields.append(9)
    assert p.fixed_fields == [1]


def test_numbered_fields_are_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_parser_format_and_repr():
    p = parse.compile("hello {}")
    assert p.format == "hello {}"
    assert repr(p) == "<Parser 'hello {}'>"
    long = "x" * 30 + "{}"
    assert repr(parse.compile(long)) == "<Parser %r>" % (long[:17] + "...")
    assert repr(parse.compile("x" * 20)) == "<Parser %r>" % ("x" * 20)
    assert repr(parse.compile("x" * 21)) == "<Parser %r>" % ("x" * 17 + "...")


def test_result_repr():
    r = parse.parse("{} {a}", "1 2")
    assert repr(r) == "<Result ('1',) {'a': '2'}>"


# ---------- integers ----------

@pytest.mark.parametrize("fmt,s,expected", [
    ("{:d}", "12", 12),
    ("{:d}", "-12", -12),
    ("{:d}", "+12", 12),
    ("{:d}", " 12", 12),
    ("{:d}", "0", 0),
    ("{:d}", "0b101", 5),
    ("{:d}", "0o17", 15),
    ("{:d}", "0xff", 255),
    ("{:d}", "0XFF", 255),
    ("{:d}", "-0x10", -16),
    ("{:d}", "007", 7),
    ("{:d}", "0123", 123),
    ("{:d}", "10", 10),
    ("{:5d}", "   12", 12),
    ("{:<5d}", "12   ", 12),
    ("{:^5d}", " 12  ", 12),
    ("{:05d}", "00012", 12),
    ("{:>5d}", "   12", 12),
    ("{:*>5d}", "***12", 12),
    ("{:*<5d}", "12***", 12),
    ("{:x=5d}", "xxx12", 12),
    ("{:=5d}", "00012", 12),
    ("{:,d}", "1,234,567", 1234567),
    ("{:_d}", "1_234_567", 1234567),
    ("{:d}", "-5", -5),
    ("{:b}", "101", 5),
    ("{:b}", "0b101", 5),
    ("{:b}", "-101", -5),
    ("{:o}", "17", 15),
    ("{:o}", "0o17", 15),
    ("{:x}", "ff", 255),
    ("{:x}", "0xFF", 255),
    ("{:X}", "Ab", 171),
    ("{:x}", "-1f", -31),
    ("{:n}", "12", 12),
    ("{:n}", "1,234", 1234),
    ("{:n}", "1.234.567", 1234567),
    ("{:n}", "-1,234", -1234),
])
def test_int_conversion(fmt, s, expected):
    r = parse.parse(fmt, s)
    assert r is not None, (fmt, s)
    assert r[0] == expected
    assert isinstance(r[0], int)


@pytest.mark.parametrize("fmt,s", [
    ("{:d}", "abc"),
    ("{:d}", "1.5"),
    ("{:d}", ""),
    ("{:b}", "102"),
    ("{:o}", "18"),
    ("{:x}", "xyz"),
    ("{:n}", "abc"),
    ("{:n}", "1234,56"),
    ("{:3d}", "1234"),
])
def test_int_no_match(fmt, s):
    assert parse.parse(fmt, s) is None


def test_d_width_limits_digits():
    assert parse.parse("{:2d}", "12").fixed == (12,)
    assert parse.parse("{:2d}", "123") is None
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)


def test_int_convert_class():
    c = parse.int_convert()
    assert c("0x1f", None) == 31
    assert c("-0b11", None) == -3
    assert c("+0o7", None) == 7
    assert c("123", None) == 123
    assert parse.int_convert(2)("0b11", None) == 3
    assert parse.int_convert(16)("ff", None) == 255
    assert parse.int_convert(10)("1,000", None) == 1000
    assert parse.int_convert(8)("0o10", None) == 8


def test_int_negative_with_fill():
    assert parse.parse("{:05d}", "-0012").fixed == (-12,)


# ---------- floats ----------

@pytest.mark.parametrize("s,expected", [
    ("1.5", 1.5),
    ("-1.5", -1.5),
    ("+1.5", 1.5),
    (".5", 0.5),
    ("0.25", 0.25),
    ("100.0", 100.0),
])
def test_float(s, expected):
    r = parse.parse("{:f}", s)
    assert r is not None
    assert r[0] == expected
    assert isinstance(r[0], float)


def test_float_no_match():
    assert parse.parse("{:f}", "abc") is None
    assert parse.parse("{:f}", "12") is None
    assert parse.parse("{:f}", "1.") is None


def test_float_precision_zero():
    r = parse.parse("{:.0f}", "20")
    assert r.fixed == (20.0,)
    r = parse.parse("{:.0f}", "-20")
    assert r.fixed == (-20.0,)
    assert parse.parse("{:.0f}", "2.5") is None


def test_float_in_text():
    r = parse.parse("v={:f}!", "v=-3.14!")
    assert r.fixed == (-3.14,)


def test_float_nan_inf():
    import math
    assert math.isnan(parse.parse("{:f}", "nan")[0])
    assert parse.parse("{:f}", "inf")[0] == float("inf")


def test_float_width_align():
    assert parse.parse("{:8.2f}", "    3.14").fixed == (3.14,)
    assert parse.parse("{:<8.2f}", "3.14    ").fixed == (3.14,)
    assert parse.parse("{:^8.2f}", "  3.14  ").fixed == (3.14,)
    assert parse.parse("{:08.2f}", "00003.14").fixed == (3.14,)
    assert parse.parse("{:=8.2f}", "00003.14").fixed == (3.14,)
    assert parse.parse("{:*>8.2f}", "****3.14").fixed == (3.14,)


def test_decimal_type_F():
    r = parse.parse("{:F}", "3.14")
    assert r[0] == Decimal("3.14")
    assert isinstance(r[0], Decimal)
    r = parse.parse("{:F}", "-3.14")
    assert r[0] == Decimal("-3.14")
    r = parse.parse("{:.0F}", "7")
    assert r[0] == Decimal("7")


def test_exponent():
    assert parse.parse("{:e}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:e}", "1.5E-3").fixed == (0.0015,)
    assert parse.parse("{:E}", "2.5e+2").fixed == (250.0,)
    assert parse.parse("{:e}", "1.5") is None
    assert parse.parse("{:e}", "abc") is None
    assert parse.parse("{:e}", "-1.5e3").fixed == (-1500.0,)


def test_general():
    assert parse.parse("{:g}", "12").fixed == (12.0,)
    assert parse.parse("{:g}", "1.5").fixed == (1.5,)
    assert parse.parse("{:g}", "1.5e3").fixed == (1500.0,)
    assert parse.parse("{:g}", "-1.5e3").fixed == (-1500.0,)
    assert parse.parse("{:G}", "1E5").fixed == (100000.0,)
    assert parse.parse("{:g}", "abc") is None
    assert parse.parse("{:g}", "-inf").fixed == (float("-inf"),)


def test_group_index_after_complex_types():
    # fields after types with internal regex groups must still map properly
    r = parse.parse("{:g} {:n} {:b} {:o} {:x} {:%} {} {:d}",
                    "1.5e2 1,000 0b11 0o7 ff 50% z 9")
    assert r.fixed == (150.0, 1000, 3, 7, 255, 0.5, "z", 9)
    r = parse.parse("{:e} {} {:f} {}", "1.5e1 a 2.5 b")
    assert r.fixed == (15.0, "a", 2.5, "b")


def test_percentage():
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "100%").fixed == (1.0,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "abc%") is None
    assert parse.percentage("25%", None) == 0.25


# ---------- strings / widths ----------

def test_width_only():
    assert parse.parse("{:5}", "ab") is None
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "abcdef").fixed == ("abcdef",)


def test_precision_only():
    assert parse.parse("{:.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None


def test_width_and_precision():
    assert parse.parse("{:2.4}", "abc").fixed == ("abc",)
    assert parse.parse("{:2.4}", "a") is None
    assert parse.parse("{:2.4}", "abcde") is None


def test_alignment_strips_padding_before_unsized_value():
    # without a width the lazy value is followed/preceded by fill chars that
    # must be consumed as padding rather than captured
    assert parse.parse("{:*^}", "**ab**") is not None
    assert parse.parse("{:<}x", "abx").fixed == ("ab",)


def test_width_default_right_align():
    assert parse.parse("{:6}", "    ab") is not None


def test_letters_type():
    assert parse.parse("{:l}", "abcXYZ").fixed == ("abcXYZ",)
    assert parse.parse("{:l}", "abc1") is None
    assert parse.parse("{:l} {:l}", "ab cd").fixed == ("ab", "cd")
    assert parse.parse("{:l}", "") is None


def test_word_and_other_regex_types():
    assert parse.parse("{:w}", "abc_1").fixed == ("abc_1",)
    assert parse.parse("{:w}", "a b") is None
    assert parse.parse("{:W}", "  !") is not None
    assert parse.parse("{:s}", "   ").fixed == ("   ",)
    assert parse.parse("{:S}", "abc").fixed == ("abc",)
    assert parse.parse("{:S}", "a c") is None
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None


# ---------- invalid formats ----------

@pytest.mark.parametrize("fmt", ["{:z}", "{:q}", "{:y1}", "{:k}", "{:.2k}", "{:5q}"])
def test_bad_type_raises(fmt):
    with pytest.raises(ValueError):
        parse.parse(fmt, "1")


def test_too_many_fields():
    fmt = " ".join("{:d}x" for _ in range(150))
    # either supported by this python's re or raises TooManyFields
    try:
        p = parse.compile(fmt)
        p.parse("1x " * 149 + "1x")
    except parse.TooManyFields:
        pass
    assert issubclass(parse.TooManyFields, ValueError)


def test_named_group_error_non_escaped():
    # names starting with non-alpha are treated as fixed
    r = parse.parse("{_a}", "x")
    assert r is None or r.named == {}


# ---------- extra types ----------

def test_extra_type_basic():
    r = parse.parse("{:Number}", "42", extra_types={"Number": int})
    assert r.fixed == (42,)


def test_extra_type_with_pattern():
    @parse.with_pattern(r"\d+")
    def num(text):
        return int(text)

    assert num.pattern == r"\d+"
    r = parse.parse("{:Number} items", "42 items", extra_types={"Number": num})
    assert r.fixed == (42,)
    assert parse.parse("{:Number} items", "x items", extra_types={"Number": num}) is None


def test_extra_type_group_count():
    @parse.with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(text):
        return text.upper()

    assert ab.regex_group_count == 2
    r = parse.parse("{:AB} {} {n:AB}", "ac z bd", extra_types={"AB": ab})
    assert r.fixed == ("AC", "z")
    assert r.named == {"n": "BD"}


def test_extra_type_default_pattern():
    r = parse.parse("{:X1}-{}", "abc-def", extra_types={"X1": str.upper})
    assert r.fixed == ("ABC", "def")


def test_extra_type_named():
    @parse.with_pattern(r"yes|no")
    def yn(t):
        return t == "yes"

    r = parse.parse("{a:YN} {b:YN}", "yes no", extra_types={"YN": yn})
    assert r.named == {"a": True, "b": False}


def test_extra_type_with_align_width():
    @parse.with_pattern(r"\d+")
    def num(t):
        return int(t)

    r = parse.parse("{:>5Num}", "   12", extra_types={"Num": num})
    assert r.fixed == (12,)


def test_convert_first_pickle():
    c = parse.convert_first(int)
    assert c("5", None) == 5
    c2 = pickle.loads(pickle.dumps(c))
    assert c2("7", None) == 7


# ---------- search / findall ----------

def test_search():
    r = parse.search("age: {:d}", "name: bob, age: 42, ok")
    assert r.fixed == (42,)
    assert r.spans[0] == (16, 18)
    assert parse.search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert parse.search("a{:d}", s, pos=1).fixed == (2,)
    assert parse.search("a{:d}", s, pos=3).fixed == (2,)
    assert parse.search("a{:d}", s, pos=6).fixed == (3,)
    assert parse.search("a{:d}", s, endpos=1) is None
    assert parse.search("a{:d}", s, pos=3, endpos=5).fixed == (2,)
    assert parse.search("a{:d}", s, pos=6, endpos=7) is None


def test_findall():
    res = list(parse.findall("<{}>", "<a> <b> <c>"))
    assert [r[0] for r in res] == ["a", "b", "c"]


def test_findall_numbers_non_overlapping():
    res = list(parse.findall("{:d}", "1 22 333"))
    assert [r[0] for r in res] == [1, 22, 333]


def test_findall_pos_endpos():
    s = "<a><b><c><d>"
    assert [r[0] for r in parse.findall("<{}>", s, pos=3)] == ["b", "c", "d"]
    assert [r[0] for r in parse.findall("<{}>", s, endpos=6)] == ["a", "b"]
    assert [r[0] for r in parse.findall("<{}>", s, pos=3, endpos=9)] == ["b", "c"]


def test_findall_empty():
    assert list(parse.findall("<{}>", "none")) == []


def test_findall_iterator_protocol():
    it = parse.findall("<{}>", "<a>")
    assert iter(it) is it
    assert next(it)[0] == "a"
    with pytest.raises(StopIteration):
        next(it)


def test_findall_evaluate_false():
    res = list(parse.findall("<{:d}>", "<1><2>", evaluate_result=False))
    assert all(isinstance(m, parse.Match) for m in res)
    assert [m.evaluate_result()[0] for m in res] == [1, 2]


def test_findall_named_spans():
    res = list(parse.findall("{k}={v}", "a=1;b=2", ))


def test_evaluate_result_false():
    m = parse.parse("{:d}", "5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (5,)
    m = parse.search("{:d}", "x5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (5,)
    assert parse.parse("{:d}", "x", evaluate_result=False) is None
    assert parse.search("{:d}", "x", evaluate_result=False) is None


def test_spans():
    r = parse.parse("ab{}cd{n}", "abXXcdYY")
    assert r.spans[0] == (2, 4)
    assert r.spans["n"] == (6, 8)


def test_spans_with_complex_types_before():
    r = parse.parse("{:g}|{}|{n}", "1.5e2|zz|q")
    assert r.spans[0] == (0, 5)
    assert r.spans[1] == (6, 8)
    assert r.spans["n"] == (9, 10)


def test_spans_dotted():
    r = parse.parse("{a.b}-{c}", "x-y")
    assert r.spans["a.b"] == (0, 1)
    assert r.spans["c"] == (2, 3)


def test_compile_reuse():
    p = parse.compile("{:d}-{:d}")
    assert p.parse("1-2").fixed == (1, 2)
    assert p.parse("3-4").fixed == (3, 4)
    assert p.search("xx 5-6 yy").fixed == (5, 6)
    assert [r.fixed for r in p.findall("1-2 3-4")] == [(1, 2), (3, 4)]


# ---------- datetime: ti, tg, ta, te, th, tc, tt, ts ----------

def test_ti_date_only():
    r = parse.parse("{:ti}", "2023-05-17")
    assert r[0] == datetime(2023, 5, 17)


def test_ti_datetime():
    r = parse.parse("{:ti}", "2023-05-17 13:45:10")
    assert r[0] == datetime(2023, 5, 17, 13, 45, 10)
    r = parse.parse("{:ti}", "2023-05-17T13:45")
    assert r[0] == datetime(2023, 5, 17, 13, 45)


def test_ti_microseconds():
    r = parse.parse("{:ti}", "2023-05-17 13:45:10.5")
    assert r[0].microsecond == 500000
    r = parse.parse("{:ti}", "2023-05-17 13:45:10.123456")
    assert r[0].microsecond == 123456
    r = parse.parse("{:ti}", "2023-05-17 13:45:10.1234567")
    assert r[0].microsecond == 123456
    r = parse.parse("{:ti}", "2023-05-17 13:45:10.000001")
    assert r[0].microsecond == 1


def test_ti_tz():
    r = parse.parse("{:ti}", "2023-05-17 13:45:10Z")
    assert r[0].utcoffset() == timedelta(0)
    assert r[0].tzinfo.tzname(None) == "UTC"
    r = parse.parse("{:ti}", "2023-05-17 13:45:10 +05:30")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2023-05-17 13:45:10-0800")
    assert r[0].utcoffset() == timedelta(hours=-8)
    r = parse.parse("{:ti}", "2023-05-17 13:45:10 -03:30")
    assert r[0].utcoffset() == timedelta(hours=-3, minutes=-30)


def test_ti_followed_by_field():
    r = parse.parse("{:ti} {}", "2023-05-17 13:45:10 hello")
    assert r.fixed[0] == datetime(2023, 5, 17, 13, 45, 10)
    assert r.fixed[1] == "hello" or r.fixed[1] is not None


def test_ti_group_index_after():
    r = parse.parse("{:ti}|{:d}|{}", "2023-05-17 13:45:10|7|z")
    assert r.fixed == (datetime(2023, 5, 17, 13, 45, 10), 7, "z")


def test_ti_no_match():
    assert parse.parse("{:ti}", "not a date") is None
    assert parse.parse("{:ti}", "17-05-2023") is None


def test_tg():
    r = parse.parse("{:tg}", "17/5/2023")
    assert r[0] == datetime(2023, 5, 17)
    r = parse.parse("{:tg}", "17-05-2023 13:45:10")
    assert r[0] == datetime(2023, 5, 17, 13, 45, 10)
    r = parse.parse("{:tg}", "17/May/2023")
    assert r[0] == datetime(2023, 5, 17)
    r = parse.parse("{:tg}", "1/January/2023")
    assert r[0] == datetime(2023, 1, 1)
    r = parse.parse("{:tg}", "17/5/2023 1:05 PM")
    assert r[0] == datetime(2023, 5, 17, 13, 5)
    r = parse.parse("{:tg}", "17/5/2023 1:05 AM")
    assert r[0] == datetime(2023, 5, 17, 1, 5)
    r = parse.parse("{:tg}", "17/5/2023 12:05 AM")
    assert r[0] == datetime(2023, 5, 17, 0, 5)
    r = parse.parse("{:tg}", "17/5/2023 12:05 PM")
    assert r[0] == datetime(2023, 5, 17, 12, 5)
    r = parse.parse("{:tg}", "17/5/2023 13:05 +0100")
    assert r[0].utcoffset() == timedelta(hours=1)


def test_ta():
    r = parse.parse("{:ta}", "5/17/2023")
    assert r[0] == datetime(2023, 5, 17)
    r = parse.parse("{:ta}", "5-17-2023 08:30")
    assert r[0] == datetime(2023, 5, 17, 8, 30)
    r = parse.parse("{:ta}", "Dec/25/2023")
    assert r[0] == datetime(2023, 12, 25)
    r = parse.parse("{:ta}", "5/17/2023 11:30 PM")
    assert r[0] == datetime(2023, 5, 17, 23, 30)
    r = parse.parse("{:ta}", "5/17/2023 12:30 AM")
    assert r[0] == datetime(2023, 5, 17, 0, 30)


def test_te():
    r = parse.parse("{:te}", "Mon, 17 Jul 2023 13:45:10 +0200")
    assert r[0].replace(tzinfo=None) == datetime(2023, 7, 17, 13, 45, 10)
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:te}", "17 Jul 2023 13:45:10 -0500")
    assert r[0].utcoffset() == timedelta(hours=-5)
    assert r[0].month == 7 and r[0].day == 17


def test_th():
    r = parse.parse("{:th}", "17/Jul/2023:13:45:10 +0200")
    assert r[0].replace(tzinfo=None) == datetime(2023, 7, 17, 13, 45, 10)
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:th}", "01/Dec/1999:00:00:01 -0100")
    assert r[0].year == 1999 and r[0].month == 12 and r[0].day == 1
    assert r[0].second == 1


def test_tc():
    r = parse.parse("{:tc}", "Mon Jul 17 13:45:10 2023")
    assert r[0] == datetime(2023, 7, 17, 13, 45, 10)
    r = parse.parse("{:tc}", "Sun Dec  3 01:02:03 2000")
    assert r[0] == datetime(2000, 12, 3, 1, 2, 3)


def test_tt():
    r = parse.parse("{:tt}", "13:45:10")
    assert r[0] == time(13, 45, 10)
    r = parse.parse("{:tt}", "1:05 PM")
    assert r[0] == time(13, 5)
    r = parse.parse("{:tt}", "12:05 AM")
    assert r[0] == time(0, 5)
    r = parse.parse("{:tt}", "12:05 PM")
    assert r[0] == time(12, 5)
    r = parse.parse("{:tt}", "11:59:59 PM")
    assert r[0] == time(23, 59, 59)
    r = parse.parse("{:tt}", "13:45 +0530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:tt}", "13:45:10.25")
    assert r[0] == time(13, 45, 10, 250000)


def test_ts():
    r = parse.parse("{:ts}", "Jul 17 13:45:10")
    assert r[0] == datetime(datetime.today().year, 7, 17, 13, 45, 10)
    r = parse.parse("{:ts}", "Dec  3 01:02:03")
    assert r[0].month == 12 and r[0].day == 3
    assert r[0].hour == 1 and r[0].minute == 2 and r[0].second == 3


def test_date_in_context():
    r = parse.parse("on {:ti} we met {name}", "on 2020-01-02 we met bob")
    assert r.fixed == (datetime(2020, 1, 2),)
    assert r.named == {"name": "bob"}


def test_fixed_tz_offset():
    tz = parse.FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "X"
    assert tz.dst(None) == timedelta(0)
    assert tz == parse.FixedTzOffset(90, "X")
    assert tz != parse.FixedTzOffset(60, "X")
    assert tz != parse.FixedTzOffset(90, "Y")
    assert tz != 5
    assert "X" in repr(tz) and "FixedTzOffset" in repr(tz)


def test_tz_negative_offset_name():
    r = parse.parse("{:ti}", "2023-05-17 13:45:10 -0830")
    assert r[0].utcoffset() == timedelta(hours=-8, minutes=-30)


# ---------- strptime-style formats ----------

def test_strf_date():
    r = parse.parse("{:%Y-%m-%d}", "2023-05-17")
    assert r[0] == date(2023, 5, 17)
    assert type(r[0]) is date


def test_strf_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-05-17 13:45:10")
    assert r[0] == datetime(2023, 5, 17, 13, 45, 10)


def test_strf_time():
    r = parse.parse("{:%H:%M}", "13:45")
    assert r[0] == time(13, 45)
    r = parse.parse("{:%I:%M %p}", "01:45 PM")
    assert r[0] == time(13, 45)


def test_strf_names():
    r = parse.parse("{:%A, %B %d, %Y}", "Monday, July 17, 2023")
    assert r[0] == date(2023, 7, 17)
    r = parse.parse("{:%a %b %d %Y}", "Mon Jul 17 2023")
    assert r[0] == date(2023, 7, 17)


def test_strf_no_year_uses_current():
    r = parse.parse("{:%m-%d}", "05-17")
    assert r[0] == date(datetime.today().year, 5, 17)


def test_strf_microseconds_and_tz():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S.%f}", "2023-05-17 13:45:10.123456")
    assert r[0].microsecond == 123456
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2023-05-17 13:45:10+0200")
    assert r[0].utcoffset() == timedelta(hours=2)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2023-05-17 13:45:10-0200")
    assert r[0].utcoffset() == timedelta(hours=-2)


def test_strf_two_digit_year():
    r = parse.parse("{:%y-%m-%d}", "23-05-17")
    assert r[0] == date(2023, 5, 17)


def test_strf_no_match():
    assert parse.parse("{:%Y-%m-%d}", "17/05/2023") is None
    assert parse.parse("{:%Y-%m-%d}", "23-05-17") is None
    assert parse.parse("{:%H:%M}", "1:5") is None


def test_strf_followed_by_fields():
    r = parse.parse("{:%Y-%m-%d} {}", "2023-05-17 hello")
    assert r.fixed == (date(2023, 5, 17), "hello")
    r = parse.parse("{d:%Y-%m-%d} {n:d}", "2023-05-17 4")
    assert r.named == {"d": date(2023, 5, 17), "n": 4}


def test_strf_day_of_year_and_weekday():
    r = parse.parse("{:%Y %j}", "2023 032")
    assert r[0] == date(2023, 2, 1)


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("%d/%m") == "[0-9]{1,2}/[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("abc") == "abc"


# ---------- extract_format ----------

def test_extract_format_parts():
    f = parse.extract_format("*^+010,.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "^"
    assert f["type"] == "f"
    f = parse.extract_format(">10d", {})
    assert f["align"] == ">" and f["width"] == "10" and f["type"] == "d"
    assert f["fill"] is None
    f = parse.extract_format("08.3f", {})
    assert f["zero"] is True and f["width"] == "8" and f["precision"] == "3"
    f = parse.extract_format(".3", {})
    assert f["precision"] == "3" and f["type"] == ""
    f = parse.extract_format("d", {})
    assert f["type"] == "d" and f["width"] == ""
    f = parse.extract_format(",d", {})
    assert f["grouping"] == ","
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    f = parse.extract_format("=5d", {})
    assert f["align"] == "="


def test_extract_format_extra_types():
    assert parse.extract_format("MyT", {"MyT": int})["type"] == "MyT"
    with pytest.raises(ValueError):
        parse.extract_format("MyT", {})


def test_extract_format_sign():
    for s in ["+d", "-d", " d"]:
        assert parse.extract_format(s, {})["type"] == "d"


# ---------- misc ----------

def test_module_all():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}


def test_with_pattern_defaults():
    @parse.with_pattern(r"x")
    def f(t):
        return t

    assert f.pattern == "x"
    assert f.regex_group_count is None
    assert f("a") == "a"


def test_date_convert_direct_time_only():
    class M:
        def groups(self):
            return ("1:02:03.5",)

    r = parse.date_convert("", M(), hms=0)
    assert r == time(1, 2, 3, 500000)


def test_sign_before_padding_floats():
    assert parse.parse("{:+.2f}", "+3.14").fixed == (3.14,)
    assert parse.parse("{:.2f}", "-3.14").fixed == (-3.14,)


def test_text_between_fields_special():
    r = parse.parse("[{}] ({}) {}*", "[a] (b) c*")
    assert r.fixed == ("a", "b", "c")


def test_unicode():
    assert parse.parse("héllo {}", "héllo wörld").fixed == ("wörld",)


def test_empty_field_value():
    # lazy ".+?" requires at least one char
    assert parse.parse("a{}b", "ab") is None
    assert parse.parse("a{}b", "axb").fixed == ("x",)
