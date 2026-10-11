import pickle
from datetime import date, datetime, time, timedelta

import pytest

import parse
from parse import (
    FixedTzOffset,
    Parser,
    RepeatedNameError,
    TooManyFields,
    compile,
    findall,
    search,
    with_pattern,
)


# ---------- basic fixed / named ----------

def test_no_fields_match():
    assert parse.parse("hello", "hello").fixed == ()
    assert parse.parse("hello", "hello!") is None


def test_fixed_fields():
    r = parse.parse("{} and {}", "a and b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a" and r[1] == "b"
    assert r[0:2] == ("a", "b")


def test_named_fields():
    r = parse.parse("{x} + {y}", "1 + 2")
    assert r.named == {"x": "1", "y": "2"}
    assert r["x"] == "1"
    assert "x" in r and "z" not in r


def test_mixed_fixed_and_named():
    r = parse.parse("{} {name} {}", "a b c")
    assert r.fixed == ("a", "c")
    assert r.named == {"name": "b"}


def test_exact_match_required():
    assert parse.parse("{}", "abc").fixed == ("abc",)
    assert parse.parse("a{}", "xab") is None
    assert parse.parse("{}a", "abx") is None


def test_multiline_dotall():
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)


def test_no_match_returns_none():
    assert parse.parse("{:d}", "abc") is None


def test_escaped_braces():
    assert parse.parse("{{}}", "{}").fixed == ()
    assert parse.parse("{{{}}}", "{abc}").fixed == ("abc",)
    assert parse.parse("{{ {} }}", "{ x }").fixed == ("x",)


def test_regex_special_chars_escaped():
    assert parse.parse("a.b {}", "a.b 1").fixed == ("1",)
    assert parse.parse("a.b {}", "aXb 1") is None
    assert parse.parse("(x)[y]*+?^$|\\ {}", "(x)[y]*+?^$|\\ 1").fixed == ("1",)
    assert parse.parse("what? {}", "what? yes").fixed == ("yes",)


def test_parse_case_sensitive_default():
    assert parse.parse("Hello {}", "hello x") is None
    assert parse.parse("Hello {}", "hello x", case_sensitive=False).fixed == ("x",)


def test_search_case_insensitive_default():
    assert search("Hello {}", "say hello x").fixed == ("x",)
    assert search("Hello {}", "say hello x", case_sensitive=True) is None


def test_compile_case_insensitive_default():
    assert compile("Hello {}").parse("hello x").fixed == ("x",)
    assert compile("Hello {}", case_sensitive=True).parse("hello x") is None


def test_dotted_names():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}


def test_bracket_names():
    r = parse.parse("{a[b]} {a[c]}", "1 2")
    assert r.named == {"a": {"b": "1", "c": "2"}}


def test_dash_names():
    r = parse.parse("{a-b}", "1")
    assert r.named == {"a-b": "1"}


def test_name_collisions():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}


def test_repeated_name_same_value():
    assert parse.parse("{a} {a}", "x x").named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.parse("{a:d} {a:w}", "1 1")


def test_repeated_name_same_type_ok():
    assert parse.parse("{a:d} {a:d}", "1 1").named == {"a": 1}


def test_named_fields_property_and_fixed_fields():
    p = compile("{a} {} {b} {}")
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1, 3] or p.fixed_fields == [1, 3]
    assert p.format == "{a} {} {b} {}"
    # properties return copies
    p.named_fields.append("zzz")
    assert p.named_fields == ["a", "b"]


def test_parser_repr():
    assert repr(Parser("{}")) == "<Parser '{}'>"
    r = repr(Parser("x" * 30))
    assert r == "<Parser %r>" % ("x" * 17 + "...")
    assert repr(Parser("x" * 20)) == "<Parser %r>" % ("x" * 20)
    assert repr(Parser("x" * 21)) == "<Parser %r>" % ("x" * 17 + "...")


def test_result_repr():
    r = parse.parse("{} {a}", "1 2")
    assert repr(r) == "<Result ('1',) {'a': '2'}>"


def test_spans():
    r = parse.parse("hi {} and {name}", "hi abc and def")
    assert r.spans[0] == (3, 6)
    assert r.spans["name"] == (11, 14)


def test_spans_with_typed_fields():
    r = parse.parse("{:d}-{:g}-{:w}", "12-3.5-ab")
    assert r.spans[0] == (0, 2)
    assert r.spans[1] == (3, 6)
    assert r.spans[2] == (7, 9)


def test_too_many_fixed_groups():
    # many complex fields should still work or raise TooManyFields
    fmt = " ".join(["{:g}"] * 60)
    try:
        r = parse.parse(fmt, " ".join(["1.5"] * 60))
    except TooManyFields:
        return
    assert r is not None and len(r.fixed) == 60


def test_many_fixed_fields_group_indexing():
    r = parse.parse("{:g} {:d} {:tg} {:d} {}", "1.5 2 1/2/2020 10:20 4 x")
    assert r.fixed[0] == 1.5 and r.fixed[1] == 2
    assert r.fixed[2] == datetime(2020, 2, 1, 10, 20)
    assert r.fixed[3:] == (4, "x")


# ---------- integer types ----------

@pytest.mark.parametrize(
    "fmt,s,expected",
    [
        ("{:d}", "12", 12),
        ("{:d}", "-12", -12),
        ("{:d}", "+12", 12),
        ("{:d}", " 12", 12),
        ("{:d}", "0", 0),
        ("{:d}", "0x1f", 31),
        ("{:d}", "0X1F", 31),
        ("{:d}", "0b101", 5),
        ("{:d}", "0o17", 15),
        ("{:d}", "-0x10", -16),
        ("{:d}", "007", 7),
        ("{:b}", "101", 5),
        ("{:b}", "0b101", 5),
        ("{:b}", "-101", -5),
        ("{:o}", "17", 15),
        ("{:o}", "0o17", 15),
        ("{:x}", "ff", 255),
        ("{:x}", "0xff", 255),
        ("{:x}", "FF", 255),
        ("{:X}", "0XFF", 255),
        ("{:x}", "-ff", -255),
        ("{:n}", "123", 123),
        ("{:n}", "1,234", 1234),
        ("{:n}", "1,234,567", 1234567),
        ("{:n}", "1.234", 1234),
        ("{:n}", "-1,234", -1234),
    ],
)
def test_integer_types(fmt, s, expected):
    r = parse.parse(fmt, s)
    assert r is not None
    assert r[0] == expected
    assert isinstance(r[0], int)


@pytest.mark.parametrize(
    "fmt,s",
    [
        ("{:d}", "1.5"),
        ("{:d}", "abc"),
        ("{:d}", ""),
        ("{:b}", "102"),
        ("{:o}", "18"),
        ("{:x}", "xyz"),
        ("{:n}", "1234,5"),
    ],
)
def test_integer_nonmatch(fmt, s):
    assert parse.parse(fmt, s) is None


def test_d_with_width():
    assert parse.parse("{:2d}", "12").fixed == (12,)
    assert parse.parse("{:2d}", "123") is None
    assert parse.parse("{:3d}", "  7").fixed == (7,)


def test_d_grouping():
    assert parse.parse("{:d}", "1,000") is None
    assert parse.parse("{:,d}", "1,000,000").fixed == (1000000,)
    assert parse.parse("{:_d}", "1_000").fixed == (1000,)
    assert parse.parse("{:,d}", "1000").fixed == (1000,)


def test_int_hex_prefix_with_base_10_literal():
    # "0" followed by short strings must stay decimal
    assert parse.parse("{:d}", "00").fixed == (0,)
    assert parse.parse("{:d}", "010").fixed == (10,)


def test_int_convert_directly():
    c = parse.int_convert()
    assert c("0b11", None) == 3
    assert c("+0x10", None) == 16
    assert c("-0o10", None) == -8
    assert c("12", None) == 12
    assert parse.int_convert(16)("ff", None) == 255
    assert parse.int_convert(2)("0b11", None) == 3
    assert parse.int_convert(8)("0o11", None) == 9


def test_alignment_ints():
    assert parse.parse("{:>5d}", "   12").fixed == (12,)
    assert parse.parse("{:<5d}", "12   ").fixed == (12,)
    assert parse.parse("{:^5d}", " 12  ").fixed == (12,)
    assert parse.parse("{:05d}", "00012").fixed == (12,)
    assert parse.parse("{:*>5d}", "***12").fixed == (12,)
    assert parse.parse("{:*<5d}", "12***").fixed == (12,)
    assert parse.parse("{:*^6d}", "**12**").fixed == (12,)
    assert parse.parse("{:0=5d}", "00012").fixed == (12,)
    assert parse.parse("{:x=5d}", "xxx12").fixed == (12,)
    assert parse.parse("{:=5d}", "00012").fixed == (12,)


def test_align_negative_with_zero_pad():
    assert parse.parse("{:=5d}", "-0012").fixed == (-12,)


def test_fill_special_chars_escaped():
    assert parse.parse("{:.>5d}", "...12").fixed == (12,)
    assert parse.parse("{:+<5d}", "12+++").fixed == (12,)
    assert parse.parse("{:*<}", "ab***").fixed == ("ab",)
    assert parse.parse("{:$>}", "$$$ab").fixed == ("ab",)
    assert parse.parse("{:)>}", ")))ab").fixed == ("ab",)
    assert parse.parse("{:[>}", "[[[ab").fixed == ("ab",)


# ---------- floats ----------

@pytest.mark.parametrize(
    "fmt,s,expected",
    [
        ("{:f}", "1.5", 1.5),
        ("{:f}", "-1.5", -1.5),
        ("{:f}", "+1.5", 1.5),
        ("{:f}", ".5", 0.5),
        ("{:f}", "12.345", 12.345),
        ("{:e}", "1.5e3", 1500.0),
        ("{:e}", "1.5E-3", 0.0015),
        ("{:e}", "-1.5e+3", -1500.0),
        ("{:g}", "1.5", 1.5),
        ("{:g}", "15", 15.0),
        ("{:g}", "1e3", 1000.0),
        ("{:g}", "1.5e-2", 0.015),
        ("{:G}", "2.5", 2.5),
        ("{:E}", "2.5e1", 25.0),
        ("{:F}", "2.5", 2.5),
    ],
)
def test_floats(fmt, s, expected):
    r = parse.parse(fmt, s)
    assert r is not None
    assert r[0] == pytest.approx(expected)


def test_f_returns_float_and_F_decimal():
    from decimal import Decimal
    assert isinstance(parse.parse("{:f}", "1.5")[0], float)
    r = parse.parse("{:F}", "1.10")[0]
    assert isinstance(r, Decimal) and r == Decimal("1.10")


def test_float_requires_point():
    assert parse.parse("{:f}", "15") is None
    assert parse.parse("{:e}", "15") is None
    assert parse.parse("{:f}", "abc") is None


def test_float_precision_zero():
    assert parse.parse("{:.0f}", "20").fixed == (20.0,)
    assert parse.parse("{:.0f}", "-20").fixed == (-20.0,)


def test_float_precision_nonzero():
    assert parse.parse("{:.2f}", "20.25").fixed == (20.25,)
    assert parse.parse("{:.2f}", "-20.25").fixed == (-20.25,)
    assert parse.parse("{:.2f}", "20") is None


def test_float_nan_inf():
    import math
    assert math.isnan(parse.parse("{:f}", "nan")[0])
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert math.isnan(parse.parse("{:g}", "nan")[0])
    assert parse.parse("{:g}", "inf")[0] == float("inf")
    assert parse.parse("{:g}", "-inf")[0] == float("-inf")
    assert parse.parse("{:e}", "inf")[0] == float("inf")


def test_float_padding():
    assert parse.parse("{:8.2f}", "   12.50").fixed == (12.5,)
    assert parse.parse("{:<8.2f}", "12.50   ").fixed == (12.5,)
    assert parse.parse("{:^8.2f}", " 12.50  ").fixed == (12.5,)
    assert parse.parse("{:08.2f}", "00012.50").fixed == (12.5,)
    assert parse.parse("{:=8.2f}", "-0012.50").fixed == (-12.5,)


def test_percentage():
    assert parse.parse("{:%}", "50%").fixed == (0.5,)
    assert parse.parse("{:%}", "12.5%").fixed == (0.125,)
    assert parse.parse("{:%}", "100%").fixed == (1.0,)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "-50%").fixed == (-0.5,)


# ---------- strings / widths ----------

def test_word_types():
    assert parse.parse("{:w}", "abc_12").fixed == ("abc_12",)
    assert parse.parse("{:w}", "abc def") is None
    assert parse.parse("{:W}", "  ,.") .fixed == ("  ,.",)
    assert parse.parse("{:W}", "abc") is None
    assert parse.parse("{:s}", " \t").fixed == (" \t",)
    assert parse.parse("{:s}", "ab") is None
    assert parse.parse("{:S}", "ab").fixed == ("ab",)
    assert parse.parse("{:S}", "a b") is None
    assert parse.parse("{:D}", "abc").fixed == ("abc",)
    assert parse.parse("{:D}", "a1") is None
    assert parse.parse("{:l}", "abc").fixed == ("abc",)
    assert parse.parse("{:l}", "ab1") is None
    assert parse.parse("{:l}", "") is None


def test_width_and_precision_strings():
    assert parse.parse("{:3}", "abc").fixed == ("abc",)
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:2.3}", "abc").fixed == ("abc",)
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "abcd") is None
    assert parse.parse("{:5}", "   ab").fixed == ("   ab",)


def test_string_alignment_strip():
    assert parse.parse("{:>}", "   ab").fixed == ("ab",)
    assert parse.parse("{:<}", "ab   ").fixed == ("ab",)
    assert parse.parse("{:^}", " ab  ").fixed == ("ab",)
    assert parse.parse("{:*^}", "**ab**").fixed == ("ab",)


def test_lazy_matching():
    r = parse.parse("{} {}", "a b c")
    assert r.fixed == ("a", "b c")
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")


def test_invalid_format_spec():
    with pytest.raises(ValueError):
        parse.parse("{:q}", "x")
    with pytest.raises(ValueError):
        Parser("{:z!}")


def test_empty_name_with_spec():
    assert parse.parse("{:d}", "3").fixed == (3,)


def test_numeric_sign_prefix_stripped_properly():
    assert parse.parse("{:d} {:d}", "-1 +2").fixed == (-1, 2)


# ---------- extra types ----------

def test_extra_types_basic():
    def shouty(text):
        return text.upper()

    shouty.pattern = r"[a-z]+"
    r = parse.parse("{:shouty}", "abc", dict(shouty=shouty))
    assert r.fixed == ("ABC",)
    assert parse.parse("{:shouty}", "ab1", dict(shouty=shouty)) is None


def test_extra_types_no_pattern():
    r = parse.parse("{:up}", "abc", dict(up=str.upper))
    assert r.fixed == ("ABC",)


def test_with_pattern():
    @with_pattern(r"\d+")
    def num(t):
        return int(t)

    assert num.pattern == r"\d+"
    assert num.regex_group_count is None
    assert parse.parse("{:num}", "42", dict(num=num)).fixed == (42,)
    assert parse.parse("{a:num}", "42", dict(num=num)).named == {"a": 42}


def test_with_pattern_group_count():
    @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(t):
        return t[::-1]

    r = parse.parse("{:ab} {:d} {:ab}", "ac 5 bd", dict(ab=ab))
    assert r.fixed == ("ca", 5, "db")
    r = parse.parse("{x:ab} {y:d}", "ac 5", dict(ab=ab))
    assert r.named == {"x": "ca", "y": 5}


def test_with_pattern_group_count_followed_by_fixed():
    @with_pattern(r"(a)(b)", regex_group_count=2)
    def ab(t):
        return t

    r = parse.parse("{:ab}|{}|{:ab}|{}", "ab|x|ab|y", dict(ab=ab))
    assert r.fixed == ("ab", "x", "ab", "y")


def test_extra_type_alignment():
    def up(t):
        return t.upper()

    up.pattern = r"[a-z]+"
    r = parse.parse("{:>5up}", "   ab", dict(up=up))
    assert r.fixed == ("AB",)


def test_extra_type_used_in_search_findall():
    def up(t):
        return t.upper()

    up.pattern = r"[a-z]+"
    assert search("<{:up}>", "x <ab> y", extra_types=dict(up=up)).fixed == ("AB",)
    res = [r.fixed for r in findall("<{:up}>", "<a><b>", extra_types=dict(up=up))]
    assert res == [("A",), ("B",)]


def test_convert_first_picklable():
    c = pickle.loads(pickle.dumps(parse.convert_first(int)))
    assert c("5", None) == 5


# ---------- search / findall ----------

def test_search_basic():
    r = search("age: {:d}", "name: bob, age: 42, ok")
    assert r.fixed == (42,)
    assert r.spans[0] == (16, 18)


def test_search_none():
    assert search("age: {:d}", "nothing here") is None


def test_search_pos_endpos():
    s = "1 2 3"
    assert search("{:d}", s, pos=2).fixed == (2,)
    assert search("{:d}", s, pos=4).fixed == (3,)
    assert search("{:d}", s, endpos=1).fixed == (1,)
    assert search("{:d}", s, pos=1, endpos=2) is None
    assert search("{:d}", s, pos=5) is None


def test_findall():
    res = list(findall("{:d}", "1 22 333"))
    assert [r.fixed[0] for r in res] == [1, 22, 333]


def test_findall_none():
    assert list(findall("{:d}", "abc")) == []


def test_findall_pos_endpos():
    s = "1 2 3 4"
    assert [r[0] for r in findall("{:d}", s, pos=2)] == [2, 3, 4]
    assert [r[0] for r in findall("{:d}", s, endpos=5)] == [1, 2, 3]
    assert [r[0] for r in findall("{:d}", s, pos=2, endpos=5)] == [2, 3]


def test_findall_named():
    res = findall("<{k}={v}>", "<a=1><b=2>")
    assert [r.named for r in res] == [{"k": "a", "v": "1"}, {"k": "b", "v": "2"}]


def test_findall_is_iterator():
    it = findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it).fixed == (1,)
    assert it.next().fixed == (2,) if hasattr(it, "next") else True
    with pytest.raises(StopIteration):
        next(it)


def test_findall_case():
    assert len(list(findall("A{:d}", "a1 A2"))) == 2
    assert len(list(findall("A{:d}", "a1 A2", case_sensitive=True))) == 1


def test_evaluate_result_false():
    m = parse.parse("{:d}", "12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (12,)
    m = search("{:d}", "x12", evaluate_result=False)
    assert m.evaluate_result().fixed == (12,)
    ms = list(findall("{:d}", "1 2", evaluate_result=False))
    assert [x.evaluate_result().fixed[0] for x in ms] == [1, 2]
    assert parse.parse("{:d}", "x", evaluate_result=False) is None


def test_parser_reuse():
    p = compile("{:d}")
    assert p.parse("1").fixed == (1,)
    assert p.parse("2").fixed == (2,)
    assert p.search("a 3").fixed == (3,)
    assert [r[0] for r in p.findall("4 5")] == [4, 5]
    assert p.parse("x") is None


# ---------- datetimes: strptime-style ----------

def test_strptime_datetime():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-11-21 13:23:27")
    assert r[0] == datetime(2023, 11, 21, 13, 23, 27)


def test_strptime_date_only():
    r = parse.parse("{:%Y-%m-%d}", "2023-11-21")
    assert r[0] == date(2023, 11, 21)
    assert not isinstance(r[0], datetime)


def test_strptime_time_only():
    r = parse.parse("{:%H:%M}", "13:23")
    assert r[0] == time(13, 23)


def test_strptime_time_with_seconds_micro():
    r = parse.parse("{:%H:%M:%S.%f}", "01:02:03.5")
    assert r[0] == time(1, 2, 3, 500000)


def test_strptime_tz():
    r = parse.parse("{:%Y-%m-%d %H:%M:%S %z}", "2023-11-21 13:23:27 +0530")
    assert r[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S %z}", "2023-11-21 13:23:27 -0800")
    assert r[0].utcoffset() == timedelta(hours=-8)


def test_strptime_names():
    r = parse.parse("{:%A %d %B %Y}", "Tuesday 21 November 2023")
    assert r[0] == date(2023, 11, 21)
    r = parse.parse("{:%a %d %b %Y}", "Tue 21 Nov 2023")
    assert r[0] == date(2023, 11, 21)


def test_strptime_ampm():
    r = parse.parse("{:%I:%M %p}", "01:30 PM")
    assert r[0] == time(13, 30)


def test_strptime_year_default():
    r = parse.parse("{:%m-%d}", "03-04")
    assert r[0] == date(datetime.today().year, 3, 4)
    r = parse.parse("{:%y-%m-%d}", "23-03-04")
    assert r[0] == date(2023, 3, 4)


def test_strptime_day_of_year():
    r = parse.parse("{:%Y %j}", "2023 032")
    assert r[0] == date(2023, 2, 1)


def test_strptime_nonmatch():
    assert parse.parse("{:%Y-%m-%d}", "23-11-21") is None
    assert parse.parse("{:%H:%M}", "1:2") is None  # %M needs 2 digits


def test_strptime_in_text():
    r = parse.parse("on {:%Y-%m-%d} at {:%H:%M}", "on 2020-01-02 at 03:04")
    assert r.fixed == (date(2020, 1, 2), time(3, 4))


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("x") == "x"


# ---------- datetimes: parse-specific t* types ----------

def test_ti():
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 +0100")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(60, "+0100"))
    assert parse.parse("{:ti}", "2011-02-03T04:05:06Z")[0] == datetime(
        2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(0, "UTC")
    )
    assert parse.parse("{:ti}", "2011-02-03")[0] == datetime(2011, 2, 3)
    assert parse.parse("{:ti}", "2011-02-03 04:05")[0] == datetime(2011, 2, 3, 4, 5)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.789")
    assert r[0] == datetime(2011, 2, 3, 4, 5, 6, 789000)
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.000789")
    assert r[0].microsecond == 789
    r = parse.parse("{:ti}", "2011-02-03 04:05:06.1234567")
    assert r[0].microsecond == 123456
    r = parse.parse("{:ti}", "2011-02-03 04:05:06 -05:30")
    assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)


def test_ti_followed_by_fields():
    r = parse.parse("{:ti}|{}|{:d}", "2011-02-03 04:05:06 +0100|a|5")
    assert r.fixed[1:] == ("a", 5)


def test_tg():
    assert parse.parse("{:tg}", "21/11/2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:tg}", "21-11-2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:tg}", "21/Nov/2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:tg}", "21/November/2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:tg}", "1/2/2023 4:05")[0] == datetime(2023, 2, 1, 4, 5)
    assert parse.parse("{:tg}", "1/2/2023 4:05 PM")[0] == datetime(2023, 2, 1, 16, 5)
    assert parse.parse("{:tg}", "1/2/2023 12:05 AM")[0] == datetime(2023, 2, 1, 0, 5)
    assert parse.parse("{:tg}", "1/2/2023 12:05 PM")[0] == datetime(2023, 2, 1, 12, 5)
    assert parse.parse("{:tg}", "1/2/2023 11:05 AM")[0] == datetime(2023, 2, 1, 11, 5)
    r = parse.parse("{:tg}", "1/2/2023 4:05:06 PM +0200")
    assert r[0] == datetime(2023, 2, 1, 16, 5, 6, tzinfo=FixedTzOffset(120, "+0200"))


def test_ta():
    assert parse.parse("{:ta}", "11/21/2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:ta}", "11-21-2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:ta}", "Nov/21/2023")[0] == datetime(2023, 11, 21)
    assert parse.parse("{:ta}", "11/21/2023 4:05:06 PM")[0] == datetime(
        2023, 11, 21, 16, 5, 6
    )
    assert parse.parse("{:ta}", "11/21/2023 12:00 AM")[0] == datetime(2023, 11, 21, 0, 0)
    r = parse.parse("{:ta}", "11/21/2023 4:05 -0530")
    assert r[0].utcoffset() == -timedelta(hours=5, minutes=30)


def test_te():
    r = parse.parse("{:te}", "Mon, 21 Nov 2023 13:14:15 +0100")
    assert r[0] == datetime(2023, 11, 21, 13, 14, 15, tzinfo=FixedTzOffset(60, "+0100"))
    r = parse.parse("{:te}", "21 Nov 2023 13:14:15 -0100")
    assert r[0].utcoffset() == timedelta(hours=-1)
    assert parse.parse("{:te}", "21 Nov 2023") is None


def test_th():
    r = parse.parse("{:th}", "21/Nov/2023:13:14:15 +0100")
    assert r[0] == datetime(2023, 11, 21, 13, 14, 15, tzinfo=FixedTzOffset(60, "+0100"))
    r = parse.parse("{:th}", "21-Nov-2023:13:14:15 -0000")
    assert r[0].day == 21


def test_tc():
    r = parse.parse("{:tc}", "Mon Nov  6 13:14:15 2023")
    assert r[0] == datetime(2023, 11, 6, 13, 14, 15)
    r = parse.parse("{:tc}", "Tue Nov 21 01:02:03 1999")
    assert r[0] == datetime(1999, 11, 21, 1, 2, 3)


def test_tt():
    assert parse.parse("{:tt}", "13:14")[0] == time(13, 14)
    assert parse.parse("{:tt}", "13:14:15")[0] == time(13, 14, 15)
    assert parse.parse("{:tt}", "1:02 PM")[0] == time(13, 2)
    assert parse.parse("{:tt}", "12:02 AM")[0] == time(0, 2)
    assert parse.parse("{:tt}", "12:02 PM")[0] == time(12, 2)
    r = parse.parse("{:tt}", "13:14:15 +0100")
    assert r[0] == time(13, 14, 15, tzinfo=FixedTzOffset(60, "+0100"))
    r = parse.parse("{:tt}", "13:14:15.5")
    assert r[0] == time(13, 14, 15, 500000)


def test_ts():
    r = parse.parse("{:ts}", "Nov 21 13:14:15")
    assert r[0] == datetime(datetime.today().year, 11, 21, 13, 14, 15)


def test_t_types_with_trailing_field():
    r = parse.parse("{:tg} {}", "1/2/2023 xyz")
    assert r[0] == datetime(2023, 2, 1) and r[1] == "xyz"
    r = parse.parse("{:th}|{:d}", "21/Nov/2023:13:14:15 +0100|7")
    assert r[1] == 7
    r = parse.parse("{:te}|{:d}", "Mon, 21 Nov 2023 13:14:15 +0100|7")
    assert r[1] == 7
    r = parse.parse("{:tc}|{:d}", "Mon Nov  6 13:14:15 2023|7")
    assert r[1] == 7
    r = parse.parse("{:tt}|{:d}", "13:14:15 +0100|7")
    assert r[1] == 7
    r = parse.parse("{:ts}|{:d}", "Nov 21 13:14:15|7")
    assert r[1] == 7
    r = parse.parse("{:ta}|{:d}", "11/21/2023 4:05 PM -0500|7")
    assert r[1] == 7
    r = parse.parse("{:tg}|{:d}", "21/11/2023 4:05 PM -0500|7")
    assert r[1] == 7


def test_t_types_named():
    r = parse.parse("{when:ti} {what}", "2011-02-03 04:05:06 +0100 hello")
    assert r["when"].year == 2011
    assert r["what"] == "hello"


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "x")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "x"
    assert tz.dst(None) == timedelta(0)
    assert tz == FixedTzOffset(90, "x")
    assert tz != FixedTzOffset(91, "x")
    assert tz != FixedTzOffset(90, "y")
    assert tz != 5
    assert repr(tz) == "<FixedTzOffset x 1:30:00>"


# ---------- extract_format ----------

def test_extract_format():
    f = parse.extract_format("*^10.3d", {})
    assert f["fill"] == "*" and f["align"] == "^"
    assert f["width"] == "10" and f["precision"] == "3" and f["type"] == "d"
    f = parse.extract_format(">5", {})
    assert f["align"] == ">" and f["fill"] is None and f["width"] == "5" and f["type"] == ""
    f = parse.extract_format("05d", {})
    assert f["zero"] is True and f["width"] == "5"
    f = parse.extract_format("+d", {})
    assert f["type"] == "d"
    f = parse.extract_format(",d", {})
    assert f["grouping"] == "," and f["type"] == "d"
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    f = parse.extract_format("<d", {})
    assert f["align"] == "<" and f["fill"] is None
    f = parse.extract_format("=d", {})
    assert f["align"] == "="
    f = parse.extract_format("x=d", {})
    assert f["fill"] == "x" and f["align"] == "="
    f = parse.extract_format("custom", {"custom": None})
    assert f["type"] == "custom"
    with pytest.raises(ValueError):
        parse.extract_format("Q", {})


def test_zero_flag_parses():
    assert parse.parse("{:05d}", "00042").fixed == (42,)


def test_strptime_type_in_extract():
    f = parse.extract_format("%Y-%m", {})
    assert f["type"] == "%Y-%m"


# ---------- misc ----------

def test_date_convert_direct():
    class M:
        def __init__(self, g):
            self._g = g

        def groups(self):
            return self._g

    d = parse.date_convert("", M(("2020-01-02",)), ymd=0)
    assert d == datetime(2020, 1, 2)
    d = parse.date_convert("", M(("01/02/2020",)), mdy=0)
    assert d == datetime(2020, 1, 2)
    d = parse.date_convert("", M(("01/02/2020",)), dmy=0)
    assert d == datetime(2020, 2, 1)


def test_percentage_function():
    assert parse.percentage("25%", None) == 0.25


def test_months_map():
    assert parse.MONTHS_MAP["Jan"] == 1 and parse.MONTHS_MAP["December"] == 12
    assert len(parse.MONTHS_MAP) == 23
    for i, n in enumerate(
        "January February March April May June July August September October November December".split(),
        1,
    ):
        assert parse.MONTHS_MAP[n] == i
        assert parse.MONTHS_MAP[n[:3]] == i


def test_public_api():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}


def test_literal_text_with_digits_and_types():
    r = parse.parse("v{:d}.{:d}.{:d}", "v1.22.3")
    assert r.fixed == (1, 22, 3)


def test_csv_like():
    r = parse.parse("{a},{b},{c:d}", "x,y,3")
    assert r.named == {"a": "x", "b": "y", "c": 3}


def test_unicode():
    assert parse.parse("héllo {}", "héllo wörld").fixed == ("wörld",)
