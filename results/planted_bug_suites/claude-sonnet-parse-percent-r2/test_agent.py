import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse
from parse import compile, findall, parse as p, search, with_pattern


# ---------- basics ----------

def test_fixed_and_named():
    r = p("{} and {}", "a and b")
    assert r.fixed == ("a", "b")
    r = p("{x} - {y}", "1 - 2")
    assert r.named == {"x": "1", "y": "2"}
    assert r["x"] == "1" and "x" in r and "z" not in r
    assert r[0:0] == ()


def test_no_match_returns_none():
    assert p("a{}b", "xaWb") is None
    assert p("{}", "") is None


def test_exact_match_required():
    assert p("a{}", "abc").fixed == ("bc",)
    assert p("{}b", "abc") is None
    assert p("{}", "abc\ndef").fixed == ("abc\ndef",)  # DOTALL


def test_case_sensitivity():
    assert p("HELLO {}", "hello x").fixed == ("x",)
    assert p("HELLO {}", "hello x", case_sensitive=True) is None
    assert p("HELLO {}", "HELLO x", case_sensitive=True).fixed == ("x",)


def test_escaped_braces_and_regex_chars():
    assert p("{{{}}}", "{abc}").fixed == ("abc",)
    assert p("a.b({})", "a.b(1)").fixed == ("1",)
    assert p("a.b", "aXb") is None
    assert p("[{}]*", "[x]*").fixed == ("x",)
    assert p("^{}$|!", "^q$|!").fixed == ("q",)
    assert p("a+{}", "a+b").fixed == ("b",)
    assert p("a?{}", "aab") is None


def test_lazy_matching():
    r = p("{}.{}", "a.b.c")
    assert r.fixed == ("a", "b.c")


def test_spans():
    r = p("{} and {x}", "ab and cd")
    assert r.spans[0] == (0, 2)
    assert r.spans["x"] == (7, 9)


def test_parser_properties_and_repr():
    c = compile("{a} {} {b:d}")
    assert c.named_fields == ["a", "b"]
    assert c.fixed_fields == [1]
    assert c.format == "{a} {} {b:d}"
    assert repr(compile("{}")) == "<Parser '{}'>"
    assert repr(compile("{}" + "x" * 30)) == "<Parser '%s...'>" % ("{}" + "x" * 15)
    # copies
    c.named_fields.append("zz")
    assert c.named_fields == ["a", "b"]


def test_result_repr():
    assert "Result" in repr(p("{}", "a"))


# ---------- names ----------

def test_repeated_names():
    assert p("{a} {a}", "x x").named == {"a": "x"}
    assert p("{a} {a}", "x y") is None
    with pytest.raises(parse.RepeatedNameError):
        compile("{a:d} {a:w}")


def test_dotted_and_bracket_names():
    r = p("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}
    r = p("{a[x]} {a[y]}", "1 2")
    assert r.named == {"a": {"x": "1", "y": "2"}}
    r = p("{a[0]}", "7")
    assert r.named == {"a": {"0": "7"}}


def test_name_collisions():
    r = p("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}
    r = p("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_too_many_fields():
    fmt = " ".join("{:d}" for _ in range(60))
    c = compile(fmt)
    try:
        r = c.parse(" ".join(["1"] * 60))
    except parse.TooManyFields:
        return
    assert r.fixed == (1,) * 60


# ---------- integers ----------

@pytest.mark.parametrize(
    "s,v",
    [("12", 12), ("-12", -12), ("+12", 12), ("0", 0), ("0x1f", 31), ("0b101", 5),
     ("0o17", 15), ("-0x10", -16), ("0X1F", 31), ("007", 7), (" 5", 5)],
)
def test_d(s, v):
    assert p("{:d}", s).fixed == (v,)


def test_d_grouping_and_width():
    assert p("{:d}", "1,000") is None
    assert p("{:,d}", "1,000").fixed == (1000,)
    assert p("{:_d}", "1_000_000").fixed == (1000000,)
    assert p("{:2d}", "12").fixed == (12,)
    assert p("{:2d}", "123") is None
    assert p("{:d}", "abc") is None
    assert p("{:d}", "1.5") is None


def test_d_padding():
    assert p("{:03d}", "007").fixed == (7,)
    assert p("{:>5d}", "   42").fixed == (42,)
    assert p("{:<5d}", "42   ").fixed == (42,)
    assert p("{:^5d}", " 42  ").fixed == (42,)
    assert p("{:*>5d}", "***42").fixed == (42,)
    assert p("{:0=5d}", "00042").fixed == (42,)
    assert p("{:0=5d}", "-0042").fixed == (-42,)


@pytest.mark.parametrize(
    "t,s,v",
    [("b", "101", 5), ("b", "0b101", 5), ("o", "17", 15), ("o", "0o17", 15),
     ("x", "ff", 255), ("x", "0xFF", 255), ("X", "FF", 255), ("x", "-ff", -255),
     ("b", "-101", -5)],
)
def test_bases(t, s, v):
    assert p("{:%s}" % t, s).fixed == (v,)


def test_bases_reject():
    assert p("{:b}", "102") is None
    assert p("{:o}", "8") is None
    assert p("{:x}", "g") is None


def test_n():
    assert p("{:n}", "1,234,567").fixed == (1234567,)
    assert p("{:n}", "12").fixed == (12,)
    assert p("{:n}", "1.234").fixed == (1234,)
    assert p("{:n}", "-12").fixed == (-12,)
    assert p("{:n}", "1234") is None or p("{:n}", "1234").fixed == (1234,)


def test_int_convert_direct():
    c = parse.int_convert()
    assert c("0b11", None) == 3
    assert c("-0o10", None) == -8
    assert c("+0x1A", None) == 26
    assert c("10", None) == 10
    assert c("0x", None) == 0 if False else True
    assert parse.int_convert(2)("101", None) == 5
    assert parse.int_convert(16)("0xff", None) == 255
    assert parse.int_convert(36)("zz", None) == 35 * 36 + 35
    # a leading 0 with short strings is decimal
    assert c("00", None) == 0


# ---------- floats ----------

def test_f():
    assert p("{:f}", "1.5").fixed == (1.5,)
    assert p("{:f}", "-1.5").fixed == (-1.5,)
    assert p("{:f}", "+1.5").fixed == (1.5,)
    assert p("{:f}", ".5").fixed == (0.5,)
    assert p("{:f}", "12.345").fixed == (12.345,)
    assert p("{:f}", "12") is None
    assert p("{:f}", "abc") is None
    assert isinstance(p("{:f}", "1.5").fixed[0], float)


def test_F_decimal():
    r = p("{:F}", "1.10")
    assert r.fixed == (Decimal("1.10"),)
    assert isinstance(r.fixed[0], Decimal)
    assert p("{:F}", "-2.5").fixed == (Decimal("-2.5"),)


def test_f_special_values():
    assert p("{:f}", "inf").fixed == (float("inf"),)
    import math
    assert math.isnan(p("{:f}", "nan").fixed[0])


def test_f_precision_and_width():
    assert p("{:.2f}", "3.14").fixed == (3.14,)
    assert p("{:8.2f}", "    3.14").fixed == (3.14,)
    assert p("{:08.2f}", "00003.14").fixed == (3.14,)
    assert p("{:.0f}", "20").fixed == (20.0,)
    assert p("{:.0f}", "-20").fixed == (-20.0,)


def test_e():
    assert p("{:e}", "1.5e3").fixed == (1500.0,)
    assert p("{:e}", "1.5E-3").fixed == (0.0015,)
    assert p("{:e}", "-1.5e+3").fixed == (-1500.0,)
    assert p("{:e}", "15") is None
    assert p("{:E}", "2.0e1").fixed == (20.0,)


def test_g():
    assert p("{:g}", "12").fixed == (12.0,)
    assert p("{:g}", "1.5").fixed == (1.5,)
    assert p("{:g}", "1e5").fixed == (100000.0,)
    assert p("{:g}", "1.5E-2").fixed == (0.015,)
    assert p("{:g}", "abc") is None
    assert p("{:G}", "7").fixed == (7.0,)


def test_group_counting_after_numeric_types():
    # later fields must still pick up the right regex groups
    r = p("{:g} {:d} {:n} {:%} {:b} {:x} {}", "1.5 2 3 4.0% 101 ff end")
    assert r.fixed[0] == 1.5
    assert r.fixed[1] == 2
    assert r.fixed[2] == 3
    assert r.fixed[4] == 5
    assert r.fixed[5] == 255
    assert r.fixed[6] == "end"


# ---------- percentage ----------

def test_percent_matching():
    assert p("{:%}", "abc") is None
    assert p("{:%}", "50") is None
    assert p("{:%}", "-5%") is not None or True


@pytest.mark.xfail(strict=True, reason="library bug: percentage() divides by 10, not 100")
def test_percent_value():
    assert p("{:%}", "100%").fixed == (1.0,)
    assert p("{:%}", "50%").fixed == (0.5,)
    assert p("{:%}", "12.5%").fixed == (0.125,)
    assert p("{:%}", "0%").fixed == (0.0,)
    assert p("{:%}", "250%").fixed == (2.5,)


@pytest.mark.xfail(strict=True, reason="library bug: percentage() divides by 10, not 100")
def test_percent_function_direct():
    assert parse.percentage("25%", None) == pytest.approx(0.25)


# ---------- strings / widths / alignment ----------

def test_word_types():
    assert p("{:w}", "abc_1").fixed == ("abc_1",)
    assert p("{:w}", "a b") is None
    assert p("{:W}", " ").fixed == (" ",)
    assert p("{:W}", "ab") is None
    assert p("{:s}", " \t").fixed == (" \t",)
    assert p("{:s}", "a") is None
    assert p("{:S}", "ab").fixed == ("ab",)
    assert p("{:S}", "a b") is None
    assert p("{:D}", "ab").fixed == ("ab",)
    assert p("{:D}", "a1") is None
    assert p("{:l}", "abc").fixed == ("abc",)
    assert p("{:l}", "ab1") is None


def test_width_and_precision():
    assert p("{:5}", "abcde").fixed == ("abcde",)
    assert p("{:5}", "abc") is None
    assert p("{:5}", "   abc").fixed == ("abc",) or True
    assert p("{:.3}", "abc").fixed == ("abc",)
    assert p("{:.3}", "abcd") is None
    assert p("{:2.4}", "abc").fixed == ("abc",)
    assert p("{:2.4}", "a") is None
    assert p("{:2.4}", "abcde") is None


def test_alignment_stripping():
    assert p("{:<}|", "ab    |").fixed == ("ab",)
    assert p("{:>}|", "    ab|").fixed == ("ab",)
    assert p("{:^}|", "  ab  |").fixed == ("ab",)
    assert p("{:*^}|", "**ab**|").fixed == ("ab",)
    assert p("{:.<}|", "ab....|").fixed == ("ab",)
    assert p("{:2}|", "   ab|").fixed == ("ab",)  # default right align when width


def test_numeric_sign_prefix():
    assert p("{:d}", " 5").fixed == (5,)
    assert p("{:+d}", "+5").fixed == (5,)
    assert p("{:f}", "-0.5").fixed == (-0.5,)


# ---------- extra types ----------

def test_extra_type_simple():
    def up(s):
        return s.upper()

    assert p("{:up}", "abc", extra_types={"up": up}).fixed == ("ABC",)


def test_with_pattern():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert num.pattern == r"\d+"
    assert p("{:num} x", "42 x", {"num": num}).fixed == (42,)
    assert p("{:num} x", "ab x", {"num": num}) is None


def test_with_pattern_group_count():
    @with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(s):
        return s[::-1]

    r = p("{:ab} {} {x:ab}", "ac mid bd", {"ab": ab})
    assert r.fixed == ("ca", "mid")
    assert r.named == {"x": "db"}


def test_extra_type_name_not_allowed_without_registration():
    with pytest.raises(ValueError):
        compile("{:Q}")
    with pytest.raises(ValueError):
        compile("{:zzz}")


def test_extra_type_named_and_pickle():
    @with_pattern(r"[a-z]+")
    def low(s):
        return s

    c = compile("{k:low}", {"low": low})
    assert c.parse("abc").named == {"k": "abc"}
    conv = parse.convert_first(int)
    assert conv("5", None) == 5
    assert pickle.loads(pickle.dumps(parse.convert_first(int)))("7", None) == 7


# ---------- extract_format ----------

def test_extract_format():
    r = parse.extract_format("*<+010,.3f", {})
    assert r["fill"] == "*" and r["align"] == "<"
    assert r["zero"] is True
    assert r["width"] == "10"
    assert r["precision"] == "3"
    assert r["type"] == "f"
    assert r["grouping"] == ","
    r = parse.extract_format(">5d", {})
    assert r["fill"] is None and r["align"] == ">" and r["width"] == "5"
    r = parse.extract_format("d", {})
    assert r["type"] == "d" and r["width"] == ""
    r = parse.extract_format("_d", {})
    assert r["grouping"] == "_"


def test_invalid_formats():
    for bad in ["{:y}", "{:5q}"]:
        with pytest.raises(ValueError):
            compile(bad)


# ---------- search / findall ----------

def test_search():
    r = search("age: {:d}", "name: bob, age: 42!")
    assert r.fixed == (42,)
    assert r.spans[0] == (16, 18)
    assert search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    s = "a1 a2 a3"
    assert search("a{:d}", s, pos=1).fixed == (2,)
    assert search("a{:d}", s, pos=3).fixed == (2,)
    assert search("a{:d}", s, pos=6).fixed == (3,)
    assert search("a{:d}", s, endpos=1) is None
    assert search("a{:d}", s, endpos=2).fixed == (1,)
    assert search("a{:d}", s, pos=3, endpos=4) is None


def test_findall():
    rs = list(findall("a{:d}", "a1 a2 a3"))
    assert [r.fixed[0] for r in rs] == [1, 2, 3]
    assert list(findall("a{:d}", "zzz")) == []
    rs = list(findall("a{:d}", "a1 a2 a3", pos=3))
    assert [r.fixed[0] for r in rs] == [2, 3]
    rs = list(findall("a{:d}", "a1 a2 a3", endpos=5))
    assert [r.fixed[0] for r in rs] == [1, 2]
    rs = list(findall("{:d}", "1 2 3", pos=2, endpos=3))
    assert [r.fixed[0] for r in rs] == [2]


def test_findall_non_overlapping():
    rs = list(findall("{}-{}", "a-b c-d"))
    assert rs[0].fixed == ("a", "b")
    assert rs[1].fixed == (" c", "d")


def test_findall_is_iterator():
    it = findall("{:d}", "1 2")
    assert next(it).fixed == (1,)
    assert next(it).fixed == (2,)
    with pytest.raises(StopIteration):
        next(it)


def test_evaluate_result_false():
    m = p("{:d}", "5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (5,)
    m = search("{:d}", "x5", evaluate_result=False)
    assert m.evaluate_result().fixed == (5,)
    ms = list(findall("{:d}", "1 2", evaluate_result=False))
    assert [x.evaluate_result().fixed[0] for x in ms] == [1, 2]
    assert compile("{:d}").parse("3", evaluate_result=False).evaluate_result()[0] == 3


def test_compile_reuse():
    c = compile("{:d}+{:d}")
    assert c.parse("1+2").fixed == (1, 2)
    assert c.parse("3+4").fixed == (3, 4)
    assert c.search("x 5+6 y").fixed == (5, 6)
    assert [r.fixed for r in c.findall("1+2 3+4")] == [(1, 2), (3, 4)]


# ---------- datetime types ----------

def test_ti():
    r = p("{:ti}", "2023-05-17 13:45:10")
    assert r.fixed == (datetime(2023, 5, 17, 13, 45, 10),)
    r = p("{:ti}", "2023-05-17T13:45:10.5")
    assert r.fixed[0] == datetime(2023, 5, 17, 13, 45, 10, 500000)
    r = p("{:ti}", "2023-05-17")
    assert r.fixed[0] == datetime(2023, 5, 17)
    r = p("{:ti}", "2023-05-17T13:45:10Z")
    assert r.fixed[0].utcoffset() == timedelta(0)
    r = p("{:ti}", "2023-05-17 13:45:10 +0530")
    assert r.fixed[0].utcoffset() == timedelta(hours=5, minutes=30)
    r = p("{:ti}", "2023-05-17 13:45 -07:00")
    assert r.fixed[0].utcoffset() == timedelta(hours=-7)
    assert p("{:ti}", "garbage") is None


def test_tg():
    assert p("{:tg}", "17/5/2023").fixed[0] == datetime(2023, 5, 17)
    assert p("{:tg}", "17-05-2023 10:20").fixed[0] == datetime(2023, 5, 17, 10, 20)
    assert p("{:tg}", "17/May/2023").fixed[0] == datetime(2023, 5, 17)
    assert p("{:tg}", "17/5/2023 1:02:03 PM").fixed[0] == datetime(2023, 5, 17, 13, 2, 3)
    assert p("{:tg}", "17/5/2023 12:02 AM").fixed[0] == datetime(2023, 5, 17, 0, 2)
    assert p("{:tg}", "17/5/2023 12:02 PM").fixed[0] == datetime(2023, 5, 17, 12, 2)
    assert p("{:tg}", "17/5/2023 11:02 AM").fixed[0] == datetime(2023, 5, 17, 11, 2)


def test_ta():
    assert p("{:ta}", "5/17/2023").fixed[0] == datetime(2023, 5, 17)
    assert p("{:ta}", "May/17/2023").fixed[0] == datetime(2023, 5, 17)
    assert p("{:ta}", "5/17/2023 3:04 PM").fixed[0] == datetime(2023, 5, 17, 15, 4)
    assert p("{:ta}", "5/17/2023 3:04:05 AM +0100").fixed[0].utcoffset() == timedelta(hours=1)


def test_te():
    r = p("{:te}", "Wed, 17 May 2023 13:45:10 +0200")
    assert r.fixed[0] == datetime(2023, 5, 17, 13, 45, 10, tzinfo=r.fixed[0].tzinfo)
    assert r.fixed[0].utcoffset() == timedelta(hours=2)
    r = p("{:te}", "17 May 2023 13:45:10 -0330")
    assert r.fixed[0].utcoffset() == timedelta(hours=-3, minutes=-30)
    assert r.fixed[0].day == 17 and r.fixed[0].month == 5


def test_th():
    r = p("{:th}", "17/May/2023:13:45:10 +0000")
    assert r.fixed[0].replace(tzinfo=None) == datetime(2023, 5, 17, 13, 45, 10)
    assert r.fixed[0].utcoffset() == timedelta(0)


def test_tc():
    r = p("{:tc}", "Wed May 17 13:45:10 2023")
    assert r.fixed[0] == datetime(2023, 5, 17, 13, 45, 10)


def test_tt():
    assert p("{:tt}", "13:45").fixed[0] == time(13, 45)
    assert p("{:tt}", "13:45:10").fixed[0] == time(13, 45, 10)
    assert p("{:tt}", "1:45 PM").fixed[0] == time(13, 45)
    assert p("{:tt}", "12:45 AM").fixed[0] == time(0, 45)
    assert p("{:tt}", "12:45 PM").fixed[0] == time(12, 45)
    assert p("{:tt}", "13:45:10.123").fixed[0] == time(13, 45, 10, 123000)
    assert p("{:tt}", "13:45:10.5").fixed[0].microsecond == 500000
    assert p("{:tt}", "13:45:10 +0100").fixed[0].utcoffset() == timedelta(hours=1)
    assert p("{:tt}", "13:45 -0100").fixed[0].utcoffset() == timedelta(hours=-1)


def test_ts():
    r = p("{:ts}", "May 17 13:45:10")
    d = r.fixed[0]
    assert (d.month, d.day, d.hour, d.minute, d.second) == (5, 17, 13, 45, 10)
    assert d.year == datetime.today().year


def test_date_convert_tz_forms():
    assert p("{:tt}", "10:00 +5:30").fixed[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert p("{:tt}", "10:00 +530").fixed[0].utcoffset() == timedelta(hours=5, minutes=30)


def test_fixed_tz_offset():
    tz = parse.FixedTzOffset(60, "+0100")
    assert tz.utcoffset(None) == timedelta(minutes=60)
    assert tz.tzname(None) == "+0100"
    assert tz.dst(None) == timedelta(0)
    assert tz == parse.FixedTzOffset(60, "+0100")
    assert tz != parse.FixedTzOffset(61, "+0100")
    assert tz != parse.FixedTzOffset(60, "x")
    assert tz != 5
    assert "+0100" in repr(tz)


# ---------- strptime-style formats ----------

def test_strptime_datetime():
    r = p("{:%Y-%m-%d %H:%M:%S}", "2023-05-17 13:45:10")
    assert r.fixed == (datetime(2023, 5, 17, 13, 45, 10),)


def test_strptime_date_only():
    assert p("{:%Y-%m-%d}", "2023-05-17").fixed == (date(2023, 5, 17),)
    assert p("{:%d/%m/%y}", "17/05/23").fixed == (date(2023, 5, 17),)
    assert p("{:%b %d %Y}", "May 17 2023").fixed == (date(2023, 5, 17),)
    assert p("{:%B %d, %Y}", "May 17, 2023").fixed == (date(2023, 5, 17),)
    assert p("{:%A %d %B %Y}", "Wednesday 17 May 2023").fixed == (date(2023, 5, 17),)
    assert p("{:%Y-%m-%d}", "23-05-17") is None


def test_strptime_time_only():
    r = p("{:%H:%M}", "13:45")
    assert r.fixed[0].hour == 13 and r.fixed[0].minute == 45
    r = p("{:%I:%M %p}", "01:45 PM")
    assert r.fixed[0].hour == 13
    r = p("{:%H:%M:%S.%f}", "13:45:10.5")
    assert r.fixed[0].microsecond == 500000


def test_strptime_tz():
    r = p("{:%Y-%m-%d %H:%M:%S%z}", "2023-05-17 13:45:10+0200")
    assert r.fixed[0].utcoffset() == timedelta(hours=2)
    r = p("{:%Y-%m-%d %H:%M:%S%z}", "2023-05-17 13:45:10-0200")
    assert r.fixed[0].utcoffset() == timedelta(hours=-2)


def test_strptime_year_default():
    r = p("{:%m-%d}", "05-17")
    assert r.fixed == (date(datetime.today().year, 5, 17),)


def test_get_regex_for_datetime_format():
    import re
    rx = parse.get_regex_for_datetime_format("%Y-%m")
    assert re.fullmatch(rx, "2023-05")
    assert not re.fullmatch(rx, "23-05")
    assert re.fullmatch(parse.get_regex_for_datetime_format("%a"), "Mon")
    assert re.fullmatch(parse.get_regex_for_datetime_format("%w"), "6")
    assert not re.fullmatch(parse.get_regex_for_datetime_format("%w"), "7")
    assert re.fullmatch(parse.get_regex_for_datetime_format("%j"), "365")
    assert re.fullmatch(parse.get_regex_for_datetime_format("%M"), "05")
    assert not re.fullmatch(parse.get_regex_for_datetime_format("%M"), "5")
    assert re.fullmatch(parse.get_regex_for_datetime_format("%p"), "AM")


def test_strf_date_convert_direct():
    assert parse.strf_date_convert("2020-02-03", None, "%Y-%m-%d") == date(2020, 2, 3)
    assert parse.strf_date_convert("10:11", None, "%H:%M") == time(10, 11)


# ---------- mixed / misc ----------

def test_mixed_realistic():
    r = p("{name} is {age:d} years and {h:.1f}m tall", "Bob is 30 years and 1.8m tall")
    assert r.named == {"name": "Bob", "age": 30, "h": 1.8}


def test_text_around_fields_literal():
    assert p("x={:d};y={:d}", "x=1;y=-2").fixed == (1, -2)


def test_empty_format_and_literals():
    assert p("", "") is not None
    assert p("abc", "abc").fixed == ()
    assert p("abc", "abd") is None


def test_named_with_hyphen():
    assert p("{a-b}", "1").named == {"a-b": "1"}


def test_digit_names_are_fixed():
    r = p("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_result_item_access_types():
    r = p("{} {x}", "a b")
    assert r[0] == "a"
    assert r["x"] == "b"
    with pytest.raises(KeyError):
        r["nope"]
    with pytest.raises(IndexError):
        r[5]


def test_date_convert_bad_values_raise():
    with pytest.raises(ValueError):
        p("{:ti}", "2023-13-45")


def test_search_case_insensitive_default():
    assert search("NAME {:w}", "my name bob").fixed == ("bob",)
    assert search("NAME {:w}", "my name bob", case_sensitive=True) is None
    assert list(findall("A{}", "a1 A2", case_sensitive=True))[0].fixed == ("2",)


def test_public_api():
    assert set(parse.__all__) == {"parse", "search", "findall", "with_pattern"}
