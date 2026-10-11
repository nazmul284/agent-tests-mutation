import pickle
from datetime import date, datetime, time, timedelta, timezone
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
    search,
    with_pattern,
)


# ---------- basics ----------

def test_no_match_returns_none():
    assert parse.parse("hello {}", "goodbye world") is None


def test_fixed_and_named():
    r = parse.parse("{} is {name}", "Bob is cool")
    assert r.fixed == ("Bob", )
    assert r.named == {"name": "cool"}
    assert r[0] == "Bob"
    assert r["name"] == "cool"
    assert "name" in r
    assert "other" not in r


def test_fixed_slice():
    r = parse.parse("{} {} {}", "a b c")
    assert r[1:] == ("b", "c")


def test_exact_match_required():
    assert parse.parse("a{}", "ab c d") is not None
    assert parse.parse("{}b", "abc") is None
    assert parse.parse("b{}", "abc") is None


def test_full_match_anchors_with_newline():
    assert parse.parse("{}x", "ab\n") is None
    r = parse.parse("{}", "a\nb")
    assert r[0] == "a\nb"  # DOTALL


def test_search_and_pos_endpos():
    r = search("age: {:d}", "name: Rufus, age: 42, x")
    assert r[0] == 42
    assert search("age: {:d}", "age: 42", pos=1) is None
    assert search("{:d}", "12 34", pos=2)[0] == 34
    assert search("{:d}", "12 34", endpos=2)[0] == 12
    assert search("{:d}", "ab 34", endpos=2) is None


def test_findall():
    res = list(findall(">{}<", "<p>some <b>bold</b> text</p>"))
    # '>' then lazily up to '<'
    assert [r[0] for r in res] == ["some ", "bold", " text"]
    assert [r[0] for r in findall("{:d}", "1 22 333")] == [1, 22, 333]
    assert [r[0] for r in findall("{:d}", "1 22 333", pos=2)] == [22, 333]
    assert [r[0] for r in findall("{:d}", "1 22 333", endpos=4)] == [1, 22]


def test_findall_evaluate_false():
    ms = list(findall("{:d}", "1 2", evaluate_result=False))
    assert len(ms) == 2
    assert ms[1].evaluate_result()[0] == 2


def test_parse_evaluate_false():
    m = parse.parse("{:d}", "12", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result().fixed == (12,)
    m = search("{:d}", "x12", evaluate_result=False)
    assert m.evaluate_result().fixed == (12,)


# ---------- case sensitivity ----------

@pytest.mark.xfail(
    strict=True,
    reason="library bug: parse() defaults to case_sensitive=True, but the "
    "docstring (and search/findall/compile) default to case-insensitive",
)
def test_parse_case_insensitive_default():
    # documented: default is case insensitive
    assert parse.parse("SPAM {}", "spam spam") is not None
    assert parse.parse("SPAM {}", "spam spam", case_sensitive=True) is None
    assert parse.parse("spam {}", "spam x", case_sensitive=True)[0] == "x"


def test_search_findall_compile_case():
    assert search("SPAM {}", "spam x") is not None
    assert search("SPAM {}", "spam x", case_sensitive=True) is None
    assert len(list(findall("A{:d}", "a1 A2"))) == 2
    assert len(list(findall("A{:d}", "a1 A2", case_sensitive=True))) == 1
    assert compile("SPAM {}").parse("spam x") is not None
    assert compile("SPAM {}", case_sensitive=True).parse("spam x") is None


# ---------- escapes / literal text ----------

def test_double_braces():
    assert parse.parse("{{{}}}", "{abc}")[0] == "abc"
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{", "{") is not None


def test_regex_chars_escaped():
    assert parse.parse("a.b {}", "a.b x")[0] == "x"
    assert parse.parse("a.b {}", "axb x") is None
    for ch in "?\\.[]()*+^$!|":
        r = parse.parse("a" + ch + "{}", "a" + ch + "z")
        assert r is not None and r[0] == "z", ch


def test_empty_format():
    assert parse.parse("", "") is not None
    assert parse.parse("", "a") is None


# ---------- types ----------

def test_int_d():
    assert parse.parse("{:d}", "42")[0] == 42
    assert parse.parse("{:d}", "-42")[0] == -42
    assert parse.parse("{:d}", "+42")[0] == 42
    assert parse.parse("{:d}", " 42")[0] == 42
    assert parse.parse("{:d}", "0x1f")[0] == 31
    assert parse.parse("{:d}", "0b101")[0] == 5
    assert parse.parse("{:d}", "0o17")[0] == 15
    assert parse.parse("{:d}", "-0x10")[0] == -16
    assert parse.parse("{:d}", "007")[0] == 7
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "1.5") is None


def test_int_d_width_and_grouping():
    assert parse.parse("{:3d}", "  7")[0] == 7
    assert parse.parse("{:2d}{:2d}", "1234").fixed == (12, 34)
    assert parse.parse("{:d}", "1,000") is None
    assert parse.parse("{:,d}", "1,000,000")[0] == 1000000
    assert parse.parse("{:_d}", "1_000")[0] == 1000
    assert parse.parse("{:d}", "1_000") is None


def test_int_n():
    assert parse.parse("{:n}", "1,234,567")[0] == 1234567
    assert parse.parse("{:n}", "1.234")[0] == 1234
    assert parse.parse("{:n}", "12")[0] == 12
    assert parse.parse("{:n}", "-1,234")[0] == -1234
    assert parse.parse("{:n}", "1234") is None


def test_bases():
    assert parse.parse("{:b}", "1011")[0] == 11
    assert parse.parse("{:b}", "0b1011")[0] == 11
    assert parse.parse("{:b}", "-0b1011")[0] == -11
    assert parse.parse("{:b}", "12") is None
    assert parse.parse("{:o}", "17")[0] == 15
    assert parse.parse("{:o}", "0o17")[0] == 15
    assert parse.parse("{:o}", "8") is None
    assert parse.parse("{:x}", "ff")[0] == 255
    assert parse.parse("{:x}", "0xFF")[0] == 255
    assert parse.parse("{:X}", "DEADbeef")[0] == 0xDEADBEEF
    assert parse.parse("{:x}", "zz") is None
    assert parse.parse("{:x}", "+a")[0] == 10


def test_float_f():
    assert parse.parse("{:f}", "3.14")[0] == 3.14
    assert parse.parse("{:f}", "-3.14")[0] == -3.14
    assert parse.parse("{:f}", "+3.14")[0] == 3.14
    assert parse.parse("{:f}", ".5")[0] == 0.5
    assert parse.parse("{:f}", "-.5")[0] == -0.5
    assert parse.parse("{:f}", "3") is None
    assert parse.parse("{:f}", "abc") is None
    assert isinstance(parse.parse("{:f}", "1.0")[0], float)


def test_float_f_special():
    assert parse.parse("{:f}", "inf")[0] == float("inf")
    assert parse.parse("{:f}", "nan")[0] != parse.parse("{:f}", "nan")[0]


def test_float_F_decimal():
    v = parse.parse("{:F}", "3.14")[0]
    assert isinstance(v, Decimal)
    assert v == Decimal("3.14")
    assert parse.parse("{:F}", "-3.14")[0] == Decimal("-3.14")


def test_float_precision_zero():
    assert parse.parse("{:.0f}", "20")[0] == 20.0


def test_float_e():
    assert parse.parse("{:e}", "1.5e3")[0] == 1500.0
    assert parse.parse("{:e}", "1.5E-3")[0] == 0.0015
    assert parse.parse("{:e}", "-1.5e+3")[0] == -1500.0
    assert parse.parse("{:e}", "15") is None
    assert parse.parse("{:e}", "-inf")[0] == float("-inf")


def test_float_g():
    assert parse.parse("{:g}", "12")[0] == 12.0
    assert parse.parse("{:g}", "1.5")[0] == 1.5
    assert parse.parse("{:g}", "1.5e3")[0] == 1500.0
    assert parse.parse("{:g}", "-2e-2")[0] == -0.02
    assert parse.parse("{:g}", "abc") is None
    assert parse.parse("{:G}", "1E3")[0] == 1000.0


def test_percentage():
    assert parse.parse("{:%}", "50%")[0] == 0.5
    assert parse.parse("{:%}", "12.5%")[0] == 0.125
    assert parse.parse("{:%}", "-10%")[0] == -0.1
    assert parse.parse("{:%}", "50") is None


def test_word_types():
    assert parse.parse("{:w}", "abc_1")[0] == "abc_1"
    assert parse.parse("{:w}", "a b") is None
    assert parse.parse("{:W}", " ,")[0] == " ,"
    assert parse.parse("{:W}", "ab") is None
    assert parse.parse("{:s}", " \t")[0] == " \t"
    assert parse.parse("{:s}", "a") is None
    assert parse.parse("{:l}", "abc")[0] == "abc"
    assert parse.parse("{:l}", "ab1") is None


def test_D_type_digits():
    assert parse.parse("{:D}", "123") is None
    assert parse.parse("{:D}", "12a") is None
    assert parse.parse("{:D}", "abc")[0] == "abc"
    assert parse.parse("{:W}", "ab c") is None
    assert parse.parse("{:S}", "ab")[0] == "ab"
    assert parse.parse("{:S}", "a b") is None


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        parse.compile("{:Q}")
    with pytest.raises(ValueError):
        parse.compile("{:y}")


# ---------- width, precision, align ----------

def test_width_text():
    assert parse.parse("{:5}", "ab   ")[0] == "ab   "
    assert parse.parse("{:4}", "abcd")[0] == "abcd"
    assert parse.parse("{:4}", "abc") is None


def test_precision_text():
    assert parse.parse("{:.2}", "ab")[0] == "ab"
    assert parse.parse("{:.2}", "abc") is None
    assert parse.parse("{:.3}", "a")[0] == "a"
    assert parse.parse("{:2.3}", "abc")[0] == "abc"
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "abcd") is None


def test_align_left():
    assert parse.parse("{:<5}|", "ab   |")[0] == "ab   "
    assert parse.parse("{:<}|", "ab   |")[0] == "ab"
    assert parse.parse("{:*<}|", "ab***|")[0] == "ab"
    assert parse.parse("{:<5d}|", "42   |")[0] == 42


def test_align_right():
    assert parse.parse("{:>}|", "   ab|")[0] == "ab"
    assert parse.parse("{:0>}|", "000ab|")[0] == "ab"
    assert parse.parse("{:>5}|", "   ab|")[0] == "   ab"


def test_align_center():
    assert parse.parse("{:^}|", "  ab  |")[0] == "ab"
    assert parse.parse("{:-^}|", "--ab--|")[0] == "ab"
    assert parse.parse("{:^5d}|", " 42  |")[0] == 42


def test_fill_special_chars():
    assert parse.parse("{:.>}|", "...ab|")[0] == "ab"
    assert parse.parse("{:*>}|", "***ab|")[0] == "ab"
    assert parse.parse("{:+<}|", "ab+++|")[0] == "ab"
    assert parse.parse("{:?>}|", "???ab|")[0] == "ab"
    assert parse.parse("{:.>d}|", "..42|")[0] == 42
    assert parse.parse("{:*>d}|", "**42|")[0] == 42


def test_numeric_alignment():
    assert parse.parse("{:>5d}", "   42")[0] == 42
    assert parse.parse("{:<5d}", "42   ")[0] == 42
    assert parse.parse("{:^5d}", " 42  ")[0] == 42
    assert parse.parse("{:05d}", "00042")[0] == 42
    assert parse.parse("{:=5d}", "00042")[0] == 42
    assert parse.parse("{:x>5d}", "xxx42")[0] == 42


def test_numeric_sign_formats():
    assert parse.parse("{:+d}", "+5")[0] == 5
    assert parse.parse("{:-d}", "-5")[0] == -5
    assert parse.parse("{: d}", " 5")[0] == 5
    assert parse.parse("{:+.2f}", "+1.50")[0] == 1.5


def test_roundtrip_with_format():
    for spec, val in [
        ("d", 42), ("5d", 42), ("<5d", 42), ("^7d", -42), ("05d", 42),
        (".2f", 3.14159), ("8.3f", -2.5), ("x", 255), ("o", 8), ("b", 5),
        ("e", 12345.678), ("%", 0.25),
    ]:
        text = ("{:" + spec + "}").format(val)
        r = parse.parse("{:" + spec + "}", text)
        assert r is not None, (spec, text)
        if isinstance(val, float):
            assert r[0] == pytest.approx(float(text.strip().rstrip("%")) / (100 if spec == "%" else 1))
        else:
            assert r[0] == val, (spec, text)


# ---------- names ----------

def test_named_fields_attr_and_index():
    r = parse.parse("{a.b} {c[d]} {c[e]}", "1 2 3")
    assert r.named == {"a.b": "1", "c": {"d": "2", "e": "3"}}
    assert r["a.b"] == "1"
    assert r["c"]["e"] == "3"


def test_name_collision():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}
    r = parse.parse("{a_b} {a.b}", "1 2")
    assert r.named == {"a_b": "1", "a.b": "2"}


def test_hyphenated_name():
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_repeated_name_backreference():
    assert parse.parse("{a} {a}", "x x").named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None
    assert parse.parse("{a:d} {a:d}", "3 3").named == {"a": 3}


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        parse.compile("{a:d} {a:w}")
    with pytest.raises(RepeatedNameError):
        parse.compile("{a} {a:d}")


def test_named_and_fixed_fields_lists():
    p = parse.compile("{} {x} {} {y:d}")
    assert p.named_fields == ["x", "y"]
    assert p.fixed_fields == [0, 2]
    # returned lists are copies
    p.named_fields.append("zzz")
    p.fixed_fields.append(9)
    assert p.named_fields == ["x", "y"]
    assert p.fixed_fields == [0, 2]
    assert p.format == "{} {x} {} {y:d}"


def test_digit_name_is_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")
    assert r.named == {}


def test_spans():
    r = parse.parse("{} and {n}", "aa and bbb")
    assert r.spans[0] == (0, 2)
    assert r.spans["n"] == (7, 10)
    r = search("x{:d}y{k:d}", "--x12y345")
    assert r.spans[0] == (3, 5)
    assert r.spans["k"] == (6, 9)


def test_spans_with_multigroup_types():
    r = parse.parse("{:g} {:d}", "1.5e3 7")
    assert r.fixed == (1500.0, 7)
    assert r.spans[0] == (0, 5)
    assert r.spans[1] == (6, 7)
    r = parse.parse("{:n} {:w}", "1,234 ab")
    assert r.fixed == (1234, "ab")


def test_mixed_complex_fields_indexing():
    r = parse.parse("{:ti} {:d} {:%} {:g} {:n} {:x}",
                    "2020-01-02 10:11:12 5 50% 1.5 1,000 ff")
    assert r.fixed == (datetime(2020, 1, 2, 10, 11, 12), 5, 0.5, 1.5, 1000, 255)


# ---------- dates ----------

def test_ti():
    assert parse.parse("{:ti}", "2012-09-17")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:ti}", "2012-09-17 10:21")[0] == datetime(2012, 9, 17, 10, 21)
    assert parse.parse("{:ti}", "2012-09-17T10:21:45")[0] == datetime(2012, 9, 17, 10, 21, 45)
    r = parse.parse("{:ti}", "2012-09-17 10:21:45.123")[0]
    assert r.microsecond == 123000
    r = parse.parse("{:ti}", "2012-09-17 10:21:45.000123")[0]
    assert r.microsecond == 123
    r = parse.parse("{:ti}", "2012-09-17 10:21:45.1234567")[0]
    assert r.microsecond == 123456


def test_ti_tz():
    r = parse.parse("{:ti}", "2012-09-17 10:21:45Z")[0]
    assert r.utcoffset() == timedelta(0)
    r = parse.parse("{:ti}", "2012-09-17 10:21:45 +0530")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2012-09-17 10:21:45-08:00")[0]
    assert r.utcoffset() == timedelta(hours=-8)
    r = parse.parse("{:ti}", "2012-09-17 10:21:45 -0230")[0]
    assert r.utcoffset() == timedelta(hours=-2, minutes=-30)


def test_tg():
    assert parse.parse("{:tg}", "17/9/2012")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:tg}", "17-Sep-2012")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:tg}", "17/September/2012")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:tg}", "1/2/2012 10:21")[0] == datetime(2012, 2, 1, 10, 21)
    assert parse.parse("{:tg}", "1/2/2012 10:21:30 PM")[0] == datetime(2012, 2, 1, 22, 21, 30)
    assert parse.parse("{:tg}", "1/2/2012 12:05 AM")[0] == datetime(2012, 2, 1, 0, 5)
    assert parse.parse("{:tg}", "1/2/2012 12:05 PM")[0] == datetime(2012, 2, 1, 12, 5)
    assert parse.parse("{:tg}", "1/2/2012 1:05 AM")[0] == datetime(2012, 2, 1, 1, 5)


def test_ta():
    assert parse.parse("{:ta}", "9/17/2012")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:ta}", "Sep/17/2012")[0] == datetime(2012, 9, 17)
    assert parse.parse("{:ta}", "9-17-2012 3:04:05 PM")[0] == datetime(2012, 9, 17, 15, 4, 5)
    r = parse.parse("{:ta}", "9/17/2012 3:04 +0100")[0]
    assert r.utcoffset() == timedelta(hours=1)


def test_te():
    r = parse.parse("{:te}", "Mon, 17 Sep 2012 10:21:45 +0100")[0]
    assert r.replace(tzinfo=None) == datetime(2012, 9, 17, 10, 21, 45)
    assert r.utcoffset() == timedelta(hours=1)
    r = parse.parse("{:te}", "17 Sep 2012 10:21:45 -0100")[0]
    assert r.utcoffset() == timedelta(hours=-1)
    assert r.day == 17


def test_th():
    r = parse.parse("{:th}", "17/Sep/2012:10:21:45 +0100")[0]
    assert r.replace(tzinfo=None) == datetime(2012, 9, 17, 10, 21, 45)
    assert r.utcoffset() == timedelta(hours=1)


def test_tc():
    r = parse.parse("{:tc}", "Mon Sep 17 10:21:45 2012")[0]
    assert r == datetime(2012, 9, 17, 10, 21, 45)
    r = parse.parse("{:tc}", "Tue Feb  5 01:02:03 2013")
    assert r[0] == datetime(2013, 2, 5, 1, 2, 3)


def test_tt():
    assert parse.parse("{:tt}", "10:21:45")[0] == time(10, 21, 45)
    assert parse.parse("{:tt}", "10:21")[0] == time(10, 21)
    assert parse.parse("{:tt}", "10:21 PM")[0] == time(22, 21)
    assert parse.parse("{:tt}", "12:21 AM")[0] == time(0, 21)
    assert parse.parse("{:tt}", "12:21 PM")[0] == time(12, 21)
    r = parse.parse("{:tt}", "10:21:45 +0200")[0]
    assert r.utcoffset() == timedelta(hours=2)
    assert parse.parse("{:tt}", "10:21:45.5")[0] == time(10, 21, 45, 500000)


def test_ts():
    r = parse.parse("{:ts}", "Sep 17 10:21:45")[0]
    assert r == datetime(datetime.today().year, 9, 17, 10, 21, 45)
    r = parse.parse("{:ts}", "Feb  3 ")[0]
    assert (r.month, r.day) == (2, 3)


def test_tz_name_ignored_and_spec():
    assert FixedTzOffset(60, "x") == FixedTzOffset(60, "x")
    assert FixedTzOffset(60, "x") != FixedTzOffset(61, "x")
    assert FixedTzOffset(60, "x") != FixedTzOffset(60, "y")
    assert FixedTzOffset(60, "x") != 5
    tz = FixedTzOffset(90, "n")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "n"
    assert tz.dst(None) == timedelta(0)
    assert "n" in repr(tz)


def test_invalid_date_raises():
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2012-13-45")


def test_strptime_style():
    assert parse.parse("{:%Y-%m-%d}", "2012-09-17")[0] == date(2012, 9, 17)
    assert parse.parse("{:%H:%M:%S}", "10:21:45")[0] == time(10, 21, 45)
    assert parse.parse("{:%Y-%m-%d %H:%M}", "2012-09-17 10:21")[0] == datetime(2012, 9, 17, 10, 21)
    assert parse.parse("{:%d/%b/%Y}", "05/Mar/2020")[0] == date(2020, 3, 5)
    assert parse.parse("{:%A, %B %d, %Y}", "Monday, March 05, 2018")[0] == date(2018, 3, 5)
    assert parse.parse("{:%y%m%d}", "200305")[0] == date(2020, 3, 5)
    assert parse.parse("{:%I:%M %p}", "03:04 PM")[0] == time(15, 4)
    r = parse.parse("{:%H:%M:%S.%f}", "10:21:45.123")[0]
    assert r == time(10, 21, 45, 123000)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2020-01-02 03:04:05+0100")[0]
    assert r.utcoffset() == timedelta(hours=1)
    assert parse.parse("{:%Y-%m-%d}", "20-09-17") is None
    assert parse.parse("{:%m/%d}", "03/05")[0] == date(datetime.today().year, 3, 5)


# ---------- custom types ----------

def test_extra_types_basic():
    def shouty(s):
        return s.upper()

    r = parse.parse("{:shouty}", "abc", extra_types={"shouty": shouty})
    assert r[0] == "ABC"


def test_extra_types_pattern():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert num.pattern == r"\d+"
    assert parse.parse("{:Num}", "12", extra_types={"Num": num})[0] == 12
    assert parse.parse("{:Num}", "ab", extra_types={"Num": num}) is None
    r = parse.parse("{a:Num}-{b:Num}", "1-2", extra_types={"Num": num})
    assert r.named == {"a": 1, "b": 2}


def test_extra_types_group_count():
    @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
    def rng(s):
        a, b = s.split("-")
        return (int(a), int(b))

    r = parse.parse("{:R} {:d} {n:w}", "1-2 3 zz", extra_types={"R": rng})
    assert r.fixed == ((1, 2), 3)
    assert r.named == {"n": "zz"}

    @with_pattern(r"a|b")
    def ab(s):
        return s

    assert ab.regex_group_count is None
    r = parse.parse("{:AB} {:d}", "b 4", extra_types={"AB": ab})
    assert r.fixed == ("b", 4)


def test_extra_types_override_builtin_name_with_align():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    r = parse.parse("{:>4Num}|", "  12|", extra_types={"Num": num})
    assert r[0] == 12


def test_with_pattern_default_group_count():
    @with_pattern("x")
    def f(s):
        return s

    assert f.regex_group_count is None
    assert f("q") == "q"


def test_search_findall_extra_types():
    @with_pattern(r"\d+")
    def num(s):
        return int(s)

    assert search("v{:N}", "xv12", extra_types={"N": num})[0] == 12
    assert [r[0] for r in findall("{:N}", "1 2", extra_types={"N": num})] == [1, 2]


def test_pickle_parser_conversion_callables():
    p = pickle.dumps(parse.int_convert(16))
    assert pickle.loads(p)("ff", None) == 255
    cf = pickle.loads(pickle.dumps(parse.convert_first(int)))
    assert cf("5", None) == 5


# ---------- int_convert ----------

def test_int_convert_direct():
    c = parse.int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0x1f", None) == 31
    assert c("0X1F", None) == 31
    assert c("0b11", None) == 3
    assert c("0o17", None) == 15
    assert c("0", None) == 0
    assert c("00", None) == 0
    assert c("1_000", None) == 1000
    assert parse.int_convert(2)("101", None) == 5
    assert parse.int_convert(16)("0xff", None) == 255
    assert parse.int_convert(8)("17", None) == 15
    assert parse.int_convert(10)("1,234", None) == 1234
    assert parse.int_convert(36)("zz", None) == 35 * 36 + 35


def test_percentage_direct():
    assert parse.percentage("25%", None) == 0.25


# ---------- Parser object ----------

def test_repr():
    assert repr(Parser("abc")) == "<Parser 'abc'>"
    long = "a" * 25
    assert repr(Parser(long)) == "<Parser %r>" % ("a" * 17 + "...")
    exactly20 = "b" * 20
    assert repr(Parser(exactly20)) == "<Parser %r>" % exactly20


def test_result_repr():
    r = parse.parse("{} {a}", "x y")
    assert repr(r) == "<Result ('x',) {'a': 'y'}>"


def test_compile_reuse():
    p = compile("{:d}+{:d}")
    assert p.parse("1+2").fixed == (1, 2)
    assert p.parse("3+4").fixed == (3, 4)
    assert p.search("x 5+6 y").fixed == (5, 6)
    assert [r.fixed for r in p.findall("1+2 3+4")] == [(1, 2), (3, 4)]


def test_parser_search_pos():
    p = compile("{:d}")
    assert p.search("1 2", pos=1)[0] == 2
    assert p.search("1 2", 0, 1)[0] == 1


def test_many_fields_ok():
    fmt = ",".join("{}" for _ in range(30))
    assert len(parse.parse(fmt, ",".join(str(i) for i in range(30))).fixed) == 30


def test_lazy_matching():
    r = parse.parse("{}-{}", "a-b-c")
    assert r.fixed == ("a", "b-c")
    r = parse.parse("{a}={b}", "k=v=w")
    assert r.named == {"a": "k", "b": "v=w"}


def test_empty_match_for_untyped_not_allowed():
    assert parse.parse("a{}b", "ab") is None
    assert parse.parse("a{}b", "axb")[0] == "x"


def test_extract_format_function():
    f = parse.extract_format("*^10.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "^"
    assert f["width"] == "10"
    assert f["precision"] == "3"
    assert f["type"] == "f"
    f = parse.extract_format(">5", {})
    assert f["align"] == ">" and f["fill"] is None and f["width"] == "5"
    f = parse.extract_format("05d", {})
    assert f["zero"] is True and f["width"] == "5" and f["type"] == "d"
    f = parse.extract_format("d", {})
    assert f["zero"] is False and f["width"] == "" and f["type"] == "d"
    f = parse.extract_format(",d", {})
    assert f["grouping"] == ","
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    f = parse.extract_format("=+5d", {})
    assert f["align"] == "="
    f = parse.extract_format("x<5", {})
    assert f["fill"] == "x" and f["align"] == "<"
    with pytest.raises(ValueError):
        parse.extract_format("5z", {})
    assert parse.extract_format("Z", {"Z": int})["type"] == "Z"


def test_allowed_types_accepted():
    for t in "nboxX%fFeEgGwWdDsSl":
        parse.compile("{:" + t + "}")
    for t in ["ti", "ta", "te", "tg", "th", "tc", "tt", "ts"]:
        parse.compile("{:" + t + "}")


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("x") == "x"


def test_date_convert_time_only_and_mm_dd():
    m = parse.compile("{:tt}").parse("1:02:03", evaluate_result=False)
    assert m.evaluate_result()[0] == time(1, 2, 3)


def test_unicode():
    assert parse.parse("héllo {}", "héllo wörld")[0] == "wörld"
    assert parse.parse("{:w}", "äb")[0] == "äb"


def test_with_slash_in_text_and_braces_content():
    assert parse.parse("{}/{}", "a/b").fixed == ("a", "b")
    assert parse.parse("[{}]", "[x]")[0] == "x"
    assert parse.parse("({:d})", "(5)")[0] == 5
    assert parse.parse("{}%", "5%")[0] == "5"


def test_search_returns_first():
    r = search("{:d}", "a 1 b 2")
    assert r[0] == 1
    assert search("{:d}", "no digits") is None
