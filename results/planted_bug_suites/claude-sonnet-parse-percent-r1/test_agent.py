import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


# ---------- basic parsing ----------

def test_fixed_and_named():
    r = parse.parse("{} and {}", "a and b")
    assert r.fixed == ("a", "b")
    r = parse.parse("{x} {y}", "1 2")
    assert r.named == {"x": "1", "y": "2"}
    assert r["x"] == "1" and "x" in r and "z" not in r


def test_no_match_returns_none():
    assert parse.parse("a{}b", "axc") is None
    assert parse.parse("{}", "") is None


def test_exact_match_required():
    assert parse.parse("{}", "abc\n") is not None
    assert parse.parse("a{}", "xab") is None
    assert parse.parse("{}b", "abx") is None
    assert parse.parse("{:d}", "12x") is None


def test_case_insensitive_default_and_sensitive():
    assert parse.parse("ABC {}", "abc x").fixed == ("x",)
    assert parse.parse("ABC {}", "abc x", case_sensitive=True) is None
    assert parse.parse("ABC {}", "ABC x", case_sensitive=True).fixed == ("x",)


def test_dotall():
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)


def test_escaped_braces():
    assert parse.parse("{{}}", "{}") is not None
    assert parse.parse("{{{}}}", "{x}").fixed == ("x",)
    assert parse.parse("{{", "{") is not None


def test_regex_chars_escaped():
    assert parse.parse("a.b{}", "a.bX").fixed == ("X",)
    assert parse.parse("a.b{}", "aXbX") is None
    for ch in "?\\.[]()*+^$!|":
        assert parse.parse("x" + ch + "{}", "x" + ch + "y").fixed == ("y",)


def test_lazy_matching():
    assert parse.parse("{}-{}", "a-b-c").fixed == ("a", "b-c")


def test_slices_and_index():
    r = parse.parse("{}/{}/{}", "a/b/c")
    assert r[0] == "a" and r[-1] == "c" and r[1:] == ("b", "c")


def test_repr():
    r = parse.parse("{} {n}", "a b")
    assert "a" in repr(r) and "'b'" in repr(r)
    p = parse.compile("short")
    assert repr(p) == "<Parser 'short'>"
    p = parse.compile("x" * 30)
    assert repr(p) == "<Parser '" + "x" * 17 + "...'>"
    p = parse.compile("y" * 20)
    assert repr(p) == "<Parser '" + "y" * 20 + "'>"


def test_parser_properties():
    p = parse.compile("{a} {} {b:d}")
    assert p.named_fields == ["a", "b"]
    assert p.fixed_fields == [1]
    assert p.format == "{a} {} {b:d}"
    p.named_fields.append("zz")
    assert p.named_fields == ["a", "b"]
    p.fixed_fields.append(9)
    assert p.fixed_fields == [1]


# ---------- names ----------

def test_dotted_and_indexed_names():
    r = parse.parse("{a.b} {a.c}", "1 2")
    assert r.named == {"a.b": "1", "a.c": "2"}
    r = parse.parse("{a[x]} {a[y]}", "1 2")
    assert r.named == {"a": {"x": "1", "y": "2"}}
    r = parse.parse("{a[x][y]}", "1")
    assert r.named == {"a": {"x": {"y": "1"}}}


def test_name_collisions():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}
    r = parse.parse("{a-b} {a_b}", "1 2")
    assert r.named == {"a-b": "1", "a_b": "2"}


def test_repeated_name_same_value():
    assert parse.parse("{a} {a}", "x x").named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None
    assert parse.parse("{a:d} {a:d}", "1 1").named == {"a": 1}


def test_repeated_name_type_mismatch():
    with pytest.raises(parse.RepeatedNameError):
        parse.parse("{a:d} {a}", "1 1")
    assert issubclass(parse.RepeatedNameError, ValueError)


def test_digit_name_is_fixed():
    r = parse.parse("{0} {1}", "a b")
    assert r.fixed == ("a", "b")


def test_spans():
    r = parse.parse("ab {x} {}", "ab foo bar")
    assert r.spans["x"] == (3, 6)
    assert r.spans[0] == (7, 10)


def test_too_many_fields():
    fmt = " ".join("{:d}" for _ in range(60))
    p = parse.compile(fmt)
    try:
        r = p.parse(" ".join(["1"] * 60))
    except parse.TooManyFields:
        return
    assert r is not None and len(r.fixed) == 60


# ---------- types ----------

def test_integer_d():
    assert parse.parse("{:d}", "42")[0] == 42
    assert parse.parse("{:d}", "-42")[0] == -42
    assert parse.parse("{:d}", "+42")[0] == 42
    assert parse.parse("{:d}", "0x1f")[0] == 31
    assert parse.parse("{:d}", "-0x1f")[0] == -31
    assert parse.parse("{:d}", "0b101")[0] == 5
    assert parse.parse("{:d}", "0o17")[0] == 15
    assert parse.parse("{:d}", "0O17")[0] == 15
    assert parse.parse("{:d}", "0B11")[0] == 3
    assert parse.parse("{:d}", "0X1F")[0] == 31
    assert parse.parse("{:d}", "007")[0] == 7
    assert parse.parse("{:d}", "0")[0] == 0
    assert parse.parse("{:d}", "1.5") is None
    assert parse.parse("{:d}", "abc") is None
    assert parse.parse("{:d}", "0b12") is None


def test_integer_d_width_and_grouping():
    assert parse.parse("{:2d}", "12")[0] == 12
    assert parse.parse("{:2d}", "123") is None
    assert parse.parse("{:3d}", " 12")[0] == 12
    assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
    assert parse.parse("{:_d}", "1_234")[0] == 1234
    assert parse.parse("{:d}", "1,234") is None


def test_integer_other_bases():
    assert parse.parse("{:b}", "101")[0] == 5
    assert parse.parse("{:b}", "0b101")[0] == 5
    assert parse.parse("{:b}", "-101")[0] == -5
    assert parse.parse("{:b}", "102") is None
    assert parse.parse("{:o}", "17")[0] == 15
    assert parse.parse("{:o}", "0o17")[0] == 15
    assert parse.parse("{:o}", "18") is None
    assert parse.parse("{:x}", "ff")[0] == 255
    assert parse.parse("{:x}", "0xFF")[0] == 255
    assert parse.parse("{:X}", "Ab")[0] == 171
    assert parse.parse("{:x}", "+1f")[0] == 31
    assert parse.parse("{:x}", "zz") is None


def test_hex_not_confused_with_prefix():
    # base explicit: "0b" digits are hex digits for x
    assert parse.parse("{:x}", "0b1")[0] == 0xB1


def test_n_type():
    assert parse.parse("{:n}", "1")[0] == 1
    assert parse.parse("{:n}", "123")[0] == 123
    assert parse.parse("{:n}", "1,234")[0] == 1234
    assert parse.parse("{:n}", "1.234.567")[0] == 1234567
    assert parse.parse("{:n}", "-12,345")[0] == -12345
    assert parse.parse("{:n}", "1234") is None
    assert parse.parse("{:n}", "x") is None


def test_float_f():
    assert parse.parse("{:f}", "1.5")[0] == 1.5
    assert parse.parse("{:f}", ".5")[0] == 0.5
    assert parse.parse("{:f}", "-1.5")[0] == -1.5
    assert parse.parse("{:f}", "+1.5")[0] == 1.5
    assert parse.parse("{:f}", "12.25")[0] == 12.25
    assert isinstance(parse.parse("{:f}", "1.5")[0], float)
    assert parse.parse("{:f}", "15") is None
    assert parse.parse("{:f}", "1.") is None
    assert parse.parse("{:f}", "abc") is None
    assert parse.parse("{:f}", "nan")[0] != parse.parse("{:f}", "nan")[0]
    assert parse.parse("{:f}", "inf")[0] == float("inf")


def test_float_f_precision_zero():
    assert parse.parse("{:.0f}", "20")[0] == 20.0
    assert parse.parse("{:.0f}", "-20")[0] == -20.0


def test_decimal_F():
    v = parse.parse("{:F}", "1.50")[0]
    assert isinstance(v, Decimal) and v == Decimal("1.50")
    assert str(v) == "1.50"
    v = parse.parse("{:F}", "-0.10")[0]
    assert v == Decimal("-0.10")


def test_float_e_and_g():
    assert parse.parse("{:e}", "1.5e3")[0] == 1500.0
    assert parse.parse("{:e}", "1.5E-3")[0] == 0.0015
    assert parse.parse("{:E}", "2.0e+2")[0] == 200.0
    assert parse.parse("{:e}", "-1.5e3")[0] == -1500.0
    assert parse.parse("{:e}", "15") is None
    assert parse.parse("{:e}", "-inf")[0] == float("-inf")
    assert parse.parse("{:g}", "12")[0] == 12.0
    assert parse.parse("{:g}", "1.5")[0] == 1.5
    assert parse.parse("{:g}", "1.5e2")[0] == 150.0
    assert parse.parse("{:g}", "-3")[0] == -3.0
    assert parse.parse("{:G}", "1E2")[0] == 100.0
    assert parse.parse("{:g}", "+inf")[0] == float("inf")
    assert parse.parse("{:g}", "x") is None


def test_g_followed_by_more_fields_group_counts():
    r = parse.parse("{:g} {:d} {}", "1.5e2 7 z")
    assert r.fixed == (150.0, 7, "z")
    r = parse.parse("{:n} {:d} {}", "1,234 7 z")
    assert r.fixed == (1234, 7, "z")
    r = parse.parse("{:b} {:o} {:x} {}", "0b11 0o7 0xf z")
    assert r.fixed == (3, 7, 15, "z")
    r = parse.parse("{:%} {:d} {}", "5% 7 z")
    assert r.fixed[1:] == (7, "z")


def test_percentage():
    assert parse.parse("{:%}", "50%")[0] == pytest.approx(0.5)
    assert parse.parse("{:%}", "100%")[0] == pytest.approx(1.0)
    assert parse.parse("{:%}", "12.5%")[0] == pytest.approx(0.125)
    assert parse.parse("{:%}", "0%")[0] == 0
    assert parse.parse("{:%}", "-5%")[0] == pytest.approx(-0.05)
    assert parse.parse("{:%}", "50") is None
    assert parse.parse("{:%}", "%") is None
    assert parse.parse("{:%}", "5.%") is None


def test_letters_type():
    assert parse.parse("{:l}", "abc")[0] == "abc"
    assert parse.parse("{:l}", "ab1") is None
    assert parse.parse("{:l} {:l}", "ab cd").fixed == ("ab", "cd")


def test_w_and_s_and_d_classes():
    assert parse.parse("{:w}", "ab_1")[0] == "ab_1"
    assert parse.parse("{:w}", "ab-1") is None
    assert parse.parse("{:W}", "-- ")[0] == "-- "
    assert parse.parse("{:S}", "abc")[0] == "abc"
    assert parse.parse("{:S}", "a b") is None
    assert parse.parse("{:s}", "  ")[0] == "  "
    assert parse.parse("{:D}", "abc")[0] == "abc"
    assert parse.parse("{:D}", "a1") is None


def test_invalid_format_spec():
    with pytest.raises(ValueError):
        parse.compile("{:q}")
    with pytest.raises(ValueError):
        parse.parse("{:z9}", "x")


# ---------- width / precision / align ----------

def test_width_precision_strings():
    assert parse.parse("{:3}", "abc")[0] == "abc"
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:3}", "abcd")[0] == "abcd"
    assert parse.parse("{:.3}", "abc")[0] == "abc"
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:.3}", "a")[0] == "a"
    assert parse.parse("{:2.3}", "abc")[0] == "abc"
    assert parse.parse("{:2.3}", "a") is None
    assert parse.parse("{:2.3}", "abcd") is None


def test_alignment_strips_fill():
    assert parse.parse("{:>}", "  abc")[0] == "abc"
    assert parse.parse("{:<}", "abc  ")[0] == "abc"
    assert parse.parse("{:^}", " abc ")[0] == "abc"
    assert parse.parse("{:*>}", "**abc")[0] == "abc"
    assert parse.parse("{:*<}", "abc**")[0] == "abc"
    assert parse.parse("{:*^}", "**abc**")[0] == "abc"
    assert parse.parse("{:.>}", "..abc")[0] == "abc"
    assert parse.parse("{:+>}", "++abc")[0] == "abc"
    assert parse.parse("{:<}|", "abc  |")[0] == "abc"


def test_width_implies_right_align():
    assert parse.parse("{:5d}", "   12")[0] == 12
    assert parse.parse("{:5}", "  abc")[0] == "  abc"
    assert parse.parse("{:<5d}", "12   ")[0] == 12


def test_numeric_alignment():
    assert parse.parse("{:>5d}", "   12")[0] == 12
    assert parse.parse("{:*>5d}", "***12")[0] == 12
    assert parse.parse("{:05d}", "00012")[0] == 12
    assert parse.parse("{:=5d}", "00012")[0] == 12
    assert parse.parse("{:^7d}", "  12   ")[0] == 12
    assert parse.parse("{:*^7d}", "**12***")[0] == 12


def test_signs_in_spec():
    assert parse.parse("{:+d}", "+5")[0] == 5
    assert parse.parse("{: d}", " 5")[0] == 5
    assert parse.parse("{:-d}", "-5")[0] == -5


def test_float_precision_spec():
    assert parse.parse("{:.2f}", "1.25")[0] == 1.25
    assert parse.parse("{:6.2f}", "  1.25")[0] == 1.25


# ---------- extra types ----------

def test_extra_types():
    def conv(s):
        return int(s) * 2

    conv.pattern = r"\d+"
    r = parse.parse("{:X}", "21", extra_types={"X": conv})
    assert r[0] == 42
    assert parse.parse("{:X}", "ab", extra_types={"X": conv}) is None


def test_extra_type_no_pattern_defaults_lazy():
    def conv(s):
        return s.upper()

    r = parse.parse("{:Y} z", "ab z", extra_types={"Y": conv})
    assert r[0] == "AB"


def test_with_pattern_and_group_count():
    @parse.with_pattern(r"(a|b)(c|d)", regex_group_count=2)
    def ab(s):
        return s[::-1]

    assert ab.pattern == r"(a|b)(c|d)"
    assert ab.regex_group_count == 2
    assert ab("xy") == "yx"
    r = parse.parse("{:AB} {:d} {}", "ac 5 z", extra_types={"AB": ab})
    assert r.fixed == ("ca", 5, "z")


def test_with_pattern_named():
    @parse.with_pattern(r"yes|no")
    def yn(s):
        return s == "yes"

    r = parse.parse("{a:YN} {b:YN}", "yes no", extra_types={"YN": yn})
    assert r.named == {"a": True, "b": False}


def test_extra_type_pickle_convert_first():
    c = parse.convert_first(int)
    assert pickle.loads(pickle.dumps(c))("5", None) == 5


def test_int_convert():
    c = parse.int_convert()
    assert c("12", None) == 12
    assert c("-12", None) == -12
    assert c("+12", None) == 12
    assert c("0x10", None) == 16
    assert c("-0b11", None) == -3
    assert c("0o10", None) == 8
    assert c("010", None) == 10
    assert c("1,000", None) == 1000
    assert parse.int_convert(16)("ff", None) == 255
    assert parse.int_convert(2)("0b11", None) == 3
    assert parse.int_convert(10)("1,234", None) == 1234
    assert parse.int_convert(36)("z", None) == 35


# ---------- search / findall ----------

def test_search():
    r = parse.search("{:d}", "abc 12 def")
    assert r[0] == 12
    assert r.spans[0] == (3, 6)
    assert parse.search("{:d}", "abc") is None


def test_search_pos_endpos():
    s = "1 22 333"
    assert parse.search("{:d}", s, pos=2)[0] == 22
    assert parse.search("{:d}", s, pos=5)[0] == 333
    assert parse.search("{:d}", s, endpos=1)[0] == 1
    assert parse.search("{:d}", s, pos=1, endpos=2) is None
    assert parse.search("{:d}", s, pos=1, endpos=3)[0] == 2


def test_findall():
    res = list(parse.findall("<{}>", "<a> <b> <c>"))
    assert [r[0] for r in res] == ["a", "b", "c"]
    assert list(parse.findall("<{}>", "none")) == []
    res = list(parse.findall("{:d}", "1 22 333", pos=2))
    assert [r[0] for r in res] == [22, 333]
    res = list(parse.findall("{:d}", "1 22 333", endpos=4))
    assert [r[0] for r in res] == [1, 22]


def test_findall_non_overlapping_and_iterator():
    it = parse.findall("{:d}", "12 34")
    assert iter(it) is it
    assert next(it)[0] == 12
    assert next(it)[0] == 34
    with pytest.raises(StopIteration):
        next(it)


def test_findall_case_sensitive():
    assert len(list(parse.findall("a{:d}", "A1 a2"))) == 2
    assert len(list(parse.findall("a{:d}", "A1 a2", case_sensitive=True))) == 1


def test_search_case_sensitive():
    assert parse.search("A{}", "xa1") is not None
    assert parse.search("A{}", "xa1", case_sensitive=True) is None


# ---------- evaluate_result=False ----------

def test_evaluate_result_false():
    m = parse.parse("{:d}", "5", evaluate_result=False)
    assert isinstance(m, parse.Match)
    assert m.evaluate_result()[0] == 5
    m = parse.search("{:d}", "x5", evaluate_result=False)
    assert m.evaluate_result()[0] == 5
    ms = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert [x.evaluate_result()[0] for x in ms] == [1, 2]
    p = parse.compile("{:d}")
    assert p.evaluate_result(p._match_re.match("9"))[0] == 9


def test_compile_reuse():
    p = parse.compile("{:d}-{:d}")
    assert p.parse("1-2").fixed == (1, 2)
    assert p.parse("3-4").fixed == (3, 4)
    assert p.search("x 5-6 y").fixed == (5, 6)
    assert p.parse("x") is None


# ---------- datetimes ----------

def test_ti():
    r = parse.parse("{:ti}", "2023-04-05 06:07:08")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8)
    r = parse.parse("{:ti}", "2023-04-05T06:07:08.123456Z")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8, 123456, tzinfo=parse.FixedTzOffset(0, "UTC"))
    r = parse.parse("{:ti}", "2023-04-05 06:07:08.5")[0]
    assert r.microsecond == 500000
    r = parse.parse("{:ti}", "2023-04-05")[0]
    assert r == datetime(2023, 4, 5)
    r = parse.parse("{:ti}", "2023-04-05 06:07")[0]
    assert r == datetime(2023, 4, 5, 6, 7)
    r = parse.parse("{:ti}", "2023-04-05 06:07:08 +0530")[0]
    assert r.utcoffset() == timedelta(hours=5, minutes=30)
    r = parse.parse("{:ti}", "2023-04-05 06:07:08-08:00")[0]
    assert r.utcoffset() == timedelta(hours=-8)
    assert parse.parse("{:ti}", "garbage") is None


def test_tg():
    assert parse.parse("{:tg}", "5/4/2023")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:tg}", "05-04-2023")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:tg}", "5/Apr/2023")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:tg}", "5/April/2023")[0] == datetime(2023, 4, 5)
    r = parse.parse("{:tg}", "5/4/2023 1:02:03 PM")[0]
    assert r == datetime(2023, 4, 5, 13, 2, 3)
    r = parse.parse("{:tg}", "5/4/2023 12:30 AM")[0]
    assert r == datetime(2023, 4, 5, 0, 30)
    r = parse.parse("{:tg}", "5/4/2023 12:30 PM")[0]
    assert r == datetime(2023, 4, 5, 12, 30)
    r = parse.parse("{:tg}", "5/4/2023 11:30 AM")[0]
    assert r == datetime(2023, 4, 5, 11, 30)


def test_ta():
    assert parse.parse("{:ta}", "4/5/2023")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:ta}", "Apr/5/2023")[0] == datetime(2023, 4, 5)
    assert parse.parse("{:ta}", "4-5-2023 3:04 PM")[0] == datetime(2023, 4, 5, 15, 4)
    r = parse.parse("{:ta}", "4/5/2023 3:04:05 AM +0100")[0]
    assert r.hour == 3 and r.utcoffset() == timedelta(hours=1)


def test_te():
    r = parse.parse("{:te}", "Wed, 05 Apr 2023 06:07:08 +0200")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8, tzinfo=parse.FixedTzOffset(120, "+0200"))
    assert r.utcoffset() == timedelta(hours=2)
    r = parse.parse("{:te}", "5 Apr 2023 06:07:08 -0330")[0]
    assert r.utcoffset() == -timedelta(hours=3, minutes=30)
    assert r.day == 5 and r.month == 4


def test_th():
    r = parse.parse("{:th}", "05/Apr/2023:06:07:08 +0200")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8, tzinfo=parse.FixedTzOffset(120, "+0200"))
    assert r.utcoffset() == timedelta(hours=2)
    r = parse.parse("{:th}", "05/Dec/2023:23:59:59 -0500")[0]
    assert r.month == 12 and r.utcoffset() == timedelta(hours=-5)


def test_tc():
    r = parse.parse("{:tc}", "Wed Apr 5 06:07:08 2023")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8)
    r = parse.parse("{:tc}", "Sun Jan 15 00:00:00 1999")[0]
    assert r == datetime(1999, 1, 15)


def test_tt():
    assert parse.parse("{:tt}", "06:07:08")[0] == time(6, 7, 8)
    assert parse.parse("{:tt}", "06:07")[0] == time(6, 7)
    assert parse.parse("{:tt}", "6:07 PM")[0] == time(18, 7)
    assert parse.parse("{:tt}", "12:07 AM")[0] == time(0, 7)
    assert parse.parse("{:tt}", "12:07 PM")[0] == time(12, 7)
    assert parse.parse("{:tt}", "06:07:08.25")[0] == time(6, 7, 8, 250000)
    r = parse.parse("{:tt}", "06:07:08 +0100")[0]
    assert r.utcoffset() == timedelta(hours=1)
    r = parse.parse("{:tt}", "06:07:08 -01:30")[0]
    assert r.utcoffset() == -timedelta(hours=1, minutes=30)
    r = parse.parse("{:tt}", "06:07:08 +100")[0]
    assert r.utcoffset() == timedelta(hours=1)


def test_ts():
    y = datetime.today().year
    r = parse.parse("{:ts}", "Apr 5 06:07:08")[0]
    assert r == datetime(y, 4, 5, 6, 7, 8)
    r = parse.parse("{:ts}", "Dec 25 ")[0]
    assert (r.month, r.day) == (12, 25)


def test_date_microsecond_padding():
    r = parse.parse("{:ti}", "2023-04-05 06:07:08.05")[0]
    assert r.microsecond == 50000
    r = parse.parse("{:ti}", "2023-04-05 06:07:08.1234567")[0]
    assert r.microsecond == 123456


def test_fixed_tz_offset():
    tz = parse.FixedTzOffset(90, "x")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "x"
    assert tz.dst(None) == timedelta(0)
    assert tz == parse.FixedTzOffset(90, "x")
    assert tz != parse.FixedTzOffset(91, "x")
    assert tz != parse.FixedTzOffset(90, "y")
    assert tz != 5
    assert "x" in repr(tz)


def test_strftime_formats():
    r = parse.parse("{:%Y-%m-%d}", "2023-04-05")[0]
    assert r == date(2023, 4, 5)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-04-05 06:07:08")[0]
    assert r == datetime(2023, 4, 5, 6, 7, 8)
    r = parse.parse("{:%H:%M}", "06:07")[0]
    assert (r.hour, r.minute) == (6, 7)
    r = parse.parse("{:%d %b %Y}", "05 Apr 2023")[0]
    assert r == date(2023, 4, 5)
    r = parse.parse("{:%A, %B %d, %Y}", "Wednesday, April 05, 2023")[0]
    assert r == date(2023, 4, 5)
    r = parse.parse("{:%I:%M %p}", "06:07 PM")[0]
    assert (r.hour, r.minute) == (18, 7)
    r = parse.parse("{:%Y-%m-%d %H:%M:%S.%f}", "2023-04-05 06:07:08.123")[0]
    assert r.microsecond == 123000
    r = parse.parse("{:%Y-%m-%d %H:%M:%S%z}", "2023-04-05 06:07:08+0200")[0]
    assert r.utcoffset() == timedelta(hours=2)
    r = parse.parse("{:%y}", "23")[0]
    assert parse.parse("{:%Y-%m-%d}", "23-4-5") is None
    r = parse.parse("{:%m/%d}", "04/05")[0]
    assert (r.month, r.day) == (4, 5)
    assert r.year == datetime.today().year
    r = parse.parse("{:%j %Y}", "095 2023")[0]
    assert r.month == 4 and r.day == 5


def test_strftime_in_context():
    r = parse.parse("on {:%Y-%m-%d} at {:d}", "on 2023-04-05 at 7")
    assert r.fixed == (date(2023, 4, 5), 7)


def test_get_regex_for_datetime_format():
    assert parse.get_regex_for_datetime_format("%Y") == "[0-9]{4}"
    assert parse.get_regex_for_datetime_format("%Y-%m") == "[0-9]{4}-[0-9]{1,2}"
    assert parse.get_regex_for_datetime_format("x") == "x"


def test_strf_date_convert_bad():
    with pytest.raises(ValueError):
        parse.strf_date_convert("x", None, "x")


def test_extract_format():
    f = parse.extract_format("*>+010,.3f", {})
    assert f["fill"] == "*" and f["align"] == ">"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "f"
    f = parse.extract_format("<5", {})
    assert f["align"] == "<" and f["fill"] is None and f["width"] == "5"
    f = parse.extract_format("d", {})
    assert f["type"] == "d" and f["width"] == ""
    f = parse.extract_format("_d", {})
    assert f["grouping"] == "_"
    f = parse.extract_format("Q", {"Q": int})
    assert f["type"] == "Q"
    with pytest.raises(ValueError):
        parse.extract_format("Q", {})


def test_parse_re_splitting():
    assert parse.parse("{a}{{b}}{c}", "1{b}2").named == {"a": "1", "c": "2"}
    assert parse.parse("a{:d}b{:d}c", "a1b2c").fixed == (1, 2)


def test_non_numeric_with_special_chars_fill():
    assert parse.parse("{:.>}", "..abc")[0] == "abc"
    assert parse.parse("{:*<}", "abc**")[0] == "abc"


def test_result_equality_of_types():
    r = parse.parse("{:d} {:f} {:w}", "1 2.5 ab")
    assert r.fixed == (1, 2.5, "ab")
    assert isinstance(r.fixed, tuple)
