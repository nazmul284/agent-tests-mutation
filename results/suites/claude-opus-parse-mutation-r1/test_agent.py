import pickle
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse
from parse import (
    FixedTzOffset,
    Match,
    Parser,
    RepeatedNameError,
    Result,
    convert_first,
    date_convert,
    extract_format,
    get_regex_for_datetime_format,
    int_convert,
    percentage,
    strf_date_convert,
    with_pattern,
)


THIS_YEAR = datetime.today().year


# ---------------------------------------------------------------------------
# with_pattern / helpers
# ---------------------------------------------------------------------------


def test_with_pattern_sets_attributes():
    @with_pattern(r"\d+", regex_group_count=3)
    def f(text):
        return text

    assert f.pattern == r"\d+"
    assert f.regex_group_count == 3
    assert f("x") == "x"


def test_with_pattern_default_group_count_is_none():
    @with_pattern(r"\d+")
    def f(text):
        return text

    assert f.regex_group_count is None


def test_convert_first():
    c = convert_first(int)
    assert c("42", None) == 42
    assert pickle.loads(pickle.dumps(c))("7", None) == 7


def test_percentage():
    assert percentage("50%", None) == 0.5
    assert percentage("12.5%", None) == 0.125


# ---------------------------------------------------------------------------
# int_convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123", 123),
        ("-123", -123),
        ("+123", 123),
        ("0xA", 10),
        ("0XA", 10),
        ("-0xA", -10),
        ("+0xA", 10),
        ("0b101", 5),
        ("0B101", 5),
        ("-0b101", -5),
        ("0o17", 15),
        ("0O17", 15),
        ("-0o17", -15),
        ("0x1f", 31),
        ("0xzz1", 1),
        ("1,234", 1234),
        ("0", 0),
        ("07", 7),
        ("012", 12),
        ("-012", -12),
    ],
)
def test_int_convert_autodetect(text, expected):
    assert int_convert()(text, None) == expected


def test_int_convert_short_prefix_is_decimal():
    # "0b" with nothing after it: not enough characters for a base prefix
    assert int_convert()("0b", None) == 0
    # "0x1" has exactly three characters -> hex detection
    assert int_convert()("0xf", None) == 15
    assert int_convert()("-0xf", None) == -15


def test_int_convert_explicit_base():
    assert int_convert(16)("ff", None) == 255
    assert int_convert(16)("0xff", None) == 255
    assert int_convert(16)("FF", None) == 255
    assert int_convert(2)("0b11", None) == 3
    assert int_convert(8)("0o17", None) == 15
    assert int_convert(10)("1.234.567", None) == 1234567
    assert int_convert(36)("z", None) == 35
    assert int_convert(16)("-ff", None) == -255


def test_int_convert_chars():
    assert int_convert.CHARS == "0123456789abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# FixedTzOffset
# ---------------------------------------------------------------------------


def test_fixed_tz_offset():
    tz = FixedTzOffset(90, "+01:30")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.tzname(None) == "+01:30"
    assert tz.dst(None) == timedelta(0)
    assert FixedTzOffset.ZERO == timedelta(0)
    assert repr(tz) == "<FixedTzOffset +01:30 1:30:00>"


def test_fixed_tz_offset_eq():
    assert FixedTzOffset(60, "a") == FixedTzOffset(60, "a")
    assert not (FixedTzOffset(60, "a") == FixedTzOffset(60, "b"))
    assert not (FixedTzOffset(60, "a") == FixedTzOffset(61, "a"))
    assert FixedTzOffset(60, "a") != FixedTzOffset(61, "a")
    assert FixedTzOffset(60, "a").__eq__(5) is NotImplemented
    assert not (FixedTzOffset(60, "a") == 5)


# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------


def test_months_map():
    names = [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ]
    for i, name in enumerate(names, 1):
        assert parse.MONTHS_MAP[name] == i
        assert parse.MONTHS_MAP[name[:3]] == i
    assert len(parse.MONTHS_MAP) == 23


def test_patterns():
    assert parse.DAYS_PAT == r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
    assert parse.MONTHS_PAT == r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    assert parse.TIME_PAT == r"(\d{1,2}:\d{1,2}(:\d{1,2}(\.\d+)?)?)"
    assert parse.AM_PAT == r"(\s+[AP]M)"
    assert parse.TZ_PAT == r"(\s+[-+]\d\d?:?\d\d)"


def test_allowed_types():
    expected = set("nboxX%fFeEgGwWdDsSl") | {
        "ti", "te", "ta", "th", "tg", "tc", "tt", "ts"
    }
    assert parse.ALLOWED_TYPES == expected


def test_regex_safety():
    for ch in "?\\.[]()*+^$!|":
        assert parse.REGEX_SAFETY.sub(r"\\\1", ch) == "\\" + ch
    assert parse.REGEX_SAFETY.sub(r"\\\1", "a") == "a"


# ---------------------------------------------------------------------------
# date_convert direct
# ---------------------------------------------------------------------------


class FakeMatch:
    def __init__(self, *groups):
        self._groups = groups

    def groups(self):
        return self._groups


def test_date_convert_time_only():
    m = FakeMatch("10:20:30")
    assert date_convert("", m, hms=0) == time(10, 20, 30)


def test_date_convert_hm():
    m = FakeMatch("2020-01-02", "10:20")
    assert date_convert("", m, ymd=0, hms=1) == datetime(2020, 1, 2, 10, 20)


def test_date_convert_microseconds():
    m = FakeMatch("1:2:3.5")
    assert date_convert("", m, hms=0) == time(1, 2, 3, 500000)
    m = FakeMatch("1:2:3.1234567")
    assert date_convert("", m, hms=0) == time(1, 2, 3, 123456)
    m = FakeMatch("1:2:3.000001")
    assert date_convert("", m, hms=0) == time(1, 2, 3, 1)


def test_date_convert_empty_hms():
    m = FakeMatch("2020-01-02", None)
    assert date_convert("", m, ymd=0, hms=1) == datetime(2020, 1, 2)


def test_date_convert_am_pm():
    def conv(t, am):
        return date_convert("", FakeMatch(t, am), hms=0, am=1)

    assert conv("12:15", " AM") == time(0, 15)
    assert conv("12:15", " PM") == time(12, 15)
    assert conv("1:15", " PM") == time(13, 15)
    assert conv("1:15", " AM") == time(1, 15)
    assert conv("11:15", "AM") == time(11, 15)
    assert conv("11:15", None) == time(11, 15)
    assert conv("12:15", None) == time(12, 15)


def test_date_convert_tz():
    def conv(tz):
        return date_convert("", FakeMatch("1:00", tz), hms=0, tz=1)

    t = conv("Z")
    assert t.tzinfo == FixedTzOffset(0, "UTC")
    t = conv(" +0130")
    assert t.tzinfo == FixedTzOffset(90, "+0130")
    t = conv("-0130")
    assert t.tzinfo == FixedTzOffset(-90, "-0130")
    t = conv("+01:30")
    assert t.tzinfo == FixedTzOffset(90, "+01:30")
    t = conv("-1:30")
    assert t.tzinfo == FixedTzOffset(-90, "-1:30")
    t = conv("+130")
    assert t.tzinfo == FixedTzOffset(90, "+130")
    t = conv("-130")
    assert t.tzinfo == FixedTzOffset(-90, "-130")
    t = conv(None)
    assert t.tzinfo is None


def test_date_convert_mdy_dmy():
    assert date_convert("", FakeMatch("2/3/2011"), mdy=0) == datetime(2011, 2, 3)
    assert date_convert("", FakeMatch("2/3/2011"), dmy=0) == datetime(2011, 3, 2)
    assert date_convert("", FakeMatch("2-Mar-2011"), dmy=0) == datetime(2011, 3, 2)
    assert date_convert("", FakeMatch("2011/3/4"), ymd=0) == datetime(2011, 3, 4)
    assert date_convert("", FakeMatch("2 Mar 2011"), dmy=0) == datetime(2011, 3, 2)


def test_date_convert_d_m_y():
    m = FakeMatch("5", "Jun", "2001")
    assert date_convert("", m, d_m_y=(0, 1, 2)) == datetime(2001, 6, 5)


def test_date_convert_mm_dd():
    m = FakeMatch("x", "Jun", "5")
    assert date_convert("", m, mm=1, dd=2) == datetime(THIS_YEAR, 6, 5)


def test_date_convert_mm_without_dd_falls_through():
    m = FakeMatch("2001-02-03", "Jun")
    assert date_convert("", m, mm=1, ymd=0) == datetime(2001, 2, 3)


# ---------------------------------------------------------------------------
# strf_date_convert / datetime format regex
# ---------------------------------------------------------------------------


def test_strf_date_convert_date():
    assert strf_date_convert("2023-01-05", None, "%Y-%m-%d") == date(2023, 1, 5)
    assert type(strf_date_convert("2023-01-05", None, "%Y-%m-%d")) is date


def test_strf_date_convert_time():
    r = strf_date_convert("10:30", None, "%H:%M")
    assert r == time(10, 30)
    assert type(r) is time


def test_strf_date_convert_datetime():
    r = strf_date_convert("2023-01-05 10:30", None, "%Y-%m-%d %H:%M")
    assert r == datetime(2023, 1, 5, 10, 30)
    assert type(r) is datetime


def test_strf_date_convert_no_year():
    r = strf_date_convert("Feb 03", None, "%b %d")
    assert r == date(THIS_YEAR, 2, 3)
    r = strf_date_convert("03/02 10:00", None, "%d/%m %H:%M")
    assert r == datetime(THIS_YEAR, 2, 3, 10, 0)


def test_strf_date_convert_two_digit_year():
    r = strf_date_convert("99-02-03", None, "%y-%m-%d")
    assert r == date(1999, 2, 3)


def test_strf_date_convert_each_date_directive_is_date():
    for fmt, s in [
        ("%a %Y", "Mon 2023"),
        ("%A %Y", "Monday 2023"),
        ("%w %Y", "1 2023"),
        ("%d", "05"),
        ("%b", "Feb"),
        ("%B", "February"),
        ("%m", "02"),
        ("%y", "23"),
        ("%Y", "2023"),
        ("%j %Y", "032 2023"),
        ("%U %Y", "05 2023"),
        ("%W %Y", "05 2023"),
    ]:
        assert type(strf_date_convert(s, None, fmt)) is date, fmt


def test_strf_date_convert_each_time_directive_is_time():
    for fmt, s in [
        ("%H", "10"),
        ("%I %p", "10 PM"),
        ("%M", "30"),
        ("%S", "30"),
        ("%f", "123"),
        ("%z", "+0100"),
    ]:
        assert type(strf_date_convert(s, None, fmt)) is time, fmt


def test_strf_date_convert_timezone():
    r = strf_date_convert("10:30 +0100", None, "%H:%M %z")
    assert r.utcoffset() == timedelta(hours=1)


def test_strf_date_convert_neither():
    with pytest.raises(ValueError, match="Datetime not a date nor a time"):
        strf_date_convert("x", None, "x")


def test_get_regex_for_datetime_format():
    assert get_regex_for_datetime_format("%Y-%m-%d") == "[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"
    assert get_regex_for_datetime_format("abc") == "abc"


def test_dt_format_regexes():
    expected = {
        "%a": "(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)",
        "%A": "(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)",
        "%w": "[0-6]",
        "%d": "[0-9]{1,2}",
        "%b": "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        "%B": "(?:January|February|March|April|May|June|July|August|September|October|November|December)",
        "%m": "[0-9]{1,2}",
        "%y": "[0-9]{2}",
        "%Y": "[0-9]{4}",
        "%H": "[0-9]{1,2}",
        "%I": "[0-9]{1,2}",
        "%p": "(?:AM|PM)",
        "%M": "[0-9]{2}",
        "%S": "[0-9]{2}",
        "%f": "[0-9]{1,6}",
        "%z": "[+|-][0-9]{2}(:?[0-9]{2})?(:?[0-9]{2})?",
        "%j": "[0-9]{1,3}",
        "%U": "[0-9]{1,2}",
        "%W": "[0-9]{1,2}",
    }
    assert parse.dt_format_to_regex == expected


# ---------------------------------------------------------------------------
# extract_format
# ---------------------------------------------------------------------------


def test_extract_format_simple():
    f = extract_format("d", {})
    assert f["type"] == "d"
    assert f["fill"] is None
    assert f["align"] is None
    assert f["zero"] is False
    assert f["width"] == ""
    assert "grouping" not in f
    assert "precision" not in f


def test_extract_format_full():
    f = extract_format("*^+010,.3f", {})
    assert f["fill"] == "*"
    assert f["align"] == "^"
    assert f["zero"] is True
    assert f["width"] == "10"
    assert f["grouping"] == ","
    assert f["precision"] == "3"
    assert f["type"] == "f"


def test_extract_format_align_only():
    for a in "<>=^":
        f = extract_format(a + "5", {})
        assert f["align"] == a
        assert f["fill"] is None
        assert f["width"] == "5"
        assert f["type"] == ""


def test_extract_format_fill_align():
    f = extract_format("x<", {})
    assert f["fill"] == "x"
    assert f["align"] == "<"
    assert f["type"] == ""


def test_extract_format_single_char_not_alignment():
    f = extract_format("s", {})
    assert f["fill"] is None
    assert f["align"] is None
    assert f["type"] == "s"


def test_extract_format_signs():
    for sign in "+- ":
        f = extract_format(sign + "5d", {})
        assert f["width"] == "5"
        assert f["type"] == "d"


def test_extract_format_underscore_grouping():
    f = extract_format("_d", {})
    assert f["grouping"] == "_"
    assert f["type"] == "d"


def test_extract_format_precision_no_digits():
    f = extract_format(".f", {})
    assert f["precision"] == ""
    assert f["type"] == "f"


def test_extract_format_precision_then_type():
    f = extract_format("3.12s", {})
    assert f["width"] == "3"
    assert f["precision"] == "12"
    assert f["type"] == "s"


def test_extract_format_zero_only():
    f = extract_format("0", {})
    assert f["zero"] is True
    assert f["width"] == ""
    f = extract_format("00", {})
    assert f["zero"] is True
    assert f["width"] == "0"


def test_extract_format_bad_type():
    with pytest.raises(ValueError, match="not recognised"):
        extract_format("q", {})
    with pytest.raises(ValueError):
        parse.parse("{:q}", "x")


def test_extract_format_extra_type_and_datetime():
    assert extract_format("q", {"q": int})["type"] == "q"
    assert extract_format("%Y-%m", {})["type"] == "%Y-%m"
    assert extract_format("ti", {})["type"] == "ti"


# ---------------------------------------------------------------------------
# Parser basics
# ---------------------------------------------------------------------------


def test_parse_plain_literal():
    r = parse.parse("hello", "hello")
    assert r.fixed == ()
    assert r.named == {}
    assert parse.parse("hello", "hellox") is None
    assert parse.parse("hello", "xhello") is None


def test_parse_fixed():
    r = parse.parse("{} {}", "a b")
    assert r.fixed == ("a", "b")
    assert r[0] == "a"
    assert r[1] == "b"
    assert r[0:2] == ("a", "b")
    assert r[-1] == "b"


def test_parse_named():
    r = parse.parse("{a} {b}", "x y")
    assert r.named == {"a": "x", "b": "y"}
    assert r["a"] == "x"
    assert "a" in r
    assert "c" not in r


def test_parse_numeric_index_fixed():
    r = parse.parse("{0} {1}", "x y")
    assert r.fixed == ("x", "y")
    assert r.named == {}


def test_parse_underscore_start_is_fixed():
    r = parse.parse("{_a}", "x")
    assert r.fixed == ("x",)
    assert r.named == {}


def test_parse_braces_escape():
    r = parse.parse("{{}} {}", "{} x")
    assert r.fixed == ("x",)
    assert parse.parse("{{{}}}", "{5}").fixed == ("5",)
    assert parse.parse("{{}}", "xy") is None


def test_parse_regex_special_literals():
    assert parse.parse("a.b{}", "axb1") is None
    assert parse.parse("a.b{}", "a.b1").fixed == ("1",)
    assert parse.parse("(x)*+?{}", "(x)*+?5").fixed == ("5",)
    assert parse.parse("[a]^$|!{}\\", "[a]^$|!5\\").fixed == ("5",)


def test_parse_mixed():
    r = parse.parse("{} {name} {}", "1 2 3")
    assert r.fixed == ("1", "3")
    assert r.named == {"name": "2"}


def test_parse_multiline():
    r = parse.parse("{}", "a\nb")
    assert r.fixed == ("a\nb",)


def test_case_insensitive_default():
    assert parse.parse("Hello {}", "hELLO x").fixed == ("x",)
    assert parse.parse("Hello {}", "hELLO x", case_sensitive=True) is None
    assert parse.parse("Hello {}", "Hello x", case_sensitive=True).fixed == ("x",)


def test_case_sensitive_search_findall_compile():
    assert parse.search("A{:d}", "a1", case_sensitive=True) is None
    assert parse.search("A{:d}", "a1")[0] == 1
    assert list(parse.findall("A{:d}", "a1", case_sensitive=True)) == []
    assert len(list(parse.findall("A{:d}", "a1"))) == 1
    assert parse.compile("A{}", case_sensitive=True).parse("a1") is None
    assert parse.compile("A{}").parse("a1").fixed == ("1",)


def test_parse_no_match_returns_none():
    assert parse.parse("{:d}", "abc") is None


def test_parse_evaluate_result_false():
    m = parse.parse("{:d}", "12", evaluate_result=False)
    assert isinstance(m, Match)
    r = m.evaluate_result()
    assert isinstance(r, Result)
    assert r.fixed == (12,)
    assert parse.parse("{:d}", "x", evaluate_result=False) is None


def test_search_basic():
    r = parse.search("age: {:d}", "name: x, age: 42 years")
    assert r.fixed == (42,)
    assert parse.search("age: {:d}", "nothing") is None


def test_search_pos_endpos():
    assert parse.search("{:d}", "1 2 3", pos=1)[0] == 2
    assert parse.search("{:d}", "1 2 3", pos=3)[0] == 3
    assert parse.search("a{:d}", "a123", endpos=3)[0] == 12
    assert parse.search("a{:d}", "a123", pos=1) is None
    assert parse.search("a{:d}", "a123")[0] == 123


def test_search_evaluate_result_false():
    m = parse.search("{:d}", "x 5 y", evaluate_result=False)
    assert isinstance(m, Match)
    assert m.evaluate_result()[0] == 5
    assert parse.search("{:d}", "x", evaluate_result=False) is None


def test_parser_search_and_findall_methods():
    p = parse.compile("{:d}")
    assert p.search("ab 12")[0] == 12
    assert p.search("ab 12 34", pos=5)[0] == 34
    assert p.search("ab 12 34", 0, 4)[0] == 1
    assert [r[0] for r in p.findall("1 2 3")] == [1, 2, 3]
    assert [r[0] for r in p.findall("1 2 3", 1)] == [2, 3]
    assert [r[0] for r in p.findall("1 2 3", 0, 3)] == [1, 2]


def test_findall():
    rs = list(parse.findall(">{}<", "<p>some <b>bold</b> text</p>"))
    assert [r[0] for r in rs] == ["some ", "bold", " text"]


def test_findall_numbers():
    assert [r[0] for r in parse.findall("{:d}", "a1 b22 c333")] == [1, 22, 333]


def test_findall_pos_endpos():
    s = "1 2 3 4"
    assert [r[0] for r in parse.findall("{:d}", s, pos=2)] == [2, 3, 4]
    assert [r[0] for r in parse.findall("{:d}", s, endpos=3)] == [1, 2]
    assert [r[0] for r in parse.findall("{:d}", s, 2, 5)] == [2, 3]


def test_findall_evaluate_result_false():
    rs = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert all(isinstance(m, Match) for m in rs)
    assert [m.evaluate_result()[0] for m in rs] == [1, 2]


def test_findall_iterator_protocol():
    it = parse.findall("{:d}", "1 2")
    assert iter(it) is it
    assert next(it)[0] == 1
    assert it.next()[0] == 2
    with pytest.raises(StopIteration):
        next(it)


def test_result_iterator_attributes():
    p = parse.compile("{:d}")
    it = p.findall("1 2", 1, 2)
    assert it.parser is p
    assert it.string == "1 2"
    assert it.pos == 1
    assert it.endpos == 2
    assert it.evaluate_result is True


def test_compile_returns_parser():
    p = parse.compile("{a:d}")
    assert isinstance(p, Parser)
    assert p.parse("5").named == {"a": 5}
    assert p.format == "{a:d}"


def test_compile_extra_types():
    p = parse.compile("{:yes}", extra_types={"yes": lambda s: s == "y"})
    assert p.parse("y")[0] is True


def test_parser_repr():
    assert repr(Parser("{}" * 10)) == "<Parser %r>" % ("{}" * 10)
    assert repr(Parser("a" * 20)) == "<Parser %r>" % ("a" * 20)
    assert repr(Parser("a" * 21)) == "<Parser %r>" % ("a" * 17 + "...")
    assert repr(Parser("abc")) == "<Parser 'abc'>"


def test_parser_fields_properties():
    p = Parser("{} {a} {} {b.c}")
    assert p.named_fields == ["a", "b_c"]
    assert p.fixed_fields == [0, 2]
    nf = p.named_fields
    nf.append("zzz")
    assert p.named_fields == ["a", "b_c"]
    ff = p.fixed_fields
    ff.append(9)
    assert p.fixed_fields == [0, 2]


def test_parser_default_extra_types():
    p = Parser("{}")
    assert p._extra_types == {}


def test_parser_flags():
    assert Parser("x")._re_flags == re.IGNORECASE | re.DOTALL
    assert Parser("x", case_sensitive=True)._re_flags == re.DOTALL


def test_parser_regex_caching():
    p = Parser("{:d}")
    assert p._match_re is p._match_re
    assert p._search_re is p._search_re


def test_match_re_invalid_group_names():
    @with_pattern(r"(?P<foo>\d+)")
    def conv(s):
        return s

    p = Parser("{:c} {:c}", extra_types={"c": conv})
    with pytest.raises(NotImplementedError):
        p.parse("1 2")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_result_repr():
    r = Result((1,), {"a": 2}, {})
    assert repr(r) == "<Result (1,) {'a': 2}>"


def test_result_attrs():
    r = Result((1, 2), {"a": 3}, {"a": (0, 1)})
    assert r.fixed == (1, 2)
    assert r.named == {"a": 3}
    assert r.spans == {"a": (0, 1)}
    assert r[1] == 2
    assert r[:1] == (1,)
    assert r["a"] == 3
    with pytest.raises(KeyError):
        r["b"]
    with pytest.raises(IndexError):
        r[5]


def test_match_object():
    p = Parser("{:d}")
    m = p.parse("3", evaluate_result=False)
    assert m.parser is p
    assert m.match.group(0) == "3"


def test_spans():
    r = parse.parse("{} {name} {}", "ab cde f")
    assert r.spans == {0: (0, 2), "name": (3, 6), 1: (7, 8)}


def test_spans_after_complex_types():
    r = parse.parse("{:g} {:ti} {x:d} {}", "1.5 2011-01-02 3 zz")
    assert r.fixed[0] == 1.5
    assert r.fixed[1] == datetime(2011, 1, 2)
    assert r.fixed[2] == "zz"
    assert r.named == {"x": 3}
    assert r.spans[0] == (0, 3)
    assert r.spans[1] == (4, 14)
    assert r.spans["x"] == (15, 16)
    assert r.spans[2] == (17, 19)


def test_search_spans():
    r = parse.search("{:d}", "abc 123 def")
    assert r.fixed == (123,)
    assert r.spans == {0: (3, 7)}


# ---------------------------------------------------------------------------
# Named field handling
# ---------------------------------------------------------------------------


def test_dotted_names():
    r = parse.parse("{a.b} {c.d.e}", "1 2")
    assert r.named == {"a.b": "1", "c.d.e": "2"}


def test_hyphenated_names():
    r = parse.parse("{a-b} {c}", "1 2")
    assert r.named == {"a-b": "1", "c": "2"}


def test_bracket_names():
    r = parse.parse("{a[b]} {a[c]} {d[e][f]}", "1 2 3")
    assert r.named == {"a": {"b": "1", "c": "2"}, "d": {"e": {"f": "3"}}}


def test_bracket_names_spans():
    r = parse.parse("{a[b]} {c}", "1 2")
    assert r.spans == {"a[b]": (0, 1), "c": (2, 3)}


def test_group_name_collision_dot_then_underscore():
    r = parse.parse("{a.b} {a_b}", "1 2")
    assert r.named == {"a.b": "1", "a_b": "2"}


def test_group_name_collision_underscore_then_dot():
    p = Parser("{a_b} {a.b}")
    assert p.named_fields == ["a_b", "a__b"]
    assert p.parse("1 2").named == {"a_b": "1", "a.b": "2"}


def test_group_name_collision_hyphen():
    p = Parser("{a_b} {a-b}")
    assert p.named_fields == ["a_b", "a__b"]
    assert p.parse("1 2").named == {"a_b": "1", "a-b": "2"}
    p = Parser("{a-b} {a_b}")
    assert p.named_fields == ["a_b", "a__b"]
    assert p.parse("1 2").named == {"a-b": "1", "a_b": "2"}


def test_group_name_collision_multiple():
    p = Parser("{a.b} {a_b} {a__b}")
    assert p.named_fields == ["a_b", "a__b", "a____b"]
    assert p.parse("1 2 3").named == {"a.b": "1", "a_b": "2", "a__b": "3"}


def test_group_name_collision_triple_dot():
    p = Parser("{a_b} {a__b} {a.b}")
    assert p.named_fields == ["a_b", "a__b", "a___b"]


def test_group_name_collision_brackets_keyerror():
    with pytest.raises(KeyError):
        Parser("{a_b_} {a[b]}")


def test_group_name_maps():
    p = Parser("{a.b}")
    assert p._group_to_name_map == {"a_b": "a.b"}
    assert p._name_to_group_map == {"a.b": "a_b"}


def test_repeated_name():
    r = parse.parse("{a} {a}", "x x")
    assert r.named == {"a": "x"}
    assert parse.parse("{a} {a}", "x y") is None


def test_repeated_name_typed():
    r = parse.parse("{a:d} {a:d}", "1 1")
    assert r.named == {"a": 1}
    assert parse.parse("{a:d} {a:d}", "1 2") is None
    p = Parser("{a:d} {a:d}")
    assert p.named_fields == ["a"]


def test_repeated_name_type_mismatch():
    with pytest.raises(RepeatedNameError):
        Parser("{a:d} {a:w}")
    with pytest.raises(RepeatedNameError):
        Parser("{a} {a:d}")
    assert issubclass(RepeatedNameError, ValueError)
    assert issubclass(parse.TooManyFields, ValueError)


def test_repeated_name_followed_by_fixed():
    r = parse.parse("{a} {a} {}", "x x y")
    assert r.fixed == ("y",)


# ---------------------------------------------------------------------------
# Type conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,text,expected",
    [
        ("{:d}", "12", 12),
        ("{:d}", "-12", -12),
        ("{:d}", "+12", 12),
        ("{:d}", " 12", 12),
        ("{:d}", "0x1F", 31),
        ("{:d}", "0b101", 5),
        ("{:d}", "0o17", 15),
        ("{:d}", "-0x1F", -31),
        ("{:,d}", "1,234", 1234),
        ("{:_d}", "1_234", 1234),
        ("{:n}", "1,234,567", 1234567),
        ("{:n}", "1.234", 1234),
        ("{:n}", "-1,234", -1234),
        ("{:n}", "12", 12),
        ("{:b}", "0b101", 5),
        ("{:b}", "101", 5),
        ("{:b}", "-101", -5),
        ("{:o}", "0o17", 15),
        ("{:o}", "17", 15),
        ("{:x}", "0x1f", 31),
        ("{:x}", "ff", 255),
        ("{:X}", "FF", 255),
        ("{:X}", "0XFF", 255),
        ("{:%}", "50%", 0.5),
        ("{:%}", "12.5%", 0.125),
        ("{:%}", "-50%", -0.5),
        ("{:f}", "1.5", 1.5),
        ("{:f}", "-1.5", -1.5),
        ("{:f}", ".5", 0.5),
        ("{:f}", "+1.5", 1.5),
        ("{:.0f}", "20", 20.0),
        ("{:.0f}", "-20", -20.0),
        ("{:.2f}", "3.14", 3.14),
        ("{:F}", "1.5", Decimal("1.5")),
        ("{:F}", "-1.25", Decimal("-1.25")),
        ("{:e}", "1.5e3", 1500.0),
        ("{:e}", "1.5E-3", 0.0015),
        ("{:e}", "-1.5e3", -1500.0),
        ("{:E}", "1.5E+3", 1500.0),
        ("{:g}", "1.5", 1.5),
        ("{:g}", "1", 1.0),
        ("{:g}", "1e10", 1e10),
        ("{:g}", "-1.5e-2", -0.015),
        ("{:G}", "2.5E2", 250.0),
        ("{:l}", "abcXYZ", "abcXYZ"),
        ("{:w}", "ab_1", "ab_1"),
        ("{:W}", "!@", "!@"),
        ("{:s}", "  ", "  "),
        ("{:S}", "abc", "abc"),
        ("{:D}", "abc", "abc"),
    ],
)
def test_type_conversion(fmt, text, expected):
    r = parse.parse(fmt, text)
    assert r is not None
    assert r[0] == expected
    assert type(r[0]) is type(expected)


@pytest.mark.parametrize(
    "fmt,text",
    [
        ("{:d}", "abc"),
        ("{:d}", "1.5"),
        ("{:n}", "1234"),
        ("{:b}", "102"),
        ("{:o}", "18"),
        ("{:x}", "fg"),
        ("{:%}", "50"),
        ("{:f}", "1"),
        ("{:f}", "1."),
        ("{:.0f}", "1.5"),
        ("{:e}", "1.5"),
        ("{:g}", "a"),
        ("{:l}", "abc1"),
        ("{:w}", "a b"),
        ("{:W}", "ab"),
        ("{:s}", "ab"),
        ("{:S}", "a b"),
        ("{:D}", "a1"),
        ("{:d}", ""),
    ],
)
def test_type_no_match(fmt, text):
    assert parse.parse(fmt, text) is None


@pytest.mark.parametrize("fmt", ["{:f}", "{:e}", "{:g}", "{:F}"])
def test_float_nan_inf(fmt):
    import math

    r = parse.parse(fmt, "nan")
    assert math.isnan(r[0])
    assert parse.parse(fmt, "NAN") is not None
    assert parse.parse(fmt, "inf")[0] == float("inf")
    assert parse.parse(fmt, "INF")[0] == float("inf")
    assert parse.parse(fmt, "-inf")[0] == float("-inf")


def test_fixed_field_after_each_type():
    cases = [
        ("{:n} {}", "1,234 z", 1234),
        ("{:b} {}", "0b1 z", 1),
        ("{:o} {}", "0o7 z", 7),
        ("{:x} {}", "0xa z", 10),
        ("{:X} {}", "A z", 10),
        ("{:%} {}", "5% z", 0.05),
        ("{:f} {}", "1.5 z", 1.5),
        ("{:F} {}", "1.5 z", Decimal("1.5")),
        ("{:e} {}", "1.5e1 z", 15.0),
        ("{:g} {}", "1.5e1 z", 15.0),
        ("{:d} {}", "12 z", 12),
        ("{:l} {}", "ab z", "ab"),
        ("{:w} {}", "ab z", "ab"),
        ("{:%Y} {}", "2020 z", date(2020, 1, 1)),
        ("{:ti} {}", "2020-01-02 z", datetime(2020, 1, 2)),
        ("{:tg} {}", "2/1/2020 z", datetime(2020, 1, 2)),
        ("{:ta} {}", "1/2/2020 z", datetime(2020, 1, 2)),
        ("{:te} {}", "2 Jan 2020 10:00 +0000 z",
         datetime(2020, 1, 2, 10, tzinfo=FixedTzOffset(0, "+0000"))),
        ("{:th} {}", "02/Jan/2020:10:00:00 +0000 z",
         datetime(2020, 1, 2, 10, tzinfo=FixedTzOffset(0, "+0000"))),
        ("{:tc} {}", "Thu Jan 2 10:00:00 2020 z", datetime(2020, 1, 2, 10)),
        ("{:tt} {}", "10:00 z", time(10, 0)),
        ("{:ts} {}", "Jan 2 10:00:00 z", datetime(THIS_YEAR, 1, 2, 10)),
        ("{:>3} {}", "  abc z", "abc"),
        ("{:.2} {}", "ab z", "ab"),
    ]
    for fmt, text, first in cases:
        r = parse.parse(fmt, text)
        assert r is not None, fmt
        assert r.fixed == (first, "z"), fmt
        assert r.spans[1] == (len(text) - 1, len(text)), fmt


def test_named_after_each_type():
    r = parse.parse("{:g} {:tg} {a}", "1.5 2/1/2020 z")
    assert r.fixed == (1.5, datetime(2020, 1, 2))
    assert r.named == {"a": "z"}


def test_d_width():
    r = parse.parse("{:2d}{:2d}", "1234")
    assert r.fixed == (12, 34)
    assert parse.parse("{:2d}", "123") is None
    r = parse.parse("{:3d}", "  7")
    assert r[0] == 7
    r = parse.parse("{:10d}", "1234567890")
    assert r[0] == 1234567890


def test_d_zero_padded():
    assert parse.parse("{:03d}", "007")[0] == 7
    assert parse.parse("{:03d}", "-07")[0] == -7


def test_d_grouping_not_allowed_without_option():
    assert parse.parse("{:d}", "1,234") is None
    assert parse.parse("{:,d}", "1_234") is None


def test_d_hex_grouped():
    assert parse.parse("{:,d}", "0x1,0") is not None


def test_align_equals_numeric():
    assert parse.parse("{:x=5d}", "-xx12")[0] == -12
    assert parse.parse("{:=5d}", "-0012")[0] == -12
    assert parse.parse("{:=f}", "001.5")[0] == 1.5


def test_align_equals_default_fill_zero_with_f():
    # without a fill, "=" means zero padding which "{:f}" alone wouldn't allow
    # before a sign
    assert parse.parse("{:=8f}", "-00001.5")[0] == -1.5


def test_alignment_strings():
    assert parse.parse("{:>3}", "  abc")[0] == "abc"
    assert parse.parse("{:<3}", "abc  ")[0] == "abc"
    assert parse.parse("{:^3}", "  abc  ")[0] == "abc"
    assert parse.parse("{:>5}", "  abc")[0] == "  abc"
    assert parse.parse("{:>}", "  abc")[0] == "abc"
    assert parse.parse("{:<}", "abc  ")[0] == "abc"
    assert parse.parse("{:^}", "  abc  ")[0] == "abc"


def test_alignment_left_doesnt_strip_leading():
    assert parse.parse("{:<}", "  abc")[0] == "  abc"
    assert parse.parse("{:>}", "abc  ")[0] == "abc  "


def test_width_implies_right_align():
    assert parse.parse("{:3}", "  abc")[0] == "abc"
    assert parse.parse("{:5}", "  abc")[0] == "  abc"
    assert parse.parse("{:5}", "abc  ")[0] == "abc  "
    assert parse.parse("{:5d}", "   12")[0] == 12


def test_no_width_no_strip():
    assert parse.parse("{}", "  abc")[0] == "  abc"
    assert parse.parse("{:d}", "   12") is None


def test_fill_characters():
    assert parse.parse("{:*^3}", "**abc**")[0] == "abc"
    assert parse.parse("{:*^}", "**abc**")[0] == "abc"
    assert parse.parse("{:x<2}", "abxxx")[0] == "ab"
    assert parse.parse("{:x>2}", "xxxab")[0] == "ab"
    assert parse.parse("{:.>}", "..abc")[0] == "abc"
    assert parse.parse("{:.<}", "abc..")[0] == "abc"


@pytest.mark.parametrize("ch", list(".\\+?*[](){}^$"))
def test_fill_characters_escaped(ch):
    if ch in "{}^":
        # braces can't be used as fill in a format spec
        return
    fmt = "{:" + ch + ">}"
    assert parse.parse(fmt, ch * 2 + "abc")[0] == "abc"
    fmt = "{:" + ch + "<}"
    assert parse.parse(fmt, "abc" + ch * 2)[0] == "abc"


def test_fill_numeric():
    assert parse.parse("{:*>5d}", "***12")[0] == 12
    assert parse.parse("{:*<5d}", "12***")[0] == 12
    assert parse.parse("{:*^6d}", "**12**")[0] == 12


def test_string_width_precision():
    assert parse.parse("{:.2}", "ab")[0] == "ab"
    assert parse.parse("{:.2}", "abc") is None
    assert parse.parse("{:.2}", "") is None
    assert parse.parse("{:3.5}", "abcde")[0] == "abcde"
    assert parse.parse("{:3.5}", "abcdef") is None
    assert parse.parse("{:3.5}", "ab") is None
    assert parse.parse("{:3}", "ab") is None
    assert parse.parse("{:3}", "abcdef")[0] == "abcdef"


def test_string_precision_lazy():
    r = parse.parse("{:.3}{}", "abcdef")
    assert r.fixed == ("a", "bcdef")
    r = parse.parse("{:2.3}{}", "abcdef")
    assert r.fixed == ("ab", "cdef")
    r = parse.parse("{:2}{}", "abcdef")
    assert r.fixed == ("ab", "cdef")


def test_sign_formats():
    assert parse.parse("{:+d}", "+5")[0] == 5
    assert parse.parse("{:-d}", "-5")[0] == -5
    assert parse.parse("{: d}", " 5")[0] == 5


def test_extra_types():
    def yesno(s):
        return s == "yes"

    yesno.pattern = r"yes|no"
    r = parse.parse("{:yn} {}", "yes z", extra_types={"yn": yesno})
    assert r.fixed == (True, "z")
    assert parse.parse("{:yn}", "maybe", extra_types={"yn": yesno}) is None


def test_extra_types_without_pattern():
    r = parse.parse("{:up}", "abc", extra_types={"up": str.upper})
    assert r[0] == "ABC"


def test_extra_types_lazy_default_pattern():
    r = parse.parse("{:up}{}", "abc", extra_types={"up": str.upper})
    assert r.fixed == ("A", "bc")


def test_extra_types_named():
    r = parse.parse("{a:up}", "abc", extra_types={"up": str.upper})
    assert r.named == {"a": "ABC"}


def test_extra_types_with_regex_group_count():
    @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
    def rng(s):
        a, b = s.split("-")
        return (int(a), int(b))

    r = parse.parse("{:r} {}", "1-2 z", extra_types={"r": rng})
    assert r.fixed == ((1, 2), "z")
    assert r.spans[1] == (4, 5)


def test_extra_types_regex_group_count_none_with_groups_breaks():
    @with_pattern(r"(\d+)")
    def num(s):
        return int(s)

    # group count not declared: the fixed field after it gets the wrong group
    r = parse.parse("{:n2} {}", "12 z", extra_types={"n2": num})
    assert r.fixed[1] != "z"


def test_extra_types_regex_group_count_zero():
    @with_pattern(r"\d+", regex_group_count=0)
    def num(s):
        return int(s)

    r = parse.parse("{:n2} {}", "12 z", extra_types={"n2": num})
    assert r.fixed == (12, "z")


def test_extra_types_override_builtin():
    r = parse.parse("{:d}", "abc", extra_types={"d": str.upper})
    assert r[0] == "ABC"


def test_datetime_format_types():
    assert parse.parse("{:%Y-%m-%d}", "2023-01-05")[0] == date(2023, 1, 5)
    assert parse.parse("{:%H:%M}", "10:30")[0] == time(10, 30)
    assert parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-01-05 10:30:15")[0] == datetime(
        2023, 1, 5, 10, 30, 15
    )
    assert parse.parse("{:%b %d}", "Feb 03")[0] == date(THIS_YEAR, 2, 3)
    assert parse.parse("{:%Y-%m-%d}", "2023-1-5")[0] == date(2023, 1, 5)
    assert parse.parse("{:%Y}", "23") is None
    assert parse.parse("{:%H:%M:%S.%f}", "10:30:15.123")[0] == time(10, 30, 15, 123000)


def test_datetime_format_with_tz():
    r = parse.parse("{:%H:%M %z}", "10:30 +01:00")[0]
    assert r.utcoffset() == timedelta(hours=1)


# ---------------------------------------------------------------------------
# Built-in date/time types
# ---------------------------------------------------------------------------


def test_ti():
    p = lambda s: parse.parse("{:ti}", s)[0]  # noqa: E731
    assert p("2011-02-03") == datetime(2011, 2, 3)
    assert p("2011-02-03T04:05") == datetime(2011, 2, 3, 4, 5)
    assert p("2011-02-03 04:05:06") == datetime(2011, 2, 3, 4, 5, 6)
    assert p("2011-02-03T04:05:06.07") == datetime(2011, 2, 3, 4, 5, 6, 70000)
    assert p("2011-02-03T04:05:06Z") == datetime(
        2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(0, "UTC")
    )
    assert p("2011-02-03T04:05:06-05:30") == datetime(
        2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(-330, "-05:30")
    )
    assert p("2011-02-03T04:05:06+0530") == datetime(
        2011, 2, 3, 4, 5, 6, tzinfo=FixedTzOffset(330, "+0530")
    )


def test_ti_tz_offset_value():
    d = parse.parse("{:ti}", "2011-02-03T04:05:06+0130")[0]
    assert d.utcoffset() == timedelta(hours=1, minutes=30)
    d = parse.parse("{:ti}", "2011-02-03T04:05:06-0130")[0]
    assert d.utcoffset() == -timedelta(hours=1, minutes=30)


def test_tg():
    p = lambda s: parse.parse("{:tg}", s)[0]  # noqa: E731
    assert p("3/2/2011") == datetime(2011, 2, 3)
    assert p("3-2-2011") == datetime(2011, 2, 3)
    assert p("3-Feb-2011") == datetime(2011, 2, 3)
    assert p("3/February/2011") == datetime(2011, 2, 3)
    assert p("3/2/2011 10:21") == datetime(2011, 2, 3, 10, 21)
    assert p("3/2/2011 10:21:36") == datetime(2011, 2, 3, 10, 21, 36)
    assert p("3/2/2011 10:21 PM") == datetime(2011, 2, 3, 22, 21)
    assert p("3/2/2011 10:21 AM") == datetime(2011, 2, 3, 10, 21)
    assert p("3/2/2011 12:21 AM") == datetime(2011, 2, 3, 0, 21)
    assert p("3/2/2011 12:21 PM") == datetime(2011, 2, 3, 12, 21)
    assert p("3/2/2011 10:21 PM +1000") == datetime(
        2011, 2, 3, 22, 21, tzinfo=FixedTzOffset(600, "+1000")
    )
    assert p("3/2/2011 10:21 -5:30") == datetime(
        2011, 2, 3, 10, 21, tzinfo=FixedTzOffset(-330, "-5:30")
    )
    assert p("3/2/2011 10:21 +530") == datetime(
        2011, 2, 3, 10, 21, tzinfo=FixedTzOffset(330, "+530")
    )


def test_ta():
    p = lambda s: parse.parse("{:ta}", s)[0]  # noqa: E731
    assert p("2/3/2011") == datetime(2011, 2, 3)
    assert p("Feb/3/2011") == datetime(2011, 2, 3)
    assert p("2-3-2011 10:21:36 PM") == datetime(2011, 2, 3, 22, 21, 36)
    assert p("2/3/2011 10:21 AM -0100") == datetime(
        2011, 2, 3, 10, 21, tzinfo=FixedTzOffset(-60, "-0100")
    )


def test_te():
    p = lambda s: parse.parse("{:te}", s)[0]  # noqa: E731
    assert p("Mon, 20 Jan 1972 10:21:36 +1000") == datetime(
        1972, 1, 20, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000")
    )
    assert p("20 Jan 1972 10:21:36 -0130") == datetime(
        1972, 1, 20, 10, 21, 36, tzinfo=FixedTzOffset(-90, "-0130")
    )
    assert parse.parse("{:te}", "20 Jan 1972 10:21:36") is None


def test_th():
    p = lambda s: parse.parse("{:th}", s)[0]  # noqa: E731
    assert p("21/Nov/2011:10:21:36 +1000") == datetime(
        2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(600, "+1000")
    )
    assert p("21-Nov-2011:10:21:36 -1000") == datetime(
        2011, 11, 21, 10, 21, 36, tzinfo=FixedTzOffset(-600, "-1000")
    )


def test_tc():
    p = lambda s: parse.parse("{:tc}", s)[0]  # noqa: E731
    assert p("Sun Sep 16 01:03:52 1973") == datetime(1973, 9, 16, 1, 3, 52)
    assert p("Sun Sep 6 01:03:52 1973") == datetime(1973, 9, 6, 1, 3, 52)


def test_tt():
    p = lambda s: parse.parse("{:tt}", s)[0]  # noqa: E731
    assert p("10:21") == time(10, 21)
    assert p("10:21:36") == time(10, 21, 36)
    assert p("10:21:36 PM") == time(22, 21, 36)
    assert p("10:21:36 AM") == time(10, 21, 36)
    assert p("12:21 AM") == time(0, 21)
    assert p("10:21:36 PM -5:30") == time(
        22, 21, 36, tzinfo=FixedTzOffset(-330, "-5:30")
    )
    assert p("10:21:36.123 +0100") == time(
        10, 21, 36, 123000, tzinfo=FixedTzOffset(60, "+0100")
    )


def test_ts():
    p = lambda s: parse.parse("{:ts}", s)[0]  # noqa: E731
    assert p("Nov  9 03:37:44") == datetime(THIS_YEAR, 11, 9, 3, 37, 44)
    assert p("Dec 31 23:59:59") == datetime(THIS_YEAR, 12, 31, 23, 59, 59)


def test_date_types_named():
    r = parse.parse("{when:ti} at {where}", "2011-02-03 at home")
    assert r.named == {"when": datetime(2011, 2, 3), "where": "home"}


def test_multiple_date_types():
    r = parse.parse("{:tg} {:ta} {}", "3/2/2011 2/3/2012 z")
    assert r.fixed == (datetime(2011, 2, 3), datetime(2012, 2, 3), "z")


# ---------------------------------------------------------------------------
# Expression generation details
# ---------------------------------------------------------------------------


def test_expression_simple():
    assert Parser("{}")._expression == "(.+?)"
    assert Parser("{a}")._expression == "(?P<a>.+?)"
    assert Parser("{{}}")._expression == r"\{\}"
    assert Parser("a.b")._expression == r"a\.b"


def test_expression_types():
    assert Parser("{:l}")._expression == "([A-Za-z]+)"
    assert Parser("{:w}")._expression == r"(\w+)"
    assert Parser("{:5}")._expression == r" *(.{5,}?)"
    assert Parser("{:.5}")._expression == r"(.{1,5}?)"
    assert Parser("{:2.5}")._expression == r" *(.{2,5}?)"
    assert Parser("{:<}")._expression == r"(.+?) *"
    assert Parser("{:^}")._expression == r" *(.+?) *"
    assert Parser("{:>}")._expression == r" *(.+?)"
    assert Parser("{:%}")._expression == r"([-+ ]?\d+(\.\d+)?%)"
    assert Parser("{:n}")._expression == r"([-+ ]?\d{1,3}([,.]\d{3})*)"
    assert Parser("{:b}")._expression == r"([-+ ]?(0[bB])?[01]+)"
    assert Parser("{:o}")._expression == r"([-+ ]?(0[oO])?[0-7]+)"
    assert Parser("{:x}")._expression == r"([-+ ]?(0[xX])?[0-9a-fA-F]+)"
    assert Parser("{:f}")._expression == r"([-+ ]?(?:\d*\.\d+|nan|NAN|inf|INF))"
    assert Parser("{:.0f}")._expression == r"([-+ ]?(?:\d+|nan|NAN|inf|INF))"
    assert Parser("{:=f}")._expression == r"([-+ ]?0*(?:\d*\.\d+|nan|NAN|inf|INF))"
    assert (
        Parser("{:d}")._expression
        == r"([-+ ]?[0-9]+|[-+ ]?0[xX][0-9a-fA-F]+|[-+ ]?0[bB][01]+|[-+ ]?0[oO][0-7]+)"
        .replace("(", "([-+ ]?", 1)
    )


def test_expression_d_width_grouping():
    assert Parser("{:3,d}")._expression == (
        r" *([-+ ]?[-+ ]?[0-9,]{1,3}|[-+ ]?0[xX][0-9a-fA-F,]{1,3}"
        r"|[-+ ]?0[bB][01,]{1,3}|[-+ ]?0[oO][0-7,]{1,3})"
    )


def test_group_index_counts():
    assert Parser("{:n}")._group_index == 2
    assert Parser("{:b}")._group_index == 2
    assert Parser("{:o}")._group_index == 2
    assert Parser("{:x}")._group_index == 2
    assert Parser("{:%}")._group_index == 2
    assert Parser("{:f}")._group_index == 1
    assert Parser("{:e}")._group_index == 1
    assert Parser("{:g}")._group_index == 3
    assert Parser("{:d}")._group_index == 1
    assert Parser("{:ti}")._group_index == 8
    assert Parser("{:tg}")._group_index == 10
    assert Parser("{:ta}")._group_index == 10
    assert Parser("{:te}")._group_index == 9
    assert Parser("{:th}")._group_index == 7
    assert Parser("{:tc}")._group_index == 9
    assert Parser("{:tt}")._group_index == 6
    assert Parser("{:ts}")._group_index == 6
    assert Parser("{}")._group_index == 1
    assert Parser("{a}")._group_index == 1
    assert Parser("{a:d}")._group_index == 1
    assert Parser("{:5}")._group_index == 1


def test_parse_re():
    assert parse.PARSE_RE.split("a{}b{x:d}c{{d}}") == [
        "a", "{}", "b", "{x:d}", "c", "{{", "d", "}}", ""
    ]
    assert parse.PARSE_RE.split("{a.b[c]}") == ["", "{a.b[c]}", ""]
    assert parse.PARSE_RE.split("{a-b}") == ["", "{a-b}", ""]


def test_module_all_and_version():
    assert parse.__all__ == ["parse", "search", "findall", "with_pattern"]
    assert parse.__version__ == "1.22.3"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_date_convert_long_tz_offset_strings():
    t = date_convert("", FakeMatch("1:00", "+013000"), hms=0, tz=1)
    assert t.tzinfo.utcoffset(None) == timedelta(minutes=90)
    t = date_convert("", FakeMatch("1:00", "+13000"), hms=0, tz=1)
    assert t.tzinfo.utcoffset(None) == timedelta(hours=13)


def test_strf_date_convert_unknown_directive_is_neither():
    with pytest.raises(ValueError, match=r"^Datetime not a date nor a time\?$"):
        strf_date_convert("10:30:00", None, "%X")


def test_extract_format_zero_then_type_X():
    f = extract_format("0X", {})
    assert f["zero"] is True
    assert f["fill"] is None
    assert f["align"] is None
    assert f["type"] == "X"


def test_extract_format_fill_align_width():
    f = extract_format("x<15d", {})
    assert f["fill"] == "x"
    assert f["align"] == "<"
    assert f["width"] == "15"
    assert f["type"] == "d"


def test_extract_format_error_message():
    with pytest.raises(ValueError, match=r"^format spec 'q' not recognised$"):
        extract_format("q", {})


def test_parser_logs_expression(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="parse"):
        Parser("{}")
    msgs = [r.getMessage() for r in caplog.records if r.name == "parse"]
    assert "format '{}' -> '(.+?)'" in msgs


def test_parser_search_default_pos():
    assert parse.compile("{:d}").search("1 2")[0] == 1
    assert parse.compile("{:d}").search("1 2").spans == {0: (0, 1)}


def test_literal_ending_with_brace():
    r = parse.parse("x}", "x}")
    assert r is not None
    assert r.fixed == ()
    assert parse.parse("x}", "x") is None


def test_duplicate_group_name_message():
    with pytest.raises(KeyError) as ei:
        Parser("{a_b_} {a[b]}")
    assert ei.value.args == ("duplicated group name 'a[b]'",)


def test_repeated_name_error_message():
    with pytest.raises(RepeatedNameError) as ei:
        Parser("{a} {a:d}")
    assert str(ei.value) == (
        "field type 'd' for field \"a\" does not match previous seen type ''"
    )


@pytest.mark.parametrize(
    "fmt,text,expected",
    [
        ("{:n}", "1,234", 1234),
        ("{:b}", "101", 5),
        ("{:o}", "17", 15),
        ("{:x}", "ff", 255),
        ("{:%}", "50%", 0.5),
        ("{:f}", "1.5", 1.5),
        ("{:e}", "1.5e1", 15.0),
        ("{:g}", "1.5e1", 15.0),
        ("{:d}", "12", 12),
        ("{:ti}", "2020-01-02 03:04", datetime(2020, 1, 2, 3, 4)),
        ("{:tg}", "2/1/2020 03:04 PM", datetime(2020, 1, 2, 15, 4)),
        ("{:ta}", "1/2/2020 03:04 PM", datetime(2020, 1, 2, 15, 4)),
        ("{:te}", "2 Jan 2020 10:00 +0000",
         datetime(2020, 1, 2, 10, tzinfo=FixedTzOffset(0, "+0000"))),
        ("{:th}", "02/Jan/2020:10:00:00 +0000",
         datetime(2020, 1, 2, 10, tzinfo=FixedTzOffset(0, "+0000"))),
        ("{:tc}", "Thu Jan 2 10:00:00 2020", datetime(2020, 1, 2, 10)),
        ("{:tt}", "10:00 PM", time(22, 0)),
        ("{:ts}", "Jan 2 10:00:00", datetime(THIS_YEAR, 1, 2, 10)),
    ],
)
def test_type_between_fixed_fields(fmt, text, expected):
    r = parse.parse("{} " + fmt + " {}", "a " + text + " z")
    assert r is not None
    assert r.fixed == ("a", expected, "z")
    r = parse.parse("{} {} " + fmt + " {}", "a b " + text + " z")
    assert r.fixed == ("a", "b", expected, "z")


def test_extra_type_group_count_after_field():
    @with_pattern(r"((\d+)-(\d+))", regex_group_count=3)
    def rng(s):
        a, b = s.split("-")
        return (int(a), int(b))

    r = parse.parse("{} {:r} {}", "a 1-2 z", extra_types={"r": rng})
    assert r.fixed == ("a", (1, 2), "z")


@pytest.mark.parametrize("fmt", ["{:e}", "{:g}"])
def test_float_case_sensitive(fmt):
    import math

    assert parse.parse(fmt, "1.5e3", case_sensitive=True)[0] == 1500.0
    assert parse.parse(fmt, "1.5E3", case_sensitive=True)[0] == 1500.0
    assert math.isnan(parse.parse(fmt, "nan", case_sensitive=True)[0])
    assert math.isnan(parse.parse(fmt, "NAN", case_sensitive=True)[0])
    assert parse.parse(fmt, "inf", case_sensitive=True)[0] == float("inf")
    assert parse.parse(fmt, "INF", case_sensitive=True)[0] == float("inf")
    assert parse.parse(fmt, "Inf", case_sensitive=True) is None


def test_ti_case_sensitive():
    p = parse.compile("{:ti}", case_sensitive=True)
    assert p.parse("2011-02-03T04:05Z")[0] == datetime(
        2011, 2, 3, 4, 5, tzinfo=FixedTzOffset(0, "UTC")
    )
    assert p.parse("2011-02-03t04:05") is None
    assert p.parse("2011-02-03 04:05z") is None


def test_fill_X_not_escaped():
    assert parse.parse("{:X<}", "abXX")[0] == "ab"
    assert parse.parse("{:X>}", "XXab")[0] == "ab"


def test_result_iterator_default_evaluates():
    p = parse.compile("{:d}")
    it = parse.ResultIterator(p, "1 2", 0, 3)
    r = next(it)
    assert isinstance(r, Result)
    assert r[0] == 1


def test_search_findall_extra_types():
    def yesno(s):
        return s == "yes"

    yesno.pattern = r"yes|no"
    et = {"yn": yesno}
    assert parse.search("{:yn}", "x yes y", extra_types=et)[0] is True
    assert [r[0] for r in parse.findall("{:yn}", "yes no yes", extra_types=et)] == [
        True, False, True
    ]
