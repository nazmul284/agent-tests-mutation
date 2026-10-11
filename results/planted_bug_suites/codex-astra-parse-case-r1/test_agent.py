"""Behavioral regression tests derived from the parser implementation."""

import math
import pickle
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import parse


@pytest.mark.parametrize("literal", ["", "plain", ".^$*+?[]()|!\\", "{{literal}}"])
def test_literals_and_regex_escaping(literal):
    text = literal.replace("{{", "{").replace("}}", "}")
    result = parse.parse(literal, text)
    assert result.fixed == ()
    assert result.named == {}
    assert result.spans == {}
    assert parse.parse(literal, text + "!") is None
    assert parse.parse(literal, "!" + text) is None


def test_fields_results_and_absolute_spans():
    result = parse.search("{}:{count:d}/{name}", "xx abc:007/Z yy", endpos=12)
    assert result.fixed == ("xx abc",)
    assert result.named == {"count": 7, "name": "Z"}
    assert result.spans == {0: (0, 6), "count": (7, 10), "name": (11, 12)}
    assert result[0] == "xx abc"
    assert result[-1] == "xx abc"
    assert result[:] == ("xx abc",)
    assert result["count"] == 7
    assert "count" in result
    assert "absent" not in result
    with pytest.raises(KeyError):
        result["absent"]
    with pytest.raises(IndexError):
        result[1]
    assert "count" in repr(result)


def test_multiline_nonempty_and_nongreedy_fields():
    assert parse.parse("{}|{}", "a|b|c").fixed == ("a", "b|c")
    assert parse.parse("<{}>", "<a\nb>")[0] == "a\nb"
    assert parse.parse("<{}>", "<>") is None
    assert parse.parse("abc", "abc\n") is None
    assert parse.parse("{2}/{0}/{}", "a/b/c").fixed == ("a", "b", "c")


@pytest.mark.parametrize("sensitive", [True, False])
def test_explicit_case_setting_across_apis(sensitive):
    expected = [] if sensitive else [3]
    assert [r[0] for r in parse.findall("ID={:d}", "id=3", case_sensitive=sensitive)] == expected
    for operation in (parse.parse, parse.search):
        result = operation("ID={:d}", "id=3", case_sensitive=sensitive)
        assert (result is None) == sensitive
        if result is not None:
            assert result.fixed == (3,)
    parser = parse.compile("ID={:d}", case_sensitive=sensitive)
    assert (parser.parse("id=3") is None) == sensitive


def test_search_and_iteration_bounds():
    text = "x [12] [34] [56] y"
    parser = parse.compile("[{:d}]")
    assert parser.parse(text) is None
    assert parser.search(text)[0] == 12
    assert parser.search(text, 6, 11).spans == {0: (8, 10)}
    assert parser.search(text, 6, 10) is None
    assert parse.search("[{:d}]", text, 6, 11)[0] == 34
    for iterator in (parser.findall(text, 6, 16), parse.findall("[{:d}]", text, 6, 16)):
        assert iter(iterator) is iterator
        assert [r[0] for r in iterator] == [34, 56]
        with pytest.raises(StopIteration):
            next(iterator)
    assert list(parser.findall("none")) == []
    assert parser.search("none") is None


@pytest.mark.parametrize("spec,text,expected", [
    ("d", "0", 0), ("d", "-42", -42), ("d", "+42", 42),
    ("d", "0XAb", 171), ("d", "-0b101", -5), ("d", "+0o17", 15),
    (",d", "1,234,567", 1234567), ("_d", "-1_234", -1234),
    ("n", "1.234.567", 1234567), ("n", "-1,234", -1234),
    ("b", "0B101", 5), ("b", "-101", -5), ("o", "0O17", 15),
    ("x", "0xAf", 175), ("X", "-AF", -175),
    ("%", "12.5%", .125), ("%", "-25%", -.25),
    ("f", "-.125", -.125), ("f", "+12.50", 12.5),
    (".0f", "-20", -20.0), (".0F", "20", Decimal("20")),
    ("F", "12.340", Decimal("12.340")),
    ("e", "-1.25e+2", -125.0), ("E", "1.25E-2", .0125),
    ("g", "12", 12.0), ("G", "-1.5E2", -150.0),
])
def test_numeric_conversions_and_capture_accounting(spec, text, expected):
    # Adjacent fields detect accidental exposure or miscounting of inner groups.
    result = parse.parse("{:" + spec + "}|{tail:d}|{}", text + "|7|end")
    assert result.fixed == (expected, "end")
    assert type(result[0]) is type(expected)
    assert result.named == {"tail": 7}
    assert result.spans[0] == (0, len(text))
    assert result.spans[1] == (len(text) + 3, len(text) + 6)


@pytest.mark.parametrize("spec", ["f", "F", "e", "E", "g", "G"])
@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_special_numbers(spec, text):
    value = parse.parse("{:" + spec + "}", text)[0]
    if text == "nan":
        assert math.isnan(value)
    else:
        assert value == float(text)


@pytest.mark.parametrize("spec,text", [("d", "abc"), ("b", "102"), ("o", "89"),
    ("x", "GH"), ("n", "12,34"), ("%", "12"), ("f", "12"),
    (".0f", "12.1"), ("e", "12.5"), ("2d", "123")])
def test_reject_invalid_numeric_input(spec, text):
    assert parse.parse("{:" + spec + "}", text) is None


@pytest.mark.parametrize("spec,text,expected", [
    (">", "   abc", "abc"), ("<", "abc   ", "abc"),
    ("^", "  abc  ", "abc"), ("*>5l", "**abc", "abc"),
    (".^5l", ".abc.", "abc"), ("+<5l", "abc++", "abc"),
    ("05d", "00042", 42), ("=5d", "-0042", -42),
    ("8d", "      42", 42), (".3", "abc", "abc"),
    ("2.4", "abc", "abc"), ("3", "abcd", "abcd"),
])
def test_padding_width_and_precision(spec, text, expected):
    assert parse.parse("{:" + spec + "}", text)[0] == expected


@pytest.mark.parametrize("spec,text", [(".3", "abcd"), ("2.4", "a"), ("2.4", "abcde"), ("3", "ab")])
def test_string_length_constraints(spec, text):
    assert parse.parse("{:" + spec + "}", text) is None


@pytest.mark.parametrize("spec,good,bad", [("w", "café_2", "a-b"), ("W", "!?", "abc"),
    ("s", " \t\n", "x"), ("S", "abc!", "a b"), ("D", "abc!", "12"),
    ("l", "Az", "abc2")])
def test_character_classes(spec, good, bad):
    assert parse.parse("{:" + spec + "}", good)[0] == good
    assert parse.parse("{:" + spec + "}", bad) is None


def test_repeated_names_and_name_collisions():
    parser = parse.compile("{a.b}/{a_b}/{a-b}/{a.b}")
    result = parser.parse("one/two/three/one")
    assert result.named == {"a.b": "one", "a_b": "two", "a-b": "three"}
    assert result.spans["a.b"] == (0, 3)
    assert parser.parse("one/two/three/four") is None
    assert parse.parse("{n:d}/{n:d}", "01/1") is None
    assert parse.parse("{n:d}/{n:d}", "01/01")["n"] == 1
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{n:d}/{n:f}")


def test_nested_dictionary_fields():
    result = parse.parse("{user[name]}:{user[stats][age]:d}:{user[id]:d}", "Ada:37:9")
    assert result.named == {"user": {"name": "Ada", "stats": {"age": 37}, "id": 9}}
    assert result.spans == {"user[name]": (0, 3), "user[stats][age]": (4, 6), "user[id]": (7, 8)}


def test_custom_patterns_groups_and_delayed_conversion():
    calls = []

    @parse.with_pattern(r"(\d+)-(\d+)", regex_group_count=2)
    def pair(text):
        calls.append(text)
        return tuple(map(int, text.split("-")))

    assert pair.pattern == r"(\d+)-(\d+)"
    assert pair.regex_group_count == 2
    parser = parse.compile("{:pair}/{named:pair}/{:d}", {"pair": pair})
    match = parser.parse("1-2/3-4/5", evaluate_result=False)
    assert isinstance(match, parse.Match)
    assert match.parser is parser
    assert calls == []
    result = match.evaluate_result()
    assert result.fixed == ((1, 2), 5)
    assert result.named == {"named": (3, 4)}
    assert result.spans == {0: (0, 3), "named": (4, 7), 1: (8, 9)}
    assert calls == ["1-2", "3-4"]
    assert parser.parse("x/3-4/5") is None


@pytest.mark.parametrize("api", [parse.parse, parse.search, parse.findall])
def test_wrappers_forward_custom_types_and_lazy_flag(api):
    calls = []

    @parse.with_pattern(r"[a-z]+")
    def upper(text):
        calls.append(text)
        return text.upper()

    result = api("<{:upper}>", "<abc>", extra_types={"upper": upper}, evaluate_result=False)
    if api is parse.findall:
        result = next(result)
    assert calls == []
    assert result.evaluate_result().fixed == ("ABC",)
    assert calls == ["abc"]


def test_custom_converter_without_pattern_and_errors():
    assert parse.parse("{:rev}", "abc", {"rev": lambda s: s[::-1]})[0] == "cba"
    assert parse.parse("{:d}", "abc", {"d": str.upper})[0] == "ABC"
    with pytest.raises(ValueError):
        parse.compile("{:unknown}")
    with pytest.raises(ValueError):
        parse.parse("{:number}", "abc", {"number": int})


@pytest.mark.parametrize("spec,text,expected", [
    ("ti", "2024-02-29", datetime(2024, 2, 29)),
    ("ti", "2024-02-29T13:14:15.1234567Z", datetime(2024, 2, 29, 13, 14, 15, 123456, timezone.utc)),
    ("ti", "2024-02-29 13:14 +0530", datetime(2024, 2, 29, 13, 14, tzinfo=timezone(timedelta(hours=5, minutes=30)))),
    ("tg", "29/February/2024 12:15 AM", datetime(2024, 2, 29, 0, 15)),
    ("ta", "Feb/29/2024 12:15 PM", datetime(2024, 2, 29, 12, 15)),
    ("tg", "29-2-2024 1:15 PM -5:30", datetime(2024, 2, 29, 13, 15, tzinfo=timezone(-timedelta(hours=5, minutes=30)))),
    ("te", "Thu, 29 Feb 2024 13:14:15 +0000", datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone.utc)),
    ("th", "29/Feb/2024:13:14:15 -0530", datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone(-timedelta(hours=5, minutes=30)))),
    ("tc", "Thu Feb 29 13:14:15 2024", datetime(2024, 2, 29, 13, 14, 15)),
    ("tt", "12:15 AM", time(0, 15)), ("tt", "12:15 PM", time(12, 15)),
    ("tt", "1:15:02.12 PM +530", time(13, 15, 2, 120000, timezone(timedelta(hours=5, minutes=30)))),
    ("%Y-%m-%d", "2024-02-29", date(2024, 2, 29)),
    ("%H:%M:%S", "13:14:15", time(13, 14, 15)),
    ("%Y/%m/%d %H:%M", "2024/02/29 13:14", datetime(2024, 2, 29, 13, 14)),
])
def test_dates_and_group_offsets(spec, text, expected):
    result = parse.parse("{:d}|{when:" + spec + "}|{}|{:d}", "8|" + text + "|end|9")
    assert result.fixed == (8, "end", 9)
    assert result["when"] == expected
    assert type(result["when"]) is type(expected)
    assert result.spans["when"] == (2, len(text) + 2)
    # Exercise conversion indexed by positional group, too.
    assert parse.parse("{:" + spec + "}|{:d}", text + "|9").fixed == (expected, 9)


def test_yearless_dates_and_invalid_calendar_date():
    year = datetime.today().year
    assert parse.parse("{:ts}", "Jan 2 03:04:05")[0] == datetime(year, 1, 2, 3, 4, 5)
    assert parse.parse("{:%m-%d}", "01-02")[0] == date(year, 1, 2)
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2023-02-29")


def test_timezone_contract():
    tz = parse.FixedTzOffset(-90, "west")
    assert tz.utcoffset(None) == timedelta(minutes=-90)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "west"
    assert tz == parse.FixedTzOffset(-90, "west")
    assert tz != parse.FixedTzOffset(90, "west")
    assert tz != parse.FixedTzOffset(-90, "other")
    assert tz != object()
    assert "west" in repr(tz)


def test_reusable_parser_properties_and_pickle():
    parser = parse.compile("{name}:{:d}:{:g}")
    assert parser.format == "{name}:{:d}:{:g}"
    named = parser.named_fields
    fixed = parser.fixed_fields
    named.clear()
    fixed.clear()
    assert parser.named_fields == ["name"]
    assert len(parser.fixed_fields) == 2
    for candidate in (parser, pickle.loads(pickle.dumps(parser))):
        for text, values in [("Ada:3:1.5", (3, 1.5)), ("Bob:4:2e2", (4, 200.0))]:
            assert candidate.parse(text).fixed == values
            assert candidate.search("!" + text).fixed == values
    assert "..." in repr(parse.compile("a" * 30))
    assert repr(parse.compile("abc")) == "<Parser 'abc'>"
