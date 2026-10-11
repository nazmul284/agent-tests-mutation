"""Behavioral tests written from the parser implementation and its API docstrings."""
import math
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
import parse


def test_mixed_results_and_absolute_spans():
    result = parse.search("{}:{count:d}:{user[name]}:{user[id]:d}", "xx apple:12:Ada:7 yy", pos=3)
    assert result.fixed == ("apple",)
    assert result.named == {"count": 12, "user": {"name": "Ada", "id": 7}}
    assert result.spans == {0: (3, 8), "count": (9, 11), "user[name]": (12, 15), "user[id]": (16, 17)}
    assert result[0] == "apple"
    assert result[-1] == "apple"
    assert result[:] == ("apple",)
    assert result["user"]["id"] == 7
    assert "user" in result and "missing" not in result
    with pytest.raises(KeyError):
        result["missing"]
    with pytest.raises(IndexError):
        result[1]


@pytest.mark.parametrize("literal", ["", "a.b*+?^$|()[]!\\", "plain text", "line\none"])
def test_literals_are_not_regex(literal):
    result = parse.parse(literal, literal)
    assert result is not None
    assert (result.fixed, result.named, result.spans) == ((), {}, {})
    assert parse.parse(literal, literal + "x") is None
    assert parse.parse(literal, "x" + literal) is None


def test_braces_multiline_and_nonempty_fields():
    assert parse.parse("{{{value}}}", "{hello}")["value"] == "hello"
    assert parse.parse("{{}}", "{}").fixed == ()
    assert parse.parse("{}", "a\nb").fixed == ("a\nb",)
    assert parse.parse("{}", "") is None
    assert parse.parse("{}|{}", "a|b|c").fixed == ("a", "b|c")
    assert parse.parse("hello", "hello\n") is None


@pytest.mark.parametrize("api", [parse.parse, parse.search, parse.findall])
def test_case_sensitivity(api):
    def results(**kw):
        value = api("HELLO {word:l}!", "hello World!", **kw)
        return list(value) if api is parse.findall else value
    matched = results()
    result = matched[0] if api is parse.findall else matched
    assert result["word"] == "World"
    assert results(case_sensitive=True) == ([] if api is parse.findall else None)


def test_search_and_iteration_bounds():
    parser = parse.compile("<{n:d}>")
    text = "x<12>y<34>z<56>"
    assert parser.parse(text) is None
    assert parser.search(text)["n"] == 12
    assert parser.search(text, 5, 10)["n"] == 34
    assert parser.search(text, 5, 9) is None
    assert parse.search("<{n:d}>", text, pos=5, endpos=10).spans == {"n": (7, 9)}
    for iterator in (parser.findall(text, 5, 10), parse.findall("<{n:d}>", text, pos=5, endpos=10)):
        assert iter(iterator) is iterator
        assert next(iterator)["n"] == 34
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
    assert [r["n"] for r in parser.findall(text)] == [12, 34, 56]
    assert list(parser.findall("nothing")) == []


@pytest.mark.parametrize("spec,text,expected", [
    ("d", "42", 42), ("d", "-42", -42), ("d", "+42", 42),
    ("d", "0XfF", 255), ("d", "-0b101", -5), ("d", "+0o17", 15),
    (",d", "1,234,567", 1234567), ("_d", "-1_234", -1234),
    ("b", "0B101", 5), ("b", "-101", -5), ("o", "0O17", 15),
    ("o", "+17", 15), ("x", "0xFf", 255), ("X", "-FF", -255),
    ("n", "1,234,567", 1234567), ("n", "-12.345", -12345),
    ("%", "12.5%", .125), ("%", "-50%", -.5),
    ("f", "-.25", -.25), ("f", "+12.50", 12.5),
    (".0f", "-20", -20.0), (".0F", "+20", Decimal("20")),
    ("F", "0.100000000000000001", Decimal("0.100000000000000001")),
    ("e", "-1.25e+2", -125.0), ("E", "1.25E-2", .0125),
    ("g", "42", 42.0), ("G", "-2.5E3", -2500.0),
])
def test_numeric_types_and_following_group(spec, text, expected):
    result = parse.parse("{:" + spec + "}|{tail:d}|{}", text + "|9|end")
    assert result.fixed == (expected, "end")
    assert type(result[0]) is type(expected)
    assert result.named == {"tail": 9}
    assert result.spans == {0: (0, len(text)), "tail": (len(text)+1, len(text)+2), 1: (len(text)+3, len(text)+6)}


@pytest.mark.parametrize("spec", ["f", "F", ".0f", ".0F", "e", "E", "g", "G"])
@pytest.mark.parametrize("text", ["nan", "NAN", "inf", "-inf", "+INF"])
def test_special_numbers(spec, text):
    value = parse.parse("{:" + spec + "}", text)[0]
    if "nan" in text.lower():
        assert math.isnan(value)
    else:
        assert math.isinf(value)
        assert (value < 0) == text.startswith("-")


@pytest.mark.parametrize("spec,text", [("b", "102"), ("o", "8"), ("x", "gg"), ("d", "1.2"), ("n", "12,34"), ("%", "10"), ("f", "1"), (".0f", "1.0"), ("e", "2e3")])
def test_invalid_numbers_do_not_match(spec, text):
    assert parse.parse("{:" + spec + "}", text) is None


@pytest.mark.parametrize("spec,good,bad", [("w", "héllo_42", "a-b"), ("W", "!?-", "abc"), ("s", " \t\n", "a"), ("S", "a-b", "a b"), ("D", "abc!", "a1"), ("l", "AbZ", "abc2")])
def test_character_classes(spec, good, bad):
    assert parse.parse("{:" + spec + "}", good)[0] == good
    assert parse.parse("{:" + spec + "}", bad) is None


@pytest.mark.parametrize("spec,value", [("08d", -42), ("+d", 42), ("6d", 42), ("0=8d", -42), ("*>8d", 42), ("*<8d", 42), ("*^8d", 42), (".>8d", 42), ("^8.2f", 12.5), (".0f", -20.0)])
def test_numeric_format_round_trip(spec, value):
    text = format(value, spec)
    assert parse.parse("{:" + spec + "}", text)[0] == value


@pytest.mark.parametrize("spec,text,expected", [("<", "abc   ", "abc"), (">", "   abc", "abc"), ("^", "  abc  ", "abc"), ("*^", "**abc**", "abc"), (".>", "...abc", "abc"), ("3", "abcd", "abcd"), (".3", "abc", "abc"), ("2.4", "abc", "abc")])
def test_string_padding_and_width(spec, text, expected):
    result = parse.parse("{:" + spec + "}", text)
    assert result[0] == expected
    start, end = result.spans[0]
    assert text[start:end] == expected


@pytest.mark.parametrize("spec,text", [("3", "ab"), (".3", "abcd"), ("2.4", "a"), ("2.4", "abcde"), ("2d", "123")])
def test_width_precision_limits(spec, text):
    assert parse.parse("{:" + spec + "}", text) is None


def test_names_collisions_nested_keys_and_repetitions():
    fmt = "{a.b}/{a_b}/{a-b}/{obj[x][y]:d}/{obj[z]}/{a.b}"
    result = parse.parse(fmt, "one/two/three/4/five/one")
    assert result.named == {"a.b": "one", "a_b": "two", "a-b": "three", "obj": {"x": {"y": 4}, "z": "five"}}
    assert result.spans["a.b"] == (0, 3)
    assert parse.parse("{n:d}:{n:d}", "12:12")["n"] == 12
    assert parse.parse("{n:d}:{n:d}", "12:012") is None
    assert parse.parse("{x}:{x}", "a:b") is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{n:d}:{n:f}")
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{:unknown}")


def test_compiled_parser_reuse_and_properties():
    parser = parse.compile("{}:{name:d}", case_sensitive=True)
    assert parser.format == "{}:{name:d}"
    fixed, named = parser.fixed_fields, parser.named_fields
    fixed.clear()
    named.clear()
    assert parser.fixed_fields == [0]
    assert parser.named_fields == ["name"]
    for text, expected in [("first:1", 1), ("second:2", 2)]:
        assert parser.parse(text)["name"] == expected
        assert parser.search("!" + text)["name"] == expected
    assert parse.parse("{2}/{0}", "a/b").fixed == ("a", "b")


@pytest.mark.parametrize("api", [parse.parse, parse.search, parse.findall])
def test_custom_conversion_is_deferred(api):
    calls = []
    @parse.with_pattern(r"[A-Z]+")
    def convert(text):
        calls.append(text)
        return text.lower()
    result = api("{x:token}", "ABC", extra_types={"token": convert}, evaluate_result=False)
    if api is parse.findall:
        result = next(result)
    assert isinstance(result, parse.Match)
    assert calls == []
    assert result.evaluate_result().named == {"x": "abc"}
    assert calls == ["ABC"]


def test_custom_capture_groups_and_default_pattern():
    @parse.with_pattern(r"(\d+)-(\d+)", regex_group_count=2)
    def pair(text):
        return tuple(map(int, text.split("-")))
    assert pair.pattern == r"(\d+)-(\d+)"
    assert pair.regex_group_count == 2
    result = parse.parse("{:pair}|{n:pair}|{:d}|{}", "1-2|3-4|5|end", {"pair": pair})
    assert result.fixed == ((1, 2), 5, "end")
    assert result.named == {"n": (3, 4)}
    assert result.spans == {0: (0, 3), "n": (4, 7), 1: (8, 9), 2: (10, 13)}
    assert parse.parse("{:reverse}", "abc", {"reverse": lambda s: s[::-1]})[0] == "cba"
    assert parse.parse("{:d}", "12", {"d": lambda s: "custom:" + s})[0] == "custom:12"


def test_conversion_errors_propagate_only_when_evaluated():
    def reject(text):
        raise ValueError("converter rejected input")
    pending = parse.parse("{:bad}", "anything", {"bad": reject}, evaluate_result=False)
    with pytest.raises(ValueError, match="converter rejected input"):
        pending.evaluate_result()
    with pytest.raises(ValueError, match="converter rejected input"):
        parse.parse("{:bad}", "anything", {"bad": reject})


@pytest.mark.parametrize("spec,text,expected", [
    ("ti", "2024-02-29", datetime(2024, 2, 29)),
    ("ti", "2024-02-29T13:14:15.1234567Z", datetime(2024, 2, 29, 13, 14, 15, 123456, timezone.utc)),
    ("ti", "2024-02-29 13:14-03:30", datetime(2024, 2, 29, 13, 14, tzinfo=timezone(timedelta(hours=-3, minutes=-30)))),
    ("tg", "29/February/2024 1:02:03 PM +0530", datetime(2024, 2, 29, 13, 2, 3, tzinfo=timezone(timedelta(hours=5, minutes=30)))),
    ("ta", "February-29-2024 12:02 AM", datetime(2024, 2, 29, 0, 2)),
    ("te", "Thu, 29 Feb 2024 13:14:15 +0000", datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone.utc)),
    ("th", "29/Feb/2024:13:14:15 -0330", datetime(2024, 2, 29, 13, 14, 15, tzinfo=timezone(timedelta(hours=-3, minutes=-30)))),
    ("tc", "Thu Feb 29 13:14:15 2024", datetime(2024, 2, 29, 13, 14, 15)),
    ("tt", "12:05 AM", time(0, 5)), ("tt", "12:05 PM", time(12, 5)),
    ("tt", "1:05:06.12 PM +530", time(13, 5, 6, 120000, timezone(timedelta(hours=5, minutes=30)))),
    ("%Y-%m-%d", "2024-02-29", date(2024, 2, 29)),
    ("%H:%M:%S.%f", "13:14:15.123", time(13, 14, 15, 123000)),
    ("%Y/%m/%d %H:%M", "2024/02/29 13:14", datetime(2024, 2, 29, 13, 14)),
])
@pytest.mark.parametrize("named", [False, True])
def test_dates_and_capture_group_offsets(spec, text, expected, named):
    field = "when" if named else ""
    result = parse.parse("{:d}|{" + field + ":" + spec + "}|{:d}", "7|" + text + "|8")
    assert result[field if named else 1] == expected
    assert type(result[field if named else 1]) is type(expected)
    assert result.fixed == ((7, 8) if named else (7, expected, 8))
    assert result.spans[field if named else 1] == (2, 2 + len(text))


def test_dates_with_current_year_and_invalid_calendar_date():
    year = datetime.today().year
    assert parse.parse("{:ts}", "Jan 2 03:04:05")[0] == datetime(year, 1, 2, 3, 4, 5)
    assert parse.parse("{:%m-%d}", "01-02")[0] == date(year, 1, 2)
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2023-02-29")
    pending = parse.parse("{:ti}", "2023-02-29", evaluate_result=False)
    assert isinstance(pending, parse.Match)


def test_fixed_timezone_contract():
    tz = parse.FixedTzOffset(-90, "west")
    assert tz.utcoffset(None) == timedelta(minutes=-90)
    assert tz.tzname(None) == "west"
    assert tz.dst(None) == timedelta(0)
    assert tz == parse.FixedTzOffset(-90, "west")
    assert tz != parse.FixedTzOffset(90, "west")
    assert tz != parse.FixedTzOffset(-90, "other")
    assert tz != object()
    assert parse.parse("{:ti}", "2024-01-02Z")[0].tzname() == "UTC"


@pytest.mark.parametrize("offset,minutes", [("+0530", 330), ("-03:30", -210), ("+0000", 0)])
def test_strptime_timezone_conversion(offset, minutes):
    result = parse.parse("{:%H:%M%z}", "13:14" + offset)
    assert result[0] == time(13, 14, tzinfo=timezone(timedelta(minutes=minutes)))
    assert result[0].utcoffset() == timedelta(minutes=minutes)
