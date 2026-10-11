from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest

import parse


def test_plain_text_is_escaped_and_braces_are_literal():
    assert parse.parse("cost ($5)? {{ok}}", "cost ($5)? {ok}").fixed == ()
    assert parse.parse("cost ($5)? {{ok}}", "cost X$5Y? {ok}") is None


def test_parse_is_anchored_but_search_is_not():
    assert parse.parse("id={:d}", "before id=12 after") is None
    result = parse.search("id={:d}", "before id=12 after")
    assert result.fixed == (12,)
    assert result.spans == {0: (10, 12)}


def test_default_matching_is_case_insensitive_and_dot_matches_newline():
    assert parse.parse("Hello {value}!", "hELLo one\ntwo!")["value"] == "one\ntwo"
    assert parse.parse("Hello", "hello", case_sensitive=True) is None
    assert parse.parse("Hello", "Hello", case_sensitive=True) is not None


def test_result_protocol_and_repr():
    result = parse.parse("{} {word}", "first second")
    assert result.fixed == ("first",)
    assert result.named == {"word": "second"}
    assert result[0] == "first"
    assert result[:] == ("first",)
    assert result["word"] == "second"
    assert "word" in result and "missing" not in result
    assert repr(result) == "<Result ('first',) {'word': 'second'}>"


def test_spans_for_named_and_positional_fields():
    result = parse.parse("A{name}B{:d}C{}", "AxyB17Cz")
    assert result.spans == {"name": (1, 3), 0: (4, 6), 1: (7, 8)}


def test_parser_metadata_is_defensively_copied_and_repr_is_shortened():
    parser = parse.compile("{first}-{}-{second:d}")
    named = parser.named_fields
    fixed = parser.fixed_fields
    named.append("damage")
    fixed.append(99)
    assert parser.named_fields == ["first", "second"]
    assert parser.fixed_fields == [1]
    assert parser.format == "{first}-{}-{second:d}"
    assert repr(parse.compile("x" * 21)) == "<Parser 'xxxxxxxxxxxxxxxxx...'>"
    assert repr(parse.compile("short")) == "<Parser 'short'>"


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        (",d", "-1,234", -1234),
        ("d", "+0xFf", 255),
        ("d", "-0B101", -5),
        ("d", "0o17", 15),
        ("b", "0B1011", 11),
        ("o", "0o71", 57),
        ("x", "0Xaf", 175),
        ("X", "FF", 255),
        ("n", "1.234.567", 1234567),
        ("%", "12.5%", 0.125),
        ("f", "-12.50", -12.5),
        ("F", "+12.50", Decimal("12.50")),
        ("e", "-1.25e+3", -1250.0),
        ("g", "2E-3", 0.002),
    ],
)
def test_numeric_conversions(spec, text, expected):
    assert parse.parse("{:" + spec + "}", text)[0] == expected


@pytest.mark.parametrize(
    ("fmt", "accepted", "rejected"),
    [
        ("{:b}", "101", "102"),
        ("{:o}", "707", "708"),
        ("{:x}", "Cafe", "Caz"),
        ("{:l}", "Letters", "abc1"),
        ("{:d}", "123", "12a"),
        ("{:.2f}", "1.25", "1"),
        ("{:.0f}", "20", "20.0"),
        ("{:%}", "10%", "10"),
    ],
)
def test_type_patterns_reject_invalid_text(fmt, accepted, rejected):
    assert parse.parse(fmt, accepted) is not None
    assert parse.parse(fmt, rejected) is None


@pytest.mark.parametrize("value", ["nan", "NAN", "inf", "INF", "-inf", "+INF"])
def test_float_special_values(value):
    result = parse.parse("{:f}", value)
    assert result is not None
    assert str(result[0]).lower().lstrip("+-") in {"nan", "inf"}


def test_width_grouping_precision_alignment_and_fill():
    assert parse.parse("{:4d}", "  12")[0] == 12
    assert parse.parse("{:4d}", "12345") is None
    assert parse.parse("{:_d}", "1_234_567")[0] == 1234567
    assert parse.parse("{:,d}", "1,234,567")[0] == 1234567
    assert parse.parse("{:<5l}", "xy   ")[0] == "xy"
    assert parse.parse("{:*>5l}", "***xy")[0] == "xy"
    assert parse.parse("{:.^5l}", ".xy..")[0] == "xy"
    assert parse.parse("{:0=5d}", "00012")[0] == 12
    assert parse.parse("{:*^7d}", "**123**")[0] == 123
    assert parse.parse("{:3.5}", "abcd")[0] == "abcd"
    assert parse.parse("{:3.5}", "ab") is None
    assert parse.parse("{:.2}", "abc") is None


def test_invalid_format_type_raises():
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{value:not_a_type}")


def test_repeated_named_field_is_a_backreference():
    assert parse.parse("{x:d}-{x:d}", "12-12").named == {"x": 12}
    assert parse.parse("{x:d}-{x:d}", "12-13") is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{x:d}-{x:f}")


def test_similar_field_names_do_not_collide():
    result = parse.parse("{a.b}|{a_b}|{a-b}", "dot|under|dash")
    assert result.named == {"a.b": "dot", "a_b": "under", "a-b": "dash"}


def test_bracket_names_expand_to_nested_dicts():
    result = parse.parse(
        "{person[name]}:{person[address][city]}:{person[address][zip]:d}",
        "Ada:Dhaka:1205",
    )
    assert result.named == {
        "person": {"name": "Ada", "address": {"city": "Dhaka", "zip": 1205}}
    }
    assert result.spans["person[name]"] == (0, 3)


def test_custom_type_decorator_conversion_and_group_accounting():
    @parse.with_pattern(r"([A-Z]+)-(\d+)", regex_group_count=2)
    def token(text):
        letters, number = text.split("-")
        return letters, int(number)

    result = parse.parse("{item:T}/{}", "ABC-12/tail", {"T": token})
    assert result.named == {"item": ("ABC", 12)}
    assert result.fixed == ("tail",)
    assert token.pattern == r"([A-Z]+)-(\d+)"
    assert token.regex_group_count == 2


def test_custom_type_without_pattern_and_none_group_count():
    def upper(text):
        return text.upper()

    upper.regex_group_count = None
    result = parse.parse("{x:U}!", "mixed!", {"U": upper})
    assert result["x"] == "MIXED"


def test_convert_first_used_by_custom_types_is_pickleable():
    converter = parse.convert_first(int)
    restored = pickle.loads(pickle.dumps(converter))
    assert restored("42", None) == 42


def test_lazy_match_can_be_evaluated_later():
    parser = parse.compile("value={value:d}")
    match = parser.parse("value=42", evaluate_result=False)
    assert isinstance(match, parse.Match)
    assert match.parser is parser
    assert match.match.group(0) == "value=42"
    assert match.evaluate_result().named == {"value": 42}


def test_search_bounds_and_lazy_result():
    parser = parse.compile("{:d}")
    assert parser.search("a11 b22", pos=4).fixed == (22,)
    assert parser.search("a11 b22", endpos=3).fixed == (11,)
    assert parser.search("a11 b22", pos=3, endpos=4) is None
    lazy = parse.search("{:d}", "x7", evaluate_result=False)
    assert isinstance(lazy, parse.Match)
    assert lazy.evaluate_result().fixed == (7,)


def test_findall_iteration_bounds_and_exhaustion():
    iterator = parse.findall("{:d}", "1, 22, 333", pos=2, endpos=7)
    assert iter(iterator) is iterator
    assert next(iterator).fixed == (22,)
    with pytest.raises(StopIteration):
        next(iterator)


def test_findall_returns_all_nonoverlapping_matches_and_lazy_matches():
    assert [r["n"] for r in parse.findall("x{n:d}", "x1 x20 x300")] == [1, 20, 300]
    lazy = list(parse.findall("x{:d}", "x1 x2", evaluate_result=False))
    assert all(isinstance(item, parse.Match) for item in lazy)
    assert [item.evaluate_result()[0] for item in lazy] == [1, 2]


@pytest.mark.parametrize(
    ("kind", "text", "expected"),
    [
        ("ti", "2024-02-29T23:05:07.123456Z", datetime(2024, 2, 29, 23, 5, 7, 123456, tzinfo=parse.FixedTzOffset(0, "UTC"))),
        ("tg", "31/December/2020 11:15 PM", datetime(2020, 12, 31, 23, 15)),
        ("ta", "December/31/2020 12:15 AM", datetime(2020, 12, 31, 0, 15)),
        ("te", "Thu, 31 Dec 2020 23:59:58 +0530", datetime(2020, 12, 31, 23, 59, 58, tzinfo=parse.FixedTzOffset(330, "+0530"))),
        ("th", "31/Dec/2020:23:59:58 -04:30", datetime(2020, 12, 31, 23, 59, 58, tzinfo=parse.FixedTzOffset(-270, "-04:30"))),
        ("tc", "Thu Dec 31 23:59:58 2020", datetime(2020, 12, 31, 23, 59, 58)),
        ("tt", "1:02:03.4 PM +0130", time(13, 2, 3, 400000, tzinfo=parse.FixedTzOffset(90, "+0130"))),
    ],
)
def test_built_in_datetime_formats(kind, text, expected):
    assert parse.parse("{:" + kind + "}", text)[0] == expected


def test_syslog_date_uses_current_year():
    value = parse.parse("{:ts}", "Jan 2 03:04:05")[0]
    assert value == datetime(datetime.today().year, 1, 2, 3, 4, 5)


def test_strftime_style_date_time_and_datetime_results():
    assert parse.parse("{:%Y-%m-%d}", "2024-02-29")[0] == date(2024, 2, 29)
    assert parse.parse("{:%H:%M:%S.%f}", "03:04:05.12")[0] == time(3, 4, 5, 120000)
    assert parse.parse("{:%Y/%m/%d %H:%M}", "2020/06/07 08:09")[0] == datetime(2020, 6, 7, 8, 9)
    current_year = datetime.today().year
    assert parse.parse("{:%b %d}", "Feb 03")[0] == date(current_year, 2, 3)


def test_fixed_timezone_protocol_equality_and_repr():
    tz = parse.FixedTzOffset(90, "X")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "X"
    assert tz == parse.FixedTzOffset(90, "X")
    assert tz != parse.FixedTzOffset(60, "X")
    assert tz != parse.FixedTzOffset(90, "Y")
    assert tz != object()
    assert repr(tz) == "<FixedTzOffset X 1:30:00>"


def test_empty_and_no_match_results():
    assert parse.parse("", "").fixed == ()
    assert parse.parse("x", "") is None
    assert parse.search("needle", "haystack") is None
    assert list(parse.findall("x{:d}", "none")) == []
