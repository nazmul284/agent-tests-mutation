from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


def test_parse_requires_the_entire_input_and_exposes_fixed_and_named_fields():
    result = parse.parse("user={} score={score:d}", "user=alice score=42")

    assert result.fixed == ("alice",)
    assert result.named == {"score": 42}
    assert result[0] == "alice"
    assert result["score"] == 42
    assert "score" in result
    assert parse.parse("value={:d}", "prefix value=3") is None
    assert parse.parse("value={:d}", "value=3 suffix") is None


def test_literal_regex_metacharacters_and_escaped_braces_are_literal():
    result = parse.parse("[x].*? + {{value}} = {}", "[x].*? + {value} = yes")
    assert result.fixed == ("yes",)
    assert parse.parse("[x].*?", "anything") is None


def test_case_sensitivity_can_be_selected_explicitly():
    assert parse.parse("Hello {name}", "hello Ada", case_sensitive=False)["name"] == "Ada"
    assert parse.parse("Hello {name}", "hello Ada", case_sensitive=True) is None


def test_compile_properties_are_defensive_and_repr_is_useful():
    parser = parse.compile("{word:l}-{:d}")
    assert parser.format == "{word:l}-{:d}"
    assert parser.named_fields == ["word"]
    assert parser.fixed_fields == [1]
    parser.named_fields.append("damage")
    parser.fixed_fields.append(99)
    assert parser.named_fields == ["word"]
    assert parser.fixed_fields == [1]
    assert repr(parser) == "<Parser '{word:l}-{:d}'>"


@pytest.mark.parametrize(
    ("specification", "text", "expected"),
    [
        ("d", "-1234", -1234),
        ("d", "+0xFf", 255),
        ("d", "0b10101", 21),
        ("d", "0o17", 15),
        ("b", "0B101", 5),
        ("o", "0o71", 57),
        ("x", "0xDeAd", 0xDEAD),
        ("n", "1.234.567", 1234567),
        ("%", "12.5%", 0.125),
        ("f", "-12.50", -12.5),
        ("F", "+12.50", Decimal("12.50")),
        ("e", "-1.25e+3", -1250.0),
        ("g", "6.02E23", 6.02e23),
    ],
)
def test_numeric_conversions(specification, text, expected):
    result = parse.parse("{:" + specification + "}", text)
    assert result.fixed == (expected,)


@pytest.mark.parametrize("text", ["nan", "NAN", "inf", "-inf", "+INF"])
def test_general_float_accepts_special_values(text):
    value = parse.parse("{:g}", text)[0]
    if "nan" in text.lower():
        assert value != value
    else:
        assert value == float(text)


def test_numeric_width_grouping_and_alignment():
    assert parse.parse("{:06d}", "000123")[0] == 123
    assert parse.parse("{:>6d}", "   -12")[0] == -12
    assert parse.parse("{:*^7d}", "**123**")[0] == 123
    assert parse.parse("{:10,d}", " 1,234,567")[0] == 1234567
    assert parse.parse("{:3d}", "1234") is None


def test_string_width_precision_alignment_and_letters_type():
    assert parse.parse("{:>5}", "   hi")[0] == "   hi"
    assert parse.parse("{:.3}", "abc")[0] == "abc"
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{word:l}", "Letters")["word"] == "Letters"
    assert parse.parse("{word:l}", "abc123") is None


def test_repeated_names_must_repeat_the_same_value_and_type():
    result = parse.parse("{x:d}+{x:d}", "12+12")
    assert result.named == {"x": 12}
    assert parse.parse("{x:d}+{x:d}", "12+13") is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{x:d} {x:f}")


def test_similar_group_names_do_not_collide():
    result = parse.parse(
        "{a.b}/{a_b}/{a-b}/{a[c]}", "dot/underscore/hyphen/bracket"
    )
    assert result.named == {
        "a.b": "dot",
        "a_b": "underscore",
        "a-b": "hyphen",
        "a": {"c": "bracket"},
    }


def test_nested_bracket_names_expand_to_dictionaries():
    result = parse.parse(
        "{person[name]} is {person[address][city]}", "Ada is London"
    )
    assert result.named == {
        "person": {"name": "Ada", "address": {"city": "London"}}
    }
    assert result["person"]["address"]["city"] == "London"


def test_spans_report_original_input_locations():
    result = parse.search("ID {}:{name} END", "before ID abc:xyz END after")
    assert result.spans == {0: (10, 13), "name": (14, 17)}


def test_search_honors_position_and_end_position():
    text = "id=1; id=22; id=333"
    assert parse.search("id={:d}", text)[0] == 1
    assert parse.search("id={:d}", text, pos=6)[0] == 22
    assert parse.search("id={:d}", text, pos=6, endpos=11)[0] == 22
    assert parse.search("id={:d}", text, pos=12, endpos=15) is None


def test_findall_is_lazy_iterable_and_respects_bounds():
    results = parse.findall("<{:d}>", "x<1> y<20> z<300>", pos=5, endpos=18)
    assert iter(results) is results
    assert [item[0] for item in results] == [20, 300]
    with pytest.raises(StopIteration):
        next(results)


def test_evaluation_can_be_deferred_for_each_public_operation():
    direct = parse.parse("x={value:d}", "x=7", evaluate_result=False)
    found = parse.search("x={value:d}", "--x=8--", evaluate_result=False)
    many = list(parse.findall("x={value:d}", "x=1 x=2", evaluate_result=False))

    assert isinstance(direct, parse.Match)
    assert direct.evaluate_result()["value"] == 7
    assert found.evaluate_result()["value"] == 8
    assert [match.evaluate_result()["value"] for match in many] == [1, 2]


def test_custom_type_converter_and_decorator_metadata():
    @parse.with_pattern(r"([A-Z]{2})-(\d+)", regex_group_count=2)
    def ticket(text):
        prefix, number = text.split("-")
        return prefix, int(number)

    parser = parse.compile("{ticket:Ticket}:{count:d}", {"Ticket": ticket})
    result = parser.parse("AB-17:4")
    assert result.named == {"ticket": ("AB", 17), "count": 4}
    assert ticket.pattern == r"([A-Z]{2})-(\d+)"
    assert ticket.regex_group_count == 2


def test_invalid_format_type_is_rejected():
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{value:not_a_type}")


@pytest.mark.parametrize(
    ("specification", "text", "expected"),
    [
        ("ti", "2024-02-29T23:05:06.123456Z", datetime(2024, 2, 29, 23, 5, 6, 123456, parse.FixedTzOffset(0, "UTC"))),
        ("tg", "31/Dec/2020 11:59 PM +0530", datetime(2020, 12, 31, 23, 59, tzinfo=parse.FixedTzOffset(330, "+0530"))),
        ("ta", "Dec/31/2020 12:01 AM -04:00", datetime(2020, 12, 31, 0, 1, tzinfo=parse.FixedTzOffset(-240, "-04:00"))),
        ("te", "Thu, 31 Dec 2020 23:59:58 +0000", datetime(2020, 12, 31, 23, 59, 58, tzinfo=parse.FixedTzOffset(0, "+0000"))),
        ("th", "31/Dec/2020:23:59:58 +0000", datetime(2020, 12, 31, 23, 59, 58, tzinfo=parse.FixedTzOffset(0, "+0000"))),
        ("tc", "Thu Dec 31 23:59:58 2020", datetime(2020, 12, 31, 23, 59, 58)),
        ("tt", "12:30:15.5 PM +0130", time(12, 30, 15, 500000, tzinfo=parse.FixedTzOffset(90, "+0130"))),
    ],
)
def test_built_in_datetime_formats(specification, text, expected):
    assert parse.parse("{:" + specification + "}", text)[0] == expected


def test_strptime_style_types_return_datetime_date_or_time():
    assert parse.parse("{:%Y-%m-%d %H:%M:%S}", "2023-07-09 04:05:06")[0] == datetime(2023, 7, 9, 4, 5, 6)
    assert parse.parse("{:%Y/%m/%d}", "2023/07/09")[0] == date(2023, 7, 9)
    parsed_time = parse.parse("{:%H:%M:%S%z}", "04:05:06+0230")[0]
    assert parsed_time.hour == 4 and parsed_time.minute == 5
    assert parsed_time.utcoffset() == timedelta(hours=2, minutes=30)


def test_fixed_timezone_protocol_and_equality():
    tz = parse.FixedTzOffset(90, "custom")
    assert tz.utcoffset(None) == timedelta(minutes=90)
    assert tz.dst(None) == timedelta(0)
    assert tz.tzname(None) == "custom"
    assert tz == parse.FixedTzOffset(90, "custom")
    assert tz != parse.FixedTzOffset(60, "custom")
    assert "custom" in repr(tz)
