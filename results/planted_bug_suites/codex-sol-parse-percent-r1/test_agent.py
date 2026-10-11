from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


def test_literal_text_is_escaped_and_entire_input_must_match():
    result = parse.parse("file.{ext}", "FILE.py")

    assert result.named == {"ext": "py"}
    assert parse.parse("file.{ext}", "prefix-file.py") is None
    assert parse.parse("a+b={}", "a+b=3").fixed == ("3",)


def test_escaped_braces_are_literals():
    result = parse.parse("{{{value:d}}}", "{42}")

    assert result["value"] == 42
    assert result.spans["value"] == (1, 3)


def test_fixed_and_named_fields_support_indexing_membership_and_slices():
    result = parse.parse("{} {name} {:d}", "hello Ada 37")

    assert result.fixed == ("hello", 37)
    assert result.named == {"name": "Ada"}
    assert result[0] == "hello"
    assert result[:] == ("hello", 37)
    assert result["name"] == "Ada"
    assert "name" in result
    assert result.spans == {0: (0, 5), 1: (10, 12), "name": (6, 9)}


@pytest.mark.parametrize(
    ("specifier", "text", "expected"),
    [
        (",d", "-1,234", -1234),
        ("d", "+0x1f", 31),
        ("b", "0b1011", 11),
        ("o", "0o17", 15),
        ("x", "0xAf", 175),
        ("n", "1.234.567", 1234567),
        ("f", "-12.50", -12.5),
        ("F", "1.25", Decimal("1.25")),
        ("e", "-1.5e+2", -150.0),
        ("g", "6.02E23", 6.02e23),
    ],
)
def test_numeric_conversions(specifier, text, expected):
    result = parse.parse("{value:" + specifier + "}", text)

    assert result is not None
    assert result["value"] == expected


def test_float_special_values_and_zero_precision_are_supported():
    assert parse.parse("{:f}", "-inf")[0] == float("-inf")
    assert parse.parse("{:.0f}", "20")[0] == 20.0


@pytest.mark.xfail(
    strict=True,
    reason="known library defect: percentage() divides by 10 instead of 100",
)
def test_percentage_is_converted_to_fraction():
    assert parse.parse("{rate:%}", "12.5%")["rate"] == pytest.approx(0.125)


def test_width_alignment_and_precision_constraints():
    assert parse.parse("[{value:>5d}]", "[   42]")["value"] == 42
    assert parse.parse("[{word:*^7l}]", "[**cat**]")["word"] == "cat"
    assert parse.parse("{word:.3}", "abc")["word"] == "abc"
    assert parse.parse("{word:.3}", "abcd") is None


def test_case_sensitivity_is_configurable():
    assert parse.parse("Hello {name}", "hello WORLD")["name"] == "WORLD"
    assert parse.parse("Hello {name}", "hello WORLD", case_sensitive=True) is None


def test_repeated_named_field_must_have_same_value_and_type():
    assert parse.parse("{word} then {word}", "same then same")["word"] == "same"
    assert parse.parse("{word} then {word}", "one then two") is None

    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{value:d} {value:f}")


def test_nested_names_and_colliding_group_names_remain_distinct():
    result = parse.parse(
        "{user[name]}:{user[address][city]} {a.b} {a_b}",
        "Ada:Dhaka dot underscore",
    )

    assert result.named == {
        "user": {"name": "Ada", "address": {"city": "Dhaka"}},
        "a.b": "dot",
        "a_b": "underscore",
    }


def test_custom_type_pattern_and_internal_captures_do_not_shift_fields():
    @parse.with_pattern(r"([A-Z]{2})-(\d{2})", regex_group_count=2)
    def code(value):
        return value.lower()

    result = parse.parse(
        "{item:code}:{count:d}", "AB-12:7", extra_types={"code": code}
    )

    assert result.named == {"item": "ab-12", "count": 7}
    assert code.pattern == r"([A-Z]{2})-(\d{2})"
    assert code.regex_group_count == 2


def test_search_honors_bounds_and_reports_absolute_spans():
    text = "skip id=12; id=345; tail"

    result = parse.search("id={value:d}", text, pos=8, endpos=20)

    assert result["value"] == 345
    assert result.spans["value"] == (15, 18)
    assert parse.search("id={value:d}", text, pos=19) is None


def test_findall_is_lazy_iterable_and_honors_bounds():
    matches = parse.findall("{word:l}", "12 one, two 34 three", pos=3, endpos=11)

    assert iter(matches) is matches
    assert [result["word"] for result in matches] == ["one", "two"]


def test_evaluation_can_be_deferred_for_parse_search_and_findall():
    direct = parse.parse("{n:d}", "42", evaluate_result=False)
    found = parse.search("{n:d}", "x42y", evaluate_result=False)
    many = list(parse.findall("{n:d}", "1 2", evaluate_result=False))

    assert isinstance(direct, parse.Match)
    assert direct.evaluate_result()["n"] == 42
    assert found.match.span() == (1, 3)
    assert found.evaluate_result()["n"] == 42
    assert [match.evaluate_result()["n"] for match in many] == [1, 2]


@pytest.mark.parametrize(
    ("specifier", "text", "expected"),
    [
        ("%Y-%m-%d", "2024-02-29", date(2024, 2, 29)),
        ("%H:%M:%S", "23:07:05", time(23, 7, 5)),
        (
            "%Y-%m-%dT%H:%M:%S",
            "2023-08-09T14:03:02",
            datetime(2023, 8, 9, 14, 3, 2),
        ),
    ],
)
def test_strptime_style_datetime_formats(specifier, text, expected):
    assert parse.parse("{when:" + specifier + "}", text)["when"] == expected


def test_builtin_iso_datetime_handles_microseconds_and_timezone():
    value = parse.parse("{when:ti}", "2020-01-02T03:04:05.123456+05:30")["when"]

    assert value.replace(tzinfo=None) == datetime(2020, 1, 2, 3, 4, 5, 123456)
    assert value.utcoffset() == timedelta(hours=5, minutes=30)


def test_parser_metadata_is_defensive_and_repr_is_readable():
    parser = parse.compile("{name}={:d}")
    names = parser.named_fields
    fixed = parser.fixed_fields
    names.append("corruption")
    fixed.append(99)

    assert parser.format == "{name}={:d}"
    assert parser.named_fields == ["name"]
    assert parser.fixed_fields == [1]
    assert repr(parser) == "<Parser '{name}={:d}'>"


def test_invalid_format_type_is_rejected():
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{value:q}")
