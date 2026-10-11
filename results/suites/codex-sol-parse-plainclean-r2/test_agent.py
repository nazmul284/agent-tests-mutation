"""Behavioral tests for the public :mod:`parse` API.

These tests intentionally exercise combinations where the generated regular
expression has internal capture groups.  Group accounting is one of the most
error-prone parts of a format-string parser.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


def test_literal_text_is_escaped_and_full_string_must_match():
    result = parse.parse("file.name+[{{}}] = {value}", "file.name+[{}] = yes")

    assert result.fixed == ()
    assert result.named == {"value": "yes"}
    assert parse.parse("value={:d}", "prefix value=3") is None
    assert parse.parse("value={:d}", "value=3 suffix") is None


def test_escaped_braces_are_literals():
    result = parse.parse("{{{name}}}", "{Ada}")
    assert result.named == {"name": "Ada"}


def test_fixed_and_named_fields_support_lookup_slices_membership_and_repr():
    result = parse.parse("{} {verb} {}", "cats chase mice")

    assert result.fixed == ("cats", "mice")
    assert result.named == {"verb": "chase"}
    assert result[0] == "cats"
    assert result[:] == ("cats", "mice")
    assert result["verb"] == "chase"
    assert "verb" in result
    assert "cats" not in result
    assert repr(result) == "<Result ('cats', 'mice') {'verb': 'chase'}>"


def test_matching_is_case_insensitive_by_default_but_can_be_strict():
    assert parse.parse("Hello {who}", "hELLo World")["who"] == "World"
    assert parse.parse("Hello {who}", "hello World", case_sensitive=True) is None


def test_parser_metadata_is_copied_and_repr_is_bounded():
    parser = parse.compile("{first} / {} / {last:d}")

    assert parser.format == "{first} / {} / {last:d}"
    assert parser.named_fields == ["first", "last"]
    assert parser.fixed_fields == [1]
    fields = parser.named_fields
    fields.append("corruption")
    assert parser.named_fields == ["first", "last"]
    assert repr(parse.compile("x" * 30)) == "<Parser 'xxxxxxxxxxxxxxxxx...'>"


def test_search_honors_pos_endpos_and_reports_absolute_spans():
    text = "skip id=12 then id=345 stop"
    result = parse.search("id={number:d}", text, pos=8, endpos=24)

    assert result.named == {"number": 345}
    assert result.spans == {"number": (19, 22)}
    assert parse.search("id={number:d}", text, pos=22) is None
    assert parse.search("id={number:d}", text, endpos=8) is None


def test_findall_is_lazy_ordered_and_honors_bounds():
    matches = parse.findall("<{:d}>", "x<1> y<22> z<333>", pos=4, endpos=14)

    assert iter(matches) is matches
    assert [result[0] for result in matches] == [22]
    assert list(matches) == []


def test_deferred_evaluation_exposes_match_and_converts_only_on_request():
    match = parse.parse("x={:d}", "x=42", evaluate_result=False)

    assert isinstance(match, parse.Match)
    assert match.match.group(0) == "x=42"
    assert match.evaluate_result().fixed == (42,)

    found = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert [item.evaluate_result()[0] for item in found] == [1, 2]


@pytest.mark.parametrize(
    ("specification", "text", "expected"),
    [
        ("_d", "-1_234", -1234),
        ("d", "+0xff", 255),
        ("d", "0B1011", 11),
        ("b", "-0b101", -5),
        ("o", "+0O17", 15),
        ("x", "0xCAFE", 0xCAFE),
        ("n", "1,234,567", 1234567),
        ("%", "12.5%", 0.125),
        ("f", "-12.50", -12.5),
        ("e", "1.25e-3", 0.00125),
        ("g", "2E+3", 2000.0),
    ],
)
def test_numeric_conversions(specification, text, expected):
    result = parse.parse("{value:" + specification + "}", text)
    assert result["value"] == expected


def test_decimal_conversion_preserves_decimal_semantics():
    result = parse.parse("{amount:F}", "0.10")
    assert result["amount"] == Decimal("0.10")
    assert isinstance(result["amount"], Decimal)


@pytest.mark.parametrize("text", ["nan", "NAN", "+inf", "-INF"])
def test_special_floating_point_values(text):
    result = parse.parse("{:g}", text)
    if "nan" in text.lower():
        assert result[0] != result[0]
    else:
        assert result[0] == float(text.lower())


def test_numeric_width_grouping_precision_and_alignment():
    assert parse.parse("{:05d}", "00042")[0] == 42
    assert parse.parse("{:>5d}", "   42")[0] == 42
    assert parse.parse("{:_^7d}", "__123__")[0] == 123
    assert parse.parse("{:8,d}", "  12,345")[0] == 12345
    assert parse.parse("{:.0f}", "20")[0] == 20.0
    assert parse.parse("{:.2}", "ab")[0] == "ab"
    assert parse.parse("{:.2}", "abc") is None


def test_letter_type_rejects_digits_and_punctuation():
    assert parse.parse("{word:l}", "Alphabet")["word"] == "Alphabet"
    assert parse.parse("{word:l}", "alpha2") is None


def test_repeated_named_field_must_have_same_value():
    result = parse.parse("{word} then {word}", "echo then echo")
    assert result.named == {"word": "echo"}
    assert parse.parse("{word} then {word}", "echo then other") is None


def test_repeated_named_field_requires_identical_format_specification():
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{value:d} {value:f}")


def test_colliding_sanitized_names_remain_distinct():
    result = parse.parse("{a.b} {a_b} {a-b}", "one two three")
    assert result.named == {"a.b": "one", "a_b": "two", "a-b": "three"}


def test_bracket_names_expand_into_nested_dictionaries():
    result = parse.parse(
        "{person[name]} is {person[address][city]}", "Ada is London"
    )
    assert result.named == {
        "person": {"name": "Ada", "address": {"city": "London"}}
    }
    assert result.spans == {
        "person[name]": (0, 3),
        "person[address][city]": (7, 13),
    }


def test_custom_type_converter_and_pattern_decorator():
    @parse.with_pattern(r"#[0-9a-f]{6}")
    def color(text):
        return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))

    result = parse.parse("color={value:Color}", "color=#12aBf0", {"Color": color})
    assert result["value"] == (0x12, 0xAB, 0xF0)
    assert color.pattern == r"#[0-9a-f]{6}"
    assert color.regex_group_count is None
    assert parse.parse("{value:Color}", "red", {"Color": color}) is None


def test_custom_converter_with_capture_groups_does_not_shift_later_fields():
    @parse.with_pattern(r"([A-Z]+)-(\d+)", regex_group_count=2)
    def ticket(text):
        prefix, number = text.split("-")
        return prefix, int(number)

    result = parse.parse(
        "{ticket:Ticket}/{count:d}/{}",
        "BUG-17/3/done",
        extra_types={"Ticket": ticket},
    )
    assert result.named == {"ticket": ("BUG", 17), "count": 3}
    assert result.fixed == ("done",)
    assert result.spans["ticket"] == (0, 6)
    assert result.spans["count"] == (7, 8)


def test_invalid_format_type_is_rejected_when_parser_is_constructed():
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{value:q}")


@pytest.mark.parametrize(
    ("specification", "text", "expected"),
    [
        ("%Y-%m-%d", "2024-02-29", date(2024, 2, 29)),
        ("%H:%M:%S.%f", "23:07:08.12", time(23, 7, 8, 120000)),
        (
            "%Y/%m/%d %H:%M:%S%z",
            "2023/04/05 06:07:08+0230",
            datetime(2023, 4, 5, 6, 7, 8, tzinfo=parse.FixedTzOffset(150, "+0230")),
        ),
    ],
)
def test_strptime_style_datetime_formats(specification, text, expected):
    assert parse.parse("{when:" + specification + "}", text)["when"] == expected


def test_iso_datetime_supports_fractional_seconds_and_utc_z():
    result = parse.parse("{when:ti}", "2020-01-02T03:04:05.1234567Z")
    assert result["when"] == datetime(
        2020,
        1,
        2,
        3,
        4,
        5,
        123456,
        tzinfo=parse.FixedTzOffset(0, "UTC"),
    )


def test_global_and_us_dates_handle_month_names_ampm_and_offsets():
    global_date = parse.parse("{when:tg}", "31/December/2021 11:59 PM +05:30")
    us_date = parse.parse("{when:ta}", "July/4/2020 12:01 AM -0400")

    assert global_date["when"] == datetime(
        2021, 12, 31, 23, 59, tzinfo=parse.FixedTzOffset(330, "+05:30")
    )
    assert us_date["when"] == datetime(
        2020, 7, 4, 0, 1, tzinfo=parse.FixedTzOffset(-240, "-0400")
    )


def test_time_format_and_fixed_timezone_protocol():
    parsed = parse.parse("at {clock:tt}", "at 12:30:15.25 AM +0130")["clock"]
    assert parsed == time(
        0, 30, 15, 250000, tzinfo=parse.FixedTzOffset(90, "+0130")
    )
    assert parsed.utcoffset() == timedelta(minutes=90)
    assert parsed.tzname() == "+0130"
    assert parsed.dst() == timedelta(0)
    assert repr(parsed.tzinfo) == "<FixedTzOffset +0130 1:30:00>"


def test_invalid_calendar_values_raise_during_conversion():
    with pytest.raises(ValueError):
        parse.parse("{when:%Y-%m-%d}", "2023-02-29")
