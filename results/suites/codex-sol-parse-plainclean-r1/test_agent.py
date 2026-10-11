from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse


def test_literal_text_and_escaped_braces_are_matched_exactly():
    result = parse.parse("{{status}}: {}", "{status}: ready")

    assert result is not None
    assert result.fixed == ("ready",)
    assert parse.parse("{{status}}: {}", "status: ready") is None
    assert parse.parse("value={}", "prefix value=3") is None


def test_parse_is_case_insensitive_by_default_and_can_be_case_sensitive():
    assert parse.parse("Hello {name}", "hello Ada")["name"] == "Ada"
    assert parse.parse("Hello {name}", "hello Ada", case_sensitive=True) is None


def test_result_supports_fixed_named_slice_membership_and_repr():
    result = parse.parse("{} {name} {}", "first Ada last")

    assert result.fixed == ("first", "last")
    assert result[0] == "first"
    assert result[:] == ("first", "last")
    assert result["name"] == "Ada"
    assert "name" in result
    assert "missing" not in result
    assert repr(result) == "<Result ('first', 'last') {'name': 'Ada'}>"


def test_result_spans_refer_to_original_input_locations():
    result = parse.search("id={id:d}, value={} yy", "xx id=42, value=blue yy")

    assert result.named == {"id": 42}
    assert result.fixed == ("blue",)
    assert result.spans == {"id": (6, 8), 0: (16, 20)}


@pytest.mark.parametrize(
    ("format_spec", "text", "expected"),
    [
        ("{:,d}", "-1,234", -1234),
        ("{:d}", "+0xFF", 255),
        ("{:b}", "0b10101", 21),
        ("{:o}", "0o17", 15),
        ("{:x}", "0xCAFE", 0xCAFE),
        ("{:n}", "1.234.567", 1234567),
        ("{:%}", "12.5%", 0.125),
        ("{:f}", "-1.25", -1.25),
        ("{:F}", "2.50", Decimal("2.50")),
        ("{:e}", "1.5e-3", 0.0015),
        ("{:g}", "6.02E23", 6.02e23),
    ],
)
def test_numeric_formats_convert_values(format_spec, text, expected):
    result = parse.parse(format_spec, text)

    assert result is not None
    assert result[0] == expected


def test_decimal_width_limits_digits_and_alignment_discards_padding():
    assert parse.parse("{:3d}", "123")[0] == 123
    assert parse.parse("{:3d}", "1234") is None
    assert parse.parse("|{:>5d}|", "|   42|")[0] == 42
    assert parse.parse("|{:*^3}|", "|**cat**|")[0] == "cat"


def test_precision_constrains_untyped_fields():
    result = parse.parse("{word:.4} then {rest}", "tool then done")

    assert result.named == {"word": "tool", "rest": "done"}
    assert parse.parse("{word:.4}", "lengthy") is None


def test_named_fields_with_dots_dashes_and_nested_keys_do_not_collide():
    result = parse.parse(
        "{user.name} {user_name} {x-y} {record[address][city]}",
        "Ada Grace tag Dhaka",
    )

    assert result.named == {
        "user.name": "Ada",
        "user_name": "Grace",
        "x-y": "tag",
        "record": {"address": {"city": "Dhaka"}},
    }


def test_repeated_named_field_requires_same_value_and_type():
    result = parse.parse("{word} and {word}", "echo and echo")

    assert result.named == {"word": "echo"}
    assert parse.parse("{word} and {word}", "echo and other") is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{value:d} {value:f}")


def test_custom_type_uses_declared_pattern_and_converter():
    @parse.with_pattern(r"#[0-9a-fA-F]{6}")
    def color(value):
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))

    result = parse.parse("color={value:color}", "color=#12aBef", {"color": color})

    assert result.named == {"value": (0x12, 0xAB, 0xEF)}
    assert color.pattern == r"#[0-9a-fA-F]{6}"
    assert parse.parse("{value:color}", "not-a-color", {"color": color}) is None


def test_custom_pattern_capture_groups_do_not_corrupt_following_fields():
    @parse.with_pattern(r"(yes|no)", regex_group_count=1)
    def boolean(value):
        return value == "yes"

    result = parse.parse("{enabled:bool}:{count:d}", "yes:12", {"bool": boolean})

    assert result.named == {"enabled": True, "count": 12}
    assert result.spans == {"enabled": (0, 3), "count": (4, 6)}


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        ("ti", "2024-02-29T23:04:05.123456Z", datetime(2024, 2, 29, 23, 4, 5, 123456)),
        ("tg", "31/Dec/2020 11:59 PM", datetime(2020, 12, 31, 23, 59)),
        ("ta", "Dec/31/2020 12:01 AM", datetime(2020, 12, 31, 0, 1)),
        ("tt", "7:08:09 PM", time(19, 8, 9)),
    ],
)
def test_builtin_datetime_formats(spec, text, expected):
    value = parse.parse("{:" + spec + "}", text)[0]

    if spec == "ti":
        assert value.replace(tzinfo=None) == expected
        assert value.utcoffset() == timedelta(0)
    else:
        assert value == expected


def test_strptime_style_formats_return_appropriate_datetime_types():
    assert parse.parse("{:%Y-%m-%d}", "2023-07-09")[0] == date(2023, 7, 9)
    assert parse.parse("{:%H:%M:%S}", "14:03:02")[0] == time(14, 3, 2)
    assert parse.parse("{:%Y-%m-%d %H:%M}", "2023-07-09 14:03")[0] == datetime(
        2023, 7, 9, 14, 3
    )


def test_search_honors_position_and_end_position():
    text = "id=1 id=22 id=333"

    assert parse.search("id={:d}", text, pos=5)[0] == 22
    assert parse.search("id={:d}", text, pos=5, endpos=10)[0] == 22
    assert parse.search("id={:d}", text, pos=10, endpos=14) is None


def test_findall_is_lazy_iterable_and_honors_bounds():
    matches = parse.findall("<{word:l}>", "x <one> <two> <three> y", pos=8, endpos=23)

    assert iter(matches) is matches
    assert [result["word"] for result in matches] == ["two", "three"]
    with pytest.raises(StopIteration):
        next(matches)


def test_unevaluated_match_can_be_evaluated_later_for_all_entry_points():
    direct = parse.parse("{:d}", "42", evaluate_result=False)
    searched = parse.search("{:d}", "x 7 y", evaluate_result=False)
    found = list(parse.findall("{:d}", "1 2", evaluate_result=False))

    assert isinstance(direct, parse.Match)
    assert direct.match.group(0) == "42"
    assert direct.evaluate_result().fixed == (42,)
    assert searched.evaluate_result().fixed == (7,)
    assert [match.evaluate_result().fixed for match in found] == [(1,), (2,)]


def test_compiled_parser_exposes_metadata_without_mutable_aliases():
    parser = parse.compile("{name:l}-{:d}")

    named = parser.named_fields
    fixed = parser.fixed_fields
    named.append("corruption")
    fixed.append(99)

    assert parser.format == "{name:l}-{:d}"
    assert parser.named_fields == ["name"]
    assert parser.fixed_fields == [1]
    assert parser.parse("Ada-5").named == {"name": "Ada"}


@pytest.mark.parametrize("bad_spec", ["{:q}", "{:wat}"])
def test_unknown_format_types_are_rejected(bad_spec):
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile(bad_spec)
