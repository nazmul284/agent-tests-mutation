from datetime import date, datetime, time, timedelta
from decimal import Decimal
import math
import pickle

import pytest

import parse


def test_literal_text_is_regex_escaped_and_braces_can_be_literal():
    result = parse.parse("a.b[0]={{value}}", "a.b[0]={VALUE}")
    assert result is not None
    assert result.fixed == ()
    assert result.named == {}
    assert parse.parse("a.b", "axb") is None


def test_parse_requires_the_whole_string_and_is_case_insensitive_by_default():
    assert parse.parse("Hello {}!", "hello World!").fixed == ("World",)
    assert parse.parse("Hello {}!", "prefix hello World!") is None
    assert parse.parse("Hello", "hello", case_sensitive=True) is None
    assert parse.parse("Hello", "Hello", case_sensitive=True) is not None


def test_result_supports_index_slice_name_membership_repr_and_spans():
    result = parse.parse("{} {word} {:d}", "one two 003")
    assert result.fixed == ("one", 3)
    assert result.named == {"word": "two"}
    assert result[0] == "one"
    assert result[:2] == ("one", 3)
    assert result["word"] == "two"
    assert "word" in result
    assert result.spans == {0: (0, 3), "word": (4, 7), 1: (8, 11)}
    assert repr(result) == "<Result ('one', 3) {'word': 'two'}>"


def test_named_attributes_hyphens_and_nested_keys_do_not_collide():
    result = parse.parse(
        "{person.name}/{person_name}/{user[first][id]}/{field-name}",
        "Ada/Lovelace/42/value",
    )
    assert result.named == {
        "person.name": "Ada",
        "person_name": "Lovelace",
        "user": {"first": {"id": "42"}},
        "field-name": "value",
    }


def test_repeated_named_field_is_a_backreference():
    assert parse.parse("{x} then {x}", "same then same").named == {"x": "same"}
    assert parse.parse("{x} then {x}", "one then two") is None
    with pytest.raises(parse.RepeatedNameError):
        parse.compile("{x:d} {x:f}")


def test_parser_metadata_is_defensive_and_repr_is_bounded():
    parser = parse.compile("{name} {}")
    named = parser.named_fields
    fixed = parser.fixed_fields
    named.append("damage")
    fixed.append(99)
    assert parser.named_fields == ["name"]
    assert parser.fixed_fields == [1]
    assert parser.format == "{name} {}"
    assert repr(parser) == "<Parser '{name} {}'>"
    assert repr(parse.compile("x" * 30)) == "<Parser 'xxxxxxxxxxxxxxxxx...'>"


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        ("{:,d}", "-1,234", -1234),
        ("{:d}", "+0xff", 255),
        ("{:d}", "0b10101", 21),
        ("{:d}", "0o17", 15),
        ("{:n}", "1.234.567", 1234567),
        ("{:b}", "0B101", 5),
        ("{:o}", "0o17", 15),
        ("{:x}", "0xAf", 175),
        ("{:X}", "FF", 255),
        ("{:%}", "12.5%", 0.125),
    ],
)
def test_integer_and_percentage_conversions(spec, text, expected):
    assert parse.parse(spec, text)[0] == expected


def test_decimal_width_and_grouping_restrict_what_matches():
    assert parse.parse("{:3d}", "123")[0] == 123
    assert parse.parse("{:3d}", "1234") is None
    assert parse.parse("{:_d}", "1_234")[0] == 1234
    assert parse.parse("{:,d}", "1,234")[0] == 1234
    assert parse.parse("{:d}", "12z") is None


@pytest.mark.parametrize("value", ["1.25", "-0.5", "+inf", "nan"])
def test_float_conversion(value):
    actual = parse.parse("{:f}", value)[0]
    expected = float(value)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def test_decimal_and_general_and_exponential_formats():
    assert parse.parse("{:F}", "-12.50")[0] == Decimal("-12.50")
    assert parse.parse("{:e}", "-1.5e+3")[0] == -1500.0
    assert parse.parse("{:g}", "6.02E23")[0] == 6.02e23
    assert parse.parse("{:.0f}", "-20")[0] == -20.0
    assert parse.parse("{:f}", "20") is None


def test_alignment_fill_width_and_precision():
    assert parse.parse("|{:>5d}|", "|   12|")[0] == 12
    assert parse.parse("|{:_<5d}|", "|12___|")[0] == 12
    assert parse.parse("|{:*^7d}|", "|**123**|")[0] == 123
    assert parse.parse("{:.3}", "abc")[0] == "abc"
    assert parse.parse("{:.3}", "abcd") is None
    assert parse.parse("{:2.4}", "abc")[0] == "abc"


def test_letter_and_unknown_format_validation():
    assert parse.parse("{:l}", "Letters")[0] == "Letters"
    assert parse.parse("{:l}", "abc1") is None
    with pytest.raises(ValueError, match="not recognised"):
        parse.compile("{:unknown}")


def test_custom_type_pattern_decorator_conversion_and_group_accounting():
    @parse.with_pattern(r"(red|green|blue)", regex_group_count=1)
    def color(text):
        return text.upper()

    parser = parse.compile("{shade:Color}-{count:d}", {"Color": color})
    result = parser.parse("green-7")
    assert color.pattern == r"(red|green|blue)"
    assert color.regex_group_count == 1
    assert result.named == {"shade": "GREEN", "count": 7}
    assert result.spans == {"shade": (0, 5), "count": (6, 7)}


def test_custom_converter_without_pattern_and_converter_errors_propagate():
    assert parse.parse("{x:Upper}", "hello", {"Upper": str.upper})["x"] == "HELLO"

    def explode(value):
        raise RuntimeError(value)

    with pytest.raises(RuntimeError, match="boom"):
        parse.parse("{:Explode}", "boom", {"Explode": explode})


def test_search_honors_position_end_position_and_reports_absolute_spans():
    text = "skip id=12 then id=34 end"
    result = parse.search("id={id:d}", text, pos=5, endpos=16)
    assert result.named == {"id": 12}
    assert result.spans["id"] == (8, 10)
    assert parse.search("id={id:d}", text, pos=11, endpos=21).named == {"id": 34}
    assert parse.search("id={id:d}", text, pos=20) is None


def test_findall_is_lazy_iterable_and_honors_bounds():
    iterator = parse.findall("{:d}", "a1 b22 c333", pos=2, endpos=8)
    assert iter(iterator) is iterator
    assert [result[0] for result in iterator] == [22]
    with pytest.raises(StopIteration):
        next(iterator)


def test_deferred_match_evaluation_for_all_entry_points():
    direct = parse.parse("{n:d}", "42", evaluate_result=False)
    assert isinstance(direct, parse.Match)
    assert direct.evaluate_result().named == {"n": 42}

    searched = parse.search("{n:d}", "x 43 y", evaluate_result=False)
    assert searched.match.span() == (1, 4)
    assert searched.evaluate_result().spans["n"] == (1, 4)
    assert searched.evaluate_result()["n"] == 43

    matches = list(parse.findall("{n:d}", "1 2", evaluate_result=False))
    assert all(isinstance(item, parse.Match) for item in matches)
    assert [item.evaluate_result()["n"] for item in matches] == [1, 2]


def test_compiled_parser_can_be_pickled_with_builtin_converter():
    restored = pickle.loads(pickle.dumps(parse.compile("{:d}")))
    assert restored.parse("123")[0] == 123


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        ("ti", "2024-02-29T23:04:05.123456Z", datetime(2024, 2, 29, 23, 4, 5, 123456)),
        ("tg", "31/Dec/2023 11:59 PM", datetime(2023, 12, 31, 23, 59)),
        ("ta", "Dec/31/2023 12:01 AM", datetime(2023, 12, 31, 0, 1)),
        ("te", "Sun, 31 Dec 2023 23:59:58 +0000", datetime(2023, 12, 31, 23, 59, 58)),
        ("th", "31/Dec/2023:23:59:58 +0000", datetime(2023, 12, 31, 23, 59, 58)),
        ("tc", "Sun Dec 31 23:59:58 2023", datetime(2023, 12, 31, 23, 59, 58)),
    ],
)
def test_builtin_datetime_formats(spec, text, expected):
    actual = parse.parse("{:" + spec + "}", text)[0]
    assert actual.replace(tzinfo=None) == expected


def test_time_formats_timezone_and_current_year_date():
    parsed_time = parse.parse("{:tt}", "12:34:56.25 PM +05:30")[0]
    assert parsed_time.replace(tzinfo=None) == time(12, 34, 56, 250000)
    assert parsed_time.utcoffset() == timedelta(hours=5, minutes=30)

    short_date = parse.parse("{:ts}", "Jan 2 03:04:05")[0]
    assert short_date == datetime(datetime.today().year, 1, 2, 3, 4, 5)


def test_strptime_style_formats_choose_date_time_or_datetime():
    assert parse.parse("{:%Y-%m-%d}", "2020-02-29")[0] == date(2020, 2, 29)
    assert parse.parse("{:%H:%M:%S}", "23:04:05")[0] == time(23, 4, 5)
    assert parse.parse("{:%Y-%m-%d %H:%M}", "2020-02-29 23:04")[0] == datetime(
        2020, 2, 29, 23, 4
    )


def test_fixed_timezone_value_semantics():
    east = parse.FixedTzOffset(90, "+01:30")
    same = parse.FixedTzOffset(90, "+01:30")
    assert east == same
    assert east != parse.FixedTzOffset(60, "+01:00")
    assert east.utcoffset(None) == timedelta(minutes=90)
    assert east.dst(None) == timedelta(0)
    assert east.tzname(None) == "+01:30"
    assert repr(east) == "<FixedTzOffset +01:30 1:30:00>"


def test_invalid_calendar_values_raise_during_conversion():
    with pytest.raises(ValueError):
        parse.parse("{:ti}", "2023-02-29")
