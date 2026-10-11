import math
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import parse
from parse import FixedTzOffset, Match, Parser, RepeatedNameError, Result


def value(format_string, text, **kwargs):
    result = parse.parse(format_string, text, **kwargs)
    assert result is not None
    return result[0]


def test_basic_parse_result_protocol_and_spans():
    result = parse.parse("Hello {name}, you have {:d} messages", "hello Ada, you have 17 messages")
    assert isinstance(result, Result)
    assert result.fixed == (17,)
    assert result.named == {"name": "Ada"}
    assert result[0] == 17
    assert result[:] == (17,)
    assert result["name"] == "Ada"
    assert "name" in result
    assert "missing" not in result
    assert result.spans == {"name": (6, 9), 0: (20, 22)}
    assert repr(result) == "<Result (17,) {'name': 'Ada'}>"


def test_parse_is_anchored_and_literal_regex_characters_are_escaped():
    fmt = "[x].+?*$^|()!\\={value}"
    assert parse.parse(fmt, "[x].+?*$^|()!\\=ok").named == {"value": "ok"}
    assert parse.parse("x{}y", "prefix xAy suffix") is None
    assert parse.parse("x{}y", "xAy") is not None


def test_literal_braces_case_sensitivity_and_dotall():
    assert parse.parse("{{{word}}}", "{YES}").named == {"word": "YES"}
    assert parse.parse("abc", "ABC") is not None
    assert parse.parse("abc", "ABC", case_sensitive=True) is None
    assert parse.parse("before {} after", "before one\ntwo after").fixed == ("one\ntwo",)


def test_parser_metadata_is_copied_and_repr_is_truncated():
    parser = Parser("{first}-{}-{third:d}")
    assert parser.format == "{first}-{}-{third:d}"
    assert parser.named_fields == ["first", "third"]
    assert parser.fixed_fields == [1]
    parser.named_fields.append("corrupt")
    parser.fixed_fields.append(99)
    assert parser.named_fields == ["first", "third"]
    assert parser.fixed_fields == [1]
    assert repr(Parser("short")) == "<Parser 'short'>"
    assert repr(Parser("abcdefghijklmnopqrstuvwxyz")) == "<Parser 'abcdefghijklmnopq...'>"


def test_evaluate_result_can_be_deferred_for_parse_and_search():
    deferred = parse.parse("{:d}", "42", evaluate_result=False)
    assert isinstance(deferred, Match)
    assert deferred.evaluate_result().fixed == (42,)
    found = parse.search("value={x:d}", "--value=7--", evaluate_result=False)
    assert isinstance(found, Match)
    assert found.match.span() == (2, 9)
    assert found.evaluate_result().named == {"x": 7}


def test_search_obeys_bounds_and_reports_absolute_spans():
    text = "aa id=12 bb id=34 cc"
    assert parse.search("id={n:d}", text).named == {"n": 12}
    assert parse.search("id={n:d}", text, pos=9).named == {"n": 34}
    assert parse.search("id={n:d}", text, pos=9, endpos=17).named == {"n": 34}
    # As with re.search, endpos may truncate a variable-width field.
    assert parse.search("id={n:d}", text, pos=9, endpos=16).named == {"n": 3}
    assert parse.search("id={n:d}", text, pos=9, endpos=14) is None
    result = parse.search("id={n:d}", text, pos=9)
    assert result.spans == {"n": (15, 17)}


def test_search_wrapper_defaults_include_position_zero_and_are_case_insensitive():
    assert parse.search("A{:d}", "a1 A2").fixed == (1,)


def test_findall_is_lazy_iterable_and_honors_bounds_and_deferred_results():
    iterator = parse.findall("<{:d}>", "x<1>y<22>z<333>", pos=4, endpos=12)
    assert iter(iterator) is iterator
    assert [r.fixed for r in iterator] == [(22,)]
    deferred = list(parse.findall("{:d}", "1 2", evaluate_result=False))
    assert all(isinstance(item, Match) for item in deferred)
    assert [item.evaluate_result().fixed for item in deferred] == [(1,), (2,)]


def test_findall_wrapper_defaults_include_position_zero_and_ignore_case():
    assert [r.fixed for r in parse.findall("A{:d}", "a1 A2")] == [(1,), (2,)]


def test_repeated_names_are_backreferences_and_types_must_agree():
    assert parse.parse("{x:d},{x:d}", "12,12").named == {"x": 12}
    assert parse.parse("{x:d},{x:d}", "12,13") is None
    with pytest.raises(RepeatedNameError, match="does not match"):
        Parser("{x:d}/{x:f}")


def test_group_name_collisions_and_nested_names_are_preserved():
    result = parse.parse(
        "{a.b}|{a_b}|{x-y}|{x_y}|{person[address][city]}",
        "dot|under|dash|under2|Dhaka",
    )
    assert result.named == {
        "a.b": "dot",
        "a_b": "under",
        "x-y": "dash",
        "x_y": "under2",
        "person": {"address": {"city": "Dhaka"}},
    }
    assert result.spans["a.b"] == (0, 3)
    assert result.spans["person[address][city]"] == (22, 27)


@pytest.mark.parametrize(
    "spec,text,expected",
    [
        (",d", "-1,234", -1234),
        ("d", "+0xff", 255),
        ("d", "0B1011", 11),
        ("d", "0o17", 15),
        ("n", "1.234.567", 1234567),
        ("b", "-0b101", -5),
        ("o", "+0O17", 15),
        ("x", "0xAf", 175),
        ("X", "FF", 255),
        ("%", "12.5%", 0.125),
        ("f", "-.25", -0.25),
        ("f", "+INF", math.inf),
        ("e", "-1.25e+3", -1250.0),
        ("g", "2.5E-2", 0.025),
        ("F", "12.50", Decimal("12.50")),
    ],
)
def test_numeric_types(spec, text, expected):
    actual = value("{:" + spec + "}", text)
    assert actual == expected
    assert type(actual) is type(expected)


def test_float_special_values_and_precision_zero():
    assert math.isnan(value("{:f}", "nan"))
    assert value("{:.0f}", "-20") == -20.0
    assert value("{:.0F}", "+20") == Decimal("20")
    assert parse.parse("{:f}", "20") is None
    assert parse.parse("{:.0f}", "20.1") is None


def test_numeric_grammar_rejects_invalid_digits_and_width_limits_digits():
    assert parse.parse("{:b}", "102") is None
    assert parse.parse("{:o}", "08") is None
    assert parse.parse("{:x}", "gh") is None
    assert parse.parse("{:3d}", "123") is not None
    assert parse.parse("{:3d}", "1234") is None


@pytest.mark.parametrize(
    "fmt,text,expected",
    [
        ("{:>5d}", "   12", 12),
        ("{:<5d}", "12   ", 12),
        ("{:^6d}", "  12  ", 12),
        ("{:#=5d}", "###12", 12),
        ("{:05d}", "00012", 12),
        ("{:*^6d}", "**12**", 12),
        ("{:2.4}", "abc", "abc"),
    ],
)
def test_width_precision_fill_and_alignment(fmt, text, expected):
    assert value(fmt, text) == expected


def test_width_precision_and_type_validation_boundaries():
    assert parse.parse("{:2.4}", "a") is None
    assert parse.parse("{:2.4}", "abcde") is None
    assert value("{:.3}", "abc") == "abc"
    assert parse.parse("{:.3}", "abcd") is None
    with pytest.raises(ValueError, match="not recognised"):
        Parser("{:q}")


def test_letter_and_string_character_types():
    assert value("{:l}", "AbCd") == "AbCd"
    assert parse.parse("{:l}", "abc1") is None
    assert value("{:w}", "abc_123") == "abc_123"
    assert value("{:W}", "-! ") == "-! "
    assert value("{:s}", " \t") == " \t"
    assert value("{:S}", "abc") == "abc"


def test_custom_type_pattern_groups_and_decorator_metadata():
    @parse.with_pattern(r"([A-Z]+)-(\d+)", regex_group_count=2)
    def token(text):
        letters, number = text.split("-")
        return letters, int(number)

    assert token.pattern == r"([A-Z]+)-(\d+)"
    assert token.regex_group_count == 2
    result = parse.parse("{item:Token}:{count:d}", "ABC-12:7", {"Token": token})
    assert result.named == {"item": ("ABC", 12), "count": 7}


def test_custom_type_default_pattern_and_none_group_count():
    def upper(text):
        return text.upper()

    upper.regex_group_count = None
    result = parse.parse("{x:upper}/{:d}", "hello/3", {"upper": upper})
    assert result.named == {"x": "HELLO"}
    assert result.fixed == (3,)


def test_convert_first_is_pickleable():
    converter = parse.convert_first(int)
    restored = pickle.loads(pickle.dumps(converter))
    assert restored("14", None) == 14


@pytest.mark.parametrize(
    "spec,text,expected",
    [
        ("ti", "2024-02-29T23:04:05.1234567Z", datetime(2024, 2, 29, 23, 4, 5, 123456, FixedTzOffset(0, "UTC"))),
        ("tg", "31/December/2020 11:59 PM +05:30", datetime(2020, 12, 31, 23, 59, tzinfo=FixedTzOffset(330, "+05:30"))),
        ("ta", "December/31/2020 12:01 AM -0430", datetime(2020, 12, 31, 0, 1, tzinfo=FixedTzOffset(-270, "-0430"))),
        ("te", "Mon, 7 Jan 2019 08:09:10 +0200", datetime(2019, 1, 7, 8, 9, 10, tzinfo=FixedTzOffset(120, "+0200"))),
        ("th", "7/Jan/2019:08:09:10 -03:30", datetime(2019, 1, 7, 8, 9, 10, tzinfo=FixedTzOffset(-210, "-03:30"))),
        ("tc", "Mon Jan 7 08:09:10 2019", datetime(2019, 1, 7, 8, 9, 10)),
        ("tt", "12:34:56.25 PM +0130", time(12, 34, 56, 250000, FixedTzOffset(90, "+0130"))),
    ],
)
def test_built_in_datetime_types(spec, text, expected):
    assert value("{:" + spec + "}", text) == expected


def test_syslog_date_uses_current_year_and_time_is_optional():
    year = datetime.today().year
    assert value("{:ts}", "Feb 3 04:05:06") == datetime(year, 2, 3, 4, 5, 6)
    assert value("{:ts}", "Feb 3 ") == datetime(year, 2, 3)


def test_iso_date_can_omit_time_and_support_offset_without_colon():
    assert value("{:ti}", "2020-01-02") == datetime(2020, 1, 2)
    assert value("{:ti}", "2020-01-02 03:04 +0530") == datetime(
        2020, 1, 2, 3, 4, tzinfo=FixedTzOffset(330, "+0530")
    )


def test_strptime_style_formats_return_date_time_or_datetime():
    current_year = datetime.today().year
    assert value("{:%Y-%m-%d}", "2023-07-09") == date(2023, 7, 9)
    assert value("{:%H:%M:%S.%f}", "06:07:08.9") == time(6, 7, 8, 900000)
    assert value("{:%d/%b %Y %H:%M}", "2/Jan 2020 03:04") == datetime(2020, 1, 2, 3, 4)
    assert value("{:%b-%d}", "Feb-03") == date(current_year, 2, 3)


def test_fixed_timezone_behavior():
    east = FixedTzOffset(90, "X")
    assert east.utcoffset(None) == timedelta(minutes=90)
    assert east.dst(None) == timedelta(0)
    assert east.tzname(None) == "X"
    assert repr(east) == "<FixedTzOffset X 1:30:00>"
    assert east == FixedTzOffset(90, "X")
    assert east != FixedTzOffset(90, "Y")
    assert east != FixedTzOffset(91, "X")
    assert east != object()


def test_multiple_date_fields_keep_group_indexes_correct():
    result = parse.parse(
        "{when:ti}|{amount:g}|{clock:tt}",
        "2020-01-02T03:04:05Z|1.5e2|9:08 AM",
    )
    assert result.named == {
        "when": datetime(2020, 1, 2, 3, 4, 5, tzinfo=FixedTzOffset(0, "UTC")),
        "amount": 150.0,
        "clock": time(9, 8),
    }


def test_public_compile_returns_reusable_parser():
    parser = parse.compile("{name}:{value:d}", case_sensitive=True)
    assert isinstance(parser, Parser)
    assert parser.parse("A:1").named == {"name": "A", "value": 1}
    assert parser.parse("a:2").named == {"name": "a", "value": 2}
    assert parse.compile("abc").parse("ABC") is not None
