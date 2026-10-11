"""Behavioral and regression tests for isodate's public API.

The examples are deliberately independent of Python's fromisoformat methods so
that these tests exercise isodate's own parsing and formatting code.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pickle

import pytest

import isodate
from isodate import Duration, ISO8601Error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-02-29", date(2024, 2, 29)),
        ("20240229", date(2024, 2, 29)),
        ("2024-060", date(2024, 2, 29)),
        ("2024060", date(2024, 2, 29)),
        ("2020-W53-7", date(2021, 1, 3)),
        ("2020W537", date(2021, 1, 3)),
        ("2024-W01", date(2024, 1, 1)),
        ("2024-02", date(2024, 2, 1)),
        ("2024", date(2024, 1, 1)),
        ("20", date(2001, 1, 1)),
    ],
)
def test_parse_date_supported_representations(text, expected):
    assert isodate.parse_date(text) == expected


def test_parse_date_defaults_and_expanded_year():
    assert isodate.parse_date("2024", defaultmonth=7, defaultday=6) == date(2024, 7, 6)
    assert isodate.parse_date("+002024-02-29", yeardigits=6) == date(2024, 2, 29)


@pytest.mark.parametrize("text", ["", "2024/01/01", "2024-1-01", "not-a-date"])
def test_parse_date_rejects_unknown_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_date(text)


@pytest.mark.parametrize("text", ["2023-02-29", "2024-13-01", "0000-01-01"])
def test_parse_date_rejects_unrepresentable_dates(text):
    with pytest.raises(ValueError):
        isodate.parse_date(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23:59:58", time(23, 59, 58)),
        ("235958", time(23, 59, 58)),
        ("T12:30", time(12, 30)),
        ("12:30.5", time(12, 30, 30)),
        ("12.5", time(12, 30)),
        ("01:02:03,1234569", time(1, 2, 3, 123456)),
    ],
)
def test_parse_time_complete_reduced_and_fractional(text, expected):
    assert isodate.parse_time(text) == expected


@pytest.mark.parametrize(
    ("text", "offset"),
    [
        ("12:00:00Z", timedelta(0)),
        ("12:00:00+05:30", timedelta(hours=5, minutes=30)),
        ("120000-0430", -timedelta(hours=4, minutes=30)),
        ("12:00+02", timedelta(hours=2)),
    ],
)
def test_parse_time_timezone_offsets(text, offset):
    parsed = isodate.parse_time(text)
    assert parsed.utcoffset() == offset


@pytest.mark.parametrize("text", ["", "1:02", "12:3", "12:00:00 UTC", "noon"])
def test_parse_time_rejects_unknown_syntax(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_time(text)


def test_parse_datetime_combines_date_time_and_timezone():
    parsed = isodate.parse_datetime("2024-02-29T23:59:58.25-04:30")
    assert parsed.replace(tzinfo=None) == datetime(2024, 2, 29, 23, 59, 58, 250000)
    assert parsed.utcoffset() == -timedelta(hours=4, minutes=30)


@pytest.mark.parametrize("text", ["2024-01-01 12:00:00", "2024-01-01", "T12:00:00"])
def test_parse_datetime_requires_valid_date_and_t_designator(text):
    with pytest.raises(ISO8601Error):
        isodate.parse_datetime(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P2W", timedelta(days=14)),
        ("P3DT4H5M6.25S", timedelta(days=3, hours=4, minutes=5, seconds=6.25)),
        ("PT0.5H", timedelta(minutes=30)),
        ("-P1DT2S", -timedelta(days=1, seconds=2)),
        ("P0D", timedelta(0)),
    ],
)
def test_parse_duration_as_timedelta(text, expected):
    parsed = isodate.parse_duration(text)
    assert type(parsed) is timedelta
    assert parsed == expected


def test_parse_duration_with_calendar_components():
    parsed = isodate.parse_duration("P1Y2M3DT4H5M6.5S")
    assert isinstance(parsed, Duration)
    assert parsed.years == Decimal("1")
    assert parsed.months == Decimal("2")
    assert parsed.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.5)


def test_duration_can_be_forced_to_duration_without_years_or_months():
    parsed = isodate.parse_duration("P2D", as_timedelta_if_possible=False)
    assert isinstance(parsed, Duration)
    assert parsed == Duration(days=2)


def test_duration_rejects_bad_values_and_non_strings():
    with pytest.raises(ISO8601Error):
        isodate.parse_duration("one day")
    with pytest.raises(TypeError):
        isodate.parse_duration(3)  # type: ignore[arg-type]


def test_duration_calendar_arithmetic_clamps_end_of_month():
    assert date(2024, 1, 31) + Duration(months=1) == date(2024, 2, 29)
    assert date(2023, 3, 31) - Duration(months=1) == date(2023, 2, 28)
    assert datetime(2020, 2, 29, 8, 15) + Duration(years=1) == datetime(2021, 2, 28, 8, 15)


def test_duration_arithmetic_and_equivalent_calendar_units():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(months=2, days=1) + Duration(months=3, hours=2) == Duration(
        months=5, days=1, hours=2
    )
    assert 3 * Duration(months=1, days=2) == Duration(months=3, days=6)
    assert timedelta(days=10) - Duration(days=3) == Duration(days=7)


def test_duration_to_timedelta_needs_exactly_one_anchor():
    duration = Duration(months=1)
    assert duration.totimedelta(start=date(2024, 1, 31)) == timedelta(days=29)
    assert duration.totimedelta(end=date(2024, 3, 31)) == timedelta(days=31)
    with pytest.raises(ValueError):
        duration.totimedelta()
    with pytest.raises(ValueError):
        duration.totimedelta(start=date(2024, 1, 1), end=date(2024, 2, 1))


def test_fractional_calendar_duration_cannot_be_applied_to_date():
    with pytest.raises(ValueError):
        date(2024, 1, 1) + Duration(months=Decimal("0.5"))


def test_isoformat_round_trips_public_types():
    d = date(2024, 2, 29)
    t = time(7, 8, 9, tzinfo=isodate.parse_tzinfo("+05:30"))
    dt = datetime.combine(d, t)
    delta = timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=600000)

    assert isodate.date_isoformat(d) == "2024-02-29"
    assert isodate.parse_date(isodate.date_isoformat(d)) == d
    assert isodate.time_isoformat(t) == "07:08:09+05:30"
    assert isodate.parse_time(isodate.time_isoformat(t)) == t
    assert isodate.datetime_isoformat(dt) == "2024-02-29T07:08:09+05:30"
    assert isodate.parse_datetime(isodate.datetime_isoformat(dt)) == dt
    assert isodate.duration_isoformat(delta) == "P2DT3H4M5.6S"
    assert isodate.parse_duration(isodate.duration_isoformat(delta)) == delta


def test_duration_isoformat_zero_negative_and_calendar_values():
    assert isodate.duration_isoformat(timedelta(0)) == "P0D"
    assert isodate.duration_isoformat(-timedelta(seconds=1)) == "-PT1S"
    assert isodate.duration_isoformat(Duration(years=1, months=2, days=3)) == "P1Y2M3D"


def test_timezone_parsing_formatting_and_pickle_identity():
    assert isodate.parse_tzinfo("") is None
    assert isodate.parse_tzinfo("Z") is isodate.UTC
    assert pickle.loads(pickle.dumps(isodate.UTC)) is isodate.UTC

    east = isodate.parse_tzinfo("+05:45")
    west = isodate.parse_tzinfo("-0330")
    assert east is not None and east.utcoffset(None) == timedelta(hours=5, minutes=45)
    assert west is not None and west.utcoffset(None) == -timedelta(hours=3, minutes=30)
    assert isodate.tz_isoformat(datetime(2024, 1, 1, tzinfo=east)) == "+05:45"
    assert isodate.tz_isoformat(datetime(2024, 1, 1, tzinfo=west), "%z") == "-0330"
    assert isodate.tz_isoformat(datetime(2024, 1, 1, tzinfo=isodate.UTC)) == "Z"
    assert isodate.tz_isoformat(datetime(2024, 1, 1)) == ""


def test_timezone_rejects_unknown_format_directive():
    with pytest.raises(ValueError):
        isodate.tz_isoformat(
            datetime(2024, 1, 1, tzinfo=isodate.FixedOffset(1, 0, "+01")), "%q"
        )


def test_custom_strftime_date_week_ordinal_and_literal_percent():
    value = date(2021, 1, 3)  # Sunday in ISO week 53 of 2020.
    assert isodate.strftime(value, "%Y-%j W%W-%w %%") == "2021-003 W53-7 %"
