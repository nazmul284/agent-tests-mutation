"""Test suite for the isodate package."""

import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    D_ALT_BAS,
    D_ALT_EXT,
    D_WEEK,
    DATE_BAS_COMPLETE,
    DATE_BAS_MONTH,
    DATE_BAS_ORD_COMPLETE,
    DATE_BAS_WEEK,
    DATE_BAS_WEEK_COMPLETE,
    DATE_CENTURY,
    DATE_EXT_COMPLETE,
    DATE_EXT_MONTH,
    DATE_EXT_ORD_COMPLETE,
    DATE_EXT_WEEK,
    DATE_EXT_WEEK_COMPLETE,
    DATE_YEAR,
    DT_BAS_COMPLETE,
    DT_EXT_COMPLETE,
    DT_EXT_ORD_COMPLETE,
    DT_EXT_WEEK_COMPLETE,
    LOCAL,
    TIME_BAS_COMPLETE,
    TIME_BAS_MINUTE,
    TIME_EXT_COMPLETE,
    TIME_EXT_MINUTE,
    TIME_HOUR,
    TZ_BAS,
    TZ_EXT,
    TZ_HOUR,
    UTC,
    Duration,
    FixedOffset,
    ISO8601Error,
    date_isoformat,
    datetime_isoformat,
    duration_isoformat,
    parse_date,
    parse_datetime,
    parse_duration,
    parse_time,
    parse_tzinfo,
    strftime,
    time_isoformat,
    tz_isoformat,
)
from isodate.duration import fquotmod, max_days_in_month
from isodate.tzinfo import DSTOFFSET, STDOFFSET, ZERO


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def offset(t):
    """Return the utc offset of a parsed time/datetime."""
    return t.utcoffset()


# ---------------------------------------------------------------------------
# duration helpers
# ---------------------------------------------------------------------------


class TestFquotmod:
    @pytest.mark.parametrize(
        "val, expected",
        [
            (1, (0, 1)),
            (12, (0, 12)),
            (13, (1, 1)),
            (24, (1, 12)),
            (25, (2, 1)),
            (0, (-1, 12)),
            (-1, (-1, 11)),
            (-11, (-1, 1)),
            (-12, (-2, 12)),
        ],
    )
    def test_month_wrapping(self, val, expected):
        div, mod = fquotmod(Decimal(val), 1, 13)
        assert (div, mod) == expected
        assert isinstance(div, int)

    def test_returns_decimal_mod(self):
        _, mod = fquotmod(Decimal("14.5"), 1, 13)
        assert mod == Decimal("2.5")


class TestMaxDaysInMonth:
    @pytest.mark.parametrize("month", [1, 3, 5, 7, 8, 10, 12])
    def test_long_months(self, month):
        assert max_days_in_month(2001, month) == 31

    @pytest.mark.parametrize("month", [4, 6, 9, 11])
    def test_short_months(self, month):
        assert max_days_in_month(2001, month) == 30

    @pytest.mark.parametrize(
        "year, days",
        [(2000, 29), (2004, 29), (1996, 29), (1900, 28), (2100, 28), (2001, 28), (2003, 28),
         (2400, 29)],
    )
    def test_february(self, year, days):
        assert max_days_in_month(year, 2) == days

    @pytest.mark.parametrize("year", range(1890, 2110))
    def test_matches_stdlib(self, year):
        for month in range(1, 13):
            nxt = date(year + (month == 12), month % 12 + 1, 1)
            assert max_days_in_month(year, month) == (nxt - timedelta(days=1)).day


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1985-04-12", date(1985, 4, 12)),
            ("19850412", date(1985, 4, 12)),
            ("1985-102", date(1985, 4, 12)),
            ("1985102", date(1985, 4, 12)),
            ("1985-W15-5", date(1985, 4, 12)),
            ("1985W155", date(1985, 4, 12)),
            ("1985-W15", date(1985, 4, 8)),
            ("1985W15", date(1985, 4, 8)),
            ("1985-04", date(1985, 4, 1)),
            ("198504", date(1985, 4, 1)),
            ("1985", date(1985, 1, 1)),
            ("19", date(1901, 1, 1)),
            ("20", date(2001, 1, 1)),
            ("2000-02-29", date(2000, 2, 29)),
            ("2000-366", date(2000, 12, 31)),
            ("2001-001", date(2001, 1, 1)),
            ("0001-01-01", date(1, 1, 1)),
            ("9999-12-31", date(9999, 12, 31)),
        ],
    )
    def test_formats(self, text, expected):
        result = parse_date(text)
        assert result == expected
        assert type(result) is date

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("2004-W53-6", date(2005, 1, 1)),
            ("2009-W01-1", date(2008, 12, 29)),
            ("2009-W53-7", date(2010, 1, 3)),
            ("2008-W01-1", date(2007, 12, 31)),
            ("2010-W01-1", date(2010, 1, 4)),
            ("2005-W01-1", date(2005, 1, 3)),
        ],
    )
    def test_week_dates_across_year_boundaries(self, text, expected):
        assert parse_date(text) == expected

    @pytest.mark.parametrize("year", range(1995, 2035))
    def test_week_dates_match_stdlib(self, year):
        for week in (1, 2, 10, 26, 52):
            for day in (1, 4, 7):
                text = "%04d-W%02d-%d" % (year, week, day)
                assert parse_date(text) == date.fromisocalendar(year, week, day), text
                basic = "%04dW%02d%d" % (year, week, day)
                assert parse_date(basic) == date.fromisocalendar(year, week, day), basic

    @pytest.mark.parametrize("year", range(1995, 2035))
    def test_reduced_week_is_monday(self, year):
        for week in (1, 2, 30, 52):
            result = parse_date("%04d-W%02d" % (year, week))
            assert result == date.fromisocalendar(year, week, 1)
            assert result.isoweekday() == 1

    @pytest.mark.parametrize("year", [1999, 2000, 2001, 2024])
    def test_ordinal_dates_match_stdlib(self, year):
        start = date(year, 1, 1)
        for day in (1, 31, 59, 60, 100, 365):
            assert parse_date("%04d-%03d" % (year, day)) == start + timedelta(days=day - 1)

    def test_default_month_and_day_for_year(self):
        assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)

    def test_default_day_for_month(self):
        assert parse_date("1985-04", defaultday=10) == date(1985, 4, 10)
        assert parse_date("198504", defaultday=10) == date(1985, 4, 10)

    def test_default_for_century(self):
        assert parse_date("19", defaultmonth=7, defaultday=4) == date(1901, 7, 4)

    def test_defaults_do_not_affect_complete_date(self):
        assert parse_date("1985-04-12", defaultmonth=6, defaultday=15) == date(1985, 4, 12)

    def test_expanded_four_digit(self):
        assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)
        assert parse_date("+19850412", expanded=True) == date(1985, 4, 12)
        assert parse_date("+1985-W15-5", expanded=True) == date(1985, 4, 12)
        assert parse_date("+1985-102", expanded=True) == date(1985, 4, 12)
        assert parse_date("+1985", expanded=True) == date(1985, 1, 1)
        assert parse_date("+19", expanded=True) == date(1901, 1, 1)

    def test_expanded_requires_sign(self):
        with pytest.raises(ISO8601Error):
            parse_date("1985-04-12", expanded=True)

    def test_sign_not_allowed_without_expanded(self):
        with pytest.raises(ISO8601Error):
            parse_date("+1985-04-12")

    def test_six_year_digits(self):
        assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+0019850412", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985-W15-5", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985-102", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985-04", yeardigits=6) == date(1985, 4, 1)
        assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
        assert parse_date("+0019", yeardigits=6) == date(1901, 1, 1)

    def test_six_year_digits_forces_sign(self):
        with pytest.raises(ISO8601Error):
            parse_date("001985-04-12", yeardigits=6)

    def test_negative_year_not_representable(self):
        with pytest.raises(ValueError):
            parse_date("-001985-04-12", yeardigits=6)

    def test_year_too_large_for_date(self):
        with pytest.raises(ValueError):
            parse_date("+010000-01-01", yeardigits=6)

    @pytest.mark.parametrize("text", ["1985-13-01", "1985-02-30", "2001-02-29", "1985-04-31"])
    def test_invalid_calendar_values(self, text):
        with pytest.raises(ValueError):
            parse_date(text)

    @pytest.mark.parametrize(
        "text",
        ["", "abcd", "1985-4-12", "85-04-12", "1985/04/12", "1985-04-12T", "1985-04-12 ",
         " 1985", "1985-W1", "198", "1", "1985-W15-55", "1985-04-1"],
    )
    def test_unrecognised(self, text):
        with pytest.raises(ISO8601Error):
            parse_date(text)

    def test_error_is_value_error(self):
        assert issubclass(ISO8601Error, ValueError)


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


class TestParseTime:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("23:20:50", time(23, 20, 50)),
            ("232050", time(23, 20, 50)),
            ("T23:20:50", time(23, 20, 50)),
            ("T232050", time(23, 20, 50)),
            ("23:20", time(23, 20)),
            ("2320", time(23, 20)),
            ("T2320", time(23, 20)),
            ("23", time(23)),
            ("T23", time(23)),
            ("00:00:00", time(0)),
            ("23:20:50.5", time(23, 20, 50, 500000)),
            ("23:20:50,5", time(23, 20, 50, 500000)),
            ("232050.5", time(23, 20, 50, 500000)),
            ("23:20:50.123456", time(23, 20, 50, 123456)),
            ("23:20:50.000001", time(23, 20, 50, 1)),
            ("23:20.5", time(23, 20, 30)),
            ("2320,8", time(23, 20, 48)),
            ("23:20.25", time(23, 20, 15)),
            ("23.5", time(23, 30)),
            ("23,3", time(23, 18)),
            ("10.25", time(10, 15)),
            ("12.0125", time(12, 0, 45)),
        ],
    )
    def test_naive_formats(self, text, expected):
        result = parse_time(text)
        assert result == expected
        assert result.tzinfo is None

    def test_fraction_truncated_not_rounded(self):
        # more than microsecond precision is truncated (floor)
        assert parse_time("23:20:50.1234567") == time(23, 20, 50, 123456)
        assert parse_time("23:20:50.9999999") == time(23, 20, 50, 999999)

    def test_minute_fraction_sub_second(self):
        # 0.0001 minute = 0.006 s
        assert parse_time("10:00.0001") == time(10, 0, 0, 6000)

    @pytest.mark.parametrize(
        "text, delta",
        [
            ("23:20:50Z", timedelta(0)),
            ("23:20:50+01:00", timedelta(hours=1)),
            ("23:20:50+0100", timedelta(hours=1)),
            ("23:20:50+01", timedelta(hours=1)),
            ("23:20:50-05:00", timedelta(hours=-5)),
            ("23:20:50-0530", timedelta(hours=-5, minutes=-30)),
            ("23:20:50+05:45", timedelta(hours=5, minutes=45)),
            ("232050-0330", timedelta(hours=-3, minutes=-30)),
            ("2320+0200", timedelta(hours=2)),
            ("23-04", timedelta(hours=-4)),
            ("23:20:50.5+12:00", timedelta(hours=12)),
        ],
    )
    def test_time_zones(self, text, delta):
        result = parse_time(text)
        assert result.utcoffset() == delta

    def test_z_is_utc_singleton(self):
        assert parse_time("10:00Z").tzinfo is UTC

    def test_fixed_offset_name(self):
        assert parse_time("10:00+05:30").tzname() == "+05:30"
        assert parse_time("10:00-0800").tzname() == "-0800"

    def test_tz_values_with_fraction(self):
        result = parse_time("23:20:50,25-02:00")
        assert result.replace(tzinfo=None) == time(23, 20, 50, 250000)
        assert result.utcoffset() == timedelta(hours=-2)

    @pytest.mark.parametrize(
        "text",
        ["", "2", "23:2", "23:20:5", "2:20:50", "23:20:50.", "23-20-50", "23:20:50 Z",
         "23:20:50+1", "abc", "23:20:50Z+01:00", "23:20:50+01:0"],
    )
    def test_unrecognised(self, text):
        with pytest.raises(ISO8601Error):
            parse_time(text)

    @pytest.mark.parametrize("text", ["25:00:00", "23:60:00", "23:20:61"])
    def test_out_of_range(self, text):
        with pytest.raises(ValueError):
            parse_time(text)


# ---------------------------------------------------------------------------
# parse_datetime
# ---------------------------------------------------------------------------


class TestParseDatetime:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
            ("19850412T101530", datetime(1985, 4, 12, 10, 15, 30)),
            ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
            ("1985-W15-5T10", datetime(1985, 4, 12, 10)),
            ("1985-04-12T10:15:30.5", datetime(1985, 4, 12, 10, 15, 30, 500000)),
        ],
    )
    def test_naive(self, text, expected):
        result = parse_datetime(text)
        assert result == expected
        assert result.tzinfo is None
        assert type(result) is datetime

    def test_utc(self):
        result = parse_datetime("1985-04-12T10:15:30Z")
        assert result.tzinfo is UTC
        assert result == datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)

    def test_offset(self):
        result = parse_datetime("1985-04-12T10:15:30+04:00")
        assert result.utcoffset() == timedelta(hours=4)
        assert result == datetime(1985, 4, 12, 6, 15, 30, tzinfo=UTC)

    def test_negative_offset(self):
        result = parse_datetime("19850412T101530-0400")
        assert result == datetime(1985, 4, 12, 14, 15, 30, tzinfo=UTC)

    @pytest.mark.parametrize(
        "text", ["1985-04-12", "1985-04-12 10:15:30", "1985-04-12T10:15T30", "T10:15", ""]
    )
    def test_invalid(self, text):
        with pytest.raises(ISO8601Error):
            parse_datetime(text)

    def test_invalid_time_part(self):
        with pytest.raises(ISO8601Error):
            parse_datetime("1985-04-12Tabc")


# ---------------------------------------------------------------------------
# tzinfo
# ---------------------------------------------------------------------------


class TestParseTzinfo:
    def test_utc(self):
        assert parse_tzinfo("Z") is UTC

    def test_empty(self):
        assert parse_tzinfo("") is None

    @pytest.mark.parametrize(
        "text, delta",
        [
            ("+01:00", timedelta(hours=1)),
            ("+0100", timedelta(hours=1)),
            ("+01", timedelta(hours=1)),
            ("-05:30", timedelta(hours=-5, minutes=-30)),
            ("-0530", timedelta(hours=-5, minutes=-30)),
            ("+00:00", timedelta(0)),
            ("+14:00", timedelta(hours=14)),
            ("-12", timedelta(hours=-12)),
        ],
    )
    def test_offsets(self, text, delta):
        tz = parse_tzinfo(text)
        assert isinstance(tz, FixedOffset)
        assert tz.utcoffset(None) == delta
        assert tz.dst(None) == ZERO
        assert tz.tzname(None) == text


class TestFixedOffsetAndUtc:
    def test_defaults(self):
        tz = FixedOffset()
        assert tz.utcoffset(None) == ZERO
        assert tz.tzname(None) == "UTC"
        assert tz.dst(None) == ZERO

    def test_values(self):
        tz = FixedOffset(-3, -30, "NST")
        assert tz.utcoffset(None) == timedelta(hours=-3, minutes=-30)
        assert tz.tzname(None) == "NST"
        assert repr(tz) == "<FixedOffset 'NST'>"

    def test_used_by_datetime(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=FixedOffset(5, 30, "IST"))
        assert dt.astimezone(UTC) == datetime(2020, 1, 1, 6, 30, tzinfo=UTC)

    def test_utc(self):
        assert UTC.utcoffset(None) == ZERO
        assert UTC.dst(None) == ZERO
        assert UTC.tzname(None) == "UTC"

    def test_utc_pickle_singleton(self):
        assert pickle.loads(pickle.dumps(UTC)) is UTC

    def test_datetime_with_utc_pickles(self):
        dt = datetime(2020, 1, 1, tzinfo=UTC)
        restored = pickle.loads(pickle.dumps(dt))
        assert restored == dt
        assert restored.tzinfo is UTC


class TestLocalTimezone:
    def test_offset_is_std_or_dst(self):
        for month in (1, 7):
            dt = datetime(2020, month, 15, 12)
            off = LOCAL.utcoffset(dt)
            assert off in (STDOFFSET, DSTOFFSET)
            assert LOCAL.dst(dt) == off - STDOFFSET
            assert isinstance(LOCAL.tzname(dt), str)

    def test_none_raises(self):
        with pytest.raises(Exception):
            LOCAL.utcoffset(None)


class TestTzIsoformat:
    def test_naive(self):
        assert tz_isoformat(datetime(2020, 1, 1)) == ""

    def test_utc(self):
        assert tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"

    def test_zero_fixed_offset_is_z(self):
        assert tz_isoformat(datetime(2020, 1, 1, tzinfo=FixedOffset(0, 0, "x"))) == "Z"

    @pytest.mark.parametrize(
        "tz, ext, bas, hour",
        [
            (FixedOffset(1, 0), "+01:00", "+0100", "+01"),
            (FixedOffset(5, 30), "+05:30", "+0530", "+05"),
            (FixedOffset(-5, 0), "-05:00", "-0500", "-05"),
            (FixedOffset(-3, -30), "-03:30", "-0330", "-03"),
            (FixedOffset(12, 45), "+12:45", "+1245", "+12"),
            (FixedOffset(0, -30), "-00:30", "-0030", "-00"),
        ],
    )
    def test_formats(self, tz, ext, bas, hour):
        dt = datetime(2020, 1, 1, tzinfo=tz)
        assert tz_isoformat(dt) == ext
        assert tz_isoformat(dt, "%Z") == ext
        assert tz_isoformat(dt, "%z") == bas
        assert tz_isoformat(dt, "%h") == hour

    def test_unknown_format(self):
        with pytest.raises(ValueError):
            tz_isoformat(datetime(2020, 1, 1, tzinfo=FixedOffset(1)), "%x")

    def test_overflow(self):
        dt = datetime(2020, 1, 1, tzinfo=FixedOffset(100))
        with pytest.raises(OverflowError):
            tz_isoformat(dt)

    def test_works_with_time(self):
        assert tz_isoformat(time(10, tzinfo=FixedOffset(2))) == "+02:00"


# ---------------------------------------------------------------------------
# strftime / date / time / datetime formatting
# ---------------------------------------------------------------------------


D = date(1985, 4, 12)
DT = datetime(1985, 4, 12, 23, 20, 50, 123456)


class TestDateFormatting:
    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (DATE_EXT_COMPLETE, "1985-04-12"),
            (DATE_BAS_COMPLETE, "19850412"),
            (DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
            (DATE_BAS_WEEK_COMPLETE, "1985W155"),
            (DATE_EXT_ORD_COMPLETE, "1985-102"),
            (DATE_BAS_ORD_COMPLETE, "1985102"),
            (DATE_EXT_WEEK, "1985-W15"),
            (DATE_BAS_WEEK, "1985W15"),
            (DATE_EXT_MONTH, "1985-04"),
            (DATE_BAS_MONTH, "198504"),
            (DATE_YEAR, "1985"),
            (DATE_CENTURY, "19"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert date_isoformat(D, fmt) == expected
        assert strftime(D, fmt) == expected

    def test_default(self):
        assert date_isoformat(D) == "1985-04-12"

    def test_padding(self):
        assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
        assert date_isoformat(date(5, 1, 2), DATE_EXT_ORD_COMPLETE) == "0005-002"
        assert date_isoformat(date(2020, 1, 6), DATE_EXT_WEEK_COMPLETE) == "2020-W02-1"

    def test_weekday_numbers(self):
        # Monday 2020-01-06 .. Sunday 2020-01-12
        for i in range(7):
            assert strftime(date(2020, 1, 6) + timedelta(days=i), "%w") == str(i + 1)

    def test_ordinal_leap_year(self):
        assert strftime(date(2000, 12, 31), "%j") == "366"
        assert strftime(date(2001, 12, 31), "%j") == "365"
        assert strftime(date(2000, 3, 1), "%j") == "061"

    def test_expanded_year(self):
        assert date_isoformat(D, DATE_EXT_COMPLETE, yeardigits=6) == "+001985-04-12"
        assert date_isoformat(D, DATE_YEAR, yeardigits=6) == "+001985"
        assert date_isoformat(D, DATE_CENTURY, yeardigits=6) == "+0019"

    def test_literal_percent_and_unknown(self):
        assert strftime(D, "%%Y=%Y %q") == "%Y=1985 %q"

    @pytest.mark.parametrize(
        "text",
        ["1985-04-12", "2000-02-29", "0001-01-01", "9999-12-31", "2020-01-06"],
    )
    def test_round_trip(self, text):
        assert date_isoformat(parse_date(text)) == text

    @pytest.mark.parametrize("d", [date(1999, 6, 1), date(2020, 7, 15), date(2016, 3, 3)])
    def test_round_trip_alternative_formats(self, d):
        for fmt in (DATE_BAS_COMPLETE, DATE_EXT_ORD_COMPLETE, DATE_BAS_ORD_COMPLETE,
                    DATE_EXT_WEEK_COMPLETE, DATE_BAS_WEEK_COMPLETE):
            assert parse_date(date_isoformat(d, fmt)) == d, fmt


class TestTimeFormatting:
    @pytest.mark.parametrize(
        "fmt, expected",
        [
            (TIME_EXT_COMPLETE, "23:20:50"),
            (TIME_BAS_COMPLETE, "232050"),
            (TIME_EXT_MINUTE, "23:20"),
            (TIME_BAS_MINUTE, "2320"),
            (TIME_HOUR, "23"),
            (TIME_EXT_COMPLETE + ".%f", "23:20:50.123456"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert time_isoformat(time(23, 20, 50, 123456), fmt) == expected

    def test_default_naive(self):
        assert time_isoformat(time(1, 2, 3)) == "01:02:03"

    def test_default_with_tz(self):
        assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
        assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(-5, 0))) == "01:02:03-05:00"

    def test_tz_formats(self):
        t = time(1, 2, 3, tzinfo=FixedOffset(5, 30))
        assert time_isoformat(t, TIME_BAS_COMPLETE + TZ_BAS) == "010203+0530"
        assert time_isoformat(t, TIME_EXT_COMPLETE + TZ_EXT) == "01:02:03+05:30"
        assert time_isoformat(t, TIME_EXT_COMPLETE + TZ_HOUR) == "01:02:03+05"

    def test_microsecond_padding(self):
        assert strftime(time(0, 0, 0, 5), "%f") == "000005"

    @pytest.mark.parametrize(
        "text", ["23:20:50", "00:00:00", "12:00:00Z", "23:59:59+05:30", "01:02:03-08:00"]
    )
    def test_round_trip(self, text):
        assert time_isoformat(parse_time(text)) == text


class TestDatetimeFormatting:
    def test_default(self):
        dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)
        assert datetime_isoformat(dt) == "1985-04-12T10:15:30Z"

    def test_naive(self):
        assert datetime_isoformat(datetime(1985, 4, 12, 10, 15, 30)) == "1985-04-12T10:15:30"

    def test_predefined(self):
        dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-4, 0))
        assert datetime_isoformat(dt, DT_EXT_COMPLETE) == "1985-04-12T10:15:30-04:00"
        assert datetime_isoformat(dt, DT_BAS_COMPLETE) == "19850412T101530-0400"
        assert datetime_isoformat(dt, DT_EXT_ORD_COMPLETE) == "1985-102T10:15:30-04:00"
        assert datetime_isoformat(dt, DT_EXT_WEEK_COMPLETE) == "1985-W15-5T10:15:30-04:00"
        assert isodate.DT_BAS_ORD_COMPLETE == "%Y%jT%H%M%S%z"
        assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "1985W155T101530-0400"

    @pytest.mark.parametrize(
        "text",
        ["1985-04-12T10:15:30", "1985-04-12T10:15:30Z", "2000-02-29T23:59:59+01:00",
         "0001-01-01T00:00:00-12:00"],
    )
    def test_round_trip(self, text):
        assert datetime_isoformat(parse_datetime(text)) == text

    def test_date_format_codes_on_datetime(self):
        assert strftime(DT, "%Y-%m-%dT%H:%M:%S.%f") == "1985-04-12T23:20:50.123456"


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


class TestParseDuration:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("P1D", timedelta(days=1)),
            ("PT1H", timedelta(hours=1)),
            ("PT1M", timedelta(minutes=1)),
            ("PT1S", timedelta(seconds=1)),
            ("PT36H", timedelta(hours=36)),
            ("P2W", timedelta(weeks=2)),
            ("P3DT4H5M6S", timedelta(days=3, hours=4, minutes=5, seconds=6)),
            ("P1DT12H", timedelta(days=1, hours=12)),
            ("PT0.5S", timedelta(milliseconds=500)),
            ("PT0,5S", timedelta(milliseconds=500)),
            ("PT1.5M", timedelta(seconds=90)),
            ("PT1.5H", timedelta(minutes=90)),
            ("P0.5D", timedelta(hours=12)),
            ("P0.5W", timedelta(days=3, hours=12)),
            ("P1W2D", timedelta(days=9)),
            ("P0D", timedelta(0)),
            ("PT0S", timedelta(0)),
            ("+P1D", timedelta(days=1)),
            ("-P1D", timedelta(days=-1)),
            ("-PT1H30M", timedelta(hours=-1, minutes=-30)),
            ("-P2W", timedelta(weeks=-2)),
            ("PT1M", timedelta(minutes=1)),
            ("PT1000S", timedelta(seconds=1000)),
        ],
    )
    def test_timedelta_results(self, text, expected):
        result = parse_duration(text)
        assert type(result) is timedelta
        assert result == expected

    def test_month_vs_minute(self):
        assert parse_duration("P1M") == Duration(months=1)
        assert parse_duration("PT1M") == timedelta(minutes=1)
        assert parse_duration("P1MT1M") == Duration(months=1, minutes=1)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("P1Y", Duration(years=1)),
            ("P2M", Duration(months=2)),
            ("P1Y2M", Duration(years=1, months=2)),
            ("P1Y2M3DT4H5M6S", Duration(years=1, months=2, days=3, hours=4, minutes=5,
                                        seconds=6)),
            ("P1Y2M3W", Duration(years=1, months=2, weeks=3)),
            ("P1,5Y", Duration(years=Decimal("1.5"))),
            ("P1.5M", Duration(months=Decimal("1.5"))),
            ("P0.5Y", Duration(months=6)),
        ],
    )
    def test_duration_results(self, text, expected):
        result = parse_duration(text)
        assert isinstance(result, Duration)
        assert result == expected

    def test_duration_components(self):
        result = parse_duration("P1Y2M3DT4H5M6.5S")
        assert result.years == 1
        assert result.months == 2
        assert isinstance(result.years, Decimal)
        assert isinstance(result.months, Decimal)
        assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6.5)

    def test_fractional_years_are_decimal_exact(self):
        result = parse_duration("P1.1Y")
        assert result.years == Decimal("1.1")

    def test_negative_duration(self):
        result = parse_duration("-P1Y2M3D")
        assert isinstance(result, Duration)
        assert result.years == -1
        assert result.months == -2
        assert result.tdelta == timedelta(days=-3)

    def test_as_timedelta_if_possible_false(self):
        result = parse_duration("P1DT2H", as_timedelta_if_possible=False)
        assert isinstance(result, Duration)
        assert result.years == 0 and result.months == 0
        assert result.tdelta == timedelta(days=1, hours=2)
        assert result == timedelta(days=1, hours=2)

    def test_as_timedelta_false_negative(self):
        result = parse_duration("-P1D", as_timedelta_if_possible=False)
        assert isinstance(result, Duration)
        assert result.tdelta == timedelta(days=-1)

    def test_alternative_extended(self):
        result = parse_duration("P0001-02-03T04:05:06")
        assert isinstance(result, Duration)
        assert result.years == 1
        assert result.months == 2
        assert result.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    def test_alternative_basic(self):
        result = parse_duration("P00010203T040506")
        assert result == Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)

    def test_alternative_fraction(self):
        result = parse_duration("P0002-01-01T00:00:01.5")
        assert result == Duration(years=2, months=1, days=1, seconds=1.5)

    @pytest.mark.parametrize(
        "text",
        ["", "P", "1D", "PD", "P1", "P1DT1", "P1H", "PT1D", "P1S", "P1D1Y", "P-1D", "P1.D",
         "XP1D", "P1DX", "P 1D", "--P1D", "P1Y2M3D4H"],
    )
    def test_invalid(self, text):
        with pytest.raises(ISO8601Error):
            parse_duration(text)

    @pytest.mark.parametrize("value", [None, 1, timedelta(1), b"P1D"])
    def test_type_error(self, value):
        with pytest.raises(TypeError):
            parse_duration(value)


# ---------------------------------------------------------------------------
# duration formatting
# ---------------------------------------------------------------------------


class TestDurationFormatting:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (timedelta(0), "P0D"),
            (timedelta(days=1), "P1D"),
            (timedelta(hours=1), "PT1H"),
            (timedelta(minutes=1), "PT1M"),
            (timedelta(seconds=1), "PT1S"),
            (timedelta(hours=36), "P1DT12H"),
            (timedelta(days=3, hours=4, minutes=5, seconds=6), "P3DT4H5M6S"),
            (timedelta(seconds=0.5), "PT0.5S"),
            (timedelta(seconds=1, microseconds=1), "PT1.000001S"),
            (timedelta(microseconds=120000), "PT0.12S"),
            (timedelta(days=1, seconds=1), "P1DT1S"),
            (timedelta(weeks=2), "P14D"),
            (timedelta(days=-1), "-P1D"),
            (timedelta(hours=-1, minutes=-30), "-PT1H30M"),
            (timedelta(seconds=-0.5), "-PT0.5S"),
            (Duration(years=1), "P1Y"),
            (Duration(months=1), "P1M"),
            (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
             "P1Y2M3DT4H5M6S"),
            (Duration(years=-1, months=-2), "-P1Y2M"),
            (Duration(days=-1), "-P1D"),
            (Duration(), "P0D"),
            (Duration(years=Decimal("1.5")), "P1.5Y"),
        ],
    )
    def test_default(self, value, expected):
        assert duration_isoformat(value) == expected

    @pytest.mark.parametrize(
        "text",
        ["P1D", "PT1H", "P1Y", "P1M", "P1Y2M3DT4H5M6S", "P1DT12H", "-P1D", "-P1Y2M3D",
         "PT0.5S", "P10Y", "-PT20M", "P2Y10M15DT10H30M20S"],
    )
    def test_round_trip(self, text):
        assert duration_isoformat(parse_duration(text)) == text

    def test_week_format(self):
        assert duration_isoformat(timedelta(weeks=3), D_WEEK) == "P3W"
        assert duration_isoformat(timedelta(days=10), D_WEEK) == "P1W"
        assert duration_isoformat(timedelta(days=-14), D_WEEK) == "-P2W"

    def test_strftime_components(self):
        td = timedelta(days=2, hours=3, minutes=4, seconds=5, microseconds=6)
        assert strftime(td, "%d %H:%M:%S.%f") == "02 03:04:05.000006"
        assert strftime(timedelta(days=21), "%W") == "03"
        assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H%M%S") == "235959"

    def test_strftime_duration_year_month(self):
        d = Duration(years=12, months=3, days=4)
        assert strftime(d, "%Y-%m-%d") == "0012-03-04"
        assert strftime(d, "%%") == "%"

    def test_alternative_format(self):
        d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
        assert duration_isoformat(d, D_ALT_EXT) == "P0001-02-03T04:05:06"
        assert duration_isoformat(d, D_ALT_BAS) == "P00010203T040506"
        assert parse_duration(duration_isoformat(d, D_ALT_EXT)) == d

    def test_unknown_codes_preserved(self):
        assert strftime(timedelta(1), "%q%d") == "%q01"


# ---------------------------------------------------------------------------
# Duration class
# ---------------------------------------------------------------------------


class TestDurationClass:
    def test_constructor(self):
        d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6,
                     weeks=7, months=8, years=9)
        assert d.years == Decimal(9)
        assert d.months == Decimal(8)
        assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)

    def test_float_months_converted_to_decimal_via_str(self):
        d = Duration(months=0.1, years=1.1)
        assert d.months == Decimal("0.1")
        assert d.years == Decimal("1.1")

    def test_attribute_passthrough(self):
        d = Duration(days=2, seconds=30, microseconds=7)
        assert d.days == 2
        assert d.seconds == 30
        assert d.microseconds == 7
        assert d.total_seconds() == timedelta(days=2, seconds=30, microseconds=7).total_seconds()

    def test_missing_attribute(self):
        with pytest.raises(AttributeError):
            Duration().nonexistent

    def test_str(self):
        assert str(Duration(years=1, months=1, days=1)) == "1 years, 1 month, 1 day, 0:00:00"
        assert str(Duration(months=2)) == "2 months, 0:00:00"
        assert str(Duration(hours=1)) == "1:00:00"
        assert str(Duration(years=2, hours=3)) == "2 years, 3:00:00"

    def test_repr(self):
        assert repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5)) == (
            "isodate.duration.Duration(1, 2, 3, years=4, months=5)"
        )

    def test_hash(self):
        a = Duration(years=1, months=2, days=3)
        b = Duration(years=1, months=2, days=3)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert hash(a) != hash(Duration(years=1, months=2, days=4))

    def test_pickle(self):
        d = Duration(years=1, months=2, days=3, seconds=4)
        restored = pickle.loads(pickle.dumps(d))
        assert restored == d
        assert restored.years == d.years and restored.months == d.months
        assert restored.tdelta == d.tdelta

    def test_neg(self):
        d = -Duration(years=1, months=2, days=3, hours=4)
        assert d.years == -1
        assert d.months == -2
        assert d.tdelta == -timedelta(days=3, hours=4)
        assert -(-d) == d

    # equality -----------------------------------------------------------

    def test_equality(self):
        assert Duration(years=1) == Duration(years=1)
        assert Duration(years=1) == Duration(months=12)
        assert Duration(years=1, months=-12) == Duration()
        assert Duration(days=1) == Duration(hours=24)
        assert not (Duration(years=1) == Duration(years=2))
        assert not (Duration(years=1, days=1) == Duration(years=1))

    def test_inequality(self):
        assert Duration(years=1) != Duration(years=2)
        assert Duration(years=1, days=1) != Duration(years=1)
        assert not (Duration(years=1) != Duration(months=12))
        assert not (Duration(days=1) != Duration(days=1))

    def test_equality_with_timedelta(self):
        assert Duration(days=1) == timedelta(days=1)
        assert timedelta(days=1) == Duration(days=1)
        assert Duration(days=1) != timedelta(days=2)
        assert not (Duration(days=1) != timedelta(days=1))
        assert Duration(years=1) != timedelta(days=365)
        assert not (Duration(years=1) == timedelta(days=365))
        assert Duration(months=1) != timedelta(0)

    def test_equality_with_other_types(self):
        assert Duration(years=1) != 1
        assert not (Duration(years=1) == "P1Y")

    # arithmetic ---------------------------------------------------------

    def test_add_durations(self):
        result = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, hours=6)
        assert isinstance(result, Duration)
        assert result.years == 5
        assert result.months == 7
        assert result.tdelta == timedelta(days=3, hours=6)

    def test_add_timedelta(self):
        result = Duration(years=1, days=1) + timedelta(hours=1)
        assert isinstance(result, Duration)
        assert result.years == 1
        assert result.tdelta == timedelta(days=1, hours=1)
        result = timedelta(hours=1) + Duration(years=1, days=1)
        assert isinstance(result, Duration)
        assert result.years == 1
        assert result.tdelta == timedelta(days=1, hours=1)

    def test_add_does_not_mutate(self):
        a = Duration(years=1, days=1)
        b = Duration(months=1, days=1)
        a + b
        a + timedelta(1)
        assert a == Duration(years=1, days=1)
        assert b == Duration(months=1, days=1)

    def test_add_unsupported(self):
        with pytest.raises(TypeError):
            Duration(days=1) + 1
        with pytest.raises(TypeError):
            "x" + Duration(days=1)

    def test_sub_durations(self):
        result = Duration(years=3, months=5, days=4) - Duration(years=1, months=7, days=1)
        assert isinstance(result, Duration)
        assert result.years == 2
        assert result.months == -2
        assert result.tdelta == timedelta(days=3)

    def test_sub_timedelta(self):
        result = Duration(years=1, days=3) - timedelta(days=1)
        assert isinstance(result, Duration)
        assert result.years == 1
        assert result.tdelta == timedelta(days=2)

    def test_rsub_timedelta(self):
        result = timedelta(days=3) - Duration(years=1, days=1)
        assert isinstance(result, Duration)
        assert result.years == -1
        assert result.months == 0
        assert result.tdelta == timedelta(days=2)

    def test_sub_unsupported(self):
        with pytest.raises(TypeError):
            Duration(days=1) - 1
        with pytest.raises(TypeError):
            1 - Duration(days=1)
        with pytest.raises(TypeError):
            Duration(days=1) - date(2000, 1, 1)

    def test_mul(self):
        d = Duration(years=1, months=2, days=3)
        for result in (d * 3, 3 * d):
            assert isinstance(result, Duration)
            assert result.years == 3
            assert result.months == 6
            assert result.tdelta == timedelta(days=9)
        assert d * 0 == Duration()
        neg = d * -1
        assert neg == -d

    def test_mul_unsupported(self):
        with pytest.raises(TypeError):
            Duration(years=1) * 1.5
        with pytest.raises(TypeError):
            Duration(years=1) * Duration(years=1)

    # date arithmetic ----------------------------------------------------

    @pytest.mark.parametrize(
        "start, dur, expected",
        [
            (date(2000, 1, 31), Duration(months=1), date(2000, 2, 29)),
            (date(2001, 1, 31), Duration(months=1), date(2001, 2, 28)),
            (date(2000, 2, 29), Duration(years=1), date(2001, 2, 28)),
            (date(2000, 2, 29), Duration(years=4), date(2004, 2, 29)),
            (date(2000, 3, 31), Duration(months=1), date(2000, 4, 30)),
            (date(2000, 12, 15), Duration(months=1), date(2001, 1, 15)),
            (date(2000, 11, 15), Duration(months=1), date(2000, 12, 15)),
            (date(2000, 12, 15), Duration(months=12), date(2001, 12, 15)),
            (date(2000, 12, 15), Duration(months=13), date(2002, 1, 15)),
            (date(2000, 1, 15), Duration(months=-1), date(1999, 12, 15)),
            (date(2000, 3, 31), Duration(months=-1), date(2000, 2, 29)),
            (date(2000, 1, 1), Duration(years=1, months=2, days=3), date(2001, 3, 4)),
            (date(2000, 1, 31), Duration(months=1, days=1), date(2000, 3, 1)),
            (date(2000, 5, 5), Duration(), date(2000, 5, 5)),
            (date(2000, 5, 5), Duration(years=-1, months=-5), date(1998, 12, 5)),
        ],
    )
    def test_add_to_date(self, start, dur, expected):
        assert start + dur == expected
        assert dur + start == expected

    def test_add_to_datetime_keeps_time_and_tz(self):
        start = datetime(2000, 1, 31, 10, 30, tzinfo=FixedOffset(2))
        result = start + Duration(months=1, hours=2)
        assert result == datetime(2000, 2, 29, 12, 30, tzinfo=FixedOffset(2))
        assert isinstance(result, datetime)
        assert result.utcoffset() == timedelta(hours=2)

    def test_add_hours_rolls_day(self):
        assert datetime(2000, 2, 28, 23) + Duration(years=1, hours=2) == datetime(2001, 3, 1, 1)

    @pytest.mark.parametrize(
        "start, dur, expected",
        [
            (date(2000, 3, 31), Duration(months=1), date(2000, 2, 29)),
            (date(2001, 3, 31), Duration(months=1), date(2001, 2, 28)),
            (date(2001, 1, 15), Duration(months=1), date(2000, 12, 15)),
            (date(2001, 1, 15), Duration(months=13), date(1999, 12, 15)),
            (date(2000, 12, 15), Duration(months=-1), date(2001, 1, 15)),
            (date(2004, 2, 29), Duration(years=1), date(2003, 2, 28)),
            (date(2001, 3, 4), Duration(years=1, months=2, days=3), date(2000, 1, 1)),
            (date(2000, 3, 1), Duration(months=1, days=1), date(2000, 1, 31)),
        ],
    )
    def test_subtract_from_date(self, start, dur, expected):
        assert start - dur == expected

    def test_subtract_from_datetime(self):
        result = datetime(2000, 3, 31, 1, 0) - Duration(months=1, hours=2)
        assert result == datetime(2000, 2, 28, 23, 0)

    def test_date_minus_parsed_duration(self):
        assert date(2000, 3, 1) - parse_duration("P1M") == date(2000, 2, 1)
        assert date(2000, 3, 1) + parse_duration("-P1M") == date(2000, 2, 1)

    def test_fractional_rejected_for_dates(self):
        with pytest.raises(ValueError):
            date(2000, 1, 1) + Duration(months=Decimal("1.5"))
        with pytest.raises(ValueError):
            date(2000, 1, 1) + Duration(years=Decimal("0.5"))
        with pytest.raises(ValueError):
            date(2000, 1, 1) - Duration(months=Decimal("1.5"))

    def test_integral_decimal_ok_for_dates(self):
        assert date(2000, 1, 1) + Duration(years=Decimal("2.0")) == date(2002, 1, 1)

    # totimedelta --------------------------------------------------------

    def test_totimedelta_start(self):
        assert Duration(months=1).totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
        assert Duration(months=1).totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
        assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)
        assert Duration(years=1).totimedelta(start=date(2001, 1, 1)) == timedelta(days=365)

    def test_totimedelta_end(self):
        assert Duration(months=1).totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
        assert Duration(months=1).totimedelta(end=date(2001, 3, 1)) == timedelta(days=28)

    def test_totimedelta_with_time(self):
        d = Duration(months=1, hours=1)
        assert d.totimedelta(start=datetime(2000, 2, 1)) == timedelta(days=29, hours=1)

    def test_totimedelta_requires_one(self):
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta()
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 1))


# ---------------------------------------------------------------------------
# package exports
# ---------------------------------------------------------------------------


def test_all_exports_exist():
    for name in isodate.__all__:
        assert hasattr(isodate, name), name


def test_format_constants():
    assert isodate.D_DEFAULT == "P%P"
    assert D_WEEK == "P%p"
    assert DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
    assert isodate.D_ALT_EXT_ORD == "P%Y-%jT%H:%M:%S"
    assert isodate.D_ALT_BAS_ORD == "P%Y%jT%H%M%S"
