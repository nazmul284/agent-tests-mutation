import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
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


# ---------------------------------------------------------------- helpers
class TestHelpers:
    @pytest.mark.parametrize(
        "year,month,expected",
        [
            (2001, 1, 31), (2001, 2, 28), (2001, 3, 31), (2001, 4, 30),
            (2001, 5, 31), (2001, 6, 30), (2001, 7, 31), (2001, 8, 31),
            (2001, 9, 30), (2001, 10, 31), (2001, 11, 30), (2001, 12, 31),
            (2004, 2, 29), (1900, 2, 28), (2000, 2, 29), (2100, 2, 28),
            (1600, 2, 29), (1996, 2, 29),
        ],
    )
    def test_max_days(self, year, month, expected):
        assert max_days_in_month(year, month) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            (5, (0, 5)), (13, (1, 1)), (0, (-1, 12)), (12, (0, 12)),
            (1, (0, 1)), (-11, (-1, 1)), (25, (2, 1)),
            (-12, (-2, 12)),
        ],
    )
    def test_fquotmod(self, val, expected):
        div, mod = fquotmod(Decimal(val), 1, 13)
        assert (div, mod) == expected


# ---------------------------------------------------------------- dates
class TestParseDate:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("19850412", date(1985, 4, 12)),
            ("1985-04-12", date(1985, 4, 12)),
            ("1985102", date(1985, 4, 12)),
            ("1985-102", date(1985, 4, 12)),
            ("1985W155", date(1985, 4, 12)),
            ("1985-W15-5", date(1985, 4, 12)),
            ("1985W15", date(1985, 4, 8)),
            ("1985-W15", date(1985, 4, 8)),
            ("198504", date(1985, 4, 1)),
            ("1985-04", date(1985, 4, 1)),
            ("1985", date(1985, 1, 1)),
            ("19", date(1901, 1, 1)),
            ("2004-W53-7", date(2005, 1, 2)),
            ("2008-W01-1", date(2007, 12, 31)),
            ("2009-W01-1", date(2008, 12, 29)),
            ("2010-W01-1", date(2010, 1, 4)),
            ("2004-W01-1", date(2003, 12, 29)),
            ("2005-W01-1", date(2005, 1, 3)),
            ("2006-W01-1", date(2006, 1, 2)),
            ("2007-W01-1", date(2007, 1, 1)),
            ("2004-W10-3", date(2004, 3, 3)),
            ("2004-366", date(2004, 12, 31)),
            ("2004-001", date(2004, 1, 1)),
            ("2003-365", date(2003, 12, 31)),
            ("2004-02-29", date(2004, 2, 29)),
            ("0001-01-01", date(1, 1, 1)),
            ("9999-12-31", date(9999, 12, 31)),
        ],
    )
    def test_formats(self, s, expected):
        assert parse_date(s) == expected

    def test_defaults_for_incomplete(self):
        assert parse_date("1985", defaultmonth=6, defaultday=15) == date(1985, 6, 15)
        assert parse_date("1985-04", defaultday=9) == date(1985, 4, 9)
        assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
        assert parse_date("198504", defaultday=9) == date(1985, 4, 9)

    def test_expanded(self):
        assert parse_date("+0019850412", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985-04-12", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985-102", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+001985W155", yeardigits=6) == date(1985, 4, 12)
        assert parse_date("+0019", yeardigits=6) == date(1901, 1, 1)
        assert parse_date("+001985", yeardigits=6) == date(1985, 1, 1)
        assert parse_date("+1985-04-12", expanded=True) == date(1985, 4, 12)

    def test_expanded_requires_sign(self):
        with pytest.raises(ISO8601Error):
            parse_date("001985-04-12", yeardigits=6)
        with pytest.raises(ISO8601Error):
            parse_date("1985-04-12", expanded=True)
        with pytest.raises(ISO8601Error):
            parse_date("+1985-04-12")

    def test_negative_year_unrepresentable(self):
        with pytest.raises(ValueError):
            parse_date("-1985-04-12", expanded=True)

    @pytest.mark.parametrize(
        "s",
        ["", "abc", "1985-4-12", "1985/04/12", "198", "19850", "1985-04-1",
         "1985-W1", "1985-0412", "198504-12", "1985-04-12 ", " 1985-04-12",
         "1985-04-12\n", "1985-W15-55", "1985-0", "12345"],
    )
    def test_invalid_format(self, s):
        with pytest.raises(ISO8601Error):
            parse_date(s)

    @pytest.mark.parametrize("s", ["1985-13-01", "1985-02-30", "2003-02-29",
                                   "1985-00-10", "1985-04-00", "1985-04-31",
                                   "0000-01-01"])
    def test_invalid_values(self, s):
        # month 00 falls back to default month only for month-only forms
        if s == "1985-00-10":
            assert parse_date(s) == date(1985, 1, 10)
            return
        with pytest.raises(ValueError):
            parse_date(s)

    def test_error_is_valueerror(self):
        assert issubclass(ISO8601Error, ValueError)

    def test_iso_error_message_contains_input(self):
        with pytest.raises(ISO8601Error, match="xyz"):
            parse_date("xyz")


class TestDateFormat:
    D = date(1985, 4, 12)

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.DATE_BAS_COMPLETE, "19850412"),
            (isodate.DATE_EXT_COMPLETE, "1985-04-12"),
            (isodate.DATE_BAS_ORD_COMPLETE, "1985102"),
            (isodate.DATE_EXT_ORD_COMPLETE, "1985-102"),
            (isodate.DATE_BAS_WEEK_COMPLETE, "1985W155"),
            (isodate.DATE_EXT_WEEK_COMPLETE, "1985-W15-5"),
            (isodate.DATE_BAS_WEEK, "1985W15"),
            (isodate.DATE_EXT_WEEK, "1985-W15"),
            (isodate.DATE_BAS_MONTH, "198504"),
            (isodate.DATE_EXT_MONTH, "1985-04"),
            (isodate.DATE_YEAR, "1985"),
            (isodate.DATE_CENTURY, "19"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert date_isoformat(self.D, fmt) == expected

    def test_default(self):
        assert date_isoformat(self.D) == "1985-04-12"

    def test_padding(self):
        assert date_isoformat(date(5, 1, 2)) == "0005-01-02"
        assert date_isoformat(date(1985, 1, 2), "%j") == "002"
        assert date_isoformat(date(2004, 12, 31), "%j") == "366"
        assert date_isoformat(date(2004, 3, 1), "%j") == "061"

    def test_week_year_boundary_weeks(self):
        assert strftime(date(2005, 1, 2), "%W") == "53"
        assert strftime(date(2007, 12, 31), "%W") == "01"

    def test_weekday(self):
        assert strftime(date(2024, 1, 1), "%w") == "1"
        assert strftime(date(2024, 1, 7), "%w") == "7"

    def test_expanded_year(self):
        assert date_isoformat(self.D, "%Y", 6) == "+001985"
        assert date_isoformat(self.D, "%C", 6) == "+0019"
        assert date_isoformat(self.D, "%Y", 4) == "1985"

    def test_percent_literal_and_unknown(self):
        assert strftime(self.D, "%%Y") == "%Y"
        assert strftime(self.D, "100%% %d") == "100% 12"
        assert strftime(self.D, "%q") == "%q"

    def test_century(self):
        assert strftime(date(1901, 1, 1), "%C") == "19"
        assert strftime(date(1900, 1, 1), "%C") == "19"
        assert strftime(date(2000, 1, 1), "%C") == "20"

    @pytest.mark.parametrize(
        "d",
        [date(1985, 4, 12), date(2004, 12, 31), date(2005, 1, 2), date(2007, 12, 31),
         date(2010, 1, 3), date(2020, 2, 29), date(1, 1, 1), date(9999, 12, 31),
         date(2008, 12, 29), date(2026, 1, 1)],
    )
    @pytest.mark.parametrize(
        "fmt",
        [isodate.DATE_BAS_COMPLETE, isodate.DATE_EXT_COMPLETE,
         isodate.DATE_BAS_ORD_COMPLETE, isodate.DATE_EXT_ORD_COMPLETE],
    )
    def test_roundtrip(self, d, fmt):
        assert parse_date(date_isoformat(d, fmt)) == d

    @pytest.mark.parametrize("d", [date(2004, 12, 31), date(2008, 12, 29), date(2010, 1, 3),
                                   date(2005, 1, 1), date(2016, 1, 3)])
    def test_week_roundtrip(self, d):
        # ISO week-year differs from calendar year near boundaries; only
        # check dates whose calendar year == ISO year.
        if d.isocalendar()[0] != d.year:
            pytest.skip("iso year differs")
        for fmt in (isodate.DATE_EXT_WEEK_COMPLETE, isodate.DATE_BAS_WEEK_COMPLETE):
            assert parse_date(date_isoformat(d, fmt)) == d


# ---------------------------------------------------------------- tz
class TestTz:
    def test_utc(self):
        assert parse_tzinfo("Z") is UTC
        assert UTC.utcoffset(None) == timedelta(0)
        assert UTC.dst(None) == timedelta(0)
        assert UTC.tzname(None) == "UTC"

    @pytest.mark.parametrize(
        "s,minutes",
        [("+01:00", 60), ("+0100", 60), ("+01", 60), ("-05:30", -330),
         ("-0530", -330), ("-05", -300), ("+14:00", 840), ("+00:30", 30),
         ("-00:30", -30), ("+23:59", 23 * 60 + 59)],
    )
    def test_parse_offset(self, s, minutes):
        tz = parse_tzinfo(s)
        assert tz.utcoffset(None) == timedelta(minutes=minutes)
        assert tz.dst(None) == timedelta(0)

    def test_parse_empty_is_none(self):
        assert parse_tzinfo("") is None

    def test_parse_name(self):
        assert parse_tzinfo("+05:30").tzname(None) == "+05:30"

    def test_unmatched_is_none(self):
        assert parse_tzinfo("abc") is None

    def test_isoformat(self):
        dt = datetime(2020, 1, 1, tzinfo=FixedOffset(5, 30, "x"))
        assert tz_isoformat(dt) == "+05:30"
        assert tz_isoformat(dt, "%Z") == "+05:30"
        assert tz_isoformat(dt, "%z") == "+0530"
        assert tz_isoformat(dt, "%h") == "+05"
        dt = datetime(2020, 1, 1, tzinfo=FixedOffset(-5, -30, "x"))
        assert tz_isoformat(dt, "%Z") == "-05:30"
        assert tz_isoformat(dt, "%z") == "-0530"
        assert tz_isoformat(dt, "%h") == "-05"

    def test_isoformat_utc_naive(self):
        assert tz_isoformat(datetime(2020, 1, 1, tzinfo=UTC)) == "Z"
        assert tz_isoformat(datetime(2020, 1, 1, tzinfo=FixedOffset(0, 0, "UTC"))) == "Z"
        assert tz_isoformat(datetime(2020, 1, 1)) == ""

    def test_isoformat_bad_format(self):
        with pytest.raises(ValueError):
            tz_isoformat(datetime(2020, 1, 1, tzinfo=FixedOffset(1, 0, "x")), "%q")

    def test_isoformat_overflow(self):
        tz = FixedOffset(100, 0, "big")
        with pytest.raises(OverflowError):
            tz_isoformat(datetime(2020, 1, 1, tzinfo=tz))
        tz = FixedOffset(99, 59, "big")
        assert tz_isoformat(datetime(2020, 1, 1, tzinfo=tz)) == "+99:59"

    def test_fixed_offset_defaults(self):
        tz = FixedOffset()
        assert tz.utcoffset(None) == timedelta(0)
        assert tz.tzname(None) == "UTC"
        assert "UTC" in repr(tz)

    def test_fixed_offset_applied(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=FixedOffset(2, 0, "x"))
        assert dt.utcoffset() == timedelta(hours=2)
        assert dt.astimezone(UTC).hour == 10

    def test_utc_pickle_identity(self):
        assert pickle.loads(pickle.dumps(UTC)) is UTC

    def test_local_timezone(self):
        from isodate.tzinfo import LOCAL
        dt = datetime(2020, 1, 1, 12, tzinfo=LOCAL)
        assert isinstance(dt.utcoffset(), timedelta)
        assert isinstance(dt.tzname(), str)
        with pytest.raises(Exception):
            LOCAL.utcoffset(None)


# ---------------------------------------------------------------- time
class TestParseTime:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("232050", time(23, 20, 50)),
            ("23:20:50", time(23, 20, 50)),
            ("2320", time(23, 20)),
            ("23:20", time(23, 20)),
            ("23", time(23)),
            ("T232050", time(23, 20, 50)),
            ("T23:20:50", time(23, 20, 50)),
            ("T23", time(23)),
            ("00:00:00", time(0, 0, 0)),
            ("235959", time(23, 59, 59)),
            ("23:20:50.5", time(23, 20, 50, 500000)),
            ("23:20:50,5", time(23, 20, 50, 500000)),
            ("232050.123456", time(23, 20, 50, 123456)),
            ("23:20:50.000001", time(23, 20, 50, 1)),
            ("23:20:50.1234567", time(23, 20, 50, 123456)),
            ("23:20:50.9999999", time(23, 20, 50, 999999)),
            ("23:20.5", time(23, 20, 30)),
            ("2320.25", time(23, 20, 15)),
            ("23.5", time(23, 30)),
            ("23.25", time(23, 15)),
            ("23,5", time(23, 30)),
            ("08.75", time(8, 45)),
        ],
    )
    def test_formats(self, s, expected):
        assert parse_time(s) == expected

    @pytest.mark.parametrize(
        "s,minutes",
        [("Z", 0), ("+01:00", 60), ("-0530", -330), ("+05", 300), ("-05:30", -330)],
    )
    def test_timezones(self, s, minutes):
        for base in ("23:20:50", "232050", "23:20", "2320", "23"):
            t = parse_time(base + s)
            assert t.utcoffset() == timedelta(minutes=minutes)
            assert (t.hour, t.minute) == (23, 20 if len(base) > 2 else 0)

    def test_z_is_utc(self):
        assert parse_time("10:00:00Z").tzinfo is UTC

    def test_naive(self):
        assert parse_time("10:00:00").tzinfo is None

    @pytest.mark.parametrize(
        "s",
        ["", "2", "232", "23:2", "23:20:5", "2:20:50", "23-20-50", "abc", "23:20:50 ",
         "23:20:50+", "23:20:50+1", "23:20:50ZZ", "23:20:50.", "t23:20:50", "23::50",
         "23:20:50+01:0", "232050.", "T"],
    )
    def test_invalid_format(self, s):
        with pytest.raises(ISO8601Error):
            parse_time(s)

    @pytest.mark.parametrize("s", ["24:00:00", "23:60:00", "23:20:60", "25", "23:61"])
    def test_invalid_values(self, s):
        with pytest.raises(ValueError):
            parse_time(s)


class TestTimeFormat:
    def test_default(self):
        assert time_isoformat(time(23, 20, 50)) == "23:20:50"
        assert time_isoformat(time(23, 20, 50, tzinfo=UTC)) == "23:20:50Z"
        assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.TIME_BAS_COMPLETE, "012003"),
            (isodate.TIME_EXT_COMPLETE, "01:20:03"),
            (isodate.TIME_BAS_MINUTE, "0120"),
            (isodate.TIME_EXT_MINUTE, "01:20"),
            (isodate.TIME_HOUR, "01"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert time_isoformat(time(1, 20, 3), fmt) == expected

    def test_microseconds(self):
        assert strftime(time(1, 2, 3, 45), "%f") == "000045"
        assert strftime(time(1, 2, 3, 456789), "%S.%f") == "03.456789"

    def test_tz_codes(self):
        t = time(1, 2, 3, tzinfo=FixedOffset(-3, -30, "x"))
        assert strftime(t, "%z") == "-0330"
        assert strftime(t, "%Z") == "-03:30"
        assert strftime(t, "%h") == "-03"

    @pytest.mark.parametrize("t", [time(0, 0, 0), time(23, 59, 59), time(12, 34, 56, 789012),
                                   time(1, 2, 3, tzinfo=UTC),
                                   time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x"))])
    def test_roundtrip(self, t):
        s = time_isoformat(t, "%H:%M:%S.%f%Z")
        assert parse_time(s) == t
        assert parse_time(s).utcoffset() == t.utcoffset()


# ---------------------------------------------------------------- datetime
class TestDatetime:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("19850412T101530", datetime(1985, 4, 12, 10, 15, 30)),
            ("1985-04-12T10:15:30", datetime(1985, 4, 12, 10, 15, 30)),
            ("1985-04-12T10:15:30Z", datetime(1985, 4, 12, 10, 15, 30, tzinfo=UTC)),
            ("1985-04-12T10:15:30.5Z", datetime(1985, 4, 12, 10, 15, 30, 500000, tzinfo=UTC)),
            ("1985-102T10:15", datetime(1985, 4, 12, 10, 15)),
            ("1985-W15-5T10", datetime(1985, 4, 12, 10)),
            ("1985-04-12T10:15:30+04:30",
             datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(4, 30, "+04:30"))),
            ("1985-04-12T10:15:30-0400",
             datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(-4, 0, "-0400"))),
        ],
    )
    def test_parse(self, s, expected):
        result = parse_datetime(s)
        assert result == expected
        assert result.utcoffset() == expected.utcoffset()

    @pytest.mark.parametrize(
        "s",
        ["1985-04-12", "1985-04-12 10:15:30", "", "T", "1985-04-12T10:15:30T",
         "1985-04-12T", "T10:15:30", "xxxx-04-12T10:15:30", "1985-04-12T25:00:00x"],
    )
    def test_invalid(self, s):
        with pytest.raises(ValueError):
            parse_datetime(s)

    def test_missing_T_is_iso_error(self):
        with pytest.raises(ISO8601Error, match="T"):
            parse_datetime("1985-04-12")

    def test_format(self):
        dt = datetime(1985, 4, 12, 10, 15, 30)
        assert datetime_isoformat(dt) == "1985-04-12T10:15:30"
        assert datetime_isoformat(dt.replace(tzinfo=UTC)) == "1985-04-12T10:15:30Z"
        assert (datetime_isoformat(dt.replace(tzinfo=FixedOffset(-4, 0, "x")))
                == "1985-04-12T10:15:30-04:00")

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.DT_EXT_COMPLETE, "1985-04-12T10:15:30+01:00"),
            (isodate.DT_BAS_COMPLETE, "19850412T101530+0100"),
            (isodate.DT_EXT_ORD_COMPLETE, "1985-102T10:15:30+01:00"),
            (isodate.DT_BAS_ORD_COMPLETE, "1985102T101530+0100"),
            (isodate.DT_EXT_WEEK_COMPLETE, "1985-W15-5T10:15:30+01:00"),
            (isodate.DT_BAS_WEEK_COMPLETE, "1985W155T101530+0100"),
        ],
    )
    def test_format_constants(self, fmt, expected):
        dt = datetime(1985, 4, 12, 10, 15, 30, tzinfo=FixedOffset(1, 0, "x"))
        assert datetime_isoformat(dt, fmt) == expected
        assert parse_datetime(expected) == dt

    def test_roundtrip_micro(self):
        dt = datetime(2020, 2, 29, 23, 59, 59, 999999, tzinfo=UTC)
        assert parse_datetime(datetime_isoformat(dt, "%Y-%m-%dT%H:%M:%S.%f%Z")) == dt


# ---------------------------------------------------------------- duration parse
class TestParseDuration:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("P1D", timedelta(days=1)),
            ("P2W", timedelta(weeks=2)),
            ("PT1H", timedelta(hours=1)),
            ("PT1M", timedelta(minutes=1)),
            ("PT1S", timedelta(seconds=1)),
            ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
            ("PT36H", timedelta(hours=36)),
            ("P0D", timedelta(0)),
            ("PT0S", timedelta(0)),
            ("PT1.5S", timedelta(seconds=1, milliseconds=500)),
            ("PT1,5S", timedelta(seconds=1.5)),
            ("P0.5D", timedelta(hours=12)),
            ("PT0.5H", timedelta(minutes=30)),
            ("P1W2D", timedelta(days=9)),
            ("PT0.000001S", timedelta(microseconds=1)),
            ("PT10M", timedelta(minutes=10)),
            ("P10D", timedelta(days=10)),
            ("-P1D", -timedelta(days=1)),
            ("-PT1H30M", -timedelta(hours=1, minutes=30)),
            ("+P1D", timedelta(days=1)),
            ("-P0D", timedelta(0)),
            ("P1DT", timedelta(days=1)),
        ],
    )
    def test_timedeltas(self, s, expected):
        result = parse_duration(s)
        assert isinstance(result, timedelta)
        assert result == expected

    @pytest.mark.parametrize(
        "s,years,months,td",
        [
            ("P1Y", 1, 0, timedelta(0)),
            ("P1M", 0, 1, timedelta(0)),
            ("P1Y2M", 1, 2, timedelta(0)),
            ("P1Y2M3DT4H5M6S", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
            ("P3Y6M4DT12H30M5S", 3, 6, timedelta(days=4, hours=12, minutes=30, seconds=5)),
            ("P1Y1W", 1, 0, timedelta(weeks=1)),
            ("P10Y", 10, 0, timedelta(0)),
            ("P12M", 0, 12, timedelta(0)),
            ("P1MT1M", 0, 1, timedelta(minutes=1)),
        ],
    )
    def test_durations(self, s, years, months, td):
        result = parse_duration(s)
        assert isinstance(result, Duration)
        assert result.years == years
        assert result.months == months
        assert result.tdelta == td

    def test_month_vs_minute(self):
        assert parse_duration("PT5M") == timedelta(minutes=5)
        assert parse_duration("P5M").months == 5

    def test_fractional_years_months(self):
        d = parse_duration("P1.5Y")
        assert d.years == Decimal("1.5")
        d = parse_duration("P0,5M")
        assert d.months == Decimal("0.5")

    def test_negative_duration(self):
        d = parse_duration("-P1Y2M3D")
        assert d.years == -1 and d.months == -2
        assert d.tdelta == -timedelta(days=3)
        d = parse_duration("-P1Y")
        assert d.years == -1 and d.tdelta == timedelta(0)

    def test_as_timedelta_if_possible_false(self):
        d = parse_duration("P1D", as_timedelta_if_possible=False)
        assert isinstance(d, Duration)
        assert d.tdelta == timedelta(days=1)
        assert d.years == 0 and d.months == 0
        d = parse_duration("-P1D", as_timedelta_if_possible=False)
        assert d.tdelta == -timedelta(days=1)

    def test_alternative_format(self):
        d = parse_duration("P0001-02-03T04:05:06")
        assert isinstance(d, Duration)
        assert (d.years, d.months) == (1, 2)
        assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
        d = parse_duration("P00010203T040506")
        assert (d.years, d.months) == (1, 2)
        assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    def test_alternative_format_ordinal(self):
        # ordinal date: year 1, day 100 -> month 4 day 10
        d = parse_duration("P0001-100T00:00:00")
        assert (d.years, d.months) == (1, 4)
        assert d.tdelta == timedelta(days=9) or d.tdelta == timedelta(days=10)

    @pytest.mark.parametrize(
        "s",
        ["", "P", "1D", "P1", "P1X", "PD", "P1D1Y", "PT1D", "P1H", "P-1D",
         "P1D ", " P1D", "P1.D", "P.5D", "P1S", "PT1Y", "PT1W", "p1d", "P1DT1D",
         "PTT1H", "P1Y2M3DT4H5M6SX", "--P1D"],
    )
    def test_invalid(self, s):
        with pytest.raises(ValueError):
            parse_duration(s)

    @pytest.mark.parametrize("bad", [None, 1, 1.5, b"P1D", ["P1D"]])
    def test_non_string(self, bad):
        with pytest.raises(TypeError):
            parse_duration(bad)


# ---------------------------------------------------------------- duration format
class TestDurationFormat:
    @pytest.mark.parametrize(
        "td,expected",
        [
            (timedelta(0), "P0D"),
            (timedelta(days=1), "P1D"),
            (timedelta(days=10), "P10D"),
            (timedelta(hours=1), "PT1H"),
            (timedelta(minutes=1), "PT1M"),
            (timedelta(seconds=1), "PT1S"),
            (timedelta(seconds=90), "PT1M30S"),
            (timedelta(hours=25), "P1DT1H"),
            (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
            (timedelta(microseconds=1), "PT0.000001S"),
            (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
            (timedelta(seconds=59, microseconds=100), "PT59.0001S"),
            (timedelta(hours=1, seconds=1), "PT1H1S"),
            (timedelta(days=1, seconds=1), "P1DT1S"),
            (timedelta(days=1, minutes=1), "P1DT1M"),
            (timedelta(hours=23, minutes=59, seconds=59), "PT23H59M59S"),
            (timedelta(days=400), "P400D"),
            (-timedelta(days=1), "-P1D"),
            (-timedelta(hours=1, minutes=30), "-PT1H30M"),
            (-timedelta(microseconds=1), "-PT0.000001S"),
            (-timedelta(seconds=1, microseconds=500000), "-PT1.5S"),
        ],
    )
    def test_timedelta(self, td, expected):
        assert duration_isoformat(td) == expected

    @pytest.mark.parametrize(
        "dur,expected",
        [
            (Duration(years=1), "P1Y"),
            (Duration(months=1), "P1M"),
            (Duration(years=1, months=2), "P1Y2M"),
            (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6),
             "P1Y2M3DT4H5M6S"),
            (Duration(years=2, hours=1), "P2YT1H"),
            (Duration(months=3, minutes=1), "P3MT1M"),
            (Duration(), "P0D"),
            (Duration(days=1), "P1D"),
            (Duration(years=-1), "-P1Y"),
            (Duration(months=-2), "-P2M"),
            (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
            (-Duration(years=1, hours=2), "-P1YT2H"),
        ],
    )
    def test_duration(self, dur, expected):
        assert duration_isoformat(dur) == expected

    def test_weeks_format(self):
        assert duration_isoformat(timedelta(days=14), "P%p") == "P2W"
        assert duration_isoformat(timedelta(days=13), "P%p") == "P1W"
        assert duration_isoformat(timedelta(days=6), isodate.D_WEEK) == "P0W"
        assert duration_isoformat(-timedelta(days=14), "P%p") == "-P2W"
        assert duration_isoformat(timedelta(days=7)) == "P7D"

    def test_alt_formats(self):
        d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
        assert duration_isoformat(d, isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
        assert duration_isoformat(d, isodate.D_ALT_BAS) == "P00010203T040506"

    def test_strftime_codes(self):
        td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
        assert strftime(td, "%d") == "03"
        assert strftime(td, "%H") == "04"
        assert strftime(td, "%M") == "05"
        assert strftime(td, "%S") == "06"
        assert strftime(td, "%f") == "000007"
        assert strftime(td, "%%") == "%"
        d = Duration(years=12, months=3)
        assert strftime(d, "%Y") == "0012"
        assert strftime(d, "%m") == "03"
        assert strftime(d, "%Y", 6) == "+000012"
        assert strftime(Duration(years=1900), "%C") == "19"
        assert strftime(Duration(years=1900), "%C", 6) == "+0019"

    def test_strftime_hours_over_day(self):
        assert strftime(timedelta(hours=50), "%d %H") == "02 02"
        assert strftime(timedelta(minutes=125), "%H:%M") == "02:05"
        assert strftime(timedelta(seconds=3661), "%H:%M:%S") == "01:01:01"

    @pytest.mark.parametrize(
        "s",
        ["P1D", "PT1H", "PT1M", "PT1S", "P1DT2H3M4S", "PT0.000001S", "PT1.5S", "-P1D",
         "P0D", "P400D", "-PT1H30M", "PT23H59M59.999999S", "P1Y", "P1M", "P1Y2M3DT4H5M6S",
         "-P1Y2M", "P2YT1H", "P11M30DT23H"],
    )
    def test_roundtrip(self, s):
        assert duration_isoformat(parse_duration(s)) == s


# ---------------------------------------------------------------- Duration class
class TestDuration:
    def test_init_and_attrs(self):
        d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5,
                     hours=6, weeks=7, months=8, years=9)
        assert d.years == 9 and d.months == 8
        assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
        assert isinstance(d.years, Decimal) and isinstance(d.months, Decimal)

    def test_float_months_exact(self):
        assert Duration(months=0.1).months == Decimal("0.1")
        assert Duration(years=1.1).years == Decimal("1.1")

    def test_getattr_delegates(self):
        d = Duration(days=2, seconds=3, microseconds=4)
        assert (d.days, d.seconds, d.microseconds) == (2, 3, 4)
        assert d.total_seconds() == timedelta(2, 3, 4).total_seconds()
        with pytest.raises(AttributeError):
            d.nonexistent

    def test_eq_ne(self):
        assert Duration(years=1) == Duration(months=12)
        assert Duration(years=1, months=1) == Duration(months=13)
        assert Duration(years=1) != Duration(years=2)
        assert Duration(months=1) != Duration(months=2)
        assert Duration(days=1) == Duration(days=1)
        assert Duration(days=1) != Duration(days=2)
        assert Duration(years=1, days=1) != Duration(years=1)
        assert not (Duration(years=1) != Duration(months=12))
        assert not (Duration(years=1) == Duration(years=2))

    def test_eq_timedelta(self):
        assert Duration(days=1) == timedelta(days=1)
        assert Duration(days=1) != timedelta(days=2)
        assert not (Duration(days=1) != timedelta(days=1))
        assert Duration(years=1) != timedelta(days=365)
        assert not (Duration(months=1) == timedelta(0))
        assert Duration(months=1) != timedelta(0)

    def test_hash(self):
        assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
        assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2

    def test_neg(self):
        d = -Duration(years=1, months=2, days=3, seconds=4)
        assert d.years == -1 and d.months == -2
        assert d.tdelta == -timedelta(days=3, seconds=4)
        assert -(-Duration(years=1, days=1)) == Duration(years=1, days=1)

    def test_add_duration(self):
        r = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
        assert isinstance(r, Duration)
        assert (r.years, r.months) == (5, 7)
        assert r.tdelta == timedelta(days=9)

    def test_add_timedelta(self):
        r = Duration(years=1, days=3) + timedelta(days=2)
        assert isinstance(r, Duration)
        assert r.years == 1 and r.tdelta == timedelta(days=5)
        r = timedelta(days=2) + Duration(months=1, days=3)
        assert isinstance(r, Duration)
        assert r.months == 1 and r.tdelta == timedelta(days=5)

    def test_add_not_implemented(self):
        with pytest.raises(TypeError):
            Duration(days=1) + 1
        with pytest.raises(TypeError):
            Duration(days=1) + "x"

    def test_add_date(self):
        assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
        assert date(2001, 1, 31) + Duration(months=1) == date(2001, 2, 28)
        assert Duration(months=1) + date(2001, 1, 31) == date(2001, 2, 28)
        assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
        assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)
        assert date(2000, 11, 15) + Duration(months=2) == date(2001, 1, 15)
        assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
        assert date(2000, 1, 15) + Duration(months=11) == date(2000, 12, 15)
        assert date(2000, 1, 15) + Duration(months=12) == date(2001, 1, 15)
        assert date(2000, 1, 15) + Duration(months=24) == date(2002, 1, 15)
        assert date(2000, 5, 31) + Duration(months=1) == date(2000, 6, 30)
        assert date(2000, 1, 1) + Duration(years=1, months=1, days=1) == date(2001, 2, 2)
        assert date(2000, 1, 31) + Duration(months=0) == date(2000, 1, 31)
        assert date(2000, 3, 31) + Duration(months=-1) == date(2000, 2, 29)
        assert date(2000, 1, 15) + Duration(months=-1) == date(1999, 12, 15)
        assert date(2000, 1, 15) + Duration(months=-12) == date(1999, 1, 15)
        assert date(2000, 1, 15) + Duration(months=-13) == date(1998, 12, 15)

    def test_add_datetime(self):
        dt = datetime(2000, 1, 31, 10, 20, 30, tzinfo=UTC)
        r = dt + Duration(months=1, hours=2)
        assert r == datetime(2000, 2, 29, 12, 20, 30, tzinfo=UTC)
        assert isinstance(r, datetime)
        assert r.tzinfo is UTC
        assert datetime(2000, 1, 1) + Duration(days=1.5) == datetime(2000, 1, 2, 12)

    def test_add_date_with_timedelta_part(self):
        assert date(2000, 1, 1) + Duration(months=1, days=30) == date(2000, 3, 2)

    def test_fractional_date_calculation_raises(self):
        with pytest.raises(ValueError):
            date(2000, 1, 1) + Duration(months=Decimal("1.5"))
        with pytest.raises(ValueError):
            date(2000, 1, 1) + Duration(years=Decimal("0.5"))
        with pytest.raises(ValueError):
            date(2000, 1, 1) - Duration(months=Decimal("1.5"))
        with pytest.raises(ValueError):
            date(2000, 1, 1) - Duration(years=Decimal("0.5"))

    def test_integral_decimal_fraction_ok(self):
        assert date(2000, 1, 1) + Duration(months=Decimal("2.0")) == date(2000, 3, 1)

    def test_sub_duration(self):
        r = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
        assert isinstance(r, Duration)
        assert (r.years, r.months) == (4, 5)
        assert r.tdelta == timedelta(days=6)

    def test_sub_timedelta(self):
        r = Duration(years=1, days=5) - timedelta(days=2)
        assert isinstance(r, Duration)
        assert r.years == 1 and r.tdelta == timedelta(days=3)
        r = timedelta(days=5) - Duration(months=1, days=2)
        assert isinstance(r, Duration)
        assert r.months == -1 and r.tdelta == timedelta(days=3)

    def test_sub_invalid(self):
        with pytest.raises(TypeError):
            Duration(days=1) - 1
        with pytest.raises(TypeError):
            1 - Duration(days=1)

    def test_rsub_date(self):
        assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
        assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
        assert date(2000, 1, 15) - Duration(months=12) == date(1999, 1, 15)
        assert date(2000, 1, 15) - Duration(months=13) == date(1998, 12, 15)
        assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
        assert date(2000, 3, 1) - Duration(years=1, months=1, days=1) == date(1999, 1, 31)
        assert date(2000, 12, 15) - Duration(months=-1) == date(2001, 1, 15)
        assert date(2000, 6, 15) - Duration(months=5) == date(2000, 1, 15)
        assert date(2000, 6, 15) - Duration(months=6) == date(1999, 12, 15)

    def test_rsub_datetime(self):
        dt = datetime(2000, 3, 31, 10, 0, 0)
        assert dt - Duration(months=1, hours=1) == datetime(2000, 2, 29, 9)

    def test_mul(self):
        r = Duration(years=1, months=2, days=3) * 3
        assert (r.years, r.months) == (3, 6)
        assert r.tdelta == timedelta(days=9)
        r = 2 * Duration(years=1, months=2, days=3)
        assert (r.years, r.months) == (2, 4)
        assert r.tdelta == timedelta(days=6)
        assert Duration(years=1) * 0 == Duration()
        assert (Duration(years=1) * -1).years == -1

    def test_mul_non_int(self):
        with pytest.raises(TypeError):
            Duration(days=1) * 1.5
        with pytest.raises(TypeError):
            Duration(days=1) * "a"

    def test_totimedelta(self):
        d = Duration(months=1)
        assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
        assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
        assert d.totimedelta(start=date(2001, 2, 1)) == timedelta(days=28)
        assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
        assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=31)
        assert Duration(years=1).totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=366)
        assert Duration(years=1, days=1).totimedelta(end=date(2001, 1, 1)) == timedelta(days=367)

    def test_totimedelta_args(self):
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta()
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))

    def test_str(self):
        assert str(Duration(days=1)) == "1 day, 0:00:00"
        assert str(Duration(years=1)) == "1 years, 0:00:00"
        assert str(Duration(months=1)) == "1 month, 0:00:00"
        assert str(Duration(months=2)) == "2 months, 0:00:00"
        assert str(Duration(years=2, months=3, hours=1)) == "2 years, 3 months, 1:00:00"
        assert str(Duration(months=Decimal("0.5"))) == "0.5 month, 0:00:00"

    def test_repr(self):
        r = repr(Duration(years=1, months=2, days=3, seconds=4, microseconds=5))
        assert r == "isodate.duration.Duration(3, 4, 5, years=1, months=2)"

    def test_pickle(self):
        d = Duration(years=1, months=2, days=3, seconds=4)
        d2 = pickle.loads(pickle.dumps(d))
        assert d2 == d
        assert (d2.years, d2.months, d2.tdelta) == (1, 2, d.tdelta)

    def test_equal_roundtrip_via_parse(self):
        assert parse_duration("P1Y2M3DT4H") == Duration(years=1, months=2, days=3, hours=4)
        assert parse_duration("P1Y") == Duration(months=12)


# ---------------------------------------------------------------- package
class TestPackage:
    def test_exports(self):
        for name in ("parse_date", "parse_time", "parse_datetime", "parse_duration",
                     "parse_tzinfo", "Duration", "UTC", "LOCAL", "FixedOffset", "strftime",
                     "date_isoformat", "time_isoformat", "datetime_isoformat",
                     "duration_isoformat", "tz_isoformat", "ISO8601Error", "D_DEFAULT",
                     "D_WEEK", "D_ALT_EXT", "D_ALT_BAS", "DT_EXT_COMPLETE", "TZ_EXT",
                     "TZ_BAS", "TZ_HOUR", "DATE_EXT_COMPLETE", "TIME_EXT_COMPLETE"):
            assert hasattr(isodate, name), name

    def test_format_constant_values(self):
        assert isodate.D_DEFAULT == "P%P"
        assert isodate.D_WEEK == "P%p"
        assert isodate.TZ_BAS == "%z"
        assert isodate.TZ_EXT == "%Z"
        assert isodate.TZ_HOUR == "%h"
        assert isodate.DATE_EXT_COMPLETE == "%Y-%m-%d"
        assert isodate.TIME_EXT_COMPLETE == "%H:%M:%S"
        assert isodate.DATE_BAS_COMPLETE == "%Y%m%d"
        assert isodate.TIME_BAS_COMPLETE == "%H%M%S"
        assert isodate.DATE_CENTURY == "%C"
