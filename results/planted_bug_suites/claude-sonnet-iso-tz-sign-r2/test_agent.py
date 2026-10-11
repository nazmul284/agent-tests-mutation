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
from isodate.isotzinfo import build_tzinfo


def offset(tz, dt=None):
    return tz.utcoffset(dt)


# ---------------------------------------------------------------- time zones


class TestTzParsing:
    def test_z_is_utc(self):
        assert parse_tzinfo("Z") is UTC

    def test_empty_is_none(self):
        assert parse_tzinfo("") is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("+01:00", timedelta(hours=1)),
            ("-01:00", timedelta(hours=-1)),
            ("+0530", timedelta(hours=5, minutes=30)),
            ("-0530", timedelta(hours=-5, minutes=-30)),
            ("+05:30", timedelta(hours=5, minutes=30)),
            ("-05:30", timedelta(hours=-5, minutes=-30)),
            ("+02", timedelta(hours=2)),
            ("-11", timedelta(hours=-11)),
            ("+00:00", timedelta(0)),
        ],
    )
    def test_offsets(self, text, expected):
        assert offset(parse_tzinfo(text)) == expected

    def test_tzname_kept(self):
        assert parse_tzinfo("+05:30").tzname(None) == "+05:30"

    def test_build_tzinfo(self):
        assert build_tzinfo(None) is None
        assert build_tzinfo("") is None
        assert build_tzinfo("Z") is UTC
        assert offset(build_tzinfo("+02:30", "+", 2, 30)) == timedelta(hours=2, minutes=30)
        assert offset(build_tzinfo("-02:30", "-", 2, 30)) == -timedelta(hours=2, minutes=30)


class TestTzFormatting:
    def dt(self, tz):
        return datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)

    def test_naive(self):
        assert tz_isoformat(datetime(2020, 1, 1)) == ""

    def test_utc(self):
        assert tz_isoformat(self.dt(UTC)) == "Z"
        assert tz_isoformat(self.dt(UTC), "%z") == "Z"

    def test_zero_fixed_offset_is_z(self):
        assert tz_isoformat(self.dt(FixedOffset(0, 0, "x"))) == "Z"

    def test_positive(self):
        dt = self.dt(FixedOffset(5, 30, "x"))
        assert tz_isoformat(dt) == "+05:30"
        assert tz_isoformat(dt, "%Z") == "+05:30"
        assert tz_isoformat(dt, "%z") == "+0530"
        assert tz_isoformat(dt, "%h") == "+05"

    def test_negative(self):
        dt = self.dt(FixedOffset(-8, 0, "x"))
        assert tz_isoformat(dt) == "-08:00"
        assert tz_isoformat(dt, "%z") == "-0800"
        assert tz_isoformat(dt, "%h") == "-08"

    def test_negative_with_minutes(self):
        dt = self.dt(FixedOffset(-3, -30, "x"))
        assert tz_isoformat(dt) == "-03:30"

    def test_bad_format(self):
        with pytest.raises(ValueError):
            tz_isoformat(self.dt(FixedOffset(1, 0, "x")), "%q")

    def test_overflow(self):
        with pytest.raises(OverflowError):
            tz_isoformat(self.dt(FixedOffset(100, 0, "x")))

    @pytest.mark.parametrize("s", ["+05:30", "-05:30", "+00:45", "-11:00", "Z"])
    def test_roundtrip(self, s):
        dt = datetime(2020, 1, 1, tzinfo=parse_tzinfo(s))
        assert tz_isoformat(dt) == s


class TestTzinfoClasses:
    def test_utc(self):
        assert UTC.utcoffset(None) == timedelta(0)
        assert UTC.dst(None) == timedelta(0)
        assert UTC.tzname(None) == "UTC"

    def test_utc_pickle(self):
        assert pickle.loads(pickle.dumps(UTC)) is UTC

    def test_fixed_offset(self):
        tz = FixedOffset(-5, -30, "EST")
        assert tz.utcoffset(None) == timedelta(hours=-5, minutes=-30)
        assert tz.tzname(None) == "EST"
        assert tz.dst(None) == timedelta(0)
        assert "EST" in repr(tz)

    def test_fixed_offset_default(self):
        tz = FixedOffset()
        assert tz.utcoffset(None) == timedelta(0)
        assert tz.tzname(None) == "UTC"

    def test_fixed_offset_in_datetime(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=FixedOffset(2, 0, "x"))
        assert dt.astimezone(UTC).hour == 10


# --------------------------------------------------------------------- dates


class TestParseDate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2012-03-04", date(2012, 3, 4)),
            ("20120304", date(2012, 3, 4)),
            ("2012-W10-7", date(2012, 3, 11)),
            ("2012W107", date(2012, 3, 11)),
            ("2012-W10-1", date(2012, 3, 5)),
            ("2012-064", date(2012, 3, 4)),
            ("2012064", date(2012, 3, 4)),
            ("2012-001", date(2012, 1, 1)),
            ("2012-366", date(2012, 12, 31)),
            ("2012-W10", date(2012, 3, 5)),
            ("2012W10", date(2012, 3, 5)),
            ("2012-03", date(2012, 3, 1)),
            ("201203", date(2012, 3, 1)),
            ("2012", date(2012, 1, 1)),
            ("20", date(2001, 1, 1)),
            ("19", date(1901, 1, 1)),
            ("0001-01-01", date(1, 1, 1)),
            ("9999-12-31", date(9999, 12, 31)),
        ],
    )
    def test_formats(self, text, expected):
        assert parse_date(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ISO week 1 contains Jan 4th
            ("2004-W01-1", date(2003, 12, 29)),
            ("2008-W01-1", date(2007, 12, 31)),
            ("2009-W01-1", date(2008, 12, 29)),
            ("2010-W01-1", date(2010, 1, 4)),
            ("2009-W53-7", date(2010, 1, 3)),
            ("2015-W01-4", date(2015, 1, 1)),
            ("2016-W01-1", date(2016, 1, 4)),
            ("2018-W01-1", date(2018, 1, 1)),
            ("2020-W53-5", date(2021, 1, 1)),
        ],
    )
    def test_week_dates_match_isocalendar(self, text, expected):
        assert parse_date(text) == expected
        assert expected.isocalendar()[:2] == (int(text[:4]), int(text[6:8]))

    def test_all_week_dates_roundtrip(self):
        d = date(2000, 1, 1)
        while d < date(2030, 1, 1):
            y, w, wd = d.isocalendar()
            assert parse_date("%04d-W%02d-%d" % (y, w, wd)) == d
            d += timedelta(days=1)

    def test_all_ordinals_roundtrip(self):
        for y in (1999, 2000, 2012, 2100):
            d = date(y, 1, 1)
            while d.year == y:
                assert parse_date("%d-%03d" % (y, d.timetuple().tm_yday)) == d
                d += timedelta(days=1)

    def test_defaults(self):
        assert parse_date("2012", defaultmonth=5, defaultday=6) == date(2012, 5, 6)
        assert parse_date("2012-03", defaultday=9) == date(2012, 3, 9)
        assert parse_date("20", defaultmonth=2, defaultday=3) == date(2001, 2, 3)

    def test_expanded(self):
        assert parse_date("+0012-03-04", expanded=True) == date(12, 3, 4)
        assert parse_date("+002012-03-04", yeardigits=6) == date(2012, 3, 4)
        assert parse_date("+002012-064", yeardigits=6) == date(2012, 3, 4)

    def test_sign_requires_expanded(self):
        with pytest.raises(ISO8601Error):
            parse_date("+2012-03-04")

    def test_expanded_requires_sign(self):
        with pytest.raises(ISO8601Error):
            parse_date("002012-03-04", yeardigits=6)

    @pytest.mark.parametrize(
        "text", ["", "abc", "2012-3-4", "2012/03/04", "2012-03-04T", "2012-03-0", " 2012-03-04"]
    )
    def test_invalid_format(self, text):
        with pytest.raises(ISO8601Error):
            parse_date(text)

    @pytest.mark.parametrize("text", ["2012-13-01", "2012-02-30", "2011-02-29"])
    def test_invalid_values(self, text):
        with pytest.raises(ValueError):
            parse_date(text)

    def test_trailing_garbage(self):
        with pytest.raises(ISO8601Error):
            parse_date("2012-03-04x")
        with pytest.raises(ISO8601Error):
            parse_date("2012-03-04\n")

    def test_leap_ordinal(self):
        assert parse_date("2011-365") == date(2011, 12, 31)
        assert parse_date("2012-060") == date(2012, 2, 29)
        assert parse_date("2011-060") == date(2011, 3, 1)


class TestDateFormat:
    D = date(2012, 3, 4)

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.DATE_BAS_COMPLETE, "20120304"),
            (isodate.DATE_EXT_COMPLETE, "2012-03-04"),
            (isodate.DATE_BAS_WEEK_COMPLETE, "2012W097"),
            (isodate.DATE_EXT_WEEK_COMPLETE, "2012-W09-7"),
            (isodate.DATE_BAS_ORD_COMPLETE, "2012064"),
            (isodate.DATE_EXT_ORD_COMPLETE, "2012-064"),
            (isodate.DATE_BAS_WEEK, "2012W09"),
            (isodate.DATE_EXT_WEEK, "2012-W09"),
            (isodate.DATE_BAS_MONTH, "201203"),
            (isodate.DATE_EXT_MONTH, "2012-03"),
            (isodate.DATE_YEAR, "2012"),
            (isodate.DATE_CENTURY, "20"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert date_isoformat(self.D, fmt) == expected

    def test_default(self):
        assert date_isoformat(self.D) == "2012-03-04"

    def test_small_year_padding(self):
        assert date_isoformat(date(5, 1, 2)) == "0005-01-02"

    def test_yeardigits(self):
        assert date_isoformat(self.D, "%Y", 6) == "+002012"
        assert date_isoformat(self.D, "%C", 6) == "+0020"

    def test_ordinal_last_day(self):
        assert date_isoformat(date(2012, 12, 31), "%j") == "366"
        assert date_isoformat(date(2011, 12, 31), "%j") == "365"
        assert date_isoformat(date(2011, 1, 1), "%j") == "001"

    def test_weekday_and_week(self):
        assert date_isoformat(date(2012, 3, 5), "%w") == "1"
        assert date_isoformat(date(2012, 3, 4), "%w") == "7"
        assert date_isoformat(date(2010, 1, 3), "%W") == "53"
        assert date_isoformat(date(2010, 1, 4), "%W") == "01"

    def test_literal_percent_and_text(self):
        assert date_isoformat(self.D, "%%%Y%%") == "%2012%"
        assert date_isoformat(self.D, "x%dy") == "x04y"

    @pytest.mark.parametrize("fmt", ["%Y%m%d", "%Y-%m-%d", "%YW%W%w", "%Y%j", "%Y-%j"])
    def test_roundtrip(self, fmt):
        for d in (date(2012, 3, 4), date(2010, 6, 3), date(2009, 7, 31), date(2000, 2, 29)):
            assert parse_date(date_isoformat(d, fmt)) == d


# --------------------------------------------------------------------- times


class TestParseTime:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("20:23:45", time(20, 23, 45)),
            ("202345", time(20, 23, 45)),
            ("T20:23:45", time(20, 23, 45)),
            ("T202345", time(20, 23, 45)),
            ("20:23", time(20, 23)),
            ("2023", time(20, 23)),
            ("20", time(20)),
            ("00:00:00", time(0)),
            ("23:59:59", time(23, 59, 59)),
            ("20:23:45.5", time(20, 23, 45, 500000)),
            ("20:23:45,5", time(20, 23, 45, 500000)),
            ("202345.123456", time(20, 23, 45, 123456)),
            ("20:23:45.000001", time(20, 23, 45, 1)),
            ("20:23.5", time(20, 23, 30)),
            ("2023,25", time(20, 23, 15)),
            ("20.5", time(20, 30)),
            ("20,25", time(20, 15)),
            ("20.75", time(20, 45)),
        ],
    )
    def test_formats(self, text, expected):
        assert parse_time(text) == expected

    def test_microsecond_truncation(self):
        assert parse_time("20:23:45.1234569") == time(20, 23, 45, 123456)

    def test_utc(self):
        t = parse_time("20:23:45Z")
        assert t.tzinfo is UTC
        assert t.replace(tzinfo=None) == time(20, 23, 45)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("20:23:45+01:00", timedelta(hours=1)),
            ("20:23:45-01:00", timedelta(hours=-1)),
            ("202345+0530", timedelta(hours=5, minutes=30)),
            ("202345-0530", timedelta(hours=-5, minutes=-30)),
            ("20:23+02", timedelta(hours=2)),
            ("20-08", timedelta(hours=-8)),
            ("20:23:45+00:00", timedelta(0)),
        ],
    )
    def test_offsets(self, text, expected):
        assert parse_time(text).utcoffset() == expected

    def test_naive(self):
        assert parse_time("20:23:45").tzinfo is None

    def test_offset_converts_correctly(self):
        dt = datetime.combine(date(2020, 1, 1), parse_time("12:00:00+02:00"))
        assert dt.astimezone(UTC).hour == 10
        dt = datetime.combine(date(2020, 1, 1), parse_time("12:00:00-02:00"))
        assert dt.astimezone(UTC).hour == 14

    @pytest.mark.parametrize("text", ["", "x", "2:23:45", "20:2", "20:23:45 ", "20:23:45Zx", "24x"])
    def test_invalid(self, text):
        with pytest.raises(ISO8601Error):
            parse_time(text)

    @pytest.mark.parametrize("text", ["25:00:00", "20:60:00", "20:23:60"])
    def test_out_of_range(self, text):
        with pytest.raises(ValueError):
            parse_time(text)


class TestTimeFormat:
    T = time(2, 3, 4, 5)

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.TIME_BAS_COMPLETE, "020304"),
            (isodate.TIME_EXT_COMPLETE, "02:03:04"),
            (isodate.TIME_BAS_MINUTE, "0203"),
            (isodate.TIME_EXT_MINUTE, "02:03"),
            (isodate.TIME_HOUR, "02"),
            ("%f", "000005"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert time_isoformat(self.T, fmt) == expected

    def test_default_naive(self):
        assert time_isoformat(time(20, 23, 45)) == "20:23:45"

    def test_default_utc(self):
        assert time_isoformat(time(20, 23, 45, tzinfo=UTC)) == "20:23:45Z"

    def test_default_offset(self):
        assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"
        assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(-5, 0, "x"))) == "01:02:03-05:00"

    def test_tz_formats(self):
        t = time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x"))
        assert time_isoformat(t, "%H%M%S%z") == "010203-0530"
        assert time_isoformat(t, "%H%h") == "01-05"

    @pytest.mark.parametrize("text", ["20:23:45", "20:23:45Z", "20:23:45+05:30", "20:23:45-03:00"])
    def test_roundtrip(self, text):
        assert time_isoformat(parse_time(text)) == text


# ------------------------------------------------------------------ datetimes


class TestDatetime:
    def test_parse(self):
        assert parse_datetime("2012-03-04T20:23:45") == datetime(2012, 3, 4, 20, 23, 45)

    def test_parse_basic(self):
        assert parse_datetime("20120304T202345") == datetime(2012, 3, 4, 20, 23, 45)

    def test_parse_tz(self):
        dt = parse_datetime("2012-03-04T20:23:45-05:00")
        assert dt.utcoffset() == timedelta(hours=-5)
        assert dt.astimezone(UTC) == datetime(2012, 3, 5, 1, 23, 45, tzinfo=UTC)
        dt = parse_datetime("2012-03-04T20:23:45+05:00")
        assert dt.astimezone(UTC) == datetime(2012, 3, 4, 15, 23, 45, tzinfo=UTC)

    def test_parse_week_and_ordinal(self):
        assert parse_datetime("2012-W10-7T01:02:03") == datetime(2012, 3, 11, 1, 2, 3)
        assert parse_datetime("2012-064T01:02:03Z") == datetime(2012, 3, 4, 1, 2, 3, tzinfo=UTC)

    def test_missing_t(self):
        with pytest.raises(ISO8601Error):
            parse_datetime("2012-03-04 20:23:45")

    def test_two_t(self):
        with pytest.raises(ISO8601Error):
            parse_datetime("2012-03-04T20:23:45T")

    def test_format_default(self):
        assert datetime_isoformat(datetime(2012, 3, 4, 20, 23, 45)) == "2012-03-04T20:23:45"
        assert (
            datetime_isoformat(datetime(2012, 3, 4, 20, 23, 45, tzinfo=UTC))
            == "2012-03-04T20:23:45Z"
        )

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.DT_EXT_COMPLETE, "2012-03-04T20:23:45+05:30"),
            (isodate.DT_BAS_COMPLETE, "20120304T202345+0530"),
            (isodate.DT_EXT_ORD_COMPLETE, "2012-064T20:23:45+05:30"),
            (isodate.DT_BAS_ORD_COMPLETE, "2012064T202345+0530"),
            (isodate.DT_EXT_WEEK_COMPLETE, "2012-W09-7T20:23:45+05:30"),
            (isodate.DT_BAS_WEEK_COMPLETE, "2012W097T202345+0530"),
        ],
    )
    def test_format_variants(self, fmt, expected):
        dt = datetime(2012, 3, 4, 20, 23, 45, tzinfo=FixedOffset(5, 30, "x"))
        assert datetime_isoformat(dt, fmt) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "2012-03-04T20:23:45Z",
            "2012-03-04T20:23:45+05:30",
            "2012-03-04T20:23:45-08:00",
            "2012-03-04T00:00:00",
        ],
    )
    def test_roundtrip(self, text):
        assert datetime_isoformat(parse_datetime(text)) == text


# ------------------------------------------------------------------ durations


class TestParseDuration:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("P1D", timedelta(days=1)),
            ("P1W", timedelta(weeks=1)),
            ("P2W", timedelta(days=14)),
            ("PT1H", timedelta(hours=1)),
            ("PT1M", timedelta(minutes=1)),
            ("PT1S", timedelta(seconds=1)),
            ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
            ("PT36H", timedelta(hours=36)),
            ("PT0.5S", timedelta(milliseconds=500)),
            ("PT0,5S", timedelta(milliseconds=500)),
            ("P0.5D", timedelta(hours=12)),
            ("PT1.5H", timedelta(minutes=90)),
            ("P1W2D", timedelta(days=9)),
            ("P0D", timedelta(0)),
            ("PT0S", timedelta(0)),
            ("-P1D", timedelta(days=-1)),
            ("-PT1H30M", -timedelta(hours=1, minutes=30)),
            ("PT1.123456S", timedelta(seconds=1, microseconds=123456)),
        ],
    )
    def test_timedelta(self, text, expected):
        r = parse_duration(text)
        assert isinstance(r, timedelta)
        assert r == expected

    def test_each_component_distinct(self):
        r = parse_duration("P1W2DT3H4M5S")
        assert r == timedelta(weeks=1, days=2, hours=3, minutes=4, seconds=5)

    def test_minutes_vs_months(self):
        r = parse_duration("PT5M")
        assert isinstance(r, timedelta) and r == timedelta(minutes=5)
        r = parse_duration("P5M")
        assert isinstance(r, Duration)
        assert r.months == 5 and r.tdelta == timedelta(0)

    def test_years_months(self):
        r = parse_duration("P1Y2M3DT4H5M6S")
        assert isinstance(r, Duration)
        assert r.years == 1 and r.months == 2
        assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    def test_years_only(self):
        r = parse_duration("P3Y")
        assert isinstance(r, Duration)
        assert r.years == 3 and r.months == 0

    def test_fractional_years(self):
        r = parse_duration("P1.5Y")
        assert r.years == Decimal("1.5")

    def test_negative_duration(self):
        r = parse_duration("-P1Y2M3D")
        assert isinstance(r, Duration)
        assert r.years == -1 and r.months == -2
        assert r.tdelta == timedelta(days=-3)

    def test_plus_sign(self):
        assert parse_duration("+P1D") == timedelta(days=1)

    def test_not_as_timedelta(self):
        r = parse_duration("P1D", as_timedelta_if_possible=False)
        assert isinstance(r, Duration)
        assert r.tdelta == timedelta(days=1)

    def test_negative_not_as_timedelta(self):
        r = parse_duration("-P1D", as_timedelta_if_possible=False)
        assert isinstance(r, Duration)
        assert r.tdelta == timedelta(days=-1)

    def test_alternative_format(self):
        r = parse_duration("P0001-02-03T04:05:06")
        assert isinstance(r, Duration)
        assert r.years == 1 and r.months == 2
        assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    def test_alternative_basic_format(self):
        r = parse_duration("P00010203T040506")
        assert r.years == 1 and r.months == 2
        assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    @pytest.mark.parametrize("text", ["P", "1D", "", "P1", "PD", "P1H", "PT1D", "P1D1Y", "P-1D"])
    def test_invalid(self, text):
        with pytest.raises(ISO8601Error):
            parse_duration(text)

    def test_not_a_string(self):
        with pytest.raises(TypeError):
            parse_duration(5)

    def test_trailing_t(self):
        # "T" designator without any time component is accepted by the regex
        # only if it matches; ensure no crash other than ISO8601Error/valid
        try:
            parse_duration("P1DT")
        except ISO8601Error:
            pass


class TestDurationFormat:
    @pytest.mark.parametrize(
        "td,expected",
        [
            (timedelta(days=1), "P1D"),
            (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
            (timedelta(hours=36), "P1DT12H"),
            (timedelta(minutes=90), "PT1H30M"),
            (timedelta(seconds=5), "PT5S"),
            (timedelta(milliseconds=500), "PT0.5S"),
            (timedelta(seconds=1, microseconds=1), "PT1.000001S"),
            (timedelta(seconds=1, microseconds=120000), "PT1.12S"),
            (timedelta(0), "P0D"),
            (timedelta(days=-1), "-P1D"),
            (timedelta(days=-1, hours=-2), "-P1DT2H"),
            (timedelta(seconds=-30), "-PT30S"),
            (timedelta(days=400), "P400D"),
        ],
    )
    def test_timedelta(self, td, expected):
        assert duration_isoformat(td) == expected

    @pytest.mark.parametrize(
        "dur,expected",
        [
            (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
            (Duration(years=1), "P1Y"),
            (Duration(months=3), "P3M"),
            (Duration(years=2, hours=1), "P2YT1H"),
            (Duration(0), "P0D"),
            (Duration(years=-1, months=-2), "-P1Y2M"),
            (Duration(years=-1, days=-2), "-P1Y2D"),
            (Duration(months=-1), "-P1M"),
            (Duration(days=-1), "-P1D"),
        ],
    )
    def test_duration(self, dur, expected):
        assert duration_isoformat(dur) == expected

    def test_weeks_format(self):
        assert duration_isoformat(timedelta(days=14), isodate.D_WEEK) == "P2W"
        assert duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
        assert duration_isoformat(timedelta(days=-14), isodate.D_WEEK) == "-P2W"

    def test_alt_format(self):
        d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
        assert duration_isoformat(d, isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
        assert duration_isoformat(d, isodate.D_ALT_BAS) == "P00010203T040506"

    def test_custom_fields(self):
        td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
        assert strftime(td, "%d|%H|%M|%S|%f|%%") == "03|04|05|06|000007|%"

    def test_custom_year_month(self):
        d = Duration(years=12, months=3)
        assert strftime(d, "%Y-%m") == "0012-03"
        assert strftime(d, "%C", 6) == "+0000"

    @pytest.mark.parametrize(
        "text",
        ["P1D", "P1DT2H3M4S", "PT0.5S", "-P1D", "P1Y2M3DT4H5M6S", "-P1Y2M", "PT1H", "P5M", "P1W"],
    )
    def test_roundtrip(self, text):
        r = parse_duration(text)
        again = parse_duration(duration_isoformat(r))
        assert again == r

    def test_week_text_roundtrip(self):
        assert parse_duration(duration_isoformat(parse_duration("P2W"))) == timedelta(days=14)


# ------------------------------------------------------------ Duration class


class TestDuration:
    def test_init(self):
        d = Duration(days=1, hours=2, minutes=3, seconds=4, microseconds=5, weeks=1, years=2, months=3)
        assert d.years == 2 and d.months == 3
        assert d.tdelta == timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5, weeks=1)

    def test_getattr_delegation(self):
        d = Duration(days=2, seconds=3)
        assert d.days == 2 and d.seconds == 3 and d.microseconds == 0
        assert d.total_seconds() == 2 * 86400 + 3

    def test_eq(self):
        assert Duration(years=1) == Duration(months=12)
        assert Duration(years=1) != Duration(months=11)
        assert Duration(days=1) == Duration(hours=24)
        assert Duration(days=1) != Duration(days=2)
        assert not (Duration(days=1) != Duration(days=1))
        assert Duration(years=1) != Duration(years=1, days=1)

    def test_eq_timedelta(self):
        assert Duration(days=1) == timedelta(days=1)
        assert Duration(days=1) != timedelta(days=2)
        assert not (Duration(days=1) != timedelta(days=1))
        assert Duration(months=1, days=1) != timedelta(days=1)
        assert not (Duration(months=1, days=1) == timedelta(days=1))

    def test_hash(self):
        assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
        assert len({Duration(days=1), Duration(days=1), Duration(days=2)}) == 2

    def test_neg(self):
        d = -Duration(years=1, months=2, days=3, seconds=4)
        assert d.years == -1 and d.months == -2
        assert d.tdelta == -timedelta(days=3, seconds=4)

    def test_add_duration(self):
        d = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
        assert d.years == 5 and d.months == 7 and d.tdelta == timedelta(days=9)

    def test_add_timedelta(self):
        d = Duration(years=1, days=1) + timedelta(days=2)
        assert isinstance(d, Duration)
        assert d.years == 1 and d.tdelta == timedelta(days=3)
        d = timedelta(days=2) + Duration(years=1, days=1)
        assert d.years == 1 and d.tdelta == timedelta(days=3)

    def test_add_date(self):
        assert Duration(years=1, months=1) + date(2000, 1, 1) == date(2001, 2, 1)
        assert date(2000, 1, 1) + Duration(years=1, months=1) == date(2001, 2, 1)

    def test_add_date_month_carry(self):
        assert date(2000, 11, 15) + Duration(months=3) == date(2001, 2, 15)
        assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
        assert date(2000, 12, 15) + Duration(months=12) == date(2001, 12, 15)
        assert date(2000, 1, 15) + Duration(months=24) == date(2002, 1, 15)

    def test_add_date_day_clamp(self):
        assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
        assert date(2001, 1, 31) + Duration(months=1) == date(2001, 2, 28)
        assert date(2000, 1, 31) + Duration(months=3) == date(2000, 4, 30)
        assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
        assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)

    def test_add_date_with_days(self):
        assert date(2000, 1, 31) + Duration(months=1, days=1) == date(2000, 3, 1)

    def test_add_datetime(self):
        dt = datetime(2000, 1, 31, 10, 30)
        assert dt + Duration(months=1, hours=2) == datetime(2000, 2, 29, 12, 30)

    def test_add_fractional_raises(self):
        with pytest.raises(ValueError):
            date(2000, 1, 1) + Duration(months=Decimal("1.5"))
        with pytest.raises(ValueError):
            Duration(years=Decimal("0.5")) + date(2000, 1, 1)

    def test_add_unsupported(self):
        with pytest.raises(TypeError):
            Duration(days=1) + 5

    def test_sub_duration(self):
        d = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
        assert d.years == 4 and d.months == 5 and d.tdelta == timedelta(days=6)

    def test_sub_timedelta(self):
        d = Duration(years=1, days=5) - timedelta(days=2)
        assert d.years == 1 and d.tdelta == timedelta(days=3)

    def test_rsub_timedelta(self):
        d = timedelta(days=5) - Duration(years=1, days=2)
        assert isinstance(d, Duration)
        assert d.years == -1 and d.tdelta == timedelta(days=3)

    def test_rsub_date(self):
        assert date(2001, 3, 1) - Duration(years=1, months=1) == date(2000, 2, 1)
        assert date(2001, 1, 15) - Duration(months=1) == date(2000, 12, 15)
        assert date(2001, 1, 15) - Duration(months=13) == date(1999, 12, 15)
        assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
        assert date(2001, 3, 31) - Duration(months=1) == date(2001, 2, 28)
        assert date(2000, 3, 10) - Duration(months=1, days=2) == date(2000, 2, 8)

    def test_rsub_fractional_raises(self):
        with pytest.raises(ValueError):
            date(2000, 1, 1) - Duration(months=Decimal("0.5"))

    def test_mul(self):
        d = Duration(years=1, months=2, days=3) * 3
        assert d.years == 3 and d.months == 6 and d.tdelta == timedelta(days=9)
        d = 2 * Duration(years=1, months=2, days=3)
        assert d.years == 2 and d.months == 4 and d.tdelta == timedelta(days=6)

    def test_mul_unsupported(self):
        with pytest.raises(TypeError):
            Duration(days=1) * "a"

    def test_totimedelta_start(self):
        assert Duration(months=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=31)
        assert Duration(months=1).totimedelta(start=date(2000, 2, 1)) == timedelta(days=29)
        assert Duration(years=1).totimedelta(start=date(2000, 1, 1)) == timedelta(days=366)

    def test_totimedelta_end(self):
        assert Duration(months=1).totimedelta(end=date(2000, 3, 1)) == timedelta(days=29)
        assert Duration(years=1).totimedelta(end=date(2001, 1, 1)) == timedelta(days=366)

    def test_totimedelta_errors(self):
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta()
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))

    def test_str(self):
        assert str(Duration(years=2, months=3)).startswith("2 years, 3 months, ")
        assert str(Duration(months=1)).startswith("1 month, ")
        assert str(Duration(years=1)).startswith("1 years, ")
        assert str(Duration(days=1)) == "1 day, 0:00:00"

    def test_repr(self):
        r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
        assert r.endswith("Duration(1, 2, 3, years=4, months=5)")

    def test_pickle(self):
        d = Duration(years=1, months=2, days=3)
        assert pickle.loads(pickle.dumps(d)) == d

    def test_float_params(self):
        d = Duration(years=1.5, months=0.25)
        assert d.years == Decimal("1.5") and d.months == Decimal("0.25")


class TestHelpers:
    @pytest.mark.parametrize(
        "year,month,expected",
        [
            (2001, 1, 31),
            (2001, 2, 28),
            (2000, 2, 29),
            (1900, 2, 28),
            (2004, 2, 29),
            (2100, 2, 28),
            (2400, 2, 29),
            (2001, 3, 31),
            (2001, 4, 30),
            (2001, 5, 31),
            (2001, 6, 30),
            (2001, 7, 31),
            (2001, 8, 31),
            (2001, 9, 30),
            (2001, 10, 31),
            (2001, 11, 30),
            (2001, 12, 31),
        ],
    )
    def test_max_days(self, year, month, expected):
        assert max_days_in_month(year, month) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            (1, (0, 1)),
            (12, (0, 12)),
            (13, (1, 1)),
            (14, (1, 2)),
            (0, (-1, 12)),
            (-1, (-1, 11)),
            (-11, (-1, 1)),
            (-12, (-2, 12)),
            (25, (2, 1)),
        ],
    )
    def test_fquotmod(self, val, expected):
        assert fquotmod(Decimal(val), 1, 13) == (expected[0], Decimal(expected[1]))


# --------------------------------------------------------------- strftime misc


class TestStrftime:
    def test_unknown_directive_untouched(self):
        assert strftime(date(2012, 3, 4), "%Q%Y") == "%Q2012"

    def test_dispatch(self):
        assert strftime(timedelta(days=1), "P%P") == "P1D"

    def test_datetime_in_date_formatter(self):
        dt = datetime(2012, 3, 4, 5, 6, 7, 8, tzinfo=UTC)
        assert strftime(dt, "%Y-%m-%dT%H:%M:%S.%f%Z") == "2012-03-04T05:06:07.000008Z"

    def test_duration_days_zero_pad(self):
        assert strftime(timedelta(days=5), "%d") == "05"
        assert strftime(timedelta(hours=5), "%H") == "05"
        assert strftime(timedelta(minutes=5), "%M") == "05"
        assert strftime(timedelta(days=14), "%W") == "02"

    def test_public_api_exports(self):
        for name in isodate.__all__:
            assert hasattr(isodate, name)
