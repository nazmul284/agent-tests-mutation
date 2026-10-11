import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    Duration,
    ISO8601Error,
    FixedOffset,
    UTC,
    parse_date,
    parse_datetime,
    parse_duration,
    parse_time,
    parse_tzinfo,
    date_isoformat,
    time_isoformat,
    datetime_isoformat,
    duration_isoformat,
    tz_isoformat,
    strftime,
)
from isodate.duration import fquotmod, max_days_in_month
from isodate.tzinfo import ZERO, LOCAL


# ---------------------------------------------------------------- helpers
class TestHelpers:
    @pytest.mark.parametrize(
        "year,month,expected",
        [
            (2023, 1, 31), (2023, 2, 28), (2024, 2, 29), (1900, 2, 28),
            (2000, 2, 29), (2100, 2, 28), (2023, 3, 31), (2023, 4, 30),
            (2023, 5, 31), (2023, 6, 30), (2023, 7, 31), (2023, 8, 31),
            (2023, 9, 30), (2023, 10, 31), (2023, 11, 30), (2023, 12, 31),
        ],
    )
    def test_max_days_in_month(self, year, month, expected):
        assert max_days_in_month(year, month) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            (Decimal(5), (0, Decimal(5))),
            (Decimal(13), (1, Decimal(1))),
            (Decimal(14), (1, Decimal(2))),
            (Decimal(0), (-1, Decimal(12))),
            (Decimal(-1), (-1, Decimal(11))),
            (Decimal(-12), (-2, Decimal(12))),
            (Decimal(1), (0, Decimal(1))),
            (Decimal(12), (0, Decimal(12))),
            (Decimal(25), (2, Decimal(1))),
        ],
    )
    def test_fquotmod(self, val, expected):
        assert fquotmod(val, 1, 13) == expected


# ---------------------------------------------------------------- dates
class TestParseDate:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("2023-05-17", date(2023, 5, 17)),
            ("20230517", date(2023, 5, 17)),
            ("0001-01-01", date(1, 1, 1)),
            ("9999-12-31", date(9999, 12, 31)),
            ("2024-02-29", date(2024, 2, 29)),
            ("2023-135", date(2023, 5, 15)),
            ("2023135", date(2023, 5, 15)),
            ("2024-366", date(2024, 12, 31)),
            ("2023-001", date(2023, 1, 1)),
            ("2023-W20-3", date(2023, 5, 17)),
            ("2023W203", date(2023, 5, 17)),
            ("2023-W01-1", date(2023, 1, 2)),
            ("2009-W01-1", date(2008, 12, 29)),
            ("2009-W53-7", date(2010, 1, 3)),
            ("2004-W53-6", date(2005, 1, 1)),
            ("2023-W20-7", date(2023, 5, 21)),
            ("2023-W20", date(2023, 5, 15)),
            ("2023W20", date(2023, 5, 15)),
            ("2009-W01", date(2008, 12, 29)),
            ("2023-05", date(2023, 5, 1)),
            ("202305", date(2023, 5, 1)),
            ("2023", date(2023, 1, 1)),
            ("19", date(1901, 1, 1)),
            ("20", date(2001, 1, 1)),
        ],
    )
    def test_valid(self, s, expected):
        assert parse_date(s) == expected

    def test_defaults_for_incomplete(self):
        assert parse_date("2023", defaultmonth=6, defaultday=15) == date(2023, 6, 15)
        assert parse_date("2023-05", defaultday=9) == date(2023, 5, 9)
        assert parse_date("19", defaultmonth=3, defaultday=4) == date(1901, 3, 4)
        # complete dates ignore defaults
        assert parse_date("2023-05-17", defaultmonth=1, defaultday=2) == date(2023, 5, 17)

    @pytest.mark.parametrize(
        "s",
        ["", "abc", "2023-5-17", "2023/05/17", "2023-05-17 ", " 2023-05-17",
         "2023-05-17\n", "20230", "202", "2023-W2", "2023-W20-", "+2023-05-17",
         "2023-05-17T00", "12345", "2023-1234"],
    )
    def test_unrecognised(self, s):
        with pytest.raises(ISO8601Error):
            parse_date(s)

    @pytest.mark.parametrize("s", ["2023-02-30", "2023-13-01", "2023-04-31",
                                   "2023-02-29", "2023-05-00", "2023-13"])
    def test_out_of_range(self, s):
        with pytest.raises(ValueError):
            parse_date(s)

    def test_error_is_valueerror(self):
        assert issubclass(ISO8601Error, ValueError)

    def test_ordinal_366_non_leap_rolls_or_errors(self):
        # 2023 has 365 days; day 366 either errors or lands in 2024.
        try:
            r = parse_date("2023-366")
        except ValueError:
            return
        assert r == date(2024, 1, 1)

    def test_expanded(self):
        assert parse_date("+002023-05-17", 6) == date(2023, 5, 17)
        assert parse_date("+2023-05-17", expanded=True) == date(2023, 5, 17)
        assert parse_date("+002023", 6) == date(2023, 1, 1)
        assert parse_date("+0020", 6) == date(2001, 1, 1)

    def test_expanded_requires_sign_for_non4(self):
        with pytest.raises(ISO8601Error):
            parse_date("002023-05-17", 6)
        with pytest.raises(ISO8601Error):
            parse_date("2023-05-17", 6)

    def test_four_digit_year_rejects_six_digit(self):
        with pytest.raises(ISO8601Error):
            parse_date("002023-05-17")

    def test_negative_year_unsupported(self):
        with pytest.raises(ValueError):
            parse_date("-2023-05-17", expanded=True)

    def test_regex_cache(self):
        a = isodate.isodates.build_date_regexps(4, False)
        assert a is isodate.isodates.build_date_regexps(4, False)
        assert a is not isodate.isodates.build_date_regexps(4, True)
        assert isodate.isodates.build_date_regexps(6) is isodate.isodates.build_date_regexps(6, True)


class TestDateFormat:
    D = date(2023, 5, 7)

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.DATE_EXT_COMPLETE, "2023-05-07"),
            (isodate.DATE_BAS_COMPLETE, "20230507"),
            (isodate.DATE_EXT_WEEK_COMPLETE, "2023-W18-7"),
            (isodate.DATE_BAS_WEEK_COMPLETE, "2023W187"),
            (isodate.DATE_EXT_ORD_COMPLETE, "2023-127"),
            (isodate.DATE_BAS_ORD_COMPLETE, "2023127"),
            (isodate.DATE_EXT_WEEK, "2023-W18"),
            (isodate.DATE_BAS_WEEK, "2023W18"),
            (isodate.DATE_EXT_MONTH, "2023-05"),
            (isodate.DATE_BAS_MONTH, "202305"),
            (isodate.DATE_YEAR, "2023"),
            (isodate.DATE_CENTURY, "20"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert date_isoformat(self.D, fmt) == expected

    def test_default(self):
        assert date_isoformat(self.D) == "2023-05-07"

    def test_small_year_padded(self):
        assert date_isoformat(date(5, 1, 2)) == "0005-01-02"

    def test_yeardigits(self):
        assert date_isoformat(self.D, isodate.DATE_EXT_COMPLETE, 6) == "+002023-05-07"
        assert date_isoformat(self.D, isodate.DATE_YEAR, 5) == "+02023"
        assert date_isoformat(self.D, isodate.DATE_CENTURY, 6) == "+0020"

    def test_ordinal_leap_year_end(self):
        assert date_isoformat(date(2024, 12, 31), isodate.DATE_EXT_ORD_COMPLETE) == "2024-366"
        assert date_isoformat(date(2024, 1, 1), isodate.DATE_EXT_ORD_COMPLETE) == "2024-001"

    def test_week_year_boundary(self):
        assert date_isoformat(date(2005, 1, 1), "%W") == "53"
        assert date_isoformat(date(2008, 12, 29), "%W") == "01"

    def test_weekday_monday_is_1(self):
        assert date_isoformat(date(2023, 5, 15), "%w") == "1"
        assert date_isoformat(date(2023, 5, 21), "%w") == "7"

    def test_percent_and_unknown(self):
        assert strftime(self.D, "%%d") == "%d"
        assert strftime(self.D, "100%% %Y") == "100% 2023"
        assert strftime(self.D, "%q %d") == "%q 07"
        assert strftime(self.D, "abc") == "abc"

    def test_roundtrip(self):
        for d in (date(2023, 5, 7), date(2024, 6, 30), date(2023, 1, 15), date(2020, 3, 1),
                  date(1999, 7, 4)):
            for fmt in (isodate.DATE_EXT_COMPLETE, isodate.DATE_BAS_COMPLETE,
                        isodate.DATE_EXT_ORD_COMPLETE, isodate.DATE_BAS_ORD_COMPLETE,
                        isodate.DATE_EXT_WEEK_COMPLETE, isodate.DATE_BAS_WEEK_COMPLETE):
                assert parse_date(date_isoformat(d, fmt)) == d, (d, fmt)

    def test_roundtrip_every_day_of_range(self):
        d = date(2020, 1, 7)
        end = date(2020, 12, 20)
        while d < end:
            assert parse_date(date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE)) == d
            assert parse_date(date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE)) == d
            d += timedelta(days=1)


# ---------------------------------------------------------------- tz
class TestTz:
    def test_utc(self):
        assert UTC.utcoffset(None) == ZERO
        assert UTC.dst(None) == ZERO
        assert UTC.tzname(None) == "UTC"

    def test_utc_pickle_singleton(self):
        assert pickle.loads(pickle.dumps(UTC)) is UTC

    def test_fixed_offset(self):
        tz = FixedOffset(5, 30, "X")
        assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
        assert tz.dst(None) == ZERO
        assert tz.tzname(None) == "X"
        assert "X" in repr(tz)

    def test_fixed_offset_default(self):
        tz = FixedOffset()
        assert tz.utcoffset(None) == ZERO
        assert tz.tzname(None) == "UTC"

    def test_fixed_offset_negative(self):
        assert FixedOffset(-3, -30, "n").utcoffset(None) == -timedelta(hours=3, minutes=30)

    def test_local(self):
        dt = datetime(2023, 1, 1, 12, tzinfo=LOCAL)
        assert isinstance(dt.utcoffset(), timedelta)
        assert isinstance(LOCAL.tzname(dt), str)
        with pytest.raises(Exception):
            LOCAL.utcoffset(None)

    @pytest.mark.parametrize(
        "s,offset",
        [
            ("+05:30", timedelta(hours=5, minutes=30)),
            ("-05:30", -timedelta(hours=5, minutes=30)),
            ("+0530", timedelta(hours=5, minutes=30)),
            ("-0530", -timedelta(hours=5, minutes=30)),
            ("+05", timedelta(hours=5)),
            ("-05", -timedelta(hours=5)),
            ("+00:00", timedelta(0)),
            ("+14:00", timedelta(hours=14)),
            ("-00:30", -timedelta(minutes=30)),
        ],
    )
    def test_parse_tzinfo_offsets(self, s, offset):
        tz = parse_tzinfo(s)
        assert tz.utcoffset(None) == offset
        assert tz.tzname(None) == s

    def test_parse_tzinfo_z_and_empty(self):
        assert parse_tzinfo("Z") is UTC
        assert parse_tzinfo("") is None

    def test_parse_tzinfo_garbage_matches_empty(self):
        # TZ_RE is not anchored; an unmatched string just yields no tz
        assert parse_tzinfo("garbage") is None

    def test_build_tzinfo(self):
        from isodate.isotzinfo import build_tzinfo
        assert build_tzinfo(None) is None
        assert build_tzinfo("") is None
        assert build_tzinfo("Z") is UTC
        tz = build_tzinfo("-03:30", "-", 3, 30)
        assert tz.utcoffset(None) == -timedelta(hours=3, minutes=30)
        tz = build_tzinfo("+03:30", "+", 3, 30)
        assert tz.utcoffset(None) == timedelta(hours=3, minutes=30)

    @pytest.mark.parametrize(
        "offset,fmt,expected",
        [
            (timedelta(hours=5, minutes=30), "%Z", "+05:30"),
            (timedelta(hours=5, minutes=30), "%z", "+0530"),
            (timedelta(hours=5, minutes=30), "%h", "+05"),
            (-timedelta(hours=5, minutes=30), "%Z", "-05:30"),
            (-timedelta(hours=5, minutes=30), "%z", "-0530"),
            (-timedelta(hours=5, minutes=30), "%h", "-05"),
            (-timedelta(minutes=30), "%Z", "-00:30"),
            (timedelta(hours=14), "%Z", "+14:00"),
            (timedelta(hours=1), "%z", "+0100"),
        ],
    )
    def test_tz_isoformat(self, offset, fmt, expected):
        dt = datetime(2023, 1, 1, tzinfo=_Tz(offset))
        assert tz_isoformat(dt, fmt) == expected

    def test_tz_isoformat_default_format(self):
        dt = datetime(2023, 1, 1, tzinfo=FixedOffset(2, 0, "x"))
        assert tz_isoformat(dt) == "+02:00"

    def test_tz_isoformat_naive_and_utc(self):
        assert tz_isoformat(datetime(2023, 1, 1)) == ""
        assert tz_isoformat(datetime(2023, 1, 1, tzinfo=UTC)) == "Z"
        assert tz_isoformat(datetime(2023, 1, 1, tzinfo=FixedOffset(0, 0, "UTC"))) == "Z"

    def test_tz_isoformat_utcoffset_none(self):
        assert tz_isoformat(datetime(2023, 1, 1, tzinfo=_NoneTz())) == ""

    def test_tz_isoformat_bad_format(self):
        dt = datetime(2023, 1, 1, tzinfo=FixedOffset(2, 0, "x"))
        with pytest.raises(ValueError):
            tz_isoformat(dt, "%q")

    def test_tz_isoformat_overflow(self):
        with pytest.raises(OverflowError):
            tz_isoformat(datetime(2023, 1, 1, tzinfo=_Big()))

    def test_tz_isoformat_dst_nonzero_is_not_z(self):
        dt = datetime(2023, 1, 1, tzinfo=_DstTz())
        assert tz_isoformat(dt) == "+00:00"


from datetime import tzinfo as _tzinfo


class _Tz(_tzinfo):
    def __init__(self, off):
        self.off = off

    def utcoffset(self, dt):
        return self.off

    def dst(self, dt):
        return timedelta(0)


class _NoneTz(_tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


class _Big(_tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=100)

    def dst(self, dt):
        return timedelta(0)


class _DstTz(_tzinfo):
    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return timedelta(hours=1)


# ---------------------------------------------------------------- time
class TestParseTime:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("12:34:56", time(12, 34, 56)),
            ("123456", time(12, 34, 56)),
            ("T12:34:56", time(12, 34, 56)),
            ("T123456", time(12, 34, 56)),
            ("00:00:00", time(0, 0, 0)),
            ("23:59:59", time(23, 59, 59)),
            ("12:34:56.789", time(12, 34, 56, 789000)),
            ("12:34:56,789", time(12, 34, 56, 789000)),
            ("123456.5", time(12, 34, 56, 500000)),
            ("12:34:56.123456", time(12, 34, 56, 123456)),
            ("12:34:56.0000001", time(12, 34, 56, 0)),
            ("12:34:56.9999999", time(12, 34, 56, 999999)),
            ("12:34", time(12, 34)),
            ("1234", time(12, 34)),
            ("12:34.5", time(12, 34, 30)),
            ("1234.5", time(12, 34, 30)),
            ("12:34,25", time(12, 34, 15)),
            ("12", time(12)),
            ("12.5", time(12, 30)),
            ("12,25", time(12, 15)),
            ("T12", time(12)),
            ("12.75", time(12, 45)),
        ],
    )
    def test_valid(self, s, expected):
        assert parse_time(s) == expected

    def test_naive_has_no_tz(self):
        assert parse_time("12:34:56").tzinfo is None

    def test_z(self):
        assert parse_time("12:34:56Z").tzinfo is UTC
        assert parse_time("12:34Z").tzinfo is UTC
        assert parse_time("12Z").tzinfo is UTC
        assert parse_time("123456Z").tzinfo is UTC

    @pytest.mark.parametrize(
        "s,off",
        [
            ("12:34:56+05:30", timedelta(hours=5, minutes=30)),
            ("12:34:56-05:30", -timedelta(hours=5, minutes=30)),
            ("123456+0530", timedelta(hours=5, minutes=30)),
            ("123456-0530", -timedelta(hours=5, minutes=30)),
            ("12:34:56+05", timedelta(hours=5)),
            ("12:34:56-05", -timedelta(hours=5)),
            ("12:34+01:00", timedelta(hours=1)),
            ("12-03", -timedelta(hours=3)),
            ("12.5+01", timedelta(hours=1)),
            ("12:34:56.5-00:30", -timedelta(minutes=30)),
        ],
    )
    def test_tz(self, s, off):
        t = parse_time(s)
        assert t.utcoffset() == off

    def test_tz_with_fraction(self):
        t = parse_time("12:34:56.5+05:30")
        assert t.microsecond == 500000
        assert t.utcoffset() == timedelta(hours=5, minutes=30)

    @pytest.mark.parametrize(
        "s",
        ["", "1", "123", "12:3", "12:34:5", "1:34:56", "ab:cd:ef", "12:34:56.", "12:34:56 ",
         " 12:34:56", "12:34:56+5", "12:34:56+05:3", "12-34-56", "12:34:56ZZ", "12:34:56\n",
         "12:34:56+0530:00", "tt12:34:56"],
    )
    def test_unrecognised(self, s):
        with pytest.raises(ISO8601Error):
            parse_time(s)

    @pytest.mark.parametrize("s", ["24:00:00", "12:60:00", "12:34:60", "25", "99:00", "12:99"])
    def test_out_of_range(self, s):
        with pytest.raises(ValueError):
            parse_time(s)

    def test_cache_built_once(self):
        a = isodate.isotime.build_time_regexps()
        assert a is isodate.isotime.build_time_regexps()
        assert len(a) == 5


class TestTimeFormat:
    T = time(7, 8, 9, 123456)

    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (isodate.TIME_EXT_COMPLETE, "07:08:09"),
            (isodate.TIME_BAS_COMPLETE, "070809"),
            (isodate.TIME_EXT_MINUTE, "07:08"),
            (isodate.TIME_BAS_MINUTE, "0708"),
            (isodate.TIME_HOUR, "07"),
            ("%H:%M:%S.%f", "07:08:09.123456"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert time_isoformat(self.T, fmt) == expected

    def test_default_naive(self):
        assert time_isoformat(time(1, 2, 3)) == "01:02:03"

    def test_default_with_tz(self):
        assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
        assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))) == "01:02:03+05:30"

    def test_tz_formats(self):
        t = time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x"))
        assert time_isoformat(t, "%H%M%S%z") == "010203-0530"
        assert time_isoformat(t, "%H%M%S%h") == "010203-05"
        assert time_isoformat(t, "%H%M%S%Z") == "010203-05:30"

    def test_microsecond_padding(self):
        assert strftime(time(0, 0, 0, 5), "%f") == "000005"

    def test_roundtrip(self):
        for t in (time(0, 0, 0), time(23, 59, 59),
                  time(1, 2, 3, tzinfo=UTC), time(1, 2, 3, tzinfo=FixedOffset(-3, -30, "x"))):
            assert parse_time(time_isoformat(t)) == t
            assert parse_time(time_isoformat(t, "%H%M%S%z")) == t
            assert parse_time(time_isoformat(t, "%H:%M:%S.%f%Z")) == t


# ---------------------------------------------------------------- datetime
class TestDatetime:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("2023-05-17T12:34:56", datetime(2023, 5, 17, 12, 34, 56)),
            ("20230517T123456", datetime(2023, 5, 17, 12, 34, 56)),
            ("2023-05-17T12:34:56.789", datetime(2023, 5, 17, 12, 34, 56, 789000)),
            ("2023-135T12:34", datetime(2023, 5, 15, 12, 34)),
            ("2023-W20-3T12", datetime(2023, 5, 17, 12)),
            ("2023-05-17T00:00:00", datetime(2023, 5, 17)),
            ("2023-05-17T23:59:59", datetime(2023, 5, 17, 23, 59, 59)),
        ],
    )
    def test_valid(self, s, expected):
        assert parse_datetime(s) == expected

    def test_tz(self):
        dt = parse_datetime("2023-05-17T12:34:56Z")
        assert dt.tzinfo is UTC
        dt = parse_datetime("2023-05-17T12:34:56+05:30")
        assert dt.utcoffset() == timedelta(hours=5, minutes=30)
        assert dt.hour == 12
        dt = parse_datetime("2023-05-17T12:34:56-08")
        assert dt.utcoffset() == -timedelta(hours=8)

    @pytest.mark.parametrize(
        "s",
        ["2023-05-17", "2023-05-17 12:34:56", "2023-05-17t12:34:56", "", "T",
         "2023-05-17T12:34:56T", "2023-05-17TT12", "T12:34:56", "2023-05-17T", "x"],
    )
    def test_errors(self, s):
        with pytest.raises(ISO8601Error):
            parse_datetime(s)

    def test_invalid_values(self):
        with pytest.raises(ValueError):
            parse_datetime("2023-02-30T12:00:00")
        with pytest.raises(ValueError):
            parse_datetime("2023-02-03T25:00:00")

    def test_format_default(self):
        assert datetime_isoformat(datetime(2023, 5, 17, 12, 34, 56)) == "2023-05-17T12:34:56"
        assert datetime_isoformat(datetime(2023, 5, 17, 12, 34, 56, tzinfo=UTC)) == "2023-05-17T12:34:56Z"
        assert (
            datetime_isoformat(datetime(2023, 5, 17, 12, 34, 56, tzinfo=FixedOffset(-7, 0, "x")))
            == "2023-05-17T12:34:56-07:00"
        )

    def test_format_constants(self):
        dt = datetime(2023, 5, 7, 1, 2, 3, tzinfo=FixedOffset(5, 30, "x"))
        assert datetime_isoformat(dt, isodate.DT_EXT_COMPLETE) == "2023-05-07T01:02:03+05:30"
        assert datetime_isoformat(dt, isodate.DT_BAS_COMPLETE) == "20230507T010203+0530"
        assert datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE) == "2023-127T01:02:03+05:30"
        assert datetime_isoformat(dt, isodate.DT_BAS_ORD_COMPLETE) == "2023127T010203+0530"
        assert datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE) == "2023-W18-7T01:02:03+05:30"
        assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "2023W187T010203+0530"

    def test_roundtrip(self):
        for dt in (datetime(2023, 5, 17, 12, 34, 56, tzinfo=UTC),
                   datetime(1999, 12, 31, 23, 59, 59),
                   datetime(2000, 2, 29, 0, 0, 0, tzinfo=FixedOffset(5, 45, "x"))):
            assert parse_datetime(datetime_isoformat(dt)) == dt
            assert parse_datetime(datetime_isoformat(dt, isodate.DT_BAS_COMPLETE)) == dt
            assert parse_datetime(datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE)) == dt
            assert parse_datetime(datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE)) == dt


# ---------------------------------------------------------------- duration parse
class TestParseDuration:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("P1D", timedelta(days=1)),
            ("P10D", timedelta(days=10)),
            ("PT1H", timedelta(hours=1)),
            ("PT1M", timedelta(minutes=1)),
            ("PT1S", timedelta(seconds=1)),
            ("P1W", timedelta(weeks=1)),
            ("P2W", timedelta(days=14)),
            ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
            ("PT0.5S", timedelta(milliseconds=500)),
            ("PT0,5S", timedelta(milliseconds=500)),
            ("P0.5D", timedelta(hours=12)),
            ("PT36H", timedelta(hours=36)),
            ("PT90M", timedelta(minutes=90)),
            ("PT1.5H", timedelta(minutes=90)),
            ("P0D", timedelta(0)),
            ("PT0S", timedelta(0)),
            ("P1W2D", timedelta(days=9)),
            ("-P1D", timedelta(days=-1)),
            ("-PT1H30M", -timedelta(hours=1, minutes=30)),
            ("-P1W", -timedelta(weeks=1)),
            ("+P1D", timedelta(days=1)),
            ("PT1.123456S", timedelta(seconds=1, microseconds=123456)),
            ("P3DT4H", timedelta(days=3, hours=4)),
        ],
    )
    def test_timedelta_results(self, s, expected):
        r = parse_duration(s)
        assert type(r) is timedelta
        assert r == expected

    @pytest.mark.parametrize(
        "s,years,months,tdelta",
        [
            ("P1Y", 1, 0, timedelta(0)),
            ("P1M", 0, 1, timedelta(0)),
            ("P1Y2M", 1, 2, timedelta(0)),
            ("P1Y2M3DT4H5M6S", 1, 2, timedelta(days=3, hours=4, minutes=5, seconds=6)),
            ("P18M", 0, 18, timedelta(0)),
            ("P1YT1H", 1, 0, timedelta(hours=1)),
            ("P1M1W", 0, 1, timedelta(weeks=1)),
            ("P0.5Y", Decimal("0.5"), 0, timedelta(0)),
            ("P1,5M", 0, Decimal("1.5"), timedelta(0)),
            ("P12Y", 12, 0, timedelta(0)),
        ],
    )
    def test_duration_results(self, s, years, months, tdelta):
        r = parse_duration(s)
        assert isinstance(r, Duration)
        assert r.years == years
        assert r.months == months
        assert r.tdelta == tdelta

    def test_negative_duration(self):
        r = parse_duration("-P1Y2M3D")
        assert isinstance(r, Duration)
        assert r.years == -1
        assert r.months == -2
        assert r.tdelta == timedelta(days=-3)

    def test_month_vs_minute(self):
        r = parse_duration("PT5M")
        assert r == timedelta(minutes=5)
        r = parse_duration("P5M")
        assert isinstance(r, Duration) and r.months == 5

    def test_as_timedelta_if_possible_false(self):
        r = parse_duration("P1D", as_timedelta_if_possible=False)
        assert isinstance(r, Duration)
        assert r.tdelta == timedelta(days=1)
        assert r.years == 0 and r.months == 0

    def test_zero_years_months_gives_timedelta(self):
        assert type(parse_duration("P0Y0M5D")) is timedelta

    @pytest.mark.parametrize(
        "s",
        ["", "P", "1D", "P1", "P1H", "PT1D", "P1S", "P1D1Y", "P1H1D", "PD", "P-1D",
         "P1D ", " P1D", "p1d", "P1.D", "P.5D", "PT1H1D", "garbage", "P1Y1", "-P",
         "--P1D", "PT1S1M", "P1M1Y", "P1D2W"],
    )
    def test_errors(self, s):
        with pytest.raises(ISO8601Error):
            parse_duration(s)

    def test_non_string(self):
        for v in (None, 5, b"P1D", timedelta(1)):
            with pytest.raises(TypeError):
                parse_duration(v)

    # alternative format
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("P0000-00-03T04:05:06", timedelta(days=3, hours=4, minutes=5, seconds=6)),
            ("P00000003T040506", timedelta(days=3, hours=4, minutes=5, seconds=6)),
        ],
    )
    def test_alternative_timedelta(self, s, expected):
        # year 0 / month 0 can't be built by datetime so this must fail clearly
        with pytest.raises(ValueError):
            parse_duration(s)

    @pytest.mark.parametrize(
        "s",
        ["P0001-02-03T04:05:06", "P00010203T040506"],
    )
    def test_alternative_duration(self, s):
        r = parse_duration(s)
        assert isinstance(r, Duration)
        assert r.years == 1
        assert r.months == 2
        assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)

    def test_alternative_with_fraction(self):
        r = parse_duration("P0001-02-03T04:05:06.5")
        assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=500000)

    def test_alternative_ordinal(self):
        r = parse_duration("P0001-045T04:05:06")
        assert isinstance(r, Duration)
        assert r.years == 1
        assert r.months == 2
        assert r.tdelta == timedelta(days=14, hours=4, minutes=5, seconds=6)

    def test_alternative_as_duration(self):
        r = parse_duration("P0001-02-03T04:05:06", as_timedelta_if_possible=False)
        assert isinstance(r, Duration)

    def test_decimal_precision_preserved(self):
        r = parse_duration("P0.1Y")
        assert r.years == Decimal("0.1")


# ---------------------------------------------------------------- duration format
class TestDurationFormat:
    @pytest.mark.parametrize(
        "td,expected",
        [
            (timedelta(days=1), "P1D"),
            (timedelta(hours=1), "PT1H"),
            (timedelta(minutes=1), "PT1M"),
            (timedelta(seconds=1), "PT1S"),
            (timedelta(0), "P0D"),
            (timedelta(days=1, hours=2, minutes=3, seconds=4), "P1DT2H3M4S"),
            (timedelta(hours=36), "P1DT12H"),
            (timedelta(days=400), "P400D"),
            (timedelta(microseconds=500000), "PT0.5S"),
            (timedelta(seconds=1, microseconds=123456), "PT1.123456S"),
            (timedelta(microseconds=1), "PT0.000001S"),
            (timedelta(seconds=59, microseconds=990000), "PT59.99S"),
            (timedelta(days=7), "P7D"),
            (timedelta(minutes=90), "PT1H30M"),
            (timedelta(hours=1, seconds=1), "PT1H1S"),
            (timedelta(days=1, seconds=1), "P1DT1S"),
            (timedelta(seconds=60), "PT1M"),
            (timedelta(seconds=3600), "PT1H"),
            (timedelta(seconds=86400), "P1D"),
            (timedelta(seconds=10), "PT10S"),
        ],
    )
    def test_timedelta(self, td, expected):
        assert duration_isoformat(td) == expected

    @pytest.mark.parametrize(
        "td,expected",
        [
            (-timedelta(days=1), "-P1D"),
            (-timedelta(hours=1), "-PT1H"),
            (-timedelta(days=1, hours=2), "-P1DT2H"),
            (-timedelta(seconds=1), "-PT1S"),
            (-timedelta(microseconds=1), "-PT0.000001S"),
            (timedelta(days=-1, hours=12), "-PT12H"),
        ],
    )
    def test_negative_timedelta(self, td, expected):
        assert duration_isoformat(td) == expected

    @pytest.mark.parametrize(
        "dur,expected",
        [
            (Duration(years=1), "P1Y"),
            (Duration(months=1), "P1M"),
            (Duration(years=1, months=2), "P1Y2M"),
            (Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), "P1Y2M3DT4H5M6S"),
            (Duration(years=2, hours=1), "P2YT1H"),
            (Duration(months=18), "P18M"),
            (Duration(years=1, days=1), "P1Y1D"),
            (Duration(years=0, months=0), "P0D"),
            (Duration(years=Decimal("1.5")), "P1.5Y"),
        ],
    )
    def test_duration(self, dur, expected):
        assert duration_isoformat(dur) == expected

    @pytest.mark.parametrize(
        "dur,expected",
        [
            (Duration(years=-1), "-P1Y"),
            (Duration(months=-3), "-P3M"),
            (Duration(years=-1, months=-2, days=-3), "-P1Y2M3D"),
            (Duration(days=-1), "-P1D"),
            (Duration(years=1, days=-1), "-P1Y1D"),
        ],
    )
    def test_negative_duration(self, dur, expected):
        assert duration_isoformat(dur) == expected

    def test_week_format(self):
        assert duration_isoformat(timedelta(weeks=2), isodate.D_WEEK) == "P2W"
        assert duration_isoformat(timedelta(days=15), isodate.D_WEEK) == "P2W"
        assert duration_isoformat(timedelta(days=6), isodate.D_WEEK) == "P0W"
        assert duration_isoformat(Duration(days=14), isodate.D_WEEK) == "P2W"
        assert duration_isoformat(-timedelta(weeks=3), isodate.D_WEEK) == "-P3W"

    def test_alt_formats(self):
        td = timedelta(days=3, hours=4, minutes=5, seconds=6)
        d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
        assert duration_isoformat(d, isodate.D_ALT_EXT) == "P0001-02-03T04:05:06"
        assert duration_isoformat(d, isodate.D_ALT_BAS) == "P00010203T040506"
        assert duration_isoformat(d, "P%Y-%m-%dT%H:%M:%S") == "P0001-02-03T04:05:06"
        assert strftime(td, "%d %H %M %S") == "03 04 05 06"

    def test_alt_ordinal_not_supported_for_duration(self):
        # %j isn't a duration directive; it is left untouched
        d = Duration(years=1, days=45)
        assert strftime(d, "%j") == "%j"

    def test_strftime_duration_directives(self):
        d = Duration(years=12, months=3, days=15, hours=7, minutes=8, seconds=9, microseconds=42)
        assert strftime(d, "%Y|%m|%d|%H|%M|%S|%f") == "0012|03|15|07|08|09|000042"
        assert strftime(d, "%C") == "00"
        assert strftime(Duration(years=1200), "%C") == "12"
        assert strftime(d, "%%") == "%"
        assert strftime(d, "%Y", 6) == "+000012"
        assert strftime(d, "%w") == "%w"

    def test_strftime_timedelta_directives(self):
        td = timedelta(days=15, hours=7, minutes=8, seconds=9, microseconds=42)
        assert strftime(td, "%d|%H|%M|%S|%f") == "15|07|08|09|000042"
        assert strftime(td, "%W") == "02"

    def test_percent_P_only(self):
        assert strftime(timedelta(days=1), "%P") == "1D"
        assert strftime(timedelta(days=15), "%p") == "2W"

    def test_roundtrip_timedelta(self):
        for td in (timedelta(days=1, hours=2, minutes=3, seconds=4),
                   timedelta(microseconds=1), timedelta(days=-5, hours=3),
                   timedelta(hours=100), timedelta(0), timedelta(seconds=59, microseconds=500)):
            assert parse_duration(duration_isoformat(td)) == td

    def test_roundtrip_duration(self):
        for d in (Duration(years=1, months=2, days=3, hours=4),
                  Duration(months=5), Duration(years=-1, months=-2, days=-3)):
            assert parse_duration(duration_isoformat(d)) == d

    def test_roundtrip_alt(self):
        d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
        assert parse_duration(duration_isoformat(d, isodate.D_ALT_EXT)) == d
        assert parse_duration(duration_isoformat(d, isodate.D_ALT_BAS)) == d


# ---------------------------------------------------------------- Duration class
class TestDurationClass:
    def test_init_defaults(self):
        d = Duration()
        assert d.years == 0 and d.months == 0
        assert d.tdelta == timedelta(0)

    def test_init_all_fields(self):
        d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5,
                     hours=6, weeks=7, months=8, years=9)
        assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
        assert d.months == 8 and d.years == 9

    def test_years_months_are_decimal(self):
        d = Duration(years=1, months=2)
        assert isinstance(d.years, Decimal) and isinstance(d.months, Decimal)
        d = Duration(years=0.1, months=0.2)
        assert d.years == Decimal("0.1") and d.months == Decimal("0.2")

    def test_attribute_proxy(self):
        d = Duration(days=3, seconds=4, microseconds=5)
        assert d.days == 3 and d.seconds == 4 and d.microseconds == 5
        assert d.total_seconds() == timedelta(3, 4, 5).total_seconds()
        with pytest.raises(AttributeError):
            d.nonexistent

    def test_str(self):
        assert str(Duration(years=2, months=3, days=1)) == "2 years, 3 months, 1 day, 0:00:00"
        assert str(Duration(months=1)) == "1 month, 0:00:00"
        assert str(Duration(years=1)) == "1 years, 0:00:00"
        assert str(Duration(hours=1)) == "1:00:00"
        assert str(Duration(months=2)) == "2 months, 0:00:00"

    def test_repr(self):
        r = repr(Duration(years=1, months=2, days=3, seconds=4, microseconds=5))
        assert r == "isodate.duration.Duration(3, 4, 5, years=1, months=2)"

    def test_hash(self):
        assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
        assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2

    def test_eq_duration(self):
        assert Duration(years=1, days=2) == Duration(years=1, days=2)
        assert Duration(years=1) != Duration(years=2)
        assert Duration(months=1) != Duration(months=2)
        assert Duration(days=1) != Duration(days=2)
        assert not (Duration(days=1) == Duration(days=2))
        assert not (Duration(days=1) != Duration(days=1))

    def test_eq_years_months_equivalent(self):
        assert Duration(years=1) == Duration(months=12)
        assert not (Duration(years=1) != Duration(months=12))
        assert Duration(years=1, months=-12) == Duration()

    def test_eq_timedelta(self):
        assert Duration(days=1) == timedelta(days=1)
        assert Duration(days=1) != timedelta(days=2)
        assert Duration(years=1) != timedelta(days=365)
        assert not (Duration(years=1) == timedelta(days=365))
        assert Duration(months=1, days=1) != timedelta(days=1)
        assert not (Duration(days=1) != timedelta(days=1))

    def test_neg(self):
        d = -Duration(years=1, months=2, days=3, seconds=4)
        assert d.years == -1 and d.months == -2
        assert d.tdelta == -timedelta(days=3, seconds=4)
        assert -(-Duration(years=1, days=1)) == Duration(years=1, days=1)

    def test_add_duration(self):
        r = Duration(years=1, months=2, days=3) + Duration(years=4, months=5, days=6)
        assert isinstance(r, Duration)
        assert r.years == 5 and r.months == 7
        assert r.tdelta == timedelta(days=9)

    def test_add_timedelta(self):
        r = Duration(years=1, days=1) + timedelta(days=2)
        assert isinstance(r, Duration)
        assert r.years == 1 and r.tdelta == timedelta(days=3)
        r = timedelta(days=2) + Duration(years=1, days=1)
        assert isinstance(r, Duration)
        assert r.years == 1 and r.tdelta == timedelta(days=3)

    def test_add_bad_type(self):
        with pytest.raises(TypeError):
            Duration(days=1) + 5
        with pytest.raises(TypeError):
            Duration(days=1) + "x"

    def test_add_date(self):
        assert date(2023, 1, 15) + Duration(months=1) == date(2023, 2, 15)
        assert Duration(months=1) + date(2023, 1, 15) == date(2023, 2, 15)
        assert date(2023, 1, 15) + Duration(years=1) == date(2024, 1, 15)
        assert date(2023, 1, 15) + Duration(years=1, months=1, days=1) == date(2024, 2, 16)

    def test_add_date_month_end_clamp(self):
        assert date(2023, 1, 31) + Duration(months=1) == date(2023, 2, 28)
        assert date(2024, 1, 31) + Duration(months=1) == date(2024, 2, 29)
        assert date(2023, 3, 31) + Duration(months=1) == date(2023, 4, 30)
        assert date(2024, 2, 29) + Duration(years=1) == date(2025, 2, 28)
        assert date(2024, 2, 29) + Duration(years=4) == date(2028, 2, 29)
        assert date(2023, 1, 30) + Duration(months=1) == date(2023, 2, 28)
        assert date(2023, 1, 29) + Duration(months=1) == date(2023, 2, 28)
        assert date(2023, 1, 28) + Duration(months=1) == date(2023, 2, 28)

    def test_add_date_month_carry(self):
        assert date(2023, 11, 15) + Duration(months=2) == date(2024, 1, 15)
        assert date(2023, 12, 15) + Duration(months=1) == date(2024, 1, 15)
        assert date(2023, 12, 15) + Duration(months=12) == date(2024, 12, 15)
        assert date(2023, 12, 15) + Duration(months=13) == date(2025, 1, 15)
        assert date(2023, 1, 15) + Duration(months=11) == date(2023, 12, 15)
        assert date(2023, 6, 15) + Duration(months=30) == date(2025, 12, 15)
        assert date(2023, 6, 15) + Duration(months=-6) == date(2022, 12, 15)
        assert date(2023, 6, 15) + Duration(months=-5) == date(2023, 1, 15)
        assert date(2023, 1, 15) + Duration(months=-1) == date(2022, 12, 15)
        assert date(2023, 1, 15) + Duration(months=-13) == date(2021, 12, 15)

    def test_add_datetime(self):
        dt = datetime(2023, 1, 31, 10, 20, 30)
        assert dt + Duration(months=1, hours=2) == datetime(2023, 2, 28, 12, 20, 30)
        assert dt + Duration(years=1, seconds=40) == datetime(2024, 1, 31, 10, 21, 10)

    def test_add_date_days_applied_after_months(self):
        # month arithmetic (with clamping) first, then the timedelta
        assert date(2023, 1, 31) + Duration(months=1, days=1) == date(2023, 3, 1)

    def test_add_date_fractional_raises(self):
        with pytest.raises(ValueError):
            date(2023, 1, 1) + Duration(months=Decimal("0.5"))
        with pytest.raises(ValueError):
            date(2023, 1, 1) + Duration(years=Decimal("0.5"))
        # whole-valued decimals are fine
        assert date(2023, 1, 1) + Duration(years=Decimal("1.0")) == date(2024, 1, 1)

    def test_sub_duration(self):
        r = Duration(years=5, months=7, days=9) - Duration(years=1, months=2, days=3)
        assert isinstance(r, Duration)
        assert r.years == 4 and r.months == 5 and r.tdelta == timedelta(days=6)

    def test_sub_timedelta(self):
        r = Duration(years=1, days=5) - timedelta(days=2)
        assert isinstance(r, Duration)
        assert r.years == 1 and r.tdelta == timedelta(days=3)

    def test_sub_bad_type(self):
        with pytest.raises(TypeError):
            Duration(days=1) - 5

    def test_rsub_timedelta(self):
        r = timedelta(days=5) - Duration(years=1, days=2)
        assert isinstance(r, Duration)
        assert r.years == -1 and r.tdelta == timedelta(days=3)

    def test_rsub_date(self):
        assert date(2023, 3, 15) - Duration(months=1) == date(2023, 2, 15)
        assert date(2023, 3, 15) - Duration(years=1) == date(2022, 3, 15)
        assert date(2023, 3, 31) - Duration(months=1) == date(2023, 2, 28)
        assert date(2023, 1, 15) - Duration(months=1) == date(2022, 12, 15)
        assert date(2023, 1, 15) - Duration(months=2) == date(2022, 11, 15)
        assert date(2023, 3, 15) - Duration(months=3) == date(2022, 12, 15)
        assert date(2023, 3, 15) - Duration(months=14) == date(2022, 1, 15)
        assert date(2023, 3, 15) - Duration(months=15) == date(2021, 12, 15)
        assert date(2024, 3, 31) - Duration(months=1) == date(2024, 2, 29)
        assert date(2023, 3, 15) - Duration(months=1, days=5) == date(2023, 2, 10)
        assert date(2023, 3, 15) - Duration(days=5) == date(2023, 3, 10)

    def test_rsub_datetime(self):
        assert datetime(2023, 3, 31, 5, 6, 7) - Duration(months=1, hours=1) == datetime(
            2023, 2, 28, 4, 6, 7
        )

    def test_rsub_fractional_raises(self):
        with pytest.raises(ValueError):
            date(2023, 1, 1) - Duration(months=Decimal("0.5"))
        with pytest.raises(ValueError):
            date(2023, 1, 1) - Duration(years=Decimal("0.5"))

    def test_rsub_bad_type(self):
        with pytest.raises(TypeError):
            5 - Duration(days=1)

    def test_mul(self):
        r = Duration(years=1, months=2, days=3, seconds=4) * 3
        assert r.years == 3 and r.months == 6
        assert r.tdelta == timedelta(days=9, seconds=12)
        r = 2 * Duration(years=1, days=1)
        assert r.years == 2 and r.tdelta == timedelta(days=2)
        r = Duration(years=1, days=1) * 0
        assert r == Duration()
        r = Duration(months=2) * -1
        assert r.months == -2

    def test_mul_non_int(self):
        with pytest.raises(TypeError):
            Duration(days=1) * 1.5
        with pytest.raises(TypeError):
            Duration(days=1) * "a"

    def test_totimedelta_start(self):
        d = Duration(months=1)
        assert d.totimedelta(start=date(2023, 1, 1)) == timedelta(days=31)
        assert d.totimedelta(start=date(2023, 2, 1)) == timedelta(days=28)
        assert d.totimedelta(start=date(2024, 2, 1)) == timedelta(days=29)
        assert Duration(years=1).totimedelta(start=date(2024, 1, 1)) == timedelta(days=366)
        assert Duration(years=1, days=1).totimedelta(start=date(2023, 1, 1)) == timedelta(days=366)

    def test_totimedelta_end(self):
        d = Duration(months=1)
        assert d.totimedelta(end=date(2023, 3, 1)) == timedelta(days=28)
        assert d.totimedelta(end=date(2023, 2, 1)) == timedelta(days=31)
        assert Duration(years=1).totimedelta(end=date(2025, 1, 1)) == timedelta(days=366)

    def test_totimedelta_errors(self):
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta()
        with pytest.raises(ValueError):
            Duration(months=1).totimedelta(start=date(2023, 1, 1), end=date(2023, 2, 1))

    def test_pickle(self):
        d = Duration(years=1, months=2, days=3, seconds=4)
        d2 = pickle.loads(pickle.dumps(d))
        assert d2 == d
        assert d2.years == 1 and d2.months == 2 and d2.tdelta == d.tdelta

    def test_copy(self):
        import copy
        d = Duration(years=1, days=3)
        assert copy.deepcopy(d) == d
        assert copy.copy(d) == d

    def test_parse_and_add_end_to_end(self):
        assert date(2023, 1, 31) + parse_duration("P1M") == date(2023, 2, 28)
        assert datetime(2023, 1, 1) + parse_duration("P1Y2M3DT4H") == datetime(2024, 3, 4, 4)
        assert date(2023, 5, 17) - parse_duration("P1Y") == date(2022, 5, 17)
        assert date(2023, 5, 17) + parse_duration("-P1Y") == date(2022, 5, 17)


# ---------------------------------------------------------------- package
class TestPackage:
    def test_all_exports_exist(self):
        for name in isodate.__all__:
            assert hasattr(isodate, name), name

    def test_format_constants(self):
        assert isodate.DATE_BAS_COMPLETE == "%Y%m%d"
        assert isodate.DATE_EXT_COMPLETE == "%Y-%m-%d"
        assert isodate.DATE_BAS_WEEK_COMPLETE == "%YW%W%w"
        assert isodate.DATE_EXT_WEEK_COMPLETE == "%Y-W%W-%w"
        assert isodate.DATE_BAS_ORD_COMPLETE == "%Y%j"
        assert isodate.DATE_EXT_ORD_COMPLETE == "%Y-%j"
        assert isodate.DATE_BAS_WEEK == "%YW%W"
        assert isodate.DATE_EXT_WEEK == "%Y-W%W"
        assert isodate.DATE_BAS_MONTH == "%Y%m"
        assert isodate.DATE_EXT_MONTH == "%Y-%m"
        assert isodate.DATE_YEAR == "%Y"
        assert isodate.DATE_CENTURY == "%C"
        assert isodate.TIME_BAS_COMPLETE == "%H%M%S"
        assert isodate.TIME_EXT_COMPLETE == "%H:%M:%S"
        assert isodate.TIME_BAS_MINUTE == "%H%M"
        assert isodate.TIME_EXT_MINUTE == "%H:%M"
        assert isodate.TIME_HOUR == "%H"
        assert isodate.TZ_BAS == "%z"
        assert isodate.TZ_EXT == "%Z"
        assert isodate.TZ_HOUR == "%h"
        assert isodate.D_DEFAULT == "P%P"
        assert isodate.D_WEEK == "P%p"
        assert isodate.D_ALT_EXT == "P%Y-%m-%dT%H:%M:%S"
        assert isodate.D_ALT_BAS == "P%Y%m%dT%H%M%S"
        assert isodate.D_ALT_EXT_ORD == "P%Y-%jT%H:%M:%S"
        assert isodate.D_ALT_BAS_ORD == "P%Y%jT%H%M%S"
        assert isodate.DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
        assert isodate.DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
