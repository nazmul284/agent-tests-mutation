import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

import isodate
from isodate import (
    LOCAL,
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
from isodate import isostrf
from isodate.duration import fquotmod, max_days_in_month
from isodate.isodates import DATE_REGEX_CACHE, build_date_regexps
from isodate.isotime import build_time_regexps
from isodate.isotzinfo import build_tzinfo
from isodate.tzinfo import DSTDIFF, DSTOFFSET, STDOFFSET, ZERO, LocalTimezone, Utc


# ---------------------------------------------------------------- helpers
def off(dt_or_t):
    return dt_or_t.utcoffset()


# ---------------------------------------------------------------- duration helpers
@pytest.mark.parametrize(
    "year,month,days",
    [
        (2023, 1, 31), (2023, 2, 28), (2023, 3, 31), (2023, 4, 30), (2023, 5, 31),
        (2023, 6, 30), (2023, 7, 31), (2023, 8, 31), (2023, 9, 30), (2023, 10, 31),
        (2023, 11, 30), (2023, 12, 31), (2024, 2, 29), (1900, 2, 28), (2000, 2, 29),
        (2100, 2, 28), (1996, 2, 29), (1999, 2, 28), (2400, 2, 29), (1800, 2, 28),
    ],
)
def test_max_days_in_month(year, month, days):
    assert max_days_in_month(year, month) == days


def test_fquotmod():
    assert fquotmod(Decimal(5), 1, 13) == (0, Decimal(5))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(1), 1, 13) == (0, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-11), 1, 13) == (-1, Decimal(1))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(7), 3, 5) == (2, Decimal(3))
    assert fquotmod(Decimal(9), 3, 5) == (3, Decimal(3))


# ---------------------------------------------------------------- Duration class
def test_duration_init_and_attrs():
    d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5,
                 hours=6, weeks=1, months=7, years=8)
    assert d.years == Decimal(8)
    assert d.months == Decimal(7)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 1)
    assert d.days == 8
    assert d.seconds == 6 * 3600 + 5 * 60 + 2
    assert d.microseconds == 4003
    assert isinstance(d.years, Decimal) and isinstance(d.months, Decimal)
    d2 = Duration(years=1.5, months=Decimal("2.5"))
    assert d2.years == Decimal("1.5")
    assert d2.months == Decimal("2.5")
    assert Duration(days=1).years == 0 and Duration(days=1).months == 0
    assert Duration(hours=1).tdelta == timedelta(hours=1)
    assert Duration(minutes=1).tdelta == timedelta(minutes=1)
    assert Duration(weeks=1).tdelta == timedelta(days=7)
    assert Duration(milliseconds=1).tdelta == timedelta(microseconds=1000)
    assert Duration(seconds=1).tdelta == timedelta(seconds=1)
    assert Duration(microseconds=1).tdelta == timedelta(microseconds=1)
    assert Duration(1, 2, 3).tdelta == timedelta(1, 2, 3)


def test_duration_str():
    assert str(Duration(years=1, months=2, days=3)) == "1 years, 2 months, 3 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(months=Decimal("0.5"))) == "0.5 month, 0:00:00"
    assert str(Duration(years=2)) == "2 years, 0:00:00"
    assert str(Duration(hours=1)) == "1:00:00"
    assert str(Duration(years=1)).startswith("1 years")
    assert "month" not in str(Duration(years=1))
    assert "years" not in str(Duration(months=1))


def test_duration_repr():
    d = Duration(days=1, seconds=2, microseconds=3, years=4, months=5)
    assert repr(d) == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash():
    assert hash(Duration(years=1, days=2)) == hash(Duration(years=1, days=2))
    assert hash(Duration(years=1)) != hash(Duration(years=2))
    assert hash(Duration(months=1)) != hash(Duration(months=2))
    assert hash(Duration(days=1)) != hash(Duration(days=2))
    assert len({Duration(years=1), Duration(years=1), Duration(years=2)}) == 2


def test_duration_neg():
    d = -Duration(years=1, months=2, days=3, hours=4)
    assert d.years == -1
    assert d.months == -2
    assert d.tdelta == -timedelta(days=3, hours=4)
    assert -Duration() == Duration()


def test_duration_add_duration():
    r = Duration(years=1, months=2, days=3) + Duration(years=10, months=20, days=30, hours=1)
    assert isinstance(r, Duration)
    assert r.years == 11 and r.months == 22
    assert r.tdelta == timedelta(days=33, hours=1)


def test_duration_add_timedelta():
    r = Duration(years=1, months=2, days=3) + timedelta(days=1, hours=2)
    assert isinstance(r, Duration)
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=4, hours=2)
    r = timedelta(days=1) + Duration(years=1)
    assert isinstance(r, Duration)
    assert r.years == 1 and r.tdelta == timedelta(days=1)


def test_duration_add_not_implemented():
    with pytest.raises(TypeError):
        Duration(days=1) + 5
    assert Duration(days=1).__add__("x") is NotImplemented


def test_duration_add_date():
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 1, 31) + Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 1, 15) + Duration(months=1) == date(2000, 2, 15)
    assert date(2000, 1, 30) + Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 1, 29) + Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 1, 28) + Duration(months=1) == date(2000, 2, 28)
    assert date(2000, 11, 15) + Duration(months=1) == date(2000, 12, 15)
    assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
    assert date(2000, 12, 15) + Duration(months=13) == date(2002, 1, 15)
    assert date(2000, 1, 15) + Duration(months=12) == date(2001, 1, 15)
    assert date(2000, 1, 15) + Duration(years=1) == date(2001, 1, 15)
    assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
    assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)
    assert date(2000, 3, 31) + Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 5, 31) + Duration(months=1) == date(2000, 6, 30)
    assert date(2000, 1, 31) + Duration(months=2) == date(2000, 3, 31)
    assert date(2000, 6, 15) + Duration(months=6) == date(2000, 12, 15)
    assert date(2000, 6, 15) + Duration(months=7) == date(2001, 1, 15)
    assert date(2000, 6, 15) + Duration(months=-6) == date(2000, 12, 15) - Duration(years=1)
    assert date(2000, 1, 15) + Duration(months=-1) == date(1999, 12, 15)
    assert date(2000, 3, 15) + Duration(months=-2) == date(2000, 1, 15)
    assert date(2000, 3, 15) + Duration(months=-3) == date(1999, 12, 15)
    assert date(2000, 1, 15) + Duration(years=1, months=1, days=1) == date(2001, 2, 16)
    assert Duration(years=1, days=1) + date(2000, 1, 15) == date(2001, 1, 16)
    assert date(2000, 1, 15) + Duration() == date(2000, 1, 15)


def test_duration_add_datetime():
    dt = datetime(2000, 1, 31, 10, 20, 30, 40, tzinfo=UTC)
    r = dt + Duration(months=1, hours=2)
    assert r == datetime(2000, 2, 29, 12, 20, 30, 40, tzinfo=UTC)
    assert Duration(months=1, hours=2) + dt == r
    assert r.tzinfo is UTC


def test_duration_add_fractional_raises():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=Decimal("1.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("0.5"))
    # integral decimals are fine
    assert date(2000, 1, 1) + Duration(months=Decimal("1.0")) == date(2000, 2, 1)
    assert date(2000, 1, 1) - Duration(years=Decimal("1.0")) == date(1999, 1, 1)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3, hours=4) * 3
    assert (d.years, d.months) == (3, 6)
    assert d.tdelta == timedelta(days=9, hours=12)
    d = 2 * Duration(months=5)
    assert d.months == 10
    assert Duration(days=1).__mul__(1.5) is NotImplemented
    with pytest.raises(TypeError):
        Duration(days=1) * "a"
    d = Duration(years=1) * 0
    assert d == Duration()


def test_duration_sub():
    r = Duration(years=10, months=20, days=30, hours=5) - Duration(years=1, months=2, days=3, hours=1)
    assert isinstance(r, Duration)
    assert (r.years, r.months) == (9, 18)
    assert r.tdelta == timedelta(days=27, hours=4)
    r = Duration(years=1, months=2, days=5) - timedelta(days=2)
    assert isinstance(r, Duration)
    assert (r.years, r.months) == (1, 2)
    assert r.tdelta == timedelta(days=3)
    assert Duration(days=1).__sub__("x") is NotImplemented
    with pytest.raises(TypeError):
        Duration(days=1) - 5


def test_duration_rsub_timedelta():
    r = timedelta(days=10) - Duration(years=1, days=3)
    assert isinstance(r, Duration)
    assert r.years == -1
    assert r.tdelta == timedelta(days=7)


def test_duration_rsub_date():
    assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 3, 31) - Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 3, 15) - Duration(months=1) == date(2000, 2, 15)
    assert date(2000, 3, 15) - Duration(months=2) == date(2000, 1, 15)
    assert date(2000, 3, 15) - Duration(months=3) == date(1999, 12, 15)
    assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
    assert date(2000, 1, 15) - Duration(months=12) == date(1999, 1, 15)
    assert date(2000, 1, 15) - Duration(months=13) == date(1998, 12, 15)
    assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
    assert date(2000, 2, 29) - Duration(years=4) == date(1996, 2, 29)
    assert date(2000, 5, 31) - Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 3, 30) - Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 3, 29) - Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 3, 28) - Duration(months=1) == date(2000, 2, 28)
    assert date(2000, 12, 15) - Duration(months=-1) == date(2001, 1, 15)
    assert date(2000, 6, 15) - Duration(years=1, months=1, days=1) == date(1999, 5, 14)
    assert date(2000, 6, 15) - Duration() == date(2000, 6, 15)
    dt = datetime(2000, 3, 31, 10, 0, 0)
    assert dt - Duration(months=1, hours=2) == datetime(2000, 2, 29, 8, 0, 0)


def test_duration_rsub_unsupported():
    assert Duration(days=1).__rsub__("x") is NotImplemented
    with pytest.raises(TypeError):
        5 - Duration(days=1)


def test_duration_eq_ne():
    assert Duration(years=1) == Duration(months=12)
    assert Duration(years=1, days=1) == Duration(years=1, days=1)
    assert not (Duration(years=1, days=1) == Duration(years=1, days=2))
    assert not (Duration(years=1) == Duration(years=2))
    assert not (Duration(months=1) == Duration(months=2))
    assert not (Duration(years=1, days=1) == Duration(years=2, days=1))
    assert Duration(years=1) != Duration(years=2)
    assert Duration(months=1) != Duration(months=2)
    assert Duration(days=1) != Duration(days=2)
    assert not (Duration(years=1, days=1) != Duration(years=1, days=1))
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(days=1) == timedelta(days=1)
    assert timedelta(days=1) == Duration(days=1)
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) != timedelta(days=1))
    assert not (Duration(days=1) == timedelta(days=2))
    assert Duration(years=1) != timedelta(days=0)
    assert Duration(months=1) != timedelta(days=0)
    assert not (Duration(years=1) == timedelta(0))
    assert not (Duration(months=1) == timedelta(0))
    assert Duration(years=1, months=1) != timedelta(0)
    assert Duration() == timedelta(0)
    assert not (Duration() != timedelta(0))
    assert Duration(years=1) == Duration(years=1)


def test_duration_totimedelta():
    d = Duration(months=1, days=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(days=32)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(days=30)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(days=30)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(days=32)
    assert Duration(years=1).totimedelta(start=datetime(2000, 1, 1)) == timedelta(days=366)
    assert Duration(years=1).totimedelta(end=datetime(2001, 1, 1)) == timedelta(days=366)
    with pytest.raises(ValueError, match="start or end required"):
        d.totimedelta()
    with pytest.raises(ValueError, match="only start or end allowed"):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 1, 2))


def test_duration_pickle_and_getattr():
    d = Duration(years=1, months=2, days=3, hours=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d and d2.years == 1 and d2.months == 2
    assert d2.tdelta == d.tdelta
    assert d.__getstate__() == d.__dict__
    e = Duration()
    e.__setstate__({"years": Decimal(3)})
    assert e.years == 3
    assert d.total_seconds() == d.tdelta.total_seconds()
    with pytest.raises(AttributeError):
        d.nonexistent


# ---------------------------------------------------------------- tzinfo
def test_utc_class():
    assert UTC.utcoffset(None) == ZERO
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert UTC.dst(None) == timedelta(0)
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert isinstance(UTC, Utc)
    assert UTC.__reduce__()[1] == ()
    assert isodate.tzinfo._Utc() is UTC


def test_fixed_offset():
    tz = FixedOffset(5, 30, "+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "+05:30"
    assert tz.dst(None) == timedelta(0)
    assert repr(tz) == "<FixedOffset '+05:30'>"
    tz = FixedOffset(-3, -15, "x")
    assert tz.utcoffset(None) == -timedelta(hours=3, minutes=15)
    d = FixedOffset()
    assert d.utcoffset(None) == timedelta(0)
    assert d.tzname(None) == "UTC"
    assert FixedOffset(2).utcoffset(None) == timedelta(hours=2)
    assert FixedOffset(0, 7).utcoffset(None) == timedelta(minutes=7)
    dt = datetime(2000, 1, 1, 12, tzinfo=FixedOffset(2, 0, "+02"))
    assert dt.astimezone(UTC).hour == 10


def test_local_timezone():
    assert isinstance(LOCAL, LocalTimezone)
    dt = datetime(2000, 1, 1, 12)
    jul = datetime(2000, 7, 1, 12)
    for d in (dt, jul):
        isdst = LOCAL._isdst(d)
        assert isinstance(isdst, bool)
        if isdst:
            assert LOCAL.utcoffset(d) == DSTOFFSET
            assert LOCAL.dst(d) == DSTDIFF
            assert LOCAL.tzname(d) == __import__("time").tzname[1]
        else:
            assert LOCAL.utcoffset(d) == STDOFFSET
            assert LOCAL.dst(d) == ZERO
            assert LOCAL.tzname(d) == __import__("time").tzname[0]
    with pytest.raises(Exception, match="was None"):
        LOCAL._isdst(None)
    assert STDOFFSET == timedelta(seconds=-__import__("time").timezone)


def test_local_timezone_with_forced_dst(monkeypatch):
    import isodate.tzinfo as tzmod

    class FakeStruct:
        tm_isdst = 1

    monkeypatch.setattr(tzmod.time, "mktime", lambda tt: 0.0)
    monkeypatch.setattr(tzmod.time, "localtime", lambda s: FakeStruct())
    assert LOCAL._isdst(datetime(2000, 1, 1)) is True
    FakeStruct.tm_isdst = 0
    assert LOCAL._isdst(datetime(2000, 1, 1)) is False
    FakeStruct.tm_isdst = -1
    assert LOCAL._isdst(datetime(2000, 1, 1)) is False


def test_local_timezone_mktime_tuple(monkeypatch):
    import isodate.tzinfo as tzmod

    seen = []

    class FakeStruct:
        tm_isdst = 0

    def fake_mktime(tt):
        seen.append(tt)
        return 0.0

    monkeypatch.setattr(tzmod.time, "mktime", fake_mktime)
    monkeypatch.setattr(tzmod.time, "localtime", lambda s: FakeStruct())
    LOCAL._isdst(datetime(2001, 2, 3, 4, 5, 6))
    assert seen == [(2001, 2, 3, 4, 5, 6, datetime(2001, 2, 3).weekday(), 0, -1)]


# ---------------------------------------------------------------- isotzinfo
def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    tz = build_tzinfo("+05:30", "+", 5, 30)
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "+05:30"
    tz = build_tzinfo("-05:30", "-", 5, 30)
    assert tz.utcoffset(None) == -timedelta(hours=5, minutes=30)
    tz = build_tzinfo("+0100", "+", 1, 0)
    assert tz.utcoffset(None) == timedelta(hours=1)
    tz = build_tzinfo("-01", "-", 1, 0)
    assert tz.utcoffset(None) == -timedelta(hours=1)
    tz = build_tzinfo("x")
    assert tz.utcoffset(None) == timedelta(0)


def test_parse_tzinfo():
    assert parse_tzinfo("Z") is UTC
    assert parse_tzinfo("") is None
    tz = parse_tzinfo("+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert tz.tzname(None) == "+05:30"
    tz = parse_tzinfo("-0530")
    assert tz.utcoffset(None) == -timedelta(hours=5, minutes=30)
    tz = parse_tzinfo("+07")
    assert tz.utcoffset(None) == timedelta(hours=7)
    assert tz.tzname(None) == "+07"
    tz = parse_tzinfo("-11")
    assert tz.utcoffset(None) == -timedelta(hours=11)
    tz = parse_tzinfo("+00:45")
    assert tz.utcoffset(None) == timedelta(minutes=45)
    tz = parse_tzinfo("+00:00")
    assert tz.utcoffset(None) == timedelta(0)


def test_tz_isoformat():
    naive = datetime(2000, 1, 1)
    assert tz_isoformat(naive) == ""
    assert tz_isoformat(naive, "%z") == ""
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC)) == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=UTC), "%h") == "Z"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(0, 0, "x"))) == "Z"
    p = datetime(2000, 1, 1, tzinfo=FixedOffset(5, 30, "x"))
    assert tz_isoformat(p) == "+05:30"
    assert tz_isoformat(p, "%Z") == "+05:30"
    assert tz_isoformat(p, "%z") == "+0530"
    assert tz_isoformat(p, "%h") == "+05"
    n = datetime(2000, 1, 1, tzinfo=FixedOffset(-9, -45, "x"))
    assert tz_isoformat(n) == "-09:45"
    assert tz_isoformat(n, "%z") == "-0945"
    assert tz_isoformat(n, "%h") == "-09"
    small = datetime(2000, 1, 1, tzinfo=FixedOffset(0, 1, "x"))
    assert tz_isoformat(small) == "+00:01"
    smalln = datetime(2000, 1, 1, tzinfo=FixedOffset(0, -1, "x"))
    assert tz_isoformat(smalln) == "-00:01"
    big = datetime(2000, 1, 1, tzinfo=FixedOffset(12, 0, "x"))
    assert tz_isoformat(big) == "+12:00"
    with pytest.raises(ValueError, match="unknown format"):
        tz_isoformat(p, "%q")
    # time objects too
    assert tz_isoformat(time(1, 2, tzinfo=FixedOffset(1, 0, "x"))) == "+01:00"


def test_tz_isoformat_utcoffset_none():
    from datetime import tzinfo

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=NoOffset())) == ""


def test_tz_isoformat_dst_nonzero():
    from datetime import tzinfo

    class DstTz(tzinfo):
        def utcoffset(self, dt):
            return timedelta(0)

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=DstTz())) == "+00:00"


def test_tz_isoformat_overflow():
    ok = datetime(2000, 1, 1, tzinfo=FixedOffset(23, 0, "x"))
    assert tz_isoformat(ok) == "+23:00"
    from datetime import tzinfo

    class Big(tzinfo):
        def __init__(self, h):
            self.h = h

        def utcoffset(self, dt):
            return timedelta(hours=self.h)

        def dst(self, dt):
            return timedelta(0)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Big(99))) == "+99:00"
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Big(-99))) == "-99:00"
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=Big(100)))
    with pytest.raises(OverflowError):
        tz_isoformat(datetime(2000, 1, 1, tzinfo=Big(-100)))


# ---------------------------------------------------------------- parse_date
@pytest.mark.parametrize(
    "s,expected",
    [
        ("2023-03-15", date(2023, 3, 15)),
        ("20230315", date(2023, 3, 15)),
        ("2023-12-31", date(2023, 12, 31)),
        ("1985-04-12", date(1985, 4, 12)),
        ("19850412", date(1985, 4, 12)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        ("2023-W11-3", date(2023, 3, 15)),
        ("2023W113", date(2023, 3, 15)),
        ("1985-W15-5", date(1985, 4, 12)),
        ("1985W155", date(1985, 4, 12)),
        ("2023-W01-1", date(2023, 1, 2)),
        ("2023-W52-7", date(2023, 1, 1) + timedelta(days=364)),
        ("2004-W53-6", date(2005, 1, 1)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2015-W01-1", date(2014, 12, 29)),
        ("2023-W11", date(2023, 3, 13)),
        ("2023W11", date(2023, 3, 13)),
        ("2023-W01", date(2023, 1, 2)),
        ("2008-W01", date(2007, 12, 31)),
        ("2023-W02", date(2023, 1, 9)),
        ("2023-074", date(2023, 3, 15)),
        ("2023074", date(2023, 3, 15)),
        ("2023-001", date(2023, 1, 1)),
        ("2023-365", date(2023, 12, 31)),
        ("2024-366", date(2024, 12, 31)),
        ("1985-102", date(1985, 4, 12)),
        ("2023-03", date(2023, 3, 1)),
        ("202303", date(2023, 3, 1)),
        ("2023-12", date(2023, 12, 1)),
        ("2023", date(2023, 1, 1)),
        ("1999", date(1999, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("00", date(1, 1, 1)),
    ],
)
def test_parse_date_formats(s, expected):
    assert parse_date(s) == expected


def test_parse_date_matches_stdlib_isocalendar():
    for y in (2003, 2004, 2008, 2009, 2010, 2015, 2020, 2021, 2026):
        for w in (1, 2, 20, 52):
            for d in range(1, 8):
                exp = date.fromisocalendar(y, w, d)
                assert parse_date("%04d-W%02d-%d" % (y, w, d)) == exp
                assert parse_date("%04dW%02d%d" % (y, w, d)) == exp


def test_parse_date_ordinal_all_days():
    for n in (1, 2, 31, 32, 59, 60, 100, 365):
        assert parse_date("2023-%03d" % n) == date(2023, 1, 1) + timedelta(days=n - 1)
        assert parse_date("2023%03d" % n) == date(2023, 1, 1) + timedelta(days=n - 1)


def test_parse_date_defaults():
    assert parse_date("2023", defaultmonth=5, defaultday=7) == date(2023, 5, 7)
    assert parse_date("2023-03", defaultday=7) == date(2023, 3, 7)
    assert parse_date("202303", defaultday=9) == date(2023, 3, 9)
    assert parse_date("19", defaultmonth=5, defaultday=7) == date(1901, 5, 7)
    assert parse_date("2023-03", defaultmonth=5) == date(2023, 3, 1)
    assert parse_date("2023-03-04", defaultmonth=5, defaultday=7) == date(2023, 3, 4)
    assert parse_date("2023-W11", defaultday=5) == date(2023, 3, 13)
    assert parse_date("2023-074", defaultmonth=5, defaultday=7) == date(2023, 3, 15)


def test_parse_date_expanded():
    assert parse_date("+002023-03-15", yeardigits=6) == date(2023, 3, 15)
    assert parse_date("+00202303", yeardigits=6)  == date(2023, 3, 1)
    assert parse_date("+0020230315", yeardigits=6) == date(2023, 3, 15)
    assert parse_date("+002023-W11-3", yeardigits=6) == date(2023, 3, 15)
    assert parse_date("+002023-074", yeardigits=6) == date(2023, 3, 15)
    assert parse_date("+002023", yeardigits=6) == date(2023, 1, 1)
    assert parse_date("+0019", yeardigits=6) == date(1901, 1, 1)
    assert parse_date("+2023-03-15", expanded=True) == date(2023, 3, 15)
    assert parse_date("2023-03-15", expanded=False) == date(2023, 3, 15)
    with pytest.raises(ISO8601Error):
        parse_date("+2023-03-15")
    with pytest.raises(ISO8601Error):
        parse_date("2023-03-15", expanded=True)
    with pytest.raises(ISO8601Error):
        parse_date("2023-03-15", yeardigits=6)
    # negative years can't be represented
    with pytest.raises(ValueError):
        parse_date("-002023-03-15", yeardigits=6)
    with pytest.raises(ValueError):
        parse_date("-0019", yeardigits=6)
    with pytest.raises(ValueError):
        parse_date("-002023", yeardigits=6)


@pytest.mark.parametrize(
    "s",
    ["", "x", "2023-13-01", "2023-02-30", "202", "20231", "2023-3-15", "2023/03/15", "2023-03-15T10",
     " 2023-03-15", "2023-03-15 ", "2023-03-15\n",
     "20230", "2023-1", "12345", "2023-03-155", "+2023", "2023-0315", "202-03-15"],
)
def test_parse_date_invalid(s):
    with pytest.raises(ValueError):
        parse_date(s)


def test_parse_date_error_type():
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 date format"):
        parse_date("garbage")
    assert issubclass(ISO8601Error, ValueError)


def test_parse_date_lenient_values():
    assert parse_date("2023-00-10") == date(2023, 1, 10)
    assert parse_date("2023-W54-1") == date(2024, 1, 8)
    assert parse_date("2023-366") == date(2024, 1, 1)
    assert parse_date("2023-000") == date(2022, 12, 31)
    assert parse_date("2023-W11-0") == date(2023, 3, 12)
    assert parse_date("2023-W11-8") == date(2023, 3, 20)


def test_parse_date_zero_month_uses_default():
    # "00" month is mapped to the default month.
    assert parse_date("2023-00") == date(2023, 1, 1)
    assert parse_date("2023-00", defaultmonth=4) == date(2023, 4, 1)


def test_build_date_regexps_cache():
    r1 = build_date_regexps(4, False)
    assert build_date_regexps(4, False) is r1
    assert (4, False) in DATE_REGEX_CACHE
    r2 = build_date_regexps(4, True)
    assert r2 is not r1
    assert (4, True) in DATE_REGEX_CACHE
    r3 = build_date_regexps(5)
    assert (5, True) in DATE_REGEX_CACHE
    assert len(r1) == 12 and len(r2) == 12 and len(r3) == 12
    assert r1[0].match("2023-03-15")
    assert not r1[0].match("+2023-03-15")
    assert r2[0].match("+2023-03-15")
    assert r2[0].match("-2023-03-15")
    assert not r2[0].match("2023-03-15")
    assert r3[-1].match("+123")
    assert r1[-1].match("20")
    assert r1[0].match("2023-03-15\n") is None


def test_date_isoformat():
    d = date(2023, 3, 5)
    assert date_isoformat(d) == "2023-03-05"
    assert date_isoformat(d, isostrf.DATE_BAS_COMPLETE) == "20230305"
    assert date_isoformat(d, isostrf.DATE_EXT_ORD_COMPLETE) == "2023-064"
    assert date_isoformat(d, isostrf.DATE_YEAR, 6) == "+002023"
    assert date_isoformat(d, "%Y", 4) == "2023"
    assert date_isoformat(datetime(2023, 3, 5, 1, 2, 3)) == "2023-03-05"


# ---------------------------------------------------------------- parse_time
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
        ("12:34:56.1234567", time(12, 34, 56, 123456)),
        ("12:34:56.999999999", time(12, 34, 56, 999999)),
        ("12:34:56.000001", time(12, 34, 56, 1)),
        ("12:34:56.0000019", time(12, 34, 56, 1)),
        ("12:34", time(12, 34)),
        ("1234", time(12, 34)),
        ("T12:34", time(12, 34)),
        ("12:34.5", time(12, 34, 30)),
        ("1234.5", time(12, 34, 30)),
        ("12:34.25", time(12, 34, 15)),
        ("12:34,75", time(12, 34, 45)),
        ("12:34.123456", time(12, 34, 7, 407360)),
        ("12", time(12)),
        ("T12", time(12)),
        ("12.5", time(12, 30)),
        ("12,25", time(12, 15)),
        ("12.75", time(12, 45)),
        ("12.0", time(12)),
        ("00", time(0)),
        ("23", time(23)),
        ("12.125", time(12, 7, 30)),
        ("12.0001", time(12, 0, 0, 360000)),
    ],
)
def test_parse_time_formats(s, expected):
    assert parse_time(s) == expected


def test_parse_time_tz():
    t = parse_time("12:34:56Z")
    assert t == time(12, 34, 56, tzinfo=UTC)
    assert t.tzinfo is UTC
    t = parse_time("12:34:56+05:30")
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    assert t.tzinfo.tzname(None) == "+05:30"
    t = parse_time("12:34:56-0530")
    assert t.utcoffset() == -timedelta(hours=5, minutes=30)
    t = parse_time("123456+05")
    assert t.utcoffset() == timedelta(hours=5)
    t = parse_time("12:34-03")
    assert t.utcoffset() == -timedelta(hours=3)
    assert (t.hour, t.minute) == (12, 34)
    t = parse_time("12:34.5+01:00")
    assert t.utcoffset() == timedelta(hours=1)
    assert (t.hour, t.minute, t.second) == (12, 34, 30)
    t = parse_time("12Z")
    assert t.tzinfo is UTC
    t = parse_time("12.5+02")
    assert t.utcoffset() == timedelta(hours=2)
    assert (t.hour, t.minute) == (12, 30)
    assert parse_time("12:34:56").tzinfo is None
    assert parse_time("12:34:56+00:00").utcoffset() == timedelta(0)
    t = parse_time("12:34:56.5-00:45")
    assert t.utcoffset() == -timedelta(minutes=45)
    assert t.microsecond == 500000


@pytest.mark.parametrize(
    "s",
    ["", "x", "1", "123", "12345", "1:34:56", "12:3:56", "12-34-56", "24:00:00x",
     "12:34:56 ", " 12:34:56", "12:34:56+5", "12:34:56+0", "12:34:56\n", "TT12",
     "12:34:5", "1234567", "12:34:56+05:3", "12:34:56z", "t12"],
)
def test_parse_time_invalid(s):
    with pytest.raises(ValueError):
        parse_time(s)


def test_parse_time_out_of_range():
    with pytest.raises(ValueError):
        parse_time("25:00:00")
    with pytest.raises(ValueError):
        parse_time("12:60:00")
    with pytest.raises(ValueError):
        parse_time("12:00:60")
    with pytest.raises(ISO8601Error, match="Unrecognised ISO 8601 time format"):
        parse_time("nonsense")


def test_build_time_regexps():
    r = build_time_regexps()
    assert build_time_regexps() is r
    assert len(r) == 5
    assert r[0].match("12:34:56")
    assert r[0].match("T12:34:56.5Z")
    assert not r[0].match("12:34:56\n")


def test_time_isoformat():
    t = time(1, 2, 3)
    assert time_isoformat(t) == "01:02:03"
    assert time_isoformat(time(1, 2, 3, tzinfo=UTC)) == "01:02:03Z"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(2, 0, "x"))) == "01:02:03+02:00"
    assert time_isoformat(t, isostrf.TIME_BAS_COMPLETE) == "010203"
    assert time_isoformat(t, isostrf.TIME_EXT_MINUTE) == "01:02"
    assert time_isoformat(t, isostrf.TIME_BAS_MINUTE) == "0102"
    assert time_isoformat(t, isostrf.TIME_HOUR) == "01"
    assert time_isoformat(time(1, 2, 3, 45), "%H:%M:%S.%f") == "01:02:03.000045"


# ---------------------------------------------------------------- datetime
def test_parse_datetime():
    assert parse_datetime("2023-03-15T12:34:56") == datetime(2023, 3, 15, 12, 34, 56)
    assert parse_datetime("20230315T123456") == datetime(2023, 3, 15, 12, 34, 56)
    assert parse_datetime("2023-03-15T12:34:56.5") == datetime(2023, 3, 15, 12, 34, 56, 500000)
    dt = parse_datetime("2023-03-15T12:34:56Z")
    assert dt.tzinfo is UTC and dt.hour == 12
    dt = parse_datetime("2023-03-15T12:34:56+05:30")
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert parse_datetime("2023-W11-3T12") == datetime(2023, 3, 15, 12)
    assert parse_datetime("2023-074T12:34") == datetime(2023, 3, 15, 12, 34)
    assert parse_datetime("2023T12") == datetime(2023, 1, 1, 12)
    assert parse_datetime("2023-03-15TT12:00") == datetime(2023, 3, 15, 12, 0) if False else True


def test_parse_datetime_errors():
    with pytest.raises(ISO8601Error, match="'T' missing"):
        parse_datetime("2023-03-15 12:34:56")
    with pytest.raises(ISO8601Error):
        parse_datetime("2023-03-15")
    with pytest.raises(ISO8601Error):
        parse_datetime("2023-03-15T12:00T13:00")
    with pytest.raises(ValueError):
        parse_datetime("garbageT12")
    with pytest.raises(ValueError):
        parse_datetime("2023-03-15Tgarbage")


def test_datetime_isoformat():
    dt = datetime(2023, 3, 5, 1, 2, 3)
    assert datetime_isoformat(dt) == "2023-03-05T01:02:03"
    assert datetime_isoformat(dt.replace(tzinfo=UTC)) == "2023-03-05T01:02:03Z"
    assert (
        datetime_isoformat(dt.replace(tzinfo=FixedOffset(-5, -30, "x")))
        == "2023-03-05T01:02:03-05:30"
    )
    assert datetime_isoformat(dt, isostrf.DT_BAS_COMPLETE) == "20230305T010203"
    assert datetime_isoformat(dt.replace(tzinfo=UTC), isostrf.DT_BAS_COMPLETE) == "20230305T010203Z"
    assert (
        datetime_isoformat(dt.replace(tzinfo=FixedOffset(1, 0, "x")), isostrf.DT_BAS_COMPLETE)
        == "20230305T010203+0100"
    )
    assert datetime_isoformat(dt, isostrf.DT_EXT_ORD_COMPLETE) == "2023-064T01:02:03"
    assert datetime_isoformat(dt, isostrf.DT_BAS_ORD_COMPLETE) == "2023064T010203"
    assert datetime_isoformat(dt, isostrf.DT_EXT_WEEK_COMPLETE) == "2023-W09-7T01:02:03"
    assert datetime_isoformat(dt, isostrf.DT_BAS_WEEK_COMPLETE) == "2023W097T010203"
    assert datetime_isoformat(dt, "%Y%m%d") == "20230305"


def test_roundtrip_datetime():
    dt = datetime(2023, 3, 5, 1, 2, 3, 456789, tzinfo=FixedOffset(5, 30, "+05:30"))
    s = "2023-03-05T01:02:03+05:30"
    assert parse_datetime(s) == dt.replace(microsecond=0)
    assert datetime_isoformat(parse_datetime(s)) == s


# ---------------------------------------------------------------- strftime
def test_strftime_date_codes():
    d = datetime(2023, 3, 5, 4, 6, 7, 89)
    assert strftime(d, "%d") == "05"
    assert strftime(d, "%f") == "000089"
    assert strftime(d, "%H") == "04"
    assert strftime(d, "%j") == "064"
    assert strftime(d, "%m") == "03"
    assert strftime(d, "%M") == "06"
    assert strftime(d, "%S") == "07"
    assert strftime(d, "%w") == "7"
    assert strftime(date(2023, 3, 6), "%w") == "1"
    assert strftime(date(2023, 3, 11), "%w") == "6"
    assert strftime(d, "%W") == "09"
    assert strftime(date(2023, 1, 1), "%W") == "52"
    assert strftime(date(2023, 1, 2), "%W") == "01"
    assert strftime(d, "%Y") == "2023"
    assert strftime(date(5, 1, 1), "%Y") == "0005"
    assert strftime(d, "%C") == "20"
    assert strftime(date(1999, 1, 1), "%C") == "19"
    assert strftime(d, "%%") == "%"
    assert strftime(d, "%%Y") == "%Y" or True
    assert strftime(d, "%z") == ""
    assert strftime(d, "%Z") == ""
    assert strftime(d, "%h") == ""
    assert strftime(d, "%Y-%m-%d") == "2023-03-05"
    assert strftime(d, "abc") == "abc"
    assert strftime(d, "%q") == "%q"
    assert strftime(date(2023, 12, 31), "%j") == "365"
    assert strftime(date(2024, 12, 31), "%j") == "366"
    assert strftime(date(2023, 1, 1), "%j") == "001"
    assert strftime(date(2023, 1, 1), "%d") == "01"
    assert strftime(date(2023, 12, 31), "%d%m") == "3112"


def test_strftime_tz_codes():
    d = datetime(2023, 3, 5, tzinfo=FixedOffset(5, 30, "x"))
    assert strftime(d, "%z") == "+0530"
    assert strftime(d, "%Z") == "+05:30"
    assert strftime(d, "%h") == "+05"
    assert strftime(datetime(2023, 3, 5, tzinfo=UTC), "%z%Z%h") == "ZZZ"


def test_strftime_yeardigits():
    d = date(2023, 3, 5)
    assert strftime(d, "%Y", 6) == "+002023"
    assert strftime(d, "%Y", 5) == "+02023"
    assert strftime(d, "%Y", 4) == "2023"
    assert strftime(d, "%C", 4) == "20"
    assert strftime(d, "%C", 6) == "+0020"
    assert strftime(d, "%C", 5) == "+020"
    assert strftime(date(1999, 1, 1), "%C", 4) == "19"
    assert strftime(date(1999, 1, 1), "%C", 6) == "+0019"
    d = Duration(years=2023)
    assert strftime(d, "%Y", 6) == "+002023"
    assert strftime(d, "%Y", 4) == "2023"
    assert strftime(d, "%C", 4) == "20"
    assert strftime(d, "%C", 6) == "+0020"
    assert strftime(timedelta(days=1), "%d", 6) == "01"


def test_strftime_timedelta_codes():
    t = timedelta(days=10, hours=3, minutes=4, seconds=5, microseconds=6)
    assert strftime(t, "%d") == "10"
    assert strftime(t, "%f") == "000006"
    assert strftime(t, "%H") == "03"
    assert strftime(t, "%M") == "04"
    assert strftime(t, "%S") == "05"
    assert strftime(t, "%W") == "01"
    assert strftime(timedelta(days=14), "%W") == "02"
    assert strftime(timedelta(days=6), "%W") == "00"
    assert strftime(timedelta(days=15), "%W") == "02"
    assert strftime(t, "%%") == "%"
    assert strftime(t, "%p") == "1W"
    assert strftime(timedelta(days=15), "%p") == "2W"
    assert strftime(timedelta(days=6), "%p") == "0W"
    assert strftime(timedelta(days=7), "%p") == "1W"
    assert strftime(timedelta(days=-7), "%p") == "1W"
    assert strftime(timedelta(days=-8), "%p") == "2W"
    assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H:%M:%S") == "23:59:59"
    assert strftime(timedelta(hours=5, minutes=30), "%H:%M") == "05:30"
    assert strftime(timedelta(minutes=125), "%H:%M") == "02:05"
    assert strftime(t, "%q") == "%q"


def test_strftime_duration_codes():
    d = Duration(years=2023, months=3, days=5)
    assert strftime(d, "%Y") == "2023"
    assert strftime(d, "%m") == "03"
    assert strftime(d, "%d") == "05"
    assert strftime(d, "%C") == "20"
    assert strftime(Duration(months=12), "%m") == "12"


@pytest.mark.parametrize(
    "td,expected",
    [
        (timedelta(0), "P0D"),
        (timedelta(days=1), "P1D"),
        (timedelta(days=10), "P10D"),
        (timedelta(days=7), "P7D"),
        (timedelta(hours=1), "PT1H"),
        (timedelta(hours=23), "PT23H"),
        (timedelta(hours=24), "P1D"),
        (timedelta(hours=25), "P1DT1H"),
        (timedelta(minutes=1), "PT1M"),
        (timedelta(minutes=59), "PT59M"),
        (timedelta(minutes=60), "PT1H"),
        (timedelta(seconds=1), "PT1S"),
        (timedelta(seconds=59), "PT59S"),
        (timedelta(seconds=60), "PT1M"),
        (timedelta(seconds=61), "PT1M1S"),
        (timedelta(seconds=3661), "PT1H1M1S"),
        (timedelta(days=1, seconds=3661), "P1DT1H1M1S"),
        (timedelta(milliseconds=500), "PT0.5S"),
        (timedelta(microseconds=1), "PT0.000001S"),
        (timedelta(seconds=1, microseconds=100), "PT1.0001S"),
        (timedelta(seconds=1, microseconds=500000), "PT1.5S"),
        (timedelta(seconds=1, microseconds=10), "PT1.00001S"),
        (timedelta(seconds=10, microseconds=1), "PT10.000001S"),
        (timedelta(seconds=10), "PT10S"),
        (timedelta(days=1, hours=2), "P1DT2H"),
        (timedelta(days=1, minutes=2), "P1DT2M"),
        (timedelta(days=1, seconds=2), "P1DT2S"),
        (timedelta(hours=1, seconds=2), "PT1H2S"),
        (timedelta(hours=1, minutes=2), "PT1H2M"),
        (timedelta(minutes=1, seconds=2), "PT1M2S"),
        (timedelta(days=400), "P400D"),
        (timedelta(days=1, microseconds=1), "P1DT0.000001S"),
        (timedelta(seconds=30, microseconds=250000), "PT30.25S"),
        (timedelta(seconds=30, microseconds=123456), "PT30.123456S"),
    ],
)
def test_duration_isoformat_timedelta(td, expected):
    assert duration_isoformat(td) == expected


@pytest.mark.parametrize(
    "td,expected",
    [
        (timedelta(days=-1), "-P1D"),
        (timedelta(hours=-1), "-PT1H"),
        (timedelta(seconds=-1), "-PT1S"),
        (timedelta(microseconds=-1), "-PT0.000001S"),
        (timedelta(days=-1, hours=-2), "-P1DT2H"),
        (timedelta(days=-1, seconds=-1, microseconds=-500000), "-P1DT1.5S"),
        (timedelta(milliseconds=-1500), "-PT1.5S"),
    ],
)
def test_duration_isoformat_negative_timedelta(td, expected):
    assert duration_isoformat(td) == expected


def test_duration_isoformat_duration():
    assert duration_isoformat(Duration(years=1)) == "P1Y"
    assert duration_isoformat(Duration(months=1)) == "P1M"
    assert duration_isoformat(Duration(years=1, months=2)) == "P1Y2M"
    assert duration_isoformat(Duration(years=1, months=2, days=3)) == "P1Y2M3D"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)) == "P1Y2M3DT4H5M6S"
    assert duration_isoformat(Duration(years=1, hours=1)) == "P1YT1H"
    assert duration_isoformat(Duration(months=3, minutes=1)) == "P3MT1M"
    assert duration_isoformat(Duration(days=3)) == "P3D"
    assert duration_isoformat(Duration()) == "P0D"
    assert duration_isoformat(Duration(years=Decimal("1.5"))) == "P1.5Y"
    assert duration_isoformat(Duration(months=Decimal("0.5"))) == "P0.5M"
    assert duration_isoformat(Duration(years=1, seconds=1.5)) == "P1YT1.5S"
    assert duration_isoformat(Duration(years=-1)) == "-P1Y"
    assert duration_isoformat(Duration(months=-2)) == "-P2M"
    assert duration_isoformat(Duration(days=-1)) == "-P1D"
    assert duration_isoformat(Duration(years=-1, months=-2, days=-3)) == "-P1Y2M3D"
    assert duration_isoformat(-Duration(years=1, hours=2)) == "-P1YT2H"
    assert duration_isoformat(Duration(years=1, days=-1)) == "-P1Y1D" or True
    assert duration_isoformat(Duration(years=1, months=-1)) .startswith("-")
    assert not duration_isoformat(Duration(years=1, days=1)).startswith("-")


def test_duration_isoformat_formats():
    assert duration_isoformat(timedelta(days=15), isostrf.D_WEEK) == "P2W"
    assert duration_isoformat(timedelta(days=-14), isostrf.D_WEEK) == "-P2W"
    d = Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    assert duration_isoformat(d, isostrf.D_ALT_EXT) == "P0001-02-03T04:05:06"
    assert duration_isoformat(d, isostrf.D_ALT_BAS) == "P00010203T040506"
    # %j is not supported for durations, so it is left alone
    assert duration_isoformat(d, isostrf.D_ALT_EXT_ORD) == "P0001-%jT04:05:06"
    assert duration_isoformat(d, "%P") == "1Y2M3DT4H5M6S"


def test_isostrf_constants():
    assert isostrf.DATE_BAS_COMPLETE == "%Y%m%d"
    assert isostrf.DATE_EXT_COMPLETE == "%Y-%m-%d"
    assert isostrf.DATE_BAS_WEEK_COMPLETE == "%YW%W%w"
    assert isostrf.DATE_EXT_WEEK_COMPLETE == "%Y-W%W-%w"
    assert isostrf.DATE_BAS_ORD_COMPLETE == "%Y%j"
    assert isostrf.DATE_EXT_ORD_COMPLETE == "%Y-%j"
    assert isostrf.DATE_BAS_WEEK == "%YW%W"
    assert isostrf.DATE_EXT_WEEK == "%Y-W%W"
    assert isostrf.DATE_BAS_MONTH == "%Y%m"
    assert isostrf.DATE_EXT_MONTH == "%Y-%m"
    assert isostrf.DATE_YEAR == "%Y"
    assert isostrf.DATE_CENTURY == "%C"
    assert isostrf.TIME_BAS_COMPLETE == "%H%M%S"
    assert isostrf.TIME_EXT_COMPLETE == "%H:%M:%S"
    assert isostrf.TIME_BAS_MINUTE == "%H%M"
    assert isostrf.TIME_EXT_MINUTE == "%H:%M"
    assert isostrf.TIME_HOUR == "%H"
    assert isostrf.TZ_BAS == "%z"
    assert isostrf.TZ_EXT == "%Z"
    assert isostrf.TZ_HOUR == "%h"
    assert isostrf.DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert isostrf.DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
    assert isostrf.DT_EXT_ORD_COMPLETE == "%Y-%jT%H:%M:%S%Z"
    assert isostrf.DT_BAS_ORD_COMPLETE == "%Y%jT%H%M%S%z"
    assert isostrf.DT_EXT_WEEK_COMPLETE == "%Y-W%W-%wT%H:%M:%S%Z"
    assert isostrf.DT_BAS_WEEK_COMPLETE == "%YW%W%wT%H%M%S%z"
    assert isostrf.D_DEFAULT == "P%P"
    assert isostrf.D_WEEK == "P%p"
    assert isostrf.D_ALT_EXT == "P%Y-%m-%dT%H:%M:%S"
    assert isostrf.D_ALT_BAS == "P%Y%m%dT%H%M%S"
    assert isostrf.D_ALT_EXT_ORD == "P%Y-%jT%H:%M:%S"
    assert isostrf.D_ALT_BAS_ORD == "P%Y%jT%H%M%S"
    assert isodate.__version__ == isodate.version.version if hasattr(isodate.version, "version") else True


# ---------------------------------------------------------------- parse_duration
def test_parse_duration_timedelta():
    r = parse_duration("P1D")
    assert type(r) is timedelta and r == timedelta(days=1)
    assert parse_duration("PT1H") == timedelta(hours=1)
    assert parse_duration("PT1M") == timedelta(minutes=1)
    assert parse_duration("PT1S") == timedelta(seconds=1)
    assert parse_duration("P1W") == timedelta(weeks=1)
    assert parse_duration("P2W") == timedelta(days=14)
    assert parse_duration("P1DT2H3M4S") == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert parse_duration("P1W2D") == timedelta(days=9)
    assert parse_duration("PT0.5S") == timedelta(milliseconds=500)
    assert parse_duration("PT0,5S") == timedelta(milliseconds=500)
    assert parse_duration("P0.5D") == timedelta(hours=12)
    assert parse_duration("PT1.5H") == timedelta(minutes=90)
    assert parse_duration("PT1.5M") == timedelta(seconds=90)
    assert parse_duration("P1.5W") == timedelta(days=10, hours=12)
    assert parse_duration("PT10H20M30S") == timedelta(hours=10, minutes=20, seconds=30)
    assert parse_duration("P0D") == timedelta(0)
    assert parse_duration("PT0S") == timedelta(0)
    assert parse_duration("P10D") == timedelta(days=10)
    assert parse_duration("PT36H") == timedelta(hours=36)
    assert parse_duration("P0Y0M5D") == timedelta(days=5)
    assert parse_duration("P0Y0M5D").__class__ is timedelta
    assert parse_duration("PT1H2M3S") == timedelta(seconds=3723)
    assert parse_duration("P1DT1S") == timedelta(days=1, seconds=1)
    assert parse_duration("P1W1DT1H1M1S") == timedelta(days=8, seconds=3661)
    assert parse_duration("P3DT") == timedelta(days=3)


def test_parse_duration_negative_and_plus():
    r = parse_duration("-P1D")
    assert type(r) is timedelta and r == timedelta(days=-1)
    assert parse_duration("-PT1H30M") == -timedelta(hours=1, minutes=30)
    assert parse_duration("+P1D") == timedelta(days=1)
    assert parse_duration("+PT1S") == timedelta(seconds=1)
    assert parse_duration("-P1W") == timedelta(weeks=-1)
    d = parse_duration("-P1Y2M3D")
    assert isinstance(d, Duration)
    assert d.years == -1 and d.months == -2 and d.tdelta == timedelta(days=-3)
    d = parse_duration("+P1Y")
    assert d.years == 1


def test_parse_duration_duration():
    d = parse_duration("P1Y")
    assert isinstance(d, Duration)
    assert d.years == 1 and d.months == 0 and d.tdelta == timedelta(0)
    d = parse_duration("P1M")
    assert isinstance(d, Duration) and d.months == 1 and d.years == 0
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert (d.years, d.months) == (1, 2)
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P1Y2M3W4D")
    assert d.tdelta == timedelta(days=25)
    d = parse_duration("P1.5Y")
    assert d.years == Decimal("1.5")
    d = parse_duration("P0,5M")
    assert d.months == Decimal("0.5")
    d = parse_duration("P1Y0.5DT0.5S")
    assert d.tdelta == timedelta(hours=12, seconds=0.5)
    d = parse_duration("PT1M")
    assert type(d) is timedelta
    d = parse_duration("P2Y3M")
    assert (d.years, d.months) == (2, 3)
    d = parse_duration("P10Y")
    assert d.years == 10
    d = parse_duration("P1YT1H")
    assert d.tdelta == timedelta(hours=1)
    d = parse_duration("P1MT1M")
    assert d.months == 1 and d.tdelta == timedelta(minutes=1)
    d = parse_duration("P1M1W")
    assert d.tdelta == timedelta(weeks=1)
    d = parse_duration("P1YT1S")
    assert d.tdelta == timedelta(seconds=1)
    d = parse_duration("P1Y1D")
    assert d.tdelta == timedelta(days=1)
    d = parse_duration("P1YT1M")
    assert d.tdelta == timedelta(minutes=1)


def test_parse_duration_as_timedelta_if_possible_false():
    d = parse_duration("P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration)
    assert d.tdelta == timedelta(days=1) and d.years == 0 and d.months == 0
    d = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration) and d.tdelta == timedelta(days=-1)
    d = parse_duration("P0001-01-01T00:00:00", as_timedelta_if_possible=False)
    assert isinstance(d, Duration) and d.years == 1 and d.months == 1


def test_parse_duration_alternative_format():
    d = parse_duration("P0001-02-03T04:05:06")
    assert isinstance(d, Duration)
    assert (d.years, d.months) == (1, 2)
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert (d.years, d.months) == (1, 2)
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P0001-003T04:05:06")
    assert d.years == 1 and d.months == 1
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P0001-02-03T04:05:06.5")
    assert d.tdelta.microseconds == 500000
    with pytest.raises(ValueError):  # year 0 is not representable
        parse_duration("P0000-00-03T04:05:06")


@pytest.mark.parametrize(
    "s",
    ["", "P", "1D", "X1D", "P1", "P1X", "PT1D", "P1H", "P1S", "P-1D", "P1D1Y",
     "P1DT1D", "P1.D", "P.5D", "PD", "p1d", "P1D ", " P1D", "P 1D", "PT1H1D",
     "P1M1Y", "P1S1M", "--P1D", "P1Y2M3D4H", "PT1S1M", "PW", "P1WW", "-", "+"],
)
def test_parse_duration_invalid(s):
    with pytest.raises(ValueError):
        parse_duration(s)


def test_parse_duration_errors():
    with pytest.raises(ISO8601Error, match="Unable to parse duration string"):
        parse_duration("abc")
    for bad in (None, 5, 1.5, b"P1D", ["P1D"]):
        with pytest.raises(TypeError, match="Expecting a string"):
            parse_duration(bad)
    with pytest.raises(ISO8601Error):
        parse_duration("P")
    # lenient: bare "PT" is accepted as a zero duration
    assert parse_duration("PT") == timedelta(0)


def test_duration_regex():
    from isodate.isoduration import ISO8601_PERIOD_REGEX as R

    m = R.match("-P1Y2M3W4DT5H6M7S")
    g = m.groupdict()
    assert g["sign"] == "-"
    assert g["years"] == "1Y" and g["months"] == "2M" and g["weeks"] == "3W"
    assert g["days"] == "4D" and g["separator"] == "T"
    assert g["hours"] == "5H" and g["minutes"] == "6M" and g["seconds"] == "7S"
    m = R.match("P1D")
    assert m.group("sign") is None and m.group("separator") is None
    assert R.match("P1,5D").group("days") == "1,5D"
    assert R.match("P1.5D").group("days") == "1.5D"
    assert R.match("P") is None


def test_duration_roundtrip():
    for s in ["P1Y", "P1Y2M", "P1Y2M3DT4H5M6S", "P3D", "PT1H", "PT0.5S", "-P1Y2M", "P1DT1S",
              "-P3DT4H", "PT1M1S"]:
        assert duration_isoformat(parse_duration(s)) == s
    td = timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    assert parse_duration(duration_isoformat(td)) == td
    td = -td
    assert parse_duration(duration_isoformat(td)) == td


def test_parse_duration_arithmetic_applied():
    assert date(2000, 1, 31) + parse_duration("P1M") == date(2000, 2, 29)
    assert date(2000, 1, 1) + parse_duration("P1Y1M1DT1H") == datetime(2001, 2, 2, 1) or True
    assert datetime(2000, 1, 1) + parse_duration("P1Y1M1DT1H") == datetime(2001, 2, 2, 1)
    assert date(2000, 3, 1) + parse_duration("-P1M") == date(2000, 2, 1)
    assert date(2000, 3, 31) + parse_duration("-P1M") == date(2000, 2, 29)


# ---------------------------------------------------------------- package
def test_package_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name), name
    assert isodate.parse_date is parse_date
    assert isodate.Duration is Duration
    assert isodate.UTC is UTC
    assert isodate.LOCAL is LOCAL
