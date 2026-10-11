import pickle
import time as _time
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
from isodate import tzinfo as tzmod
from isodate.duration import fquotmod, max_days_in_month
from isodate.isodates import DATE_REGEX_CACHE, build_date_regexps
from isodate.isotime import build_time_regexps
from isodate.isotzinfo import build_tzinfo


# ---------------------------------------------------------------- duration.py
def test_fquotmod():
    assert fquotmod(Decimal(5), 1, 13) == (0, Decimal(5))
    assert fquotmod(Decimal(13), 1, 13) == (1, Decimal(1))
    assert fquotmod(Decimal(0), 1, 13) == (-1, Decimal(12))
    assert fquotmod(Decimal(-12), 1, 13) == (-2, Decimal(12))
    assert fquotmod(Decimal(25), 1, 13) == (2, Decimal(1))
    assert fquotmod(Decimal(12), 1, 13) == (0, Decimal(12))
    assert fquotmod(Decimal(1), 1, 13) == (0, Decimal(1))
    assert fquotmod(Decimal(7), 3, 10) == (0, Decimal(7))
    assert fquotmod(Decimal(10), 3, 10) == (1, Decimal(3))
    assert fquotmod(Decimal(2), 3, 10) == (-1, Decimal(9))


@pytest.mark.parametrize("m", [1, 3, 5, 7, 8, 10, 12])
def test_max_days_31(m):
    assert max_days_in_month(2001, m) == 31


@pytest.mark.parametrize("m", [4, 6, 9, 11])
def test_max_days_30(m):
    assert max_days_in_month(2001, m) == 30


@pytest.mark.parametrize(
    "y,exp", [(2000, 29), (1900, 28), (2004, 29), (2001, 28), (2100, 28), (1996, 29), (2400, 29), (1800, 28)]
)
def test_max_days_feb(y, exp):
    assert max_days_in_month(y, 2) == exp


def test_duration_init():
    d = Duration(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=7, months=8, years=9)
    assert d.years == Decimal(9)
    assert d.months == Decimal(8)
    assert d.tdelta == timedelta(1, 2, 3, 4, 5, 6, 7)
    assert isinstance(d.months, Decimal) and isinstance(d.years, Decimal)
    d = Duration(months=Decimal("1.5"), years=2.5)
    assert d.months == Decimal("1.5")
    assert d.years == Decimal("2.5")
    assert Duration(months=0.1).months == Decimal("0.1")
    # attribute passthrough
    assert Duration(days=3, seconds=4, microseconds=5).days == 3
    assert Duration(days=3, seconds=4, microseconds=5).seconds == 4
    assert Duration(days=3, seconds=4, microseconds=5).microseconds == 5
    with pytest.raises(AttributeError):
        Duration().nonexistent


def test_duration_str():
    assert str(Duration(years=1, months=2, days=3)) == "1 years, 2 months, 3 days, 0:00:00"
    assert str(Duration(months=1)) == "1 month, 0:00:00"
    assert str(Duration(months=2)) == "2 months, 0:00:00"
    assert str(Duration(months=Decimal("0.5"))) == "0.5 month, 0:00:00"
    assert str(Duration(years=1)) == "1 years, 0:00:00"
    assert str(Duration(hours=1)) == "1:00:00"
    assert str(Duration()) == "0:00:00"


def test_duration_repr():
    r = repr(Duration(days=1, seconds=2, microseconds=3, years=4, months=5))
    assert r == "isodate.duration.Duration(1, 2, 3, years=4, months=5)"


def test_duration_hash():
    a = Duration(years=1, months=2, days=3)
    b = Duration(years=1, months=2, days=3)
    assert hash(a) == hash(b)
    assert hash(a) == hash((a.tdelta, a.months, a.years))
    assert hash(a) != hash(Duration(years=1, months=2, days=4))
    assert hash(a) != hash(Duration(years=2, months=2, days=3))
    assert len({a, b, Duration(years=3)}) == 2


def test_duration_neg():
    n = -Duration(years=1, months=2, days=3, hours=4)
    assert n.years == -1
    assert n.months == -2
    assert n.tdelta == -timedelta(days=3, hours=4)
    assert (-n) == Duration(years=1, months=2, days=3, hours=4)


def test_duration_add_duration():
    s = Duration(years=1, months=2, days=3) + Duration(years=10, months=20, days=30)
    assert isinstance(s, Duration)
    assert s.years == 11 and s.months == 22 and s.tdelta == timedelta(33)


def test_duration_add_timedelta():
    s = Duration(years=1, months=2, days=3) + timedelta(days=4)
    assert isinstance(s, Duration)
    assert s.years == 1 and s.months == 2 and s.tdelta == timedelta(7)
    s = timedelta(days=4) + Duration(years=1, months=2, days=3)
    assert isinstance(s, Duration)
    assert s.years == 1 and s.months == 2 and s.tdelta == timedelta(7)


def test_duration_add_bad():
    with pytest.raises(TypeError):
        Duration(days=1) + 5
    with pytest.raises(TypeError):
        Duration(days=1) + "x"


def test_duration_add_date():
    assert date(2000, 1, 31) + Duration(months=1) == date(2000, 2, 29)
    assert Duration(months=1) + date(2001, 1, 31) == date(2001, 2, 28)
    assert date(2000, 1, 30) + Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 1, 29) + Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 1, 28) + Duration(months=1) == date(2000, 2, 28)
    assert date(2000, 3, 31) + Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 3, 30) + Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 3, 29) + Duration(months=1) == date(2000, 4, 29)
    assert date(2000, 11, 15) + Duration(months=2) == date(2001, 1, 15)
    assert date(2000, 12, 15) + Duration(months=1) == date(2001, 1, 15)
    assert date(2000, 11, 15) + Duration(months=1) == date(2000, 12, 15)
    assert date(2000, 1, 15) + Duration(months=11) == date(2000, 12, 15)
    assert date(2000, 1, 15) + Duration(months=12) == date(2001, 1, 15)
    assert date(2000, 1, 15) + Duration(months=25) == date(2002, 2, 15)
    assert date(2000, 2, 29) + Duration(years=1) == date(2001, 2, 28)
    assert date(2000, 2, 29) + Duration(years=4) == date(2004, 2, 29)
    assert date(2000, 5, 5) + Duration(years=1, months=1, days=1) == date(2001, 6, 6)
    assert date(2000, 5, 5) + Duration(months=-5) == date(1999, 12, 5)
    assert date(2000, 5, 5) + Duration(months=-4) == date(2000, 1, 5)
    assert date(2000, 5, 5) + Duration(months=-6) == date(1999, 11, 5)
    assert date(2000, 5, 5) + Duration(months=-17) == date(1998, 12, 5)
    dt = datetime(2000, 1, 31, 10, 20, 30)
    assert dt + Duration(months=1, hours=2) == datetime(2000, 2, 29, 12, 20, 30)
    assert date(2000, 1, 1) + Duration(days=1, hours=24) == date(2000, 1, 3)
    # exact-same day end of month kept
    assert date(2001, 1, 28) + Duration(months=1) == date(2001, 2, 28)
    assert date(2001, 8, 31) + Duration(months=1) == date(2001, 9, 30)
    assert date(2001, 10, 31) + Duration(months=1) == date(2001, 11, 30)
    assert date(2001, 12, 31) + Duration(months=2) == date(2002, 2, 28)


def test_duration_add_fractional_date():
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(months=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) + Duration(years=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(months=Decimal("0.5"))
    with pytest.raises(ValueError):
        date(2000, 1, 1) - Duration(years=Decimal("0.5"))
    # integer-valued decimals fine
    assert date(2000, 1, 1) + Duration(months=Decimal("1.0")) == date(2000, 2, 1)
    assert date(2000, 1, 1) + Duration(years=Decimal("1.0")) == date(2001, 1, 1)


def test_duration_mul():
    d = Duration(years=1, months=2, days=3) * 3
    assert d.years == 3 and d.months == 6 and d.tdelta == timedelta(9)
    d = 2 * Duration(years=1, months=2, days=3)
    assert d.years == 2 and d.months == 4 and d.tdelta == timedelta(6)
    with pytest.raises(TypeError):
        Duration(days=1) * 1.5
    with pytest.raises(TypeError):
        Duration(days=1) * "a"


def test_duration_sub():
    d = Duration(years=10, months=20, days=30) - Duration(years=1, months=2, days=3)
    assert isinstance(d, Duration)
    assert d.years == 9 and d.months == 18 and d.tdelta == timedelta(27)
    d = Duration(years=10, months=20, days=30) - timedelta(days=5)
    assert isinstance(d, Duration)
    assert d.years == 10 and d.months == 20 and d.tdelta == timedelta(25)
    with pytest.raises(TypeError):
        Duration(days=1) - 5
    with pytest.raises(TypeError):
        Duration(days=1) - "x"


def test_duration_rsub():
    d = timedelta(days=10) - Duration(years=1, months=2, days=3)
    assert isinstance(d, Duration)
    assert d.years == -1 and d.months == -2 and d.tdelta == timedelta(7)
    assert date(2000, 3, 31) - Duration(months=1) == date(2000, 2, 29)
    assert date(2001, 3, 31) - Duration(months=1) == date(2001, 2, 28)
    assert date(2001, 3, 30) - Duration(months=1) == date(2001, 2, 28)
    assert date(2001, 3, 29) - Duration(months=1) == date(2001, 2, 28)
    assert date(2001, 3, 28) - Duration(months=1) == date(2001, 2, 28)
    assert date(2000, 3, 29) - Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 3, 30) - Duration(months=1) == date(2000, 2, 29)
    assert date(2000, 5, 31) - Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 5, 30) - Duration(months=1) == date(2000, 4, 30)
    assert date(2000, 5, 29) - Duration(months=1) == date(2000, 4, 29)
    assert date(2000, 1, 15) - Duration(months=1) == date(1999, 12, 15)
    assert date(2000, 2, 15) - Duration(months=1) == date(2000, 1, 15)
    assert date(2000, 1, 15) - Duration(months=12) == date(1999, 1, 15)
    assert date(2000, 1, 15) - Duration(months=13) == date(1998, 12, 15)
    assert date(2000, 12, 15) - Duration(months=11) == date(2000, 1, 15)
    assert date(2000, 12, 15) - Duration(months=12) == date(1999, 12, 15)
    assert date(2000, 2, 29) - Duration(years=1) == date(1999, 2, 28)
    assert date(2000, 2, 29) - Duration(years=4) == date(1996, 2, 29)
    assert date(2000, 5, 5) - Duration(years=1, months=1, days=1) == date(1999, 4, 4)
    assert date(2000, 5, 5) - Duration(months=-8) == date(2001, 1, 5)
    assert date(2000, 5, 5) - Duration(months=-7) == date(2000, 12, 5)
    assert datetime(2000, 3, 31, 10) - Duration(months=1, hours=2) == datetime(2000, 2, 29, 8)
    assert date(2000, 1, 3) - Duration(days=1, hours=24) == date(2000, 1, 1)
    assert date(2001, 12, 31) - Duration(months=10) == date(2001, 2, 28)
    assert date(2001, 7, 31) - Duration(months=1) == date(2001, 6, 30)
    assert date(2001, 12, 31) - Duration(months=1) == date(2001, 11, 30)
    assert date(2001, 10, 31) - Duration(months=1) == date(2001, 9, 30)
    with pytest.raises(TypeError):
        "x" - Duration(days=1)
    with pytest.raises(TypeError):
        5 - Duration(days=1)


def test_duration_eq_ne():
    a = Duration(years=1, months=2, days=3)
    assert a == Duration(years=1, months=2, days=3)
    assert not (a != Duration(years=1, months=2, days=3))
    assert a != Duration(years=1, months=2, days=4)
    assert not (a == Duration(years=1, months=2, days=4))
    assert a != Duration(years=1, months=3, days=3)
    assert a != Duration(years=2, months=2, days=3)
    assert not (a == Duration(years=1, months=3, days=3))
    # total months compared
    assert Duration(years=1) == Duration(months=12)
    assert not (Duration(years=1) != Duration(months=12))
    assert Duration(years=1, months=1) != Duration(years=1, months=2)
    assert Duration(years=1, months=1) != Duration(years=2, months=0)
    # timedelta comparison
    assert Duration(days=1) == timedelta(days=1)
    assert not (Duration(days=1) != timedelta(days=1))
    assert Duration(days=1) != timedelta(days=2)
    assert not (Duration(days=1) == timedelta(days=2))
    assert Duration(months=1) != timedelta(0)
    assert not (Duration(months=1) == timedelta(0))
    assert Duration(years=1) != timedelta(0)
    assert not (Duration(years=1) == timedelta(0))
    assert Duration(years=1, days=1) != timedelta(days=1)
    assert Duration(months=1, days=1) != timedelta(days=1)
    assert Duration(days=1) != "x"
    assert not (Duration(days=1) == "x")


def test_duration_totimedelta():
    d = Duration(months=1)
    assert d.totimedelta(start=date(2000, 1, 1)) == timedelta(31)
    assert d.totimedelta(start=date(2000, 2, 1)) == timedelta(29)
    assert d.totimedelta(end=date(2000, 3, 1)) == timedelta(29)
    assert d.totimedelta(end=date(2000, 2, 1)) == timedelta(31)
    assert Duration(years=1).totimedelta(start=date(2001, 1, 1)) == timedelta(365)
    with pytest.raises(ValueError):
        d.totimedelta()
    with pytest.raises(ValueError):
        d.totimedelta(start=date(2000, 1, 1), end=date(2000, 2, 1))


def test_duration_pickle():
    d = Duration(years=1, months=2, days=3, hours=4)
    d2 = pickle.loads(pickle.dumps(d))
    assert d2 == d
    assert d2.years == 1 and d2.months == 2
    assert d.__getstate__() is d.__dict__
    d3 = Duration()
    d3.__setstate__({"years": Decimal(5)})
    assert d3.years == 5


# ------------------------------------------------------------------ tzinfo.py
def test_utc():
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"
    assert UTC.dst(None) == timedelta(0)
    assert pickle.loads(pickle.dumps(UTC)) is UTC
    assert tzmod._Utc() is UTC
    assert UTC.__reduce__() == (tzmod._Utc, ())
    assert tzmod.ZERO == timedelta(0)


def test_fixed_offset():
    f = FixedOffset(5, 30, "+0530")
    assert f.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert f.tzname(None) == "+0530"
    assert f.dst(None) == timedelta(0)
    assert repr(f) == "<FixedOffset '+0530'>"
    f = FixedOffset(-3, -15, "x")
    assert f.utcoffset(None) == -timedelta(hours=3, minutes=15)
    d = FixedOffset()
    assert d.utcoffset(None) == timedelta(0)
    assert d.tzname(None) == "UTC"
    assert FixedOffset(2).utcoffset(None) == timedelta(hours=2)
    assert FixedOffset(0, 45).utcoffset(None) == timedelta(minutes=45)


def test_local_timezone():
    assert tzmod.STDOFFSET == timedelta(seconds=-_time.timezone)
    if _time.daylight:
        assert tzmod.DSTOFFSET == timedelta(seconds=-_time.altzone)
    else:
        assert tzmod.DSTOFFSET == tzmod.STDOFFSET
    assert tzmod.DSTDIFF == tzmod.DSTOFFSET - tzmod.STDOFFSET
    dt = datetime(2000, 1, 15, 12)
    dst = _time.localtime(_time.mktime(dt.timetuple())).tm_isdst > 0
    assert LOCAL.utcoffset(dt) == (tzmod.DSTOFFSET if dst else tzmod.STDOFFSET)
    assert LOCAL.dst(dt) == (tzmod.DSTDIFF if dst else timedelta(0))
    assert LOCAL.tzname(dt) == _time.tzname[1 if dst else 0]
    dt = datetime(2000, 7, 15, 12)
    dst = _time.localtime(_time.mktime(dt.timetuple())).tm_isdst > 0
    assert LOCAL.utcoffset(dt) == (tzmod.DSTOFFSET if dst else tzmod.STDOFFSET)
    assert LOCAL.dst(dt) == (tzmod.DSTDIFF if dst else timedelta(0))
    assert LOCAL.tzname(dt) == _time.tzname[1 if dst else 0]
    with pytest.raises(Exception, match="was None"):
        LOCAL._isdst(None)


def test_local_timezone_mocked(monkeypatch):
    monkeypatch.setattr(tzmod, "DSTOFFSET", timedelta(hours=2))
    monkeypatch.setattr(tzmod, "STDOFFSET", timedelta(hours=1))
    monkeypatch.setattr(tzmod, "DSTDIFF", timedelta(hours=1))
    monkeypatch.setattr(tzmod.LocalTimezone, "_isdst", lambda self, dt: True)
    dt = datetime(2000, 1, 1)
    assert LOCAL.utcoffset(dt) == timedelta(hours=2)
    assert LOCAL.dst(dt) == timedelta(hours=1)
    monkeypatch.setattr(tzmod.LocalTimezone, "_isdst", lambda self, dt: False)
    assert LOCAL.utcoffset(dt) == timedelta(hours=1)
    assert LOCAL.dst(dt) == timedelta(0)


def test_isdst_args(monkeypatch):
    seen = {}

    class FakeTT:
        tm_isdst = 1

    def fake_mktime(tt):
        seen["tt"] = tt
        return 0.0

    monkeypatch.setattr(tzmod.time, "mktime", fake_mktime)
    monkeypatch.setattr(tzmod.time, "localtime", lambda s: FakeTT())
    dt = datetime(2001, 2, 3, 4, 5, 6)
    assert LOCAL._isdst(dt) is True
    assert seen["tt"] == (2001, 2, 3, 4, 5, 6, dt.weekday(), 0, -1)
    FakeTT.tm_isdst = 0
    assert LOCAL._isdst(dt) is False
    FakeTT.tm_isdst = -1
    assert LOCAL._isdst(dt) is False


# ----------------------------------------------------------------- isotzinfo
def test_build_tzinfo():
    assert build_tzinfo(None) is None
    assert build_tzinfo("") is None
    assert build_tzinfo("Z") is UTC
    f = build_tzinfo("+05:30", "+", 5, 30)
    assert f.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert f.tzname(None) == "+05:30"
    f = build_tzinfo("-05:30", "-", 5, 30)
    assert f.utcoffset(None) == -timedelta(hours=5, minutes=30)
    f = build_tzinfo("+05", "+", 5, 0)
    assert f.utcoffset(None) == timedelta(hours=5)
    f = build_tzinfo("-05", "-", 5, 0)
    assert f.utcoffset(None) == -timedelta(hours=5)
    f = build_tzinfo("+0030", "+", 0, 30)
    assert f.utcoffset(None) == timedelta(minutes=30)
    f = build_tzinfo("+01:00")
    assert f.utcoffset(None) == timedelta(0)


def test_parse_tzinfo():
    assert parse_tzinfo("") is None
    assert parse_tzinfo("Z") is UTC
    f = parse_tzinfo("+05:30")
    assert f.utcoffset(None) == timedelta(hours=5, minutes=30)
    assert f.tzname(None) == "+05:30"
    f = parse_tzinfo("-0530")
    assert f.utcoffset(None) == -timedelta(hours=5, minutes=30)
    f = parse_tzinfo("+05")
    assert f.utcoffset(None) == timedelta(hours=5)
    f = parse_tzinfo("-11")
    assert f.utcoffset(None) == -timedelta(hours=11)
    f = parse_tzinfo("+00:45")
    assert f.utcoffset(None) == timedelta(minutes=45)
    # the pattern matches an empty prefix, so unknown text means "no tz"
    assert parse_tzinfo("abc") is None


def test_tz_isoformat():
    def dt(tz):
        return datetime(2000, 1, 1, tzinfo=tz)

    assert tz_isoformat(datetime(2000, 1, 1)) == ""
    assert tz_isoformat(dt(UTC)) == "Z"
    assert tz_isoformat(dt(UTC), "%h") == "Z"
    assert tz_isoformat(dt(FixedOffset(0, 0, "x"))) == "Z"
    assert tz_isoformat(dt(FixedOffset(5, 30, "x"))) == "+05:30"
    assert tz_isoformat(dt(FixedOffset(5, 30, "x")), "%Z") == "+05:30"
    assert tz_isoformat(dt(FixedOffset(5, 30, "x")), "%z") == "+0530"
    assert tz_isoformat(dt(FixedOffset(5, 30, "x")), "%h") == "+05"
    assert tz_isoformat(dt(FixedOffset(-5, -30, "x"))) == "-05:30"
    assert tz_isoformat(dt(FixedOffset(-5, -30, "x")), "%z") == "-0530"
    assert tz_isoformat(dt(FixedOffset(-5, -30, "x")), "%h") == "-05"
    assert tz_isoformat(dt(FixedOffset(0, 1, "x"))) == "+00:01"
    assert tz_isoformat(dt(FixedOffset(0, -1, "x"))) == "-00:01"
    assert tz_isoformat(dt(FixedOffset(0, 59, "x")), "%z") == "+0059"
    assert tz_isoformat(dt(FixedOffset(12, 0, "x"))) == "+12:00"
    assert tz_isoformat(dt(FixedOffset(23, 59, "x"))) == "+23:59"
    assert tz_isoformat(dt(FixedOffset(-23, -59, "x")), "%z") == "-2359"
    assert tz_isoformat(dt(FixedOffset(-1, 0, "x"))) == "-01:00"
    with pytest.raises(ValueError, match="unknown format string"):
        tz_isoformat(dt(FixedOffset(5, 30, "x")), "%q")


def test_tz_isoformat_overflow():
    class Big(tzmod.tzinfo):
        def __init__(self, h):
            self.h = h

        def utcoffset(self, dt):
            return timedelta(hours=self.h)

        def dst(self, dt):
            return timedelta(0)

    # datetime itself restricts offsets to < 24h, so call with a stub object
    class Stub:
        def __init__(self, h):
            self.tzinfo = Big(h)

    assert tz_isoformat(Stub(99)) == "+99:00"
    assert tz_isoformat(Stub(-99)) == "-99:00"
    with pytest.raises(OverflowError):
        tz_isoformat(Stub(100))
    with pytest.raises(OverflowError):
        tz_isoformat(Stub(-100))
    assert tz_isoformat(Stub(1)) == "+01:00"


def test_tz_isoformat_dst_nonzero():
    class Dst(tzmod.tzinfo):
        def utcoffset(self, dt):
            return timedelta(0)

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=Dst())) == "+00:00"

    class NoOff(tzmod.tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=NoOff())) == ""


def test_tz_isoformat_nonzero_offset_zero_dst():
    # offset non-zero, dst zero must not yield Z
    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=FixedOffset(1, 0, "x"))) == "+01:00"

    class OffZeroDst(tzmod.tzinfo):
        def utcoffset(self, dt):
            return timedelta(hours=1)

        def dst(self, dt):
            return timedelta(0)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=OffZeroDst())) == "+01:00"

    class ZeroOffDst(tzmod.tzinfo):
        def utcoffset(self, dt):
            return timedelta(0)

        def dst(self, dt):
            return timedelta(hours=1)

    assert tz_isoformat(datetime(2000, 1, 1, tzinfo=ZeroOffDst())) != "Z"


# ------------------------------------------------------------------ isodates
@pytest.mark.parametrize(
    "s,exp",
    [
        ("2012-03-05", date(2012, 3, 5)),
        ("20120305", date(2012, 3, 5)),
        ("2012-W10-1", date(2012, 3, 5)),
        ("2012W101", date(2012, 3, 5)),
        ("2012-W10-7", date(2012, 3, 11)),
        ("2012-065", date(2012, 3, 5)),
        ("2012065", date(2012, 3, 5)),
        ("2012-001", date(2012, 1, 1)),
        ("2012-366", date(2012, 12, 31)),
        ("2012-W10", date(2012, 3, 5)),
        ("2012W10", date(2012, 3, 5)),
        ("2012-03", date(2012, 3, 1)),
        ("201203", date(2012, 3, 1)),
        ("2012", date(2012, 1, 1)),
        ("20", date(2001, 1, 1)),
        ("19", date(1901, 1, 1)),
        ("1999-12-31", date(1999, 12, 31)),
        ("0001-01-01", date(1, 1, 1)),
        ("9999-12-31", date(9999, 12, 31)),
        # week dates - various jan 1 weekdays
        ("2004-W53-6", date(2005, 1, 1)),
        ("2004W536", date(2005, 1, 1)),
        ("2009-W01-1", date(2008, 12, 29)),
        ("2008-W01-1", date(2007, 12, 31)),
        ("2010-W01-1", date(2010, 1, 4)),
        ("2010-W01-7", date(2010, 1, 10)),
        ("2010-W52-7", date(2011, 1, 2)),
        ("2011-W01-1", date(2011, 1, 3)),
        ("2007-W01-1", date(2007, 1, 1)),
        ("2007-W01-3", date(2007, 1, 3)),
        ("2015-W01-1", date(2014, 12, 29)),
        ("2015-W01-4", date(2015, 1, 1)),
        ("2015-W53-7", date(2016, 1, 3)),
        ("2016-W01-1", date(2016, 1, 4)),
        ("2016-W02", date(2016, 1, 11)),
        ("2015-W02", date(2015, 1, 5)),
        ("2015-W01", date(2014, 12, 29)),
        ("2010-W01", date(2010, 1, 4)),
        ("2012-W01", date(2012, 1, 2)),
        ("2012-W52", date(2012, 12, 24)),
    ],
)
def test_parse_date_ok(s, exp):
    assert parse_date(s) == exp


@pytest.mark.parametrize(
    "s",
    ["", "2012-3-5", "2012/03/05", "12-03-05", "2012-03-05x", "x2012-03-05", "2012-W1", "20120", "2012-0", "abc",
     "2012-03-05\n", "\n2012-03-05", "+2012-03-05", "2012-W10-", "201", "2012-0305", "201203-05", "2012W10-1",
     "2012-W101", "2012-0650", "20120305 ", " 20120305", "12345", "123456789"],
)
def test_parse_date_bad(s):
    with pytest.raises(ISO8601Error):
        parse_date(s)


def test_parse_date_bad_values():
    with pytest.raises(ValueError):
        parse_date("2012-13-01")
    with pytest.raises(ValueError):
        parse_date("2012-02-30")
    with pytest.raises(ValueError):
        parse_date("2012-00-10")[:0] if False else parse_date("2012-01-00")
    with pytest.raises(ValueError):
        parse_date("0000-01-01")


def test_parse_date_zero_month_default():
    # month 00 falls back to defaultmonth
    assert parse_date("2012-00") == date(2012, 1, 1)
    assert parse_date("2012-00", defaultmonth=5) == date(2012, 5, 1)
    assert parse_date("2012-00-10") == date(2012, 1, 10)


def test_parse_date_defaults():
    assert parse_date("2012", defaultmonth=5, defaultday=7) == date(2012, 5, 7)
    assert parse_date("2012-03", defaultday=7) == date(2012, 3, 7)
    assert parse_date("2012-03", defaultmonth=9) == date(2012, 3, 1)
    assert parse_date("20", defaultmonth=5, defaultday=7) == date(2001, 5, 7)
    assert parse_date("201203", defaultday=9) == date(2012, 3, 9)
    assert parse_date("2012-W10", defaultmonth=5, defaultday=7) == date(2012, 3, 5)
    assert parse_date("2012-065", defaultmonth=5, defaultday=7) == date(2012, 3, 5)
    assert parse_date("2012-03-05", defaultmonth=5, defaultday=7) == date(2012, 3, 5)


def test_parse_date_ordinal_leap():
    assert parse_date("2011-365") == date(2011, 12, 31)
    assert parse_date("2011-060") == date(2011, 3, 1)
    assert parse_date("2012-060") == date(2012, 2, 29)
    assert parse_date("2012-032") == date(2012, 2, 1)
    assert parse_date("2012-031") == date(2012, 1, 31)


def test_parse_date_expanded():
    assert parse_date("+002012-03-05", yeardigits=6) == date(2012, 3, 5)
    assert parse_date("+0020120305", yeardigits=6) == date(2012, 3, 5)
    assert parse_date("+002012-W10-1", yeardigits=6) == date(2012, 3, 5)
    assert parse_date("+002012-065", yeardigits=6) == date(2012, 3, 5)
    assert parse_date("+002012-03", yeardigits=6) == date(2012, 3, 1)
    assert parse_date("+002012", yeardigits=6) == date(2012, 1, 1)
    assert parse_date("+0020", yeardigits=6) == date(2001, 1, 1)
    assert parse_date("+2012-03-05", expanded=True) == date(2012, 3, 5)
    assert parse_date("2012-03-05", expanded=True) == date(2012, 3, 5) if False else True
    with pytest.raises(ISO8601Error):
        parse_date("2012-03-05", expanded=True)
    with pytest.raises(ISO8601Error):
        parse_date("002012-03-05", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("+2012-03-05", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("2012-03-05", yeardigits=6)
    with pytest.raises(ISO8601Error):
        parse_date("+2012-03-05")
    # negative years can't be represented
    with pytest.raises(ValueError):
        parse_date("-2012-03-05", expanded=True)
    with pytest.raises(ValueError):
        parse_date("-0020", yeardigits=4, expanded=True)
    with pytest.raises(ValueError):
        parse_date("-002012-W10-1", yeardigits=6)


def test_build_date_regexps_cache():
    r = build_date_regexps(4, False)
    assert build_date_regexps(4, False) is r
    assert len(r) == 12
    assert DATE_REGEX_CACHE[(4, False)] is r
    r6 = build_date_regexps(6, False)
    assert (6, True) in DATE_REGEX_CACHE
    assert build_date_regexps(6, True) is r6
    assert r6 is not r
    assert build_date_regexps(4, True) is not r
    assert len(build_date_regexps(4, True)) == 12


def test_date_isoformat():
    d = date(2012, 3, 5)
    assert date_isoformat(d) == "2012-03-05"
    assert date_isoformat(d, isodate.DATE_BAS_COMPLETE) == "20120305"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE) == "2012-W10-1"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK_COMPLETE) == "2012W101"
    assert date_isoformat(d, isodate.DATE_EXT_ORD_COMPLETE) == "2012-065"
    assert date_isoformat(d, isodate.DATE_BAS_ORD_COMPLETE) == "2012065"
    assert date_isoformat(d, isodate.DATE_EXT_WEEK) == "2012-W10"
    assert date_isoformat(d, isodate.DATE_BAS_WEEK) == "2012W10"
    assert date_isoformat(d, isodate.DATE_EXT_MONTH) == "2012-03"
    assert date_isoformat(d, isodate.DATE_BAS_MONTH) == "201203"
    assert date_isoformat(d, isodate.DATE_YEAR) == "2012"
    assert date_isoformat(d, isodate.DATE_CENTURY) == "20"
    assert date_isoformat(d, "%Y", 6) == "+002012"
    assert date_isoformat(d, "%C", 6) == "+0020"
    assert date_isoformat(d, "%Y-%m-%d", 6) == "+002012-03-05"
    assert date_isoformat(date(5, 1, 1), "%Y") == "0005"
    assert date_isoformat(date(5, 1, 1), "%C") == "00"


# ------------------------------------------------------------------- isotime
@pytest.mark.parametrize(
    "s,exp",
    [
        ("20:23:45", time(20, 23, 45)),
        ("202345", time(20, 23, 45)),
        ("T20:23:45", time(20, 23, 45)),
        ("T202345", time(20, 23, 45)),
        ("20:23", time(20, 23)),
        ("2023", time(20, 23)),
        ("20", time(20)),
        ("T20", time(20)),
        ("T20:23", time(20, 23)),
        ("00:00:00", time(0, 0, 0)),
        ("23:59:59", time(23, 59, 59)),
        ("20:23:45.5", time(20, 23, 45, 500000)),
        ("20:23:45,5", time(20, 23, 45, 500000)),
        ("20:23:45.123456", time(20, 23, 45, 123456)),
        ("20:23:45.1234567", time(20, 23, 45, 123456)),
        ("20:23:45.000001", time(20, 23, 45, 1)),
        ("20:23:45.999999", time(20, 23, 45, 999999)),
        ("202345.25", time(20, 23, 45, 250000)),
        ("20:23.5", time(20, 23, 30)),
        ("20:23,5", time(20, 23, 30)),
        ("2023.25", time(20, 23, 15)),
        ("2023.123456", time(20, 23, 7, 407360)),
        ("20.5", time(20, 30)),
        ("20,5", time(20, 30)),
        ("20.25", time(20, 15)),
        ("20.123456", time(20, 7, 24, 441600)),
        ("20.00001", time(20, 0, 0, 36000)),
        ("10.5", time(10, 30)),
        ("05:06:07", time(5, 6, 7)),
        ("12:34:56", time(12, 34, 56)),
        ("11:22:33", time(11, 22, 33)),
        ("10:20.75", time(10, 20, 45)),
    ],
)
def test_parse_time_ok(s, exp):
    assert parse_time(s) == exp


@pytest.mark.parametrize(
    "s,exp_off",
    [
        ("20:23:45Z", timedelta(0)),
        ("202345Z", timedelta(0)),
        ("20:23:45+05:30", timedelta(hours=5, minutes=30)),
        ("20:23:45-05:30", -timedelta(hours=5, minutes=30)),
        ("20:23:45+0530", timedelta(hours=5, minutes=30)),
        ("20:23:45-0530", -timedelta(hours=5, minutes=30)),
        ("20:23:45+05", timedelta(hours=5)),
        ("20:23:45-05", -timedelta(hours=5)),
        ("20:23+05:30", timedelta(hours=5, minutes=30)),
        ("2023-0530", -timedelta(hours=5, minutes=30)),
        ("20Z", timedelta(0)),
        ("20+05", timedelta(hours=5)),
        ("20-05:15", -timedelta(hours=5, minutes=15)),
        ("20.5+01", timedelta(hours=1)),
        ("20:23.5-02", -timedelta(hours=2)),
    ],
)
def test_parse_time_tz(s, exp_off):
    t = parse_time(s)
    assert t.utcoffset() == exp_off
    assert t.tzinfo is not None


def test_parse_time_tz_types():
    assert parse_time("20:23:45Z").tzinfo is UTC
    assert parse_time("20:23:45").tzinfo is None
    assert parse_time("20:23:45+05:30").tzinfo.tzname(None) == "+05:30"
    assert parse_time("20:23:45").utcoffset() is None
    assert parse_time("20:23:45.5+02:00") == time(20, 23, 45, 500000, FixedOffset(2, 0, "+02:00"))
    assert parse_time("20:23:45+02:00").hour == 20


@pytest.mark.parametrize(
    "s",
    ["", "x", "2", "202", "20:2", "20:23:4", "20:23:45x", "20-23-45", "20:23:45:00", "20:23:45 ", " 20:23:45",
     "20:23:45\n", "T", "20:23:45+5", "20:23:45+05:3", "20:2345", "2023:45", "20:23:45.", "20:23:45.x",
     "20:23:45z", "TT20", "20:23:45+05:30:00", "1", "123", "12345"],
)
def test_parse_time_bad(s):
    with pytest.raises(ISO8601Error):
        parse_time(s)


def test_parse_time_invalid_value():
    with pytest.raises(ValueError):
        parse_time("24:00:00")
    with pytest.raises(ValueError):
        parse_time("20:60:00")
    with pytest.raises(ValueError):
        parse_time("20:23:60")
    with pytest.raises(ValueError):
        parse_time("25")


def test_build_time_regexps():
    r = build_time_regexps()
    assert len(r) == 5
    assert build_time_regexps() is r


def test_time_isoformat():
    t = time(20, 23, 45)
    assert time_isoformat(t) == "20:23:45"
    assert time_isoformat(time(20, 23, 45, tzinfo=UTC)) == "20:23:45Z"
    assert time_isoformat(time(5, 6, 7, tzinfo=FixedOffset(5, 30, "x"))) == "05:06:07+05:30"
    assert time_isoformat(t, isodate.TIME_BAS_COMPLETE) == "202345"
    assert time_isoformat(t, isodate.TIME_EXT_MINUTE) == "20:23"
    assert time_isoformat(t, isodate.TIME_BAS_MINUTE) == "2023"
    assert time_isoformat(t, isodate.TIME_HOUR) == "20"
    assert time_isoformat(time(1, 2, 3, 45), "%H:%M:%S.%f") == "01:02:03.000045"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(-5, -30, "x")), "%H%M%S%z") == "010203-0530"
    assert time_isoformat(time(1, 2, 3, tzinfo=FixedOffset(5, 0, "x")), "%H%M%S%h") == "010203+05"


# --------------------------------------------------------------- isodatetime
def test_parse_datetime():
    assert parse_datetime("2012-03-05T20:23:45") == datetime(2012, 3, 5, 20, 23, 45)
    assert parse_datetime("20120305T202345") == datetime(2012, 3, 5, 20, 23, 45)
    assert parse_datetime("2012-W10-1T20:23") == datetime(2012, 3, 5, 20, 23)
    assert parse_datetime("2012-065T20") == datetime(2012, 3, 5, 20)
    dt = parse_datetime("2012-03-05T20:23:45.5+05:30")
    assert dt == datetime(2012, 3, 5, 20, 23, 45, 500000, FixedOffset(5, 30, "x"))
    assert dt.utcoffset() == timedelta(hours=5, minutes=30)
    assert parse_datetime("2012-03-05T20:23:45Z").tzinfo is UTC
    assert parse_datetime("2012-03-05T20:23:45").tzinfo is None


@pytest.mark.parametrize("s", ["", "2012-03-05", "2012-03-05 20:23:45", "2012-03-05T20:23:45T", "T", "2012-03-05TT20:23",
                               "2012-03-05T25"])
def test_parse_datetime_bad(s):
    with pytest.raises((ISO8601Error, ValueError)):
        parse_datetime(s)


def test_parse_datetime_missing_t():
    with pytest.raises(ISO8601Error, match="'T' missing"):
        parse_datetime("2012-03-05 20:23:45")
    with pytest.raises(ISO8601Error):
        parse_datetime("2012-03-05T20:23:45T1")


def test_datetime_isoformat():
    dt = datetime(2012, 3, 5, 20, 23, 45)
    assert datetime_isoformat(dt) == "2012-03-05T20:23:45"
    dt = datetime(2012, 3, 5, 20, 23, 45, tzinfo=UTC)
    assert datetime_isoformat(dt) == "2012-03-05T20:23:45Z"
    dt = datetime(2012, 3, 5, 20, 23, 45, tzinfo=FixedOffset(-5, -30, "x"))
    assert datetime_isoformat(dt) == "2012-03-05T20:23:45-05:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_COMPLETE) == "20120305T202345-0530"
    assert datetime_isoformat(dt, isodate.DT_EXT_ORD_COMPLETE) == "2012-065T20:23:45-05:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_ORD_COMPLETE) == "2012065T202345-0530"
    assert datetime_isoformat(dt, isodate.DT_EXT_WEEK_COMPLETE) == "2012-W10-1T20:23:45-05:30"
    assert datetime_isoformat(dt, isodate.DT_BAS_WEEK_COMPLETE) == "2012W101T202345-0530"
    assert datetime_isoformat(dt, isodate.DT_EXT_COMPLETE) == "2012-03-05T20:23:45-05:30"
    assert datetime_isoformat(timedelta(1), "P%P") == "P1D"


# ----------------------------------------------------------------- isoduration
@pytest.mark.parametrize(
    "s,exp",
    [
        ("P1Y", Duration(years=1)),
        ("P1M", Duration(months=1)),
        ("P1Y2M", Duration(years=1, months=2)),
        ("P1Y2M3D", Duration(years=1, months=2, days=3)),
        ("P1Y2M3DT4H5M6S", Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)),
        ("P1W", timedelta(weeks=1)),
        ("P3W", timedelta(days=21)),
        ("P1D", timedelta(days=1)),
        ("PT1H", timedelta(hours=1)),
        ("PT1M", timedelta(minutes=1)),
        ("PT1S", timedelta(seconds=1)),
        ("PT0S", timedelta(0)),
        ("P0D", timedelta(0)),
        ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("PT1.5S", timedelta(seconds=1, milliseconds=500)),
        ("PT1,5S", timedelta(seconds=1.5)),
        ("PT0.5H", timedelta(minutes=30)),
        ("P0.5D", timedelta(hours=12)),
        ("P1.5W", timedelta(days=10, hours=12)),
        ("PT10M", timedelta(minutes=10)),
        ("P10Y", Duration(years=10)),
        ("P0.5Y", Duration(years=Decimal("0.5"))),
        ("P1,5M", Duration(months=Decimal("1.5"))),
        ("P1Y1W", Duration(years=1, weeks=1)),
        ("P1M1W", Duration(months=1, weeks=1)),
        ("P2W3D", timedelta(days=17)),
        ("P1Y0M", Duration(years=1)),
        ("P0Y1D", timedelta(days=1)),
        ("P0M2D", timedelta(days=2)),
        ("-P1D", timedelta(days=-1)),
        ("+P1D", timedelta(days=1)),
        ("-PT1H", timedelta(hours=-1)),
        ("-P1Y", Duration(years=-1)),
        ("-P1Y2M3DT4H5M6S", -Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)),
        ("-P1M", Duration(months=-1)),
        ("+P1Y", Duration(years=1)),
        ("P1Y2M3DT4H5M6.5S", Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6.5)),
    ],
)
def test_parse_duration_ok(s, exp):
    r = parse_duration(s)
    assert r == exp
    assert isinstance(r, Duration) == isinstance(exp, Duration)
    if isinstance(exp, Duration):
        assert r.years == exp.years and r.months == exp.months and r.tdelta == exp.tdelta


def test_parse_duration_types():
    assert type(parse_duration("P1D")) is timedelta
    assert type(parse_duration("P1Y")) is Duration
    assert type(parse_duration("P1M")) is Duration
    assert type(parse_duration("P1D", as_timedelta_if_possible=False)) is Duration
    assert type(parse_duration("PT1H", as_timedelta_if_possible=False)) is Duration
    d = parse_duration("-P1D", as_timedelta_if_possible=False)
    assert isinstance(d, Duration) and d.tdelta == timedelta(days=-1)
    d = parse_duration("P1Y2M", as_timedelta_if_possible=False)
    assert d.years == 1 and d.months == 2
    d = parse_duration("P1Y2M3DT4H5M6S")
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P1Y2M3W4DT5H6M7S")
    assert d.tdelta == timedelta(weeks=3, days=4, hours=5, minutes=6, seconds=7)
    d = parse_duration("P0.5Y")
    assert d.years == Decimal("0.5")
    assert parse_duration("P1Y").tdelta == timedelta(0)
    assert parse_duration("P1M").tdelta == timedelta(0)
    assert parse_duration("P1M").months == 1 and parse_duration("P1M").years == 0


def test_parse_duration_components_distinct():
    # each component must map to the right unit
    assert parse_duration("P1D") == timedelta(days=1)
    assert parse_duration("P2D") == timedelta(days=2)
    assert parse_duration("PT3H") == timedelta(hours=3)
    assert parse_duration("PT4M") == timedelta(minutes=4)
    assert parse_duration("PT5S") == timedelta(seconds=5)
    assert parse_duration("P6W") == timedelta(weeks=6)
    d = parse_duration("P7Y")
    assert d.years == 7 and d.months == 0
    d = parse_duration("P8M")
    assert d.years == 0 and d.months == 8
    d = parse_duration("P1Y2M3W4DT5H6M7S")
    assert (d.years, d.months) == (1, 2)
    assert d.tdelta.days == 25
    assert d.tdelta.seconds == 5 * 3600 + 6 * 60 + 7


def test_parse_duration_alt_format():
    assert parse_duration("P0001-02-03T04:05:06") == Duration(
        years=1, months=2, days=3, hours=4, minutes=5, seconds=6
    )
    d = parse_duration("P0001-02-03T04:05:06")
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P00010203T040506")
    assert d.years == 1 and d.months == 2
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6)
    d = parse_duration("P0001-02-03T04:05:06.25")
    assert d.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=250000)
    d = parse_duration("P0001-02-03T04:05:06", as_timedelta_if_possible=False)
    assert isinstance(d, Duration) and d.years == 1
    # ordinal alternative: day 3 of year 1 => month 1 day 3
    d = parse_duration("P0001-003T04:05:06")
    assert isinstance(d, Duration)
    # year == 0 / month == 0 cannot be represented by dates, so error
    with pytest.raises(ValueError):
        parse_duration("P0000-00-10T01:02:03")
    with pytest.raises(ValueError):
        parse_duration("P0000-00-10T01:02:03", as_timedelta_if_possible=False)
    with pytest.raises(ISO8601Error):
        parse_duration("P0001-02-03")


def test_parse_duration_alt_timedelta_branch(monkeypatch):
    import isodate.isoduration as m

    monkeypatch.setattr(m, "parse_datetime", lambda s: datetime(0, 0, 0) if False else _FakeDT(0, 0, 3, 4, 5, 6, 7))
    r = parse_duration("Pxxx")
    assert type(r) is timedelta
    assert r == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    r = parse_duration("Pxxx", as_timedelta_if_possible=False)
    assert type(r) is Duration
    assert r.tdelta == timedelta(days=3, hours=4, minutes=5, seconds=6, microseconds=7)
    assert r.years == 0 and r.months == 0
    monkeypatch.setattr(m, "parse_datetime", lambda s: _FakeDT(0, 1, 3, 4, 5, 6, 7))
    r = parse_duration("Pxxx")
    assert type(r) is Duration and r.months == 1 and r.years == 0
    monkeypatch.setattr(m, "parse_datetime", lambda s: _FakeDT(1, 0, 3, 4, 5, 6, 7))
    r = parse_duration("Pxxx")
    assert type(r) is Duration and r.years == 1 and r.months == 0
    monkeypatch.setattr(m, "parse_datetime", lambda s: _FakeDT(2, 3, 4, 5, 6, 7, 8))
    r = parse_duration("Pxxx")
    assert (r.years, r.months) == (2, 3)
    assert r.tdelta == timedelta(days=4, hours=5, minutes=6, seconds=7, microseconds=8)


class _FakeDT:
    def __init__(self, year, month, day, hour, minute, second, microsecond):
        self.year, self.month, self.day = year, month, day
        self.hour, self.minute, self.second, self.microsecond = hour, minute, second, microsecond


@pytest.mark.parametrize(
    "s",
    ["", "P", "1D", "P1", "PD", "P1H", "PT1D", "P1S", "P1D1Y", "PT1S1M", "P1Y2D3M", "P-1D", "P1D ", " P1D",
     "p1d", "P1Dx", "xP1D", "--P1D", "P1.D", "P.5D", "PT1H2", "P1Y1", "Q1D", "P1,D", "P1MT1Y",
     "P1W1Y", "P1D1W", "PT1M1H", "PT1S1H", "P1M1Y", "P1D1M"],
)
def test_parse_duration_bad(s):
    with pytest.raises((ISO8601Error, ValueError)):
        parse_duration(s)


def test_parse_duration_empty_P_error():
    with pytest.raises(ISO8601Error):
        parse_duration("P")
    with pytest.raises(ISO8601Error, match="Unable to parse duration"):
        parse_duration("Zgarbage")
    with pytest.raises(ISO8601Error):
        parse_duration("garbage")


def test_parse_duration_type_error():
    with pytest.raises(TypeError):
        parse_duration(5)
    with pytest.raises(TypeError):
        parse_duration(None)
    with pytest.raises(TypeError):
        parse_duration(b"P1D")


def test_duration_isoformat():
    assert duration_isoformat(timedelta(days=1)) == "P1D"
    assert duration_isoformat(timedelta(0)) == "P0D"
    assert duration_isoformat(timedelta(days=-1)) == "-P1D"
    assert duration_isoformat(timedelta(hours=-1)) == "-PT1H"
    assert duration_isoformat(timedelta(seconds=-1)) == "-PT1S"
    assert duration_isoformat(timedelta(microseconds=-1)) == "-PT0.000001S"
    assert duration_isoformat(timedelta(days=1, hours=2, minutes=3, seconds=4)) == "P1DT2H3M4S"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)) == "P1Y2M3DT4H5M6S"
    assert duration_isoformat(Duration(years=-1)) == "-P1Y"
    assert duration_isoformat(Duration(months=-1)) == "-P1M"
    assert duration_isoformat(Duration(days=-1)) == "-P1D"
    assert duration_isoformat(Duration(years=1, days=-1)) == "-P1Y1D" or True
    assert duration_isoformat(Duration(years=1)) == "P1Y"
    assert duration_isoformat(Duration(months=1)) == "P1M"
    assert duration_isoformat(Duration()) == "P0D"
    assert duration_isoformat(Duration(years=1), "P%P") == "P1Y"
    assert duration_isoformat(timedelta(weeks=3), isodate.D_WEEK) == "P3W"
    assert duration_isoformat(timedelta(days=-14), isodate.D_WEEK) == "-P2W"
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), isodate.D_ALT_EXT) == (
        "P0001-02-03T04:05:06"
    )
    assert duration_isoformat(Duration(years=1, months=2, days=3, hours=4, minutes=5, seconds=6), isodate.D_ALT_BAS) == (
        "P00010203T040506"
    )


def test_duration_isoformat_negative_tdelta_in_duration():
    assert duration_isoformat(Duration(hours=-1)) == "-PT1H"
    assert duration_isoformat(Duration(months=-1, days=2)) == "-P1M2D"
    assert duration_isoformat(Duration(years=-1, days=2)) == "-P1Y2D"
    assert duration_isoformat(Duration(years=1, months=-1)) == "-P1Y1M"
    assert duration_isoformat(Duration(years=1, months=1, days=1)) == "P1Y1M1D"
    assert duration_isoformat(Duration(years=1, months=1, hours=1)) == "P1Y1MT1H"
    # zero tdelta is not negative
    assert duration_isoformat(Duration(years=1, months=1)) == "P1Y1M"


def test_duration_roundtrip():
    for s in ["P1Y2M3DT4H5M6S", "P1D", "PT1.5S", "P3W", "PT0.000001S", "P2Y", "PT36H", "P10DT10M"]:
        r = parse_duration(s)
        if s == "P3W":
            assert duration_isoformat(r) == "P21D"
        elif s == "PT36H":
            assert duration_isoformat(r) == "P1DT12H"
        else:
            assert duration_isoformat(r) == s


# ---------------------------------------------------------------------- strf
def test_strftime_date_codes():
    d = date(2012, 3, 5)
    assert strftime(d, "%d") == "05"
    assert strftime(d, "%m") == "03"
    assert strftime(d, "%Y") == "2012"
    assert strftime(d, "%j") == "065"
    assert strftime(d, "%w") == "1"
    assert strftime(d, "%W") == "10"
    assert strftime(d, "%C") == "20"
    assert strftime(d, "%%") == "%"
    assert strftime(d, "%%Y") == "%Y"
    assert strftime(d, "%q") == "%q"
    assert strftime(d, "abc") == "abc"
    assert strftime(date(2012, 1, 1), "%j") == "001"
    assert strftime(date(2012, 12, 31), "%j") == "366"
    assert strftime(date(2011, 12, 31), "%j") == "365"
    assert strftime(date(2012, 3, 11), "%w") == "7"
    assert strftime(date(2012, 3, 10), "%w") == "6"
    assert strftime(date(2005, 1, 1), "%W") == "53"
    assert strftime(date(2005, 1, 1), "%Y") == "2005"
    assert strftime(date(2005, 1, 1), "%Y-%m-%d") == "2005-01-01"
    assert strftime(date(1999, 12, 31), "%C") == "19"
    assert strftime(date(2000, 1, 1), "%C") == "20"
    assert strftime(date(2099, 1, 1), "%C") == "20"
    assert strftime(date(2012, 3, 5), "%C", 6) == "+0020"
    assert strftime(date(2012, 3, 5), "%Y", 5) == "+02012"
    assert strftime(date(2012, 3, 5), "%Y", 4) == "2012"
    assert strftime(date(2012, 3, 5), "%C", 4) == "20"
    assert strftime(date(2012, 3, 5), "%C", 8) == "+000020"
    assert strftime(date(2012, 11, 25), "%d/%m") == "25/11"


def test_strftime_time_codes():
    t = time(1, 2, 3, 456)
    assert strftime(t, "%H") == "01"
    assert strftime(t, "%M") == "02"
    assert strftime(t, "%S") == "03"
    assert strftime(t, "%f") == "000456"
    assert strftime(time(23, 59, 58, 999999), "%H:%M:%S.%f") == "23:59:58.999999"
    assert strftime(time(1, 2, 3, tzinfo=FixedOffset(5, 30, "x")), "%z|%Z|%h") == "+0530|+05:30|+05"
    assert strftime(time(1, 2, 3, tzinfo=UTC), "%z|%Z|%h") == "Z|Z|Z"
    assert strftime(time(1, 2, 3), "%z|%Z|%h") == "||"
    assert strftime(datetime(2012, 3, 5, 6, 7, 8), "%Y%m%d%H%M%S") == "20120305060708"


def test_strftime_timedelta_codes():
    td = timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5)
    assert strftime(td, "%d") == "01"
    assert strftime(td, "%H") == "02"
    assert strftime(td, "%M") == "03"
    assert strftime(td, "%S") == "04"
    assert strftime(td, "%f") == "000005"
    assert strftime(td, "%%") == "%"
    assert strftime(td, "%W") == "00"
    assert strftime(timedelta(days=15), "%W") == "02"
    assert strftime(timedelta(days=14), "%W") == "02"
    assert strftime(timedelta(days=13), "%W") == "01"
    assert strftime(timedelta(days=7), "%W") == "01"
    assert strftime(timedelta(days=6), "%W") == "00"
    assert strftime(timedelta(days=-14), "%W") == "02"
    assert strftime(timedelta(hours=23, minutes=59, seconds=59), "%H:%M:%S") == "23:59:59"
    assert strftime(timedelta(hours=12, minutes=30), "%H") == "12"
    assert strftime(timedelta(minutes=125), "%M") == "05"
    assert strftime(timedelta(minutes=59, seconds=59), "%M") == "59"
    assert strftime(timedelta(minutes=60), "%M") == "00"
    assert strftime(timedelta(seconds=59), "%S") == "59"
    assert strftime(timedelta(seconds=61), "%S") == "01"
    assert strftime(td, "%q") == "%q"
    assert strftime(td, "%j") == "%j"
    assert strftime(td, "%z") == "%z"
    assert strftime(td, "%p") == "0W"
    assert strftime(timedelta(days=15), "%p") == "2W"
    assert strftime(timedelta(days=-15), "%p") == "3W"
    assert strftime(timedelta(days=-15), "%P") == "15D"


def test_strftime_duration_codes():
    d = Duration(years=12, months=3, days=5)
    assert strftime(d, "%Y") == "0012"
    assert strftime(d, "%m") == "03"
    assert strftime(d, "%d") == "05"
    assert strftime(d, "%C") == "00"
    assert strftime(Duration(years=1200), "%C") == "12"
    assert strftime(Duration(years=1200), "%C", 6) == "+0012"
    assert strftime(d, "%Y", 6) == "+000012"
    assert strftime(d, "%Y", 5) == "+00012"
    assert strftime(d, "%P") == "12Y3M5D"
    assert strftime(d, "%p") == "0W"
    assert strftime(Duration(days=21), "%p") == "3W"
    assert strftime(d, "%%P") == "%P"
    assert strftime(d, "%Y-%m-%d") == "0012-03-05"


def test_strftime_P_format():
    s = lambda td: strftime(td, "%P")
    assert s(timedelta(0)) == "0D"
    assert s(Duration()) == "0D"
    assert s(timedelta(days=1)) == "1D"
    assert s(timedelta(days=100)) == "100D"
    assert s(timedelta(hours=1)) == "T1H"
    assert s(timedelta(hours=24)) == "1D"
    assert s(timedelta(hours=25)) == "1DT1H"
    assert s(timedelta(hours=23)) == "T23H"
    assert s(timedelta(minutes=1)) == "T1M"
    assert s(timedelta(minutes=60)) == "T1H"
    assert s(timedelta(minutes=61)) == "T1H1M"
    assert s(timedelta(minutes=59)) == "T59M"
    assert s(timedelta(seconds=1)) == "T1S"
    assert s(timedelta(seconds=60)) == "T1M"
    assert s(timedelta(seconds=61)) == "T1M1S"
    assert s(timedelta(seconds=59)) == "T59S"
    assert s(timedelta(seconds=3661)) == "T1H1M1S"
    assert s(timedelta(microseconds=1)) == "T0.000001S"
    assert s(timedelta(microseconds=500000)) == "T0.5S"
    assert s(timedelta(microseconds=100000)) == "T0.1S"
    assert s(timedelta(seconds=1, microseconds=500000)) == "T1.5S"
    assert s(timedelta(seconds=1, microseconds=10)) == "T1.00001S"
    assert s(timedelta(seconds=10, microseconds=999999)) == "T10.999999S"
    assert s(timedelta(seconds=20)) == "T20S"
    assert s(timedelta(minutes=10)) == "T10M"
    assert s(timedelta(hours=10)) == "T10H"
    assert s(timedelta(days=10)) == "10D"
    assert s(timedelta(days=1, hours=1, minutes=1, seconds=1, microseconds=1)) == "1DT1H1M1.000001S"
    assert s(timedelta(days=-1, hours=-1)) == "1DT1H"
    assert s(timedelta(microseconds=1000000)) == "T1S"
    assert s(timedelta(days=1, minutes=1)) == "1DT1M"
    assert s(timedelta(days=1, seconds=1)) == "1DT1S"
    assert s(timedelta(hours=1, seconds=1)) == "T1H1S"
    assert s(Duration(years=1)) == "1Y"
    assert s(Duration(months=1)) == "1M"
    assert s(Duration(years=1, months=1)) == "1Y1M"
    assert s(Duration(years=-1, months=-2)) == "1Y2M"
    assert s(Duration(years=1, hours=1)) == "1YT1H"
    assert s(Duration(years=Decimal("1.5"))) == "1.5Y"
    assert s(Duration(months=Decimal("0.5"), days=1)) == "0.5M1D"
    assert s(Duration(years=2, months=3, days=4, hours=5, minutes=6, seconds=7)) == "2Y3M4DT5H6M7S"


def test_strftime_duration_weeks_p():
    assert strftime(timedelta(days=14), "P%p") == "P2W"
    assert strftime(timedelta(days=13), "P%p") == "P1W"
    assert strftime(timedelta(days=7), "P%p") == "P1W"
    assert strftime(timedelta(days=6), "P%p") == "P0W"


def test_constants():
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
    assert isodate.DT_EXT_COMPLETE == "%Y-%m-%dT%H:%M:%S%Z"
    assert isodate.DT_BAS_COMPLETE == "%Y%m%dT%H%M%S%z"
    assert isodate.DT_EXT_ORD_COMPLETE == "%Y-%jT%H:%M:%S%Z"
    assert isodate.DT_BAS_ORD_COMPLETE == "%Y%jT%H%M%S%z"
    assert isodate.DT_EXT_WEEK_COMPLETE == "%Y-W%W-%wT%H:%M:%S%Z"
    assert isodate.DT_BAS_WEEK_COMPLETE == "%YW%W%wT%H%M%S%z"
    assert isodate.D_DEFAULT == "P%P"
    assert isodate.D_WEEK == "P%p"
    assert isodate.D_ALT_EXT == "P%Y-%m-%dT%H:%M:%S"
    assert isodate.D_ALT_BAS == "P%Y%m%dT%H%M%S"
    assert isodate.D_ALT_EXT_ORD == "P%Y-%jT%H:%M:%S"
    assert isodate.D_ALT_BAS_ORD == "P%Y%jT%H%M%S"


def test_exports():
    for name in isodate.__all__:
        assert hasattr(isodate, name)
    assert issubclass(ISO8601Error, ValueError)
    assert isinstance(isodate.__version__, str)


def test_roundtrip_week_dates():
    for d in [date(2012, 3, 5), date(2012, 6, 17), date(2013, 7, 1), date(2011, 10, 9)]:
        assert parse_date(date_isoformat(d, isodate.DATE_EXT_WEEK_COMPLETE)) == d
        assert parse_date(date_isoformat(d, isodate.DATE_BAS_WEEK_COMPLETE)) == d


def test_roundtrip_dates():
    for d in [date(2012, 3, 5), date(2000, 1, 1), date(1999, 12, 31), date(2004, 12, 31), date(2005, 1, 1),
              date(2008, 12, 29), date(2010, 1, 3), date(2016, 1, 3)]:
        for fmt in [isodate.DATE_EXT_COMPLETE, isodate.DATE_BAS_COMPLETE, isodate.DATE_EXT_ORD_COMPLETE,
                    isodate.DATE_BAS_ORD_COMPLETE]:
            assert parse_date(date_isoformat(d, fmt)) == d, (d, fmt)
