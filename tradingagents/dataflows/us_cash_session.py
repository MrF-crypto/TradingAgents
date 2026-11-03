"""US cash-equity session flags for a Binance print that may trade around the clock.

The regular session used here is 09:30–16:00 America/New_York, Monday to Friday,
excluding the NYSE's standard full-day holidays:

- New Year's Day, Juneteenth (from 2021), Independence Day, Christmas Day,
  observed on Friday when the date is a Saturday and on Monday when it is a Sunday
- the third Monday of January (Martin Luther King Jr. Day) and of February
  (Washington's Birthday)
- Good Friday (the Friday before Easter Sunday)
- the last Monday of May (Memorial Day)
- the first Monday of September (Labor Day)
- the fourth Thursday of November (Thanksgiving)

Early closes (the day after Thanksgiving, Christmas Eve, and the weekday before
Independence Day when that date is not already a full closure) are not marked
closed. This is the recurring rule set, not a download of the NYSE calendar for
a particular year, so a one-off closure is not flagged.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# Bar open is inside the regular session when it falls in this half-open range.
# A Binance 1h kline is aligned to UTC, which is an exact ET hour. The 09:00 ET
# hour opens before 09:30, so it is classified as outside the regular session
# even though the cash open sits inside that hour.
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)

# Sub-daily intervals documented for both spot and USD-M klines. A daily (or
# coarser) bar is a UTC period, not a US session, and is not given an intraday label.
SUB_DAILY_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h"}
)


def easter_sunday(year: int) -> date:
    """Easter Sunday, Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(day: date) -> date:
    """Saturday holidays are observed Friday; Sunday holidays, Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth ``weekday`` of a month (Monday is 0)."""
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


@lru_cache(maxsize=64)
def nyse_full_day_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures whose rule falls in ``year`` (plus a New Year observed in December)."""
    days = {
        _observed(date(year, 1, 1)),
        _observed(date(year, 7, 4)),
        _observed(date(year, 12, 25)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
    }
    # NYSE began closing for Juneteenth when it became a federal holiday.
    if year >= 2021:
        days.add(_observed(date(year, 6, 19)))
    return frozenset(days)


def is_nyse_full_day_holiday(day: date) -> bool:
    """True when ``day`` is a full-day cash closure, including an observed New Year on Dec 31."""
    return day in nyse_full_day_holidays(day.year) or day in nyse_full_day_holidays(day.year + 1)


def cash_session_status(day: date) -> str:
    """``open``, ``weekend``, or ``holiday`` for the US cash session on ``day``."""
    if day.weekday() >= 5:
        return "weekend"
    if is_nyse_full_day_holiday(day):
        return "holiday"
    return "open"


def et_parts(open_time_ms: int) -> tuple[datetime, date]:
    """Bar open as an America/New_York datetime and its local date."""
    opened = datetime.fromtimestamp(open_time_ms / 1000, tz=UTC).astimezone(NY)
    return opened, opened.date()


def session_label(open_time_ms: int, interval: str) -> str:
    """Why a bar is, or is not, a regular-session print.

    Sub-daily bars are labeled ``regular``, ``outside_regular_hours``,
    ``weekend``, or ``holiday`` from the bar's open time. Daily and coarser
    bars are labeled ``utc_period`` because their open is UTC midnight (or a
    multi-day boundary), which is not a US session.
    """
    if interval not in SUB_DAILY_INTERVALS:
        return "utc_period"
    opened, day = et_parts(open_time_ms)
    status = cash_session_status(day)
    if status != "open":
        return status
    if REGULAR_OPEN <= opened.timetz().replace(tzinfo=None) < REGULAR_CLOSE:
        return "regular"
    return "outside_regular_hours"


def window_millis(start_date: str, end_date: str) -> tuple[int, int]:
    """Inclusive UTC millis for ``[start 00:00 ET, end 24:00 ET)``.

    Binance treats ``endTime`` as inclusive, so the returned end is one
    millisecond before the next ET midnight. A bar that opens on the following
    New York date is outside the analysis date.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=NY)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=NY) + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1
