"""Compare a configured Binance symbol with the underlying US equity close.

The ticker → symbol map is configuration. Nothing in the spot or USD-M docs
reviewed for this module states which symbols are tokenized US stocks, or that
a ticker becomes a symbol by a fixed suffix. Observations from one public read
(2026-11-03, ``GET https://data-api.binance.vision/api/v3/exchangeInfo`` and
``GET /api/v3/klines``) are not copied into the default map:

- Spot symbols such as ``AAPLBUSDT`` and ``TSLABUSDT`` were ``TRADING``, and
  klines came back. That is a catalog observation, not a naming rule. Binance
  listing notices have used different spellings for different names (Micron as
  ``MUB``, Tesla as ``TSLAB``), so a suffix must not be assumed.
- An earlier call to ``https://fapi.binance.com`` from a geo-blocked network
  returned HTTP 451, so that environment could not read ``contractType``.
  A later live ``GET /fapi/v1/exchangeInfo`` on 2026-11-03 (not geo-blocked)
  showed TSLAUSDT, NVDAUSDT, AAPLUSDT, and MUUSDT as ``contractType``
  ``TRADIFI_PERPETUAL``, ``status`` ``TRADING``, ``underlyingType`` ``EQUITY``,
  ``underlyingSubType`` ``["TradFi"]``. Funding history and open interest
  answered for MUUSDT and NVDAUSDT. Funding and open interest are requested
  for ``PERPETUAL`` and for ``TRADIFI_PERPETUAL``. The USD-M connector enum
  lists ``PERPETUAL`` and not ``TRADIFI_PERPETUAL``; the second value is
  accepted because that catalog returned it.

``/sapi/v1/equity/*`` is not called. The official stocks connector marks those
routes ``MARKET_DATA``, which requires an API key.

Premium/discount is ``(Binance close - stock close) / stock close``. A positive
number means the Binance print is above the stock. Prices are research context,
not an order.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, date
from decimal import Decimal, InvalidOperation

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.us_cash_session import (
    SUB_DAILY_INTERVALS,
    cash_session_status,
    et_parts,
    session_label,
    window_millis,
)
from tradingagents.dataflows.vendors import binance_public as public
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

logger = logging.getLogger(__name__)

_MAX_DAYS = 45
_MAX_FUNDING_ROWS = 8
_MAX_OI_ROWS = 8
# The statistics page says the latest month; the USD-M connector says 30 days.
_OI_RETENTION_DAYS = 31


def _today() -> date:
    return date.today()


def lookup_mapping(ticker: str) -> dict | None:
    """Configured spot and futures symbols for ``ticker``, or None.

    A string value is a spot symbol only. A dict may set ``spot_symbol`` and
    ``futures_symbol``. Empty strings are ignored. Keys match the run's ticker
    without regard to case.
    """
    raw_map = (get_config().get("binance_tradfi") or {}).get("symbol_map") or {}
    if not isinstance(raw_map, dict) or not ticker:
        return None
    raw = raw_map.get(ticker)
    if raw is None:
        raw = raw_map.get(ticker.upper())
    if raw is None:
        folded = {str(key).upper(): value for key, value in raw_map.items()}
        raw = folded.get(ticker.upper())
    if raw is None:
        return None
    if isinstance(raw, str):
        spot, futures = raw.strip(), ""
    elif isinstance(raw, dict):
        spot = str(raw.get("spot_symbol") or "").strip()
        futures = str(raw.get("futures_symbol") or "").strip()
    else:
        return None
    if not spot and not futures:
        return None
    return {"spot_symbol": spot.upper(), "futures_symbol": futures.upper()}


def _settings() -> dict:
    block = get_config().get("binance_tradfi") or {}
    interval = str(block.get("kline_interval") or "1h")
    return {
        "interval": interval,
        "spot_base_url": str(block.get("spot_base_url") or public.DEFAULT_SPOT_BASE),
        "futures_base_url": str(block.get("futures_base_url") or public.DEFAULT_FUTURES_BASE),
        # Default on: a ticker with no futures_symbol is looked up in the live
        # USD-M catalog. Set false to skip that call.
        "discover_futures": bool(block.get("discover_futures", True)),
    }


def _decimal(value) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _price(value: Decimal) -> str:
    """Up to four decimal places, without trailing zeros."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _pct(binance_close: Decimal, stock_close: Decimal) -> str | None:
    if stock_close == 0:
        return None
    points = (binance_close - stock_close) / stock_close * Decimal(100)
    return f"{points:+.2f}%"


def _stock_closes(ticker: str, start_date: str, end_date: str) -> tuple[dict[date, Decimal], str | None]:
    """Session closes from the existing equity source, or an error string.

    ``fill_gaps`` is false so a holiday is a missing bar, not the previous
    close copied onto that date.
    """
    try:
        frame = load_ohlcv(ticker, end_date, fill_gaps=False)
    except VendorError as exc:
        return {}, f"Underlying closes are unavailable ({exc})."
    except Exception as exc:
        logger.warning("Equity closes failed for %s: %s", ticker, exc)
        return {}, f"Underlying closes are unavailable ({exc})."
    if frame is None or frame.empty or "Close" not in frame.columns or "Date" not in frame.columns:
        return {}, "Underlying closes are unavailable (no rows)."
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    closes: dict[date, Decimal] = {}
    for stamp, close in zip(frame["Date"], frame["Close"], strict=False):
        try:
            day = stamp.date() if hasattr(stamp, "date") else date.fromisoformat(str(stamp)[:10])
        except ValueError:
            continue
        if day < start or day > end:
            continue
        parsed = _decimal(close)
        if parsed is None:
            continue
        closes[day] = parsed
    return closes, None


def _oi_period(interval: str) -> str:
    return interval if interval in public.OI_HIST_PERIODS else "1h"


def _unmapped(ticker: str) -> str:
    return (
        f"# Binance TradFi for {ticker}\n\n"
        f"No Binance symbol is configured for {ticker}. "
        f"`binance_tradfi.symbol_map` is empty for this ticker, so no Binance "
        f"endpoint was called and no premium is computed. The spot and USD-M docs "
        f"do not define a ticker-to-symbol rule for tokenized US stocks; this "
        f"report does not invent one. Set `spot_symbol` and, separately, "
        f"`futures_symbol` to symbols you have confirmed. Other data for {ticker} "
        f"is unchanged."
    )


def _fetch_price_bars(mapping: dict, settings: dict, start_ms: int, end_ms: int):
    """Return ``(bars, venue, error)``. ``venue`` is ``spot`` or ``futures``."""
    interval = settings["interval"]
    if mapping["spot_symbol"]:
        try:
            bars = public.get_spot_klines(
                mapping["spot_symbol"],
                interval,
                start_ms,
                end_ms,
                base_url=settings["spot_base_url"],
            )
            return bars, "spot", None
        except VendorError as exc:
            return [], "spot", exc
    try:
        bars = public.get_futures_klines(
            mapping["futures_symbol"],
            interval,
            start_ms,
            end_ms,
            base_url=settings["futures_base_url"],
        )
        return bars, "futures", None
    except VendorError as exc:
        return [], "futures", exc


def _onboard_ms(contract: dict) -> int | None:
    """``onboardDate`` in milliseconds, or None when the catalog omits it.

    The USD-M exchangeInfo schema includes ``onboardDate``. A missing field is
    not treated as "listed forever": a historical run will not invent a listing
    date.
    """
    raw = contract.get("onboardDate")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _contract_phrase(contract: dict) -> str:
    subs = contract.get("underlyingSubType")
    return (
        f"{contract.get('symbol')} with contractType {contract.get('contractType')} "
        f"(status {contract.get('status')}, underlyingType {contract.get('underlyingType')}, "
        f"underlyingSubType {subs})"
    )


def _funding_decision(contract: dict, end_ms: int, *, historical: bool, explicit: bool) -> tuple[bool, str]:
    """Whether funding and open interest may be read for this catalog row.

    Rows are still clipped to the analysis window by the caller. A contract
    whose ``onboardDate`` is after that window is not read at all. An explicit
    ``futures_symbol`` with no ``onboardDate`` is still read, with that gap
    stated. Auto-discovery on a historical run requires ``onboardDate``.
    """
    if not public.accepts_funding(contract):
        return False, (
            f"USD-M `exchangeInfo` lists {_contract_phrase(contract)}, which is "
            f"neither PERPETUAL nor TRADIFI_PERPETUAL. "
            f"Funding and open interest were not requested."
        )
    onboard = _onboard_ms(contract)
    if onboard is not None and onboard > end_ms:
        return False, (
            f"USD-M `exchangeInfo` lists {_contract_phrase(contract)}, but its "
            f"onboardDate is after this window. Funding and open interest were not "
            f"requested, so a later listing is not read into the analysis date."
        )
    vintage = ""
    if onboard is None and historical and explicit:
        vintage = (
            " exchangeInfo did not include onboardDate, so the listing vintage was "
            "not checked; returned rows are still limited to the window."
        )
    elif onboard is None and not historical:
        vintage = (
            " onboardDate was absent. The window ends today, so the current catalog "
            "was used."
        )
    return True, (
        f"USD-M `exchangeInfo` lists {_contract_phrase(contract)}. "
        f"Funding history and open interest were requested.{vintage}"
    )


def discover_tradfi_perpetual(
    symbols: list, ticker: str, end_ms: int, *, historical: bool
) -> tuple[dict | None, str]:
    """Pick one USD-M perpetual for ``ticker`` from a live exchangeInfo catalog.

    A row matches only when ``baseAsset`` equals the ticker, ``underlyingType``
    is ``EQUITY``, ``underlyingSubType`` contains ``TradFi``, ``status`` is
    ``TRADING``, and ``contractType`` is ``PERPETUAL`` or ``TRADIFI_PERPETUAL``.
    Those field values were observed on 2026-11-03 for TSLAUSDT, NVDAUSDT,
    AAPLUSDT, and MUUSDT. They are not a naming rule from the REST docs.
    Several matches prefer a single ``USDT`` quote; more than one remaining
    match is refused. A historical window ignores a row with no ``onboardDate``
    and a row listed after ``end_ms``.
    """
    base = ticker.upper()
    found = []
    for item in symbols:
        if not isinstance(item, dict):
            continue
        if str(item.get("baseAsset") or "").upper() != base:
            continue
        if item.get("underlyingType") != "EQUITY":
            continue
        subs = item.get("underlyingSubType") or []
        if isinstance(subs, str):
            subs = [subs]
        if "TradFi" not in subs:
            continue
        if item.get("status") != "TRADING":
            continue
        if not public.accepts_funding(item):
            continue
        found.append(item)
    usdt = [item for item in found if str(item.get("quoteAsset") or "").upper() == "USDT"]
    candidates = usdt if usdt else found
    if not candidates:
        return None, (
            f"No USD-M symbol in the current exchangeInfo catalog has baseAsset {base}, "
            f"underlyingType EQUITY, underlyingSubType TradFi, status TRADING, and "
            f"contractType PERPETUAL or TRADIFI_PERPETUAL. "
            f"Funding and open interest were not requested."
        )
    if len(candidates) > 1:
        names = ", ".join(sorted(str(item.get("symbol")) for item in candidates))
        return None, (
            f"exchangeInfo has more than one TradFi perpetual for baseAsset {base} "
            f"({names}). None was selected. Set `futures_symbol` to choose one."
        )
    contract = candidates[0]
    onboard = _onboard_ms(contract)
    if onboard is not None and onboard > end_ms:
        return None, (
            f"The current catalog lists {_contract_phrase(contract)}, but onboardDate "
            f"is after this window. It was not used."
        )
    if onboard is None and historical:
        return None, (
            f"The current catalog lists {_contract_phrase(contract)}, but it has no "
            f"onboardDate. A historical run does not assume that contract existed on "
            f"the analysis date. Funding and open interest were not requested. "
            f"Set `futures_symbol` to override."
        )
    use, sentence = _funding_decision(contract, end_ms, historical=historical, explicit=False)
    if not use:
        return None, sentence
    return contract, (
        f"Futures symbol {contract.get('symbol')} was not in `symbol_map`. {sentence} "
        f"Selection used the live catalog (baseAsset {base}, underlyingType EQUITY, "
        f"underlyingSubType TradFi), which describes today's listings."
    )


def _resolve_futures(ticker: str, configured: str, settings: dict, end_ms: int, end_date: str):
    """``(contract or None, futures_symbol, sentence, from_catalog)``.

    An explicit ``futures_symbol`` wins over catalog discovery.
    """
    historical = is_historical(end_date)
    if not configured and not settings["discover_futures"]:
        return None, "", (
            "No futures symbol is configured and futures discovery is off, so funding "
            "and open interest were not requested."
        ), False
    try:
        symbols = public.get_futures_exchange_info(base_url=settings["futures_base_url"])
    except VendorError as exc:
        logger.warning("USD-M exchangeInfo failed: %s", exc)
        if configured:
            return None, configured, (
                f"Perpetual status of {configured} was NOT confirmed ({exc}). "
                f"Funding and open interest were not requested."
            ), False
        return None, "", (
            f"Futures discovery failed ({exc}). "
            f"Funding and open interest were not requested."
        ), False
    if configured:
        contract = next(
            (item for item in symbols if str(item.get("symbol") or "").upper() == configured),
            None,
        )
        if contract is None:
            return None, configured, (
                f"{configured} is not in USD-M `GET /fapi/v1/exchangeInfo`. "
                f"Funding and open interest were not requested."
            ), False
        use, sentence = _funding_decision(
            contract, end_ms, historical=historical, explicit=True
        )
        return (contract if use else None), configured, sentence, False
    contract, sentence = discover_tradfi_perpetual(
        symbols, ticker, end_ms, historical=historical
    )
    symbol = str(contract.get("symbol") or "").upper() if contract else ""
    return contract, symbol, sentence, contract is not None


def _labeled(bars, interval: str) -> list[dict]:
    labeled = []
    for bar in bars:
        opened, day = et_parts(bar.open_time_ms)
        labeled.append(
            {
                "bar": bar,
                "day": day,
                "opened": opened,
                "label": session_label(bar.open_time_ms, interval),
                "close": _decimal(bar.close),
            }
        )
    return labeled


def _choose(day_bars: list[dict]) -> tuple[dict | None, bool]:
    usable = [item for item in day_bars if item["close"] is not None]
    if not usable:
        return None, False
    regular = [item for item in usable if item["label"] == "regular"]
    pool = regular or usable
    return max(pool, key=lambda item: item["bar"].open_time_ms), bool(regular)


def _render_days(labeled, closes: dict[date, Decimal], start: date, end: date, interval: str) -> list[str]:
    if interval not in SUB_DAILY_INTERVALS:
        return [
            "No session premium table. "
            f"Interval `{interval}` is a UTC period (the spot docs interpret `1d` in UTC "
            "unless `timeZone` is set, and a fixed offset does not follow US daylight saving). "
            "Those bars are not compared with the US cash close.",
            "",
            "| UTC open | Close | Volume |",
            "| --- | --- | --- |",
            *[
                f"| {item['opened'].astimezone(UTC).strftime('%Y-%m-%d %H:%M')} "
                f"| {item['bar'].close} | {item['bar'].volume} |"
                for item in labeled[:_MAX_DAYS]
            ],
        ]

    by_day: dict[date, list] = defaultdict(list)
    for item in labeled:
        if start <= item["day"] <= end:
            by_day[item["day"]].append(item)
    days = sorted(set(by_day) | {day for day in closes if start <= day <= end})
    hidden = 0
    if len(days) > _MAX_DAYS:
        hidden = len(days) - _MAX_DAYS
        days = days[-_MAX_DAYS:]

    counts = defaultdict(int)
    for item in labeled:
        counts[item["label"]] += 1
    closed = counts["weekend"] + counts["holiday"] + counts["outside_regular_hours"]
    lines = [
        f"Sub-daily bars in the window: {len(labeled)}. "
        f"Opened inside 09:30–16:00 America/New_York on a cash session: {counts['regular']}. "
        f"Opened while the US cash session was closed (weekend, full-day holiday, or outside "
        f"those hours): {closed}. "
        "A 1h bar is aligned to UTC. The hour that opens at 09:00 ET is labeled outside "
        "the regular session because 09:30 falls inside it, not at its open.",
        "",
        "| Date (ET) | US cash session | Stock close | Binance close | Binance bar open (ET) | Premium/discount | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if hidden:
        lines.append(f"({hidden} earlier date(s) omitted; the latest {_MAX_DAYS} are shown.)")
    cursor = None
    for day in sorted(closes):
        if days and day < days[0]:
            cursor = (day, closes[day])
    for day in days:
        if day in closes:
            cursor = (day, closes[day])
        status = cash_session_status(day)
        day_bars = by_day.get(day) or []
        if not day_bars:
            lines.append(
                f"| {day.isoformat()} | {status} | "
                f"{_price(closes[day]) if day in closes else ''} |  |  |  | "
                f"no Binance bar with an open on this New York date |"
            )
            continue
        chosen, used_regular = _choose(day_bars)
        if chosen is None:
            lines.append(
                f"| {day.isoformat()} | {status} | "
                f"{_price(closes[day]) if day in closes else ''} |  |  |  | unreadable Binance close |"
            )
            continue
        binance_close = chosen["close"]
        stock_cell = ""
        premium = ""
        if status == "open" and day in closes and binance_close is not None:
            stock_cell = _price(closes[day])
            premium = _pct(binance_close, closes[day]) or ""
            note = (
                "regular-session bar (last 1h open before 16:00 ET; not the closing auction)"
                if used_regular
                else "cash session was open; no regular-session bar, so the last bar that New York date is shown"
            )
        elif status == "open" and binance_close is not None and cursor is not None:
            stock_cell = _price(cursor[1])
            premium = _pct(binance_close, cursor[1]) or ""
            when = "regular-session bar" if used_regular else "bar opened outside 09:30–16:00 ET"
            note = (
                f"no equity close for this date yet; premium uses last stock close {cursor[0].isoformat()}. "
                f"Binance {when}."
            )
        elif binance_close is not None and cursor is not None:
            stock_cell = _price(cursor[1])
            premium = _pct(binance_close, cursor[1]) or ""
            ref = "same session" if cursor[0] == day else f"last stock close {cursor[0].isoformat()}"
            note = (
                f"US cash session {status}; premium uses the {ref}, not a simultaneous print. "
                f"The token can trade while the stock does not."
            )
        else:
            note = f"US cash session {status}; no stock close to compare"
        opened = chosen["opened"].strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"| {day.isoformat()} | {status} | {stock_cell} | {_price(binance_close) if binance_close is not None else ''} "
            f"| {opened} | {premium} | {note} |"
        )
    return lines


def _render_funding(rows: list[dict], error: str | None) -> list[str]:
    lines = ["", "## Funding rate history", ""]
    if error:
        lines.append(error)
        return lines
    if not rows:
        lines.append("`GET /fapi/v1/fundingRate` returned no settlements in the window.")
        return lines
    rates = []
    for item in rows:
        parsed = _decimal(item.get("fundingRate"))
        if parsed is not None:
            rates.append(parsed)
    if rates:
        average = sum(rates, Decimal(0)) / Decimal(len(rates))
        lines.append(
            f"{len(rows)} settlement(s) in the window. "
            f"Simple average of the published fundingRate values: {average:.8f} "
            f"({average * 100:.4f}% per settlement). Not annualized."
        )
    lines += [
        "",
        "| Funding time (ET) | fundingRate | As percent of notional | markPrice |",
        "| --- | --- | --- | --- |",
    ]
    shown = rows[-_MAX_FUNDING_ROWS:]
    if len(rows) > _MAX_FUNDING_ROWS:
        lines.append(f"(showing the latest {_MAX_FUNDING_ROWS} of {len(rows)}.)")
    for item in shown:
        opened, _day = et_parts(int(item["fundingTime"]))
        rate = _decimal(item.get("fundingRate"))
        percent = f"{rate * 100:.4f}%" if rate is not None else ""
        lines.append(
            f"| {opened.strftime('%Y-%m-%d %H:%M')} | {item.get('fundingRate')} | {percent} | {item.get('markPrice', '')} |"
        )
    return lines


def _render_oi(rows: list[dict], error: str | None, end: date, current: dict | None, current_note: str) -> list[str]:
    lines = ["", "## Open interest", ""]
    lines.append(current_note)
    if current:
        lines.append(
            f"Present `GET /fapi/v1/openInterest`: openInterest={current.get('openInterest')} "
            f"time={current.get('time')} symbol={current.get('symbol')}. "
            f"This is a live snapshot."
        )
    if error:
        lines.append(error)
        return lines
    if not rows:
        age = (_today() - end).days
        if age > _OI_RETENTION_DAYS:
            lines.append(
                "`GET /futures/data/openInterestHist` returned no rows. Binance documents "
                "this route as only the latest month (the USD-M connector says 30 days), "
                "so an empty older window is not a reading of zero open interest."
            )
        else:
            lines.append("`GET /futures/data/openInterestHist` returned no rows in the window.")
        return lines
    first = _decimal(rows[0].get("sumOpenInterest"))
    last = _decimal(rows[-1].get("sumOpenInterest"))
    if first is not None and last is not None:
        lines.append(
            f"{len(rows)} open-interest point(s) in the window. "
            f"sumOpenInterest { _price(first) } → { _price(last) }."
        )
    lines += [
        "",
        "| Time (ET) | sumOpenInterest | sumOpenInterestValue |",
        "| --- | --- | --- |",
    ]
    shown = rows[-_MAX_OI_ROWS:]
    if len(rows) > _MAX_OI_ROWS:
        lines.append(f"(showing the latest {_MAX_OI_ROWS} of {len(rows)}.)")
    for item in shown:
        opened, _day = et_parts(int(item["timestamp"]))
        lines.append(
            f"| {opened.strftime('%Y-%m-%d %H:%M')} | {item.get('sumOpenInterest')} | "
            f"{item.get('sumOpenInterestValue')} |"
        )
    return lines


def get_binance_tradfi_premium(symbol: str, start_date: str, end_date: str) -> str:
    """Premium/discount of a Binance symbol versus the equity close.

    ``symbol`` is the run's ticker, not a Binance symbol. Spot symbols come
    only from ``binance_tradfi.symbol_map``. A futures symbol in that map wins.
    Otherwise, when ``discover_futures`` is on, one USD-M contract is taken
    from the live exchangeInfo catalog.
    """
    settings = _settings()
    if settings["interval"] not in public.KLINE_INTERVALS:
        return (
            f"# Binance TradFi for {symbol}\n\n"
            f"kline_interval {settings['interval']!r} is not a documented interval. "
            f"No endpoint was called."
        )
    configured = lookup_mapping(symbol) or {"spot_symbol": "", "futures_symbol": ""}
    if not configured["spot_symbol"] and not configured["futures_symbol"] and not settings["discover_futures"]:
        return _unmapped(symbol)

    start_ms, end_ms = window_millis(start_date, end_date)
    contract, futures_symbol, perpetual_sentence, from_catalog = _resolve_futures(
        symbol, configured["futures_symbol"], settings, start_ms, end_date
    )
    mapping = {"spot_symbol": configured["spot_symbol"], "futures_symbol": futures_symbol}
    if not mapping["spot_symbol"] and not mapping["futures_symbol"]:
        return (
            f"# Binance TradFi for {symbol}\n\n"
            f"{perpetual_sentence}\n\n"
            f"No spot symbol is configured, so no premium was computed."
        )

    bars, venue, price_error = _fetch_price_bars(mapping, settings, start_ms, end_ms)
    is_perp = contract is not None and public.accepts_funding(contract)

    funding: list[dict] = []
    funding_error = None
    oi_rows: list[dict] = []
    oi_error = None
    current_oi = None
    if is_perp:
        try:
            funding = public.get_funding_rates(
                futures_symbol, start_ms, end_ms, base_url=settings["futures_base_url"]
            )
        except VendorError as exc:
            funding_error = f"Funding history failed ({exc})."
        try:
            oi_rows = public.get_open_interest_hist(
                futures_symbol,
                _oi_period(settings["interval"]),
                start_ms,
                end_ms,
                base_url=settings["futures_base_url"],
            )
        except VendorError as exc:
            oi_error = f"Open-interest history failed ({exc})."
        if is_historical(end_date):
            current_note = (
                "Present open interest (`GET /fapi/v1/openInterest`) is a live snapshot "
                f"with no historical vintage, so it is withheld for {end_date}."
            )
        else:
            current_note = "Present open interest is included because the window ends today."
            try:
                current_oi = public.get_open_interest(
                    futures_symbol, base_url=settings["futures_base_url"]
                )
            except VendorError as exc:
                current_note = f"Present open interest failed ({exc})."
    else:
        current_note = "Present open interest was not requested."

    if not bars and price_error is not None and not is_perp:
        raise price_error

    closes, stock_error = _stock_closes(symbol, start_date, end_date)
    labeled = _labeled(bars, settings["interval"])
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if venue == "spot":
        price_source = (
            f"Token price: `GET {public.SPOT_KLINES_PATH}` on `{settings['spot_base_url']}` "
            f"symbol `{mapping['spot_symbol']}` interval `{settings['interval']}`."
        )
    else:
        price_source = (
            f"Price: `GET {public.FUTURES_KLINES_PATH}` on `{settings['futures_base_url']}` "
            f"symbol `{mapping['futures_symbol']}` interval `{settings['interval']}`. "
            f"This is the USD-M contract kline, not a confirmed spot-token price."
        )
    if from_catalog:
        origin = (
            "The futures symbol was selected from the current exchangeInfo catalog, "
            "not from a documented ticker-to-symbol rule."
        )
    elif mapping["futures_symbol"] or mapping["spot_symbol"]:
        origin = "The symbol was taken from config. It is not a documented ticker-to-symbol rule."
    else:
        origin = "No futures symbol was configured or discovered."

    lines = [
        f"# Binance TradFi vs underlying for {symbol}",
        f"# Window: {start_date} to {end_date} (America/New_York calendar dates)",
        "",
        price_source,
        perpetual_sentence,
        "Premium/discount = (Binance close − stock close) / stock close. "
        "Positive means the Binance print is above the stock.",
        origin,
        "",
    ]
    if price_error is not None:
        lines.append(f"Binance klines failed ({price_error}).")
    if stock_error:
        lines.append(stock_error)
    lines.append("")
    lines.extend(_render_days(labeled, closes, start, end, settings["interval"]))
    if is_perp:
        lines.extend(_render_funding(funding, funding_error))
        lines.extend(_render_oi(oi_rows, oi_error, end, current_oi, current_note))
    elif mapping["futures_symbol"]:
        lines += ["", "## Funding rate and open interest", "", current_note, perpetual_sentence]
    lines += [
        "",
        "## Sources and limits",
        "",
        "- Spot klines: `GET /api/v3/klines` "
        "(https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data), "
        f"host `{settings['spot_base_url']}`.",
        "- USD-M klines, `exchangeInfo`, funding history, present open interest, "
        "and open-interest history use the public routes on "
        f"`{settings['futures_base_url']}`: `/fapi/v1/klines`, `/fapi/v1/exchangeInfo`, "
        "`/fapi/v1/fundingRate`, `/fapi/v1/openInterest`, `/futures/data/openInterestHist`.",
        "- Funding and open interest are requested only when `exchangeInfo` lists the "
        "futures symbol with `contractType` PERPETUAL or TRADIFI_PERPETUAL. "
        "TRADIFI_PERPETUAL was observed on 2026-11-03 for TSLAUSDT, NVDAUSDT, "
        "AAPLUSDT, and MUUSDT (`underlyingType` EQUITY, `underlyingSubType` "
        "[TradFi]). An explicit `futures_symbol` overrides catalog discovery. "
        "A historical run does not adopt a discovered contract that lacks "
        "`onboardDate` or whose `onboardDate` is after the window.",
        "- `/sapi/v1/equity/*` is not called. Those routes require an API key.",
        "- No order, account, or signed endpoint is called.",
        "- Equity closes come from the existing Yahoo path, filtered through the window end.",
        "- Full-day US holidays follow the standard NYSE rules (weekends, and New Year, "
        "MLK Day, Washington's Birthday, Good Friday, Memorial Day, Juneteenth from 2021, "
        "Independence Day, Labor Day, Thanksgiving, Christmas, with weekend observance). "
        "Early closes are not marked closed. A one-off halt is not flagged.",
        "",
    ]
    return "\n".join(lines)
