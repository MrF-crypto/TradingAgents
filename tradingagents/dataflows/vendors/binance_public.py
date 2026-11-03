"""Public Binance market data. No API key, no signature, no account call.

Paths and parameters below are the ones published by Binance. This module does
not call ``/sapi/`` (the stocks endpoints require an ``X-MBX-APIKEY`` header
even for market data) and does not call any order or account route.

Spot klines
    GET /api/v3/klines
    https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
    https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
    Market-data-only host (no API key):
    https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md

USD-M futures (base host https://fapi.binance.com)
    https://developers.binance.com/docs/derivatives/usds-margined-futures
    GET /fapi/v1/klines
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
    GET /fapi/v1/exchangeInfo
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information
    GET /fapi/v1/fundingRate
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
    GET /fapi/v1/openInterest
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest
    GET /futures/data/openInterestHist
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics

The official USD-M connector documents the same paths:
https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py

What this module does not know
    The spot and USD-M docs describe ``symbol`` as the exchange's trading
    symbol (their examples are pairs such as BTCUSDT). They do not say which
    symbols, if any, are tokenized US stocks, and they do not say that a stock
    ticker maps to a symbol by adding a letter. ``contractType`` in the USD-M
    connector enum includes PERPETUAL, CURRENT_MONTH, NEXT_MONTH,
    CURRENT_QUARTER, and NEXT_QUARTER.

    A live ``GET /fapi/v1/exchangeInfo`` on 2026-11-03 (from a network that is
    not geo-blocked) returned ``contractType`` ``TRADIFI_PERPETUAL``,
    ``status`` ``TRADING``, ``underlyingType`` ``EQUITY``, and
    ``underlyingSubType`` ``["TradFi"]`` for TSLAUSDT, NVDAUSDT, AAPLUSDT, and
    MUUSDT. ``GET /fapi/v1/fundingRate`` and ``GET /fapi/v1/openInterest``
    answered for MUUSDT and NVDAUSDT. ``TRADIFI_PERPETUAL`` is therefore
    treated as a perpetual alongside ``PERPETUAL``. The connector enum above
    does not list ``TRADIFI_PERPETUAL``; acceptance is from that catalog read,
    not from a naming rule in the docs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

logger = logging.getLogger(__name__)

# Documented market-data hosts. Overridable from config so a test can point
# both at one stub, and so a caller can select another published spot host
# (api.binance.com and api1–api4). The default spot host is the keyless
# market-data host from the spot docs.
DEFAULT_SPOT_BASE = "https://data-api.binance.vision"
DEFAULT_FUTURES_BASE = "https://fapi.binance.com"

SPOT_KLINES_PATH = "/api/v3/klines"
FUTURES_KLINES_PATH = "/fapi/v1/klines"
FUTURES_EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"
FUTURES_FUNDING_PATH = "/fapi/v1/fundingRate"
FUTURES_OPEN_INTEREST_PATH = "/fapi/v1/openInterest"
FUTURES_OPEN_INTEREST_HIST_PATH = "/futures/data/openInterestHist"

# Spot rest-api.md: kline limit default 500, maximum 1000.
# The USD-M connector documents kline limit default 500, maximum 1000.
# (A third-party OpenAPI mirror says 1500; this client stays at the connector's 1000.)
# Funding history: default 100, maximum 1000 (connector and the funding-rate page).
# openInterestHist: default 30, maximum 500; period enum as on the statistics page.
# The statistics page says only the latest month is available; the connector says 30 days.
SPOT_KLINE_LIMIT = 1000
FUTURES_KLINE_LIMIT = 1000
FUNDING_LIMIT = 1000
OI_HIST_LIMIT = 500
OI_HIST_PERIODS = frozenset({"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"})

# Intervals published for USD-M klines. Spot also publishes ``1s``, which this
# client does not request: a second-level pull is not a session comparison.
# PERPETUAL is the connector enum. TRADIFI_PERPETUAL was observed on
# 2026-11-03 for USDT-margined equity contracts (underlyingSubType TradFi).
FUNDING_CONTRACT_TYPES = frozenset({"PERPETUAL", "TRADIFI_PERPETUAL"})

KLINE_INTERVALS = frozenset(
    {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    }
)

REQUEST_TIMEOUT = 30
# A window of hourly bars is a few hundred rows. Twenty pages is far past the
# documented limit and stops a cursor that does not advance.
_MAX_PAGES = 20


@dataclass(frozen=True)
class Kline:
    """One kline. Field order matches the published array layout."""

    open_time_ms: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time_ms: int
    trades: int


def _binance_error(response: requests.Response) -> tuple[int | None, str]:
    try:
        body = response.json()
    except ValueError:
        return None, (response.text or "")[:200]
    if isinstance(body, dict) and "code" in body and "msg" in body:
        try:
            code = int(body["code"])
        except (TypeError, ValueError):
            code = None
        return code, str(body.get("msg") or "")
    return None, ""


def public_get(base_url: str, path: str, params: dict | None = None) -> Any:
    """GET a public path. Raises a vendor error; never attaches an API key."""
    url = base_url.rstrip("/") + path
    query = {key: value for key, value in (params or {}).items() if value is not None}
    try:
        response = requests.get(url, params=query, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"{path} request failed: {exc}") from exc

    code, message = _binance_error(response)
    restricted = code == 0 and "restricted location" in message.lower()
    if response.status_code == 451 or restricted:
        raise VendorUnavailableError(
            f"{path} HTTP {response.status_code}: Binance refused the public "
            f"request from this location ({message or 'restricted location'})"
        )
    if code == -1121:
        symbol = str(query.get("symbol") or "")
        raise NoMarketDataError(symbol, symbol, message or "Invalid symbol.")
    if response.status_code >= 400 or (code is not None and code < 0):
        detail = message or (response.text or "")[:200]
        raise VendorUnavailableError(
            f"{path} HTTP {response.status_code}: {detail}".strip()
        )
    try:
        return response.json()
    except ValueError as exc:
        raise VendorUnavailableError(f"{path} returned non-JSON") from exc


def _parse_kline(row: list) -> Kline | None:
    # Spot docs: [open time, open, high, low, close, volume, close time, quote
    # volume, trade count, taker buy base, taker buy quote, ignore].
    # USD-M uses the same array. A short row is ignored rather than guessed.
    if not isinstance(row, list) or len(row) < 9:
        return None
    try:
        return Kline(
            open_time_ms=int(row[0]),
            open=str(row[1]),
            high=str(row[2]),
            low=str(row[3]),
            close=str(row[4]),
            volume=str(row[5]),
            close_time_ms=int(row[6]),
            trades=int(row[8]),
        )
    except (TypeError, ValueError):
        return None


def get_klines(
    base_url: str,
    path: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
) -> list[Kline]:
    """Klines for ``symbol`` with ``startTime`` and ``endTime`` in milliseconds.

    Pages forward from each page's last open time. Rows outside
    ``[start_ms, end_ms]`` are dropped so a bar the exchange included past the
    analysis date cannot reach the report.
    """
    if interval not in KLINE_INTERVALS:
        raise ValueError(
            f"interval {interval!r} is not a documented spot/USD-M kline interval"
        )
    symbol = symbol.upper()
    rows: list[Kline] = []
    seen: set[int] = set()
    cursor = start_ms
    for _ in range(_MAX_PAGES):
        page = public_get(
            base_url,
            path,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        if not isinstance(page, list) or not page:
            break
        parsed = [item for item in (_parse_kline(row) for row in page) if item is not None]
        if not parsed:
            break
        for item in parsed:
            if item.open_time_ms < start_ms or item.open_time_ms > end_ms:
                continue
            if item.open_time_ms in seen:
                continue
            seen.add(item.open_time_ms)
            rows.append(item)
        last_open = max(item.open_time_ms for item in parsed)
        nxt = last_open + 1
        if nxt <= cursor or len(page) < limit or last_open >= end_ms:
            break
        cursor = nxt
    rows.sort(key=lambda item: item.open_time_ms)
    return rows


def get_spot_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = DEFAULT_SPOT_BASE,
) -> list[Kline]:
    return get_klines(
        base_url, SPOT_KLINES_PATH, symbol, interval, start_ms, end_ms, limit=SPOT_KLINE_LIMIT
    )


def get_futures_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = DEFAULT_FUTURES_BASE,
) -> list[Kline]:
    return get_klines(
        base_url,
        FUTURES_KLINES_PATH,
        symbol,
        interval,
        start_ms,
        end_ms,
        limit=FUTURES_KLINE_LIMIT,
    )


def accepts_funding(contract: dict | None) -> bool:
    """True for contract types that publish funding and open interest.

    ``PERPETUAL`` is the documented enum value. ``TRADIFI_PERPETUAL`` is
    included because a 2026-11-03 ``exchangeInfo`` read returned it for
    TSLAUSDT, NVDAUSDT, AAPLUSDT, and MUUSDT, and funding plus open interest
    answered for MUUSDT and NVDAUSDT.
    """
    return bool(contract) and contract.get("contractType") in FUNDING_CONTRACT_TYPES


def get_futures_exchange_info(*, base_url: str = DEFAULT_FUTURES_BASE) -> list[dict]:
    """Every symbol in ``GET /fapi/v1/exchangeInfo``.

    The official connector takes no symbol filter, so the full catalog is read.
    """
    payload = public_get(base_url, FUTURES_EXCHANGE_INFO_PATH, None)
    if not isinstance(payload, dict):
        raise VendorUnavailableError(f"{FUTURES_EXCHANGE_INFO_PATH} did not return an object")
    symbols = payload.get("symbols") or []
    if not isinstance(symbols, list):
        raise VendorUnavailableError(f"{FUTURES_EXCHANGE_INFO_PATH} symbols was not a list")
    return [item for item in symbols if isinstance(item, dict)]


def get_futures_contract(symbol: str, *, base_url: str = DEFAULT_FUTURES_BASE) -> dict | None:
    """The USD-M ``exchangeInfo`` entry for ``symbol``, or None when it is absent.

    Absence is not the same as a perpetual: callers must read ``contractType``.
    """
    wanted = symbol.upper()
    for item in get_futures_exchange_info(base_url=base_url):
        if str(item.get("symbol") or "").upper() == wanted:
            return item
    return None


def get_funding_rates(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = DEFAULT_FUTURES_BASE,
) -> list[dict]:
    """Funding settlements in ``[start_ms, end_ms]``, oldest first.

    Only call this after ``accepts_funding`` is true (``PERPETUAL`` or
    ``TRADIFI_PERPETUAL``).
    """
    symbol = symbol.upper()
    rows: list[dict] = []
    seen: set[int] = set()
    cursor = start_ms
    for _ in range(_MAX_PAGES):
        page = public_get(
            base_url,
            FUTURES_FUNDING_PATH,
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FUNDING_LIMIT,
            },
        )
        if not isinstance(page, list) or not page:
            break
        times = []
        for item in page:
            if not isinstance(item, dict) or "fundingTime" not in item:
                continue
            try:
                when = int(item["fundingTime"])
            except (TypeError, ValueError):
                continue
            times.append(when)
            if when < start_ms or when > end_ms or when in seen:
                continue
            seen.add(when)
            rows.append(item)
        if not times:
            break
        last = max(times)
        nxt = last + 1
        if nxt <= cursor or len(page) < FUNDING_LIMIT or last >= end_ms:
            break
        cursor = nxt
    rows.sort(key=lambda item: int(item["fundingTime"]))
    return rows


def get_open_interest_hist(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = DEFAULT_FUTURES_BASE,
) -> list[dict]:
    """Historical open interest. Only the latest month is served.

    See the statistics page (latest month) and the USD-M connector (30 days).
    An empty list from an older window is that limit, not a zero reading.
    """
    if period not in OI_HIST_PERIODS:
        period = "1h"
    symbol = symbol.upper()
    rows: list[dict] = []
    seen: set[int] = set()
    cursor = start_ms
    for _ in range(_MAX_PAGES):
        page = public_get(
            base_url,
            FUTURES_OPEN_INTEREST_HIST_PATH,
            {
                "symbol": symbol,
                "period": period,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": OI_HIST_LIMIT,
            },
        )
        if not isinstance(page, list) or not page:
            break
        times = []
        for item in page:
            if not isinstance(item, dict) or "timestamp" not in item:
                continue
            try:
                when = int(item["timestamp"])
            except (TypeError, ValueError):
                continue
            times.append(when)
            if when < start_ms or when > end_ms or when in seen:
                continue
            seen.add(when)
            rows.append(item)
        if not times:
            break
        last = max(times)
        nxt = last + 1
        if nxt <= cursor or len(page) < OI_HIST_LIMIT or last >= end_ms:
            break
        cursor = nxt
    rows.sort(key=lambda item: int(item["timestamp"]))
    return rows


def get_open_interest(symbol: str, *, base_url: str = DEFAULT_FUTURES_BASE) -> dict:
    """Present open interest. This is a live snapshot, not a historical vintage."""
    payload = public_get(
        base_url, FUTURES_OPEN_INTEREST_PATH, {"symbol": symbol.upper()}
    )
    if not isinstance(payload, dict):
        raise VendorUnavailableError(f"{FUTURES_OPEN_INTEREST_PATH} did not return an object")
    return payload


def smoke_public(spot_symbol: str = "BTCUSDT", futures_symbol: str | None = None) -> str:
    """One live read of the public endpoints. No API key and no order call.

    ``BTCUSDT`` is the example pair in Binance's own kline docs, not a stock.
    Any other symbol is only a probe the caller asked for. A futures HTTP 451
    means this network could not read ``contractType``; it is not a finding
    that the contract does or does not exist.
    """
    lines = [
        "Binance public smoke (no API key).",
        f"Spot host {DEFAULT_SPOT_BASE} GET {SPOT_KLINES_PATH} symbol={spot_symbol} interval=1d limit via start/end omitted, limit=2.",
    ]
    try:
        page = public_get(
            DEFAULT_SPOT_BASE,
            SPOT_KLINES_PATH,
            {"symbol": spot_symbol.upper(), "interval": "1d", "limit": 2},
        )
    except (VendorUnavailableError, NoMarketDataError) as exc:
        lines.append(f"spot klines: {type(exc).__name__}: {exc}")
    else:
        parsed = [item for item in (_parse_kline(row) for row in page or []) if item]
        lines.append(f"spot klines: {len(parsed)} bar(s)")
        for item in parsed:
            lines.append(
                f"  open_ms={item.open_time_ms} close={item.close} volume={item.volume}"
            )
    if not futures_symbol:
        lines.append("futures: not probed")
        return "\n".join(lines)
    lines.append(
        f"Futures host {DEFAULT_FUTURES_BASE} GET {FUTURES_EXCHANGE_INFO_PATH} "
        f"looking for {futures_symbol} (full catalog; the connector takes no symbol filter)."
    )
    try:
        contract = get_futures_contract(futures_symbol)
    except (VendorUnavailableError, NoMarketDataError) as exc:
        lines.append(f"futures exchangeInfo: {type(exc).__name__}: {exc}")
        lines.append(
            "Perpetual status was not confirmed, so funding and open interest were not requested."
        )
        return "\n".join(lines)
    if contract is None:
        lines.append(
            f"futures exchangeInfo: {futures_symbol} is not in the catalog. "
            "Funding and open interest were not requested."
        )
        return "\n".join(lines)
    contract_type = contract.get("contractType")
    lines.append(
        f"futures exchangeInfo: symbol={contract.get('symbol')} "
        f"contractType={contract_type} status={contract.get('status')} "
        f"underlyingType={contract.get('underlyingType')}"
    )
    if not accepts_funding(contract):
        lines.append(
            "contractType is neither PERPETUAL nor TRADIFI_PERPETUAL, "
            "so funding and open interest were not requested."
        )
        return "\n".join(lines)
    try:
        funding = public_get(
            DEFAULT_FUTURES_BASE,
            FUTURES_FUNDING_PATH,
            {"symbol": futures_symbol.upper(), "limit": 1},
        )
        lines.append(f"fundingRate latest row: {funding[-1] if isinstance(funding, list) and funding else funding}")
    except (VendorUnavailableError, NoMarketDataError) as exc:
        lines.append(f"fundingRate: {type(exc).__name__}: {exc}")
    try:
        oi = get_open_interest(futures_symbol)
        lines.append(
            f"openInterest: openInterest={oi.get('openInterest')} time={oi.get('time')}"
        )
    except (VendorUnavailableError, NoMarketDataError) as exc:
        lines.append(f"openInterest: {type(exc).__name__}: {exc}")
    return "\n".join(lines)
