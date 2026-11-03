"""Binance TradFi research reader: public market data only, mocked HTTP.

The default symbol map is empty. A normal ticker must not call Binance.
Funding and open interest are requested only after exchangeInfo reports
contractType PERPETUAL or TRADIFI_PERPETUAL.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.binance_public as public
import tradingagents.dataflows.vendors.binance_tradfi as tradfi
from tradingagents.agents import tools
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.us_cash_session import (
    cash_session_status,
    easter_sunday,
    nyse_full_day_holidays,
    session_label,
)

NY = ZoneInfo("America/New_York")
SPOT = "https://spot.test"
FUTURES = "https://futures.test"


def _ms(year, month, day, hour, minute=0) -> int:
    opened = datetime(year, month, day, hour, minute, tzinfo=NY)
    return int(opened.timestamp() * 1000)


def _kline(open_ms: int, close: str, volume: str = "10") -> list:
    return [
        open_ms, close, close, close, close, volume,
        open_ms + 3_600_000 - 1, "0", 3, "0", "0", "0",
    ]


def _response(status: int, payload):
    class Response:
        status_code = status
        text = payload if isinstance(payload, str) else json.dumps(payload)

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    return Response()


def _stock_frame():
    return pd.DataFrame({
        "Date": [pd.Timestamp("2026-07-02")],
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [10.0],
    })


def _map(**overrides):
    block = {
        "symbol_map": {"AAPL": {"spot_symbol": "AAPLBUSDT"}},
        "kline_interval": "1h",
        "spot_base_url": SPOT,
        "futures_base_url": FUTURES,
        "discover_futures": False,
    }
    block.update(overrides)
    set_config({"binance_tradfi": block})


@pytest.mark.unit
class TestCashSession:
    def test_easter_and_good_friday(self):
        assert easter_sunday(2024) == date(2024, 3, 31)
        assert easter_sunday(2025) == date(2025, 4, 20)
        holidays = nyse_full_day_holidays(2024)
        assert date(2024, 3, 29) in holidays  # Good Friday

    def test_fixed_and_floating_holidays(self):
        holidays = nyse_full_day_holidays(2024)
        assert date(2024, 1, 1) in holidays
        assert date(2024, 1, 15) in holidays   # MLK, third Monday
        assert date(2024, 2, 19) in holidays   # Washington's Birthday
        assert date(2024, 5, 27) in holidays   # Memorial Day
        assert date(2024, 6, 19) in holidays   # Juneteenth
        assert date(2024, 7, 4) in holidays
        assert date(2024, 9, 2) in holidays    # Labor Day
        assert date(2024, 11, 28) in holidays  # Thanksgiving
        assert date(2024, 12, 25) in holidays

    def test_weekend_observation(self):
        # 2026-07-04 is a Saturday, so the cash market closes Friday 2026-07-03.
        assert date(2026, 7, 4).weekday() == 5
        assert cash_session_status(date(2026, 7, 3)) == "holiday"
        assert cash_session_status(date(2026, 7, 4)) == "weekend"
        # 2022-01-01 is a Saturday, observed Friday 2021-12-31.
        assert date(2022, 1, 1).weekday() == 5
        assert cash_session_status(date(2021, 12, 31)) == "holiday"

    def test_a_regular_weekday_is_open(self):
        assert date(2026, 7, 2).weekday() == 3
        assert cash_session_status(date(2026, 7, 2)) == "open"
        assert date(2026, 11, 3).weekday() == 1
        assert cash_session_status(date(2026, 11, 3)) == "open"

    def test_bar_open_classifies_the_session(self):
        assert session_label(_ms(2026, 7, 2, 15), "1h") == "regular"
        assert session_label(_ms(2026, 7, 2, 9), "1h") == "outside_regular_hours"
        assert session_label(_ms(2026, 7, 2, 20), "1h") == "outside_regular_hours"
        assert session_label(_ms(2026, 7, 3, 15), "1h") == "holiday"
        assert session_label(_ms(2026, 7, 4, 12), "1h") == "weekend"
        assert session_label(_ms(2026, 7, 2, 0), "1d") == "utc_period"


@pytest.mark.unit
class TestMapping:
    def test_discovery_off_does_not_call_binance(self):
        set_config({"binance_tradfi": {"discover_futures": False, "symbol_map": {}}})

        def refuse(*_a, **_k):
            raise AssertionError("Binance was called while discovery is off and the map is empty")

        with mock.patch.object(public.requests, "get", refuse), \
                mock.patch.object(tradfi, "load_ohlcv", refuse):
            out = tradfi.get_binance_tradfi_premium("NVDA", "2026-07-02", "2026-07-04")
        assert "No Binance symbol is configured for NVDA" in out
        assert "does not invent" in out

    def test_string_value_is_a_spot_symbol_only(self):
        _map(symbol_map={"aapl": "aaplusdt"})
        found = tradfi.lookup_mapping("AAPL")
        assert found == {"spot_symbol": "AAPLUSDT", "futures_symbol": ""}

    def test_partial_config_keeps_the_default_interval(self):
        set_config({"binance_tradfi": {"symbol_map": {"AAPL": "AAPLBUSDT"}}})
        settings = tradfi._settings()
        assert settings["interval"] == "1h"
        assert settings["spot_base_url"] == public.DEFAULT_SPOT_BASE


@pytest.mark.unit
class TestPremium:
    def _bars(self):
        return [
            _kline(_ms(2026, 7, 2, 15), "200"),
            _kline(_ms(2026, 7, 2, 20), "201"),
            _kline(_ms(2026, 7, 3, 15), "210"),
            _kline(_ms(2026, 7, 4, 12), "220"),
            _kline(_ms(2026, 7, 5, 15), "99999"),
        ]

    def test_premium_uses_the_regular_bar_and_flags_closures(self, monkeypatch):
        _map()
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            assert "signature" not in (params or {})
            assert "timestamp" not in (params or {})
            return _response(200, self._bars())

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")

        assert calls and all(url.startswith(SPOT + "/api/v3/klines") for url, _ in calls)
        assert all("/sapi/" not in url for url, _ in calls)
        assert calls[0][1]["symbol"] == "AAPLBUSDT"
        assert calls[0][1]["interval"] == "1h"
        assert "startTime" in calls[0][1] and "endTime" in calls[0][1]
        july2 = next(line for line in out.splitlines() if line.startswith("| 2026-07-02"))
        assert "| 200 |" in july2
        assert "201" not in july2
        assert "+100.00%" in july2
        assert "regular-session" in july2
        july3 = next(line for line in out.splitlines() if line.startswith("| 2026-07-03"))
        assert "holiday" in july3 and "210" in july3 and "+110.00%" in july3
        assert "not a simultaneous print" in july3
        july4 = next(line for line in out.splitlines() if line.startswith("| 2026-07-04"))
        assert "weekend" in july4 and "220" in july4
        assert "99999" not in out
        assert "GET /api/v3/klines" in out
        assert "not a documented ticker-to-symbol rule" in out

    def test_an_open_date_without_a_stock_close_is_not_called_a_session_print(self, monkeypatch):
        # The cash date can be a weekday before that session's close exists.
        _map()
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())

        def fake_get(url, params=None, timeout=None):
            return _response(200, [_kline(_ms(2026, 7, 6, 7), "90")])

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-06")
        july6 = next(line for line in out.splitlines() if line.startswith("| 2026-07-06"))
        assert "no equity close for this date yet" in july6
        assert "outside 09:30" in july6
        assert "2026-07-02" in july6

    def test_klines_are_paged_with_the_documented_parameters(self):
        _map()
        first = [_kline(_ms(2026, 7, 2, 10), "200"), _kline(_ms(2026, 7, 2, 11), "201")]
        second = [_kline(_ms(2026, 7, 2, 12), "202")]
        seen = []

        def fake_get(url, params=None, timeout=None):
            seen.append(params)
            return _response(200, first if len(seen) == 1 else second)

        with mock.patch.object(public, "SPOT_KLINE_LIMIT", 2), \
                mock.patch.object(public.requests, "get", fake_get):
            rows = public.get_spot_klines(
                "AAPLBUSDT", "1h", _ms(2026, 7, 2, 10), _ms(2026, 7, 2, 16), base_url=SPOT
            )
        assert [row.close for row in rows] == ["200", "201", "202"]
        assert seen[1]["startTime"] == _ms(2026, 7, 2, 11) + 1
        assert seen[0]["limit"] == 2
        assert set(seen[0]) == {"symbol", "interval", "startTime", "endTime", "limit"}


@pytest.mark.unit
class TestPerpetualGate:
    def _spot_get(self, contract_type, funding_rate="0.00010000", extra_funding=None):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            if url.endswith("/api/v3/klines"):
                return _response(200, [_kline(_ms(2026, 7, 2, 15), "200")])
            if url.endswith("/fapi/v1/exchangeInfo"):
                if contract_type == "HTTP451":
                    return _response(451, {
                        "code": 0,
                        "msg": "Service unavailable from a restricted location",
                    })
                return _response(200, {"symbols": [{
                    "symbol": "AAPLUSDT",
                    "contractType": contract_type,
                    "status": "TRADING",
                    "underlyingType": "EQUITY",
                }]})
            if url.endswith("/fapi/v1/fundingRate"):
                rows = [{
                    "symbol": "AAPLUSDT",
                    "fundingRate": funding_rate,
                    "fundingTime": _ms(2026, 7, 2, 16),
                    "markPrice": "200",
                }]
                if extra_funding:
                    rows.append(extra_funding)
                return _response(200, rows)
            if url.endswith("/futures/data/openInterestHist"):
                assert params["period"] == "1h"
                return _response(200, [
                    {
                        "symbol": "AAPLUSDT",
                        "sumOpenInterest": "10",
                        "sumOpenInterestValue": "2000",
                        "timestamp": _ms(2026, 7, 2, 15),
                    },
                    {
                        "symbol": "AAPLUSDT",
                        "sumOpenInterest": "99",
                        "sumOpenInterestValue": "99999",
                        "timestamp": _ms(2026, 7, 5, 12),
                    },
                ])
            if url.endswith("/fapi/v1/openInterest"):
                return _response(200, {"symbol": "AAPLUSDT", "openInterest": "10", "time": 1})
            raise AssertionError(url)

        return calls, fake_get

    def test_funding_and_open_interest_follow_a_confirmed_perpetual(self, monkeypatch):
        _map(symbol_map={"AAPL": {"spot_symbol": "AAPLBUSDT", "futures_symbol": "AAPLUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: True)
        calls, fake_get = self._spot_get("PERPETUAL", extra_funding={
            "symbol": "AAPLUSDT",
            "fundingRate": "0.77700000",
            "fundingTime": _ms(2026, 7, 5, 12),
            "markPrice": "1",
        })
        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
        paths = [url.split(FUTURES, 1)[-1] for url, _ in calls if url.startswith(FUTURES)]
        assert "/fapi/v1/exchangeInfo" in paths
        assert "/fapi/v1/fundingRate" in paths
        assert "/futures/data/openInterestHist" in paths
        assert "/fapi/v1/openInterest" not in paths
        assert "/fapi/v1/klines" not in paths
        assert "contractType PERPETUAL" in out
        assert "0.00010000" in out
        assert "0.77700000" not in out
        assert "| 10 |" in out or "sumOpenInterest 10" in out
        assert "99" not in out
        assert "withheld" in out

    def test_a_live_window_includes_present_open_interest(self, monkeypatch):
        _map(symbol_map={"AAPL": {"spot_symbol": "AAPLBUSDT", "futures_symbol": "AAPLUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: False)
        calls, fake_get = self._spot_get("PERPETUAL")
        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
        assert any(url.endswith("/fapi/v1/openInterest") for url, _ in calls)
        assert "openInterest=10" in out

    def test_a_non_perpetual_does_not_request_funding_or_open_interest(self, monkeypatch):
        _map(symbol_map={"AAPL": {"spot_symbol": "AAPLBUSDT", "futures_symbol": "AAPLUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        calls, fake_get = self._spot_get("CURRENT_QUARTER")
        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
        paths = [url for url, _ in calls]
        assert not any(url.endswith("/fapi/v1/fundingRate") for url in paths)
        assert not any("openInterest" in url for url in paths)
        assert "neither PERPETUAL nor TRADIFI_PERPETUAL" in out
        assert "+100.00%" in out

    def test_a_restricted_futures_host_does_not_confirm_a_perpetual(self, monkeypatch):
        _map(symbol_map={"AAPL": {"spot_symbol": "AAPLBUSDT", "futures_symbol": "AAPLUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        calls, fake_get = self._spot_get("HTTP451")
        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
        assert not any(url.endswith("/fapi/v1/fundingRate") for url, _ in calls)
        assert "NOT confirmed" in out
        assert "+100.00%" in out

    def test_old_empty_open_interest_is_not_reported_as_zero(self, monkeypatch):
        _map(symbol_map={"AAPL": {"futures_symbol": "AAPLUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: True)
        monkeypatch.setattr(tradfi, "_today", lambda: date(2026, 11, 3))

        def fake_get(url, params=None, timeout=None):
            if url.endswith("/fapi/v1/klines"):
                return _response(200, [_kline(_ms(2026, 7, 2, 15), "200")])
            if url.endswith("/fapi/v1/exchangeInfo"):
                return _response(200, {"symbols": [{
                    "symbol": "AAPLUSDT", "contractType": "PERPETUAL", "status": "TRADING",
                }]})
            if url.endswith("/fapi/v1/fundingRate"):
                return _response(200, [])
            if url.endswith("/futures/data/openInterestHist"):
                return _response(200, [])
            raise AssertionError(url)

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
        assert "not a reading of zero open interest" in out
        assert "USD-M contract kline" in out


@pytest.mark.unit
class TestRouting:
    def test_invalid_symbol_is_no_data(self):
        _map()

        def fake_get(url, params=None, timeout=None):
            return _response(400, {"code": -1121, "msg": "Invalid symbol."})

        with mock.patch.object(public.requests, "get", fake_get):
            with pytest.raises(NoMarketDataError):
                tradfi.get_binance_tradfi_premium("AAPL", "2026-07-02", "2026-07-04")
            out = router.route_to_vendor(
                "get_binance_tradfi_premium", "AAPL", "2026-07-02", "2026-07-04"
            )
        assert out.startswith("NO_DATA_AVAILABLE")

    def test_category_is_optional_and_named(self):
        assert router.get_category_for_method("get_binance_tradfi_premium") == "binance_tradfi"
        assert "binance_tradfi" in router.OPTIONAL_CATEGORIES
        assert "binance" in router.VENDOR_METHODS["get_binance_tradfi_premium"]

    def test_the_tool_clamps_the_window_to_the_run_date(self):
        with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
            tools.get_binance_tradfi_premium.func(
                "AAPL", "2026-08-01", "2026-09-14", trade_date="2026-08-14"
            )
        assert routed.call_args.args == (
            "get_binance_tradfi_premium", "AAPL", "2026-08-01", "2026-08-14"
        )

    def test_a_normal_ticker_still_uses_the_equity_vendor(self):
        _map(symbol_map={})
        assert router.get_vendor("core_stock_apis", "get_stock_data") == "yfinance"


def _tradfi_row(symbol, base, **extra):
    """Field shape observed on 2026-11-03 for USDT-margined equity perps."""
    row = {
        "symbol": symbol,
        "contractType": "TRADIFI_PERPETUAL",
        "status": "TRADING",
        "underlyingType": "EQUITY",
        "underlyingSubType": ["TradFi"],
        "baseAsset": base,
        "quoteAsset": "USDT",
        "onboardDate": _ms(2026, 1, 28, 0),
    }
    row.update(extra)
    return row


_CATALOG = [
    _tradfi_row("TSLAUSDT", "TSLA"),
    _tradfi_row("NVDAUSDT", "NVDA"),
    _tradfi_row("AAPLUSDT", "AAPL"),
    _tradfi_row("MUUSDT", "MU"),
    {
        "symbol": "BTCUSDT",
        "contractType": "PERPETUAL",
        "status": "TRADING",
        "underlyingType": "COIN",
        "underlyingSubType": ["PoW"],
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
    },
]


@pytest.mark.unit
class TestTradFiDiscovery:
    def test_observed_rows_match_their_base_asset(self):
        end = _ms(2026, 7, 5, 0)
        for symbol, base in (("TSLAUSDT", "TSLA"), ("NVDAUSDT", "NVDA"), ("AAPLUSDT", "AAPL"), ("MUUSDT", "MU")):
            found, sentence = tradfi.discover_tradfi_perpetual(
                _CATALOG, base, end, historical=True
            )
            assert found["symbol"] == symbol
            assert found["contractType"] == "TRADIFI_PERPETUAL"
            assert "TRADIFI_PERPETUAL" in sentence
        # A crypto perpetual is not a TradFi equity, even when the base matches.
        none, sentence = tradfi.discover_tradfi_perpetual(_CATALOG, "BTC", end, historical=True)
        assert none is None
        assert "No USD-M symbol" in sentence

    def test_a_historical_row_without_onboard_date_is_not_assumed(self):
        catalog = [_tradfi_row("MUUSDT", "MU", onboardDate=None)]
        found, sentence = tradfi.discover_tradfi_perpetual(
            catalog, "MU", _ms(2026, 7, 5, 0), historical=True
        )
        assert found is None
        assert "onboardDate" in sentence
        assert "does not assume" in sentence

    def test_a_contract_listed_after_the_window_is_not_used(self):
        catalog = [_tradfi_row("MUUSDT", "MU", onboardDate=_ms(2026, 12, 1, 0))]
        found, sentence = tradfi.discover_tradfi_perpetual(
            catalog, "MU", _ms(2026, 7, 5, 0), historical=True
        )
        assert found is None
        assert "after this window" in sentence

    def test_two_usdt_matches_are_not_guessed(self):
        catalog = [
            _tradfi_row("MUUSDT", "MU"),
            _tradfi_row("MU2USDT", "MU"),
        ]
        found, sentence = tradfi.discover_tradfi_perpetual(
            catalog, "MU", _ms(2026, 7, 5, 0), historical=False
        )
        assert found is None
        assert "MU2USDT" in sentence and "MUUSDT" in sentence

    def test_discovered_tradifi_perpetual_requests_funding_and_open_interest(self, monkeypatch):
        set_config({
            "binance_tradfi": {
                "symbol_map": {},
                "discover_futures": True,
                "kline_interval": "1h",
                "spot_base_url": SPOT,
                "futures_base_url": FUTURES,
            }
        })
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: True)
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            if url.endswith("/fapi/v1/exchangeInfo"):
                return _response(200, {"symbols": _CATALOG})
            if url.endswith("/fapi/v1/klines"):
                assert params["symbol"] == "MUUSDT"
                return _response(200, [_kline(_ms(2026, 7, 2, 15), "870")])
            if url.endswith("/fapi/v1/fundingRate"):
                assert params["symbol"] == "MUUSDT"
                return _response(200, [{
                    "symbol": "MUUSDT",
                    "fundingRate": "0.00037717",
                    "fundingTime": _ms(2026, 7, 2, 16),
                    "markPrice": "874.63",
                }])
            if url.endswith("/futures/data/openInterestHist"):
                return _response(200, [{
                    "symbol": "MUUSDT",
                    "sumOpenInterest": "165656.12",
                    "sumOpenInterestValue": "1",
                    "timestamp": _ms(2026, 7, 2, 15),
                }])
            if url.endswith("/fapi/v1/openInterest"):
                raise AssertionError("present open interest is withheld for a historical window")
            raise AssertionError(url)

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("MU", "2026-07-02", "2026-07-04")
        assert "TRADIFI_PERPETUAL" in out
        assert "0.00037717" in out
        assert "874.63" in out
        assert "165656.12" in out
        assert "selected from the current exchangeInfo catalog" in out
        assert any(url.endswith("/fapi/v1/fundingRate") for url, _ in calls)

    def test_an_explicit_futures_symbol_overrides_discovery(self, monkeypatch):
        _map(symbol_map={"MU": {"futures_symbol": "NVDAUSDT"}}, discover_futures=True)
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: False)
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params or {}))
            if url.endswith("/fapi/v1/exchangeInfo"):
                return _response(200, {"symbols": _CATALOG})
            if url.endswith("/fapi/v1/klines"):
                return _response(200, [_kline(_ms(2026, 7, 2, 15), "200")])
            if url.endswith("/fapi/v1/fundingRate"):
                return _response(200, [{
                    "symbol": params["symbol"],
                    "fundingRate": "0.00010480",
                    "fundingTime": _ms(2026, 7, 2, 16),
                    "markPrice": "204.05",
                }])
            if url.endswith("/futures/data/openInterestHist"):
                return _response(200, [])
            if url.endswith("/fapi/v1/openInterest"):
                assert params["symbol"] == "NVDAUSDT"
                return _response(200, {
                    "symbol": "NVDAUSDT", "openInterest": "347156.24", "time": _ms(2026, 7, 2, 16),
                })
            raise AssertionError(url)

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("MU", "2026-07-02", "2026-07-04")
        funding = [params for url, params in calls if url.endswith("/fapi/v1/fundingRate")]
        assert [params["symbol"] for params in funding] == ["NVDAUSDT"]
        assert all(params.get("symbol") != "MUUSDT" for _url, params in calls)
        assert "0.00010480" in out
        assert "347156.24" in out
        assert "204.05" in out
        assert "taken from config" in out

    def test_documented_perpetual_is_still_accepted(self, monkeypatch):
        _map(symbol_map={"BTC": {"futures_symbol": "BTCUSDT"}})
        monkeypatch.setattr(tradfi, "load_ohlcv", lambda *a, **k: _stock_frame())
        monkeypatch.setattr(tradfi, "is_historical", lambda _day: True)

        def fake_get(url, params=None, timeout=None):
            if url.endswith("/fapi/v1/exchangeInfo"):
                return _response(200, {"symbols": _CATALOG})
            if url.endswith("/fapi/v1/klines"):
                return _response(200, [_kline(_ms(2026, 7, 2, 15), "100")])
            if url.endswith("/fapi/v1/fundingRate"):
                return _response(200, [{
                    "symbol": "BTCUSDT",
                    "fundingRate": "0.00010000",
                    "fundingTime": _ms(2026, 7, 2, 16),
                    "markPrice": "1",
                }])
            if url.endswith("/futures/data/openInterestHist"):
                return _response(200, [])
            raise AssertionError(url)

        with mock.patch.object(public.requests, "get", fake_get):
            out = tradfi.get_binance_tradfi_premium("BTC", "2026-07-02", "2026-07-04")
        assert "contractType PERPETUAL" in out
        assert "0.00010000" in out
