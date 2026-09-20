"""Rebuild the TSLA/SPY example offline, or explicitly refresh Yahoo snapshots.

Run ``python prepare_real_example.py`` to read the committed raw JSON files.
Only ``--download`` accesses the network. Prices are Yahoo's retrospectively
adjusted daily closes; this is not an original-vendor point-in-time dataset.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SYMBOLS = ("TSLA", "SPY")
SAMPLE_START = "2023-01-01"
SAMPLE_END_EXCLUSIVE = "2025-01-01"
PERIOD1 = 1672531200
PERIOD2 = 1735689600
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "real"


def source_url(symbol):
    return (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d")


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_prices(payload, symbol):
    """Validate Yahoo bytes and return exchange-local daily OHLCV/adjusted close."""
    try:
        chart = json.loads(payload)["chart"]
        if chart.get("error") is not None or len(chart.get("result") or []) != 1:
            raise ValueError(f"Yahoo returned an error or unexpected results for {symbol}")
        result = chart["result"][0]
        metadata = result["meta"]
        if metadata.get("symbol") != symbol:
            raise ValueError(f"Expected {symbol}, received {metadata.get('symbol')}")
        exchange_timezone = metadata["exchangeTimezoneName"]
        if exchange_timezone != "America/New_York":
            raise ValueError(f"Unexpected exchange timezone: {exchange_timezone}")
        timestamps = result["timestamp"]
        if not timestamps:
            raise ValueError(f"No price observations returned for {symbol}")
        sessions = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(exchange_timezone)
        dates = sessions.normalize().tz_localize(None)
        if dates.hasnans or not dates.is_unique:
            raise ValueError(f"Invalid or duplicate exchange-local trading dates for {symbol}")
        if not dates.is_monotonic_increasing:
            raise ValueError(f"Trading dates are not in increasing order for {symbol}")
        if ((dates < pd.Timestamp(SAMPLE_START)) | (dates >= pd.Timestamp(SAMPLE_END_EXCLUSIVE))).any():
            raise ValueError(f"Yahoo returned dates outside the requested sample for {symbol}")
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        fields = {"Open": quote["open"], "High": quote["high"], "Low": quote["low"],
                  "Close": quote["close"], "Volume": quote["volume"], "Adj_Close": adjusted}
        if any(len(values) != len(dates) for values in fields.values()):
            raise ValueError(f"Price-array lengths do not match timestamps for {symbol}")
        frame = pd.DataFrame(fields, index=pd.DatetimeIndex(dates, name="Date"))
        for column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        adjusted_values = frame["Adj_Close"].to_numpy(dtype=float)
        if not np.isfinite(adjusted_values).all() or (adjusted_values <= 0).any():
            raise ValueError(f"Missing, invalid or nonpositive adjusted close for {symbol}; no raw-close fallback")
        return frame, metadata
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Missing or malformed required Yahoo fields for {symbol}: {error}") from error


def _download_snapshots(raw_dir):
    """Validate both responses before replacing local raw snapshots."""
    pending, acquisition = {}, {}
    for symbol in SYMBOLS:
        url = source_url(symbol)
        request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        parse_prices(payload, symbol)
        pending[symbol] = payload
        acquisition[symbol] = {
            "url": url,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "raw_json": f"{symbol}_2023_2024.yahoo.json",
        }
    # Calendar mismatches are explicit failures, never inner-joined away.
    frames = {symbol: parse_prices(payload, symbol)[0] for symbol, payload in pending.items()}
    _validate_matching_calendar(frames)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol, payload in pending.items():
        (raw_dir / acquisition[symbol]["raw_json"]).write_bytes(payload)
    _json_write(raw_dir / "download_metadata.json", acquisition)


def _validate_matching_calendar(frames):
    asset_dates = frames["TSLA"].index
    benchmark_dates = frames["SPY"].index
    if not asset_dates.equals(benchmark_dates):
        only_asset = asset_dates.difference(benchmark_dates).strftime("%Y-%m-%d").tolist()
        only_benchmark = benchmark_dates.difference(asset_dates).strftime("%Y-%m-%d").tolist()
        raise ValueError("TSLA/SPY trading calendars differ; no sessions were dropped. "
                         f"Only TSLA: {only_asset}; only SPY: {only_benchmark}")


def _build_rating_events(data_dir):
    """Preserve reported announcement dates when rebuilding the curated events."""
    records_path = data_dir / "raw" / "ratings_source_records.csv"
    if not records_path.exists():
        raise FileNotFoundError("Missing data/real/raw/ratings_source_records.csv; restore the cited, manually curated rating records.")
    records = pd.read_csv(records_path)
    required = {"event_id", "announcement_date", "firm", "event_type", "source_url"}
    if not required.issubset(records.columns):
        raise ValueError(f"Rating source records need columns: {sorted(required)}")
    if records[list(required)].isna().any().any() or records["event_id"].duplicated().any():
        raise ValueError("Rating records require nonmissing fields and unique event IDs")
    if not records["event_type"].isin(["upgrade", "downgrade"]).all():
        raise ValueError("Rating event types must be upgrade or downgrade")
    events = records.rename(columns={"announcement_date": "event_date"})[
        ["event_id", "event_date", "firm", "event_type"]
    ].copy()
    dates = pd.to_datetime(events["event_date"], format="%Y-%m-%d", errors="raise")
    if ((dates < pd.Timestamp(SAMPLE_START)) | (dates >= pd.Timestamp(SAMPLE_END_EXCLUSIVE))).any():
        raise ValueError("Rating announcement date falls outside the price sample")
    if events.duplicated(["firm", "event_date"]).any():
        raise ValueError("Duplicate rating firm/announcement-date pair")
    events.to_csv(data_dir / "events.csv", index=False)
    return {
        "nature": "Manually transcribed public reporting, not complete vendor rating history",
        "source_records": "raw/ratings_source_records.csv",
        "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "events_csv": "events.csv",
        "events_csv_sha256": hashlib.sha256((data_dir / "events.csv").read_bytes()).hexdigest(),
        "event_count": len(events),
        "counts_by_event_type": events["event_type"].value_counts().sort_index().to_dict(),
        "source_urls": records["source_url"].tolist(),
        "date_policy": "Preserve reported announcement dates, including weekends. The event-study backend maps each to the next supplied trading session; unverified publication timestamps do not establish analyst-note release times.",
        "selection_warning": "Small hand-selected illustration with incomplete coverage, overlapping windows and possible contemporaneous news confounding; not a representative causal sample.",
    }


def build_real_example(download=False, data_dir=DEFAULT_DATA_DIR):
    """Create raw price CSVs, aligned decimal returns and a provenance manifest."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    if download:
        _download_snapshots(raw_dir)
    acquisition_path = raw_dir / "download_metadata.json"
    if not acquisition_path.exists():
        raise FileNotFoundError("Raw snapshot metadata is missing. Run with --download to acquire the example.")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    frames, sources = {}, []
    for symbol in SYMBOLS:
        details = acquisition[symbol]
        raw_path = raw_dir / details["raw_json"]
        payload = raw_path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != details["sha256"]:
            raise ValueError(f"Raw JSON SHA-256 mismatch for {symbol}; snapshot or metadata was modified")
        frame, metadata = parse_prices(payload, symbol)
        frames[symbol] = frame
        sources.append({
            "symbol": symbol,
            "role": "asset" if symbol == "TSLA" else "market_benchmark_proxy",
            "provider": "Yahoo Finance chart API",
            "url": details["url"],
            "history_url": f"https://finance.yahoo.com/quote/{symbol}/history/",
            "downloaded_utc": details["downloaded_utc"],
            "raw_json": f"raw/{details['raw_json']}",
            "raw_json_sha256": actual_hash,
            "raw_json_bytes": len(payload),
            "parsed_prices_csv": f"raw/{symbol}_prices.csv",
            "exchange_timezone": metadata["exchangeTimezoneName"],
            "currency": metadata.get("currency"),
            "first_trading_date": frame.index.min().strftime("%Y-%m-%d"),
            "last_trading_date": frame.index.max().strftime("%Y-%m-%d"),
            "price_observations": len(frame),
            "missing_adjusted_closes": int(frame["Adj_Close"].isna().sum()),
        })
    _validate_matching_calendar(frames)
    adjusted = pd.DataFrame({symbol: frame["Adj_Close"] for symbol, frame in frames.items()})
    returns = adjusted.pct_change(fill_method=None).rename(columns={"TSLA": "ret", "SPY": "mkt"})
    returns.index.name = "Date"
    if np.isinf(returns.to_numpy(dtype=float)).any():
        raise ValueError("Nonfinite return created from adjusted prices")
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol, frame in frames.items():
        frame.to_csv(raw_dir / f"{symbol}_prices.csv", date_format="%Y-%m-%d", float_format="%.15g")
    returns_path = data_dir / "returns.csv"
    returns.to_csv(returns_path, date_format="%Y-%m-%d", float_format="%.15g", na_rep="")
    ratings = _build_rating_events(data_dir)
    finite_rows = np.isfinite(returns.to_numpy(dtype=float)).all(axis=1)
    manifest = {
        "dataset": "TSLA versus SPY daily adjusted-close return example",
        "requested_sample": {"start_inclusive": SAMPLE_START, "end_exclusive": SAMPLE_END_EXCLUSIVE},
        "sources": sources,
        "ratings": ratings,
        "alignment": {
            "calendar": "Exchange-local daily dates, converted from UTC Yahoo timestamps using America/New_York",
            "matching_sessions": len(adjusted),
            "calendars_identical": True,
            "sessions_dropped": 0,
            "first_trading_date": returns.index.min().strftime("%Y-%m-%d"),
            "last_trading_date": returns.index.max().strftime("%Y-%m-%d"),
        },
        "transformation": {
            "formula": "ret_t = TSLA_Adj_Close_t / TSLA_Adj_Close_(t-1) - 1; mkt_t = SPY_Adj_Close_t / SPY_Adj_Close_(t-1) - 1",
            "units": "decimal daily returns; 0.01 means 1%",
            "pandas_operation": "adjusted_close.pct_change(fill_method=None)",
            "missing_data_policy": "No fill, calendar compression, or unadjusted-close fallback. Duplicate or mismatched calendars and missing adjusted closes fail validation. First-row returns remain NaN because no preceding sample price exists.",
            "benchmark": "SPY adjusted-close return is a tradable broad US equity benchmark proxy, not a fitted market model or a market excess-return factor.",
        },
        "output": {
            "returns_csv": "returns.csv",
            "returns_csv_sha256": hashlib.sha256(returns_path.read_bytes()).hexdigest(),
            "rows": len(returns),
            "complete_return_rows": int(finite_rows.sum()),
            "rows_with_missing_return": int((~finite_rows).sum()),
            "first_complete_return_date": returns.index[finite_rows][0].strftime("%Y-%m-%d") if finite_rows.any() else None,
        },
        "limitations": [
            "Yahoo adjusted prices can be revised retrospectively for corporate actions and corrections; this snapshot is not an original vendor point-in-time database.",
            "The raw endpoint may include metadata outside the requested historical period. Returns use only the validated historical timestamp/price arrays.",
            "Identical TSLA/SPY calendars do not independently prove exchange-calendar completeness; both sources could omit the same session.",
            "Rating events are rebuilt from separately cited manual source records. Only prices are downloaded by this script.",
        ],
        "reproduce": "python prepare_real_example.py (offline); python prepare_real_example.py --download (explicit network refresh)",
    }
    _json_write(data_dir / "source_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly refresh the two Yahoo raw JSON snapshots")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Example data directory")
    args = parser.parse_args()
    manifest = build_real_example(download=args.download, data_dir=args.data_dir)
    alignment = manifest["alignment"]
    print(f"Prepared {alignment['matching_sessions']} matched TSLA/SPY sessions, "
          f"{alignment['first_trading_date']} through {alignment['last_trading_date']}.")
    print(f"Complete return rows: {manifest['output']['complete_return_rows']}; "
          f"missing initial return retained: {manifest['output']['rows_with_missing_return']}.")
    print(f"Saved returns and provenance to {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()
