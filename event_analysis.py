"""Auditable market-adjusted event study using supplied trading sessions.

Returns are decimal daily total returns. A window of 5 means 11 observed
trading sessions, [-5, +5], with day zero the first session on/after the event.
No returns or missing sessions are filled. The supplied index must therefore
represent the complete intended trading calendar. Inference treats events as
independent observations; overlapping windows require additional care.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


EVENT_COLUMNS = ["event_id", "firm", "event_type", "event_date"]
METADATA_COLUMNS = EVENT_COLUMNS + ["effective_event_date"]
PANEL_COLUMNS = METADATA_COLUMNS + [
    "event_time", "ret_date", "ret", "mkt", "abnormal_return", "car"
]
EVENT_SUMMARY_COLUMNS = METADATA_COLUMNS + ["event_shift_days", "car"]
GROUP_COLUMNS = [
    "event_type", "event_time", "n", "aar", "caar", "ci_lower", "ci_upper",
    "tstat", "pvalue", "aar_ci_lower", "aar_ci_upper"
]
CAR_SUMMARY_COLUMNS = [
    "event_type", "n", "mean_car", "std_car", "ci_lower", "ci_upper", "tstat", "pvalue"
]


@dataclass(frozen=True)
class AnalysisResult:
    """Event paths, endpoint CARs, pointwise group estimates and audit trail."""

    panel: pd.DataFrame
    event_summary: pd.DataFrame
    group_summary: pd.DataFrame
    exclusions: pd.DataFrame


def _mean_statistics(values):
    """Two-sided one-sample t inference; degenerate uncertainty is undefined."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    result = dict(n=n, mean=mean, std=std, ci_lower=np.nan, ci_upper=np.nan,
                  tstat=np.nan, pvalue=np.nan)
    if n < 2 or np.ptp(values) == 0 or not np.isfinite(std) or std == 0:
        return result
    standard_error = std / np.sqrt(n)
    margin = float(student_t.ppf(0.975, n - 1)) * standard_error
    statistic = mean / standard_error
    result.update(ci_lower=mean - margin, ci_upper=mean + margin,
                  tstat=statistic, pvalue=float(2 * student_t.sf(abs(statistic), n - 1)))
    return result


def _validate_returns(returns):
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame")
    if not returns.columns.is_unique or not {"ret", "mkt"}.issubset(returns.columns):
        raise ValueError("returns must have unique columns including ret and mkt")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must use a DatetimeIndex")
    if returns.index.hasnans or returns.index.tz is not None:
        raise ValueError("return dates must be valid timezone-naive daily dates")
    if not returns.index.equals(returns.index.normalize()):
        raise ValueError("return dates must be daily dates at midnight")
    if not returns.index.is_unique:
        raise ValueError("duplicate return dates are not allowed")
    # Keep invalid rows on the calendar so they cannot silently shorten a window.
    clean = returns.loc[:, ["ret", "mkt"]].copy().sort_index()
    for column in clean:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean


def _parse_event_date(value):
    # Numeric values would otherwise be silently interpreted as nanoseconds.
    if value is None or isinstance(value, (int, float, np.number, bool)):
        return pd.NaT
    try:
        date = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    if pd.isna(date) or date.tzinfo is not None:
        return pd.NaT
    return date.normalize()


def analyze_events(returns, events, window=5, max_event_shift_days=7):
    """Compute AR=ret-mkt and additive CAR over complete trading-day windows.

    ``events`` requires event_id, event_date, firm and event_type columns.
    Event IDs and valid (firm, event_date) pairs must be unique. Accepted types
    are ``upgrade`` and ``downgrade``. Non-trading event dates move to the next
    supplied session, at most ``max_event_shift_days`` calendar days later.
    Malformed events, unavailable/full windows and invalid returns are logged
    in ``exclusions``. Input DataFrames are never modified.

    Group confidence intervals are pointwise 95% Student t intervals across
    individual event CAR paths, not sums of daily AR confidence intervals.
    For fewer than two observations or zero variance, intervals and tests are
    NaN. These intervals are neither simultaneous nor dependence-adjusted.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (isinstance(max_event_shift_days, bool)
            or not isinstance(max_event_shift_days, (int, np.integer))
            or max_event_shift_days < 0):
        raise ValueError("max_event_shift_days must be a nonnegative integer")
    clean_returns = _validate_returns(returns)
    if not isinstance(events, pd.DataFrame):
        raise ValueError("events must be a pandas DataFrame")
    if not events.columns.is_unique or not set(EVENT_COLUMNS).issubset(events.columns):
        raise ValueError("events require unique event_id, event_date, firm, event_type columns")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("event_id values must be nonmissing and unique")
    clean_events = events.loc[:, EVENT_COLUMNS].copy()
    clean_events["parsed_date"] = clean_events["event_date"].map(_parse_event_date)
    valid_pairs = clean_events.dropna(subset=["parsed_date", "firm"])
    if valid_pairs.duplicated(["firm", "parsed_date"]).any():
        raise ValueError("duplicate (firm, event_date) events are not allowed")

    calendar = clean_returns.index
    paths, endpoints, excluded = [], [], []
    for row in clean_events.to_dict("records"):
        metadata = {column: row[column] for column in EVENT_COLUMNS}
        metadata["effective_event_date"] = pd.NaT
        event_date = row["parsed_date"]
        reason = None
        if pd.isna(event_date):
            reason = "invalid_event_date"
        elif not isinstance(row["firm"], str) or not row["firm"].strip():
            reason = "invalid_firm"
        elif not isinstance(row["event_type"], str) or row["event_type"] not in {"upgrade", "downgrade"}:
            reason = "invalid_event_type"
        else:
            position = int(calendar.searchsorted(event_date, side="left"))
            if position == len(calendar):
                reason = "no_next_trading_session"
            else:
                effective_date = calendar[position]
                metadata["effective_event_date"] = effective_date
                shift = (effective_date - event_date).days
                if shift > max_event_shift_days:
                    reason = "event_shift_exceeds_limit"
                elif position - window < 0 or position + window >= len(calendar):
                    reason = "incomplete_event_window"
                else:
                    observations = clean_returns.iloc[position - window:position + window + 1]
                    if not np.isfinite(observations.to_numpy(dtype=float)).all():
                        reason = "invalid_window_returns"
        if reason is not None:
            excluded.append({**metadata, "reason": reason})
            continue

        # Preserve the original date in audit outputs, including weekend dates.
        abnormal = observations["ret"].to_numpy(dtype=float) - observations["mkt"].to_numpy(dtype=float)
        cumulative = abnormal.cumsum()
        if not np.isfinite(abnormal).all() or not np.isfinite(cumulative).all():
            excluded.append({**metadata, "reason": "invalid_window_returns"})
            continue
        path = observations.reset_index(names="ret_date")
        for column, value in metadata.items():
            path[column] = value
        path["event_time"] = np.arange(-window, window + 1)
        path["abnormal_return"] = abnormal
        path["car"] = cumulative
        paths.append(path.loc[:, PANEL_COLUMNS])
        endpoints.append({**metadata, "event_shift_days": shift, "car": cumulative[-1]})

    panel = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(columns=PANEL_COLUMNS)
    event_summary = pd.DataFrame(endpoints, columns=EVENT_SUMMARY_COLUMNS)
    exclusions = pd.DataFrame(excluded, columns=METADATA_COLUMNS + ["reason"])
    groups = []
    for (event_type, event_time), observations in panel.groupby(["event_type", "event_time"], sort=True):
        car_stats = _mean_statistics(observations["car"])
        ar_stats = _mean_statistics(observations["abnormal_return"])
        groups.append(dict(event_type=event_type, event_time=event_time,
                           n=car_stats["n"], aar=ar_stats["mean"], caar=car_stats["mean"],
                           ci_lower=car_stats["ci_lower"], ci_upper=car_stats["ci_upper"],
                           tstat=car_stats["tstat"], pvalue=car_stats["pvalue"],
                           aar_ci_lower=ar_stats["ci_lower"], aar_ci_upper=ar_stats["ci_upper"]))
    return AnalysisResult(panel, event_summary, pd.DataFrame(groups, columns=GROUP_COLUMNS), exclusions)


def summarize_car(event_summary):
    """Summarize full-window event CARs by event type using a two-sided t test."""
    rows = []
    for event_type, events in event_summary.groupby("event_type", sort=True):
        stats = _mean_statistics(events["car"])
        rows.append(dict(event_type=event_type, n=stats["n"], mean_car=stats["mean"],
                         std_car=stats["std"], ci_lower=stats["ci_lower"],
                         ci_upper=stats["ci_upper"], tstat=stats["tstat"], pvalue=stats["pvalue"]))
    return pd.DataFrame(rows, columns=CAR_SUMMARY_COLUMNS)
