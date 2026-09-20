"""Deterministic synthetic inputs; injected effects are not empirical findings."""

import numpy as np
import pandas as pd


def make_demo_data(seed=42, n_events=32):
    """Return synthetic daily returns and separated, balanced event labels.

    A fixed-seed normal market and independent residual form one fictional
    asset. Upgrade/downgrade labels alternate every 18 weekday sessions. On
    event days 0, +1 and +2 we deliberately inject positive/negative effects,
    respectively. This demonstrates the visual pipeline, not predictability.
    The weekday calendar is synthetic and does not model exchange holidays.
    """
    if (isinstance(n_events, bool) or not isinstance(n_events, (int, np.integer))
            or n_events < 2 or n_events % 2):
        raise ValueError("n_events must be a positive even integer of at least 2")
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2022-01-03", periods=n_events * 18 + 20, name="date")
    market = rng.normal(0.0002, 0.007, len(calendar))
    asset = market + rng.normal(0.0, 0.003, len(calendar))
    events = []
    for number, position in enumerate(range(10, 10 + n_events * 18, 18)):
        event_type = "upgrade" if number % 2 == 0 else "downgrade"
        sign = 1 if event_type == "upgrade" else -1
        amplitude = rng.uniform(0.8, 1.2)
        asset[position:position + 3] += sign * amplitude * np.array([0.012, 0.006, 0.003])
        events.append(dict(event_id=number + 1, event_date=calendar[position].strftime("%Y-%m-%d"),
                           firm=f"Synthetic Research {number % 4 + 1}", event_type=event_type))
    returns = pd.DataFrame({"ret": asset, "mkt": market}, index=calendar)
    return returns, pd.DataFrame(events)
