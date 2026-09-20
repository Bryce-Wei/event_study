"""Reproducible event-study figures from local returns and rating events.

Run ``python visualize.py --demo`` for explicitly synthetic data, or provide
``--returns CSV --events CSV --label TSLA``. No network access is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

from demo_data import make_demo_data
from event_analysis import analyze_events, summarize_car

COLORS = {"upgrade": "#087F8C", "downgrade": "#D5654A"}
LABELS = {"upgrade": "Upgrade", "downgrade": "Downgrade"}
INK, MUTED, GRID = "#192C43", "#627287", "#E4EAF0"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 14, "axes.titlepad": 18,
        "figure.facecolor": "#F6F8FB", "axes.facecolor": "white",
        "savefig.facecolor": "#F6F8FB", "svg.fonttype": "none",
    })


def decorate(ax, title, ylabel):
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))


def event_axis(ax, window):
    ax.axvline(0, color=INK, linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlim(-window - 0.5, window + 0.5)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Trading sessions relative to event (t = 0)")


def plot_aar(ax, result, window):
    decorate(ax, "01  |  Average abnormal return", "AAR / day")
    for etype, offset in [("upgrade", -0.18), ("downgrade", 0.18)]:
        g = result.group_summary.query("event_type == @etype")
        if g.empty:
            continue
        ax.bar(g.event_time + offset, g.aar, width=0.34,
               color=COLORS[etype], label=f"{LABELS[etype]} (n={int(g.n.iloc[0])})")
    event_axis(ax, window)
    ax.legend(frameon=False, loc="best", fontsize=10)


def plot_caar(ax, result, window, show_ci=False):
    decorate(ax, "02  |  Cumulative average abnormal return", "CAAR from window start")
    for etype in COLORS:
        g = result.group_summary.query("event_type == @etype")
        if g.empty:
            continue
        x, y = g.event_time.to_numpy(), g.caar.to_numpy()
        if not show_ci:
            paths = result.panel.loc[result.panel.event_type == etype]
            for _, path in paths.groupby("event_id"):
                ax.plot(path.event_time, path.car, color=COLORS[etype],
                        alpha=0.17, linewidth=1, zorder=1)
        ax.plot(x, y, marker="o", markersize=4, linewidth=2.3,
                color=COLORS[etype], label=LABELS[etype])
        if show_ci:
            ax.fill_between(x, g.ci_lower.to_numpy(), g.ci_upper.to_numpy(),
                            color=COLORS[etype], alpha=0.13, linewidth=0)
    event_axis(ax, window)
    ax.legend(frameon=False, loc="best", fontsize=10)
    note = "Shading: pointwise 95% t intervals*" if show_ci else "Thin lines: individual event CAR paths"
    ax.text(0.02, 0.97, note, transform=ax.transAxes,
            va="top", fontsize=9, color=MUTED)
    ax.margins(y=0.22)


def plot_caar_intervals(ax, result, window):
    plot_caar(ax, result, window, show_ci=True)
    ax.set_title("05  |  Uncertainty: pointwise 95% t intervals", loc="left")


def plot_distribution(ax, result, window):
    decorate(ax, f"03  |  Event CAR distribution [-{window}, +{window}]", "Event CAR")
    groups = [g for g in COLORS if (result.event_summary.event_type == g).any()]
    rng = np.random.default_rng(2026)
    for pos, etype in enumerate(groups, 1):
        values = result.event_summary.loc[result.event_summary.event_type == etype, "car"].to_numpy()
        boxes = ax.boxplot([values], positions=[pos], widths=0.42, patch_artist=True,
                           showfliers=False, medianprops={"color": INK, "linewidth": 1.8},
                           whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        boxes["boxes"][0].set(facecolor=COLORS[etype], alpha=0.14, edgecolor=COLORS[etype])
        ax.scatter(pos + rng.uniform(-0.09, 0.09, len(values)), values,
                   color=COLORS[etype], edgecolors="white", linewidths=0.7,
                   s=50, alpha=0.9, zorder=4)
        ax.scatter([pos], [np.mean(values)], marker="D", s=65, color=INK, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1),
                  [f"{LABELS[g]}\nn = {(result.event_summary.event_type == g).sum()}" for g in groups])
    ax.set_xlim(0.45, len(groups) + 0.55)
    ax.text(0.02, 0.97, "Dot: one event   |   Diamond: group mean", transform=ax.transAxes,
            va="top", fontsize=9, color=MUTED)
    ax.margins(y=0.25)


def plot_events(ax, result, window):
    decorate(ax, "04  |  Each rating event", "Event CAR")
    events = result.event_summary.sort_values(["effective_event_date", "event_id"])
    for i, row in enumerate(events.itertuples()):
        c = COLORS[row.event_type]
        ax.vlines(i, 0, row.car, color=c, linewidth=2, alpha=0.6)
        ax.scatter(i, row.car, color=c, s=52, edgecolor="white", zorder=4)
    if len(events) <= 12:
        labels = [f"{r.effective_event_date:%Y-%m-%d}\n{r.firm}" for r in events.itertuples()]
        ax.set_xticks(range(len(events)), labels, rotation=35, ha="right", fontsize=8)
    else:
        ax.set_xlabel("Events in chronological order")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.margins(x=0.07, y=0.2)


PLOTS = [("01_aar", plot_aar), ("02_caar", plot_caar),
         ("03_car_distribution", plot_distribution), ("04_event_car", plot_events),
         ("05_caar_confidence_intervals", plot_caar_intervals)]


def header(fig, title, subtitle, badge):
    fig.text(0.065, 0.96, "EVENT STUDY  /  RESEARCH NOTE", fontsize=10,
             color=MUTED, weight="bold", va="top")
    fig.text(0.065, 0.915, title, fontsize=24, weight="bold", va="top")
    fig.text(0.065, 0.855, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.94, 0.96, badge, fontsize=9, color=INK, ha="right", va="top",
             bbox={"facecolor": "#E8EEF4", "edgecolor": "none", "pad": 7})


def render(result, output_dir, label, window, synthetic):
    style()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    n = len(result.event_summary)
    badge = "SYNTHETIC DEMO" if synthetic else "OBSERVED DATA / CURATED SAMPLE"
    subtitle = f"{n} complete events  |  Window: [-{window}, +{window}] sessions  |  AR = asset return - benchmark return"
    foot = ("Synthetic returns with injected effects; no empirical investment conclusion."
            if synthetic else "Selected events only; small, potentially dependent sample. No causal or predictive claim.")
    for filename, draw in PLOTS:
        fig, ax = plt.subplots(figsize=(12, 7.4))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.74, bottom=0.23)
        header(fig, f"{label} | Rating-event response", subtitle, badge)
        draw(ax, result, window)
        fig.text(0.065, 0.035, foot, fontsize=9, color=MUTED)
        if filename == "05_caar_confidence_intervals":
            fig.text(0.065, 0.068, "*Descriptive t intervals assume independent events; not cluster-adjusted or simultaneous bands.",
                     fontsize=8.5, color=MUTED)
        fig.savefig(figures / f"{filename}.png", dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12.8))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.79, bottom=0.16, hspace=0.53, wspace=0.23)
    header(fig, f"{label} | Analyst rating events", subtitle, badge)
    for (_, draw), ax in zip(PLOTS, axs.flat):
        draw(ax, result, window)
    fig.text(0.065, 0.045, foot, fontsize=10, color=MUTED)
    fig.text(0.065, 0.072, "Uncertainty estimates and sample limitations are reported separately in the README and tables.",
             fontsize=9, color=MUTED)
    fig.savefig(figures / "overview.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Use reproducible synthetic data")
    mode.add_argument("--returns", type=Path, help="CSV with Date,ret,mkt (decimal returns)")
    parser.add_argument("--events", type=Path, help="CSV with event_id,event_date,firm,event_type")
    parser.add_argument("--label", default=None)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.demo and args.events:
        parser.error("--events cannot be combined with --demo")
    if not args.demo and args.events is None:
        parser.error("--events is required with --returns")
    if args.demo:
        returns, events = make_demo_data(seed=args.seed)
    else:
        returns = pd.read_csv(args.returns, parse_dates=["Date"], index_col="Date")
        events = pd.read_csv(args.events)
    result = analyze_events(returns, events, window=args.window)
    if result.event_summary.empty:
        reasons = result.exclusions.to_string(index=False)
        parser.error("No complete event windows remain.\n" + reasons)
    out = args.output_dir or Path("outputs/demo" if args.demo else "outputs/local")
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_car(result.event_summary)
    for name, frame in {"event_panel": result.panel, "events": result.event_summary,
                        "event_time_summary": result.group_summary, "summary": summary,
                        "excluded_events": result.exclusions}.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    if args.demo:
        sample_dir = out / "inputs"
        sample_dir.mkdir(exist_ok=True)
        returns.rename_axis("Date").to_csv(sample_dir / "returns.csv")
        events.to_csv(sample_dir / "events.csv", index=False)
    manifest = {"data_kind": "synthetic" if args.demo else "local_csv",
                "label": args.label or ("Synthetic sample" if args.demo else "Imported sample"),
                "seed": args.seed if args.demo else None, "window_sessions": args.window,
                "input_events": len(events), "included_events": len(result.event_summary),
                "excluded_events": len(result.exclusions), "method": "market_adjusted",
                "returns_input": str(args.returns) if args.returns else None,
                "events_input": str(args.events) if args.events else None,
                "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (out / "run_metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    render(result, out, manifest["label"], args.window, args.demo)
    print(summary.to_string(index=False))
    print(f"Included {len(result.event_summary)} / {len(events)} events. Saved figures and tables to {out.resolve()}")
    return result


if __name__ == "__main__":
    main()
