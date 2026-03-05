"""Visualization module for backtest results.

Provides plotting functions for equity curves, drawdowns, and returns distributions.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot equity curve over time.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        The matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_df["timestamp"], equity_df["equity"], linewidth=1.5, color="blue")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_drawdown(
    equity_df: pd.DataFrame,
    title: str = "Drawdown",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot drawdown over time.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        The matplotlib Figure object
    """
    equity = equity_df["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(equity_df["timestamp"], drawdown, 0, alpha=0.3, color="red")
    ax.plot(equity_df["timestamp"], drawdown, linewidth=1, color="red")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Add horizontal line at max drawdown
    max_dd = drawdown.min()
    ax.axhline(y=max_dd, color="darkred", linestyle="--", alpha=0.7)
    ax.annotate(
        f"Max DD: {max_dd:.1f}%",
        xy=(equity_df["timestamp"].iloc[0], max_dd),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        color="darkred",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_returns_distribution(
    equity_df: pd.DataFrame,
    title: str = "Daily Returns Distribution",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot distribution of daily returns.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        The matplotlib Figure object
    """
    returns = equity_df["equity"].pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(returns, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(
        returns.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {returns.mean():.2f}%",
    )
    ax.axvline(0, color="black", linestyle="-", alpha=0.5)

    # Add standard deviation bands
    std = returns.std()
    ax.axvline(
        returns.mean() + std,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"+1 Std: {returns.mean() + std:.2f}%",
    )
    ax.axvline(
        returns.mean() - std,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"-1 Std: {returns.mean() - std:.2f}%",
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_monthly_returns(
    equity_df: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot monthly returns as a heatmap.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        The matplotlib Figure object
    """
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Calculate monthly returns
    monthly = df["equity"].resample("ME").last().pct_change() * 100
    monthly = monthly.dropna()

    # Create pivot table for heatmap
    monthly_df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return (%)")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            if j + 1 in pivot.columns:
                val = pivot.iloc[i, pivot.columns.get_loc(j + 1)]
                if not pd.isna(val):
                    color = "white" if abs(val) > 5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def create_full_report(
    equity_df: pd.DataFrame,
    output_dir: str = "output",
    show: bool = False,
) -> None:
    """Generate full visual report with all charts.

    Args:
        equity_df: DataFrame with 'timestamp' and 'equity' columns
        output_dir: Directory to save output files
        show: Whether to display plots (default False for batch processing)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating visual reports in {output_dir}/...")

    # Generate all plots
    plot_equity_curve(
        equity_df,
        save_path=f"{output_dir}/equity_curve.png",
        show=show,
    )
    plt.close()

    plot_drawdown(
        equity_df,
        save_path=f"{output_dir}/drawdown.png",
        show=show,
    )
    plt.close()

    plot_returns_distribution(
        equity_df,
        save_path=f"{output_dir}/returns_distribution.png",
        show=show,
    )
    plt.close()

    # Monthly returns heatmap (only if we have enough data)
    if len(equity_df) > 60:  # At least ~3 months of data
        plot_monthly_returns(
            equity_df,
            save_path=f"{output_dir}/monthly_returns.png",
            show=show,
        )
        plt.close()

    print(f"Report saved to {output_dir}/")
