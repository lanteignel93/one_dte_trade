import os  # For save_path in plot_daily_pnl_bar
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates

# Plotting library imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import scipy.stats as stats  # For confidence interval calculation (norm.ppf)
import seaborn as sns

from one_dte_trade import (
    analyzer_config as pre_loaded_analyzer_config,
)

# Also import the dataclass type for type hinting
from one_dte_trade.config import (
    AnalyzerConfig,
)


class Analyzer:
    """
    Analyzes DataFrames, providing descriptive statistics, correlation matrices,
    PnL statistics, and various plots. Designed to work with outputs from data
    pipelines like OneDTEDataPipeline. It uses pre-loaded configurations.
    """

    def __init__(self):
        """
        Initializes the Analyzer.
        It uses the pre-loaded 'analyzer_config' made available
        at the package level (expected to be loaded by __init__.py).
        """
        self.config: AnalyzerConfig = pre_loaded_analyzer_config

        if self.config is None:  # Should ideally not happen if __init__.py loads it
            print(
                "Analyzer Critical Warning: Pre-loaded analyzer_config is None. Instantiating a default AnalyzerConfig."
            )
            self.config = AnalyzerConfig()  # Fallback to default dataclass values

        # Optional: Validate if the loaded config has essential attributes
        # This helps catch issues if the pre-loaded config object is not as expected.
        essential_attrs = [
            "DEFAULT_PLOT_STYLE",
            "DEFAULT_FIGSIZE",
            "DEFAULT_ANNUALIZATION_FACTOR",
        ]
        for attr in essential_attrs:
            if not hasattr(self.config, attr):
                print(
                    f"Analyzer Warning: Loaded AnalyzerConfig is missing expected attribute '{attr}'. "
                    "Functionality relying on this config might use hardcoded defaults or fail."
                )
                # You could set a hardcoded default here if critical, e.g.:
                # if attr == 'DEFAULT_FIGSIZE': setattr(self.config, attr, (14, 7))

        print(
            f"Analyzer initialized with plot style from config: {getattr(self.config, 'DEFAULT_PLOT_STYLE', 'N/A')}"
        )

    def get_descriptive_stats(
        self,
        df: pl.DataFrame,
        columns_to_analyze: Optional[List[str]] = None,
        percentiles: Optional[List[float]] = None,
    ) -> pl.DataFrame:
        """
        Calculates descriptive statistics for specified numeric columns in the DataFrame.
        Statistics include: count, null_count, mean, std, min, specified percentiles, and max.
        """
        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for descriptive stats is empty.")
            return pl.DataFrame()

        if columns_to_analyze is None:
            columns_to_analyze = [
                col.name for col in df if col.dtype in pl.NUMERIC_DTYPES
            ]
        else:
            actual_columns = [col for col in columns_to_analyze if col in df.columns]
            if len(actual_columns) != len(columns_to_analyze):
                missing = set(columns_to_analyze) - set(actual_columns)
                print(
                    f"Analyzer Warning: Columns not found for descriptive stats and will be skipped: {missing}"
                )
            columns_to_analyze = actual_columns

        if not columns_to_analyze:
            print(
                "Analyzer Warning: No valid columns found or specified for descriptive statistics."
            )
            return pl.DataFrame()

        current_percentiles = (
            percentiles
            if percentiles is not None
            else getattr(
                self.config,
                "DEFAULT_PERCENTILES",
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            )
        )

        summary_list = []
        for col_name in columns_to_analyze:
            if df[col_name].dtype not in pl.NUMERIC_DTYPES:
                print(
                    f"Analyzer Warning: Column '{col_name}' is not numeric. Skipping for descriptive stats."
                )
                continue

            stats = df.select(
                [
                    pl.lit(col_name).alias("column"),
                    pl.col(col_name).count().alias("count"),
                    pl.col(col_name).null_count().alias("null_count"),
                    pl.col(col_name).mean().alias("mean"),
                    pl.col(col_name).std().alias("std"),
                    pl.col(col_name).min().alias("min"),
                    *[
                        pl.col(col_name)
                        .quantile(p, interpolation="linear")
                        .alias(f"p{int(p*100)}")
                        for p in current_percentiles
                    ],
                    pl.col(col_name).max().alias("max"),
                ]
            )
            summary_list.append(stats)

        return pl.concat(summary_list) if summary_list else pl.DataFrame()

    def get_correlation_matrix(
        self,
        df: pl.DataFrame,
        columns_to_correlate: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> pl.DataFrame:
        """Calculates the correlation matrix for specified numeric columns."""
        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for correlation matrix is empty.")
            return pl.DataFrame()

        if columns_to_correlate is None:
            columns_to_correlate = [
                col.name for col in df if col.dtype in pl.NUMERIC_DTYPES
            ]
        else:
            actual_columns = [col for col in columns_to_correlate if col in df.columns]
            if len(actual_columns) != len(columns_to_correlate):
                missing = set(columns_to_correlate) - set(actual_columns)
                print(
                    f"Analyzer Warning: Columns not found for correlation and will be skipped: {missing}"
                )
            columns_to_correlate = [
                col for col in actual_columns if df[col].dtype in pl.NUMERIC_DTYPES
            ]

            if len(columns_to_correlate) != len(actual_columns):
                non_numeric_skipped = set(actual_columns) - set(columns_to_correlate)
                if non_numeric_skipped:
                    print(
                        f"Analyzer Warning: Non-numeric columns skipped for correlation: {non_numeric_skipped}"
                    )

        if not columns_to_correlate or len(columns_to_correlate) < 2:
            print(
                "Analyzer Warning: Need at least two valid numeric columns to calculate a correlation matrix."
            )
            return pl.DataFrame()

        subset_df = df.select(columns_to_correlate).drop_nulls()

        if subset_df.height < 2:
            print(
                "Analyzer Warning: Not enough non-null data points to compute correlation matrix after dropping nulls."
            )
            return pl.DataFrame(
                {
                    "column": columns_to_correlate,
                    **{
                        col: [None] * (len(columns_to_correlate))
                        for col in columns_to_correlate
                    },
                }
            )

        corr_data = []
        for col_x in columns_to_correlate:
            row_corrs = {"column": col_x}
            for col_y in columns_to_correlate:
                try:
                    correlation_value = subset_df.select(
                        pl.corr(col_x, col_y, method=method)
                    ).item()
                except Exception as e:
                    print(
                        f"Analyzer Info: Could not compute correlation between '{col_x}' and '{col_y}'. Error: {e}. Setting to None."
                    )
                    correlation_value = None
                row_corrs[col_y] = correlation_value
            corr_data.append(row_corrs)

        return pl.DataFrame(corr_data, schema=["column"] + columns_to_correlate)

    def calculate_pnl_statistics(
        self,
        df: pl.DataFrame,
        pnl_col_name: str,
        annualization_factor: Optional[int] = None,
        risk_free_rate_daily: Optional[float] = None,
        portfolio_size_for_drawdown: Optional[float] = None,
    ) -> pl.DataFrame:
        """Calculates various performance statistics from a daily PnL column."""
        if pnl_col_name not in df.columns:
            print(
                f"Analyzer Error: PnL column '{pnl_col_name}' not found in DataFrame."
            )
            return pl.DataFrame(
                {"Statistic": [], "Value": []},
                schema={"Statistic": pl.String, "Value": pl.Float64},
            )
        if df.is_empty():
            print(
                f"Analyzer Warning: Input DataFrame for PnL stats is empty (column '{pnl_col_name}')."
            )
            return pl.DataFrame(
                {"Statistic": [], "Value": []},
                schema={"Statistic": pl.String, "Value": pl.Float64},
            )

        pnl_series = df[pnl_col_name].drop_nulls()

        ann_factor = (
            annualization_factor
            if annualization_factor is not None
            else getattr(self.config, "DEFAULT_ANNUALIZATION_FACTOR", 252)
        )
        rf_daily = (
            risk_free_rate_daily
            if risk_free_rate_daily is not None
            else getattr(self.config, "DEFAULT_RISK_FREE_RATE_DAILY", 0.0)
        )
        portfolio_base = (
            portfolio_size_for_drawdown
            if portfolio_size_for_drawdown is not None
            else getattr(
                self.config, "DEFAULT_PORTFOLIO_SIZE_FOR_DRAWDOWN", 1_000_000.0
            )
        )

        stat_names_list = [
            "Mean Daily PnL",
            "Std Dev Daily PnL",
            "Annualized Sharpe Ratio",
            "Win Rate (%)",
            "Max Daily Win",
            "Max Daily Loss",
            f"Worst 5-Day Cum PnL (% of {portfolio_base:,.0f})",
            f"Worst 20-Day Cum PnL (% of {portfolio_base:,.0f})",
            "Skewness",
            "Kurtosis",
            "Average Win Amount",
            "Average Loss Amount",
            "Total Return (%) over Period",
            "Annualized Total Return (%)",
        ]

        if pnl_series.len() == 0:
            print(
                f"Analyzer Warning: PnL column '{pnl_col_name}' is empty after dropping nulls. Returning nulls for stats."
            )
            return pl.DataFrame(
                {"Statistic": stat_names_list, "Value": [None] * len(stat_names_list)},
                schema={"Statistic": pl.String, "Value": pl.Float64},
            )

        mean_pnl = pnl_series.mean()
        std_pnl = pnl_series.std()

        sharpe_ratio = None
        if std_pnl is not None and std_pnl != 0 and mean_pnl is not None:
            sharpe_ratio = ((mean_pnl - rf_daily) / std_pnl) * (ann_factor**0.5)
        elif (
            mean_pnl is not None and mean_pnl == 0 and (std_pnl is None or std_pnl == 0)
        ):
            sharpe_ratio = 0.0

        wins = pnl_series.filter(pnl_series > 0).len()
        total_days_with_pnl = pnl_series.len()
        win_rate = (
            (float(wins) / total_days_with_pnl * 100)
            if total_days_with_pnl > 0
            else 0.0
        )

        max_daily_win = pnl_series.max()
        max_daily_loss = pnl_series.min()

        def calculate_worst_n_day_pnl_pct(
            series: pl.Series, n: int, base_value: float
        ) -> Optional[float]:
            if series.len() < n or base_value == 0:
                return None
            rolling_sum_pnl = series.rolling_sum(window_size=n, min_periods=n)
            min_rolling_sum_abs = rolling_sum_pnl.min()
            return (
                (min_rolling_sum_abs / base_value * 100)
                if min_rolling_sum_abs is not None
                else None
            )

        worst_5_day_pnl_pct = calculate_worst_n_day_pnl_pct(
            pnl_series, 5, portfolio_base
        )
        worst_20_day_pnl_pct = calculate_worst_n_day_pnl_pct(
            pnl_series, 20, portfolio_base
        )

        skewness = pnl_series.skew() if total_days_with_pnl > 2 else None
        kurt = pnl_series.kurtosis() if total_days_with_pnl > 3 else None

        winners_series = pnl_series.filter(pnl_series > 0)
        losers_series = pnl_series.filter(pnl_series < 0)
        avg_win_amount = winners_series.mean() if winners_series.len() > 0 else 0.0
        avg_loss_amount = losers_series.mean() if losers_series.len() > 0 else 0.0

        total_pnl_sum = pnl_series.sum()
        total_return_pct = (
            (total_pnl_sum / portfolio_base * 100)
            if portfolio_base != 0 and total_pnl_sum is not None
            else None
        )

        annualized_total_return_pct = None
        if total_return_pct is not None and total_days_with_pnl > 0 and ann_factor > 0:
            annualized_total_return_pct = (
                ((1 + total_return_pct / 100) ** (ann_factor / total_days_with_pnl)) - 1
            ) * 100

        stat_values_list = [
            mean_pnl,
            std_pnl,
            sharpe_ratio,
            win_rate,
            max_daily_win,
            max_daily_loss,
            worst_5_day_pnl_pct,
            worst_20_day_pnl_pct,
            skewness,
            kurt,
            avg_win_amount,
            avg_loss_amount,
            total_return_pct,
            annualized_total_return_pct,
        ]

        processed_stat_values = []
        for v in stat_values_list:
            if isinstance(v, (int, float)):
                is_problematic_float = isinstance(v, float) and (
                    np.isnan(v) or np.isinf(v)
                )
                processed_stat_values.append(None if is_problematic_float else float(v))
            elif v is None:
                processed_stat_values.append(None)
            else:
                print(
                    f"Analyzer Warning: Unexpected type in PnL stats: {type(v)}, value: {v}. Setting to None."
                )
                processed_stat_values.append(None)

        return pl.DataFrame(
            {"Statistic": stat_names_list, "Value": processed_stat_values},
            schema={"Statistic": pl.String, "Value": pl.Float64},
        )

    def plot_timeseries(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        style: Optional[str] = None,
        line_color: Optional[str] = None,
        line_width: float = 1.5,
        marker: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        date_format_str: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Creates and displays a styled time series plot from a Polars DataFrame."""
        if df.is_empty() or time_col not in df.columns or value_col not in df.columns:
            print(
                f"Analyzer Warning: DataFrame empty or required columns ('{time_col}', '{value_col}') not found for timeseries plot."
            )
            return
        if df[value_col].drop_nulls().is_empty():
            print(
                f"Analyzer Warning: Value column '{value_col}' is all nulls or empty after dropping nulls. Skipping timeseries plot."
            )
            return

        plot_title = (
            title
            if title is not None
            else f"{value_col.replace('_', ' ').title()} over Time"
        )
        plot_xlabel = (
            xlabel if xlabel is not None else time_col.replace("_", " ").title()
        )
        plot_ylabel = (
            ylabel if ylabel is not None else value_col.replace("_", " ").title()
        )
        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (14, 7))
        )
        plot_date_format = (
            date_format_str
            if date_format_str is not None
            else getattr(self.config, "DEFAULT_DATE_FORMAT_STR", "%Y-%m-%d")
        )
        plot_line_color = (
            line_color
            if line_color is not None
            else getattr(self.config, "DEFAULT_TIMESERIES_LINE_COLOR", "maroon")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Using default. Error: {e}"
            )

        df_sorted = df.sort(time_col)
        time_data_total = df_sorted[time_col]
        value_data_total = df_sorted[value_col]

        plt.figure(figsize=plot_figsize)
        plt.plot(
            time_data_total.to_list(),
            value_data_total.to_list(),
            color=plot_line_color,
            linewidth=line_width,
            marker=marker if marker else "",
            label="Total Cumulative PnL",
        )

        plot_long_short_breakdown = False
        position_col_for_breakdown = None
        daily_pnl_col_for_breakdown = None
        base_signal_name = None

        if value_col == "ret_uprc_signal_cum_pnl":
            base_signal_name = "ret_uprc_signal"
        elif value_col == "ret_sprc_signal_cum_pnl":
            base_signal_name = "ret_sprc_signal"

        if base_signal_name:
            position_col_for_breakdown = f"{base_signal_name}_pos"
            daily_pnl_col_for_breakdown = f"{base_signal_name}_pnl"
            plot_long_short_breakdown = True

        if plot_long_short_breakdown:
            if (
                position_col_for_breakdown in df_sorted.columns
                and daily_pnl_col_for_breakdown in df_sorted.columns
            ):
                short_df = df_sorted.filter(pl.col(position_col_for_breakdown) < 0)
                if not short_df.is_empty():
                    short_df_cum_pnl = short_df.with_columns(
                        pl.col(daily_pnl_col_for_breakdown)
                        .cum_sum()
                        .alias("short_pnl_contribution")
                    )
                    plt.plot(
                        short_df_cum_pnl[time_col].to_list(),
                        short_df_cum_pnl["short_pnl_contribution"].to_list(),
                        color="green",
                        linewidth=line_width,
                        marker=marker if marker else "",
                        label="Short PnL (Cumulative)",
                    )
                long_df = df_sorted.filter(pl.col(position_col_for_breakdown) > 0)
                if not long_df.is_empty():
                    long_df_cum_pnl = long_df.with_columns(
                        pl.col(daily_pnl_col_for_breakdown)
                        .cum_sum()
                        .alias("long_pnl_contribution")
                    )
                    plt.plot(
                        long_df_cum_pnl[time_col].to_list(),
                        long_df_cum_pnl["long_pnl_contribution"].to_list(),
                        color="blue",
                        linewidth=line_width,
                        marker=marker if marker else "",
                        label="Long PnL (Cumulative)",
                    )
            else:
                print(
                    f"Analyzer Warning: Required columns for long/short PnL breakdown ('{position_col_for_breakdown}', '{daily_pnl_col_for_breakdown}') not found. Skipping breakdown for {value_col}."
                )

        plt.title(
            plot_title,
            fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
            pad=20,
        )
        plt.xlabel(
            plot_xlabel,
            fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
            labelpad=15,
        )
        plt.ylabel(
            plot_ylabel,
            fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
            labelpad=15,
        )
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(plot_date_format))
        plt.gcf().autofmt_xdate()
        y_formatter = mticker.ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(y_formatter)
        plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout(pad=1.5)
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Analyzer Error: Error saving plot to {save_path}: {e}")
        plt.show()

    def plot_histogram_kde(
        self,
        df: pl.DataFrame,
        value_col: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Density",
        bins: Optional[Any] = None,
        style: Optional[str] = None,
        hist_color: Optional[str] = None,
        kde_color: Optional[str] = None,
        kde_linewidth: float = 2.0,
        figsize: Optional[tuple[float, float]] = None,
        filter_outliers_abs_limit: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Creates histogram with KDE, optionally filtering extreme outliers."""
        if df.is_empty() or value_col not in df.columns:
            print(
                f"Analyzer Warning: DataFrame empty or value column '{value_col}' not found for histogram/KDE."
            )
            return

        data_to_plot_series = df[value_col].drop_nulls()
        if data_to_plot_series.is_empty():
            print(
                f"Analyzer Warning: Value column '{value_col}' is all nulls after drop_nulls. Skipping histogram/KDE."
            )
            return

        if filter_outliers_abs_limit is not None:
            original_len = data_to_plot_series.len()
            data_to_plot_series = data_to_plot_series.filter(
                data_to_plot_series.abs() < filter_outliers_abs_limit
            )
            if data_to_plot_series.len() < original_len:
                print(
                    f"Analyzer Info: Filtered {original_len - data_to_plot_series.len()} outliers from '{value_col}' (abs limit: {filter_outliers_abs_limit})."
                )

        if data_to_plot_series.is_empty():
            print(
                f"Analyzer Warning: Value column '{value_col}' is empty after outlier filtering. Skipping histogram/KDE."
            )
            return

        plot_title = (
            title
            if title is not None
            else f"Distribution of {value_col.replace('_', ' ').title()}"
        )
        plot_xlabel = (
            xlabel if xlabel is not None else value_col.replace("_", " ").title()
        )
        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (10, 6))
        )
        plot_bins = (
            bins
            if bins is not None
            else getattr(self.config, "DEFAULT_HISTOGRAM_BINS", "auto")
        )
        plot_hist_color = (
            hist_color
            if hist_color is not None
            else getattr(self.config, "DEFAULT_HISTOGRAM_COLOR", "cornflowerblue")
        )
        plot_kde_color = (
            kde_color
            if kde_color is not None
            else getattr(self.config, "DEFAULT_KDE_COLOR", "red")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Using default. Error: {e}"
            )

        plt.figure(figsize=plot_figsize)
        sns.histplot(
            data_to_plot_series.to_numpy(),
            bins=plot_bins,
            kde=True,
            stat="density",
            color=plot_hist_color,
            line_kws={"linewidth": kde_linewidth, "color": plot_kde_color},
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(
            plot_title,
            fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
            pad=20,
        )
        plt.xlabel(
            plot_xlabel,
            fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
            labelpad=15,
        )
        plt.ylabel(
            ylabel,
            fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
            labelpad=15,
        )
        ax = plt.gca()
        x_formatter = mticker.ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(x_formatter)
        plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout(pad=1.5)
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Analyzer Error: Error saving plot to {save_path}: {e}")
        plt.show()

    def plot_pnl_bars(
        self,
        df: pl.DataFrame,
        date_col: str,
        pnl_col: str,
        title_base: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        style: Optional[str] = None,
        positive_color: Optional[str] = None,
        negative_color: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        date_format_str: Optional[str] = None,
        save_path_base: Optional[str] = None,
        aggregation_period: str = "daily",
    ) -> None:
        """Creates and displays bar plots of PnL, aggregated by specified period."""
        if df.is_empty() or date_col not in df.columns or pnl_col not in df.columns:
            print(
                f"Analyzer Warning: DataFrame empty or required columns ('{date_col}', '{pnl_col}') not found for PnL bar plot."
            )
            return

        df_cleaned = df.drop_nulls(subset=[date_col, pnl_col])
        if df_cleaned.is_empty():
            print(
                f"Analyzer Info: DataFrame is empty after dropping nulls in '{date_col}' or '{pnl_col}'. Cannot plot PnL bars."
            )
            return

        plot_xlabel = (
            xlabel if xlabel is not None else date_col.replace("_", " ").title()
        )
        plot_ylabel = (
            ylabel if ylabel is not None else pnl_col.replace("_", " ").title()
        )
        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (15, 7))
        )
        plot_date_format = (
            date_format_str
            if date_format_str is not None
            else getattr(self.config, "DEFAULT_DATE_FORMAT_STR", "%Y-%m-%d")
        )
        plot_title_base = (
            title_base
            if title_base is not None
            else f"{pnl_col.replace('_', ' ').title()} PnL"
        )
        plot_positive_color = (
            positive_color
            if positive_color is not None
            else getattr(
                self.config, "DEFAULT_PNL_BAR_POSITIVE_COLOR", "mediumseagreen"
            )
        )
        plot_negative_color = (
            negative_color
            if negative_color is not None
            else getattr(self.config, "DEFAULT_PNL_BAR_NEGATIVE_COLOR", "lightcoral")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Using default. Error: {e}"
            )

        if not isinstance(df_cleaned[date_col].dtype, (pl.Date, pl.Datetime)):
            print(
                f"Analyzer Warning: Date column '{date_col}' is not Date/Datetime. Attempting cast."
            )
            try:
                df_cleaned = df_cleaned.with_columns(pl.col(date_col).cast(pl.Date))
            except Exception as cast_err:
                print(
                    f"Analyzer Error: Failed to cast date column '{date_col}': {cast_err}. Cannot plot."
                )
                return

        def _plot_single_aggregated_pnl_chart(
            data_df,
            x_data_col_name,
            y_data_col_name,
            current_plot_title,
            current_save_path,
        ):
            dates_to_plot = data_df[x_data_col_name].to_list()
            pnl_values_to_plot = data_df[y_data_col_name].to_list()
            if not dates_to_plot:
                return
            colors = [
                plot_positive_color if pnl >= 0 else plot_negative_color
                for pnl in pnl_values_to_plot
            ]
            plt.figure(figsize=plot_figsize)
            num_bars = len(dates_to_plot)
            bar_width_map = {"daily": 0.8, "weekly": 5.0, "monthly": 20.0}
            bar_width = bar_width_map.get(aggregation_period, 0.8)
            if num_bars > 0 and num_bars < 30 and aggregation_period != "daily":
                bar_width = max(
                    1.0,
                    (30 / num_bars) * (bar_width_map.get(aggregation_period, 0.8) / 5),
                )
            plt.bar(dates_to_plot, pnl_values_to_plot, color=colors, width=bar_width)
            plt.title(
                current_plot_title,
                fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
                pad=20,
            )
            plt.xlabel(
                plot_xlabel,
                fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                labelpad=15,
            )
            plt.ylabel(
                plot_ylabel,
                fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                labelpad=15,
            )
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter(plot_date_format))
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            y_formatter = mticker.ScalarFormatter(useOffset=False)
            y_formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.tight_layout(pad=1.5)
            if current_save_path:
                try:
                    plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                    print(f"Plot saved to {current_save_path}")
                except Exception as e:
                    print(f"Analyzer Error saving plot to {current_save_path}: {e}")
            plt.show()

        if aggregation_period == "daily":
            df_with_period = df_cleaned.sort(date_col).with_columns(
                [
                    pl.col(date_col).dt.year().alias("plot_year"),
                    (((pl.col(date_col).dt.month() - 1) // 6) + 1).alias(
                        "plot_semester"
                    ),
                ]
            )
            period_groups = (
                df_with_period.select(["plot_year", "plot_semester"])
                .unique()
                .sort(["plot_year", "plot_semester"])
            )
            if period_groups.is_empty():
                print("Analyzer Info: No data periods for daily PnL plot.")
                return
            for row in period_groups.iter_rows(named=True):
                year, semester = row["plot_year"], row["plot_semester"]
                df_period_daily = df_with_period.filter(
                    (pl.col("plot_year") == year)
                    & (pl.col("plot_semester") == semester)
                )
                if df_period_daily.is_empty():
                    continue
                min_d, max_d = (
                    df_period_daily[date_col].min(),
                    df_period_daily[date_col].max(),
                )
                period_title = f"{plot_title_base} - Daily ({year} H{semester}: {min_d:%b %Y} - {max_d:%b %Y})"
                current_sp = None
                if save_path_base:
                    base, ext = os.path.splitext(save_path_base)
                    current_sp = f"{base}_daily_{year}H{semester}{ext}"
                _plot_single_aggregated_pnl_chart(
                    df_period_daily, date_col, pnl_col, period_title, current_sp
                )

        elif aggregation_period in ["weekly", "monthly"]:
            truncate_by = "1w" if aggregation_period == "weekly" else "1mo"
            agg_date_col_name = f"{aggregation_period}_start_date"
            df_aggregated = (
                df_cleaned.group_by(
                    pl.col(date_col).dt.truncate(truncate_by).alias(agg_date_col_name)
                )
                .agg(pl.col(pnl_col).sum().alias(pnl_col))
                .sort(agg_date_col_name)
            )
            if df_aggregated.is_empty():
                print(f"Analyzer Info: No data after {aggregation_period} aggregation.")
                return
            agg_title = f"{plot_title_base} - {aggregation_period.capitalize()}"
            current_sp = None
            if save_path_base:
                base, ext = os.path.splitext(save_path_base)
                current_sp = f"{base}_{aggregation_period}{ext}"
            _plot_single_aggregated_pnl_chart(
                df_aggregated, agg_date_col_name, pnl_col, agg_title, current_sp
            )
        else:
            print(
                f"Analyzer Error: Unsupported aggregation_period '{aggregation_period}'. Choose 'daily', 'weekly', or 'monthly'."
            )

    def plot_signal_decile_sharpe_barchart(
        self,
        df: pl.DataFrame,
        signal_bases: List[str],
        num_deciles: int = 10,
        figsize: Optional[tuple[float, float]] = None,
        style: Optional[str] = None,
        bar_color: Optional[str] = None,
        save_path_prefix: Optional[str] = None,
    ) -> None:
        """Creates bar plots of annualized Sharpe ratios for deciles of long and short signals."""
        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for decile analysis is empty.")
            return
        if not signal_bases:
            print("Analyzer Warning: No signal_bases provided for decile analysis.")
            return

        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (12, 7))
        )
        ann_factor = getattr(self.config, "DEFAULT_ANNUALIZATION_FACTOR", 252)
        rf_daily = getattr(self.config, "DEFAULT_RISK_FREE_RATE_DAILY", 0.0)
        default_long_color = getattr(
            self.config, "DEFAULT_DECILE_BAR_COLOR_LONG", "mediumseagreen"
        )
        default_short_color = getattr(
            self.config, "DEFAULT_DECILE_BAR_COLOR_SHORT", "lightcoral"
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Error: {e}"
            )

        base_decile_qcut_labels = [f"D{i+1}" for i in range(num_deciles)]

        for base_signal_name in signal_bases:
            signal_strength_col = base_signal_name
            position_col = f"{base_signal_name}_pos"
            daily_pnl_col = f"{base_signal_name}_pnl"

            if not all(
                c in df.columns
                for c in [signal_strength_col, position_col, daily_pnl_col]
            ):
                print(
                    f"Analyzer Warning: Missing required columns for base signal '{base_signal_name}'. Skipping."
                )
                continue

            for position_type in ["long", "short"]:
                current_plot_bar_color = bar_color  # Use argument if provided
                if (
                    current_plot_bar_color is None
                ):  # Fallback to specific or general config
                    current_plot_bar_color = (
                        default_long_color
                        if position_type == "long"
                        else default_short_color
                    )

                filtered_df = (
                    df.filter(pl.col(position_col) > 0)
                    if position_type == "long"
                    else df.filter(pl.col(position_col) < 0)
                )
                title_suffix = f"{position_type.capitalize()} Positions"

                if filtered_df.is_empty():
                    print(
                        f"Analyzer Info: No {position_type} positions for signal '{base_signal_name}'. Skipping."
                    )
                    continue

                try:
                    df_with_deciles = filtered_df.with_columns(
                        pl.col(signal_strength_col)
                        .qcut(
                            num_deciles,
                            labels=base_decile_qcut_labels,
                            allow_duplicates=True,
                        )
                        .alias("signal_decile_label")
                    )
                except Exception as e:
                    print(
                        f"Analyzer Error: Could not create deciles for {base_signal_name} ({position_type}). Error: {e}. Skipping."
                    )
                    continue

                plot_data_list = []
                for decile_qcut_label in base_decile_qcut_labels:
                    current_decile_data = df_with_deciles.filter(
                        pl.col("signal_decile_label") == decile_qcut_label
                    )
                    min_signal, max_signal, sharpe = np.nan, np.nan, np.nan
                    decile_axis_label = f"{decile_qcut_label} (No Data)"
                    if not current_decile_data.is_empty():
                        min_signal = current_decile_data[signal_strength_col].min()
                        max_signal = current_decile_data[signal_strength_col].max()
                        if min_signal is not None and max_signal is not None:
                            decile_axis_label = f"{min_signal:.2f} to {max_signal:.2f}"
                        decile_pnl_series = current_decile_data[
                            daily_pnl_col
                        ].drop_nulls()
                        if decile_pnl_series.len() >= 2:
                            mean_pnl, std_pnl = (
                                decile_pnl_series.mean(),
                                decile_pnl_series.std(),
                            )
                            if (
                                std_pnl is not None
                                and std_pnl != 0
                                and mean_pnl is not None
                            ):
                                sharpe = ((mean_pnl - rf_daily) / std_pnl) * (
                                    ann_factor**0.5
                                )
                            elif (
                                mean_pnl is not None
                                and mean_pnl == 0
                                and (std_pnl is None or std_pnl == 0)
                            ):
                                sharpe = 0.0
                    plot_data_list.append(
                        {
                            "DecileAxisLabel": decile_axis_label,
                            "SharpeRatio": sharpe,
                            "_sort_key": base_decile_qcut_labels.index(
                                decile_qcut_label
                            ),
                        }
                    )

                if not plot_data_list:
                    print(
                        f"Analyzer Info: No decile data for {base_signal_name} - {title_suffix}."
                    )
                    continue
                plot_display_df = pl.DataFrame(plot_data_list).sort("_sort_key")
                plot_display_df_cleaned = plot_display_df.filter(
                    pl.col("SharpeRatio").is_not_nan()
                )
                if plot_display_df_cleaned.is_empty():
                    print(
                        f"Analyzer Info: No non-NaN Sharpe ratios for {base_signal_name} - {title_suffix}."
                    )
                    continue

                plt.figure(figsize=plot_figsize)
                sns.barplot(
                    x="DecileAxisLabel",
                    y="SharpeRatio",
                    data=plot_display_df_cleaned.to_pandas(),
                    color=current_plot_bar_color,
                    errorbar=None,
                )
                plot_title = f"Annualized Sharpe Ratio by Signal Decile\n({base_signal_name} - {title_suffix})"
                plt.title(
                    plot_title,
                    fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
                    pad=15,
                )
                plt.xlabel(
                    "Signal Strength Decile (Range)",
                    fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                    labelpad=10,
                )
                plt.ylabel(
                    "Annualized Sharpe Ratio",
                    fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                    labelpad=10,
                )
                ax = plt.gca()
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
                y_formatter = mticker.FormatStrFormatter("%.2f")
                ax.yaxis.set_major_formatter(y_formatter)
                plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
                plt.tight_layout()
                if save_path_prefix:
                    current_save_path = f"{save_path_prefix}sharpe_barchart_{base_signal_name}_{position_type}_deciles.png"
                    try:
                        plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                        print(f"Plot saved to {current_save_path}")
                    except Exception as e:
                        print(f"Analyzer Error saving plot to {current_save_path}: {e}")
                plt.show()

    def plot_signal_autocorrelation(
        self,
        df: pl.DataFrame,
        signal_cols_to_analyze: List[str],
        max_lags: int = 40,
        figsize: Optional[tuple[float, float]] = None,
        style: Optional[str] = None,
        line_color: Optional[str] = None,
        marker_color: Optional[str] = None,
        show_confidence_intervals: bool = True,
        confidence_level: float = 0.95,
        save_path_prefix: Optional[str] = None,
    ) -> None:
        """Calculates and plots the Autocorrelation Function (ACF) for specified signal columns."""
        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for ACF analysis is empty.")
            return
        if not signal_cols_to_analyze:
            print("Analyzer Warning: No signal_cols_to_analyze for ACF plots.")
            return

        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (12, 6))
        )
        plot_line_color = (
            line_color
            if line_color is not None
            else getattr(self.config, "DEFAULT_ACF_LINE_COLOR", "steelblue")
        )
        plot_marker_color = (
            marker_color
            if marker_color is not None
            else getattr(self.config, "DEFAULT_ACF_MARKER_COLOR", "crimson")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Error: {e}"
            )

        for signal_col in signal_cols_to_analyze:
            if signal_col not in df.columns:
                print(
                    f"Analyzer Warning: Signal column '{signal_col}' not found. Skipping ACF."
                )
                continue
            series = df[signal_col].drop_nulls()
            if series.len() < max_lags + 1 or series.len() < 2:
                print(
                    f"Analyzer Warning: Not enough data in '{signal_col}' for ACF up to {max_lags} lags. Skipping."
                )
                continue

            acf_values = [1.0]
            temp_df_for_corr = pl.DataFrame({"original": series})
            for i in range(1, max_lags + 1):
                acf_val = np.nan
                if series.len() - i >= 2:
                    try:
                        lagged_series_name = f"lagged_{i}"
                        df_for_lag_corr = temp_df_for_corr.with_columns(
                            pl.col("original").shift(i).alias(lagged_series_name)
                        )
                        corr_result = df_for_lag_corr.select(
                            pl.corr("original", lagged_series_name)
                        ).item()
                        if corr_result is not None:
                            acf_val = corr_result
                    except Exception as e:
                        print(
                            f"Analyzer Info: Error calculating ACF for lag {i} of '{signal_col}': {e}. Setting to NaN."
                        )
                acf_values.append(acf_val)

            lags_range = np.arange(0, max_lags + 1)
            valid_indices = [i for i, val in enumerate(acf_values) if not np.isnan(val)]
            if not valid_indices:
                print(
                    f"Analyzer Warning: All ACF values NaN for '{signal_col}'. Skipping plot."
                )
                continue
            plot_lags, plot_acf_values = (
                lags_range[valid_indices],
                np.array(acf_values)[valid_indices],
            )

            plt.figure(figsize=plot_figsize)
            markerline, stemlines, baseline = plt.stem(
                plot_lags, plot_acf_values, linefmt="-", markerfmt="o", basefmt="-"
            )
            plt.setp(stemlines, "color", plot_line_color, "linewidth", 1.5)
            plt.setp(
                markerline, "color", plot_marker_color, "markersize", 5
            )  # Set marker color explicitly
            plt.setp(
                baseline, "color", "black", "linewidth", 0.5
            )  # Ensure baseline is visible

            if show_confidence_intervals:
                n_obs = series.len()
                if n_obs > 1:
                    conf_val = stats.norm.ppf(1 - (1 - confidence_level) / 2) / np.sqrt(
                        n_obs
                    )
                    plt.axhline(y=conf_val, color="gray", linestyle="--", linewidth=0.8)
                    plt.axhline(
                        y=-conf_val, color="gray", linestyle="--", linewidth=0.8
                    )
                    plt.fill_between(
                        lags_range,
                        -conf_val,
                        conf_val,
                        alpha=0.1,
                        color="gray",
                        label=f"{int(confidence_level*100)}% CI",
                    )

            plot_title = f"Autocorrelation Function (ACF) for {signal_col.replace('_', ' ').title()}"
            plt.title(
                plot_title,
                fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
                pad=15,
            )
            plt.xlabel(
                "Lag",
                fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                labelpad=10,
            )
            plt.ylabel(
                "Autocorrelation",
                fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                labelpad=10,
            )
            plt.xticks(np.arange(0, max_lags + 1, step=max(1, max_lags // 10)))
            plt.ylim([-1.05, 1.05])
            plt.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
            if show_confidence_intervals and n_obs > 1:
                plt.legend(loc="upper right")
            plt.tight_layout()
            if save_path_prefix:
                current_save_path = f"{save_path_prefix}acf_{signal_col}.png"
                try:
                    plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                    print(f"Plot saved to {current_save_path}")
                except Exception as e:
                    print(f"Analyzer Error saving plot to {current_save_path}: {e}")
            plt.show()

    def plot_column_decile_signal_sharpe_barchart(
        self,
        df: pl.DataFrame,
        columns_to_decile: List[str],
        signal_bases: List[str],
        num_deciles: int = 10,
        figsize: Optional[tuple[float, float]] = None,
        style: Optional[str] = None,
        bar_color_long: Optional[str] = None,
        bar_color_short: Optional[str] = None,
        save_path_prefix: Optional[str] = None,
    ) -> None:
        """For each column_to_decile and each signal_base, creates bar plots of the
        annualized Sharpe ratio of the signal's PnL, bucketed by deciles of the
        column_to_decile. Plots are generated separately for long and short positions.
        The x-axis shows decile ranges of the column_to_decile."""

        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for decile analysis is empty.")
            return
        if not columns_to_decile:
            print("Analyzer Warning: No columns_to_decile provided.")
            return
        if not signal_bases:
            print("Analyzer Warning: No signal_bases provided.")
            return

        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (12, 7))
        )
        ann_factor = getattr(self.config, "DEFAULT_ANNUALIZATION_FACTOR", 252)
        rf_daily = getattr(self.config, "DEFAULT_RISK_FREE_RATE_DAILY", 0.0)
        default_bar_color_long = (
            bar_color_long
            if bar_color_long is not None
            else getattr(self.config, "DEFAULT_DECILE_BAR_COLOR_LONG", "mediumseagreen")
        )
        default_bar_color_short = (
            bar_color_short
            if bar_color_short is not None
            else getattr(self.config, "DEFAULT_DECILE_BAR_COLOR_SHORT", "lightcoral")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Error: {e}"
            )

        base_decile_qcut_labels = [f"D{i+1}" for i in range(num_deciles)]

        for decile_col_name in columns_to_decile:
            if decile_col_name not in df.columns:
                print(
                    f"Analyzer Warning: Column '{decile_col_name}' for deciling not found. Skipping."
                )
                continue
            if df[decile_col_name].drop_nulls().is_empty():
                print(
                    f"Analyzer Warning: Column '{decile_col_name}' for deciling is all nulls or empty. Skipping."
                )
                continue

            for base_signal_name in signal_bases:
                position_col = f"{base_signal_name}_pos"
                daily_pnl_col = f"{base_signal_name}_pnl"
                if not all(c in df.columns for c in [position_col, daily_pnl_col]):
                    print(
                        f"Analyzer Warning: Missing PnL/Pos columns for signal '{base_signal_name}' with decile col '{decile_col_name}'. Skipping."
                    )
                    continue

                for position_type in ["long", "short"]:
                    current_bar_color = (
                        default_bar_color_long
                        if position_type == "long"
                        else default_bar_color_short
                    )

                    filtered_df_by_pos = (
                        df.filter(pl.col(position_col) > 0)
                        if position_type == "long"
                        else df.filter(pl.col(position_col) < 0)
                    )
                    title_suffix = f"{position_type.capitalize()} Positions (Signal: {base_signal_name})"

                    if (
                        filtered_df_by_pos.is_empty()
                        or filtered_df_by_pos[decile_col_name].drop_nulls().is_empty()
                    ):
                        print(
                            f"Analyzer Info: No {position_type} positions with valid '{decile_col_name}' data for signal '{base_signal_name}'. Skipping."
                        )
                        continue

                    try:
                        df_with_deciles = filtered_df_by_pos.with_columns(
                            pl.col(decile_col_name)
                            .qcut(
                                num_deciles,
                                labels=base_decile_qcut_labels,
                                allow_duplicates=True,
                            )
                            .alias(f"{decile_col_name}_decile_label")
                        )
                    except Exception as e:
                        print(
                            f"Analyzer Error: Could not create deciles for '{decile_col_name}' ({base_signal_name} - {position_type}). Error: {e}. Skipping."
                        )
                        continue

                    plot_data_list = []
                    for i, decile_qcut_label_val in enumerate(base_decile_qcut_labels):
                        current_decile_data = df_with_deciles.filter(
                            pl.col(f"{decile_col_name}_decile_label")
                            == decile_qcut_label_val
                        )
                        min_val, max_val, sharpe = np.nan, np.nan, np.nan
                        decile_axis_label = f"{decile_qcut_label_val} (No Data)"
                        if not current_decile_data.is_empty():
                            min_val, max_val = (
                                current_decile_data[decile_col_name].min(),
                                current_decile_data[decile_col_name].max(),
                            )
                            if min_val is not None and max_val is not None:
                                decile_axis_label = f"{min_val:.2f} to {max_val:.2f}"
                            pnl_series_for_decile = current_decile_data[
                                daily_pnl_col
                            ].drop_nulls()
                            if pnl_series_for_decile.len() >= 2:
                                mean_pnl, std_pnl = (
                                    pnl_series_for_decile.mean(),
                                    pnl_series_for_decile.std(),
                                )
                                if (
                                    std_pnl is not None
                                    and std_pnl != 0
                                    and mean_pnl is not None
                                ):
                                    sharpe = ((mean_pnl - rf_daily) / std_pnl) * (
                                        ann_factor**0.5
                                    )
                                elif (
                                    mean_pnl is not None
                                    and mean_pnl == 0
                                    and (std_pnl is None or std_pnl == 0)
                                ):
                                    sharpe = 0.0
                        plot_data_list.append(
                            {
                                "DecileAxisLabel": decile_axis_label,
                                "SharpeRatio": sharpe,
                                "_sort_key": i,
                            }
                        )

                    if not plot_data_list:
                        print(
                            f"Analyzer Info: No decile data to plot for '{decile_col_name}' ({base_signal_name} - {title_suffix})."
                        )
                        continue
                    plot_display_df = pl.DataFrame(plot_data_list).sort("_sort_key")
                    plot_display_df_cleaned = plot_display_df.filter(
                        pl.col("SharpeRatio").is_not_nan()
                    )
                    if plot_display_df_cleaned.is_empty():
                        print(
                            f"Analyzer Info: No non-NaN Sharpe ratios for '{decile_col_name}' deciles ({base_signal_name} - {title_suffix})."
                        )
                        continue

                    plt.figure(figsize=plot_figsize)
                    sns.barplot(
                        x="DecileAxisLabel",
                        y="SharpeRatio",
                        data=plot_display_df_cleaned.to_pandas(),
                        color=current_bar_color,
                        errorbar=None,
                    )
                    plot_title = f"Sharpe Ratio of '{base_signal_name}' PnL\nby Deciles of '{decile_col_name.replace('_',' ').title()}' ({title_suffix})"
                    plt.title(
                        plot_title,
                        fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
                        pad=15,
                    )
                    plt.xlabel(
                        f"Decile Range of {decile_col_name.replace('_',' ').title()}",
                        fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                        labelpad=10,
                    )
                    plt.ylabel(
                        "Annualized Sharpe Ratio",
                        fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                        labelpad=10,
                    )
                    ax = plt.gca()
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
                    y_formatter = mticker.FormatStrFormatter("%.2f")
                    ax.yaxis.set_major_formatter(y_formatter)
                    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    if save_path_prefix:
                        if os.path.dirname(save_path_prefix) and not os.path.exists(
                            os.path.dirname(save_path_prefix)
                        ):
                            os.makedirs(
                                os.path.dirname(save_path_prefix), exist_ok=True
                            )
                        filename_decile_col = decile_col_name.replace("/", "_")
                        current_save_path = f"{save_path_prefix}sharpe_by_{filename_decile_col}_deciles_for_{base_signal_name}_{position_type}.png"
                        try:
                            plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                            print(f"Plot saved to {current_save_path}")
                        except Exception as e:
                            print(
                                f"Analyzer Error saving plot to {current_save_path}: {e}"
                            )
                    plt.show()

    def plot_column_fixed_bin_signal_sharpe_barchart(
        self,
        df: pl.DataFrame,
        columns_to_bin: List[str],
        signal_bases: List[str],
        fixed_breaks: List[float] = [-2, -1, -0.5, 0, 0.5, 1, 2],
        figsize: Optional[tuple[float, float]] = None,
        style: Optional[str] = None,
        bar_color_long: Optional[str] = None,
        bar_color_short: Optional[str] = None,
        save_path_prefix: Optional[str] = None,
    ) -> None:
        """For each column_to_bin and each signal_base, creates bar plots of the
        annualized Sharpe ratio of the signal's PnL, bucketed by fixed bins of the
        column_to_bin. Plots are generated separately for long and short positions.
        The x-axis shows the fixed bin ranges."""

        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for fixed bin analysis is empty.")
            return
        if not columns_to_bin:
            print("Analyzer Warning: No columns_to_bin provided.")
            return
        if not signal_bases:
            print("Analyzer Warning: No signal_bases provided.")
            return
        if not fixed_breaks or len(fixed_breaks) < 1:
            print(
                "Analyzer Warning: fixed_breaks must contain at least one break point."
            )
            return

        sorted_breaks = sorted(list(set(fixed_breaks)))
        cut_breaks_for_polars = sorted_breaks

        bin_labels = [f"<= {sorted_breaks[0]:.2f}"]
        for i in range(len(sorted_breaks) - 1):
            bin_labels.append(f"({sorted_breaks[i]:.2f}, {sorted_breaks[i+1]:.2f}]")
        bin_labels.append(f"> {sorted_breaks[-1]:.2f}")

        if len(bin_labels) != (len(sorted_breaks) + 1):
            print(
                f"Analyzer Error: Mismatch in generated bin labels ({len(bin_labels)}) and expected bins ({len(sorted_breaks)+1}). Check breaks."
            )
            return

        plot_style = (
            style
            if style is not None
            else getattr(self.config, "DEFAULT_PLOT_STYLE", "seaborn-v0_8-darkgrid")
        )
        plot_figsize = (
            figsize
            if figsize is not None
            else getattr(self.config, "DEFAULT_FIGSIZE", (12, 7))
        )
        ann_factor = getattr(self.config, "DEFAULT_ANNUALIZATION_FACTOR", 252)
        rf_daily = getattr(self.config, "DEFAULT_RISK_FREE_RATE_DAILY", 0.0)
        # Use provided bar colors, or fall back to config, or then to hardcoded defaults
        default_bar_color_long = (
            bar_color_long
            if bar_color_long is not None
            else getattr(self.config, "DEFAULT_DECILE_BAR_COLOR_LONG", "mediumseagreen")
        )
        default_bar_color_short = (
            bar_color_short
            if bar_color_short is not None
            else getattr(self.config, "DEFAULT_DECILE_BAR_COLOR_SHORT", "lightcoral")
        )

        try:
            sns.set_theme(style=plot_style)
        except Exception as e:
            print(
                f"Analyzer Warning: Could not set seaborn style '{plot_style}'. Error: {e}"
            )

        for bin_col_name in columns_to_bin:
            if bin_col_name not in df.columns:
                print(
                    f"Analyzer Warning: Column '{bin_col_name}' for binning not found. Skipping."
                )
                continue
            if df[bin_col_name].drop_nulls().is_empty():
                print(
                    f"Analyzer Warning: Column '{bin_col_name}' for binning is all nulls or empty. Skipping."
                )
                continue

            for base_signal_name in signal_bases:
                position_col = f"{base_signal_name}_pos"
                daily_pnl_col = f"{base_signal_name}_pnl"
                if not all(c in df.columns for c in [position_col, daily_pnl_col]):
                    print(
                        f"Analyzer Warning: Missing PnL/Pos columns for signal '{base_signal_name}' with bin '{bin_col_name}'. Skipping."
                    )
                    continue

                for position_type in ["long", "short"]:
                    current_bar_color = (
                        default_bar_color_long
                        if position_type == "long"
                        else default_bar_color_short
                    )

                    filtered_df_by_pos = (
                        df.filter(pl.col(position_col) > 0)
                        if position_type == "long"
                        else df.filter(pl.col(position_col) < 0)
                    )
                    title_suffix = f"{position_type.capitalize()} Positions (Signal: {base_signal_name})"

                    if (
                        filtered_df_by_pos.is_empty()
                        or filtered_df_by_pos[bin_col_name].drop_nulls().is_empty()
                    ):
                        print(
                            f"Analyzer Info: No {position_type} positions with valid '{bin_col_name}' data for signal '{base_signal_name}'. Skipping."
                        )
                        continue

                    try:
                        df_with_bins = filtered_df_by_pos.with_columns(
                            pl.col(bin_col_name)
                            .cut(breaks=cut_breaks_for_polars, labels=bin_labels)
                            .alias(f"{bin_col_name}_fixed_bin_label")
                        )
                    except Exception as e:
                        print(
                            f"Analyzer Error: Could not create fixed bins for '{bin_col_name}' ({base_signal_name} - {position_type}). Error: {e}. Skipping."
                        )
                        continue

                    plot_data_list = []
                    for i, bin_label_val in enumerate(bin_labels):
                        current_bin_data = df_with_bins.filter(
                            pl.col(f"{bin_col_name}_fixed_bin_label") == bin_label_val
                        )
                        sharpe = np.nan
                        if not current_bin_data.is_empty():
                            pnl_series_for_bin = current_bin_data[
                                daily_pnl_col
                            ].drop_nulls()
                            if pnl_series_for_bin.len() >= 2:
                                mean_pnl, std_pnl = (
                                    pnl_series_for_bin.mean(),
                                    pnl_series_for_bin.std(),
                                )
                                if (
                                    std_pnl is not None
                                    and std_pnl != 0
                                    and mean_pnl is not None
                                ):
                                    sharpe = ((mean_pnl - rf_daily) / std_pnl) * (
                                        ann_factor**0.5
                                    )
                                elif (
                                    mean_pnl is not None
                                    and mean_pnl == 0
                                    and (std_pnl is None or std_pnl == 0)
                                ):
                                    sharpe = 0.0
                        plot_data_list.append(
                            {
                                "BinAxisLabel": bin_label_val,
                                "SharpeRatio": sharpe,
                                "_sort_key": i,
                            }
                        )

                    if not plot_data_list:
                        print(
                            f"Analyzer Info: No bin data to plot for '{bin_col_name}' ({base_signal_name} - {title_suffix})."
                        )
                        continue
                    plot_display_df = pl.DataFrame(plot_data_list).sort("_sort_key")
                    plot_display_df_cleaned = plot_display_df.filter(
                        pl.col("SharpeRatio").is_not_nan()
                    )
                    if plot_display_df_cleaned.is_empty():
                        print(
                            f"Analyzer Info: No non-NaN Sharpe ratios for '{bin_col_name}' fixed bins ({base_signal_name} - {title_suffix})."
                        )
                        continue

                    plt.figure(figsize=plot_figsize)
                    sns.barplot(
                        x="BinAxisLabel",
                        y="SharpeRatio",
                        data=plot_display_df_cleaned.to_pandas(),
                        color=current_bar_color,
                        errorbar=None,
                    )
                    plot_title = f"Sharpe Ratio of '{base_signal_name}' PnL\nby Fixed Bins of '{bin_col_name.replace('_',' ').title()}' ({title_suffix})"
                    plt.title(
                        plot_title,
                        fontsize=getattr(self.config, "DEFAULT_TITLE_FONTSIZE", 16),
                        pad=15,
                    )
                    plt.xlabel(
                        f"Fixed Bin Range of {bin_col_name.replace('_',' ').title()}",
                        fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                        labelpad=10,
                    )
                    plt.ylabel(
                        "Annualized Sharpe Ratio",
                        fontsize=getattr(self.config, "DEFAULT_LABEL_FONTSIZE", 12),
                        labelpad=10,
                    )
                    ax = plt.gca()
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
                    y_formatter = mticker.FormatStrFormatter("%.2f")
                    ax.yaxis.set_major_formatter(y_formatter)
                    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    if save_path_prefix:
                        if os.path.dirname(save_path_prefix) and not os.path.exists(
                            os.path.dirname(save_path_prefix)
                        ):
                            os.makedirs(
                                os.path.dirname(save_path_prefix), exist_ok=True
                            )
                        filename_bin_col = bin_col_name.replace("/", "_")
                        current_save_path = f"{save_path_prefix}sharpe_by_{filename_bin_col}_fixedbins_for_{base_signal_name}_{position_type}.png"
                        try:
                            plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                            print(f"Plot saved to {current_save_path}")
                        except Exception as e:
                            print(
                                f"Analyzer Error saving plot to {current_save_path}: {e}"
                            )
                    plt.show()

    def perform_full_analysis(
        self,
        df: pl.DataFrame,
        stats_cols: Optional[List[str]] = None,
        corr_cols: Optional[List[str]] = None,
        corr_method: str = "pearson",
        pnl_stat_cols: Optional[List[str]] = None,
        timeseries_plot_configs: Optional[List[Dict[str, Any]]] = None,
        histogram_plot_configs: Optional[List[Dict[str, Any]]] = None,
        pnl_bar_plot_configs: Optional[List[Dict[str, Any]]] = None,
        signal_decile_sharpe_configs: Optional[List[Dict[str, Any]]] = None,
        column_decile_signal_sharpe_configs: Optional[List[Dict[str, Any]]] = None,
        column_fixed_bin_signal_sharpe_configs: Optional[
            List[Dict[str, Any]]
        ] = None,  # New
        acf_plot_configs: Optional[List[Dict[str, Any]]] = None,
        save_plots_path_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis including descriptive statistics, correlations,
        PnL statistics, and generates various plots including signal ACF plots,
        Sharpe ratios by deciles of specified columns, and Sharpe ratios by fixed bins.
        """
        analysis_results: Dict[str, Any] = {}

        if not isinstance(df, pl.DataFrame):
            print("Analyzer Error: Input 'df' must be a Polars DataFrame.")
            return analysis_results
        if df.is_empty():
            print("Analyzer Warning: Input DataFrame for full analysis is empty.")
            analysis_results["descriptive_stats"] = pl.DataFrame()
            analysis_results["correlation_matrix"] = pl.DataFrame()
            analysis_results["pnl_statistics"] = pl.DataFrame()
            return analysis_results

        # --- Descriptive Statistics ---
        if stats_cols is not None:
            print("Analyzer: Performing descriptive statistics...")
            analysis_results["descriptive_stats"] = self.get_descriptive_stats(
                df,
                columns_to_analyze=stats_cols,
                percentiles=self.config.DEFAULT_PERCENTILES,
            )
        else:
            analysis_results["descriptive_stats"] = pl.DataFrame()

        # --- Correlation Analysis ---
        if corr_cols is not None:
            print("Analyzer: Performing correlation analysis...")
            analysis_results["correlation_matrix"] = self.get_correlation_matrix(
                df, columns_to_correlate=corr_cols, method=corr_method
            )
        else:
            analysis_results["correlation_matrix"] = pl.DataFrame()

        # --- PnL Statistics ---
        if pnl_stat_cols:
            print("Analyzer: Calculating PnL statistics...")
            pnl_stats_list = []
            for pnl_c in pnl_stat_cols:
                if pnl_c in df.columns:
                    stats_df = self.calculate_pnl_statistics(df, pnl_col_name=pnl_c)
                    pnl_stats_list.append(
                        stats_df.with_columns(pl.lit(pnl_c).alias("PnL Source"))
                    )
                else:
                    print(
                        f"Analyzer Warning: PnL column '{pnl_c}' for stats not found in DataFrame."
                    )
            if pnl_stats_list:
                analysis_results["pnl_statistics"] = (
                    pl.concat(pnl_stats_list)
                    if len(pnl_stats_list) > 1
                    else pnl_stats_list[0]
                )
            else:
                analysis_results["pnl_statistics"] = pl.DataFrame()
        else:
            analysis_results["pnl_statistics"] = pl.DataFrame()

        # --- Plotting ---
        if (
            save_plots_path_prefix
            and os.path.dirname(save_plots_path_prefix)
            and not os.path.exists(os.path.dirname(save_plots_path_prefix))
        ):
            os.makedirs(os.path.dirname(save_plots_path_prefix), exist_ok=True)

        if timeseries_plot_configs:
            print("Analyzer: Generating timeseries plots...")
            for i, conf in enumerate(timeseries_plot_configs):
                save_path = None
                value_col_for_filename = conf.get("value_col", f"plot{i}")
                if not isinstance(value_col_for_filename, str):
                    value_col_for_filename = f"plot{i}"
                if save_plots_path_prefix:
                    save_path = f"{save_plots_path_prefix}timeseries_{value_col_for_filename}.png"
                plot_conf = conf.copy()
                plot_conf["save_path"] = save_path
                self.plot_timeseries(df, **plot_conf)

        if histogram_plot_configs:
            print("Analyzer: Generating histogram/KDE plots...")
            for i, conf in enumerate(histogram_plot_configs):
                save_path = None
                value_col_for_filename = conf.get("value_col", f"hist{i}")
                if not isinstance(value_col_for_filename, str):
                    value_col_for_filename = f"hist{i}"
                if save_plots_path_prefix:
                    save_path = f"{save_plots_path_prefix}histogram_{value_col_for_filename}.png"
                plot_conf = conf.copy()
                plot_conf["save_path"] = save_path
                self.plot_histogram_kde(df, **plot_conf)

        if pnl_bar_plot_configs:
            print("Analyzer: Generating PnL bar plots...")
            for i, conf in enumerate(pnl_bar_plot_configs):
                save_path_base = None
                pnl_col_for_filename = conf.get("pnl_col", f"pnl{i}")
                if not isinstance(pnl_col_for_filename, str):
                    pnl_col_for_filename = f"pnl{i}"
                if save_plots_path_prefix:
                    save_path_base = (
                        f"{save_plots_path_prefix}pnl_bars_{pnl_col_for_filename}"
                    )
                plot_conf = conf.copy()
                plot_conf["save_path_base"] = save_path_base
                self.plot_pnl_bars(df, **plot_conf)

        if signal_decile_sharpe_configs:
            print("Analyzer: Generating signal decile Sharpe bar charts...")
            for conf in signal_decile_sharpe_configs:
                plot_conf = conf.copy()
                current_save_prefix = save_plots_path_prefix
                if "save_path_prefix" in plot_conf:
                    current_save_prefix = plot_conf.pop("save_path_prefix")

                if hasattr(self, "plot_signal_decile_sharpe_barchart"):
                    self.plot_signal_decile_sharpe_barchart(
                        df, save_path_prefix=current_save_prefix, **plot_conf
                    )
                else:
                    if hasattr(
                        self, "plot_signal_decile_sharpe_histograms"
                    ):  # Fallback if you still have the old histogram version
                        self.plot_signal_decile_sharpe_histograms(
                            df, save_path_prefix=current_save_prefix, **plot_conf
                        )
                    else:
                        print(
                            "Analyzer Warning: Method for signal decile Sharpe plotting (barchart or histogram) not found."
                        )

        if column_decile_signal_sharpe_configs:
            print("Analyzer: Generating column decile signal Sharpe bar charts...")
            for conf in column_decile_signal_sharpe_configs:
                plot_conf = conf.copy()
                current_save_prefix = save_plots_path_prefix
                if "save_path_prefix" in plot_conf:
                    current_save_prefix = plot_conf.pop("save_path_prefix")

                self.plot_column_decile_signal_sharpe_barchart(
                    df, save_path_prefix=current_save_prefix, **plot_conf
                )

        if column_fixed_bin_signal_sharpe_configs:  # New plot type
            print("Analyzer: Generating column fixed-bin signal Sharpe bar charts...")
            for conf in column_fixed_bin_signal_sharpe_configs:
                plot_conf = conf.copy()
                current_save_prefix = save_plots_path_prefix
                if "save_path_prefix" in plot_conf:
                    current_save_prefix = plot_conf.pop("save_path_prefix")
                self.plot_column_fixed_bin_signal_sharpe_barchart(
                    df, save_path_prefix=current_save_prefix, **plot_conf
                )

        if acf_plot_configs:
            print("Analyzer: Generating signal autocorrelation plots...")
            for conf in acf_plot_configs:
                plot_conf = conf.copy()
                current_save_prefix = save_plots_path_prefix
                if "save_path_prefix" in plot_conf:
                    current_save_prefix = plot_conf.pop("save_path_prefix")

                self.plot_signal_autocorrelation(
                    df, save_path_prefix=current_save_prefix, **plot_conf
                )

        print("Analyzer: Full analysis complete.")
        return analysis_results
