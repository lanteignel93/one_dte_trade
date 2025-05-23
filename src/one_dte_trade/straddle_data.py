import polars as pl
from intake import Catalog  # For type hinting self.cat

# Import the global/default cat loader
from . import cat as global_cat_loader
from one_dte_trade import (
    clock,
    logger,
)

# Import the pre-instantiated straddle_config object.
# This assumes that the __init__.py of the parent package (e.g., 'one_dte_trade')
# has loaded the configurations and made 'straddle_config' available for import.
# If this script (straddle_data.py) is in the same directory as that __init__.py:
from one_dte_trade import (
    straddle_config as pre_loaded_straddle_config,
)

# Import the StraddleDataConfig class definition for type hinting self.config
# This assumes 'config.py' (where StraddleDataConfig is defined) is in the same
# directory as this script (e.g., both are in 'one_dte_trade').
from one_dte_trade.config import (
    StraddleDataConfig,  # For type hinting self.straddle_config
)

# If this script is one level deeper, it might be: from .. import straddle_config


class StraddleDataPipeline:
    """
    Handles fetching, processing, and structuring straddle option data.
    It uses pre-loaded configurations made available at the package level.
    """

    def __init__(self):
        """
        Initializes the pipeline.
        It uses pre-loaded configuration objects (straddle_config)
        and the global 'cat' loader.
        """
        # Use the pre-loaded straddle_config from the package
        self.config: StraddleDataConfig = pre_loaded_straddle_config

        # Use the globally imported cat loader
        self.cat: Catalog = global_cat_loader

        if self.config is None:
            print(
                "StraddleDataPipeline Warning: Pre-loaded straddle_config is None. Using default StraddleDataConfig()."
            )
            # This fallback might be hit if the package __init__.py fails to load configs
            # or if this module is imported before __init__.py fully initializes them.
            self.config = StraddleDataConfig()

        print(
            f"StraddleDataPipeline initialized with TICKER_ID: {self.config.TICKER_ID}"
        )

    def _get_base_processed_lazyframe(self) -> pl.LazyFrame:
        """
        Fetches raw option and security data, performs initial join and processing.
        This method encapsulates the common data retrieval and preparation steps:
        - Filters by security ID (TICKER_ID) and minimum date.
        - Selects relevant columns.
        - Calculates mid-price ('sprc') for options.
        - Joins option data with security closing prices.
        - Drops intermediate price columns.
        - Renames columns to a consistent format.
        """
        option_prices_lazy = (
            self.cat.om_int_option_price()
            .lazy()
            .filter(pl.col("securityid").eq(self.config.TICKER_ID))
            .select(["callput", "strike", "date", "bestbid", "bestoffer", "expiration"])
            .filter(
                pl.col("date").ge(self.config.min_date)
            )  # Uses min_date property from config
            .with_columns(sprc=(pl.col("bestbid") + pl.col("bestoffer")) / 2)
        )

        security_prices_lazy = (
            self.cat.om_int_security_price()
            .lazy()
            .filter(pl.col("securityid").eq(self.config.TICKER_ID))
            .select(["date", "closeprice"])
        )

        joined_lazy = option_prices_lazy.join(
            security_prices_lazy, on="date", how="inner"
        )

        processed_lazy = joined_lazy.drop(["bestbid", "bestoffer"]).rename(
            {
                "callput": "okey_cp",
                "strike": "okey_xx",
                "closeprice": "uprc",
            }
        )
        return processed_lazy

    def _calculate_time_to_expiration(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Adds the 'time_to_expiration' column in total days."""
        return lf.with_columns(
            time_to_expiration=(pl.col("expiration") - pl.col("date")).dt.total_days()
        )

    @clock()
    @logger
    def get_open_straddle_data(self) -> pl.DataFrame:
        """
        Processes and returns data for opening straddles.
        """
        base_lf = self._get_base_processed_lazyframe()
        lf_with_tte = self._calculate_time_to_expiration(base_lf)

        open_lf = (
            lf_with_tte.with_columns(
                (pl.col("okey_xx") * pl.lit(1 / 1000)).alias("okey_xx")
            )
            .with_columns(dist_to_strike=(pl.col("okey_xx") - pl.col("uprc")).abs())
            .filter(pl.col("time_to_expiration").gt(0))
            .sort(["date", "dist_to_strike", "time_to_expiration"])
            .group_by(["date", "okey_cp"], maintain_order=True)
            .first()
        )
        return open_lf.collect()

    @clock()
    @logger
    def get_close_straddle_data(self) -> pl.DataFrame:
        """
        Processes and returns data relevant for closing straddles,
        specifically targeting options with less than 1 day to expiration.
        """
        base_lf = self._get_base_processed_lazyframe()
        lf_with_tte = self._calculate_time_to_expiration(base_lf)

        # Logic for closing straddles: typically options expiring very soon (TTE < 1)
        close_lf = lf_with_tte.with_columns(
            (pl.col("okey_xx") * pl.lit(1 / 1000)).alias("okey_xx")
        ).filter(pl.col("time_to_expiration").lt(1))
        return close_lf.collect()

    @logger
    def merge_open_trade_close_trade(
        self, open_df: pl.DataFrame, close_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Merges open and close straddle data.
        It joins open positions with their corresponding closing data, assuming
        closing occurs on the expiration date of the open position.
        """
        if open_df.is_empty():
            print(
                "StraddleDataPipeline Warning: open_df is empty in merge_open_trade_close_trade. Returning empty DataFrame."
            )
            # Return an empty DataFrame with expected schema if possible, or just empty.
            # For simplicity, returning empty. Downstream should handle.
            return pl.DataFrame()
        # close_df can be empty if no closing trades match, join will handle this.

        merged_df = open_df.join(
            close_df,
            left_on=["okey_cp", "okey_xx", "expiration"],
            right_on=["okey_cp", "okey_xx", "date"],
            how="left",
            suffix="_right",
        )

        columns_to_drop = ["time_to_expiration_right", "expiration_right"]
        if "time_to_expiration" in merged_df.columns:  # From open_df
            columns_to_drop.append("time_to_expiration")

        # Ensure columns to drop actually exist to avoid errors
        actual_columns_to_drop = [
            col for col in columns_to_drop if col in merged_df.columns
        ]

        processed_df = merged_df
        if actual_columns_to_drop:
            processed_df = merged_df.drop(actual_columns_to_drop)

        processed_df = (
            processed_df.rename(
                {
                    "uprc": "uprc_day_0",
                    "sprc": "sprc_day_0",
                    "uprc_right": "uprc_day_1",
                    "sprc_right": "sprc_day_1",
                }
            )
            .unique()
            .sort(["date", "okey_cp"])
            .filter(~pl.col("date").is_in(self.config.DATE_PROBLEMS))
            # The filter ~pl.any_horizontal(pl.all().is_null()) is very strict.
            # It removes rows if ANY column is null, which means rows from open_df
            # that didn't find a match in close_df (and thus have nulls for _day_1 columns)
            # will be dropped. This makes the left join behave like an inner join in terms of row retention.
            # If you want to keep all open trades, comment out or adjust this filter.
            .filter(~pl.any_horizontal(pl.all().is_null()))
        )
        return processed_df

    @logger
    def pivot_data(self, merged_df: pl.DataFrame) -> pl.DataFrame:
        """
        Pivots the merged data to have separate columns for Call and Put option prices.
        """
        if merged_df.is_empty():
            print(
                "StraddleDataPipeline Warning: merged_df is empty in pivot_data. Returning empty DataFrame."
            )
            return pl.DataFrame()

        index_cols_base = [
            "date",
            "okey_xx",
            "uprc_day_0",
            "expiration",
            "dist_to_strike",
        ]
        # Add uprc_day_1 to index if it exists and is not all nulls (due to the strict filter in merge, it should be non-null if rows exist)
        if (
            "uprc_day_1" in merged_df.columns
            and not merged_df["uprc_day_1"].is_null().all()
        ):
            index_cols_base.append("uprc_day_1")

        actual_index_cols = [col for col in index_cols_base if col in merged_df.columns]

        # Check if 'okey_cp' exists and has expected values for pivoting
        if (
            "okey_cp" not in merged_df.columns
            or merged_df["okey_cp"].drop_nulls().n_unique() < 1
        ):
            print(
                "StraddleDataPipeline Warning: 'okey_cp' column missing or has no distinct values for pivoting. Returning unpivoted data."
            )
            return merged_df  # Or an empty DF with expected pivoted schema

        pivoted_df = merged_df.pivot(
            values=["sprc_day_0", "sprc_day_1"],
            index=actual_index_cols,
            columns="okey_cp",  # Use 'columns' for clarity
            aggregate_function=None,  # Assumes unique combinations
            separator="_",
        )

        # This filter also makes the result very strict, keeping only rows where ALL cells are non-null.
        return pivoted_df.filter(~pl.any_horizontal(pl.all().is_null()))

    @logger
    def calculate_straddle_prices(self, pivoted_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates straddle prices (sum of call and put) for day_0 and day_1,
        and the absolute difference between underlying price at close and strike.
        """
        if pivoted_df.is_empty():
            print(
                "StraddleDataPipeline Warning: pivoted_df is empty in calculate_straddle_prices. Returning empty DataFrame."
            )
            return pl.DataFrame()

        sprc_d0_call = "sprc_day_0_C"
        sprc_d0_put = "sprc_day_0_P"
        sprc_d1_call = "sprc_day_1_C"
        sprc_d1_put = "sprc_day_1_P"

        # Check if all required pivoted columns exist
        required_pivot_cols = [
            sprc_d0_call,
            sprc_d0_put,
            sprc_d1_call,
            sprc_d1_put,
            "uprc_day_1",
            "okey_xx",
        ]
        missing_pivot_cols = [
            col for col in required_pivot_cols if col not in pivoted_df.columns
        ]

        if missing_pivot_cols:
            # If strict filtering in pivot_data was applied, these columns should exist if pivoted_df is not empty.
            # However, if pivot_data returned early or filters were changed, this check is useful.
            print(
                f"StraddleDataPipeline Warning: Pivoted DataFrame is missing required columns for straddle price calculation: {missing_pivot_cols}. Returning input DataFrame."
            )
            return pivoted_df  # Or return empty with expected schema

        df_with_straddle_prices = pivoted_df.with_columns(
            (pl.col(sprc_d0_call) + pl.col(sprc_d0_put)).alias("straddle_day_0"),
            (pl.col(sprc_d1_call) + pl.col(sprc_d1_put)).alias("straddle_day_1"),
            ((pl.col("uprc_day_1") - pl.col("okey_xx")).abs()).alias(
                "uprc_diff"
            ),  # uprc_diff is now absolute
        )
        return df_with_straddle_prices
