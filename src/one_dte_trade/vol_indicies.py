from typing import Any, Dict, List  # Added Optional for clarity

import polars as pl
from intake import Catalog  # For type hinting self.cat

# Import the global/default cat loader
from . import cat as global_cat_loader
from one_dte_trade import (
    clock,
    logger,
)

# If FeatureBuilderConfig or another config object were to hold VIX_SECURITY_ID etc.,
# you would import the pre-loaded config instance here.
# For example:
# from . import feature_builder_config # If it contained VOL_COLS and their IDs


# --- Volatility Data Class ---
class VolatilityIndicesFetcher:
    """
    Fetches and consolidates daily prices for a predefined set of volatility-related securities.
    Prices are calculated as (bidlow + askhigh) / 2.
    It uses a globally available 'cat' loader.
    """

    # These are defined as class attributes. If they need to be configurable via JSON,
    # they should ideally be part of a config dataclass (e.g., FeatureBuilderConfig or a new one)
    # which would then be imported as a pre-loaded instance.
    # For this modification, we'll keep them as class attributes as per your provided script.
    VIX_SECURITY_ID_TUPLE: tuple[int, str] = (
        117801,
        "VIX",
    )  # Renamed to avoid conflict if class name is VIX_SECURITY_ID
    VVIX_SECURITY_ID_TUPLE: tuple[int, str] = (152892, "VVIX")
    VIXM_SECURITY_ID_TUPLE: tuple[int, str] = (145764, "VIXM")

    # This list is constructed from the class attributes above.
    security_configs: List[Dict[str, Any]] = [
        {"id": VIX_SECURITY_ID_TUPLE[0], "name": VIX_SECURITY_ID_TUPLE[1]},
        {"id": VVIX_SECURITY_ID_TUPLE[0], "name": VVIX_SECURITY_ID_TUPLE[1]},
        {"id": VIXM_SECURITY_ID_TUPLE[0], "name": VIXM_SECURITY_ID_TUPLE[1]},
    ]

    def __init__(self):
        """
        Initializes the fetcher.
        It uses the globally imported 'cat' loader from 'onepipeline.conf'.
        """
        self.cat: Catalog = global_cat_loader
        # If security_configs were instance-based from a config object:
        # self.security_configs = imported_volatility_config.SECURITY_DEFINITIONS
        print("VolatilityIndicesFetcher initialized.")

    @clock()
    @logger
    def _fetch_price_for_security(
        self, sec_id: int, price_col_name: str
    ) -> pl.DataFrame:
        """
        Fetches and calculates the daily price for a single security.
        The price is the midpoint of 'bidlow' and 'askhigh'.

        Args:
            sec_id: The security ID to fetch data for.
            price_col_name: The name to use for the calculated price column (e.g., "VIX").

        Returns:
            A Polars DataFrame with 'date' and the named price column.
            Returns an empty DataFrame with the correct schema if data cannot be fetched
            or if the price cannot be calculated.
        """
        try:
            source_data_lazy = self.cat.om_int_security_price().lazy()

            price_df = (
                source_data_lazy.filter(pl.col("securityid").eq(sec_id))
                .with_columns(
                    ((pl.col("bidlow") + pl.col("askhigh")) / 2).alias(price_col_name)
                )
                .select(["date", price_col_name])
                .collect()
                .drop_nulls(subset=[price_col_name])
            )
            return price_df
        except Exception as e:
            print(
                f"VolatilityIndicesFetcher Warning: Could not fetch/process data for sec_id {sec_id} (column: {price_col_name}). Error: {e}"
            )
            return pl.DataFrame(
                {
                    "date": pl.Series(dtype=pl.Date),
                    price_col_name: pl.Series(dtype=pl.Float64),
                }
            )

    @logger
    def calculate_vol_ts(self, merged_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the VIX term structure ratio (VIXM / VIX) and adds it as a new column 'VIX_TS'.
        Handles potential division by zero or nulls gracefully.
        """
        if "VIX" not in merged_df.columns or "VIXM" not in merged_df.columns:
            print(
                "VolatilityIndicesFetcher Warning: 'VIX' or 'VIXM' column not found in merged_df for VIX_TS calculation. Returning DataFrame as is."
            )
            # Add an empty VIX_TS column for schema consistency if desired
            if "VIX_TS" not in merged_df.columns:
                return merged_df.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("VIX_TS")
                )
            return merged_df

        return merged_df.with_columns(
            pl.when((pl.col("VIX").is_not_null()) & (pl.col("VIX") != 0))
            .then(pl.col("VIXM") / pl.col("VIX"))
            .otherwise(None)
            .alias("VIX_TS")
        )

    @clock()
    @logger
    def get_all_volatility_prices(self) -> pl.DataFrame:
        """
        Fetches daily prices for VIX, VVIX, and VIXM, merges them,
        and calculates the VIXM/VIX term structure ratio ("VIX_TS").

        Returns:
            A Polars DataFrame with a 'date' column, columns for each
            security's price (e.g., 'VIX', 'VVIX', 'VIXM'), and 'VIX_TS'.
        """
        list_of_individual_dfs: List[pl.DataFrame] = []
        # Uses the class attribute self.security_configs
        for config_item in self.security_configs:
            df_sec = self._fetch_price_for_security(
                config_item["id"], config_item["name"]
            )
            if not df_sec.is_empty():
                list_of_individual_dfs.append(df_sec)

        if not list_of_individual_dfs:
            schema_dict = {"date": pl.Date}
            for config_item in self.security_configs:
                schema_dict[config_item["name"]] = pl.Float64
            # Add VIX_TS to schema for empty result consistency
            if any(sc["name"] == "VIX" for sc in self.security_configs) and any(
                sc["name"] == "VIXM" for sc in self.security_configs
            ):
                schema_dict["VIX_TS"] = pl.Float64
            return pl.DataFrame(schema=schema_dict)

        merged_df = list_of_individual_dfs[0]
        for i in range(1, len(list_of_individual_dfs)):
            merged_df = merged_df.join(
                list_of_individual_dfs[i], on="date", how="full", coalesce=True
            )

        # It's possible that after joins, there are multiple rows for the same date if the
        # underlying data source had duplicates for a security on a given date before _fetch_price_for_security's logic.
        # The .group_by("date").first() ensures one row per date.
        # However, _fetch_price_for_security usually implies one price per sec_id per date.
        # If duplicates are not expected, this group_by might hide issues.
        # For safety, keeping it, but it's worth understanding if it's strictly needed.
        # If _fetch_price_for_security guarantees uniqueness of date per security df,
        # and join keys are just 'date', then 'full' join might create rows with many nulls
        # if date ranges don't align perfectly. coalesce=True helps manage the date column.
        merged_df = merged_df.sort("date")  # Sort before potential group_by

        # Ensure all configured columns exist, filling with nulls if a security had no data
        for config_item in self.security_configs:
            if config_item["name"] not in merged_df.columns:
                merged_df = merged_df.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(config_item["name"])
                )

        # Calculate VIX_TS
        # The calculate_vol_ts method will add VIX_TS if VIX and VIXM are present
        merged_df = merged_df.group_by("date").first().sort("date")
        final_df = self.calculate_vol_ts(merged_df)

        return final_df.sort("date")  # Final sort
