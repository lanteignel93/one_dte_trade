import polars as pl

# --- Import your main pipeline/component classes ---
from . import cat  # Your global cat loader
from one_dte_trade import (
    clock,
    feature_builder_config,
    logger,
    # Other configs like signal_generator_config, backtester_config, analyzer_config
    # would be imported here if OneDTEDataPipeline directly used them.
    straddle_config,
)

# Also import the dataclass types if needed for type hinting within this file,
# though they are primarily used by the components being instantiated.
from one_dte_trade.config import (
    FeatureBuilderConfig,  # For type hinting self.feature_builder_config
    StraddleDataConfig,  # For type hinting self.straddle_config
)
from one_dte_trade.features import (
    FeatureBuilder,
)
from one_dte_trade.straddle_data import (
    StraddleDataPipeline,
)
from one_dte_trade.vol_indicies import (
    VolatilityIndicesFetcher,
)


class OneDTEDataPipeline:
    """
    Orchestrates the generation of a combined dataset focused on 1DTE (One Day To Expiration)
    scenarios, including processed straddle option data, volatility index prices,
    and optionally, engineered features.
    """

    def __init__(self):
        """
        Initializes the 1DTE data pipeline orchestrator.
        It uses pre-loaded configuration objects imported from the package level
        (expected to be loaded by the package's __init__.py from project_settings.json).
        """
        print("OneDTEDataPipeline: Initializing with pre-loaded configurations...")

        # Store the imported, pre-loaded configs
        self.straddle_config: StraddleDataConfig = straddle_config
        self.feature_builder_config: FeatureBuilderConfig = feature_builder_config
        self.cat_loader = cat  # Use the globally imported cat

        # Instantiate internal pipeline components
        self.straddle_pipeline = StraddleDataPipeline()
        self.vol_fetcher = VolatilityIndicesFetcher()

        # Instantiate FeatureBuilder here, as it's now a consistent part
        self.feature_builder = FeatureBuilder()
        print("OneDTEDataPipeline: Components initialized.")

    @clock()
    @logger
    def get_1dte_data(
        self,
        generate_features: bool = True,  # Controls whether features are generated
    ) -> pl.DataFrame:
        """
        Generates a combined DataFrame focused on 1DTE scenarios, optionally
        enriching it with features like volatility Z-scores and SPX returns.

        The process involves:
        1. Running the full StraddleDataPipeline to get processed straddle data.
        2. Running the VolatilityIndicesFetcher to get base volatility data.
        3. If `generate_features` is True:
            a. Calculating Z-scores for specified volatility columns.
            b. Calculating SPX returns using the internal FeatureBuilder instance.
        4. Performing left joins to attach the (potentially feature-enhanced) volatility
           data and SPX returns (if calculated) to the straddle data, matching on 'date'.

        Args:
            generate_features: If True, volatility Z-scores and SPX returns will be
                               calculated and included. Defaults to True.

        Returns:
            A Polars DataFrame containing the combined and potentially feature-enriched
            data, sorted by date.
        """
        print("Starting 1DTE data generation...")

        # Step 1: Process Straddle Data
        print("  Processing straddle data for 1DTE context...")
        open_data = self.straddle_pipeline.get_open_straddle_data()
        print(f"    Open data rows: {len(open_data)}")
        close_data = self.straddle_pipeline.get_close_straddle_data()
        print(f"    Close data (TTE < 1) rows: {len(close_data)}")

        if open_data.is_empty():
            print(
                "  Warning: Open straddle data is empty. Resulting 1DTE data will lack opening leg info."
            )

        merged_straddle_data = self.straddle_pipeline.merge_open_trade_close_trade(
            open_data, close_data
        )
        print(f"    Merged straddle data rows: {len(merged_straddle_data)}")

        pivoted_straddle_data = self.straddle_pipeline.pivot_data(merged_straddle_data)
        print(f"    Pivoted straddle data rows: {len(pivoted_straddle_data)}")

        final_straddle_df = self.straddle_pipeline.calculate_straddle_prices(
            pivoted_straddle_data
        )
        print(f"    Final straddle data with prices rows: {len(final_straddle_df)}")

        # Step 2: Fetch Volatility Data
        print("  Fetching volatility index data...")
        volatility_df = self.vol_fetcher.get_all_volatility_prices()  # Includes VIX_TS
        print(f"    Volatility index data rows: {len(volatility_df)}")

        spx_ret_df = None  # Initialize to None
        if generate_features:
            print("  Applying features using internal FeatureBuilder...")
            if not volatility_df.is_empty():
                volatility_df = self.feature_builder.calculate_vol_zscore(volatility_df)
                print(f"    Volatility data with Z-scores rows: {len(volatility_df)}")
            else:
                print("    Skipping Z-score calculation as volatility_df is empty.")

            # Calculate SPX returns using the stored straddle_config
            spx_ret_df = self.feature_builder.calculate_spx_ret(self.straddle_config)
            print(
                f"    SPX returns calculated, rows: {len(spx_ret_df if spx_ret_df is not None else [])}"
            )
        else:
            print("  Feature generation skipped.")

        # Step 3: Combine Data
        print("  Combining straddle, volatility, and SPX returns data...")
        if final_straddle_df.is_empty():
            print(
                "  Info: Processed straddle data is empty for 1DTE context. Combined DataFrame will be empty or only contain vol/SPX data if they exist."
            )
            # If final_straddle_df is empty, the left join will result in an empty df
            # or a df with only dates if other DFs have dates not in final_straddle_df (unlikely with left join).
            # To ensure a predictable schema even when empty, one might construct an empty DF with all expected columns.
            # For now, relying on join behavior.
            if volatility_df.is_empty() and (
                spx_ret_df is None or spx_ret_df.is_empty()
            ):
                return pl.DataFrame()  # Return truly empty if all inputs are empty

        combined_df = final_straddle_df

        if not volatility_df.is_empty():
            combined_df = combined_df.join(volatility_df, on="date", how="left")

        if spx_ret_df is not None and not spx_ret_df.is_empty():
            combined_df = combined_df.join(spx_ret_df, on="date", how="left")

        print(f"    Combined 1DTE data generated with rows: {len(combined_df)}")

        return combined_df.sort("date", nulls_last=True)


if __name__ == "__main__":
    feature_builder = FeatureBuilder()
    pipeline = OneDTEDataPipeline()
    df = pipeline.get_1dte_data()
    df.write_parquet("/htaa/llanteigne/1dte.parquet")
