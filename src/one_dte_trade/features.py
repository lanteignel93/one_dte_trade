import polars as pl
from intake import Catalog  # For type hinting self.cat
import numpy as np

# Import the global/default cat loader
from . import cat as global_cat_loader
from one_dte_trade import (
    clock,
    logger,
    requires_columns,
)

# Import the pre-instantiated feature_builder_config object.
# This assumes that the __init__.py of the parent package (e.g., 'one_dte_trade')
# has loaded the configurations and made 'feature_builder_config' available for import.
from one_dte_trade import (
    feature_builder_config as pre_loaded_feature_builder_config,
)

# Import the FeatureBuilderConfig and StraddleDataConfig class definitions for type hinting
# This assumes 'config.py' is in the same directory or accessible in the package.
from one_dte_trade.config import (
    FeatureBuilderConfig,
    StraddleDataConfig,
)


class FeatureBuilder:
    """
    Builds various financial features based on input data and configurations.
    This class can calculate:
    1. Z-scores of specified volatility columns based on their deviation from EWMA.
    2. Log returns for a specified ticker (e.g., SPX).
    It uses pre-loaded configurations.
    """

    def __init__(self):
        """
        Initializes the FeatureBuilder.
        It uses the pre-loaded 'feature_builder_config' and the global 'cat' loader.
        """
        self.config: FeatureBuilderConfig = pre_loaded_feature_builder_config

        if self.config is None:
            print(
                "FeatureBuilder Warning: Pre-loaded feature_builder_config is None. Using default FeatureBuilderConfig()."
            )
            self.config = FeatureBuilderConfig()  # Fallback

        # Validate and store configuration parameters from the pre-loaded config
        if not (isinstance(self.config.VOL_EWMA, int) and self.config.VOL_EWMA > 0):
            raise ValueError(
                "VOL_EWMA in FeatureBuilderConfig must be a positive integer."
            )
        self.vol_ewma_half_life = self.config.VOL_EWMA

        if not (
            isinstance(self.config.ZSCORE_TIME, int) and self.config.ZSCORE_TIME > 0
        ):
            raise ValueError(
                "ZSCORE_TIME in FeatureBuilderConfig must be a positive integer."
            )
        self.zscore_window = self.config.ZSCORE_TIME

        if not (
            isinstance(self.config.VOL_COLS, list)
            and all(isinstance(col, str) for col in self.config.VOL_COLS)
        ):
            raise ValueError(
                "VOL_COLS in FeatureBuilderConfig must be a list of strings."
            )
        self.vol_cols_to_process = self.config.VOL_COLS

        self.cat: Catalog = global_cat_loader
        print(
            f"FeatureBuilder initialized with VOL_EWMA: {self.vol_ewma_half_life}, ZSCORE_TIME: {self.zscore_window}"
        )

    @logger
    @requires_columns(FeatureBuilderConfig().VOL_COLS)  # Fallback
    def _calculate_vol_ewma(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Internal method to calculate the Exponential Weighted Moving Average (EWMA)
        for the configured volatility columns.
        The resulting EWMA columns are named '{vol_col}_ewma_hl_{half_life}_zscore'.
        """
        if df.is_empty():
            print(
                "FeatureBuilder Warning: Input DataFrame to _calculate_vol_ewma is empty. Returning as is."
            )
            return df

        ewma_expressions = []
        for vol_col in self.vol_cols_to_process:
            if vol_col not in df.columns:
                print(
                    f"FeatureBuilder Warning: Column '{vol_col}' not found in DataFrame for EWMA calculation. Skipping."
                )
                continue

            ewma_col_name = f"{vol_col}_ewma_hl_{self.vol_ewma_half_life}_zscore"
            ewma_expressions.append(
                pl.col(vol_col)
                .ewm_mean(half_life=self.vol_ewma_half_life, adjust=True)
                .alias(ewma_col_name)
            )

        return df.with_columns(ewma_expressions) if ewma_expressions else df

    @logger
    @requires_columns(FeatureBuilderConfig().VOL_COLS)  # Fallback
    def calculate_vol_zscore(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates a Z-score for specified volatility columns, overwriting intermediate
        columns named '{vol_col}_ewma_hl_{half_life}_zscore' at each step.
        """
        if df.is_empty():
            print(
                "FeatureBuilder Warning: Input DataFrame to calculate_vol_zscore is empty. Returning as is."
            )
            return df

        # Step 1: Calculate EWMA
        df_with_ewma = self._calculate_vol_ewma(df=df)

        # Step 2: Calculate Difference from EWMA and overwrite
        diff_expressions = []
        for vol_col in self.vol_cols_to_process:
            intermediate_col_name = (
                f"{vol_col}_ewma_hl_{self.vol_ewma_half_life}_zscore"
            )
            if vol_col not in df_with_ewma.columns:
                print(
                    f"FeatureBuilder Warning: Original column '{vol_col}' not found for difference calculation. Skipping for this column."
                )
                continue
            if intermediate_col_name not in df_with_ewma.columns:
                print(
                    f"FeatureBuilder Warning: EWMA column '{intermediate_col_name}' not found for difference calculation. Skipping for {vol_col}."
                )
                continue
            diff_expressions.append(
                (pl.col(vol_col) - pl.col(intermediate_col_name)).alias(
                    intermediate_col_name
                )
            )

        df_with_diff = df_with_ewma
        if diff_expressions:
            df_with_diff = df_with_ewma.with_columns(diff_expressions)

        # Step 3: Calculate Z-score of the Difference and overwrite
        zscore_expressions = []
        for vol_col in self.vol_cols_to_process:
            diff_from_ewma_col_name = (
                f"{vol_col}_ewma_hl_{self.vol_ewma_half_life}_zscore"
            )
            if diff_from_ewma_col_name not in df_with_diff.columns:
                print(
                    f"FeatureBuilder Warning: Difference column '{diff_from_ewma_col_name}' not found for Z-score calculation. Skipping for {vol_col}."
                )
                continue

            deviation_series = pl.col(diff_from_ewma_col_name)
            rolling_mean_of_deviation = deviation_series.rolling_mean(
                window_size=self.zscore_window
            )
            rolling_std_of_deviation = deviation_series.rolling_std(
                window_size=self.zscore_window
            )

            zscore_expressions.append(
                pl.when(
                    rolling_std_of_deviation.is_not_null()
                    & (rolling_std_of_deviation != 0)
                )
                .then(
                    (deviation_series - rolling_mean_of_deviation)
                    / rolling_std_of_deviation
                )
                .otherwise(None)
                .alias(diff_from_ewma_col_name)
            )

        return (
            df_with_diff.with_columns(zscore_expressions)
            if zscore_expressions
            else df_with_diff
        )

    @clock()
    @logger
    def _get_spx_data(self, straddle_config: StraddleDataConfig) -> pl.LazyFrame:
        """
        Fetches raw date and close price data for the SPX ticker specified
        in StraddleDataConfig.
        """
        if not isinstance(straddle_config, StraddleDataConfig):
            raise TypeError(
                "straddle_config must be an instance of StraddleDataConfig."
            )
        try:
            return (
                self.cat.om_int_security_price()
                .lazy()
                .filter(pl.col("securityid").eq(straddle_config.TICKER_ID))
                .select(["date", "closeprice"])
            )
        except Exception as e:
            print(
                f"FeatureBuilder Error: Could not fetch SPX data for TICKER_ID {straddle_config.TICKER_ID}. Error: {e}"
            )
            return pl.LazyFrame({"date": [], "closeprice": []}).with_columns(
                [pl.col("date").cast(pl.Date), pl.col("closeprice").cast(pl.Float64)]
            )

    @logger
    def _compute_spx_ret(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """
        Computes daily log returns from a LazyFrame of SPX close prices.
        """
        if "closeprice" not in lf.columns:
            print(
                "FeatureBuilder Warning: 'closeprice' column not found in LazyFrame for SPX return calculation. Returning empty DataFrame."
            )
            return (
                pl.DataFrame(
                    {
                        "date": pl.Series(dtype=pl.Date),
                        "ret": pl.Series(dtype=pl.Float64),
                    }
                )
                # Ensure schema consistency even if empty, and handle potential duplicates if any
                .group_by("date", maintain_order=True)
                .first()
            )

        return (
            lf.sort("date")
            .with_columns(ret=pl.col("closeprice").log().diff())
            .drop("closeprice")
            .collect()
            .drop_nulls(subset=["ret"])
            .group_by("date", maintain_order=True)
            .first()  # Ensure unique date after collect
        )

    @logger
    def calculate_spx_ret(self, straddle_config: StraddleDataConfig) -> pl.DataFrame:
        """
        Calculates daily log returns for the SPX.
        Orchestrates fetching SPX price data and computing returns.
        Requires StraddleDataConfig to identify the SPX ticker.
        """
        lf = self._get_spx_data(straddle_config=straddle_config)
        return self._compute_spx_ret(lf=lf)


if __name__ == "__main__":
    # This __main__ block demonstrates instantiating FeatureBuilder.
    # It assumes that 'pre_loaded_feature_builder_config' is available due to
    # the package's __init__.py having run.
    # It also assumes 'StraddleDataConfig' is available for calculate_spx_ret.

    print("--- Example: Running FeatureBuilder with Pre-loaded Configs ---")

    try:
        # FeatureBuilder is now initialized without arguments
        feature_builder_instance = FeatureBuilder()
        print(
            f"FeatureBuilder initialized. VOL_COLS: {feature_builder_instance.vol_cols_to_process}"
        )

        # For calculate_spx_ret, we still need a StraddleDataConfig instance.
        # In a full application, this might also be a pre-loaded config.
        # For this example, we'll instantiate it directly or assume it's pre-loaded.

        # Option 1: Use a pre-loaded straddle_config if available from your package's __init__.py
        # from . import straddle_config as pre_loaded_straddle_config_for_spx
        # current_straddle_config = pre_loaded_straddle_config_for_spx

        # Option 2: Instantiate directly for this example if pre-loaded is not set up for this specific __main__
        current_straddle_config = StraddleDataConfig()  # Uses defaults
        print(
            f"Using StraddleDataConfig with TICKER_ID: {current_straddle_config.TICKER_ID} for SPX returns."
        )

        spx_returns = feature_builder_instance.calculate_spx_ret(
            straddle_config=current_straddle_config
        )
        print("\nSPX Returns:")
        if not spx_returns.is_empty():
            print(spx_returns.head())
        else:
            print("SPX returns DataFrame is empty.")

        # Example for Z-score (requires a DataFrame with VIX, VVIX, VIX_TS)
        # You would typically get this 'volatility_df' from VolatilityIndicesFetcher
        import datetime

        mock_vol_df = pl.DataFrame(
            {
                "date": [datetime.date(2023, 1, i + 1) for i in range(30)],
                "VIX": [12.0 + np.random.randn() for _ in range(30)],
                "VVIX": [100.0 + np.random.randn() * 5 for _ in range(30)],
                "VIX_TS": [1.1 + np.random.randn() * 0.1 for _ in range(30)],
                "OTHER_COL": [1] * 30,
            }
        )
        print("\nCalculating Vol Z-scores on mock data...")
        vol_with_zscores = feature_builder_instance.calculate_vol_zscore(mock_vol_df)
        if not vol_with_zscores.is_empty():
            print(vol_with_zscores.head())
            print(f"Columns in Z-score output: {vol_with_zscores.columns}")
        else:
            print("Z-score DataFrame is empty.")

    except ImportError as e:
        print(
            f"ImportError in __main__: {e}. This example requires the package structure and __init__.py to be correctly set up."
        )
    except Exception as e:
        print(f"An error occurred in __main__: {e}")
