import itertools
from typing import Dict, List  # Added Any for broader compatibility if needed

import polars as pl

from one_dte_trade import (
    logger,
    requires_columns,
)

# Import the pre-instantiated signal_generator_config object.
# This assumes that the __init__.py of the parent package (e.g., 'one_dte_trade')
# has loaded the configurations and made 'signal_generator_config' available for import.
# If this script (e.g., signal_generator.py) is in the same directory as that __init__.py:
from one_dte_trade import (
    signal_generator_config as pre_loaded_signal_generator_config,
)

# Import the SignalGeneratorConfig class definition for type hinting
# This assumes 'config.py' is in the same directory or accessible in the package.
# StraddleDataConfig might not be directly needed by __main__ if OneDTEDataPipeline
# also uses pre-loaded configs, but FeatureBuilderConfig might be if FeatureBuilder is used.
from one_dte_trade.config import (
    SignalGeneratorConfig,
)

# If this script is one level deeper, it might be: from .. import signal_generator_config
# For the __main__ block, we'll need OneDTEDataPipeline and its dependencies
# These imports assume they are correctly structured within your package
from one_dte_trade.data import (
    OneDTEDataPipeline,
)


class SignalGenerator:
    """
    Generates trading signals based on calculated straddle returns.
    It processes a DataFrame containing pivoted straddle prices (open/close for calls/puts)
    and applies logic for returns, lags, and weighted signals.
    It uses pre-loaded configurations.
    """

    return_columns_bases = [
        "ret_sprc",
        "ret_uprc",
    ]  # Base names for return calculations

    def __init__(self):
        """
        Initializes the SignalGenerator.
        It uses the pre-loaded 'signal_generator_config' made available
        at the package level (expected to be loaded by __init__.py).
        """
        self.config: SignalGeneratorConfig = pre_loaded_signal_generator_config

        if self.config is None:
            print(
                "SignalGenerator Warning: Pre-loaded signal_generator_config is None. Using default SignalGeneratorConfig()."
            )
            self.config = SignalGeneratorConfig()  # Fallback

        if len(self.config.LAGS) != len(self.config.SIGNAL_WEIGHTS):
            raise ValueError(
                "LAGS and SIGNAL_WEIGHTS in SignalGeneratorConfig must have the same length."
            )
        if not all(isinstance(lag, int) and lag > 0 for lag in self.config.LAGS):
            raise ValueError(
                "LAGS in SignalGeneratorConfig must be a tuple of positive integers."
            )
        if not all(isinstance(w, (int, float)) for w in self.config.SIGNAL_WEIGHTS):
            raise ValueError(
                "SIGNAL_WEIGHTS in SignalGeneratorConfig must be a tuple of numbers."
            )

        self.lags = self.config.LAGS
        self.signal_weights = self.config.SIGNAL_WEIGHTS

        # This internal mapping should be initialized here if _calculate_signal_components depends on it being an instance attribute.
        self._temp_weighted_col_names_map: Dict[str, List[str]] = {
            ret_col_base: [] for ret_col_base in self.return_columns_bases
        }
        print(
            f"SignalGenerator initialized with LAGS: {self.lags}, WEIGHTS: {self.signal_weights}"
        )

    @logger
    @requires_columns(["straddle_day_0", "straddle_day_1", "uprc_diff"])
    def _calculate_straddle_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates two types of percentage returns based on pre-calculated straddle values.

        Args:
            df: Input DataFrame, expected to contain columns:
                'straddle_day_0': Price of the straddle at opening.
                'straddle_day_1': Price of the straddle at closing.
                'uprc_diff': Difference between underlying price at close (uprc_day_1)
                             and the strike price (okey_xx). This method will take
                             the absolute value of this difference for its calculation.

        Returns:
            DataFrame with added columns: 'ret_sprc' and 'ret_uprc'.
        """
        required_cols = ["straddle_day_0", "straddle_day_1", "uprc_diff"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame for _calculate_straddle_returns is missing required columns: {missing_cols}"
            )

        df_with_returns = df.with_columns(
            ret_sprc=pl.when(pl.col("straddle_day_0") != 0)
            .then(
                (pl.col("straddle_day_0") - pl.col("straddle_day_1"))
                / pl.col("straddle_day_0")
            )
            .otherwise(None),
            ret_uprc=pl.when(pl.col("straddle_day_0") != 0)
            .then(
                (pl.col("straddle_day_0") - pl.col("uprc_diff").abs())
                / pl.col("straddle_day_0")
            )
            .otherwise(None),
        )
        return df_with_returns

    @logger
    def _calculate_signal_components(
        self, df_with_returns: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculates lagged returns, weighted lagged returns, and final signals.
        """
        df_current = df_with_returns

        # Re-initialize map for each call to ensure it's clean for this specific DataFrame processing
        self._temp_weighted_col_names_map: Dict[str, List[str]] = {
            ret_col_base: [] for ret_col_base in self.return_columns_bases
        }

        lag_expressions = []
        for ret_col_base, lag_val in itertools.product(
            self.return_columns_bases, self.lags
        ):
            if ret_col_base not in df_current.columns:
                print(
                    f"SignalGenerator Warning: Base return column '{ret_col_base}' not found. Skipping lag {lag_val} for it."
                )
                continue
            lag_expressions.append(
                pl.col(ret_col_base)
                .shift(lag_val)
                .alias(f"{ret_col_base}_lag_{lag_val}")
            )

        if lag_expressions:
            df_current = df_current.with_columns(lag_expressions)

        weighted_lag_expressions = []
        for ret_col_base in self.return_columns_bases:
            for lag_val, weight_val in zip(self.lags, self.signal_weights):
                original_lagged_col_name = f"{ret_col_base}_lag_{lag_val}"

                if original_lagged_col_name not in df_current.columns:
                    print(
                        f"SignalGenerator Warning: Lagged column '{original_lagged_col_name}' not found. Skipping its weighted version."
                    )
                    continue

                weighted_col_alias = f"{ret_col_base}_lag_{lag_val}_{weight_val}"
                weighted_lag_expressions.append(
                    (pl.col(original_lagged_col_name) * weight_val).alias(
                        weighted_col_alias
                    )
                )
                self._temp_weighted_col_names_map[ret_col_base].append(
                    weighted_col_alias
                )

        if weighted_lag_expressions:
            df_current = df_current.with_columns(weighted_lag_expressions)

        signal_final_expressions = []
        for ret_col_base in self.return_columns_bases:
            cols_to_sum_for_signal = self._temp_weighted_col_names_map[ret_col_base]
            actual_cols_to_sum = [
                cn for cn in cols_to_sum_for_signal if cn in df_current.columns
            ]

            if not actual_cols_to_sum:
                print(
                    f"SignalGenerator Warning: No valid weighted columns found to sum for '{ret_col_base}_signal'. Signal will be null."
                )
                signal_final_expressions.append(
                    pl.lit(None, dtype=pl.Float64).alias(f"{ret_col_base}_signal")
                )
                continue

            signal_final_expressions.append(
                pl.sum_horizontal(actual_cols_to_sum).alias(f"{ret_col_base}_signal")
            )

        if signal_final_expressions:
            df_current = df_current.with_columns(signal_final_expressions)

        return df_current

    @logger
    def generate_signals(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generates trading signals by first calculating straddle returns and then
        applying lag and weighting logic.
        """
        if input_df.is_empty():
            print(
                "SignalGenerator Warning: Input DataFrame to generate_signals is empty. Returning as is."
            )
            return input_df

        df_with_returns = self._calculate_straddle_returns(input_df)

        if df_with_returns.is_empty() and not input_df.is_empty():
            print(
                "SignalGenerator Warning: DataFrame became empty after calculating straddle returns."
            )
            empty_signal_cols = [
                pl.lit(None, dtype=pl.Float64).alias(f"{rcb}_signal")
                for rcb in self.return_columns_bases
            ]
            # Return the df_with_returns which is empty but might have the return columns, plus empty signal columns
            return (
                df_with_returns.with_columns(empty_signal_cols)
                if not df_with_returns.is_empty()
                else pl.DataFrame().with_columns(empty_signal_cols)
            )

        df_with_signals = self._calculate_signal_components(df_with_returns)

        # Consider the impact of this drop_nulls(). It will remove rows with leading nulls from shift operations.
        # If you need to keep all original rows and have null signals for initial periods,
        # you might remove this or make it more specific (e.g., drop_nulls(subset=["ret_sprc_signal", "ret_uprc_signal"])).
        return df_with_signals.drop_nulls()


if __name__ == "__main__":
    print("--- Example: Running SignalGenerator with Pre-loaded Configs ---")

    # OneDTEDataPipeline now uses configs loaded by its package's __init__.py
    # (or defaults if loading failed or __init__.py wasn't run prior to its import)
    # It also instantiates its own FeatureBuilder using pre-loaded feature_builder_config.

    # Ensure your package's __init__.py has loaded FeatureBuilderConfig correctly
    # and that the FeatureBuilder class is correctly imported and used within OneDTEDataPipeline.
    # For this __main__ block, we assume OneDTEDataPipeline is correctly set up
    # to use its pre-loaded configs.

    try:
        # This pipeline instance will use pre-loaded straddle_config and feature_builder_config
        pipeline = OneDTEDataPipeline()

        # The get_1dte_data method uses the internal FeatureBuilder
        df_from_pipeline = pipeline.get_1dte_data(
            generate_features=True
        )  # Ensure features are generated

        if df_from_pipeline.is_empty():
            print(
                "DataFrame from OneDTEDataPipeline is empty. Cannot generate signals."
            )
        else:
            print(
                f"DataFrame from OneDTEDataPipeline received with columns: {df_from_pipeline.columns}"
            )
            # SignalGenerator also uses pre-loaded signal_generator_config
            signal_gen = SignalGenerator()
            df_with_signals = signal_gen.generate_signals(input_df=df_from_pipeline)

            print("\n--- DataFrame with Signals ---")
            if not df_with_signals.is_empty():
                with pl.Config(
                    tbl_rows=10,
                    tbl_cols=df_with_signals.width // 2
                    if df_with_signals.width > 20
                    else df_with_signals.width,
                ):  # Adjust display
                    print(df_with_signals.head())
                print(
                    f"Signals generated. Final DataFrame shape: {df_with_signals.shape}"
                )
                print(f"Columns: {df_with_signals.columns}")
            else:
                print("Signal generation resulted in an empty DataFrame.")

    except ImportError as e:
        print(
            f"ImportError in __main__: {e}. This example requires the package structure and __init__.py to be correctly set up."
        )
    except Exception as e:
        print(f"An error occurred in __main__: {e}")
