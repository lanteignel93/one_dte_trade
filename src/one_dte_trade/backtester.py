from typing import Dict, List  # Added Optional

import polars as pl

# Import the pre-instantiated backtester_config object.
# This assumes that the __init__.py of the parent package (e.g., 'one_dte_trade')
# has loaded the configurations and made 'backtester_config' available for import.
from one_dte_trade import (
    backtester_config as pre_loaded_backtester_config,
)
from one_dte_trade import (
    logger,
    requires_columns,
)

# Import the BacktesterConfig class definition for type hinting
# Assumes 'config.py' is in the same directory or accessible in the package.
from one_dte_trade.config import (
    BacktesterConfig,
)

# For the __main__ block, we'll need other components and their configs/loaders
# These imports assume they are correctly structured and that their respective
# config objects are also pre-loaded by the package's __init__.py if they follow the same pattern.
from one_dte_trade.data import (
    OneDTEDataPipeline,
)  # For __main__
from one_dte_trade.strategy_signal import (
    SignalGenerator,
)  # For __main__


class Backtester:
    """
    Performs a vectorized backtest based on input signals and price data.
    It uses pre-loaded configurations.

    The backtesting process involves:
    1. Calculating capital allocated per trade based on signal strength.
    2. Determining the number of positions (e.g., straddles) to take.
    3. Calculating daily Profit and Loss (PnL) for each signal.
    4. Calculating cumulative PnL for each signal.
    """

    # These define the signals to use and their corresponding price sources for PnL calculation.
    # Assumes a 1:1 mapping by index.
    # These could also be moved to BacktesterConfig if they need to be configurable.
    _signal_cols: List[str] = ["ret_sprc_signal", "ret_uprc_signal"]
    _open_price_col_for_positions: str = (
        "straddle_day_0"  # Column used as the price to determine number of positions
    )
    _pnl_calculation_sources: List[Dict[str, str]] = [
        {
            "signal_name_base": "ret_sprc",
            "close_ref_col": "straddle_day_1",
            "open_ref_col": "straddle_day_0",
        },
        {
            "signal_name_base": "ret_uprc",
            "close_ref_col": "uprc_diff",  # This is abs(uprc_day_1 - okey_xx) from StraddleDataPipeline
            "open_ref_col": "straddle_day_0",
        },
    ]

    def __init__(self):
        """
        Initializes the Backtester.
        It uses the pre-loaded 'backtester_config' made available
        at the package level (expected to be loaded by __init__.py).
        """
        self.config: BacktesterConfig = pre_loaded_backtester_config

        if self.config is None:
            print(
                "Backtester Warning: Pre-loaded backtester_config is None. Using default BacktesterConfig()."
            )
            self.config = BacktesterConfig()  # Fallback

        self.capital = self.config.CAPITAL
        self.weight_per_trade = self.config.WEIGHT_PER_TRADE

        if not 0 < self.weight_per_trade <= 1:
            raise ValueError(
                "WEIGHT_PER_TRADE in BacktesterConfig must be between 0 (exclusive) and 1 (inclusive)."
            )

        self.capital_per_trade = self.capital * self.weight_per_trade

        # Validate signal configurations (basic check)
        if len(self._signal_cols) != len(self._pnl_calculation_sources):
            raise ValueError(
                "Internal Backtester Mismatch: _signal_cols and _pnl_calculation_sources lengths differ."
            )
        for i, signal_col_name in enumerate(self._signal_cols):
            if not signal_col_name.startswith(
                self._pnl_calculation_sources[i]["signal_name_base"]
            ):
                print(
                    f"Backtester Warning: Signal column '{signal_col_name}' might not align with PnL config base '{self._pnl_calculation_sources[i]['signal_name_base']}'."
                )
        print(
            f"Backtester initialized with CAPITAL: {self.capital:,.0f}, WEIGHT_PER_TRADE: {self.weight_per_trade}"
        )

    @requires_columns(_signal_cols)
    @logger
    def _calculate_capital_per_trade(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the capital allocated to each trade based on the signal value.
        Assumes signal values are multipliers for `self.capital_per_trade`.
        """
        expressions = []
        for signal_col_name in self._signal_cols:
            if signal_col_name not in df.columns:
                raise ValueError(
                    f"Signal column '{signal_col_name}' not found in DataFrame for capital allocation."
                )
            expressions.append(
                (pl.col(signal_col_name) * self.capital_per_trade).alias(
                    f"{signal_col_name}_capital"
                )
            )
        return df.with_columns(expressions)

    @requires_columns([_open_price_col_for_positions])
    @logger
    def _calculate_number_positions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the number of positions (e.g., straddles) to take based on
        allocated capital and the opening price of the instrument.
        """
        expressions = []
        opening_price_col = self._open_price_col_for_positions
        if opening_price_col not in df.columns:
            raise ValueError(
                f"Opening price column '{opening_price_col}' for position sizing not found."
            )

        for signal_col_name in self._signal_cols:
            allocated_capital_col = f"{signal_col_name}_capital"
            if allocated_capital_col not in df.columns:
                raise ValueError(
                    f"Allocated capital column '{allocated_capital_col}' not found for position sizing."
                )

            expressions.append(
                pl.when(
                    pl.col(opening_price_col).is_not_null()
                    & (pl.col(opening_price_col) != 0)
                )
                .then(pl.col(allocated_capital_col) / pl.col(opening_price_col))
                .otherwise(0.0)  # Set position to 0 if opening price is null or zero
                .alias(f"{signal_col_name}_pos")
            )
        return df.with_columns(expressions)

    @logger
    def _calculate_daily_pnl(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the daily Profit and Loss (PnL) for each signal.
        PnL = number_of_positions * (reference_closing_price - reference_opening_price)
        """
        expressions = []
        for pnl_config in self._pnl_calculation_sources:
            signal_col_name = f"{pnl_config['signal_name_base']}_signal"
            position_col = f"{signal_col_name}_pos"
            pnl_col = f"{signal_col_name}_pnl"  # Daily PnL column

            close_ref_col = pnl_config["close_ref_col"]
            open_ref_col = pnl_config["open_ref_col"]

            if position_col not in df.columns:
                raise ValueError(
                    f"Position column '{position_col}' not found for PnL calculation."
                )
            if close_ref_col not in df.columns:
                raise ValueError(
                    f"PnL closing reference column '{close_ref_col}' not found."
                )
            if open_ref_col not in df.columns:
                raise ValueError(
                    f"PnL opening reference column '{open_ref_col}' not found."
                )

            expressions.append(
                (
                    pl.col(position_col)
                    * (pl.col(close_ref_col) - pl.col(open_ref_col))
                ).alias(pnl_col)
            )
        return df.with_columns(expressions)

    @requires_columns([f"{signal_col}_pnl" for signal_col in _signal_cols])
    @logger
    def _calculate_cumulative_pnl(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the cumulative PnL for each signal.
        """
        expressions = []
        for signal_col_name in self._signal_cols:
            daily_pnl_col = f"{signal_col_name}_pnl"
            if daily_pnl_col not in df.columns:
                raise ValueError(
                    f"Daily PnL column '{daily_pnl_col}' not found for cumulative PnL calculation."
                )
            expressions.append(
                pl.col(daily_pnl_col).cum_sum().alias(f"{signal_col_name}_cum_pnl")
            )
        return df.with_columns(expressions)

    @requires_columns(_signal_cols + [_open_price_col_for_positions])
    @logger
    def generate_backtest_results(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Runs the full backtesting pipeline on the input DataFrame.
        """
        if df.is_empty():
            print("Backtester Warning: Input DataFrame is empty. Returning empty.")
            return df

        required_input_cols = set(self._signal_cols)
        required_input_cols.add(self._open_price_col_for_positions)
        for pnl_conf in self._pnl_calculation_sources:
            required_input_cols.add(pnl_conf["open_ref_col"])
            required_input_cols.add(pnl_conf["close_ref_col"])

        missing_cols = [col for col in required_input_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing essential columns for backtesting: {missing_cols}"
            )

        df_processed = self._calculate_capital_per_trade(df)
        df_processed = self._calculate_number_positions(df_processed)
        df_processed = self._calculate_daily_pnl(df_processed)
        df_processed = self._calculate_cumulative_pnl(df_processed)
        return df_processed


if __name__ == "__main__":
    print("--- Example: Running Backtester with Pre-loaded Configs ---")

    # This __main__ block assumes that OneDTEDataPipeline, SignalGenerator,
    # and FeatureBuilder have also been updated to use their respective
    # pre-loaded configs from the package's __init__.py.

    try:
        # These classes should now be initializable without config arguments
        # if their __init__ methods were updated to use pre-loaded configs.
        # OneDTEDataPipeline was updated in 'one_dte_pipeline_updated_configs'
        # SignalGenerator was updated in 'signal_generator_updated_configs'
        # FeatureBuilder would need a similar update if not done yet.

        # For FeatureBuilder, its config is feature_builder_config.
        # OneDTEDataPipeline internally creates FeatureBuilder using pre-loaded feature_builder_config.

        pipeline = (
            OneDTEDataPipeline()
        )  # Uses pre-loaded straddle_config & feature_builder_config
        print("\nGenerating 1DTE data...")
        df_from_pipeline = pipeline.get_1dte_data(generate_features=True)

        if df_from_pipeline.is_empty():
            print(
                "DataFrame from OneDTEDataPipeline is empty. Cannot proceed with signals or backtest."
            )
        else:
            print(
                f"DataFrame from OneDTEDataPipeline received. Shape: {df_from_pipeline.shape}"
            )

            signal_gen = SignalGenerator()  # Uses pre-loaded signal_generator_config
            print("\nGenerating signals...")
            df_with_signals = signal_gen.generate_signals(input_df=df_from_pipeline)

            if df_with_signals.is_empty():
                print(
                    "Signal generation resulted in an empty DataFrame. Cannot proceed with backtest."
                )
            else:
                print(f"Signals generated. Shape: {df_with_signals.shape}")

                backtester_instance = Backtester()  # Uses pre-loaded backtester_config
                print("\nRunning backtest...")
                df_backtest_output = backtester_instance.generate_backtest_results(
                    df_with_signals
                )

                print("\n--- DataFrame with Backtest Results ---")
                if not df_backtest_output.is_empty():
                    with pl.Config(
                        tbl_rows=10,
                        tbl_cols=df_backtest_output.width // 2
                        if df_backtest_output.width > 20
                        else df_backtest_output.width,
                    ):
                        print(df_backtest_output.head())
                    print(
                        f"Backtest results generated. Final DataFrame shape: {df_backtest_output.shape}"
                    )
                    print(f"Columns: {df_backtest_output.columns}")
                else:
                    print("Backtest resulted in an empty DataFrame.")

    except ImportError as e:
        print(
            f"ImportError in __main__: {e}. This example requires the package structure, "
            "__init__.py (for pre-loaded configs), and all component classes to be correctly set up."
        )
    except Exception as e:
        print(f"An error occurred in __main__: {e}")
