import datetime
import json  # Still needed for the from_json_file method to read the file
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Type, TypeVar

# Assuming config_loader.py is in a 'utils' subdirectory relative to this config.py
# Adjust the import path based on your actual project structure.
# If config_loader.py is in the same directory, it would be:
# from .config_loader import load_config_from_json_data
from .utils.config_loader import (
    load_config_from_json_data,
)  # Preferred for organization

T = TypeVar("T")


@dataclass
class StraddleDataConfig:
    """Configuration for the straddle data processing script."""

    DATEMIN_TUPLE: Tuple[int, int, int] = (2023, 1, 1)
    TICKER_ID: int = 108105
    DATE_PROBLEMS: List[datetime.date] = field(
        default_factory=lambda: [datetime.date(2025, 5, 12), datetime.date(2025, 1, 8)]
    )

    @property
    def min_date(self) -> datetime.date:
        """Returns the minimum date as a datetime.date object."""
        return datetime.date(*self.DATEMIN_TUPLE)

    @classmethod
    def from_json_file(cls: Type[T], json_path: str) -> T:
        """Loads an instance of StraddleDataConfig from a JSON file."""
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JSON config file not found at {json_path} for {cls.__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path} for {cls.__name__}: {e}"
            )
        return load_config_from_json_data(cls, config_data)


@dataclass
class SignalGeneratorConfig:
    """Configuration for the SignalGenerator."""

    LAGS: Tuple[int, ...] = (1, 2, 3)
    SIGNAL_WEIGHTS: Tuple[float, ...] = (0.6, 0.3, 0.1)

    @classmethod
    def from_json_file(cls: Type[T], json_path: str) -> T:
        """Loads an instance of SignalGeneratorConfig from a JSON file."""
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JSON config file not found at {json_path} for {cls.__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path} for {cls.__name__}: {e}"
            )
        return load_config_from_json_data(cls, config_data)


@dataclass
class BacktesterConfig:
    """Configuration for the Backtester."""

    CAPITAL: float = 1_000_000.0
    WEIGHT_PER_TRADE: float = 0.05

    @classmethod
    def from_json_file(cls: Type[T], json_path: str) -> T:
        """Loads an instance of BacktesterConfig from a JSON file."""
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JSON config file not found at {json_path} for {cls.__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path} for {cls.__name__}: {e}"
            )
        return load_config_from_json_data(cls, config_data)


@dataclass
class FeatureBuilderConfig:
    """Configuration for the FeatureBuilder."""

    VOL_EWMA: int = 5
    VOL_COLS: List[str] = field(default_factory=lambda: ["VIX", "VVIX", "VIX_TS"])
    ZSCORE_TIME: int = 252

    @classmethod
    def from_json_file(cls: Type[T], json_path: str) -> T:
        """Loads an instance of FeatureBuilderConfig from a JSON file."""
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JSON config file not found at {json_path} for {cls.__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path} for {cls.__name__}: {e}"
            )
        return load_config_from_json_data(cls, config_data)


@dataclass
class AnalyzerConfig:
    """Configuration for the Analyzer."""

    DEFAULT_ANNUALIZATION_FACTOR: int = 252
    DEFAULT_RISK_FREE_RATE_DAILY: float = 0.0
    DEFAULT_PORTFOLIO_SIZE_FOR_DRAWDOWN: float = 1_000_000.0
    DEFAULT_PLOT_STYLE: str = "seaborn-v0_8-darkgrid"
    DEFAULT_FIGSIZE: Tuple[float, float] = (14, 7)
    DEFAULT_TITLE_FONTSIZE: int = 16
    DEFAULT_LABEL_FONTSIZE: int = 12
    DEFAULT_DATE_FORMAT_STR: str = "%Y-%m-%d"
    DEFAULT_HISTOGRAM_BINS: Any = "auto"
    DEFAULT_PERCENTILES: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    DEFAULT_TIMESERIES_LINE_COLOR: str = "maroon"
    DEFAULT_HISTOGRAM_COLOR: str = "cornflowerblue"
    DEFAULT_KDE_COLOR: str = "red"
    DEFAULT_PNL_BAR_POSITIVE_COLOR: str = "mediumseagreen"
    DEFAULT_PNL_BAR_NEGATIVE_COLOR: str = "lightcoral"
    DEFAULT_DECILE_BAR_COLOR_LONG: str = "mediumseagreen"
    DEFAULT_DECILE_BAR_COLOR_SHORT: str = "lightcoral"
    DEFAULT_ACF_LINE_COLOR: str = "steelblue"
    DEFAULT_ACF_MARKER_COLOR: str = "crimson"

    @classmethod
    def from_json_file(cls: Type[T], json_path: str) -> T:
        """Loads an instance of AnalyzerConfig from a JSON file."""
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JSON config file not found at {json_path} for {cls.__name__}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from {json_path} for {cls.__name__}: {e}"
            )
        return load_config_from_json_data(cls, config_data)
