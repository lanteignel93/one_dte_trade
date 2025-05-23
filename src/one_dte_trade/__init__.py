# __init__.py
# This file is at the root of your package (e.g., 20250515_1DTE_trade/__init__.py)
import polars as pl
import os

# Import your dataclass definitions from config.py (in the same package)
from .config import (
    AnalyzerConfig,
    BacktesterConfig,
    FeatureBuilderConfig,
    SignalGeneratorConfig,
    StraddleDataConfig,
)

# Import the loader function from utils.config_loader.py (in a subpackage)
from .utils.config_loader import load_all_project_configs
from .utils.decorators import clock, logger, requires_columns
from .utils.mock_loader import MockCatLoader

print("Initializing package and loading configurations...")

# --- Define the mapping of JSON keys to your dataclass types ---
# This map should match the top-level keys in your project_settings.json
CONFIG_CLASS_MAP = {
    "StraddleDataConfig": StraddleDataConfig,
    "SignalGeneratorConfig": SignalGeneratorConfig,
    "BacktesterConfig": BacktesterConfig,
    "FeatureBuilderConfig": FeatureBuilderConfig,
    "AnalyzerConfig": AnalyzerConfig,
    # Add other config classes here if you have more
}

# --- Determine the path to the JSON configuration file ---
# This assumes __file__ (path to this __init__.py) is at the root of your package
# and the 'configs' directory is a subdirectory of this package.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_CONFIG_PATH = os.path.join(PACKAGE_ROOT, "configs", "project_settings.json")
CACHED_DATA_DIR = os.path.join(PACKAGE_ROOT, "cached_data")

# --- Instantiate MockCatLoader ---
# Define paths to your cached Parquet files
MOCK_DATA_PATHS = {
    "om_int_security_price": os.path.join(
        CACHED_DATA_DIR, "cached_om_int_security_price.parquet"
    ),
    "om_int_option_price": os.path.join(
        CACHED_DATA_DIR, "cached_om_int_option_price.parquet"
    ),
    # Add other data sources if you cache them
}

# This 'cat' instance will be the mock loader used throughout the package
try:
    cat = MockCatLoader(data_paths=MOCK_DATA_PATHS)
    print("MockCatLoader initialized. Will use cached Parquet files.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize MockCatLoader. Error: {e}")

    # Fallback or raise error if data loading is critical
    class FallbackCatLoader:  # A dummy loader that returns empty DataFrames
        def om_int_security_price(self):
            return pl.DataFrame()

        def om_int_option_price(self):
            return pl.DataFrame()

    cat = FallbackCatLoader()
    print("WARNING: Using FallbackCatLoader due to MockCatLoader initialization error.")

# --- Load all configurations ---
# The `load_all_project_configs` function will return a dictionary where keys
# are the names of the config classes (e.g., "StraddleDataConfig") and values
# are the instantiated config objects.

_loaded_configs = {}  # Use a temporary variable
if os.path.exists(JSON_CONFIG_PATH):
    try:
        _loaded_configs = load_all_project_configs(JSON_CONFIG_PATH, CONFIG_CLASS_MAP)
        print(f"Configurations loaded successfully from {JSON_CONFIG_PATH}")
    except Exception as e:
        print(
            f"CRITICAL ERROR: Failed to load project configurations from {JSON_CONFIG_PATH}. Error: {e}"
        )
        # Depending on your application's needs, you might want to raise the error
        # or allow the application to continue with default-instantiated configs (if possible).
        # For now, we'll proceed, and individual configs might be None or defaults.
else:
    print(
        f"WARNING: Main configuration file not found at {JSON_CONFIG_PATH}. "
        "Configuration objects will be instantiated with their default values."
    )

# --- Make instantiated config objects available at the package level ---
# You can choose how to name them. Using lowercase versions of the class names is common.
straddle_config: StraddleDataConfig = _loaded_configs.get(
    "StraddleDataConfig", StraddleDataConfig()
)
signal_generator_config: SignalGeneratorConfig = _loaded_configs.get(
    "SignalGeneratorConfig", SignalGeneratorConfig()
)
backtester_config: BacktesterConfig = _loaded_configs.get(
    "BacktesterConfig", BacktesterConfig()
)
feature_builder_config: FeatureBuilderConfig = _loaded_configs.get(
    "FeatureBuilderConfig", FeatureBuilderConfig()
)
analyzer_config: AnalyzerConfig = _loaded_configs.get(
    "AnalyzerConfig", AnalyzerConfig()
)

# Optional: You can define __all__ to control what `from your_package import *` imports
__all__ = [
    "straddle_config",
    "signal_generator_config",
    "backtester_config",
    "feature_builder_config",
    "analyzer_config",
    # Also export the dataclass types themselves if users need to type hint them
    "StraddleDataConfig",
    "SignalGeneratorConfig",
    "BacktesterConfig",
    "FeatureBuilderConfig",
    "AnalyzerConfig",
    "clock",
    "logger",
    "requires_columns",
    "cat",
]

print("Package initialization complete. Config objects are now available.")
