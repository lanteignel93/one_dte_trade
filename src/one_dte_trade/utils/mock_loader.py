# In: your_package_name/utils/mock_loader.py
import polars as pl
import os
from typing import Dict

class MockCatLoader:
    """
    A mock data catalog loader that reads data from cached Parquet files.
    Mimics the necessary methods of onepipeline.conf.cat for offline use.
    """
    def __init__(self, data_paths: Dict[str, str]):
        """
        Initializes the MockCatLoader.

        Args:
            data_paths: A dictionary mapping data source names (e.g., 
                        "om_int_security_price") to their Parquet file paths.
        """
        self.data_paths = data_paths
        self._loaded_data = {} # Optional: Cache loaded DataFrames in memory

        # Validate paths at initialization
        for source_name, path in self.data_paths.items():
            if not os.path.exists(path):
                print(f"MockCatLoader Warning: Parquet file for source '{source_name}' not found at '{path}'. Method calls will fail.")

    def _load_source(self, source_name: str) -> pl.DataFrame:
        """
        Loads a data source from its Parquet file.
        Caches the DataFrame in memory after the first load.
        """
        if source_name in self._loaded_data:
            return self._loaded_data[source_name].clone() # Return a clone to prevent accidental modification

        path = self.data_paths.get(source_name)
        if path is None:
            raise ValueError(f"MockCatLoader Error: No path configured for data source '{source_name}'.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"MockCatLoader Error: Parquet file for '{source_name}' not found at '{path}'.")
        
        print(f"MockCatLoader: Loading '{source_name}' from '{path}'...")
        try:
            df = pl.read_parquet(path)
            self._loaded_data[source_name] = df # Cache it
            return df.clone()
        except Exception as e:
            raise IOError(f"MockCatLoader Error: Failed to read Parquet file '{path}' for source '{source_name}'. Error: {e}")

    def om_int_security_price(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame for security prices from a cached Parquet file.
        The DataFrame is expected to be usable with .lazy() by downstream code.
        """
        return self._load_source("om_int_security_price")

    def om_int_option_price(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame for option prices from a cached Parquet file.
        The DataFrame is expected to be usable with .lazy() by downstream code.
        """
        return self._load_source("om_int_option_price")

    # Add other methods here if your pipeline uses more data sources from `cat`,
    # e.g., cat.some_other_data_source() -> self._load_source("some_other_data_source")

