import functools
import logging
import time
from typing import List

import polars as pl


class Colors:
    def __init__(self):
        # Regular colors
        self.red = "\033[91m"
        self.green = "\033[92m"
        self.blue = "\033[94m"
        self.yellow = "\033[93m"
        self.magenta = "\033[95m"
        self.cyan = "\033[96m"
        self.white = "\033[97m"
        self.reset = "\033[0m"


DEFAULT_FMT = "[{self.time_color}{elapsed:0.8f}s{self.reset_color}] {name}({args}) -> {self.result_color}{result}{self.reset_color}"


class clock:
    def __init__(self, fmt=DEFAULT_FMT):
        self.fmt = fmt
        self.time_color = Colors().green
        self.result_color = Colors().red
        self.reset_color = Colors().reset

    def __call__(self, func):
        def clocked(*_args, **_kwargs):
            t0 = time.perf_counter()
            _result = func(*_args, **_kwargs)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            args = ", ".join(repr(arg) for arg in _args)
            args += "," + ", ".join(
                f"{repr(arg)}={_kwargs.get(arg)}" for arg in _kwargs
            )
            result = repr(_result)
            print(self.fmt.format(**locals()))
            return _result

        return clocked


def requires_columns(required_cols: List[str]):
    """Decorator to check if an input DataFrame has the required columns."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper_requires_columns(*args, **kwargs):
            # Assume the DataFrame is the first positional argument after 'self' (for methods)
            # or the first argument for functions. This might need adjustment.
            df_arg_index = 0
            if (
                args
                and hasattr(args[0], "__class__")
                and not isinstance(args[0], pl.DataFrame)
            ):  # Likely 'self'
                df_arg_index = 1

            if len(args) > df_arg_index and isinstance(
                args[df_arg_index], pl.DataFrame
            ):
                df = args[df_arg_index]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    raise ValueError(
                        f"Function {func.__name__!r} missing required columns: {missing}"
                    )
            else:
                # Could also check kwargs if df is passed by keyword
                found_in_kwargs = False
                for kw_name, kw_val in kwargs.items():
                    if isinstance(
                        kw_val, pl.DataFrame
                    ):  # Simple check, might need refinement
                        df = kw_val
                        missing = [
                            col for col in required_cols if col not in df.columns
                        ]
                        if missing:
                            raise ValueError(
                                f"Function {func.__name__!r} missing required columns from kwarg '{kw_name}': {missing}"
                            )
                        found_in_kwargs = True
                        break
                if not found_in_kwargs:
                    print(
                        f"Warning: Decorator @requires_columns could not find DataFrame argument for {func.__name__!r}"
                    )

            return func(*args, **kwargs)

        return wrapper_requires_columns

    return decorator


# Configure basic logging (you can customize this further)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def logger(func):
    """Log the function call, arguments, and result."""

    @functools.wraps(func)
    def wrapper_logger(*args, **kwargs):
        # args_repr = [repr(a) for a in args]
        # kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        # signature = ", ".join(args_repr + kwargs_repr)

        # For complex objects like DataFrames, logging the full repr might be too much.
        # Consider logging types or shapes for DataFrames.
        args_summary = []
        for i, arg in enumerate(args):
            if isinstance(arg, pl.DataFrame):
                args_summary.append(f"arg{i}(DataFrame shape={arg.shape})")
            elif (
                hasattr(arg, "__class__") and "Config" in arg.__class__.__name__
            ):  # For config objects
                args_summary.append(f"arg{i}({arg.__class__.__name__})")
            else:
                args_summary.append(repr(arg))

        kwargs_summary = []
        for k, v in kwargs.items():
            if isinstance(v, pl.DataFrame):
                kwargs_summary.append(f"{k}=DataFrame(shape={v.shape})")
            elif hasattr(v, "__class__") and "Config" in v.__class__.__name__:
                kwargs_summary.append(f"{k}={v.__class__.__name__}")
            else:
                kwargs_summary.append(f"{k}={v!r}")

        signature_summary = ", ".join(args_summary + kwargs_summary)

        logging.info(f"Calling {func.__name__}({signature_summary})")
        value = func(*args, **kwargs)

        if isinstance(value, pl.DataFrame):
            logging.info(
                f"{func.__name__!r} returned DataFrame with shape {value.shape}"
            )
        else:
            logging.info(f"{func.__name__!r} returned {value!r}")
        return value

    return wrapper_logger
