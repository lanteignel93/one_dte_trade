import datetime
import itertools

import polars as pl

from . import cat

DATEMIN = (2023, 1, 1)
SIGNAL_WEIGHTS = (0.6, 0.3, 0.1)
LAGS = (1, 2, 3)
CAPITAL = 1e6
ALLOCATED_WEIGHT = 0.05
SPX_ID = 108105
DATE_PROBLEMS = [datetime.date(2025, 5, 12), datetime.date(2025, 1, 8)]


def open_straddle_data():
    return (
        cat.om_int_option_price()
        .lazy()
        .filter(pl.col("securityid").eq(SPX_ID))
        .select(
            [
                "callput",
                "strike",
                "date",
                "bestbid",
                "bestoffer",
                "expiration",
            ]
        )
        .filter(pl.col("date").ge(datetime.date(*DATEMIN)))
        .with_columns(sprc=(pl.col("bestbid") + pl.col("bestoffer")) / 2)
        .join(
            cat.om_int_security_price()
            .lazy()
            .filter(pl.col("securityid").eq(SPX_ID))
            .select(["date", "closeprice"]),
            on="date",
        )
        .drop(["bestbid", "bestoffer"])
        .rename(
            {
                "callput": "okey_cp",
                "strike": "okey_xx",
                "closeprice": "uprc",
            }
        )
        .with_columns(
            time_to_expiration=(pl.col("expiration") - pl.col("date")).dt.total_days()
        )
        .with_columns(pl.col("okey_xx") * pl.lit(1 / 1000))
        .with_columns(dist_to_strike=(pl.col("okey_xx") - pl.col("uprc")).abs())
        .filter(pl.col("time_to_expiration").gt(0))
        .sort(["date", "dist_to_strike", "time_to_expiration"])
        .group_by(["date", "okey_cp"], maintain_order=True)
        .first()
        .collect()
    )


def close_straddle_data():
    return (
        cat.om_int_option_price()
        .lazy()
        .filter(pl.col("securityid").eq(SPX_ID))
        .select(
            [
                "callput",
                "strike",
                "date",
                "bestbid",
                "bestoffer",
                "expiration",
            ]
        )
        .filter(pl.col("date").ge(datetime.date(*DATEMIN)))
        .with_columns(sprc=(pl.col("bestbid") + pl.col("bestoffer")) / 2)
        .join(
            cat.om_int_security_price()
            .lazy()
            .filter(pl.col("securityid").eq(SPX_ID))
            .select(["date", "closeprice"]),
            on="date",
        )
        .drop(["bestbid", "bestoffer"])
        .rename(
            {
                "callput": "okey_cp",
                "strike": "okey_xx",
                "closeprice": "uprc",
            }
        )
        .with_columns(
            time_to_expiration=(pl.col("expiration") - pl.col("date")).dt.total_days()
        )
        .with_columns(pl.col("okey_xx") * pl.lit(1 / 1000))
        .filter(pl.col("time_to_expiration").lt(1))
        .collect()
    )


def merge_open_close(open_df, close_df):
    return (
        open_df.join(
            close_df,
            left_on=["okey_cp", "okey_xx", "expiration"],
            right_on=["okey_cp", "okey_xx", "date"],
            how="left",
        )
        .drop(
            [
                "time_to_expiration",
                "time_to_expiration_right",
                "expiration_right",
            ]
        )
        .rename(
            {
                "uprc": "open_uprc",
                "sprc": "open_sprc",
                "uprc_right": "close_uprc",
                "sprc_right": "close_sprc",
            }
        )
        .unique()
        .sort(["date", "okey_cp"])
        .filter(~pl.col("date").is_in(DATE_PROBLEMS))
        # .filter(~pl.any_horizontal(pl.all().is_null()))
    )


def pivot_data(df):
    return df.pivot(
        "okey_cp",
        index=[
            "date",
            "okey_xx",
            "open_uprc",
            "expiration",
            "dist_to_strike",
            "close_uprc",
        ],
        values=["open_sprc", "close_sprc"],
    ).filter(~pl.any_horizontal(pl.all().is_null()))


def calculate_straddle_returns(df):
    return df.with_columns(
        open_straddle=pl.col("open_sprc_C") + pl.col("open_sprc_P"),
        close_straddle=pl.col("close_sprc_C") + pl.col("close_sprc_P"),
        uprc_diff=(pl.col("close_uprc") - pl.col("okey_xx")).abs(),
    ).with_columns(
        ret_sprc=(pl.col("open_straddle") - pl.col("close_straddle"))
        / pl.col("open_straddle"),
        ret_uprc=(pl.col("open_straddle") - pl.col("uprc_diff"))
        / pl.col("open_straddle"),
    )


def calculate_signal(df):
    return (
        df.with_columns(
            [
                pl.col(ret_col).shift(lag).alias(f"{ret_col}_lag_{lag}")
                for ret_col, lag in itertools.product(["ret_sprc", "ret_uprc"], LAGS)
            ]
        )
        .with_columns(
            [
                (weight * pl.col(f"{ret_col}_lag_{lag}")).alias(
                    f"{ret_col}_lag_{lag}_{weight}"
                )
                for ret_col, (lag, weight) in itertools.product(
                    ["ret_sprc", "ret_uprc"], zip(LAGS, SIGNAL_WEIGHTS)
                )
            ]
        )
        .with_columns(
            [
                (
                    pl.fold(
                        acc=pl.lit(0, dtype=pl.Float64),
                        function=lambda acc, x: acc + x,
                        exprs=[
                            f"{ret_col}_lag_{lag}_{weight}"
                            for lag, weight in zip(LAGS, SIGNAL_WEIGHTS)
                        ],
                    )
                ).alias(f"{ret_col}_signal")
                for ret_col in ["ret_sprc", "ret_uprc"]
            ]
        )
    )


def calculate_pnl(df):
    return df.with_columns(
        [
            (
                pl.col(signal_col)
                * (pl.col(ref_col) - pl.col("open_straddle"))
                * pl.lit(CAPITAL)
                * pl.lit(ALLOCATED_WEIGHT)
                / pl.col("open_straddle")
            ).alias(f"{signal_col}_pnl")
            for signal_col, ref_col in zip(
                ["ret_sprc_signal", "ret_uprc_signal"], ["close_straddle", "uprc_diff"]
            )
        ]
    ).with_columns(
        [
            pl.col(signal_pnl).cum_sum().alias(f"{signal_pnl}_cumsum")
            for signal_pnl in [
                f"ret_{source}_signal_pnl" for source in ["sprc", "uprc"]
            ]
        ]
    )


if __name__ == "__main__":
    open_df = open_straddle_data()
    close_df = close_straddle_data()
    df = merge_open_close(open_df, close_df)
    df = pivot_data(df)
    df = calculate_straddle_returns(df)
    df = calculate_signal(df)
    df = calculate_pnl(df)
    df.to_pandas().to_csv("/htaa/llanteigne/1dte.csv")
