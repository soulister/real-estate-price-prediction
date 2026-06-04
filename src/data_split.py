import pandas as pd


def add_time_blocks(
    df: pd.DataFrame,
    date_col: str = "timestamp",
    n_blocks: int = 6,
    block_col: str = "time_block",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df[block_col] = pd.qcut(
        df[date_col],
        q=n_blocks,
        labels=[f"B{i}" for i in range(1, n_blocks + 1)]
    )

    return df


def summarize_time_blocks(
    df: pd.DataFrame,
    date_col: str = "timestamp",
    block_col: str = "time_block",
) -> pd.DataFrame:
    summary = (
        df.groupby(block_col)
        .agg(
            n_transactions=(date_col, "count"),
            min_date=(date_col, "min"),
            max_date=(date_col, "max"),
        )
        .reset_index()
    )

    summary["time_diff"] = summary["max_date"] - summary["min_date"]

    return summary


def get_time_folds():
    return [
        {
            "fold": 1,
            "train_blocks": ["B1", "B2"],
            "val_block": "B3",
        },
        {
            "fold": 2,
            "train_blocks": ["B1", "B2", "B3"],
            "val_block": "B4",
        },
        {
            "fold": 3,
            "train_blocks": ["B1", "B2", "B3", "B4"],
            "val_block": "B5",
        },
    ]


def get_dev_holdout_split(
    df: pd.DataFrame,
    block_col: str = "time_block",
):
    dev_df = df[df[block_col].isin(["B1", "B2", "B3", "B4", "B5"])].copy()
    holdout_df = df[df[block_col] == "B6"].copy()

    return dev_df, holdout_df


def split_fold(
    df: pd.DataFrame,
    fold: dict,
    block_col: str = "time_block",
):
    train_df = df[df[block_col].isin(fold["train_blocks"])].copy()
    val_df = df[df[block_col] == fold["val_block"]].copy()

    return train_df, val_df