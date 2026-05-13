"""
This module provides functions to load the Enron email dataset from a CSV file,
print a summary of the dataset, and save a sample for further processing. It includes
a command-line interface for flexible usage.
"""

from pathlib import Path
import argparse

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_email_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load an Enron-like email CSV file."""
    return pd.read_csv(path, nrows=nrows)


def save_processed_sample(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a processed dataframe sample for repeatable experiments."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a compact summary useful at the start of exploration."""
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst rows:")
    print(df.head().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an Enron-like email CSV and save a processed sample."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / "emails.csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_sample.csv",
        help="Path where the loaded sample should be saved.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=1000,
        help="Number of rows to load. Use 0 to load the full file.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only display the summary without saving the sample.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nrows = None if args.nrows == 0 else args.nrows

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Put the Enron CSV in data/raw/ or pass a path with --input."
        )

    df = load_email_csv(args.input, nrows=nrows)
    print_dataset_summary(df)

    if not args.no_save:
        save_processed_sample(df, args.output)
        print(f"\nSaved sample to: {args.output}")


if __name__ == "__main__":
    main()
