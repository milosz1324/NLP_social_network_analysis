from pathlib import Path
import argparse
import re
import pandas as pd

from src.config import PROCESSED_DATA_DIR

DATE_UTC_COL = "date_utc"


EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

FORWARDED_ORIGINAL_RE = re.compile(
    r"(?is)(-{2,}\s*original message\s*-{2,}|from:\s.+?sent:\s.+?to:\s.+?subject:)"
)

ENRON_FORWARDED_HEADER_RE = re.compile(
    r"(?is)^-{2,}\s*forwarded by .+?-{2,}\s*"
)

LEGAL_FOOTER_RE = re.compile(
    r"(?is)(this e-mail and any files transmitted with it|"
    r"this message is intended only for|"
    r"confidentiality notice).*$"
)


def extract_emails(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(EMAIL_REGEX, str(text).lower())


def normalize_email(email: str) -> str:
    return email.strip().lower()


def serialize(list_: list[str]) -> str:
    return ", ".join(list_)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_email_body(text: str | None) -> str:
    if not text:
        return ""

    text = str(text)
    text = ENRON_FORWARDED_HEADER_RE.sub("", text)
    text = FORWARDED_ORIGINAL_RE.sub("", text)
    text = LEGAL_FOOTER_RE.sub("", text)

    return normalize_whitespace(text)


def parse_email_message(raw_message: str | None) -> dict[str, str]:
    if not raw_message:
        raw_message = ""

    raw_message = str(raw_message)

    emails = [normalize_email(e) for e in extract_emails(raw_message)]

    sender = emails[0] if emails else ""
    recipients = list(set(emails))

    body = clean_email_body(raw_message)

    return {
        "sender": sender,
        "recipients": serialize(recipients),
        "body": body,
        "clean_body": body,
    }


def preprocess_emails(df: pd.DataFrame, message_col: str = "message") -> pd.DataFrame:
    if message_col not in df.columns:
        raise ValueError(f"Missing column: {message_col}")

    parsed = df[message_col].apply(parse_email_message).apply(pd.Series)
    processed = pd.concat([df, parsed], axis=1)

    if "date" in processed.columns:
        processed[DATE_UTC_COL] = pd.to_datetime(
            processed["date"],
            errors="coerce",
            utc=True,
            format="mixed",
        )

    return processed


def filter_by_date_window(df, start_date=None, end_date=None, date_col=DATE_UTC_COL):
    if not start_date and not end_date:
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    mask = dates.notna()

    if start_date:
        mask &= dates >= pd.to_datetime(start_date, utc=True)

    if end_date:
        mask &= dates <= pd.to_datetime(end_date, utc=True)

    return df.loc[mask].copy()


def validate_date_window(start_date: str | None, end_date: str | None) -> None:
    """Fail early when CLI date arguments are invalid."""
    parsed_start = None
    parsed_end = None

    if start_date:
        try:
            parsed_start = pd.to_datetime(start_date, utc=True)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --start-date: {start_date}. Use a valid date, e.g. 2001-01-01."
            ) from exc

    if end_date:
        try:
            parsed_end = pd.to_datetime(end_date, utc=True)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --end-date: {end_date}. Use a valid date, e.g. 2001-06-30."
            ) from exc

    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")


def filter_enron_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only emails where sender AND recipients are from enron.com"""
    required_columns = {"sender", "recipients"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required column(s): {missing}. "
            "Run preprocess_emails() before filter_enron_only()."
        )

    def is_enron(email: str) -> bool:
        return isinstance(email, str) and "@enron.com" in email.lower()

    # sender must be from enron company
    df = df[df["sender"].apply(is_enron)].copy()

    # recipients: zostaw tylko enronowych
    def filter_recipients(value):
        if pd.isna(value):
            return ""

        recipients = [
            r.strip()
            for r in str(value).split(",")
            if is_enron(r)
        ]

        return ", ".join(recipients)

    df["recipients"] = df["recipients"].apply(filter_recipients)

    # delete rows without recipients
    df = df[df["recipients"].str.len() > 0]

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse raw Enron email messages into metadata and cleaned text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_sample.csv",
        help="Path to CSV with a raw message column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_preprocessed.csv",
        help="Path where the parsed CSV should be saved.",
    )
    parser.add_argument(
        "--message-col",
        default="message",
        help="Name of the column containing raw email messages.",
    )
    parser.add_argument(
        "--start-date",
        help="Keep only messages on or after this date, e.g. 2001-10-01.",
    )
    parser.add_argument(
        "--end-date",
        help="Keep only messages on or before this date, e.g. 2001-12-31.",
    )
    parser.add_argument(
        "--drop-raw-message",
        action="store_true",
        help="Do not keep the original raw message column in the output CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.start_date or args.end_date:
        validate_date_window(args.start_date, args.end_date)

    df = pd.read_csv(args.input)
    print(f"Loaded rows: {len(df)}", flush=True)
    print("Parsing email headers and bodies...", flush=True)
    processed = preprocess_emails(df, message_col=args.message_col)
    processed = filter_enron_only(processed)
    print(f"After filtering to Enron-only emails: {len(processed)}", flush=True)
    rows_after_enron = len(processed)

    if args.start_date or args.end_date:
        print("Applying date filter...", flush=True)
        processed = filter_by_date_window(
            processed,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Rows after date filter: {len(processed)}", flush=True)

    if args.drop_raw_message and args.message_col in processed.columns:
        processed = processed.drop(columns=[args.message_col])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)

    print(f"Rows after Enron filter (before optional date filter): {rows_after_enron}")
    print(f"Columns: {', '.join(processed.columns)}")
    print(f"Saved preprocessed emails to: {args.output}")


if __name__ == "__main__":
    main()
