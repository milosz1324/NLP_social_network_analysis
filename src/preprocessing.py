from email import policy
from email.message import Message
from email.parser import Parser
from email.utils import getaddresses
from pathlib import Path
import argparse
import re

import pandas as pd

from src.config import PROCESSED_DATA_DIR

DATE_COL = "date"
DATE_UTC_COL = "date_utc"


FORWARDED_ORIGINAL_RE = re.compile(
    r"(?is)("
    r"-{2,}\s*original message\s*-{2,}|"
    r"from:\s.+?sent:\s.+?to:\s.+?subject:"
    r")"
)
ENRON_FORWARDED_HEADER_RE = re.compile(
    r"(?is)^-{2,}\s*forwarded by .+?-{2,}\s*"
)
LEGAL_FOOTER_RE = re.compile(
    r"(?is)(this e-mail and any files transmitted with it|"
    r"this message is intended only for|"
    r"confidentiality notice).*$"
)
EMAIL_PARSER = Parser(policy=policy.default)


def normalize_email_address(address: str) -> str:
    """Return a lower-cased email address or an empty string."""
    return address.strip().lower()


def parse_address_header(value: str | None) -> list[str]:
    """Parse a comma-separated email header into normalized addresses."""
    if not value:
        return []

    addresses = []
    for _, address in getaddresses([value]):
        normalized = normalize_email_address(address)
        if normalized:
            addresses.append(normalized)

    return addresses


def serialize_addresses(addresses: list[str]) -> str:
    """Store multiple addresses in a graph_builder-compatible format."""
    return ", ".join(addresses)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while keeping the message readable."""
    return re.sub(r"\s+", " ", text).strip()


def strip_quoted_threads(text: str) -> str:
    """Remove obvious forwarded/replied message blocks from an email body."""
    match = FORWARDED_ORIGINAL_RE.search(text)
    return text[: match.start()] if match and match.start() > 0 else text


def strip_enron_forwarded_header(text: str) -> str:
    """Remove Enron's forwarded-by separator without deleting forwarded content."""
    return ENRON_FORWARDED_HEADER_RE.sub("", text)


def strip_legal_footer(text: str) -> str:
    """Remove common legal footers found in corporate email."""
    return LEGAL_FOOTER_RE.sub("", text)


def clean_email_body(text: str | None) -> str:
    """Apply lightweight domain cleaning for Enron-style email bodies."""
    if not text:
        return ""

    text = strip_enron_forwarded_header(text)
    text = strip_quoted_threads(text)
    text = strip_legal_footer(text)
    return normalize_whitespace(text)


def _decode_payload(message: Message) -> str:
    payload = message.get_payload(decode=True)
    if payload is None:
        raw_payload = message.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""

    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract_body(message: Message) -> str:
    """Extract plain text body from an email message."""
    if message.is_multipart():
        parts = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                parts.append(_decode_payload(part))
        return "\n".join(parts).strip()

    return _decode_payload(message).strip()


def parse_email_message(raw_message: str | None) -> dict[str, str]:
    """Parse one raw Enron email into metadata and cleaned body fields."""
    if not raw_message:
        raw_message = ""

    message = EMAIL_PARSER.parsestr(str(raw_message))

    to_addresses = parse_address_header(message.get("To"))
    cc_addresses = parse_address_header(message.get("Cc"))
    bcc_addresses = parse_address_header(message.get("Bcc"))
    recipients = to_addresses + cc_addresses + bcc_addresses
    body = extract_body(message)

    return {
        "message_id": str(message.get("Message-ID", "")).strip(),
        "date": str(message.get("Date", "")).strip(),
        "sender": serialize_addresses(parse_address_header(message.get("From"))),
        "recipients": serialize_addresses(recipients),
        "to": serialize_addresses(to_addresses),
        "cc": serialize_addresses(cc_addresses),
        "bcc": serialize_addresses(bcc_addresses),
        "subject": str(message.get("Subject", "")).strip(),
        "body": body,
        "clean_body": clean_email_body(body),
    }


def preprocess_emails(df: pd.DataFrame, message_col: str = "message") -> pd.DataFrame:
    """Add parsed metadata and cleaned body columns to an email dataframe."""
    if message_col not in df.columns:
        raise ValueError(f"Missing required column: {message_col}")

    parsed = df[message_col].apply(parse_email_message).apply(pd.Series)
    processed = pd.concat([df, parsed], axis=1)
    processed[DATE_UTC_COL] = pd.to_datetime(
        processed[DATE_COL],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    return processed


def filter_by_date_window(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = DATE_UTC_COL,
) -> pd.DataFrame:
    """Filter rows to an inclusive UTC date window."""
    if not start_date and not end_date:
        return df

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed", utc=True)
    mask = dates.notna()

    if start_date:
        start = pd.to_datetime(start_date, utc=True)
        mask &= dates >= start

    if end_date:
        end = pd.to_datetime(end_date, utc=True)
        if end.time() == pd.Timestamp.min.time():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= dates <= end

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


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    validate_date_window(args.start_date, args.end_date)

    print(f"Reading input CSV: {args.input}", flush=True)
    df = pd.read_csv(args.input)
    print(f"Loaded rows: {len(df)}", flush=True)
    print("Parsing email headers and bodies...", flush=True)
    processed = preprocess_emails(df, message_col=args.message_col)
    processed = filter_enron_only(processed)
    print(f"After filtering to Enron-only emails: {len(processed)}", flush=True)
    before_filter = len(processed)
    date_filter = args.start_date and args.end_date
    if date_filter:
        print("Applying date filter...", flush=True)
        processed = filter_by_date_window(
            processed,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if args.drop_raw_message:
        processed = processed.drop(columns=[args.message_col])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving output CSV: {args.output}", flush=True)
    processed.to_csv(args.output, index=False)

    print(f"Rows parsed: {before_filter}")
    if date_filter: 
        print(f"Rows after date filter: {len(processed)}")
    print(f"Columns: {', '.join(processed.columns)}")
    print(f"Saved preprocessed emails to: {args.output}")


if __name__ == "__main__":
    main()
