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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "emails_sample.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR / "emails_preprocessed.csv")
    parser.add_argument("--message-col", default="message")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    print("Loaded:", len(df))

    processed = preprocess_emails(df, args.message_col)

    processed = filter_by_date_window(processed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Rows:", len(processed))


if __name__ == "__main__":
    main()
