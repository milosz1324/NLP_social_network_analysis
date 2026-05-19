import argparse
import time
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

from src.preprocess.preprocessing import filter_enron_only, preprocess_emails
from src.nlp.ner import extract_people_batch
from src.graphs.graph_builder import build_metadata_graph
from src.graphs.graph_ner_builder import build_ner_graph
from src.nlp.relation_classifier import classify_relation


def _step(label: str) -> float:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    return time.time()


def _done(t0: float) -> None:
    print(f"  done in {time.time() - t0:.1f}s")

OUTPUT_DIR = Path("outputs/tables")


def normalize_person(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()

    if "/" in name:
        name = name.split("/")[0]

    if "@" in name:
        name = name.split("@")[0]

    name = "".join(ch for ch in name if ch.isalnum() or ch in " ._-")

    return name.strip()


def is_valid_person(name: str) -> bool:
    if not name or len(name) < 3:
        return False

    bad = {
        "subject", "message", "please", "thanks", "regards",
        "forwarded", "original", "sent", "email", "attached"
    }

    return name not in bad


def safe_graph_to_df(graph):
    rows = [
        {
            "source": s,
            "target": t,
            "weight": d.get("weight", 1),
            "relation_source": d.get("relation_source", ""),
        }
        for s, t, d in graph.edges(data=True)
    ]

    return pd.DataFrame(rows).sort_values(
        by=["weight", "source", "target"],
        ascending=[False, True, True],
    ) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "relation_source"]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/emails.csv")
    parser.add_argument("--nrows", type=int, default=1000)
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/emails_with_ner.csv"),
        help="Path for preprocessed emails enriched with relation_type and mentioned_people.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    t = _step(f"[1/6] Loading CSV  (nrows={args.nrows or 'all'})")
    df = pd.read_csv(args.input)
    if args.nrows > 0:
        df = df.sample(n=args.nrows, random_state=42).reset_index(drop=True)
    print(f"  Rows loaded: {len(df):,}")
    _done(t)

    # ------------------------------------------------------------------
    t = _step("[2/6] Preprocessing emails (parse headers, clean body)")
    df = preprocess_emails(df)
    df = filter_enron_only(df)
    print(f"  Rows after Enron filter: {len(df):,}")
    _done(t)

    # ------------------------------------------------------------------
    t = _step("[3/6] Rule-based relation tagging")
    tqdm.pandas(desc="classify_relation", unit="email")
    df["relation_type"] = df["clean_body"].progress_apply(classify_relation)
    print("\n  Relation type counts:")
    for label, cnt in df["relation_type"].value_counts().items():
        print(f"    {label}: {cnt:,}")
    _done(t)

    # ------------------------------------------------------------------
    t = _step("[4/6] NER — extracting mentioned people (spaCy batch)")
    raw_mentions = extract_people_batch(df["clean_body"].fillna("").tolist())
    df["mentioned_people"] = [
        [p for p in (normalize_person(x) for x in lst) if is_valid_person(p)]
        for lst in raw_mentions
    ]
    total_mentions = df["mentioned_people"].apply(len).sum()
    all_people = [p for row in df["mentioned_people"] for p in row]
    print(f"  Total mentions: {total_mentions:,}")
    print(f"  Unique people:  {len(set(all_people)):,}")
    print(f"  Top 10: {Counter(all_people).most_common(10)}")
    _done(t)

    # ------------------------------------------------------------------
    t = _step("[5/6] Saving enriched emails")
    args.processed_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.processed_output, index=False)
    print(f"  Saved: {args.processed_output}")
    _done(t)

    # ------------------------------------------------------------------
    t = _step("[6/6] Building graphs and saving edge CSVs")
    metadata_graph = build_metadata_graph(df)
    ner_graph = build_ner_graph(df)

    print(f"  Metadata graph — nodes: {metadata_graph.number_of_nodes():,}  edges: {metadata_graph.number_of_edges():,}")
    print(f"  NER graph      — nodes: {ner_graph.number_of_nodes():,}  edges: {ner_graph.number_of_edges():,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "metadata_edges.csv"
    ner_path = OUTPUT_DIR / "ner_edges.csv"
    safe_graph_to_df(metadata_graph).to_csv(metadata_path, index=False)
    safe_graph_to_df(ner_graph).to_csv(ner_path, index=False)
    print(f"  Saved: {metadata_path}")
    print(f"  Saved: {ner_path}")
    _done(t)

    # ------------------------------------------------------------------
    total = time.time() - pipeline_start
    print(f"\n{'='*55}")
    print(f"  Pipeline complete in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
