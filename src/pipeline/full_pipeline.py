import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

from src.preprocess.preprocessing import filter_enron_only, preprocess_emails
from src.nlp.ner import extract_people, clean_people
from src.graphs.graph_builder import build_metadata_graph
from src.graphs.graph_ner_builder import build_ner_graph
from src.nlp.relation_classifier import classify_relation

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

    df = pd.read_csv(args.input)

    if args.nrows > 0:
        df = df.sample(n=args.nrows, random_state=42).reset_index(drop=True)

    print(f"Loaded rows: {len(df)}")

    df = preprocess_emails(df)
    df = filter_enron_only(df)

    print(f"After Enron filter: {len(df)} rows")

    df["relation_type"] = df["clean_body"].apply(classify_relation)

    print("\nRelation types:")
    print(df["relation_type"].value_counts())

    df["mentioned_people"] = df["clean_body"].apply(
        lambda x: clean_people(extract_people(x))
    )

    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: [normalize_person(p) for p in lst]
    )

    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: [p for p in lst if is_valid_person(p)]
    )

    total_mentions = df["mentioned_people"].apply(len).sum()
    print("\nTotal detected people mentions:", total_mentions)

    print("\nNon-empty examples:")
    print(df[df["mentioned_people"].str.len() > 0][["mentioned_people"]].head(10))

    all_people = [p for row in df["mentioned_people"] for p in row]

    print("\nUnique people found:", len(set(all_people)))
    print("\nTop mentioned people:")
    print(Counter(all_people).most_common(20))

    print("NER completed (clean + normalize)")

    args.processed_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.processed_output, index=False)
    print("Enriched emails:", args.processed_output)

    metadata_graph = build_metadata_graph(df)
    ner_graph = build_ner_graph(df)

    print("\n=== GRAPH SUMMARY ===")

    print("\n[Metadata Graph]")
    print("Nodes:", metadata_graph.number_of_nodes())
    print("Edges:", metadata_graph.number_of_edges())

    print("\n[NER Graph]")
    print("Nodes:", ner_graph.number_of_nodes())
    print("Edges:", ner_graph.number_of_edges())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_DIR / "metadata_edges.csv"
    ner_path = OUTPUT_DIR / "ner_edges.csv"

    safe_graph_to_df(metadata_graph).to_csv(metadata_path, index=False)
    safe_graph_to_df(ner_graph).to_csv(ner_path, index=False)

    print("\n=== FILES SAVED ===")
    print("Metadata edges:", metadata_path)
    print("NER edges:", ner_path)


if __name__ == "__main__":
    main()
