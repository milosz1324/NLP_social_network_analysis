import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_emails
from src.ner import extract_people, clean_people
from src.graph_builder import build_metadata_graph, graph_to_edges_df
from src.graph_ner_builder import build_ner_graph
from src.alias_resolution import resolve_people


OUTPUT_DIR = Path("outputs/tables")

def to_surname(name: str) -> str:
    parts = name.strip().lower().split()
    if len(parts) == 0:
        return name
    return parts[-1]  # surname only

def parse_args():
    parser = argparse.ArgumentParser(description="Enron NLP + SNA pipeline")

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/emails.csv",
        help="Path to raw emails CSV"
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=1000,
        help="Number of rows to load (0 = full dataset)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input,
        nrows=None if args.nrows == 0 else args.nrows
    )

    print(f"Loaded rows: {len(df)}")

    df = preprocess_emails(df)

    df["mentioned_people"] = df["clean_body"].apply(
        lambda x: clean_people(extract_people(x))
    )

    all_names = set(
        name
        for sublist in df["mentioned_people"]
        for name in sublist
    )

    alias_map = resolve_people(all_names)

    # 6. APPLY NORMALIZATION + SURNAME REDUCTION
    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: [
            to_surname(alias_map[name])
            for name in lst
        ]
    )

    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: list(set(lst))
    )

    print("NER completed (with alias + surname normalization)")

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

    metadata_df = graph_to_edges_df(metadata_graph)
    metadata_path = OUTPUT_DIR / "metadata_edges.csv"
    metadata_df.to_csv(metadata_path, index=False)

    ner_df = graph_to_edges_df(ner_graph)
    ner_path = OUTPUT_DIR / "ner_edges.csv"
    ner_df.to_csv(ner_path, index=False)

    print("\n=== FILES SAVED ===")
    print("Metadata edges:", metadata_path)
    print("NER edges:", ner_path)

if __name__ == "__main__":
    main()
