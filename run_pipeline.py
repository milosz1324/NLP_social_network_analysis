import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_emails
from src.ner import extract_people, clean_people
from src.graph_builder import build_metadata_graph, graph_to_edges_df
from src.graph_ner_builder import build_ner_graph


OUTPUT_DIR = Path("outputs/tables")


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

    # 1. LOAD DATA
    df = pd.read_csv(
        args.input,
        nrows=None if args.nrows == 0 else args.nrows
    )

    print(f"Loaded rows: {len(df)}")

    # 2. PREPROCESSING
    df = preprocess_emails(df)

    # 3. NER
    df["mentioned_people"] = df["clean_body"].apply(
        lambda x: clean_people(extract_people(x))
    )

    print("NER completed")

    # 4. METADATA GRAPH
    metadata_graph = build_metadata_graph(df)

    # 5. NER GRAPH
    ner_graph = build_ner_graph(df)

    # 6. SUMMARY
    print("\n=== GRAPH SUMMARY ===")

    print("\n[Metadata Graph]")
    print("Nodes:", metadata_graph.number_of_nodes())
    print("Edges:", metadata_graph.number_of_edges())

    print("\n[NER Graph]")
    print("Nodes:", ner_graph.number_of_nodes())
    print("Edges:", ner_graph.number_of_edges())

    # 7. SAVE OUTPUTS (KLUCZOWE DLA RAPORTU)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # metadata edges
    metadata_df = graph_to_edges_df(metadata_graph)
    metadata_path = OUTPUT_DIR / "metadata_edges.csv"
    metadata_df.to_csv(metadata_path, index=False)

    # ner edges
    ner_df = graph_to_edges_df(ner_graph)
    ner_path = OUTPUT_DIR / "ner_edges.csv"
    ner_df.to_csv(ner_path, index=False)

    print("\n=== FILES SAVED ===")
    print("Metadata edges:", metadata_path)
    print("NER edges:", ner_path)


if __name__ == "__main__":
    main()
