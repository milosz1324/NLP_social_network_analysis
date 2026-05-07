import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import filter_enron_only, preprocess_emails
from src.ner import extract_people, clean_people
from src.graph_builder import build_metadata_graph, graph_to_edges_df
from src.graph_ner_builder import build_ner_graph
from src.alias_resolution import resolve_people


OUTPUT_DIR = Path("outputs/tables")

# ---------------------------
# NORMALIZATION HELPERS
# ---------------------------
def normalize_enron_name(name: str) -> str:
    """Handle Enron-style names like 'gorny/hou' → 'gorny'"""
    name = name.lower().strip()

    if "/" in name:
        name = name.split("/")[0]

    return name.strip()


def to_surname(name: str) -> str:
    parts = name.strip().lower().split()
    if len(parts) == 0:
        return name
    return parts[-1]


def is_valid_person(name: str) -> bool:
    """Lightweight filter (NOT too aggressive!)"""
    name = name.strip().lower()

    if len(name) < 3:
        return False

    bad_words = {
        "subject", "corporation", "email", "message",
        "please", "thanks", "regards", "attached",
        "forwarded", "original", "sent"
    }

    if name in bad_words:
        return False

    return True


# ---------------------------
# SAFE GRAPH EXPORT
# ---------------------------
def safe_graph_to_df(graph):
    rows = []
    for source, target, data in graph.edges(data=True):
        rows.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1),
            "relation_source": data.get("relation_source", ""),
        })

    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight", "relation_source"])

    return pd.DataFrame(rows).sort_values(
        by=["weight", "source", "target"],
        ascending=[False, True, True],
    )


# ---------------------------
# CLI
# ---------------------------
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


# ---------------------------
# MAIN
# ---------------------------
def main():
    args = parse_args()

    df = pd.read_csv(
        args.input,
        nrows=None if args.nrows == 0 else args.nrows
    )

    print(f"Loaded rows: {len(df)}")

    # ---------------------------
    # PREPROCESSING
    # ---------------------------
    df = preprocess_emails(df)

    df = filter_enron_only(df)

    print(f"After Enron filter: {len(df)} rows")

    # ---------------------------
    # NER EXTRACTION
    # ---------------------------
    df["mentioned_people"] = df["clean_body"].apply(
        lambda x: clean_people(extract_people(x))
    )

    # ---------------------------
    # NORMALIZE (Enron-specific)
    # ---------------------------
    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: [normalize_enron_name(name) for name in lst]
    )

    # ---------------------------
    # FILTER
    # ---------------------------
    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: [name for name in lst if is_valid_person(name)]
    )

    # DEBUG (important)
    print("\nSample mentioned_people:")
    print(df["mentioned_people"].head(5).to_list())

    # ---------------------------
    # ALIAS RESOLUTION
    # ---------------------------
    all_names = set(
        name
        for sublist in df["mentioned_people"]
        for name in sublist
    )

    alias_map = resolve_people(all_names)

    # ---------------------------
    # SURNAME REDUCTION
    # ---------------------------
    df["mentioned_people"] = df["mentioned_people"].apply(
        lambda lst: list({
            to_surname(alias_map.get(name, name))
            for name in lst
        })
    )

    print("NER completed (clean + normalize + alias + surname)")

    # ---------------------------
    # GRAPHS
    # ---------------------------
    metadata_graph = build_metadata_graph(df)
    ner_graph = build_ner_graph(df)

    print("\n=== GRAPH SUMMARY ===")

    print("\n[Metadata Graph]")
    print("Nodes:", metadata_graph.number_of_nodes())
    print("Edges:", metadata_graph.number_of_edges())

    print("\n[NER Graph]")
    print("Nodes:", ner_graph.number_of_nodes())
    print("Edges:", ner_graph.number_of_edges())

    # ---------------------------
    # SAVE
    # ---------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = safe_graph_to_df(metadata_graph)
    metadata_path = OUTPUT_DIR / "metadata_edges.csv"
    metadata_df.to_csv(metadata_path, index=False)

    ner_df = safe_graph_to_df(ner_graph)
    ner_path = OUTPUT_DIR / "ner_edges.csv"
    ner_df.to_csv(ner_path, index=False)

    print("\n=== FILES SAVED ===")
    print("Metadata edges:", metadata_path)
    print("NER edges:", ner_path)


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":
    main()
