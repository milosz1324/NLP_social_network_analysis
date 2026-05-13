from pathlib import Path
import argparse
import networkx as nx
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TABLES_DIR


def clean_email(email: str) -> str:
    if not isinstance(email, str):
        return ""
    return email.strip().lower()


def _split_recipients(value: object) -> list[str]:
    if pd.isna(value):
        return []

    parts = str(value).replace(";", ",").split(",")

    cleaned = []
    for item in parts:
        item = clean_email(item)

        if not item:
            continue

        if "@" not in item:
            continue

        cleaned.append(item)

    return cleaned


def build_metadata_graph(
    df: pd.DataFrame,
    sender_col: str = "sender",
    recipients_col: str = "recipients",
) -> nx.DiGraph:

    G = nx.DiGraph()

    for _, row in df.iterrows():

        sender = clean_email(row.get(sender_col, ""))
        if not sender or "@" not in sender:
            continue

        recipients = _split_recipients(row.get(recipients_col))

        for recipient in recipients:

            if recipient == sender:
                continue

            relation = row.get("relation_type", "other")

            if G.has_edge(sender, recipient):
                G[sender][recipient]["weight"] += 1
            else:
                G.add_edge(
                    sender,
                    recipient,
                    weight=1,
                    relation_source="metadata",
                    relation_type=relation,
                )

    return G


def graph_to_edges_df(G: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {
            "source": s,
            "target": t,
            "weight": d.get("weight", 1),
            "relation_source": d.get("relation_source", ""),
            "relation_type": d.get("relation_type", "unknown"),
        }
        for s, t, d in G.edges(data=True)
    ]

    return pd.DataFrame(rows).sort_values(
        by=["weight", "source", "target"],
        ascending=[False, True, True],
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_preprocessed.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "metadata_edges.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    G = build_metadata_graph(df)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    df_out = graph_to_edges_df(G)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
