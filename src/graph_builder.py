from collections.abc import Iterable
from pathlib import Path
import argparse

import networkx as nx
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TABLES_DIR


def _split_recipients(value: object) -> list[str]:
    if pd.isna(value):
        return []

    return [
        item.strip().lower()
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    ]


def build_metadata_graph(
    df: pd.DataFrame,
    sender_col: str = "sender",
    recipients_col: str = "recipients",
    ) -> nx.DiGraph:
    """Build a weighted directed graph from sender-recipient metadata."""
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        sender = str(row.get(sender_col, "")).strip().lower()
        if not sender:
            continue

        for recipient in _split_recipients(row.get(recipients_col)):
            if graph.has_edge(sender, recipient):
                graph[sender][recipient]["weight"] += 1
            else:
                graph.add_edge(sender, recipient, weight=1, relation_source="metadata")

    return graph


def add_mentioned_people_edges(
    graph: nx.DiGraph,
    sender: str,
    mentioned_people: Iterable[str],
    ) -> None:
    """Add sender-to-mentioned-person edges extracted from message content."""
    sender = sender.strip().lower()
    if not sender:
        return

    for person in mentioned_people:
        target = person.strip().lower()
        if not target or target == sender:
            continue

        if graph.has_edge(sender, target):
            graph[sender][target]["weight"] += 1
        else:
            graph.add_edge(sender, target, weight=1, relation_source="ner")


def graph_to_edges_df(graph: nx.DiGraph) -> pd.DataFrame:
    """Convert graph edges to a dataframe that can be saved as CSV."""
    rows = []
    for source, target, data in graph.edges(data=True):
        rows.append(
            {
                "source": source,
                "target": target,
                "weight": data.get("weight", 1),
                "relation_source": data.get("relation_source", ""),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["weight", "source", "target"],
        ascending=[False, True, True],
    )


def print_graph_summary(graph: nx.DiGraph, top_n: int = 10) -> None:
    """Print compact graph statistics and strongest edges."""
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    if graph.number_of_edges() == 0:
        return

    print(f"\nTop {top_n} edges by weight:")
    top_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: edge[2].get("weight", 1),
        reverse=True,
    )[:top_n]

    for source, target, data in top_edges:
        print(f"- {source} -> {target}: weight={data.get('weight', 1)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sender-to-recipient metadata graph from preprocessed emails."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_preprocessed.csv",
        help="Path to preprocessed emails CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "metadata_edges.csv",
        help="Path where graph edges CSV should be saved.",
    )
    parser.add_argument(
        "--sender-col",
        default="sender",
        help="Name of the sender column.",
    )
    parser.add_argument(
        "--recipients-col",
        default="recipients",
        help="Name of the recipients column.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of strongest edges to print.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print graph summary without saving edges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    graph = build_metadata_graph(
        df,
        sender_col=args.sender_col,
        recipients_col=args.recipients_col,
    )
    print_graph_summary(graph, top_n=args.top_n)

    if not args.no_save:
        edges_df = graph_to_edges_df(graph)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(args.output, index=False)
        print(f"\nSaved graph edges to: {args.output}")


if __name__ == "__main__":
    main()
