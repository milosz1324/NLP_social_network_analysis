from pathlib import Path
import argparse

import networkx as nx
import pandas as pd

from src.config import TABLES_DIR


def load_graph_from_edges(
    path: str | Path,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
) -> nx.DiGraph:
    """Load a weighted directed graph from an edge-list CSV."""
    edges = pd.read_csv(path)
    graph = nx.DiGraph()

    for _, row in edges.iterrows():
        source = str(row[source_col]).strip().lower()
        target = str(row[target_col]).strip().lower()
        if not source or not target:
            continue

        weight = float(row.get(weight_col, 1))
        graph.add_edge(
            source,
            target,
            weight=weight,
            distance=1 / weight if weight else 1,
            relation_source=row.get("relation_source", ""),
        )

    return graph


def betweenness_ranking(graph: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """Return top nodes by betweenness centrality."""
    scores = nx.betweenness_centrality(graph, weight="distance")
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return pd.DataFrame(ranking, columns=["node", "betweenness_centrality"])


def node_metrics_ranking(graph: nx.DiGraph, top_n: int = 20) -> pd.DataFrame:
    """Return node-level metrics useful for communication network analysis."""
    betweenness = nx.betweenness_centrality(graph, weight="distance")
    in_degree = dict(graph.in_degree(weight="weight"))
    out_degree = dict(graph.out_degree(weight="weight"))
    degree = dict(graph.degree(weight="weight"))

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "node": node,
                "betweenness_centrality": betweenness.get(node, 0.0),
                "weighted_in_degree": in_degree.get(node, 0.0),
                "weighted_out_degree": out_degree.get(node, 0.0),
                "weighted_degree": degree.get(node, 0.0),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "betweenness_centrality",
                "weighted_degree",
                "weighted_out_degree",
                "node",
            ],
            ascending=[False, False, False, True],
        )
        .head(top_n)
    )


def graph_summary(graph: nx.Graph) -> dict[str, int | float]:
    """Return basic graph size metrics."""
    weak_components = list(nx.weakly_connected_components(graph))
    largest_component_size = max((len(component) for component in weak_components), default=0)

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "weakly_connected_components": len(weak_components),
        "largest_weak_component_size": largest_component_size,
    }


def save_summary(summary: dict[str, int | float], output_path: str | Path) -> None:
    """Save graph summary metrics as a two-column CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
        output_path,
        index=False,
    )


def print_summary(summary: dict[str, int | float]) -> None:
    """Print graph summary metrics."""
    print("Graph summary:")
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"- {metric}: {value:.6f}")
        else:
            print(f"- {metric}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze graph metrics for the Enron communication network."
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=TABLES_DIR / "metadata_edges.csv",
        help="Path to graph edge-list CSV.",
    )
    parser.add_argument(
        "--ranking-output",
        type=Path,
        default=TABLES_DIR / "metadata_node_metrics.csv",
        help="Path where node metrics ranking should be saved.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=TABLES_DIR / "metadata_graph_summary.csv",
        help="Path where graph summary should be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top nodes to save in the ranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.edges.exists():
        raise FileNotFoundError(f"Edges file not found: {args.edges}")

    graph = load_graph_from_edges(args.edges)
    summary = graph_summary(graph)
    ranking = node_metrics_ranking(graph, top_n=args.top_n)

    print_summary(summary)
    print(f"\nTop {len(ranking)} nodes by betweenness centrality:")
    print(ranking.to_string(index=False))

    args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.ranking_output, index=False)
    save_summary(summary, args.summary_output)

    print(f"\nSaved node metrics to: {args.ranking_output}")
    print(f"Saved graph summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
