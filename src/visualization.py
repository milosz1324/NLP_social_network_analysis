from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def save_static_graph(
    graph: nx.Graph,
    output_path: str | Path,
    max_nodes: int = 80,
) -> None:
    """Save a readable static graph preview for reports."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if graph.number_of_nodes() > max_nodes:
        nodes = sorted(graph.degree, key=lambda item: item[1], reverse=True)[:max_nodes]
        graph = graph.subgraph([node for node, _ in nodes]).copy()

    plt.figure(figsize=(14, 10))
    positions = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, positions, node_size=120, alpha=0.8)
    nx.draw_networkx_edges(graph, positions, arrows=True, alpha=0.25)
    nx.draw_networkx_labels(graph, positions, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
