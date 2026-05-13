from pathlib import Path
import pandas as pd
import networkx as nx
import re

from src.config import TABLES_DIR


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()

    name = name.replace("@enron.com", "")
    name = name.split("@")[0]
    name = name.replace("_", " ")
    name = name.replace(".", " ")

    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def load_graph(edges_path: str | Path) -> nx.DiGraph:
    df = pd.read_csv(edges_path)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        s = normalize(row["source"])
        t = normalize(row["target"])

        if not s or not t or s == t:
            continue

        w = float(row.get("weight", 1))

        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    return G


def compute_betweenness(G: nx.DiGraph) -> dict:
    return nx.betweenness_centrality(G, weight="weight")


def find_hidden_brokers(meta_scores, ner_scores, top_n=20):
    rows = []

    all_nodes = set(meta_scores) | set(ner_scores)

    for node in all_nodes:
        meta = meta_scores.get(node, 0.0)
        ner = ner_scores.get(node, 0.0)

        rows.append({
            "node": node,
            "metadata_bc": meta,
            "ner_bc": ner,
            "difference": ner - meta
        })

    df = pd.DataFrame(rows)

    return df.sort_values("difference", ascending=False).head(top_n)


def main():
    metadata_path = TABLES_DIR / "metadata_edges.csv"
    ner_path = TABLES_DIR / "ner_edges.csv"

    print("Loading graphs...")
    G_meta = load_graph(metadata_path)
    G_ner = load_graph(ner_path)

    print("Computing centrality...")
    meta_scores = compute_betweenness(G_meta)
    ner_scores = compute_betweenness(G_ner)

    print("Finding hidden brokers...")
    hidden = find_hidden_brokers(meta_scores, ner_scores)

    print("\n=== HIDDEN BROKERS ===")
    print(hidden.to_string(index=False))

    out = TABLES_DIR / "hidden_brokers.csv"
    hidden.to_csv(out, index=False)

    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
