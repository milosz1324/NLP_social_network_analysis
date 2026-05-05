from pathlib import Path
import argparse
import pandas as pd
import networkx as nx


def canonicalize(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()
    name = name.replace("@enron.com", "")
    name = name.replace(".", " ")
    name = " ".join(name.split())
    return name


def build_ner_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in df.iterrows():

        sender = canonicalize(row.get("sender", ""))
        if not sender:
            continue

        mentions = row.get("mentioned_people", [])

        if not isinstance(mentions, (list, tuple)):
            continue

        for m in mentions:

            target = canonicalize(m)

            if not target:
                continue

            if target == sender:
                continue

            if G.has_edge(sender, target):
                G[sender][target]["weight"] += 1
            else:
                G.add_edge(
                    sender,
                    target,
                    weight=1,
                    relation_source="ner",
                )

    return G


def graph_to_edges_df(G: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {
            "source": s,
            "target": t,
            "weight": d.get("weight", 1),
            "relation_source": "ner",
        }
        for s, t, d in G.edges(data=True)
    ]

    return pd.DataFrame(rows).sort_values(
        by=["weight", "source", "target"],
        ascending=[False, True, True],
    ) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "relation_source"]
    )


def save_graph(G: nx.DiGraph, output_path: Path):
    df = graph_to_edges_df(G)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved NER edges → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default="outputs/data/ner_input.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="outputs/tables/ner_edges.csv"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"Loaded rows: {len(df)}")

    G = build_ner_graph(df)

    print("\n=== NER GRAPH SUMMARY ===")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    save_graph(G, args.output)


if __name__ == "__main__":
    main()
