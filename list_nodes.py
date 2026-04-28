from pathlib import Path
import argparse
import pandas as pd


def extract_nodes(edges_path: Path) -> pd.DataFrame:
    df = pd.read_csv(edges_path)

    # zbierz wszystkie node'y
    nodes = set(df["source"]).union(set(df["target"]))

    nodes_df = pd.DataFrame(sorted(nodes), columns=["node"])
    return nodes_df


def parse_args():
    parser = argparse.ArgumentParser(description="List graph nodes sorted alphabetically")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/tables/metadata_edges.csv"),
        help="Path to edges CSV"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/nodes_list.csv"),
        help="Where to save node list"
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Print nodes to console"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")

    nodes_df = extract_nodes(args.input)

    # save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(args.output, index=False)

    print(f"Saved {len(nodes_df)} nodes to {args.output}")

    # optional print
    if args.print:
        print("\n=== NODES ===")
        print(nodes_df.to_string(index=False))


if __name__ == "__main__":
    main()
