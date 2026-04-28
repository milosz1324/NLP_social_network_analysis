import networkx as nx

def build_ner_graph(df) -> nx.DiGraph:
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        sender = str(row["sender"]).strip().lower()
        people = row.get("mentioned_people", [])

        if not sender:
            continue

        for person in people:
            if person == sender:
                continue

            if graph.has_edge(sender, person):
                graph[sender][person]["weight"] += 1
            else:
                graph.add_edge(sender, person, weight=1, relation_source="ner")

    return graph
