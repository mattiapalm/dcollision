from pathlib import Path
import networkx as nx
import json
from networkx.readwrite import json_graph
from py2neo import Graph, Node, Relationship


# Base path
BASE = Path(__file__).resolve().parent.parent

# Path to subfolders
real_world_dags_dir   = BASE / "Real_world_dags"

# Connect to your database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "graph000"))

# Run a query
data = graph.run("MATCH (a)-[r]->(b) RETURN a, r, b").data()

print(data[:3])  # print first few results


G = nx.DiGraph()

for rel in graph.run("MATCH (a)-[r]->(b) RETURN a.name AS source, b.name AS target"):
    G.add_edge(rel["source"], rel["target"])
    
    
# Convert to JSON-compatible dict
data = json_graph.node_link_data(G, edges="links")

# Save to JSON file
with open(real_world_dags_dir / "reducedcovid.json", "w") as f:
    json.dump(data, f, indent=2)
    
    
with open(real_world_dags_dir / "reducedcovid.json") as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)