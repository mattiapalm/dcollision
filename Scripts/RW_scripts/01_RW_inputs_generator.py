########--------------- Requirements ---------------########

from pathlib import Path
from pgmpy.readwrite import BIFReader
import json
import networkx as nx
import random
import pandas as pd
import pickle

##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolder
RW_dags_dir   = BASE / "DAGs/Real_world_dags"

# === 1. Load DAG from BIF file ===

names_of_graphs = ['SACHS', 'CHILD', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN', 'SMALLCOVID', 'REDUCEDCOVID', 'COVID', 'CNSAMPLEDAG']

path_to_sachs = RW_dags_dir / "sachs.bif"
path_to_child = RW_dags_dir / "child.bif"
path_to_barley = RW_dags_dir / "barley.bif"
path_to_win95pts = RW_dags_dir / "win95pts.bif"
path_to_link = RW_dags_dir / "link.bif"
path_to_munin = RW_dags_dir / "munin.bif"
path_to_smallcovid19 = RW_dags_dir / "smallcovid19.json"
path_to_reducedcovid = RW_dags_dir / "reducedcovid.json"
path_to_covid19 = RW_dags_dir / "covid19.json"
path_to_cnsampledag = RW_dags_dir / "cnsampledag.json"

all_paths_to_data = [path_to_sachs,
                     path_to_child,
                     path_to_barley,
                     path_to_win95pts,
                     path_to_link,
                     path_to_munin,
                     path_to_smallcovid19,
                     path_to_reducedcovid,
                     path_to_covid19,
                     path_to_cnsampledag]

data_files = dict(zip(names_of_graphs, all_paths_to_data))


RW_X_inputs, Z_inputs = dict(), dict()
range_10 = list(range(10))
range_frac = [ r / 10 for r in range_10]

its = 2

for name in names_of_graphs:
    if name in ['SACHS', 'CHILD', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN']:
        reader = BIFReader(data_files[name])
        model = reader.get_model()
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())
    else:
        with open(data_files[name]) as f:
            data = json.load(f)
        G = nx.DiGraph()
        G = nx.json_graph.node_link_graph(data, edges="links")
    
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    
    to_be_removed = G.nodes() - largest_cc
    G.remove_nodes_from(to_be_removed)
    
    print('The graph', name, 'has', len(G.nodes()), 'nodes and', len(G.edges()), 'edges')

    # -------

    N_nodes = len(G.nodes())
    
    range_ = [int(N_nodes * f) for f in range_frac]
    to_be_added_X = [1,2,5,10,20,50]
    to_be_added_Z = [1,2,5,10,20,50]
    for ind in range(len(to_be_added_X)):
        a = to_be_added_X[ind]
        if a >= int(N_nodes*0.9):
            to_be_added_X = to_be_added_X[:ind]
            to_be_added_Z = to_be_added_Z[:ind]
            break
    
    range_X, range_Z = range_.copy(), range_.copy()
    range_X = range_X + to_be_added_X; range_Z = range_Z + to_be_added_Z
    range_Z = list(set(range_Z)); range_Z.sort()
    range_X = list(set(range_X)); range_X.remove(0); range_X.sort()
    
    dict_XZ = {(i,j):0 for i in range_X for j in range_Z if i+j < N_nodes}
    temp_df = pd.DataFrame(
        [(i, j, val) for (i, j), val in dict_XZ.items()],
        columns=["|X|", "|Z|", "value"]
    )
    
    rts_df = temp_df.pivot(index="|X|", columns="|Z|", values="value")

    tot_pairs = rts_df.count().sum()
    
    RW_X_inputs[name], Z_inputs[name] = {}, {}
    for card_X in range_X:
        for card_Z in range_Z:
            card_union = card_X + card_Z
            if card_union < N_nodes:
                
                X_instances, Z_instances = {}, {}
                for h in range(its):
                    sample_nodes = random.sample(list(G.nodes()), card_union)
                    X_instances[h] = list(sample_nodes[:card_X])
                    Z_instances[h] = list(sample_nodes[card_X:])
                    
                RW_X_inputs[name][(card_X, card_Z)] = X_instances
                Z_inputs[name][(card_X, card_Z)] = Z_instances

# Directory where to save
save_dir = BASE / "Results/Inputs"
save_dir.mkdir(exist_ok=True)  # does not create it if it does not exist

# Save files
with open(save_dir / "RW_X_inputs.pkl", "wb") as f:
    pickle.dump(RW_X_inputs, f)

with open(save_dir / "RW_Z_inputs.pkl", "wb") as f:
    pickle.dump(Z_inputs, f)           