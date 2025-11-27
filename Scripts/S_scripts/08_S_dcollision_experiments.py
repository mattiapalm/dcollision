# --- requirements ---

from pathlib import Path
from pgmpy.readwrite import BIFReader
import json
import networkx as nx
from py2neo import Graph, Node, Relationship
import time
import pandas as pd
import sys
import pickle

##### ====== #####

# Neo4j connection settings
host = "bolt://localhost:7687"
username = "neo4j"
neo4j_psw = "graph000"

# Base path
BASE = Path(__file__).resolve().parent.parent

# Path to subfolders
synthetic_world_dags_dir   = BASE / "DAGs/Synthetic_dags"
inputs_dir = BASE / "Results/Inputs"
all_runtimes_dir = BASE /"Results/Runtimes/All_runtimes"

# Get queries
sys.path.append(str(BASE / "Scripts"))
from queries import (
    query_dcollision_reset,
    query_dcollision_partial_reset,
    query_dcollision_1of4,
    query_dcollision_2of4,
    query_dcollision_3of4,
    query_dcollision_4of4,
    query_dcollision_4of4_1,
    query_dcollision_4of4_2,
    query_dcollision_4of4_3,
    query_dcollision_4of4_apoc
)


##### Retrieve DAGs' path ####

names_of_graphs = ['SACHS', 'CHILD', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN', 'REDUCEDCOVID', 'COVID', 'CNSAMPLEDAG']

path_to_sachs = real_world_dags_dir / "sachs.bif"
path_to_child = real_world_dags_dir / "child.bif"
path_to_barley = real_world_dags_dir / "barley.bif"
path_to_win95pts = real_world_dags_dir / "win95pts.bif"
path_to_link = real_world_dags_dir / "link.bif"
path_to_munin = real_world_dags_dir / "munin.bif"
path_to_covid19 = real_world_dags_dir / "covid19.json"
path_to_smallcovid19 = real_world_dags_dir / "smallcovid19.json"
path_to_reducedcovid19 = real_world_dags_dir / "reducedcovid.json"
path_to_cnsampledag = real_world_dags_dir / "cnsampledag.json"

all_paths_to_dags = [path_to_sachs,
                     path_to_child,
                     path_to_barley,
                     path_to_win95pts,
                     path_to_link,
                     path_to_munin,
                     path_to_reducedcovid19,
                     path_to_covid19,
                     path_to_cnsampledag]

names_of_graphs = ['REDUCEDCOVID']
all_paths_to_dags = [path_to_reducedcovid19]

data_files = dict(zip(names_of_graphs, all_paths_to_dags))

########

with open(inputs_dir / "X_inputs.pkl", "rb") as f:
    X_inputs = pickle.load(f)

with open(inputs_dir / "Z_inputs.pkl", "rb") as f:
    Z_inputs = pickle.load(f)

#######

all_runtimes_T_dict = {}
all_runtimes_Qn_dict = {}
all_runtimes_tot_n_dict = {}
all_runtimes_Qa_dict = {}
all_runtimes_tot_a_dict = {}

# all_mean_runtimes_T_dict = {}
# all_mean_runtimes_Q_dict = {}
# all_mean_runtimes_tot_dict = {}

# Y_all_dict = {}

for name in names_of_graphs:
    all_runtimes_T_dict[name] = {}
    all_runtimes_Qn_dict[name] = {}
    all_runtimes_tot_n_dict[name] = {}
    all_runtimes_Qa_dict[name] = {}
    all_runtimes_tot_a_dict[name] = {}
    # all_runtimes_tot_a_dict[name] = {}
    # all_mean_runtimes_T_dict[name] = pd.DataFrame()
    # all_mean_runtimes_Q_dict[name] = pd.DataFrame()
    # all_mean_runtimes_tot_dict[name] = pd.DataFrame()
    #Y_all_dict[name] = {}

for name in names_of_graphs:
    if name in ['SACHS', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN']:
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
    G.remove_nodes_from(G.nodes() - largest_cc)
    
    print('The graph', name, 'has', len(G.nodes()), 'nodes and', len(G.edges()), 'edges')

    # === 3. Connect to Neo4j ===
    
    graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "graph000"))
    
    # Clear database 
    graph_db.delete_all()
    
    # === 4. Push nodes and edges into Neo4j ===
    # Create nodes
    for node in G.nodes():
        graph_db.merge(Node("Variable", name=node), "Variable", "name")
    
    # Create edges
    for u, v in G.edges():
        node_u = graph_db.nodes.match("Variable", name=u).first()
        node_v = graph_db.nodes.match("Variable", name=v).first()
        rel = Relationship(node_u, "CAUSES", node_v)
        graph_db.merge(rel)
    
# -------

    N_nodes = len(G.nodes())
    ub = N_nodes-1             # Upper bound on the number of edges in a simple path
    
    # Obtain the DAG-specific query
    query_dcollision_4of4_complete = (   query_dcollision_4of4_1 + str(ub)
                                       + query_dcollision_4of4_2 + str(ub)
                                       + query_dcollision_4of4_3 )
    
    X_inputs_current, Z_inputs_current = X_inputs[name], Z_inputs[name]
    range_X, range_Z = [], []
    
    for k in X_inputs_current.keys():
        range_X.append(k[0])
        range_Z.append(k[1])
    range_X = list(set(range_X)); range_X.sort()
    range_Z = list(set(range_Z)); range_Z.sort()
    
    all_pairs = []
    
    for card_X in range_X:
        for card_Z in range_Z:
            card_union = card_X + card_Z
            if card_union < N_nodes:
                all_pairs.append((card_X, card_Z))
                
    
    # dict_XZ = {(i,j):0 for i in range_X for j in range_Z if i+j < N_nodes}
    # temp_df = pd.DataFrame(
    #     [(i, j, val) for (i, j), val in dict_XZ.items()],
    #     columns=["|X|", "|Z|", "value"]
    # )
    
    # rts_df = temp_df.pivot(index="|X|", columns="|Z|", values="value")
    
    # mean_runtimes_T = rts_df.copy()
    # mean_runtimes_Q = rts_df.copy()
    # mean_runtimes_tot = rts_df.copy()
    
    tot_pairs = len(all_pairs)
    n_pair=1
    
    graph_db.run(query_dcollision_reset)
    saved_op = []
    for pair in all_pairs:
        # in range_X:
        # for card_Z in range_Z:
        #     card_union = card_X + card_Z
        #     if card_union < N_nodes:
                
                X_instances = X_inputs_current[pair]
                Z_instances = Z_inputs_current[pair]
                
                T_rts = []
                Qn_rts, Qa_rts = [], []
                tot_n_rts, tot_a_rts = [], []
                
                # Y_all_list = []
                
                its = len(X_instances)
                #its = 1000
                
                for h in range(its):
                    
                    print(name, n_pair, '/', tot_pairs, '|X|:', card_X, '|Z|:', card_Z, h+1, 'MM')
                    
                    X, Z = X_instances[h], Z_instances[h]
                    
                    params_T = {"Z_names": Z}
                    params_Q = {"X_names": X, "Z_names": Z}
                    
                    #=== Mattia's method for P2
                    
                    # Run the algorithm
                    
                    start_T = time.time()
                    graph_db.run(query_dcollision_1of4, parameters=params_T)
                    graph_db.run(query_dcollision_2of4)
                    graph_db.run(query_dcollision_3of4)
                    end_T = time.time()
                    Y_all = graph_db.run(query_dcollision_4of4_complete, parameters=params_Q).evaluate()
                    end_Qn = time.time()
                    graph_db.run(query_dcollision_partial_reset)
                    start_Qa = time.time()
                    Y_all = graph_db.run(query_dcollision_4of4_apoc, parameters=params_Q).evaluate()
                    end_Qa = time.time()
                    graph_db.run(query_dcollision_reset)
                    
                    # Y_con = list(G.nodes() - Y_all - set(X) - set(Z))
                    
                    # Y_all_list.append(Y_all)
                
                    # Save the runtimes
                    T_rt = end_T - start_T; T_rts.append(T_rt)
                    Qn_rt = end_Qn - end_T; Qn_rts.append(Qn_rt)
                    tot_n_rt = T_rt + Qn_rt; tot_n_rts.append(tot_n_rt)
                    Qa_rt = end_Qa - start_Qa; Qa_rts.append(Qa_rt)
                    tot_a_rt = T_rt + Qa_rt; tot_a_rts.append(tot_a_rt)
                    
                    if Qn_rt > 1: saved_op.append([(X, Z), Qn_rt])
                
                all_runtimes_T_dict[name][pair] = T_rts
                all_runtimes_Qn_dict[name][pair] = Qn_rts
                all_runtimes_tot_n_dict[name][pair] = tot_n_rts
                all_runtimes_Qa_dict[name][pair] = Qa_rts
                all_runtimes_tot_a_dict[name][pair] = tot_a_rts
                #Y_all_dict[name][pair] = Y_all_list
                
                # mean_T_rt = np.mean(T_rts)
                # mean_Q_rt = np.mean(Q_rts)
                # mean_tot = np.mean(tot_rts)
                
                # mean_runtimes_T.loc[card_X,card_Z] = mean_T_rt 
                # mean_runtimes_Q.loc[card_X,card_Z] = mean_Q_rt
                # mean_runtimes_tot.loc[card_X,card_Z] = mean_tot
                
                n_pair += 1
        
    # all_mean_runtimes_T_dict[name] = mean_runtimes_T
    # all_mean_runtimes_Q_dict[name] = mean_runtimes_Q
    # all_mean_runtimes_tot_dict[name] = mean_runtimes_tot
    
# # Save to disk
# with open("all_mean_runtimes_T_dict.pkl", "wb") as f:
#     pickle.dump(all_mean_runtimes_T_dict, f)

# with open("all_mean_runtimes_Q_dict.pkl", "wb") as f:
#     pickle.dump(all_mean_runtimes_Q_dict, f)

# with open("all_mean_runtimes_tot_dict.pkl", "wb") as f:
#     pickle.dump(all_mean_runtimes_tot_dict, f)
    
# Save to disk
with open(all_runtimes_dir / "R_all_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(all_runtimes_T_dict, f)

with open(all_runtimes_dir / "R_all_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(all_runtimes_Qn_dict, f)

with open(all_runtimes_dir / "R_all_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(all_runtimes_tot_n_dict, f)
    
with open(all_runtimes_dir / "R_all_runtimes_Qa_dict.pkl", "wb") as f:
    pickle.dump(all_runtimes_Qa_dict, f)

with open(all_runtimes_dir / "R_all_runtimes_tot_a_dict.pkl", "wb") as f:
    pickle.dump(all_runtimes_tot_a_dict, f)
    
    
    
#with open("Y_all_smallcovid.pkl", "wb") as f:
    #pickle.dump(Y_all_dict['SMALLCOVID'], f)
    
    
                
            
                    