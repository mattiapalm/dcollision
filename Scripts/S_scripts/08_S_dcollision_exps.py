########--------------- Requirements ---------------########

import importlib
import subprocess
import sys

def require(package, pip_name=None):
    """
    Try to import a package. If not installed, install it using pip.
    package:   import name
    pip_name:  name used in pip install (if different)
    """
    pip_name = pip_name or package
    try:
        return importlib.import_module(package)
    except ImportError:
        print(f"[INFO] '{package}' not found. Installing '{pip_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(package)
    
Path = require("pathlib").Path
nx = require("networkx")
Graph = require("py2neo", "py2neo").Graph
Node = require("py2neo", "py2neo").Node
Relationship = require("py2neo", "py2neo").Relationship
time = require("time")
pd = require("pandas")
pickle = require("pickle")

##### ====== #####

# Neo4j connection settings
host = "bolt://localhost:7687"
username = "neo4j"
neo4j_psw = "graph000"

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
S_dags_dir   = BASE / "DAGs/Synthetic_dags"
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

graph_names = []

for dag in S_dags_dir.iterdir():
    if dag.is_file() and dag.name != "S_all_graphs_dict.pkl" and dag.suffix == ".pkl":
        graph_names.append(dag.name[:-4])
        
graph_names.sort()

    
########--------------- Initialize the runtimes dictionaries ----------########

all_runtimes_T_dict = {}
all_runtimes_Qn_dict = {}
all_runtimes_tot_n_dict = {}
all_runtimes_Qa_dict = {}
all_runtimes_tot_a_dict = {}

# Y_all_dict = {}

for t in graph_names:
    all_runtimes_T_dict[t] = {}
    all_runtimes_Qn_dict[t] = {}
    all_runtimes_tot_n_dict[t] = {}
    all_runtimes_Qa_dict[t] = {}
    all_runtimes_tot_a_dict[t] = {}
    
########--------------- Upload the inputs files ---------------########

with open(inputs_dir / "S_X_inputs.pkl", "rb") as f:
    S_X_inputs = pickle.load(f)

with open(inputs_dir / "S_Z_inputs.pkl", "rb") as f:
    S_Z_inputs = pickle.load(f)


card_V_list = [10**3, 10**4]#, 10**5, 10**6, 5*10**6]

########--------------- Iterates over the DAGs ---------------########

file = open("S_DC_execution.txt", "w")

for name in graph_names:
    
    with open(S_dags_dir / name+'.pkl', "rb") as f:
        G = pickle.load(f)
        
    N_nodes = len(G.nodes())    # Number of nodes in the DAG

    to_be_printed = f"The graph {name} has {N_nodes} nodes and {len(G.edges())} edges"
    file.write(to_be_printed+'\n')
    file.flush()
    print(to_be_printed)

    ### Connect to Neo4j
    
    graph_db = Graph(host, auth=(username, neo4j_psw))
    
    # Clear database 
    graph_db.delete_all()
    
    ### Push nodes and edges into Neo4j
    
    # Push nodes
    for node in G.nodes():
        graph_db.merge(Node("Variable", name=node), "Variable", "name")
    
    # Push edges
    for u, v in G.edges():
        node_u = graph_db.nodes.match("Variable", name=u).first()
        node_v = graph_db.nodes.match("Variable", name=v).first()
        rel = Relationship(node_u, "CAUSES", node_v)
        graph_db.merge(rel)
        
# -------

    ub = N_nodes-1             # Upper bound on the number of edges in a simple path
    
    # Obtain the DAG-specific query
    query_dcollision_4of4_complete = (   query_dcollision_4of4_1 + str(ub)
                                       + query_dcollision_4of4_2 + str(ub)
                                       + query_dcollision_4of4_3 )
    
    # Retrieve inputs for the current DAG
    S_X_inputs_current, S_Z_inputs_current = S_X_inputs[name], S_Z_inputs[name]
    range_X, range_Z = [], []
    
    # Retrieve the cardinality pairs (|X|, |Z|)
    
    #
    for k in S_X_inputs_current.keys():
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
    
    tot_pairs = len(all_pairs)
    #
    
    n_pair=1   # keeps truck of the current cardinality pair
    
    graph_db.run(query_dcollision_reset)
    saved_op = []

for pair in all_pairs:
            
        card_X, card_Z = pair  # unfolds |X| and |Z|
    
        X_instances = S_X_inputs_current[pair]
        Z_instances = S_Z_inputs_current[pair]
        
        # Initialize the runtimes' lists
        T_rts = []
        Qn_rts, Qa_rts = [], []
        tot_n_rts, tot_a_rts = [], []
        
        its = len(X_instances)   # number of iterations (inputs) for the pair
        
        for h in range(its):
            
            # Takes the input of the current iteration
            X, Z = X_instances[h], Z_instances[h]
            
            params_T = {"Z_names": Z}
            params_Q = {"X_names": X, "Z_names": Z}
            
            ### Execution of the d-collision graph method
            
            # Run the algorithm
            
            # Transformation
            start_T = time.time()
            graph_db.run(query_dcollision_1of4, parameters=params_T)
            graph_db.run(query_dcollision_2of4)
            graph_db.run(query_dcollision_3of4)
            end_T = time.time()
            
            # Native
            graph_db.run(query_dcollision_4of4_complete, parameters=params_Q).evaluate()
            end_Qn = time.time()
            
            # Partial reset
            graph_db.run(query_dcollision_partial_reset)
            
            # APOC
            start_Qa = time.time()
            graph_db.run(query_dcollision_4of4_apoc, parameters=params_Q).evaluate()
            end_Qa = time.time()
            
            # Reset
            graph_db.run(query_dcollision_reset)
            
            # Y_con = list(G.nodes() - Y_all - set(X) - set(Z))
        
            # Save iteration's runtimes
            T_rt = end_T - start_T; T_rts.append(T_rt)
            Qn_rt = end_Qn - end_T; Qn_rts.append(Qn_rt)
            tot_n_rt = T_rt + Qn_rt; tot_n_rts.append(tot_n_rt)
            Qa_rt = end_Qa - start_Qa; Qa_rts.append(Qa_rt)
            tot_a_rt = T_rt + Qa_rt; tot_a_rts.append(tot_a_rt)
            
            all_runtimes_T_dict[name][pair] = T_rts
            all_runtimes_Qn_dict[name][pair] = Qn_rts
            all_runtimes_tot_n_dict[name][pair] = tot_n_rts
            all_runtimes_Qa_dict[name][pair] = Qa_rts
            all_runtimes_tot_a_dict[name][pair] = tot_a_rts
            
            if Qn_rt > 1: saved_op.append([(X, Z), Qn_rt])
        
            # Save to disk
            
            with open(all_runtimes_dir / "S_all_runtimes_T_dict.pkl", "wb") as f:
                pickle.dump(all_runtimes_T_dict, f)
        
            with open(all_runtimes_dir / "S_all_runtimes_Qn_dict.pkl", "wb") as f:
                pickle.dump(all_runtimes_Qn_dict, f)
        
            with open(all_runtimes_dir / "S_all_runtimes_tot_n_dict.pkl", "wb") as f:
                pickle.dump(all_runtimes_tot_n_dict, f)
                
            with open(all_runtimes_dir / "S_all_runtimes_Qa_dict.pkl", "wb") as f:
                pickle.dump(all_runtimes_Qa_dict, f)
        
            with open(all_runtimes_dir / "S_all_runtimes_tot_a_dict.pkl", "wb") as f:
                pickle.dump(all_runtimes_tot_a_dict, f)
                
            to_be_printed = f"{name} {n_pair} / {tot_pairs}; |X|: {card_X}, |Z|: {card_Z}; it: {h+1} DC"
            file.write(to_be_printed+f"\nT: {T_rt}; N: {Qn_rt}; A: {Qa_rt}\n")
            file.flush()
            print(to_be_printed)
            
        n_pair += 1
            
file.close()