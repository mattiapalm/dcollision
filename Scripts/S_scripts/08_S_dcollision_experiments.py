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

graph_types = ['BA', 'ER', 'LF', 'TR']
current_run_types = ['LF', 'TR']
current_run_dim = ['2']

current_run_names = []
for t in current_run_types:
    for d in current_run_dim:
        name = t+d
        current_run_names.append(name)
        


########--------------- Iterates over the DAGs ---------------########

file = open("S_DC_execution.txt", "w")

for name in current_run_names:
    
    dag_name = name+'.pkl'
    
    with open(S_dags_dir / dag_name, "rb") as f:
        G = pickle.load(f)
        
    N_nodes = len(G.nodes())    # Number of nodes in the DAG

    to_be_printed = f"The graph {name} has {N_nodes} nodes and {len(G.edges())} edges"
    file.write(to_be_printed+'\n')
    file.flush()
    print(to_be_printed)
        
# -------

    ub = N_nodes-1             # Upper bound on the number of edges in a simple path
    
    # Obtain the DAG-specific query
    query_dcollision_4of4_complete = (   query_dcollision_4of4_1 + str(ub)
                                       + query_dcollision_4of4_2 + str(ub)
                                       + query_dcollision_4of4_3 )
    
########--------------- Upload the inputs files ---------------########

    with open(inputs_dir / f"S_X_inputs_{dag_name}", "rb") as f:
        X_inputs = pickle.load(f)

    with open(inputs_dir / f"S_Z_inputs_{dag_name}", "rb") as f:
        Z_inputs = pickle.load(f)
     
    # Retrieve the cardinality pairs (|X|, |Z|)
    
    all_pairs = X_inputs.keys()
    tot_pairs = len(all_pairs)
    #
    
    n_pair=1   # keeps truck of the current cardinality pair
    
    
########--------------- Initialize the runtimes dictionaries ----------########

    all_runtimes_T_dict = {}
    all_runtimes_Qn_dict = {}
    all_runtimes_tot_n_dict = {}
    # all_runtimes_Qa_dict = {}
    # all_runtimes_tot_a_dict = {}
        
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
    
    for pair in all_pairs:
                
            card_X, card_Z = pair  # unfolds |X| and |Z|
        
            X_instances = X_inputs[pair]
            Z_instances = Z_inputs[pair]
            
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
                
                # # APOC
                # start_Qa = time.time()
                # graph_db.run(query_dcollision_4of4_apoc, parameters=params_Q).evaluate()
                # end_Qa = time.time()
                
                # Reset
                graph_db.run(query_dcollision_reset)
            
                # Save iteration's runtimes
                T_rt = end_T - start_T; T_rts.append(T_rt)
                Qn_rt = end_Qn - end_T; Qn_rts.append(Qn_rt)
                tot_n_rt = T_rt + Qn_rt; tot_n_rts.append(tot_n_rt)
                # Qa_rt = end_Qa - start_Qa; Qa_rts.append(Qa_rt)
                # tot_a_rt = T_rt + Qa_rt; tot_a_rts.append(tot_a_rt)
                
                all_runtimes_T_dict[pair] = T_rts
                all_runtimes_Qn_dict[pair] = Qn_rts
                all_runtimes_tot_n_dict[pair] = tot_n_rts
                # all_runtimes_Qa_dict[pair] = Qa_rts
                # all_runtimes_tot_a_dict[pair] = tot_a_rts
            
                # Save to disk
                
                with open(all_runtimes_dir / f"S_all_runtimes_T_{name}.pkl", "wb") as f:
                    pickle.dump(all_runtimes_T_dict, f)
            
                with open(all_runtimes_dir / f"S_all_runtimes_Qn_{name}.pkl", "wb") as f:
                    pickle.dump(all_runtimes_Qn_dict, f)
            
                with open(all_runtimes_dir / f"S_all_runtimes_tot_n_{name}.pkl", "wb") as f:
                    pickle.dump(all_runtimes_tot_n_dict, f)
                    
                # with open(all_runtimes_dir / "S_all_runtimes_Qa_{name}.pkl", "wb") as f:
                #     pickle.dump(all_runtimes_Qa_dict, f)
            
                # with open(all_runtimes_dir / "S_all_runtimes_tot_a_{name}.pkl", "wb") as f:
                #     pickle.dump(all_runtimes_tot_a_dict, f)
                    
                to_be_printed = f"{name} {n_pair} / {tot_pairs}; |X|: {card_X}, |Z|: {card_Z}; it: {h+1} DC"
                file.write(to_be_printed+f"\nT: {T_rt}; N: {Qn_rt}")#"; A: {Qa_rt}\n")
                file.flush()
                print(to_be_printed)
                
            n_pair += 1
            
file.close()