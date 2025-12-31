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

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
S_dags_dir   = BASE / "DAGs/Synthetic_dags/Previous"
inputs_dir = BASE / "Results/Inputs"
all_runtimes_dir = BASE /"Results/Runtimes/All_runtimes"

graph_names = []

for dag in S_dags_dir.iterdir():
    if dag.is_file() and dag.suffix == ".pkl":
        graph_names.append(dag.name[:-4])
        
graph_names.sort()

name1, name2 = "TR2", "TR2"

with open(inputs_dir / f"S_X_inputs_{name2}.pkl", "rb") as f:
    X_inputs = pickle.load(f)
with open(inputs_dir / f"S_Z_inputs_{name2}.pkl", "rb") as f:
    Z_inputs = pickle.load(f)

with open(inputs_dir / f"Previous/S_X_inputs_{name1}.pkl", "rb") as f:
    PX_inputs = pickle.load(f)
with open(inputs_dir / f"Previous/S_Z_inputs_{name1}.pkl", "rb") as f:
    PZ_inputs = pickle.load(f)
    
print(X_inputs == PX_inputs)
print(Z_inputs == PZ_inputs)



name = "BA1"+'.pkl'
with open(S_dags_dir / name, "rb") as f:
    G = pickle.load(f)
    
N_nodes = len(G.nodes())    # Number of nodes in the DAG
E_edges = len(G.edges())
D = 2* E_edges / ( N_nodes * (N_nodes -1))
print("Density= ", D)

largest_cc = max(nx.weakly_connected_components(G), key=len)

to_be_removed = G.nodes() - largest_cc
G.remove_nodes_from(to_be_removed)

print('The graph', name, 'has', len(G.nodes()), 'nodes and', len(G.edges()), 'edges')
"""
name = "ER0"+'.pkl'
with open(S_dags_dir / name, "rb") as f:
    G2 = pickle.load(f)
    
print(G == G2)
    
N_nodes = len(G.nodes())    # Number of nodes in the DAG
E_edges = len(G.edges())
D = 2* E_edges / ( N_nodes * (N_nodes -1))
print("Density= ", D)

largest_cc = max(nx.weakly_connected_components(G), key=len)

to_be_removed = G.nodes() - largest_cc
G2.remove_nodes_from(to_be_removed)

"""

"""## Read transformation runtimes
with open(all_runtimes_dir / f"S_all_runtimes_T_{name[:-4]}.pkl", "rb") as f:
    S_all_runtimes_T_dict = pickle.load(f)

## Read query runtimes

# Native
with open(all_runtimes_dir / f"S_all_runtimes_Qn_{name[:-4]}.pkl", "rb") as f:
    S_all_runtimes_Qn_dict = pickle.load(f)
# APOC
with open(all_runtimes_dir / f"S_all_runtimes_tot_n_{name[:-4]}.pkl", "rb") as f:
    S_all_runtimes_tot_n_dict = pickle.load(f)"""