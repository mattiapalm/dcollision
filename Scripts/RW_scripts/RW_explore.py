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
BIFReader = require("pgmpy.readwrite", "pgmpy").BIFReader
json = require("json")
nx = require("networkx")
Graph = require("py2neo", "py2neo").Graph
Node = require("py2neo", "py2neo").Node
Relationship = require("py2neo", "py2neo").Relationship
time = require("time")
pickle = require("pickle")

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
RW_dags_dir   = BASE / "DAGs/Real_world_dags"


########--------------- Retrieve DAGs' path ---------------########

names_of_graphs = ['SACHS', 'C01', 'C02', 'CHILD', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']

path_to_sachs = RW_dags_dir / "sachs.bif"
path_to_c01 = RW_dags_dir / "c01.json"
path_to_c02 = RW_dags_dir / "c02.json"
path_to_child = RW_dags_dir / "child.bif"
path_to_covid19 = RW_dags_dir / "covid19.json"
path_to_barley = RW_dags_dir / "barley.bif"
path_to_win95pts = RW_dags_dir / "win95pts.bif"
path_to_cnsdag = RW_dags_dir / "cnsdag.json"
path_to_link = RW_dags_dir / "link.bif"
path_to_munin = RW_dags_dir / "munin.bif"

all_paths_to_dags = [path_to_sachs,
                     path_to_c01,
                     path_to_c02,
                     path_to_child,
                     path_to_covid19,
                     path_to_barley,
                     path_to_win95pts,
                     path_to_cnsdag,
                     path_to_link,
                     path_to_munin
                     ]

data_files = dict(zip(names_of_graphs, all_paths_to_dags))

dim_dict = {}

range_9 = list(range(1,10))
range_frac = [ r / 10 for r in range_9]


for name in names_of_graphs:
    
    ### Open the DAG
    
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

    ## Keep only the largest connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G.remove_nodes_from(G.nodes() - largest_cc)
    N_nodes = len(G.nodes())   # Number of nodes in the DAG
    E_edges = len(G.edges())
    
    range_ = [int(N_nodes * f) for f in range_frac]
    
    dim_dict[name] = {'V': N_nodes, 'E' : E_edges, '0': (1,0)}
    for i in range(9):
        k = i+1
        dim_dict[name][k] = (range_[i], range_[i])
        

    to_be_printed = f"The graph {name} has {N_nodes} nodes and {E_edges} edges"
    print(to_be_printed)

# # Save files
# with open(BASE / "Results/dim_dict.pkl", "wb") as f:
#     pickle.dump(dim_dict, f)


