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


##### ====== #####

# Neo4j connection settings
host = "bolt://localhost:7687"
username = "neo4j"
neo4j_psw = "graph000"

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
RW_dags_dir   = BASE / "DAGs/Real_world_dags"
inputs_dir = BASE / "Results/Inputs"
all_runtimes_dir = BASE /"Results/Runtimes/All_runtimes"

# Get queries
sys.path.append(str(BASE / "Scripts"))
from queries import query_baseline
from queries import query_baseline_1, query_baseline_2

########--------------- Retrieve DAGs' path ---------------########

names_of_graphs = ['SACHS', 'SMALLCOVID', 'CHILD', 'REDUCEDCOVID']

path_to_sachs = RW_dags_dir / "sachs.bif"
path_to_smallcovid19 = RW_dags_dir / "smallcovid19.json"
path_to_child = RW_dags_dir / "child.bif"
path_to_reducedcovid = RW_dags_dir / "reducedcovid.json"
#path_to_covid19 = RW_dags_dir / "covid19.json"

all_paths_to_dags = [ path_to_sachs,
                      path_to_smallcovid19,
                      path_to_child,
                      path_to_reducedcovid#,
                      #path_to_covid19
                    ]

data_files = dict(zip(names_of_graphs, all_paths_to_dags))

########--------------- Upload the inputs files ---------------########

with open(inputs_dir / "RW_X_inputs.pkl", "rb") as f:
    RW_X_inputs = pickle.load(f)

with open(inputs_dir / "RW_Z_inputs.pkl", "rb") as f:
    RW_Z_inputs = pickle.load(f)

########--------------- Initialize the runtimes dictionaries ----------########

RW_all_runtimes_baseline_dict = {}
for name in names_of_graphs:
    RW_all_runtimes_baseline_dict[name] = {}

########--------------- Iterates over the DAGs ---------------########

text_file = open("RW_Baseline_execution.txt", "w")

### Open the DAG

for name in names_of_graphs:
    if name in ['SACHS', 'CHILD']:
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

    ## Keep only the largest connecred component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G.remove_nodes_from(G.nodes() - largest_cc)
    N_nodes = len(G.nodes()) # Number of nodes in the DAG
    
    to_be_printed = f"The graph {name} has {N_nodes} nodes and {len(G.edges())} edges"
    text_file.write(to_be_printed+'\n')
    text_file.flush()
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
    query_baseline_complete = (   query_baseline_1 + str(ub)
                                + query_baseline_2         )
    
    # Retrieve inputs for the current DAG
    RW_X_inputs_current, RW_Z_inputs_current = RW_X_inputs[name], RW_Z_inputs[name]
    range_X, range_Z = [], []
    
    # Retrieve the cardinality pairs (|X|, |Z|)
    
    #
    for k in RW_X_inputs_current.keys():
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
    
    n_pair=1        # keeps truck of the current cardinality pair
    
    saved_op = []
    for pair in all_pairs:
            
        card_X, card_Z = pair  # unfolds |X| and |Z|
    
        X_instances = RW_X_inputs_current[pair]
        Z_instances = RW_Z_inputs_current[pair]
        
        B_rts = [] # initialises the runtimes' list
        
        its = len(X_instances) # number of iterations (inputs) for the pair
        
        for h in range(its):
            
            # Takes the input of the current iteration
            X, Z = X_instances[h], Z_instances[h]
            
            params = {"X_names": X, "Z_names": Z}
            
            # Baseline
            start = time.time()
            graph_db.run(query_baseline_complete, parameters=params).evaluate()
            end = time.time()
            
            # Save the runtimes
            rt_b = end - start
            B_rts.append(rt_b)
        
            RW_all_runtimes_baseline_dict[name][pair] = B_rts
            
            # Save to disk
            with open(all_runtimes_dir / "RW_all_runtimes_baseline_dict.pkl", "wb") as f:
                pickle.dump(RW_all_runtimes_baseline_dict, f)
                
            to_be_printed = f"{name} {n_pair} / {tot_pairs}; |X|: {card_X}, |Z|: {card_Z}; it: {h+1} B; rt:{rt_b}"
            text_file.write(to_be_printed+'\n')
            text_file.flush()
            print(to_be_printed)
            
        n_pair += 1
        
text_file.close()