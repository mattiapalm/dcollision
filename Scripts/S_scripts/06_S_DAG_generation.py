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
random = require("random")
np = require("numpy")
pickle = require("pickle")
time = require("time")


##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
synthetic_dags_dir   = BASE / "DAGs/Synthetic_dags"


########--------------- User-defined functions ---------------########

def transform_into_DAG(G, seed=None):
    
    """
    Takes as input a generic graph, and returns a DAG with the same node set.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        NetworkX graph.
    
    seed : int, optional
        Optional random seed. The default is None.

    Returns
    -------
    DAG : networkx.classes.graph.DiGraph
          A DAG with the same node set as G. 
    """
    
    # Step 1: Assign a random topological order
    ordering = list(G.nodes())
    rng = random.Random(seed)
    rng.shuffle(ordering)
    rank = {node: i for i, node in enumerate(ordering)}
    
    # Step 2: Direct edges according to order
    DAG = nx.DiGraph()
    DAG.add_nodes_from(G.nodes())
    
    for u, v in G.edges():
        if rank[u] < rank[v]:
            DAG.add_edge(u, v)
        else:
            DAG.add_edge(v, u)
    
    return DAG

def random_sequence_sum(n, low, high, total, seed=None):
    
    """
    Returns a list of integers, where the i-th element of the list corresponds
    to the number of nodes in the i+1-th layer of the DAG to be created.
    
    Parameters
    ----------
    n : int
        Number of elements in the output list.
    low : int
        The minimum value for each integer in the output list.
    high : int
        The maximum value for each integer in the output list.
    total : int
        The number of nodes in the layered DAG to be created.
    seed : int, optional
        Optional random seed. The default is None.

    Returns
    -------
    seq : list of ints
        
    """
    
    rng = random.Random(seed)

    # Start from the minimum
    seq = [low] * n
    remaining = total - n * low

    # Distribute remaining units randomly
    while remaining > 0:
        i = rng.randrange(n)
        if seq[i] < high:
            seq[i] += 1
            remaining -= 1

    rng.shuffle(seq)
    return seq


def layered_dag(nodes_per_layer, p, seed=None):
    """
    Create a layered (feed-forward) DAG.

    Parameters
    ----------
    nodes_per_layer : list of ints
        Number of nodes in each layer.
    p : float
        Probability of existence of an edge from a vertex in layer i to another
        in layer i+1
    seed : int, optional
        Optional random seed. The default is None.

    Returns
    -------
    DAG : networkx.DiGraph
        DESCRIPTION.
    layers : list of lists of node IDs
        list of lists of node IDs.

    """
    
    np.random.seed(seed)

    DAG = nx.DiGraph()
    
    layers = []
    node_id = 0
    for n in nodes_per_layer:
        layer_nodes = list(range(node_id, node_id + n))
        layers.append(layer_nodes)
        DAG.add_nodes_from(layer_nodes)
        node_id += n
    
    # Add edges from layer i to layer i+1 (or beyond if desired)
    for i in range(len(layers) - 1):
        
        U = layers[i]
        V = layers[i+1]
        
        size_U, size_V = len(U), len(V)
        for v in V:
            k = np.random.binomial(n=size_U-1, p=p) + 1   # random number of elements to sample
            subset = random.sample(U, k)
            for u in U:
                if u in subset:
                    DAG.add_edge(u, v)
                    
        for u in U:
            if len(DAG.out_edges(u))==0:
                k = np.random.binomial(n=size_V-1, p=p) + 1
                subset = random.sample(V, k)
                for v in V:
                    if v in subset:
                        DAG.add_edge(u, v)
        
    return DAG, layers

########--------------- DAGs creation ---------------########

graph_types = ['ER', 'BA', 'LF', 'TR']

# Initialise the dictionaries of DAGs
all_graphs_dict = {}
for t in graph_types:
    all_graphs_dict[t] = {}

card_V_list0 = [10**3, 10**4, 2*10**4]
card_V_list1 = [10**3, 10**4]#, 10**6, 5*10**6]

D = 0.00375
random.seed(2026)

for N in card_V_list1:
    
    index = card_V_list0.index(N)
    
    ########--------------- BARABASI-ALBERT ---------------########
    
    seed = 2026
    G_ba = nx.barabasi_albert_graph(N, m=3)
    DAG_ba = transform_into_DAG(G_ba, seed)
    
    print(f"BA {N} has {len(DAG_ba.edges())} edges.")
    print("BA Is DAG:", nx.is_directed_acyclic_graph(DAG_ba))        
    #all_graphs_dict['BA'][N] = DAG_ba
    
    to_be_saved = f"BA{index}.pkl"
    with open(synthetic_dags_dir / to_be_saved, "wb") as f:
        pickle.dump(DAG_ba, f)
    
    ########--------------- ERDOS-RENYI ---------------########
    
    start = time.time()
    G_er = nx.erdos_renyi_graph(n=N, p=D, directed=False)
    DAG_er = transform_into_DAG(G_er, seed)
    
    print(f"ER {N} has {len(DAG_er.edges())} edges.")
    print("ER Is DAG:", nx.is_directed_acyclic_graph(DAG_er))
    all_graphs_dict['ER'][N] = DAG_er
    
    to_be_saved = f"ER{index}.pkl"
    with open(synthetic_dags_dir / to_be_saved, "wb") as f:
        pickle.dump(DAG_er, f)
    end = time.time()
    
    print(f"{index} ", end-start)
    
    ########--------------- LAYERED / FEED-FORWARD ---------------########
    
    seed = 2026
    n_layers = int(np.sqrt(N))+1
    low = int(n_layers/2); high = int(3*n_layers/2)
    exv = 5 / (low-1)
    
    n_p_layer = random_sequence_sum(n=n_layers, low=low, high=high, total=N, seed=seed)
    DAG_lf, layers = layered_dag(n_p_layer, p = exv, seed=seed)
    
    print(f"LF {N} has {len(DAG_lf.edges())} edges.")
    print("LF Is DAG:", nx.is_directed_acyclic_graph(DAG_lf))
    all_graphs_dict['LF'][N] = DAG_lf
    
    to_be_saved = f"LF{index}.pkl"
    with open(synthetic_dags_dir / to_be_saved, "wb") as f:
        pickle.dump(DAG_lf, f)
    
    ########--------------- TREE ---------------########
    
    T_und = nx.random_labeled_rooted_tree(N)
    root = T_und.graph['root']
    DAG_tr = nx.bfs_tree(T_und, source=root)
    
    print(f"TR {N} has {len(DAG_tr.edges())} edges.")
    print("TR Is DAG:", nx.is_directed_acyclic_graph(DAG_tr))
    #all_graphs_dict['TR'][N] = DAG_tr
    
    to_be_saved = f"TR{index}.pkl"
    with open(synthetic_dags_dir / to_be_saved, "wb") as f:
        pickle.dump(DAG_tr, f)

