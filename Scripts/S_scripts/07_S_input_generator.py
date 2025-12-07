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
random = require("random")
time = require("time")
pd = require("pandas")
pickle = require("pickle")

##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to directories
S_dags_dir  = BASE / "DAGs/Synthetic_dags"
inputs_dir = BASE / "Results/Inputs"
inputs_dir.mkdir(exist_ok=True)  # does not create it if it does not exist

########------------------------------########

graph_names = []
graph_types = ['BA','ER','LF','TR']

for dag in S_dags_dir.iterdir():
    if dag.is_file() and dag.suffix == ".pkl":
        graph_names.append(dag.name)
        
graph_names.sort()
    
#
range_frac = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]

# Number of iterations
its = 30

text_file = open("S_Inputs_execution.txt", "w")
time0=time.time()

for t in graph_types:
    prefix = t

######## Load synthetic DAGs ########

    for dag in S_dags_dir.iterdir():
        if (dag.is_file()
            and dag.name.startswith(prefix)
            and ('3' in dag.name or '4' in dag.name)
            and dag.suffix == ".pkl"):
            time1 = time.time()
            with open(dag, "rb") as f:
                G = pickle.load(f)
                
            dag_name = dag.name[:-4]
    
            N = len(G.nodes())    # number of nodes in the graph
            
            to_be_printed = f"The graph {dag.name} has {N} nodes and {len(G.edges())} edges."
            text_file.write(to_be_printed+'\n')
            text_file.flush()
            print(to_be_printed)
            
            ### Compute inputs dimension
            #
            range_ = [int(N * f) for f in range_frac] + [1,2]
            range_ = list(set(range_)); range_.sort()
            range_X, range_Z = range_.copy(), range_.copy()
            range_X.remove(0)
            
            dict_XZ = {(i,j):0 for i in range_X for j in range_Z if i+j < N}
            temp_df = pd.DataFrame(
                [(i, j, val) for (i, j), val in dict_XZ.items()],
                columns=["|X|", "|Z|", "value"]
            )
            
            rts_df = temp_df.pivot(index="|X|", columns="|Z|", values="value")
            #
            
            # Number of input dimensions
            tot_pairs = rts_df.count().sum()
            
            S_X_inputs, S_Z_inputs = {}, {}
            
            ########--------------- Generate inputs ---------------########
            
            random.seed(2026)
            for card_X in range_X:
                for card_Z in range_Z:
                    card_union = card_X + card_Z
                    if card_union < N:
                        print(card_X, card_Z)
                        
                        X_instances, Z_instances = {}, {}
                        for h in range(its):
                            sample_nodes = random.sample(list(G.nodes()), card_union)
                            X_instances[h] = list(sample_nodes[:card_X])
                            Z_instances[h] = list(sample_nodes[card_X:])
                            
                        S_X_inputs[(card_X, card_Z)] = X_instances
                        S_Z_inputs[(card_X, card_Z)] = Z_instances
        
            # Save to disk
            with open(inputs_dir / f"S_X_inputs_{dag_name}.pkl", "wb") as f:
                pickle.dump(S_X_inputs, f)
            
            with open(inputs_dir / f"S_Z_inputs_{dag_name}.pkl", "wb") as f:
                pickle.dump(S_Z_inputs, f)
            
            time2 = time.time()
            text_file.write(f"Elapsed time = {time2-time1}\n")
            text_file.flush()

text_file.write(f"Total time = {time2-time0}\n")
text_file.flush()
text_file.close()
        
            
        
        
                        