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
np = require("numpy")
pickle = require("pickle")
pd = require("pandas")

##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
runtimes_dir = BASE / "Results/Runtimes"
all_runtimes_dir = runtimes_dir / "All_runtimes"
mean_runtimes_dir = runtimes_dir / "Mean_runtimes"
var_runtimes_dir = runtimes_dir / "Variances_of_the_runtimes"

##### ====== #####

### Graphs' names

graph_types = ['BA', 'ER', 'LF', 'TR']
current_run_types = ['BA', 'LF', 'TR']
current_run_dim = ['3', '4']

current_run_names = []
for t in current_run_types:
    for d in current_run_dim:
        name = t+d
        current_run_names.append(name)
        
S_all_runtimes_T_dict = {}
S_all_runtimes_Qn_dict = {}     
S_all_runtimes_tot_n_dict = {}     
 
########--------------- Load the runtimes' files ---------------########

for name in current_run_names:
    
    ## Read transformation runtimes
    with open(all_runtimes_dir / f"S_all_runtimes_T_{name}.pkl", "rb") as f:
        S_all_runtimes_T_dict[name] = pickle.load(f)
    
    ## Read query runtimes
    
    # Native
    with open(all_runtimes_dir / f"S_all_runtimes_Qn_{name}.pkl", "rb") as f:
        S_all_runtimes_Qn_dict[name] = pickle.load(f)
    
    ## Read total runtimes
    
    # Native
    with open(all_runtimes_dir / f"S_all_runtimes_tot_n_{name}.pkl", "rb") as f:
        S_all_runtimes_tot_n_dict[name] = pickle.load(f)
    
 
    ########--------------- Retrieve statistics ---------------########
    
    
    ## Retrieve means and variances of runtimes per each (|X|, |Z|)
    
    # Initialization
    S_all_mean_runtimes_T_dict = {}
    S_all_mean_runtimes_Qn_dict, S_all_mean_runtimes_tot_n_dict = {}, {}
    S_all_var_runtimes_T_dict = {}
    S_all_var_runtimes_Qn_dict, S_all_var_runtimes_tot_n_dict = {}, {}
    
    S_all_runtimes_list = [
                              S_all_runtimes_T_dict,
                              S_all_runtimes_Qn_dict,
                              S_all_runtimes_tot_n_dict,
                           ]
    
    # Parallel lists for iteration
    means_dict_list = [
        S_all_mean_runtimes_T_dict,
        S_all_mean_runtimes_Qn_dict,
        S_all_mean_runtimes_tot_n_dict,
    ]
    
    vars_dict_list = [
        S_all_var_runtimes_T_dict,
        S_all_var_runtimes_Qn_dict,
        S_all_var_runtimes_tot_n_dict,
    ]

# Compute mean and variance DataFrames

for algo_runtimes, mean_dict, var_dict in zip(S_all_runtimes_list, means_dict_list, vars_dict_list):
        
    for name in current_run_names:
        
        subdict = algo_runtimes[name]
        
        # Extract all |X| and |Z| values
        X_values = sorted({i for (i, j) in subdict.keys()})
        Z_values = sorted({j for (i, j) in subdict.keys()})
        
        # Initialize DataFrames
        df_mean = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        df_var = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        
        # Fill with mean and variance
        for (i, j), runtimes in subdict.items():
            if len(runtimes) > 0:
                df_mean.loc[i, j] = np.mean(runtimes)
                df_var.loc[i, j] = np.var(runtimes, ddof=1)  # sample variance
            else:
                df_mean.loc[i, j] = np.nan
                df_var.loc[i, j] = np.nan
        
        # Store results
        mean_dict[name] = df_mean
        var_dict[name] = df_var


###

# Retrieve means and variances of the runtimes for the transformation with |Z| fixed

# Means
T_means = {name: df.mean(axis=0) for name, df in S_all_mean_runtimes_T_dict.items()}
S_all_mean_runtimes_T_Zfix = pd.DataFrame(T_means)
S_all_mean_runtimes_T_Zfix = S_all_mean_runtimes_T_Zfix.sort_index()

# Variances
T_vars = {name: df.mean(axis=0) for name, df in S_all_var_runtimes_T_dict.items()}
S_all_var_runtimes_T_Zfix = pd.DataFrame(T_vars)
S_all_var_runtimes_T_Zfix = S_all_var_runtimes_T_Zfix.sort_index()

print(S_all_mean_runtimes_T_Zfix)

# S_all_mean_runtimes_T_Zfix_dict, S_all_var_runtimes_T_Zfix_dict = {}, {}

# for name in graph_names:
#     S_all_mean_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     S_all_var_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     S_all_mean_runtimes_T_Zfix_dict[name] = S_all_mean_runtimes_T_dict[name].mean(axis=0)
#     S_all_var_runtimes_T_Zfix_dict[name] = S_all_var_runtimes_T_dict[name].mean(axis=0)
    
    
########--------------- Save to disks ---------------########


# Means

with open(mean_runtimes_dir / "S_all_mean_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_T_dict, f)

with open(mean_runtimes_dir / "S_all_mean_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_Qn_dict, f)
    
with open(mean_runtimes_dir / "S_all_mean_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_tot_n_dict, f)
      
with open(mean_runtimes_dir/ "S_all_mean_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_T_Zfix, f)
    
    
# Variances    

with open(var_runtimes_dir / "S_all_var_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_T_dict, f)

with open(var_runtimes_dir / "S_all_var_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_Qn_dict, f)
    
with open(var_runtimes_dir / "S_all_var_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_tot_n_dict, f)

with open(var_runtimes_dir/ "S_all_var_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_T_Zfix, f)
    

    
# for name in names:
#     all_mean_runtimes_T_Zfix_APOC[name].to_excel("all_mean_runtimes_T_Zfix_APOC_"+name+".xlsx", index=False)
#     all_mean_runtimes_T_Zfix_Native[name].to_excel("all_mean_runtimes_T_Zfix_Native_"+name+".xlsx", index=False)    

# all_mean_runtimes_T_Zfix_APOC.to_excel("all_mean_runtimes_T_Zfix_APOC_.xlsx", index=False)
# all_mean_runtimes_T_Zfix_Native.to_excel("all_mean_runtimes_T_Zfix_Native_.xlsx", index=False)    """