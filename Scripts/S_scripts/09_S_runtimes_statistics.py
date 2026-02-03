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
plt = require("matplotlib.pyplot")
sns = require("seaborn")
copy = require("copy")

##### ====== #####

def filter_by_mod_zscore(values, threshold=10):
    """
    Compute the list of values without otliers
    
    Parameters
    ----------
    values: list of floats
    threshold: float
    
    Returns
    -------
    filtered_rts: array of floats
        
    """
    values = np.asarray(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        mzs_values = np.zeros_like(values, dtype=float)
    else:
        mzs_values = 0.6745 * (values - median) / mad
    
    mzs_abs_values = np.abs(mzs_values)
    filtered_idx = [i for i, z in enumerate(mzs_abs_values) if z <= threshold]
    return filtered_idx

##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
runtimes_dir = BASE / "Results/Runtimes"
all_runtimes_dir = runtimes_dir / "All_runtimes"
mean_runtimes_dir = runtimes_dir / "Mean_runtimes"
var_runtimes_dir = runtimes_dir / "Vars_and_sds"

##### ====== #####

### Graphs' names

graph_types = ['BA', 'ER', 'LF', 'TR']
# current_run_types = ['BA', 'ER', 'LF', 'TR']
# current_run_dim = ['0', '1', '2']
graph_names = ['ER0', 'BA0', 'LF0', 'TR0', 'ER1', 'BA1', 'LF1', 'TR1', 'ER2', 'BA2', 'LF2', 'TR2']

# current_run_names = []
# for t in current_run_types:
#     for d in current_run_dim:
#         name = t+d
#         current_run_names.append(name)
        
S_all_runtimes_T_dict = {}
S_all_runtimes_Qn_dict = {}     
S_all_runtimes_tot_n_dict = {}     
 
########--------------- Load the runtimes' files ---------------########

for name in graph_names:
    
    ## Read d-collision graph generation runtimes
    with open(all_runtimes_dir / f"S_all_runtimes_T_{name}.pkl", "rb") as f:
        S_all_runtimes_T_dict[name] = pickle.load(f)
    
    ## Read Identification of d-separated nodes runtimes
    
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
S_all_sd_runtimes_T_dict = {}
S_all_sd_runtimes_Qn_dict, S_all_sd_runtimes_tot_n_dict = {}, {}
S_overall_means_dict, S_overall_sd_dict = {}, {}

S_all_runtimes_list = [
                          S_all_runtimes_T_dict,
                          S_all_runtimes_Qn_dict,
                          S_all_runtimes_tot_n_dict
                      ]

S_means_dict_list = [
    S_all_mean_runtimes_T_dict,
    S_all_mean_runtimes_Qn_dict,
    S_all_mean_runtimes_tot_n_dict
]

S_vars_dict_list = [
    S_all_var_runtimes_T_dict,
    S_all_var_runtimes_Qn_dict,
    S_all_var_runtimes_tot_n_dict
]

S_sd_dict_list = [
    S_all_sd_runtimes_T_dict,
    S_all_sd_runtimes_Qn_dict,
    S_all_sd_runtimes_tot_n_dict
]

# Compute mean and variance DataFrames

for algo_runtimes, mean_dict, var_dict, sd_dict in zip(S_all_runtimes_list, S_means_dict_list, S_vars_dict_list, S_sd_dict_list):
        
    for name in graph_names:
        
        subdict = algo_runtimes[name]
        
        # Extract all |X| and |Z| values
        X_values = sorted({i for (i, j) in subdict.keys()})
        Z_values = sorted({j for (i, j) in subdict.keys()})
        
        # Initialize DataFrames
        df_mean = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        df_var = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        df_sd = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        
        # Fill with mean and variance
        for (i, j), runtimes in subdict.items():
            if len(runtimes) > 0:
                df_mean.loc[i, j] = np.mean(runtimes)
                df_var.loc[i, j] = np.var(runtimes, ddof=1)  # sample variance
                df_sd.loc[i, j] = np.std(runtimes, ddof=1)
            else:
                df_mean.loc[i, j] = np.nan
                df_var.loc[i, j] = np.nan
                df_sd.loc[i, j] = np.nan
        
        # Store results
        mean_dict[name] = df_mean
        var_dict[name] = df_var
        sd_dict[name] = df_sd


###

for name in graph_names:
    S_overall_means_dict[name] = {}
    S_overall_sd_dict[name] = {}
    ov_mu_T = np.mean(S_all_mean_runtimes_T_dict[name])
    ov_mu_Qn = np.mean(S_all_mean_runtimes_Qn_dict[name])
    ov_mu_tot_n = np.mean(S_all_mean_runtimes_tot_n_dict[name])
    ov_sd_T = np.nanstd(S_all_mean_runtimes_T_dict[name].to_numpy())
    ov_sd_Qn = np.nanstd(S_all_mean_runtimes_Qn_dict[name].to_numpy())
    ov_sd_tot_n = np.nanstd(S_all_mean_runtimes_tot_n_dict[name].to_numpy())
    S_overall_means_dict[name]['T'] = ov_mu_T
    S_overall_means_dict[name]['Qn'] = ov_mu_Qn
    S_overall_means_dict[name]['tot_n'] = ov_mu_tot_n
    S_overall_sd_dict[name]['T'] = ov_sd_T
    S_overall_sd_dict[name]['Qn'] = ov_sd_Qn
    S_overall_sd_dict[name]['tot_n'] = ov_sd_tot_n
    
# Retrieve means and variances of the runtimes for the d-collision graph generation with |Z| fixed

# Means
T_means = {name: df.mean(axis=0) for name, df in S_all_mean_runtimes_T_dict.items()}
S_all_mean_runtimes_T_Zfix = pd.DataFrame(T_means)
S_all_mean_runtimes_T_Zfix = S_all_mean_runtimes_T_Zfix.sort_index()

# Variances
T_vars = {name: df.mean(axis=0) for name, df in S_all_var_runtimes_T_dict.items()}
S_all_var_runtimes_T_Zfix = pd.DataFrame(T_vars)
S_all_var_runtimes_T_Zfix = S_all_var_runtimes_T_Zfix.sort_index()

# Standard deviations
T_sds = {name: df.mean(axis=0) for name, df in S_all_sd_runtimes_T_dict.items()}
S_all_sd_runtimes_T_Zfix = pd.DataFrame(T_vars)
S_all_sd_runtimes_T_Zfix = S_all_sd_runtimes_T_Zfix.sort_index()
    
    
# Mean total runtimes with |X U Z| fixed to be plotted

rrts_idx = S_all_mean_runtimes_T_Zfix.columns
X_percs, Z_percs = [1,2,5,10,20,50,80], [0,1,2,5,10,20,50,80]
X_frac, Z_frac = [n/100 for n in X_percs], [n/100 for n in Z_percs]
X_dims = [1, 2, '1%', '2%', '5%','10%','20%','50%','80%']
Z_dims = [0, 1, 2, '1%', '2%', '5%', '10%', '20%', '50%', '80%']


            
# Mean runtimes of T phase with |Z| fixed to be plotted

S_means_over_dim_Zfix = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
df = S_all_mean_runtimes_T_Zfix
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    for C in Z_dims:
        if type(C) == int:
            size_Z = C
        else:
            f = int(C[:-1])/100
            size_Z = int(N*f)
        t = round(df.loc[size_Z, r], 3)
        S_means_over_dim_Zfix.loc[r,C] = t
        
        
# Mean runtimes of Qn phase with |X|=1 fixed to be plotted

S_means_Qn_over_Z_X1 = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
for r in graph_names:
    row = S_all_mean_runtimes_Qn_dict[r].loc[1]
    row = [round(s,3) for s in row]
    S_means_Qn_over_Z_X1.loc[r] = list(row)
    
# Mean runtimes of Qn phase with |X|= 0.01*|V| fixed to be plotted

S_means_Qn_over_Z_X1perc = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    idx = int(N*0.01)
    row = S_all_mean_runtimes_Qn_dict[r].loc[idx]
    row = [round(s,3) for s in row]
    S_means_Qn_over_Z_X1perc.loc[r] = list(row)

    
# Mean runtimes of Qn phase with |Z|=1 fixed to be plotted

S_means_Qn_over_X_Z1 = pd.DataFrame(
    index=graph_names,
    columns=X_dims
)
for r in graph_names:
    col = S_all_mean_runtimes_Qn_dict[r][1]
    col = [round(s,3) for s in col]
    S_means_Qn_over_X_Z1.loc[r] = list(col)
    
# Mean runtimes of Qn phase with |Z|= 0.01*|V| fixed to be plotted

S_means_Qn_over_X_Z1perc = pd.DataFrame(
    index=graph_names,
    columns=X_dims
)
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    idx = int(N*0.01)
    col = S_all_mean_runtimes_Qn_dict[r][idx]
    col = [round(s,3) for s in col]
    S_means_Qn_over_X_Z1perc.loc[r] = list(col)

    
    
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
    
# with open(mean_runtimes_dir / "S_means_over_dim.pkl", "wb") as f:
#     pickle.dump(S_means_over_dim, f)

with open(mean_runtimes_dir / "S_means_over_dim_Zfix.pkl", "wb") as f:
    pickle.dump(S_means_over_dim_Zfix, f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_Z_X1, f)

with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1perc.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_Z_X1perc, f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_Z_X1, f)

with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1perc.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_X_Z1perc, f)
    
# Variances    

with open(var_runtimes_dir / "S_all_var_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_T_dict, f)

with open(var_runtimes_dir / "S_all_var_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_Qn_dict, f)
    
with open(var_runtimes_dir / "S_all_var_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_tot_n_dict, f)

with open(var_runtimes_dir/ "S_all_var_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_T_Zfix, f)
    
# Standard deviations    

with open(var_runtimes_dir / "S_all_sd_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_T_dict, f)

with open(var_runtimes_dir / "S_all_sd_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_Qn_dict, f)
    
with open(var_runtimes_dir / "S_all_sd_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_tot_n_dict, f)
    
with open(var_runtimes_dir/ "S_all_sd_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_T_Zfix, f)
    

    


########--------------- Without Outliers ---------------########
    

S_all_runtimes_WO_T_dict = copy.deepcopy(S_all_runtimes_T_dict)
S_all_runtimes_WO_Qn_dict = copy.deepcopy(S_all_runtimes_Qn_dict)
S_all_runtimes_WO_tot_n_dict = copy.deepcopy(S_all_runtimes_tot_n_dict)

for name in graph_names:
    for pair in S_all_runtimes_WO_T_dict[name].keys():
        idx_n = filter_by_mod_zscore(S_all_runtimes_WO_tot_n_dict[name][pair])
        # if len(idx_n) < 30:
        #     print(name, pair, len(idx_n))
        S_all_runtimes_WO_tot_n_dict[name][pair] = [S_all_runtimes_WO_tot_n_dict[name][pair][i] for i in idx_n]
        S_all_runtimes_WO_T_dict[name][pair] = [S_all_runtimes_WO_T_dict[name][pair][i] for i in idx_n]
        S_all_runtimes_WO_Qn_dict[name][pair] = [S_all_runtimes_WO_Qn_dict[name][pair][i] for i in idx_n]
    

# Initialization
S_all_mean_runtimes_WO_T_dict = {}
S_all_mean_runtimes_WO_Qn_dict, S_all_mean_runtimes_WO_tot_n_dict = {}, {}
S_all_var_runtimes_WO_T_dict = {}
S_all_var_runtimes_WO_Qn_dict, S_all_var_runtimes_WO_tot_n_dict = {}, {}
S_all_sd_runtimes_WO_T_dict= {}
S_all_sd_runtimes_WO_Qn_dict, S_all_sd_runtimes_WO_tot_n_dict = {}, {}
S_overall_means_WO_dict, S_overall_sd_WO_dict = {}, {}

S_all_runtimes_WO_list = [
                          S_all_runtimes_WO_T_dict,
                          S_all_runtimes_WO_Qn_dict,
                          S_all_runtimes_WO_tot_n_dict
                          ]

# Parallel lists for iteration
means_WO_dict_list = [
    S_all_mean_runtimes_WO_T_dict,
    S_all_mean_runtimes_WO_Qn_dict,
    S_all_mean_runtimes_WO_tot_n_dict
]

vars_WO_dict_list = [
    S_all_var_runtimes_WO_T_dict,
    S_all_var_runtimes_WO_Qn_dict,
    S_all_var_runtimes_WO_tot_n_dict
]

sds_WO_dict_list = [
    S_all_sd_runtimes_WO_T_dict,
    S_all_sd_runtimes_WO_Qn_dict,
    S_all_sd_runtimes_WO_tot_n_dict
]

for algo_runtimes, mean_dict, var_dict, sd_dict in zip(S_all_runtimes_WO_list, means_WO_dict_list, vars_WO_dict_list, sds_WO_dict_list):
        
    for name in graph_names:
        
        subdict = algo_runtimes[name]
        
        # Extract all |X| and |Z| values
        X_values = sorted({i for (i, j) in subdict.keys()})
        Z_values = sorted({j for (i, j) in subdict.keys()})
        
        # Initialize DataFrames
        df_mean = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        df_var = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        df_sd = pd.DataFrame(index=X_values, columns=Z_values, dtype=float)
        
        # Fill with mean and variance
        for (i, j), runtimes in subdict.items():
            if len(runtimes) > 0:
                df_mean.loc[i, j] = np.mean(runtimes)
                df_var.loc[i, j] = np.var(runtimes, ddof=1)  # sample variance
                df_sd.loc[i, j] = np.std(runtimes, ddof=1)
            else:
                df_mean.loc[i, j] = np.nan
                df_var.loc[i, j] = np.nan
                df_sd.loc[i, j] = np.nan
        
        # Store results
        mean_dict[name] = df_mean
        var_dict[name] = df_var
        sd_dict[name] = df_sd


# Retrieve means and variances of the runtimes for d-collision graph generation with |Z| fixed

# Means
T_means = {name: df.mean(axis=0) for name, df in S_all_mean_runtimes_WO_T_dict.items()}
S_all_mean_runtimes_WO_T_Zfix = pd.DataFrame(T_means)
S_all_mean_runtimes_WO_T_Zfix = S_all_mean_runtimes_WO_T_Zfix.sort_index()

# Variances
T_vars = {name: df.mean(axis=0) for name, df in S_all_var_runtimes_WO_T_dict.items()}
S_all_var_runtimes_WO_T_Zfix = pd.DataFrame(T_vars)
S_all_var_runtimes_WO_T_Zfix = S_all_var_runtimes_WO_T_Zfix.sort_index()

# Standard deviations
T_sds = {name: df.mean(axis=0) for name, df in S_all_sd_runtimes_WO_T_dict.items()}
S_all_sd_runtimes_WO_T_Zfix = pd.DataFrame(T_vars)
S_all_sd_runtimes_WO_T_Zfix = S_all_sd_runtimes_WO_T_Zfix.sort_index()

for name in graph_names:
    S_overall_means_WO_dict[name] = {}
    S_overall_sd_WO_dict[name] = {}
    ov_mu_T = np.mean(S_all_mean_runtimes_WO_T_dict[name])
    ov_mu_Qn = np.mean(S_all_mean_runtimes_WO_Qn_dict[name])
    ov_mu_tot_n = np.mean(S_all_mean_runtimes_WO_tot_n_dict[name])
    ov_sd_T = np.nanstd(S_all_mean_runtimes_WO_T_dict[name].to_numpy())
    ov_sd_Qn = np.nanstd(S_all_mean_runtimes_WO_Qn_dict[name].to_numpy())
    ov_sd_tot_n = np.nanstd(S_all_mean_runtimes_WO_tot_n_dict[name].to_numpy())
    S_overall_means_WO_dict[name]['T'] = ov_mu_T
    S_overall_means_WO_dict[name]['Qn'] = ov_mu_Qn
    S_overall_means_WO_dict[name]['tot_n'] = ov_mu_tot_n
    S_overall_sd_WO_dict[name]['T'] = ov_sd_T
    S_overall_sd_WO_dict[name]['Qn'] = ov_sd_Qn
    S_overall_sd_WO_dict[name]['tot_n'] = ov_sd_tot_n

            
# Mean runtimes of T phase with |Z| fixed to be plotted
          
S_means_over_dim_Zfix_WO = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
df = S_all_mean_runtimes_WO_T_Zfix
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    for C in Z_dims:
        if type(C) == int:
            size_Z = C
        else:
            f = int(C[:-1])/100
            size_Z = int(N*f)
        t = round(df.loc[size_Z, r], 3)
        S_means_over_dim_Zfix_WO.loc[r,C] = t
        
# Mean runtimes of Qn phase with |X|=1 fixed to be plotted

S_means_Qn_over_Z_X1_WO = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
for r in graph_names:
    row = S_all_mean_runtimes_WO_Qn_dict[r].loc[1]
    row = [round(s,3) for s in row]
    S_means_Qn_over_Z_X1_WO.loc[r] = list(row)
        
# Mean runtimes of Qn phase with |X|= 0.01*|V| fixed to be plotted

S_means_Qn_over_Z_X1perc_WO = pd.DataFrame(
    index=graph_names,
    columns=Z_dims
)
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    idx = int(N*0.01)
    row = S_all_mean_runtimes_WO_Qn_dict[r].loc[idx]
    row = [round(s,3) for s in row]
    S_means_Qn_over_Z_X1perc_WO.loc[r] = list(row)  


# Mean runtimes of Qn phase with |Z|=1 fixed to be plotted

S_means_Qn_over_X_Z1_WO = pd.DataFrame(
    index=graph_names,
    columns=X_dims
)
for r in graph_names:
    col = S_all_mean_runtimes_Qn_dict[r][1]
    col = [round(s,3) for s in col]
    S_means_Qn_over_X_Z1_WO.loc[r] = list(col)
    
# Mean runtimes of Qn phase with |Z|= 0.01*|V| fixed to be plotted

S_means_Qn_over_X_Z1perc_WO = pd.DataFrame(
    index=graph_names,
    columns=X_dims
)
for r in graph_names:
    if r == 'ER0':
        N = 903
    elif int(r[2]) == 0:
        N = 1000
    elif int(r[2]) == 1:
        N = 10000
    else:
        N = 20000
    idx = int(N*0.01)
    col = S_all_mean_runtimes_Qn_dict[r][idx]
    col = [round(s,3) for s in col]
    S_means_Qn_over_X_Z1perc_WO.loc[r] = list(col)   
        
### Proportion of instances accounted for the computation of WO statistics
 
S_WO_proportions_n = {}
for name in graph_names:
    df_n = S_all_mean_runtimes_WO_tot_n_dict[name].copy()
    for idx in df_n.index:
        for col in df_n.columns:
            if not np.isnan(df_n.loc[idx,col]):
                perc_n = len(S_all_runtimes_WO_tot_n_dict[name][(idx, col)]) / 30
                df_n.loc[idx, col] = perc_n
    S_WO_proportions_n[name] = df_n
    


    
# Means

with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_WO_T_dict, f)

with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_WO_Qn_dict, f)
    
with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_WO_tot_n_dict, f)
    
with open(mean_runtimes_dir/ "S_all_mean_runtimes_WO_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_mean_runtimes_WO_T_Zfix, f)

# with open(mean_runtimes_dir / "S_means_over_dim_WO.pkl", "wb") as f:
#     pickle.dump(S_means_over_dim_WO, f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1_WO.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_Z_X1_WO, f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1perc_WO.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_Z_X1perc_WO, f)

with open(mean_runtimes_dir / "S_means_over_dim_Zfix_WO.pkl", "wb") as f:
    pickle.dump(S_means_over_dim_Zfix_WO, f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1_WO.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_X_Z1_WO, f)

with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1perc_WO.pkl", "wb") as f:
    pickle.dump(S_means_Qn_over_X_Z1perc_WO, f)

# Variances    

with open(var_runtimes_dir / "S_all_var_runtimes_WO_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_WO_T_dict, f)

with open(var_runtimes_dir / "S_all_var_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_WO_Qn_dict, f)
    
with open(var_runtimes_dir / "S_all_var_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_WO_tot_n_dict, f)
    
with open(var_runtimes_dir/ "S_all_var_runtimes_WO_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_var_runtimes_WO_T_Zfix, f)
    
# Standard deviations    

with open(var_runtimes_dir / "S_all_sd_runtimes_WO_T_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_WO_T_dict, f)

with open(var_runtimes_dir / "S_all_sd_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_WO_Qn_dict, f)
    
with open(var_runtimes_dir / "S_all_sd_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_WO_tot_n_dict, f)
        
with open(var_runtimes_dir/ "S_all_sd_runtimes_WO_T_Zfix.pkl", "wb") as f:
    pickle.dump(S_all_sd_runtimes_WO_T_Zfix, f)

# proportions

with open(runtimes_dir / "Outliers/S_WO_proportions_n.pkl", "wb") as f:
    pickle.dump(S_WO_proportions_n, f)
