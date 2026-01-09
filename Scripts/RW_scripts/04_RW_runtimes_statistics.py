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
copy = require("copy")
plt = require("matplotlib.pyplot")
sns = require("seaborn")
sts = require("scipy.stats")


def filter_by_mod_zscore(values, threshold=3.5):
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
    #filtered_rts = [rt for rt, z in zip(values, mzs_abs_values) if z <= threshold]
    return filtered_idx

##### ====== #####

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
runtimes_dir = BASE / "Results/Runtimes"
all_runtimes_dir = runtimes_dir / "All_runtimes"
all_runtimes_dir = BASE / "Results/Previous/1All_runtimes"
mean_runtimes_dir = runtimes_dir / "Mean_runtimes"
var_runtimes_dir = runtimes_dir / "Variances_of_the_runtimes"

##### ====== #####

### Graphs' names

graph_names = ['SACHS', 'C01', 'C02', 'CHILD', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']
baseline_graph_names = ['SACHS', 'C01', 'C02', 'CHILD']

########--------------- Load the runtimes' files ---------------########
    
## Read transformation runtimes
with open(all_runtimes_dir / "RW_all_runtimes_T_dict.pkl", "rb") as f:
    RW_all_runtimes_T_dict = pickle.load(f)

## Read query runtimes

# Native
with open(all_runtimes_dir / "RW_all_runtimes_Qn_dict.pkl", "rb") as f:
    RW_all_runtimes_Qn_dict = pickle.load(f)
# APOC
with open(all_runtimes_dir / "RW_all_runtimes_Qa_dict.pkl", "rb") as f:
    RW_all_runtimes_Qa_dict = pickle.load(f)
    
## Read total runtimes

# Native
with open(all_runtimes_dir / "RW_all_runtimes_tot_n_dict.pkl", "rb") as f:
    RW_all_runtimes_tot_n_dict = pickle.load(f)
# APOC
with open(all_runtimes_dir / "RW_all_runtimes_tot_a_dict.pkl", "rb") as f:
    RW_all_runtimes_tot_a_dict = pickle.load(f)

## Baseline runtimes
with open(all_runtimes_dir / "RW_all_runtimes_baseline_dict.pkl", "rb") as f:
    RW_all_runtimes_baseline_dict = pickle.load(f)
    
    
all_dicts = [RW_all_runtimes_T_dict,
             RW_all_runtimes_Qn_dict,
             RW_all_runtimes_Qa_dict,
             RW_all_runtimes_tot_n_dict,
             RW_all_runtimes_tot_a_dict,
             RW_all_runtimes_baseline_dict
             ]

 
########--------------- Retrieve statistics ---------------########


## Retrieve means and variances of runtimes per each (|X|, |Z|)

# Initialization
RW_all_mean_runtimes_T_dict = {}
RW_all_mean_runtimes_Qn_dict, RW_all_mean_runtimes_tot_n_dict = {}, {}
RW_all_mean_runtimes_Qa_dict, RW_all_mean_runtimes_tot_a_dict = {}, {}
RW_all_mean_runtimes_baseline_dict = {}
RW_all_var_runtimes_T_dict = {}
RW_all_var_runtimes_Qn_dict, RW_all_var_runtimes_tot_n_dict = {}, {}
RW_all_var_runtimes_Qa_dict, RW_all_var_runtimes_tot_a_dict = {}, {}
RW_all_var_runtimes_baseline_dict = {}
RW_all_sd_runtimes_T_dict = {}
RW_all_sd_runtimes_Qn_dict, RW_all_sd_runtimes_tot_n_dict = {}, {}
RW_all_sd_runtimes_Qa_dict, RW_all_sd_runtimes_tot_a_dict = {}, {}
RW_all_sd_runtimes_baseline_dict = {}
RW_overall_means_dict, RW_overall_sd_dict = {}, {}

RW_all_runtimes_list = [
                          RW_all_runtimes_T_dict,
                          RW_all_runtimes_Qn_dict,
                          RW_all_runtimes_tot_n_dict,
                          RW_all_runtimes_Qa_dict,
                          RW_all_runtimes_tot_a_dict,
                       ]

# Parallel lists for iteration
means_dict_list = [
    RW_all_mean_runtimes_T_dict,
    RW_all_mean_runtimes_Qn_dict,
    RW_all_mean_runtimes_tot_n_dict,
    RW_all_mean_runtimes_Qa_dict,
    RW_all_mean_runtimes_tot_a_dict,
]

vars_dict_list = [
    RW_all_var_runtimes_T_dict,
    RW_all_var_runtimes_Qn_dict,
    RW_all_var_runtimes_tot_n_dict,
    RW_all_var_runtimes_Qa_dict,
    RW_all_var_runtimes_tot_a_dict
]

sd_dict_list = [
    RW_all_sd_runtimes_T_dict,
    RW_all_sd_runtimes_Qn_dict,
    RW_all_sd_runtimes_tot_n_dict,
    RW_all_sd_runtimes_Qa_dict,
    RW_all_sd_runtimes_tot_a_dict
]


# Compute mean and variance DataFrames

for algo_runtimes, mean_dict, var_dict, sd_dict in zip(RW_all_runtimes_list, means_dict_list, vars_dict_list, sd_dict_list):
        
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
        

for name in baseline_graph_names:
    
    subdict = RW_all_runtimes_baseline_dict[name]
    
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
    RW_all_mean_runtimes_baseline_dict[name] = df_mean
    RW_all_var_runtimes_baseline_dict[name] = df_var
    RW_all_sd_runtimes_baseline_dict[name] = df_sd


###

# Retrieve means and variances of the runtimes for the transformation with |Z| fixed

# Means
T_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_T_dict.items()}
RW_all_mean_runtimes_T_Zfix = pd.DataFrame(T_means)
RW_all_mean_runtimes_T_Zfix = RW_all_mean_runtimes_T_Zfix.sort_index()

# Variances
T_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_T_dict.items()}
RW_all_var_runtimes_T_Zfix = pd.DataFrame(T_vars)
RW_all_var_runtimes_T_Zfix = RW_all_var_runtimes_T_Zfix.sort_index()

# Standard deviations
T_sds = {name: df.mean(axis=0) for name, df in RW_all_sd_runtimes_T_dict.items()}
RW_all_sd_runtimes_T_Zfix = pd.DataFrame(T_vars)
RW_all_sd_runtimes_T_Zfix = RW_all_sd_runtimes_T_Zfix.sort_index()

# RW_all_mean_runtimes_T_Zfix_dict, RW_all_var_runtimes_T_Zfix_dict = {}, {}

# for name in graph_names:
#     RW_all_mean_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     RW_all_var_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     RW_all_mean_runtimes_T_Zfix_dict[name] = RW_all_mean_runtimes_T_dict[name].mean(axis=0)
#     RW_all_var_runtimes_T_Zfix_dict[name] = RW_all_var_runtimes_T_dict[name].mean(axis=0)

for name in graph_names:
    RW_overall_means_dict[name] = {}
    RW_overall_sd_dict[name] = {}
    ov_mu_T = np.mean(RW_all_mean_runtimes_T_dict[name])
    ov_mu_Qn = np.mean(RW_all_mean_runtimes_Qn_dict[name])
    ov_mu_tot_n = np.mean(RW_all_mean_runtimes_tot_n_dict[name])
    ov_mu_Qa = np.mean(RW_all_mean_runtimes_Qa_dict[name])
    ov_mu_tot_a = np.mean(RW_all_mean_runtimes_tot_a_dict[name])
    ov_sd_T = np.nanstd(RW_all_mean_runtimes_T_dict[name].to_numpy())
    ov_sd_Qn = np.nanstd(RW_all_mean_runtimes_Qn_dict[name].to_numpy())
    ov_sd_tot_n = np.nanstd(RW_all_mean_runtimes_tot_n_dict[name].to_numpy())
    ov_sd_Qa = np.nanstd(RW_all_mean_runtimes_Qa_dict[name].to_numpy())
    ov_sd_tot_a = np.nanstd(RW_all_mean_runtimes_tot_a_dict[name].to_numpy())
    RW_overall_means_dict[name]['T'] = ov_mu_T
    RW_overall_means_dict[name]['Qn'] = ov_mu_Qn
    RW_overall_means_dict[name]['tot_n'] = ov_mu_tot_n
    RW_overall_means_dict[name]['Qa'] = ov_mu_Qa
    RW_overall_means_dict[name]['tot_a'] = ov_mu_tot_a
    RW_overall_sd_dict[name]['T'] = ov_sd_T
    RW_overall_sd_dict[name]['Qn'] = ov_sd_Qn
    RW_overall_sd_dict[name]['tot_n'] = ov_sd_tot_n
    RW_overall_sd_dict[name]['Qa'] = ov_sd_Qa
    RW_overall_sd_dict[name]['tot_a'] = ov_sd_tot_a
    
for name in baseline_graph_names:
    ov_mu_b = np.mean(RW_all_mean_runtimes_baseline_dict[name])
    ov_sd_b = np.std(RW_all_mean_runtimes_baseline_dict[name])
    RW_overall_means_dict[name]['b'] = ov_mu_b
    RW_overall_sd_dict[name]['b'] = ov_sd_b
    
    
########--------------- Save to disks ---------------########


# Means

with open(mean_runtimes_dir / "RW_all_mean_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_T_dict, f)

with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_Qn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_tot_n_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_Qa_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_tot_a_dict, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_T_Zfix, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_baseline_dict, f)
    
# Variances    

with open(var_runtimes_dir / "RW_all_var_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_T_dict, f)

with open(var_runtimes_dir / "RW_all_var_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_T_Zfix, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_baseline_dict, f)
    
# Standard deviations    

with open(var_runtimes_dir / "RW_all_sd_runtimes_T_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_T_dict, f)

with open(var_runtimes_dir / "RW_all_sd_runtimes_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_sd_runtimes_T_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_T_Zfix, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_baseline_dict, f)
    

    
# for name in names:
#     all_mean_runtimes_T_Zfix_APOC[name].to_excel("all_mean_runtimes_T_Zfix_APOC_"+name+".xlsx", index=False)
#     all_mean_runtimes_T_Zfix_Native[name].to_excel("all_mean_runtimes_T_Zfix_Native_"+name+".xlsx", index=False)    

# all_mean_runtimes_T_Zfix_APOC.to_excel("all_mean_runtimes_T_Zfix_APOC_.xlsx", index=False)
# all_mean_runtimes_T_Zfix_Native.to_excel("all_mean_runtimes_T_Zfix_Native_.xlsx", index=False)    


########--------------- Without Outliers ---------------########

    

RW_all_runtimes_WO_Tn_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WO_Ta_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WO_Qn_dict = copy.deepcopy(RW_all_runtimes_Qn_dict)
RW_all_runtimes_WO_tot_n_dict = copy.deepcopy(RW_all_runtimes_tot_n_dict)
RW_all_runtimes_WO_Qa_dict = copy.deepcopy(RW_all_runtimes_Qa_dict)
RW_all_runtimes_WO_tot_a_dict = copy.deepcopy(RW_all_runtimes_tot_a_dict)
RW_all_runtimes_WO_baseline_dict = copy.deepcopy(RW_all_runtimes_baseline_dict)

for name in graph_names:
    for pair in RW_all_runtimes_WO_tot_n_dict[name].keys():
        idx_n = filter_by_mod_zscore(RW_all_runtimes_WO_tot_n_dict[name][pair])
        RW_all_runtimes_WO_tot_n_dict[name][pair] = [RW_all_runtimes_WO_tot_n_dict[name][pair][i] for i in idx_n]
        RW_all_runtimes_WO_Tn_dict[name][pair] = [RW_all_runtimes_WO_Tn_dict[name][pair][i] for i in idx_n]
        RW_all_runtimes_WO_Qn_dict[name][pair] = [RW_all_runtimes_WO_Qn_dict[name][pair][i] for i in idx_n]
        idx_a = filter_by_mod_zscore(RW_all_runtimes_WO_tot_a_dict[name][pair])
        RW_all_runtimes_WO_tot_a_dict[name][pair] = [RW_all_runtimes_WO_tot_a_dict[name][pair][i] for i in idx_a]
        RW_all_runtimes_WO_Ta_dict[name][pair] = [RW_all_runtimes_WO_Ta_dict[name][pair][i] for i in idx_a]
        RW_all_runtimes_WO_Qa_dict[name][pair] = [RW_all_runtimes_WO_Qa_dict[name][pair][i] for i in idx_a]
        



# Initialization
RW_all_mean_runtimes_WO_Tn_dict, RW_all_mean_runtimes_WO_Ta_dict = {}, {}
RW_all_mean_runtimes_WO_Qn_dict, RW_all_mean_runtimes_WO_tot_n_dict = {}, {}
RW_all_mean_runtimes_WO_Qa_dict, RW_all_mean_runtimes_WO_tot_a_dict = {}, {}
RW_all_mean_runtimes_WO_baseline_dict = {}
RW_all_var_runtimes_WO_Tn_dict, RW_all_var_runtimes_WO_Ta_dict = {}, {}
RW_all_var_runtimes_WO_Qn_dict, RW_all_var_runtimes_WO_tot_n_dict = {}, {}
RW_all_var_runtimes_WO_Qa_dict, RW_all_var_runtimes_WO_tot_a_dict = {}, {}
RW_all_var_runtimes_WO_baseline_dict = {}
RW_all_sd_runtimes_WO_Tn_dict, RW_all_sd_runtimes_WO_Ta_dict = {}, {}
RW_all_sd_runtimes_WO_Qn_dict, RW_all_sd_runtimes_WO_tot_n_dict = {}, {}
RW_all_sd_runtimes_WO_Qa_dict, RW_all_sd_runtimes_WO_tot_a_dict = {}, {}
RW_all_sd_runtimes_WO_baseline_dict = {}
RW_overall_means_WO_dict, RW_overall_sd_WO_dict = {}, {}


RW_all_runtimes_WO_list = [
                          RW_all_runtimes_WO_Tn_dict,
                          RW_all_runtimes_WO_Qn_dict,
                          RW_all_runtimes_WO_tot_n_dict,
                          RW_all_runtimes_WO_Ta_dict,
                          RW_all_runtimes_WO_Qa_dict,
                          RW_all_runtimes_WO_tot_a_dict,
                          ]

# Parallel lists for iteration
means_WO_dict_list = [
    RW_all_mean_runtimes_WO_Tn_dict,
    RW_all_mean_runtimes_WO_Qn_dict,
    RW_all_mean_runtimes_WO_tot_n_dict,
    RW_all_mean_runtimes_WO_Ta_dict,
    RW_all_mean_runtimes_WO_Qa_dict,
    RW_all_mean_runtimes_WO_tot_a_dict
    ]

vars_WO_dict_list = [
    RW_all_var_runtimes_WO_Tn_dict,
    RW_all_var_runtimes_WO_Qn_dict,
    RW_all_var_runtimes_WO_tot_n_dict,
    RW_all_var_runtimes_WO_Ta_dict,
    RW_all_var_runtimes_WO_Qa_dict,
    RW_all_var_runtimes_WO_tot_a_dict
    ]

sds_WO_dict_list = [
    RW_all_sd_runtimes_WO_Tn_dict,
    RW_all_sd_runtimes_WO_Qn_dict,
    RW_all_sd_runtimes_WO_tot_n_dict,
    RW_all_sd_runtimes_WO_Ta_dict,
    RW_all_sd_runtimes_WO_Qa_dict,
    RW_all_sd_runtimes_WO_tot_a_dict
    ]

for algo_runtimes, mean_dict, var_dict, sd_dict in zip(RW_all_runtimes_WO_list, means_WO_dict_list, vars_WO_dict_list, sds_WO_dict_list):
        
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
                df_var.loc[i, j] = np.std(runtimes, ddof=1)
            else:
                df_mean.loc[i, j] = np.nan
                df_var.loc[i, j] = np.nan
                df_sd.loc[i, j] = np.nan
        
        # Store results
        mean_dict[name] = df_mean
        var_dict[name] = df_var
        sd_dict[name] = df_sd
        
for name in baseline_graph_names:
    for pair in RW_all_runtimes_WO_baseline_dict[name].keys():
        idx_b = filter_by_mod_zscore(RW_all_runtimes_WO_baseline_dict[name][pair])
        RW_all_runtimes_WO_baseline_dict[name][pair] = [RW_all_runtimes_WO_baseline_dict[name][pair][i] for i in idx_n]
        
for name in baseline_graph_names:
    
    subdict = RW_all_runtimes_WO_baseline_dict[name]
    
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
    RW_all_mean_runtimes_WO_baseline_dict[name] = df_mean
    RW_all_var_runtimes_WO_baseline_dict[name] = df_var
    RW_all_sd_runtimes_WO_baseline_dict[name] = df_sd


# Retrieve means and variances of the runtimes for the transformation with |Z| fixed

# Means
Tn_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_WO_Tn_dict.items()}
RW_all_mean_runtimes_WO_Tn_Zfix = pd.DataFrame(Tn_means)
RW_all_mean_runtimes_WO_Tn_Zfix = RW_all_mean_runtimes_WO_Tn_Zfix.sort_index()

Ta_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_WO_Ta_dict.items()}
RW_all_mean_runtimes_WO_Ta_Zfix = pd.DataFrame(Ta_means)
RW_all_mean_runtimes_WO_Ta_Zfix = RW_all_mean_runtimes_WO_Ta_Zfix.sort_index()

# Variances
Tn_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_WO_Tn_dict.items()}
RW_all_var_runtimes_WO_Tn_Zfix = pd.DataFrame(Tn_vars)
RW_all_var_runtimes_WO_Tn_Zfix = RW_all_var_runtimes_WO_Tn_Zfix.sort_index()

Ta_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_WO_Ta_dict.items()}
RW_all_var_runtimes_WO_Ta_Zfix = pd.DataFrame(Ta_vars)
RW_all_var_runtimes_WO_Ta_Zfix = RW_all_var_runtimes_WO_Ta_Zfix.sort_index()

# Standard deviations
Tn_sds = {name: df.mean(axis=0) for name, df in RW_all_sd_runtimes_WO_Tn_dict.items()}
RW_all_sd_runtimes_WO_Tn_Zfix = pd.DataFrame(Tn_vars)
RW_all_sd_runtimes_WO_Tn_Zfix = RW_all_sd_runtimes_WO_Tn_Zfix.sort_index()

Ta_sds = {name: df.mean(axis=0) for name, df in RW_all_sd_runtimes_WO_Ta_dict.items()}
RW_all_sd_runtimes_WO_Ta_Zfix = pd.DataFrame(Ta_vars)
RW_all_sd_runtimes_WO_Ta_Zfix = RW_all_sd_runtimes_WO_Ta_Zfix.sort_index()

for name in graph_names:
    RW_overall_means_WO_dict[name] = {}
    RW_overall_sd_WO_dict[name] = {}
    ov_mu_Tn = np.mean(RW_all_mean_runtimes_WO_Tn_dict[name])
    ov_mu_Ta = np.mean(RW_all_mean_runtimes_WO_Ta_dict[name])
    ov_mu_Qn = np.mean(RW_all_mean_runtimes_WO_Qn_dict[name])
    ov_mu_tot_n = np.mean(RW_all_mean_runtimes_WO_tot_n_dict[name])
    ov_mu_Qa = np.mean(RW_all_mean_runtimes_WO_Qa_dict[name])
    ov_mu_tot_a = np.mean(RW_all_mean_runtimes_WO_tot_a_dict[name])
    ov_sd_Tn = np.nanstd(RW_all_mean_runtimes_WO_Tn_dict[name].to_numpy())
    ov_sd_Ta = np.nanstd(RW_all_mean_runtimes_WO_Ta_dict[name].to_numpy())
    ov_sd_Qn = np.nanstd(RW_all_mean_runtimes_WO_Qn_dict[name].to_numpy())
    ov_sd_tot_n = np.nanstd(RW_all_mean_runtimes_WO_tot_n_dict[name].to_numpy())
    ov_sd_Qa = np.nanstd(RW_all_mean_runtimes_WO_Qa_dict[name].to_numpy())
    ov_sd_tot_a = np.nanstd(RW_all_mean_runtimes_WO_tot_a_dict[name].to_numpy())
    RW_overall_means_WO_dict[name]['Tn'] = ov_mu_Tn
    RW_overall_means_WO_dict[name]['Ta'] = ov_mu_Ta
    RW_overall_means_WO_dict[name]['Qn'] = ov_mu_Qn
    RW_overall_means_WO_dict[name]['tot_n'] = ov_mu_tot_n
    RW_overall_means_WO_dict[name]['Qa'] = ov_mu_Qa
    RW_overall_means_WO_dict[name]['tot_a'] = ov_mu_tot_a
    RW_overall_sd_WO_dict[name]['Tn'] = ov_sd_Tn
    RW_overall_sd_WO_dict[name]['Ta'] = ov_sd_Ta
    RW_overall_sd_WO_dict[name]['Qn'] = ov_sd_Qn
    RW_overall_sd_WO_dict[name]['tot_n'] = ov_sd_tot_n
    RW_overall_sd_WO_dict[name]['Qa'] = ov_sd_Qa
    RW_overall_sd_WO_dict[name]['tot_a'] = ov_sd_tot_a
    
for name in baseline_graph_names:
    ov_mu_b = np.mean(RW_all_mean_runtimes_WO_baseline_dict[name])
    ov_sd_b = np.nanstd(RW_all_mean_runtimes_WO_baseline_dict[name].to_numpy())
    RW_overall_means_WO_dict[name]['b'] = ov_mu_b
    RW_overall_sd_WO_dict[name]['b'] = ov_sd_b
    


    
# Means

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Tn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Ta_dict, f)

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Qn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_tot_n_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Qa_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_tot_a_dict, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_WO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Tn_Zfix, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_WO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Ta_Zfix, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_baseline_dict, f)
    
# Variances    

with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Tn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Ta_dict, f)

with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_WO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Tn_Zfix, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_WO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Ta_Zfix, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_baseline_dict, f)
    
# Standard deviations    

with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Tn_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Ta_dict, f)

with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_sd_runtimes_WO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Tn_Zfix, f)
    
with open(var_runtimes_dir/ "RW_all_sd_runtimes_WO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_Ta_Zfix, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WO_baseline_dict, f)



        