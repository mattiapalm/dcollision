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

graph_names = ['SACHS', 'CHILD', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN', 'SMALLCOVID', 'REDUCEDCOVID', 'COVID', 'CNSAMPLEDAG']
baseline_graph_names = ['SACHS', 'CHILD', 'SMALLCOVID', 'REDUCEDCOVID']

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

# Compute mean and variance DataFrames

for algo_runtimes, mean_dict, var_dict in zip(RW_all_runtimes_list, means_dict_list, vars_dict_list):
        
    for name in graph_names:
        
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

for name in baseline_graph_names:
    
    subdict = RW_all_runtimes_baseline_dict[name]
    
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
    RW_all_mean_runtimes_baseline_dict[name] = df_mean
    RW_all_var_runtimes_baseline_dict[name] = df_var


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

print(RW_all_mean_runtimes_T_Zfix)

# RW_all_mean_runtimes_T_Zfix_dict, RW_all_var_runtimes_T_Zfix_dict = {}, {}

# for name in graph_names:
#     RW_all_mean_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     RW_all_var_runtimes_T_dict.setdefault(name, pd.DataFrame())
#     RW_all_mean_runtimes_T_Zfix_dict[name] = RW_all_mean_runtimes_T_dict[name].mean(axis=0)
#     RW_all_var_runtimes_T_Zfix_dict[name] = RW_all_var_runtimes_T_dict[name].mean(axis=0)
    
    
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
    

    
# for name in names:
#     all_mean_runtimes_T_Zfix_APOC[name].to_excel("all_mean_runtimes_T_Zfix_APOC_"+name+".xlsx", index=False)
#     all_mean_runtimes_T_Zfix_Native[name].to_excel("all_mean_runtimes_T_Zfix_Native_"+name+".xlsx", index=False)    

# all_mean_runtimes_T_Zfix_APOC.to_excel("all_mean_runtimes_T_Zfix_APOC_.xlsx", index=False)
# all_mean_runtimes_T_Zfix_Native.to_excel("all_mean_runtimes_T_Zfix_Native_.xlsx", index=False)    


########--------------- Without Outliers ---------------########

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
    filtered_rts = [rt for rt, z in zip(values, mzs_abs_values) if z <= threshold]
    return filtered_rts
    

RW_all_runtimes_WO_T_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WO_Qn_dict = copy.deepcopy(RW_all_runtimes_Qn_dict)
RW_all_runtimes_WO_tot_n_dict = copy.deepcopy(RW_all_runtimes_tot_n_dict)
RW_all_runtimes_WO_Qa_dict = copy.deepcopy(RW_all_runtimes_Qa_dict)
RW_all_runtimes_WO_tot_a_dict = copy.deepcopy(RW_all_runtimes_tot_a_dict)
RW_all_runtimes_WO_baseline_dict = copy.deepcopy(RW_all_runtimes_baseline_dict)

for name in graph_names:
    for pair in RW_all_runtimes_WO_T_dict[name].keys():
        RW_all_runtimes_WO_T_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_T_dict[name][pair])
        RW_all_runtimes_WO_Qn_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_Qn_dict[name][pair])
        RW_all_runtimes_WO_tot_n_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_tot_n_dict[name][pair])
        RW_all_runtimes_WO_Qa_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_Qa_dict[name][pair])
        RW_all_runtimes_WO_tot_a_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_tot_a_dict[name][pair])

# Initialization
RW_all_mean_runtimes_WO_T_dict = {}
RW_all_mean_runtimes_WO_Qn_dict, RW_all_mean_runtimes_WO_tot_n_dict = {}, {}
RW_all_mean_runtimes_WO_Qa_dict, RW_all_mean_runtimes_WO_tot_a_dict = {}, {}
RW_all_mean_runtimes_WO_baseline_dict = {}
RW_all_var_runtimes_WO_T_dict = {}
RW_all_var_runtimes_WO_Qn_dict, RW_all_var_runtimes_WO_tot_n_dict = {}, {}
RW_all_var_runtimes_WO_Qa_dict, RW_all_var_runtimes_WO_tot_a_dict = {}, {}
RW_all_var_runtimes_WO_baseline_dict = {}


RW_all_runtimes_WO_list = [
                          RW_all_runtimes_WO_T_dict,
                          RW_all_runtimes_WO_Qn_dict,
                          RW_all_runtimes_WO_tot_n_dict,
                          RW_all_runtimes_WO_Qa_dict,
                          RW_all_runtimes_WO_tot_a_dict,
                          ]

# Parallel lists for iteration
means_WO_dict_list = [
    RW_all_mean_runtimes_WO_T_dict,
    RW_all_mean_runtimes_WO_Qn_dict,
    RW_all_mean_runtimes_WO_tot_n_dict,
    RW_all_mean_runtimes_WO_Qa_dict,
    RW_all_mean_runtimes_WO_tot_a_dict,
    ]

vars_WO_dict_list = [
    RW_all_var_runtimes_WO_T_dict,
    RW_all_var_runtimes_WO_Qn_dict,
    RW_all_var_runtimes_WO_tot_n_dict,
    RW_all_var_runtimes_WO_Qa_dict,
    RW_all_var_runtimes_WO_tot_a_dict
]

for algo_runtimes, mean_dict, var_dict in zip(RW_all_runtimes_WO_list, means_WO_dict_list, vars_WO_dict_list):
        
    for name in graph_names:
        
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
        
for name in baseline_graph_names:
    for pair in RW_all_runtimes_WO_baseline_dict[name].keys():
        RW_all_runtimes_WO_baseline_dict[name][pair] = filter_by_mod_zscore(RW_all_runtimes_WO_baseline_dict[name][pair])

for name in baseline_graph_names:
    
    subdict = RW_all_runtimes_WO_baseline_dict[name]
    
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
    RW_all_mean_runtimes_WO_baseline_dict[name] = df_mean
    RW_all_var_runtimes_WO_baseline_dict[name] = df_var


# Retrieve means and variances of the runtimes for the transformation with |Z| fixed

# Means
T_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_WO_T_dict.items()}
RW_all_mean_runtimes_WO_T_Zfix = pd.DataFrame(T_means)
RW_all_mean_runtimes_WO_T_Zfix = RW_all_mean_runtimes_WO_T_Zfix.sort_index()

# Variances
T_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_WO_T_dict.items()}
RW_all_var_runtimes_WO_T_Zfix = pd.DataFrame(T_vars)
RW_all_var_runtimes_WO_T_Zfix = RW_all_var_runtimes_WO_T_Zfix.sort_index()
    
    
# Means

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_T_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_T_dict, f)

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Qn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_tot_n_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_Qa_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_tot_a_dict, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_WO_T_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_T_Zfix, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WO_baseline_dict, f)
    
# Variances    

with open(var_runtimes_dir / "RW_all_var_runtimes_WO_T_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_T_dict, f)

with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_WO_T_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_T_Zfix, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WO_baseline_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WO_baseline_dict, f)
        
        
####

runtimes1 = RW_all_runtimes_Qn_dict['REDUCEDCOVID'][(1,1)]

plt.hist(runtimes1, bins=60)
plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Distribution")
plt.show()

sns.violinplot(x=runtimes1)
plt.xlabel("Runtime")
plt.title("Runtime Violin Plot")
plt.show()



runtimes2 = RW_all_runtimes_WO_Qn_dict['REDUCEDCOVID'][(1,1)]
print(len(runtimes2))

plt.hist(runtimes2, bins=30)
plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Distribution")
plt.show()


sns.violinplot(x=runtimes2)
plt.xlabel("Runtime")
plt.title("Runtime Violin Plot")
plt.show()
