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

def filter_by_mod_zscore(values, threshold):
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
    
    filtered_idx = [i for i, z in enumerate(mzs_values) if z <= threshold]
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

graph_names = ['SACHS', 'C01', 'C02', 'CHILD', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']
baseline_graph_names = ['SACHS', 'C01', 'C02', 'CHILD']
small_graphs = ['SACHS', 'C01', 'C02', 'CHILD', 'COVID']

########--------------- Load the runtimes' files ---------------########
    
## Load transformation runtimes
with open(all_runtimes_dir / "RW_all_runtimes_T_dict.pkl", "rb") as f:
    RW_all_runtimes_T_dict = pickle.load(f)

## Load query runtimes

# Native
with open(all_runtimes_dir / "RW_all_runtimes_Qn_dict.pkl", "rb") as f:
    RW_all_runtimes_Qn_dict = pickle.load(f)
# APOC
with open(all_runtimes_dir / "RW_all_runtimes_Qa_dict.pkl", "rb") as f:
    RW_all_runtimes_Qa_dict = pickle.load(f)
    
## Load total runtimes

# Native
with open(all_runtimes_dir / "RW_all_runtimes_tot_n_dict.pkl", "rb") as f:
    RW_all_runtimes_tot_n_dict = pickle.load(f)
# APOC
with open(all_runtimes_dir / "RW_all_runtimes_tot_a_dict.pkl", "rb") as f:
    RW_all_runtimes_tot_a_dict = pickle.load(f)

## Baseline runtimes
with open(all_runtimes_dir / "RW_all_runtimes_baseline_dict.pkl", "rb") as f:
    RW_all_runtimes_baseline_dict = pickle.load(f)

### Load inputs dimentions

with open(BASE / "Results/dim_dict.pkl", "rb") as f:
    dim_dict = pickle.load(f)
    
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
                          RW_all_runtimes_tot_a_dict
                       ]

# Parallel lists for iteration
means_dict_list = [
    RW_all_mean_runtimes_T_dict,
    RW_all_mean_runtimes_Qn_dict,
    RW_all_mean_runtimes_tot_n_dict,
    RW_all_mean_runtimes_Qa_dict,
    RW_all_mean_runtimes_tot_a_dict
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

X_dims =  [1, '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
Z_dims =  [0, '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
XZ_dims = [1, '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
rrts_cols0 = list(range(0,91,10))
range_frac = [n/100 for n in rrts_cols0]

for dic_dag in RW_all_runtimes_list:
    for dag in graph_names:
        dic_rw = dic_dag[dag]
        N = dim_dict[dag]['V']

        range_Z = [int(N*f) for f in range_frac]
        range_X = [1] + [int(N*f) for f in range_frac[1:]]

        pairs = {
            (size_X, size_Z)
            for size_X in range_X
            for size_Z in range_Z
            if size_X + size_Z < N
        }

        for k in list(dic_rw.keys()):
            if k not in pairs:
                dic_rw.pop(k)

for dag in baseline_graph_names:
    dic_rw = RW_all_runtimes_baseline_dict[dag]
    N = dim_dict[dag]['V']

    range_Z = [int(N*f) for f in range_frac]
    range_X = [1] + [int(N*f) for f in range_frac[1:]]

    pairs = {
        (size_X, size_Z)
        for size_X in range_X
        for size_Z in range_Z
        if size_X + size_Z < N
    }

    for k in list(dic_rw.keys()):
        if k not in pairs:
            dic_rw.pop(k)                



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

### Overall means and standard deviations

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


### Retrieve means and variances of the runtimes for the transformation with |Z| fixed

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

# Mean total runtimes with |X U Z| fixed to be plotted


rrts_idx = RW_all_mean_runtimes_T_Zfix.columns


RW_means_over_dim_tot_n = pd.DataFrame(
    index=rrts_idx,
    columns=XZ_dims
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_tot_n_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_tot_n.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_tot_n.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_tot_n.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_tot_n.loc[r,C] = RW_means_over_dim_tot_n.loc[r,idx]

# Mean runtimes of Qn phase with |X U Z| fixed to be plotted
            
RW_means_over_dim_Qn = pd.DataFrame(
    index=rrts_idx,
    columns=XZ_dims
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_Qn_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_Qn.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_Qn.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_Qn.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_Qn.loc[r,C] = RW_means_over_dim_Qn.loc[r,idx]
            
# Mean runtimes of T phase with |Z| fixed to be plotted
            
RW_means_over_dim_Zfix = pd.DataFrame(
    index=rrts_idx,
    columns=Z_dims
)
df = RW_all_mean_runtimes_T_Zfix
for r in rrts_idx:
    N = dim_dict[r]['V']
    range_Z = [int(N*f) for f in range_frac]
    for C in Z_dims:
        if C==0:
            c=0
        else:
            c=int(int(C[:2])/10)
        size_Z = range_Z[c]
        # if size_Z == range_Z[c-1]:
        #     continue
        t = round(df.loc[size_Z, r], 3)
        RW_means_over_dim_Zfix.loc[r,C] = t
        if np.isnan(RW_means_over_dim_Zfix.loc[r,C]):
            idx =str((c-1)*10)+'%'
            RW_means_over_dim_Zfix.loc[r,C] = RW_means_over_dim_Zfix.loc[r,idx]
        
    
########--------------- Save to disk ---------------########


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
    
with open(mean_runtimes_dir / "RW_overall_means_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_means_dict, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_tot_n.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_tot_n, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Qn.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Qn, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Zfix.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Zfix, f)
    
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

with open(var_runtimes_dir / "RW_overall_sd_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_sd_dict, f)
     


########--------------- Without Outliers ---------------########

    

RW_all_runtimes_WO_Tn_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WO_Ta_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WO_Qn_dict = copy.deepcopy(RW_all_runtimes_Qn_dict)
RW_all_runtimes_WO_tot_n_dict = copy.deepcopy(RW_all_runtimes_tot_n_dict)
RW_all_runtimes_WO_Qa_dict = copy.deepcopy(RW_all_runtimes_Qa_dict)
RW_all_runtimes_WO_tot_a_dict = copy.deepcopy(RW_all_runtimes_tot_a_dict)
RW_all_runtimes_WO_baseline_dict = copy.deepcopy(RW_all_runtimes_baseline_dict)

threshold = 5
for name in graph_names:
    for pair in RW_all_runtimes_WO_tot_n_dict[name].keys():
        idx_n = filter_by_mod_zscore(RW_all_runtimes_WO_tot_n_dict[name][pair], threshold = threshold)
        RW_all_runtimes_WO_tot_n_dict[name][pair] = [RW_all_runtimes_WO_tot_n_dict[name][pair][i] for i in idx_n]
        RW_all_runtimes_WO_Tn_dict[name][pair] = [RW_all_runtimes_WO_Tn_dict[name][pair][i] for i in idx_n]
        RW_all_runtimes_WO_Qn_dict[name][pair] = [RW_all_runtimes_WO_Qn_dict[name][pair][i] for i in idx_n]
        idx_a = filter_by_mod_zscore(RW_all_runtimes_WO_tot_a_dict[name][pair], threshold = threshold)
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
                          RW_all_runtimes_WO_tot_a_dict
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
    for pair in RW_all_runtimes_WO_baseline_dict[name].keys():
        idx_b = filter_by_mod_zscore(RW_all_runtimes_WO_baseline_dict[name][pair], threshold = threshold)
        RW_all_runtimes_WO_baseline_dict[name][pair] = [RW_all_runtimes_WO_baseline_dict[name][pair][i] for i in idx_b]

        
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


### Overall means and standard deviations

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
    
# Mean total runtimes with |X U Z| fixed to be plotted
    
RW_means_over_dim_tot_n_WO = pd.DataFrame(
    index=rrts_idx,
    columns=XZ_dims
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_WO_tot_n_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_tot_n_WO.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_tot_n_WO.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_tot_n_WO.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_tot_n_WO.loc[r,C] = RW_means_over_dim_tot_n_WO.loc[r,idx]
                    
            
# Mean runtimes of Qn phase with |X U Z| fixed to be plotted
            
RW_means_over_dim_Qn_WO = pd.DataFrame(
    index=rrts_idx,
    columns=XZ_dims
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_WO_Qn_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_Qn_WO.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_Qn_WO.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_Qn_WO.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_Qn_WO.loc[r,C] = RW_means_over_dim_Qn_WO.loc[r,idx]

# Mean runtimes of T phase with |Z| fixed to be plotted   
            
RW_means_over_dim_Zfix_WO = pd.DataFrame(
    index=rrts_idx,
    columns=Z_dims
)
df = RW_all_mean_runtimes_WO_Tn_Zfix
for r in rrts_idx:
    N = dim_dict[r]['V']
    range_Z = [int(N*f) for f in range_frac]
    for C in Z_dims:
        if C==0:
            c=0
        else:
            c = int(int(C[:2])/10)
        size_Z = range_Z[c]
        # if size_Z == range_Z[c-1]:
        #     continue
        t = round(df.loc[size_Z, r], 3)
        RW_means_over_dim_Zfix_WO.loc[r,C] = t
        if np.isnan(RW_means_over_dim_Zfix_WO.loc[r,C]):
            idx =str((c-1)*10)+'%'
            RW_means_over_dim_Zfix_WO.loc[r,C] = RW_means_over_dim_Zfix_WO.loc[r,idx]
    
### Proportion of instances accounted for the computation of WO statistics
    
RW_WO_proportions_n, RW_WO_proportions_a, RW_WO_proportions_b = {}, {}, {}
for name in graph_names:
    df_n, df_a = RW_all_mean_runtimes_WO_tot_n_dict[name].copy(), RW_all_mean_runtimes_WO_tot_a_dict[name].copy()
    if name in baseline_graph_names:
        df_b = RW_all_mean_runtimes_WO_baseline_dict[name].copy()
    for idx in df_n.index:
        for col in df_n.columns:
            if not np.isnan(df_n.loc[idx,col]):
                perc_n = len(RW_all_runtimes_WO_tot_n_dict[name][(idx, col)]) / 100
                perc_a = len(RW_all_runtimes_WO_tot_a_dict[name][(idx, col)]) / 100
                df_n.loc[idx, col], df_a.loc[idx, col] = perc_n, perc_a
                if name in baseline_graph_names:
                    perc_b = len(RW_all_runtimes_WO_baseline_dict[name][(idx, col)]) / 100
                    df_b.loc[idx, col] = perc_b
    RW_WO_proportions_n[name] = df_n
    RW_WO_proportions_a[name] = df_a
    if name in baseline_graph_names:
        RW_WO_proportions_b[name] = df_b

    
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
    
with open(mean_runtimes_dir / "RW_overall_means_WO_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_means_WO_dict, f)

with open(mean_runtimes_dir / "RW_means_over_dim_tot_n_WO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_tot_n_WO, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Qn_WO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Qn_WO, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Zfix_WO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Zfix_WO, f)
    
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
    
with open(var_runtimes_dir / "RW_overall_sd_WO_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_sd_WO_dict, f)
    
# Proportions

with open(BASE / "Results/Runtimes/Proportions/RW_WO_proportions_n.pkl", "wb") as f:
    pickle.dump(RW_WO_proportions_n, f)
    
with open(BASE / "Results/Runtimes/Proportions/RW_WO_proportions_a.pkl", "wb") as f:
    pickle.dump(RW_WO_proportions_a, f)
    
with open(BASE / "Results/Runtimes/Proportions/RW_WO_proportions_b.pkl", "wb") as f:
    pickle.dump(RW_WO_proportions_b, f)
    
    
########--------------- Without Some Outliers ---------------########

    

RW_all_runtimes_WSO_Tn_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WSO_Ta_dict = copy.deepcopy(RW_all_runtimes_T_dict)
RW_all_runtimes_WSO_Qn_dict = copy.deepcopy(RW_all_runtimes_Qn_dict)
RW_all_runtimes_WSO_tot_n_dict = copy.deepcopy(RW_all_runtimes_tot_n_dict)
RW_all_runtimes_WSO_Qa_dict = copy.deepcopy(RW_all_runtimes_Qa_dict)
RW_all_runtimes_WSO_tot_a_dict = copy.deepcopy(RW_all_runtimes_tot_a_dict)



# Initialization
RW_all_mean_runtimes_WSO_Tn_dict = copy.deepcopy(RW_all_mean_runtimes_T_dict)
RW_all_mean_runtimes_WSO_Ta_dict = copy.deepcopy(RW_all_mean_runtimes_T_dict)
RW_all_mean_runtimes_WSO_Qn_dict  = copy.deepcopy(RW_all_mean_runtimes_Qn_dict)
RW_all_mean_runtimes_WSO_tot_n_dict = copy.deepcopy(RW_all_mean_runtimes_tot_n_dict)
RW_all_mean_runtimes_WSO_Qa_dict = copy.deepcopy(RW_all_mean_runtimes_Qa_dict)
RW_all_mean_runtimes_WSO_tot_a_dict = copy.deepcopy(RW_all_mean_runtimes_tot_a_dict)


RW_all_var_runtimes_WSO_Tn_dict = copy.deepcopy(RW_all_var_runtimes_T_dict)
RW_all_var_runtimes_WSO_Ta_dict = copy.deepcopy(RW_all_var_runtimes_T_dict)
RW_all_var_runtimes_WSO_Qn_dict = copy.deepcopy(RW_all_var_runtimes_Qn_dict)
RW_all_var_runtimes_WSO_tot_n_dict = copy.deepcopy(RW_all_var_runtimes_tot_n_dict)
RW_all_var_runtimes_WSO_Qa_dict = copy.deepcopy(RW_all_var_runtimes_Qa_dict)
RW_all_var_runtimes_WSO_tot_a_dict = copy.deepcopy(RW_all_var_runtimes_tot_a_dict)

RW_all_sd_runtimes_WSO_Tn_dict = copy.deepcopy(RW_all_sd_runtimes_T_dict)
RW_all_sd_runtimes_WSO_Ta_dict = copy.deepcopy(RW_all_sd_runtimes_T_dict)
RW_all_sd_runtimes_WSO_Qn_dict = copy.deepcopy(RW_all_sd_runtimes_Qn_dict)
RW_all_sd_runtimes_WSO_tot_n_dict = copy.deepcopy(RW_all_sd_runtimes_tot_n_dict)
RW_all_sd_runtimes_WSO_Qa_dict = copy.deepcopy(RW_all_sd_runtimes_Qa_dict)
RW_all_sd_runtimes_WSO_tot_a_dict = copy.deepcopy(RW_all_sd_runtimes_tot_a_dict)



RW_all_runtimes_WSO_list = [
                          RW_all_runtimes_WSO_Tn_dict,
                          RW_all_runtimes_WSO_Qn_dict,
                          RW_all_runtimes_WSO_tot_n_dict,
                          RW_all_runtimes_WSO_Ta_dict,
                          RW_all_runtimes_WSO_Qa_dict,
                          RW_all_runtimes_WSO_tot_a_dict
                          ]

# Parallel lists for iteration
means_WSO_dict_list = [
    RW_all_mean_runtimes_WSO_Tn_dict,
    RW_all_mean_runtimes_WSO_Qn_dict,
    RW_all_mean_runtimes_WSO_tot_n_dict,
    RW_all_mean_runtimes_WSO_Ta_dict,
    RW_all_mean_runtimes_WSO_Qa_dict,
    RW_all_mean_runtimes_WSO_tot_a_dict
    ]

vars_WSO_dict_list = [
    RW_all_var_runtimes_WSO_Tn_dict,
    RW_all_var_runtimes_WSO_Qn_dict,
    RW_all_var_runtimes_WSO_tot_n_dict,
    RW_all_var_runtimes_WSO_Ta_dict,
    RW_all_var_runtimes_WSO_Qa_dict,
    RW_all_var_runtimes_WSO_tot_a_dict
    ]

sds_WSO_dict_list = [
    RW_all_sd_runtimes_WSO_Tn_dict,
    RW_all_sd_runtimes_WSO_Qn_dict,
    RW_all_sd_runtimes_WSO_tot_n_dict,
    RW_all_sd_runtimes_WSO_Ta_dict,
    RW_all_sd_runtimes_WSO_Qa_dict,
    RW_all_sd_runtimes_WSO_tot_a_dict
    ]

RW_overall_means_WSO_dict = {}
RW_overall_sd_WSO_dict = {}

WSO_names = ['SACHS', 'C01', 'C02']

for ar, ar_WO, mu, mu_WO, var, var_WO, sd, sd_WO in zip(RW_all_runtimes_WSO_list, RW_all_runtimes_WO_list, means_WSO_dict_list, means_WO_dict_list, vars_WSO_dict_list, vars_WO_dict_list, sds_WSO_dict_list, sds_WO_dict_list):    
    for name in WSO_names:
        ar[name] = ar_WO[name]
        mu[name] = mu_WO[name]
        var[name] = var_WO[name]
        sd[name] = sd_WO[name]
    
        


# Retrieve means and variances of the runtimes for the transformation with |Z| fixed

# Means
Tn_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_WSO_Tn_dict.items()}
RW_all_mean_runtimes_WSO_Tn_Zfix = pd.DataFrame(Tn_means)
RW_all_mean_runtimes_WSO_Tn_Zfix = RW_all_mean_runtimes_WSO_Tn_Zfix.sort_index()

Ta_means = {name: df.mean(axis=0) for name, df in RW_all_mean_runtimes_WSO_Ta_dict.items()}
RW_all_mean_runtimes_WSO_Ta_Zfix = pd.DataFrame(Ta_means)
RW_all_mean_runtimes_WSO_Ta_Zfix = RW_all_mean_runtimes_WSO_Ta_Zfix.sort_index()

# Variances
Tn_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_WSO_Tn_dict.items()}
RW_all_var_runtimes_WSO_Tn_Zfix = pd.DataFrame(Tn_vars)
RW_all_var_runtimes_WSO_Tn_Zfix = RW_all_var_runtimes_WSO_Tn_Zfix.sort_index()

Ta_vars = {name: df.mean(axis=0) for name, df in RW_all_var_runtimes_WSO_Ta_dict.items()}
RW_all_var_runtimes_WSO_Ta_Zfix = pd.DataFrame(Ta_vars)
RW_all_var_runtimes_WSO_Ta_Zfix = RW_all_var_runtimes_WSO_Ta_Zfix.sort_index()

# Standard deviations
Tn_sds = {name: df.mean(axis=0) for name, df in RW_all_sd_runtimes_WSO_Tn_dict.items()}
RW_all_sd_runtimes_WSO_Tn_Zfix = pd.DataFrame(Tn_vars)
RW_all_sd_runtimes_WSO_Tn_Zfix = RW_all_sd_runtimes_WSO_Tn_Zfix.sort_index()

Ta_sds = {name: df.mean(axis=0) for name, df in RW_all_sd_runtimes_WSO_Ta_dict.items()}
RW_all_sd_runtimes_WSO_Ta_Zfix = pd.DataFrame(Ta_vars)
RW_all_sd_runtimes_WSO_Ta_Zfix = RW_all_sd_runtimes_WSO_Ta_Zfix.sort_index()


### Overall means and standard deviations

for name in graph_names:
    RW_overall_means_WSO_dict[name] = {}
    RW_overall_sd_WSO_dict[name] = {}
    ov_mu_Tn = np.mean(RW_all_mean_runtimes_WSO_Tn_dict[name])
    ov_mu_Ta = np.mean(RW_all_mean_runtimes_WSO_Ta_dict[name])
    ov_mu_Qn = np.mean(RW_all_mean_runtimes_WSO_Qn_dict[name])
    ov_mu_tot_n = np.mean(RW_all_mean_runtimes_WSO_tot_n_dict[name])
    ov_mu_Qa = np.mean(RW_all_mean_runtimes_WSO_Qa_dict[name])
    ov_mu_tot_a = np.mean(RW_all_mean_runtimes_WSO_tot_a_dict[name])
    ov_sd_Tn = np.nanstd(RW_all_mean_runtimes_WSO_Tn_dict[name].to_numpy())
    ov_sd_Ta = np.nanstd(RW_all_mean_runtimes_WSO_Ta_dict[name].to_numpy())
    ov_sd_Qn = np.nanstd(RW_all_mean_runtimes_WSO_Qn_dict[name].to_numpy())
    ov_sd_tot_n = np.nanstd(RW_all_mean_runtimes_WSO_tot_n_dict[name].to_numpy())
    ov_sd_Qa = np.nanstd(RW_all_mean_runtimes_WSO_Qa_dict[name].to_numpy())
    ov_sd_tot_a = np.nanstd(RW_all_mean_runtimes_WSO_tot_a_dict[name].to_numpy())
    RW_overall_means_WSO_dict[name]['Tn'] = ov_mu_Tn
    RW_overall_means_WSO_dict[name]['Ta'] = ov_mu_Ta
    RW_overall_means_WSO_dict[name]['Qn'] = ov_mu_Qn
    RW_overall_means_WSO_dict[name]['tot_n'] = ov_mu_tot_n
    RW_overall_means_WSO_dict[name]['Qa'] = ov_mu_Qa
    RW_overall_means_WSO_dict[name]['tot_a'] = ov_mu_tot_a
    RW_overall_sd_WSO_dict[name]['Tn'] = ov_sd_Tn
    RW_overall_sd_WSO_dict[name]['Ta'] = ov_sd_Ta
    RW_overall_sd_WSO_dict[name]['Qn'] = ov_sd_Qn
    RW_overall_sd_WSO_dict[name]['tot_n'] = ov_sd_tot_n
    RW_overall_sd_WSO_dict[name]['Qa'] = ov_sd_Qa
    RW_overall_sd_WSO_dict[name]['tot_a'] = ov_sd_tot_a
    
    
# Mean total runtimes with |X U Z| fixed to be plotted
    
RW_means_over_dim_tot_n_WSO = copy.deepcopy(RW_means_over_dim_tot_n_WO)
for r in WSO_names:
    df = RW_all_mean_runtimes_WSO_tot_n_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_tot_n_WSO.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_tot_n_WSO.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_tot_n_WSO.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_tot_n_WSO.loc[r,C] = RW_means_over_dim_tot_n_WSO.loc[r,idx]
                    
            
# Mean runtimes of Qn phase with |X U Z| fixed to be plotted
            
RW_means_over_dim_Qn_WSO = copy.deepcopy(RW_means_over_dim_Qn_WO)
for r in WSO_names:
    df = RW_all_mean_runtimes_WSO_Qn_dict[r]
    N = dim_dict[r]['V']
    range_ = [int(N*f) for f in range_frac]
    for C in XZ_dims:
        if C == 1:
            RW_means_over_dim_Qn_WSO.loc[r,C] = df.loc[(1,0)]
        else:
            c = int(int(C[:2])/10)
            m = 0
            tot = 0
            for c1 in range(c+1):
                c2 = c-c1
                C1, C2 = range_[c1], range_[c2]
                if C1 == 0:
                    if C2+1< N*((c+1)/10):
                        C1=1
                    else:
                        continue
                if (r in small_graphs and int((C1+C2)*(10/N)) == c) or r not in small_graphs:
                    s = df.loc[C1, C2]
                    tot += s
                    m += 1
            if m !=0:
               mu = round(tot/m, 3)
               RW_means_over_dim_Qn_WSO.loc[r,C] = mu
            elif np.isnan(RW_means_over_dim_Qn_WSO.loc[r,C]):
                idx = str((c-1)*10)+'%'
                RW_means_over_dim_Qn_WSO.loc[r,C] = RW_means_over_dim_Qn_WSO.loc[r,idx]

# Mean runtimes of T phase with |Z| fixed to be plotted   
            
RW_means_over_dim_Zfix_WSO = copy.deepcopy(RW_means_over_dim_Zfix_WO)
df = RW_all_mean_runtimes_WSO_Tn_Zfix
for r in WSO_names:
    N = dim_dict[r]['V']
    range_Z = [int(N*f) for f in range_frac]
    for C in Z_dims:
        if C==0:
            c=0
        else:
            c = int(int(C[:2])/10)
        size_Z = range_Z[c]
        # if size_Z == range_Z[c-1]:
        #     continue
        t = round(df.loc[size_Z, r], 3)
        RW_means_over_dim_Zfix_WSO.loc[r,C] = t
        if np.isnan(RW_means_over_dim_Zfix_WSO.loc[r,C]):
            idx =str((c-1)*10)+'%'
            RW_means_over_dim_Zfix_WSO.loc[r,C] = RW_means_over_dim_Zfix_WSO.loc[r,idx]
    

    
# Means

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Tn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Ta_dict, f)

with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Qn_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_tot_n_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Qa_dict, f)
    
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_tot_a_dict, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_WSO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Tn_Zfix, f)
    
with open(mean_runtimes_dir/ "RW_all_mean_runtimes_WSO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_mean_runtimes_WSO_Ta_Zfix, f)
    
with open(mean_runtimes_dir / "RW_overall_means_WSO_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_means_WSO_dict, f)

with open(mean_runtimes_dir / "RW_means_over_dim_tot_n_WSO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_tot_n_WSO, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Qn_WSO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Qn_WSO, f)
    
with open(mean_runtimes_dir / "RW_means_over_dim_Zfix_WSO.pkl", "wb") as f:
    pickle.dump(RW_means_over_dim_Zfix_WSO, f)
    
# Variances    

with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Tn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Ta_dict, f)

with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_var_runtimes_WSO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_WSO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Tn_Zfix, f)
    
with open(var_runtimes_dir/ "RW_all_var_runtimes_WSO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_var_runtimes_WSO_Ta_Zfix, f)
    

    
# Standard deviations    

with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_Tn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Tn_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_Ta_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Ta_dict, f)

with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_Qn_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Qn_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_tot_n_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_tot_n_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_Qa_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Qa_dict, f)
    
with open(var_runtimes_dir / "RW_all_sd_runtimes_WSO_tot_a_dict.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_tot_a_dict, f)
    
with open(var_runtimes_dir/ "RW_all_sd_runtimes_WSO_Tn_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Tn_Zfix, f)
    
with open(var_runtimes_dir/ "RW_all_sd_runtimes_WSO_Ta_Zfix.pkl", "wb") as f:
    pickle.dump(RW_all_sd_runtimes_WSO_Ta_Zfix, f)
 
with open(var_runtimes_dir / "RW_overall_sd_WSO_dict.pkl", "wb") as f:
    pickle.dump(RW_overall_sd_WSO_dict, f)
    

