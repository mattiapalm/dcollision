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
plt = require("matplotlib.pyplot", "matplotlib")
pd = require("pandas")
os = require("os")
PIL = require("PIL")

###----------------------###

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
runtimes_dir = BASE / "Results/Runtimes"
all_runtimes_dir = runtimes_dir / "All_runtimes"
mean_runtimes_dir = runtimes_dir / "Mean_runtimes"
var_runtimes_dir = runtimes_dir / "Vars_and_sds"
proportions_dir = runtimes_dir / "Proportions"
out_dir = BASE / "Results/Plots"
os.makedirs(out_dir, exist_ok=True)


########--------------- Load runtimes' files ---------------########


### Read mean runtimes

## Read mean transformation runtimes

# All
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_T_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_T_dict = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Tn_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_Tn_dict = pickle.load(f)
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Tn_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WSO_Tn_dict = pickle.load(f)
    
# |Z| fixed
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_T_Zfix.pkl", "rb") as f:
#     RW_all_mean_runtimes_T_Zfix = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Tn_Zfix.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_Tn_Zfix = pickle.load(f)
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Tn_Zfix.pkl", "rb") as f:
    RW_all_mean_runtimes_WSO_Tn_Zfix = pickle.load(f)

## Read mean query runtimes

# Native
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qn_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_Qn_dict = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_Qn_dict = pickle.load(f)
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_Qn_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WSO_Qn_dict = pickle.load(f)

# # APOC
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qa_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_Qa_dict = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qa_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_Qa_dict = pickle.load(f)

## Read mean total runtimes

# Native
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_n_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_tot_n_dict  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_tot_n_dict  = pickle.load(f)
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WSO_tot_n_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WSO_tot_n_dict  = pickle.load(f)

# # APOC
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_a_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_tot_a_dict  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_a_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_tot_a_dict  = pickle.load(f)
    
# Baseline
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_baseline_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_baseline_dict  = pickle.load(f)
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_baseline_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_baseline_dict  = pickle.load(f)
    
# Overall means
# with open(mean_runtimes_dir / "RW_overall_means_dict.pkl", "rb") as f:
#     RW_overall_means_dict  = pickle.load(f)
with open(mean_runtimes_dir / "RW_overall_means_WO_dict.pkl", "rb") as f:
    RW_overall_means_WO_dict  = pickle.load(f)
with open(mean_runtimes_dir / "RW_overall_means_WSO_dict.pkl", "rb") as f:
    RW_overall_means_WSO_dict  = pickle.load(f)
    
# Means over dimension

# with open(mean_runtimes_dir / "RW_means_over_dim_tot_n.pkl", "rb") as f:
#     RW_means_over_dim  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_means_over_dim_tot_n_WO.pkl", "rb") as f:
#     RW_means_over_dim_WO  = pickle.load(f)
with open(mean_runtimes_dir / "RW_means_over_dim_tot_n_WSO.pkl", "rb") as f:
    RW_means_over_dim_WSO  = pickle.load(f)
    
# with open(mean_runtimes_dir / "RW_means_over_dim_Qn.pkl", "rb") as f:
#     RW_means_over_dim_Qn  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_means_over_dim_Qn_WO.pkl", "rb") as f:
#     RW_means_over_dim_Qn_WO  = pickle.load(f)
with open(mean_runtimes_dir / "RW_means_over_dim_Qn_WSO.pkl", "rb") as f:
    RW_means_over_dim_Qn_WSO  = pickle.load(f)

# with open(mean_runtimes_dir / "RW_means_over_dim_Zfix.pkl", "rb") as f:
#     RW_means_over_dim_Zfix  = pickle.load(f)    
# with open(mean_runtimes_dir / "RW_means_over_dim_Zfix_WO.pkl", "rb") as f:
#     RW_means_over_dim_Zfix_WO  = pickle.load(f)
with open(mean_runtimes_dir / "RW_means_over_dim_Zfix_WSO.pkl", "rb") as f:
    RW_means_over_dim_Zfix_WSO  = pickle.load(f)
    
# ### Read variances of the runtimes

# ## Read variances of transformation runtimes

# # All
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_T_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_T_dict = pickle.load(f)
    
# # |Z| fixed
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_T_Zfix.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_T_Zfix = pickle.load(f)

# ## Read variances of query runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_Qn_dict = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_Qa_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_Qa_dict = pickle.load(f)

# ## Read variances of total runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_tot_n_dict  = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_tot_a_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_tot_a_dict  = pickle.load(f)
    
# # Baseline
# with open(var_runtimes_dir / "RW_all_var_runtimes_WO_baseline_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_WO_baseline_dict  = pickle.load(f) 

### Read standard deviations of the runtimes

## Load standard deviations of transformation runtimes

# All
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Tn_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_Tn_dict = pickle.load(f)
    
# |Z| fixed
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Tn_Zfix.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_T_Zfix = pickle.load(f)

## Load standard deviations of query runtimes

# Native
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qn_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_Qn_dict = pickle.load(f)

# APOC
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qa_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_Qa_dict = pickle.load(f)

## Load standard deviations of total runtimes

# Native
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_n_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_tot_n_dict  = pickle.load(f)

# APOC
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_a_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_tot_a_dict  = pickle.load(f)
    
# Baseline
with open(var_runtimes_dir / "RW_all_sd_runtimes_baseline_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_baseline_dict  = pickle.load(f)
with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_baseline_dict.pkl", "rb") as f:
    RW_all_sd_runtimes_WO_baseline_dict  = pickle.load(f)
    
# Overall standard deviations
with open(var_runtimes_dir / "RW_overall_sd_dict.pkl", "rb") as f:
    RW_overall_sd_dict  = pickle.load(f)   
with open(var_runtimes_dir / "RW_overall_sd_WO_dict.pkl", "rb") as f:
    RW_overall_sd_WO_dict  = pickle.load(f)


## Load proportions of WO runtimes

# Native
with open(proportions_dir / "RW_WO_proportions_n.pkl", "rb") as f:
    RW_WO_proportions_n  = pickle.load(f)

# APOC
with open(proportions_dir / "RW_WO_proportions_a.pkl", "rb") as f:
    RW_WO_proportions_a  = pickle.load(f)
    
# Baseline
with open(proportions_dir / "RW_WO_proportions_b.pkl", "rb") as f:
    RW_WO_proportions_b  = pickle.load(f)
    
##########========================================================############

### Graphs' names

graph_names = ['SACHS', 'C01', 'C02', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']
baseline_graph_names = ['SACHS', 'C01', 'C02']
"""
########--------------- PLOTS ---------------########

########--------------- Native VS Baseline ---------------########

for name in baseline_graph_names:
    
    df1 = RW_all_mean_runtimes_WO_tot_n_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_WO_baseline_dict[name]     # second dataframe
    
    print(name)

    for row_idx in df1.index:
        
        y1 = df1.loc[row_idx]
        y2 = df2.loc[row_idx]
    
        mask = ~df1.loc[row_idx].isna()   # use original mask
        x = df1.columns[mask].astype(int)
    
        plt.figure()
        plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
        plt.plot(x, y2[mask], 'o--', color='red', label='Baseline')
        plt.xticks(x)
        plt.title(name + f" |X| = {row_idx}")
        plt.xlabel("|Z|")
        plt.ylabel("mean t")
        plt.xlim(left=0, right=1.05*x.max())
        plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
        plt.legend()
        plt.grid(True)
        plt.show()
        
    for col in df1.columns:
        
        y1 = df1[col]
        y2 = df2[col]
    
        mask = ~df1[col].isna()
        x = df1.index[mask]
    
        plt.figure()
        plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
        plt.plot(x, y2[mask], 'o--', color='red', label='Baseline')
        plt.xticks(x)
        plt.title(name + f" |Z| = {col}")
        plt.xlabel("|X|")
        plt.ylabel("mean t")
        plt.xlim(left=0, right=1.05*x.max())
        plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
        plt.legend()
        plt.grid(True)
        plt.show()
    
########------––-- Native and Baseline with row/column fixed ----––----########
    
for name in baseline_graph_names:
    
    df1 = RW_all_mean_runtimes_WO_tot_n_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_WO_baseline_dict[name]     # second dataframe
    
    dfs = {'Native': df1, 'Baseline': df2}
    
    for k in dfs.keys():
        
        df = dfs[k]

        colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))  # distinct colors
    
        # Interpolate along rows (vary row index)
        df_col = df.interpolate(method='linear', axis=0, limit_area='inside')
        
        plt.figure()
 
        for idx, col in enumerate(df.columns):
            y_full = df_col[col]
            mask = ~df[col].isna()
            x = df.index[mask].astype(int)
            y = y_full[mask]
            
            plt.plot(x, y, 'o-', label=f"{col}", color=colors[idx])
            
        plt.xticks(df.index)
        plt.xlabel("|X|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |Z| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Interpolate along columns (vary column index)
        df_row = df.interpolate(method='linear', axis=1, limit_area='inside')
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(df.index)))  # distinct colors
        
        plt.figure()
        for idx, row in enumerate(df.index):
            y_full = df_row.loc[row]
            mask = ~df.loc[row].isna()
            x = df.columns[mask].astype(int)
            y = y_full[mask]
            
            plt.plot(x, y, 'o-', label=f"{row}", color=colors[idx])
            
        plt.xticks(df.columns)
        plt.xlabel("|Z|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |X| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        

########------––-- Native - Transformation only ----––----########

RW_mean_T_Zfix_interpolated = RW_all_mean_runtimes_WO_T_Zfix.interpolate(method='index', limit_area='inside')

### |Z| fixed
for col in graph_names:
    
    mask = RW_all_mean_runtimes_WO_T_Zfix[col].notna()
    x = RW_all_mean_runtimes_WO_T_Zfix.index[mask].astype(int)
    y = RW_mean_T_Zfix_interpolated.loc[mask, col]
    
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(f'Transformation runtime for {col}')
    plt.xlabel('|Z|')
    plt.ylabel('mean rt')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()
    
plt.figure() 
for col in graph_names:
    
    mask = RW_all_mean_runtimes_WO_T_Zfix[col].notna()
    x = RW_all_mean_runtimes_WO_T_Zfix.index[mask].astype(int)
    y = RW_mean_T_Zfix_interpolated.loc[mask, col]
    plt.plot(x, y, marker='o', label=col)

plt.xticks(RW_mean_T_Zfix_interpolated.index)
plt.title('Transformation runtime')
plt.xlabel('|Z|')
plt.ylabel('mean rt')
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.show()
        
# ### PLOTS FIND Y_ALL QUERY

# ### LINEAR INTERPOLATION

# ## LI Native and APOC comparison

# for name in graph_names:
    
#     df1 = RW_all_mean_runtimes_WO_Qn_dict[name]     # first dataframe
#     df2 = RW_all_mean_runtimes_WO_Qa_dict[name]     # second dataframe

    
#     for row_idx in df1.index:
        
#         y1 = df1.loc[row_idx]
#         y2 = df2.loc[row_idx]
    
#         mask = ~df1.loc[row_idx].isna()   # use original mask
#         x = df1.columns[mask].astype(int)
    
#         plt.figure()
#         plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
#         plt.plot(x, y2[mask], 'o--', color='red', label='APOC')
    
#         plt.title(name + f" |X| = {row_idx}")
#         plt.xlabel("|Z|")
#         plt.ylabel("mean t")
#         plt.xlim(left=0, right=1.05*x.max())
#         plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
#     for col in df1.columns:
        
#         y1 = df1[col]
#         y2 = df2[col]
    
#         mask = ~df1[col].isna()
#         x = df1.index[mask]
    
#         plt.figure()
#         plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
#         plt.plot(x, y2[mask], 'o--', color='red', label='APOC')
    
#         plt.title(name + f" |Z| = {col}")
#         plt.xlabel("|X|")
#         plt.ylabel("mean t")
#         plt.xlim(left=0, right=1.05*x.max())
#         plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
        

########------––-- Native VS APOC with row/column fixed ----––----########
    
for name in graph_names:
    
    df1 = RW_all_mean_runtimes_WO_Qn_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_WO_Qa_dict[name]     # second dataframe
    
    dfs = {'Native': df1, 'APOC': df2}
    
    for k in dfs.keys():
        
        df = dfs[k]
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))  # distinct colors
    
        # Interpolate along rows (vary row index)
        df_col = df.interpolate(method='linear', axis=0, limit_area='inside')
        
        plt.figure()
        
        for idx, col in enumerate(df.columns):
            y_full = df_col[col]
            mask = ~df[col].isna()
            x = df.index[mask].astype(int)
            y = y_full[mask]
        
            plt.plot(x, y, 'o-', label=f"{col}", color=colors[idx])
        
        plt.xlabel("|X|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |Z| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Interpolate along columns (vary column index)
        df_row = df.interpolate(method='linear', axis=1, limit_area='inside')
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(df.index)))  # distinct colors
        
        plt.figure()
        
        for idx, row in enumerate(df.index):
            y_full = df_row.loc[row]
            mask = ~df.loc[row].isna()
            x = df.columns[mask].astype(int)
            y = y_full[mask]
        
            plt.plot(x, y, 'o-', label=f"{row}", color=colors[idx])

        
        plt.xlabel("|Z|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |X| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        
### Main comparisons

dict_sachs = {'|X| fixed' : [1,2,6], '|Z| fixed' : [0,2]}
dict_c01 = {'|X| fixed' : [1,9], '|Z| fixed': [0,1,2]}
dict_c02 = {'|X| fixed' : [1,9], '|Z| fixed': [0,1,2]}
#dict_child = {'|X| fixed' : [1,2], '|Z| fixed' : [0,5]}
dict_covid = {'|X| fixed' : [1,10,22], '|Z| fixed' : [0,2]}
dict_barley = {'|X| fixed' : [1,9,33], '|Z| fixed' : [0,9,28]}
dict_win95pt = {'|X| fixed' : [1,2,45,53], '|Z| fixed' : [0,15]}
dict_cnsdag = {'|X| fixed' : [1,16,83,116], '|Z| fixed' : [0,16,116]}
dict_link = {'|X| fixed' : [1,71,642], '|Z| fixed' : [0,71,642]}
dict_munin = {'|X| fixed' : [1,104,936], '|Z| fixed' : [0,104,936]}

dict_fixed = {}
dict_fixed['SACHS'] = dict_sachs
dict_fixed['C01'] = dict_c01
dict_fixed['C02'] = dict_c02
#dict_fixed['CHILD'] = dict_child
dict_fixed['COVID'] = dict_covid
dict_fixed['BARLEY'] = dict_barley
dict_fixed['WIN95PTS'] = dict_win95pt
dict_fixed['CNSDAG'] = dict_cnsdag
dict_fixed['LINK'] = dict_link
dict_fixed['MUNIN'] = dict_munin


for name in graph_names:
    
    df1 = RW_all_mean_runtimes_WO_Qn_dict[name]
    df2 = RW_all_mean_runtimes_WO_Qa_dict[name]
    
    rows_to_keep = [r for r in dict_fixed[name]['|X| fixed'] if r in df1.index]
    cols = df1.columns  # full set of Z values

    # Interpolate along columns (axis=1)
    norm_interp = df1.loc[rows_to_keep].interpolate(axis=1, limit_area='inside')
    apoc_interp = df2.loc[rows_to_keep].interpolate(axis=1, limit_area='inside')
    
    x = cols

    plt.figure(figsize=(8, 5))

    for r in rows_to_keep:
        y1 = norm_interp.loc[r]
        y2 = apoc_interp.loc[r]
        plt.plot(x, y1, marker='o', label=f"|X|={r} Native")
        plt.plot(x, y2, marker='s', linestyle='--', label=f"|X|={r} APOC")
    plt.xticks(x)
    plt.title(f"{name}: Runtime Comparison (|Z| varies)")
    plt.xlabel("|Z|")
    plt.ylabel("runtime")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    cols_to_keep = [c for c in dict_fixed[name]['|Z| fixed'] if c in df1.columns]
    rows = df1.index  # full set of X values

    # Interpolate along rows (axis=0)
    norm_interp = df1[cols_to_keep].interpolate(axis=0, limit_area='inside')
    apoc_interp = df2[cols_to_keep].interpolate(axis=0, limit_area='inside')
    
    x = rows

    plt.figure(figsize=(8, 5))

    for c in cols_to_keep:
        y1 = norm_interp[c]
        y2 = apoc_interp[c]
        plt.plot(rows, y1, marker='o', label=f"|Z|={c} Native")
        plt.plot(rows, y2, marker='s', linestyle='--', label=f"|Z|={c} APOC")
    plt.xticks(x)
    plt.title(f"{name}: Runtime Comparison (|X| varies)")
    plt.xlabel("|X|")
    plt.ylabel("runtime")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    rrts_idx = RW_all_mean_runtimes_WO_Tn_Zfix.columns
    rrts_cols0 = [0,1,2,3,4]
    range_frac = [n/10 for n in range(0,5)]
    relevant_rts0 = pd.DataFrame(
        index=rrts_idx,
        columns=rrts_cols0
    )
    for r in rrts_idx:
        df = RW_all_mean_runtimes_WO_tot_n_dict[r]
        N = dim_dict[r]['V']
        range_X = [int(N*f) for f in range_frac]
        for c in rrts_cols0:
            if c == 0:
                relevant_rts0.loc[r,c] = df.loc[(1,0)]  
            elif c in range(1,5):
                C = range_X[c]
                if C != 0:
                    c1, c2 = C, C
                else:
                    c1, c2 = 1, 0
                relevant_rts0.loc[r,c] = df.loc[(c1,c2)]
                
    relevant_rts0.rename(columns={0: (1,0), 1: '(10%,10%)', 2: '(20%,20%)', 3:'(30%,30%)', 4:'(40%,40%)'}, inplace=True)

"""


with open(BASE / "Results/dim_dict.pkl", "rb") as f:
    dim_dict = pickle.load(f)
    
"""new_cols = {n: n/100 for n in range(0,91,10)}
new_cols[0] = '0'
new_cols2 = new_cols.copy()
new_cols2[0] = (1,0)


rrts_idx = RW_all_mean_runtimes_WO_Tn_Zfix.columns
rrts_cols1 = ['Mean','Largest']
relevant_rts1_WO = pd.DataFrame(
    index=rrts_idx,
    columns=rrts_cols1
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_WO_tot_n_dict[r]
    N = dim_dict[r]['V']
    for c in rrts_cols1:
        if c == 'Mean':
           mu = float(np.mean(df))
           relevant_rts1_WO.loc[r,c] = mu
        else:
            w = float(np.max(df))
            matches = df.where(df == w).stack()
            i = list(matches.index)[0]
            relevant_rts1_WO.loc[r,c] = (w, i)            
relevant_rts1_WO.rename(columns=new_cols, inplace=True)

## Mean total runtimes WO

df = RW_means_over_dim_WO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title("Trend of the mean total runtime")
plt.xlabel("Sizes of the input $X \cup Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes T WO

df = RW_means_over_dim_Zfix_WO
df.rename(columns=new_cols, inplace=True)
x = df.columns.astype(str)   # convert tuple → string
plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)
plt.title("Mean runtimes of the d-collision graph generation phase")
plt.xlabel("Proportion of nodes in the input $Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes Qn WO

df = RW_means_over_dim_Qn_WO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title("Mean runtimes of the d-separated nodes identification phase")
plt.xlabel("Proportion of nodes in the input $X \cup Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Baseline data WO

bs_cols = [0,'Mean']
relevant_bs = pd.DataFrame(
    index=baseline_graph_names,
    columns=bs_cols
)
for r in baseline_graph_names:
    df_means = RW_all_mean_runtimes_WO_baseline_dict[r]
    df_sds = RW_all_sd_runtimes_WO_baseline_dict[r]
    df_props = RW_WO_proportions_b[r]
    for c in bs_cols:
        if c == 0:
            mu = round(df_means.loc[1,0], 3)
            sd = round(df_sds.loc[1,0], 3)
            prop = df_props.loc[1,0]
            relevant_bs.loc[r,c] = (mu, sd, prop)
        elif c == 'Mean':
           mu = round(RW_overall_means_WO_dict[r]['b'],3)
           sd = round(RW_overall_sd_WO_dict[r]['b'],3)
           prop = round(np.mean(df_props),3)
           relevant_bs.loc[r,c] = (mu, sd, prop)
relevant_bs.rename(columns={0: (1,0)}, inplace=True)


# Bars T
Tdf_idx = graph_names
Tdf2_cols = ['Mean', 'Largest']
Tdf2 = pd.DataFrame(
    index=Tdf_idx,
    columns=Tdf2_cols
)
for r in Tdf2.index:
    df = RW_all_mean_runtimes_WO_Tn_dict[r]
    for c in Tdf2_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1_WO.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        Tdf2.loc[r,c] = round(v, 3)    
Tdf2.rename(columns={0: (1,0)}, inplace=True)

# Bars tot
totn_idx = graph_names
totn_cols = ['Mean', 'Largest']
totn = pd.DataFrame(
    index=totn_idx,
    columns=totn_cols
)
for r in totn.index:
    df = RW_all_mean_runtimes_WO_tot_n_dict[r]
    for c in totn_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1_WO.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        totn.loc[r,c] = round(v, 3)    
totn.rename(columns={0: (1,0)}, inplace=True)

xti = [f"{dag}\n {relevant_rts1_WO.loc[dag,'Largest'][1]}" for dag in graph_names]
graph_names = ['SACHS', 'C01', 'C02', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']

df1 = totn
df2 = Tdf2

for col in df1.columns:
    plt.figure()
    if col == 'Largest':
        x = xti
    else:
        x = df1.index
    plt.bar(x, df1[col], label = 'd-sep. nodes id.')
    plt.bar(x, df2[col], label = 'd-coll. gr. gen.')
    plt.xticks(rotation=45)   # 45 degrees
    plt.tight_layout()
    if col == (1,0):
        tit = "Mean runtimes for (|X|,|Z|)=(1,0)"
    elif col == "Mean":
        tit = "Overall mean runtimes"
    else:
        tit = "Largest mean runtimes"
    plt.title(tit)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 0.6, 0.2), reverse=True)
    plt.show()
    
    
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)

ax1, ax2 = axes

df = RW_means_over_dim_Qn_WO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

for name in graph_names:
    y = df.loc[name].astype(float)
    ax1.plot(x, y, marker='o', label=name)

ax1.title("Mean runtimes of the d-separated nodes identification phase")
ax1.xlabel("Proportion of nodes in the input $X \cup Z$")
plt.ylabel("Mean runtime in seconds")
plt.grid(True)
plt.legend()
plt.show()"""

    



##################


"""
rrts_idx = RW_all_mean_runtimes_T_Zfix.columns
rrts_cols1 = ['Mean','Largest']
relevant_rts1 = pd.DataFrame(
    index=rrts_idx,
    columns=rrts_cols1
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_tot_n_dict[r]
    N = dim_dict[r]['V']
    for c in rrts_cols1:
        if c == 'Mean':
           mu = float(np.mean(df))
           relevant_rts1.loc[r,c] = mu
        else:
            w = float(np.max(df))
            matches = df.where(df == w).stack()
            i = list(matches.index)[0]
            relevant_rts1.loc[r,c] = (w, i)            
relevant_rts1.rename(columns=new_cols, inplace=True)

## Mean total runtimes

df = RW_means_over_dim
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title("Trend of the mean total runtime")
plt.xlabel("Sizes of the input $X \cup Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes T

df = RW_means_over_dim_Zfix
df.rename(columns=new_cols, inplace=True)
x = df.columns.astype(str)   # convert tuple → string
plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)
plt.title("Mean runtimes of the d-collision graph generation phase")
plt.xlabel("Proportion of nodes in the input $Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes Qn

df = RW_means_over_dim_Qn
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title("Mean runtimes of the d-separated nodes identification phase")
plt.xlabel("Proportion of nodes in the input $X \cup Z$")
plt.ylabel("Mean runtime")
plt.grid(True)
plt.legend()
plt.show()

## Baseline data

bs_cols = [0,'Mean']
relevant_bs = pd.DataFrame(
    index=baseline_graph_names,
    columns=bs_cols
)
for r in baseline_graph_names:
    df_means = RW_all_mean_runtimes_baseline_dict[r]
    df_sds = RW_all_sd_runtimes_baseline_dict[r]
    for c in bs_cols:
        if c == 0:
            mu = round(df_means.loc[1,0], 3)
            sd = round(df_sds.loc[1,0], 3)
            relevant_bs.loc[r,c] = (mu, sd)
        elif c == 'Mean':
           mu = round(RW_overall_means_dict[r]['b'],3)
           sd = round(RW_overall_sd_dict[r]['b'],3)
           relevant_bs.loc[r,c] = (mu, sd,)
relevant_bs.rename(columns={0: (1,0)}, inplace=True)


# Bars T
Tdf_idx = graph_names
Tdf2_cols = ['Mean', 'Largest']
Tdf2 = pd.DataFrame(
    index=Tdf_idx,
    columns=Tdf2_cols
)
for r in Tdf2.index:
    df = RW_all_mean_runtimes_T_dict[r]
    for c in Tdf2_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        Tdf2.loc[r,c] = round(v, 3)    
Tdf2.rename(columns={0: (1,0)}, inplace=True)

# Bars tot
totn_idx = graph_names
totn_cols = ['Mean', 'Largest']
totn = pd.DataFrame(
    index=totn_idx,
    columns=totn_cols
)
for r in totn.index:
    df = RW_all_mean_runtimes_tot_n_dict[r]
    for c in totn_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        totn.loc[r,c] = round(v, 3)    
totn.rename(columns={0: (1,0)}, inplace=True)

xti = [f"{dag}\n {relevant_rts1.loc[dag,'Largest'][1]}" for dag in graph_names]
graph_names = ['SACHS', 'C01', 'C02', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']

df1 = totn
df2 = Tdf2

for col in df1.columns:
    plt.figure()
    if col == 'Largest':
        x = xti
    else:
        x = df1.index
    plt.bar(x, df1[col], label = 'd-sep. nodes id.')
    plt.bar(x, df2[col], label = 'd-coll. gr. gen.')
    plt.xticks(rotation=45)   # 45 degrees
    plt.tight_layout()
    if col == (1,0):
        tit = "Mean runtimes for (|X|,|Z|)=(1,0)"
    elif col == "Mean":
        tit = "Overall mean runtimes"
    else:
        tit = "Largest mean runtimes"
    plt.title(tit)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 0.6, 0.2), reverse=True)
    plt.show()
    

fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)

ax1, ax2 = axes

df = RW_means_over_dim_Qn
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

for name in graph_names:
    y = df.loc[name].astype(float)
    ax1.plot(x, y, marker='o', label=name)

ax1.title("Mean runtimes of the d-separated nodes identification phase")
ax1.xlabel("Proportion of nodes in the input $X \cup Z$")
plt.ylabel("Mean runtime in seconds")
plt.grid(True)
plt.legend()
plt.show()
"""

new_cols = {n: n/100 for n in range(0,91,10)}
new_cols[0] = '0'
new_cols2 = new_cols.copy()
new_cols2[0] = (1,0)


rrts_idx = RW_all_mean_runtimes_WSO_Tn_Zfix.columns
rrts_cols1 = ['Mean','Largest']
relevant_rts1_WSO = pd.DataFrame(
    index=rrts_idx,
    columns=rrts_cols1
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_WSO_tot_n_dict[r]
    N = dim_dict[r]['V']
    for c in rrts_cols1:
        if c == 'Mean':
           mu = float(np.mean(df))
           relevant_rts1_WSO.loc[r,c] = mu
        else:
            w = float(np.max(df))
            matches = df.where(df == w).stack()
            i = list(matches.index)[0]
            relevant_rts1_WSO.loc[r,c] = (w, i)            
relevant_rts1_WSO.rename(columns=new_cols, inplace=True)

## Mean total runtimes WO

df = RW_means_over_dim_WSO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title(r"$\mathbf{Trend \; of \; the \; mean \; total  \; runtime}$")
plt.xlabel(r"$\mathbf{Proportion \; of  \;nodes \; in \; the \; input \; X \cup Z}$")
plt.ylabel(r"$\mathbf{Mean \; runtime \; in  \; seconds}$")
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes T WO

df = RW_means_over_dim_Zfix_WSO
df.rename(columns=new_cols, inplace=True)
x = df.columns.astype(str)   # convert tuple → string
plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)
plt.title(r"$\mathbf{Mean \; runtimes \; for \; d-collision \; graph \; generation}$", fontsize=16)
plt.xlabel(r"$\mathbf{Proportion \; of  \;nodes \; in \; the \; input \; Z}$", fontsize=14)
plt.ylabel(r"$\mathbf{Mean \; runtime \; in  \; seconds}$", fontsize=14)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()

## Mean runtimes Qn WO

df = RW_means_over_dim_Qn_WSO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title(r"$\mathbf{Mean \; runtimes \; for \; Identification \; of \; d-separated \; nodes}$", fontsize=16)
plt.xlabel(r"$\mathbf{Proportion \; of  \;nodes \; in \; the \; input \; X \cup Z}$", fontsize=14)
plt.ylabel(r"$\mathbf{Mean \; runtime \; in  \; seconds}$", fontsize=14)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()

## Baseline data WO

bs_cols = [0,'Mean']
relevant_bs = pd.DataFrame(
    index=baseline_graph_names,
    columns=bs_cols
)
for r in baseline_graph_names:
    df_means = RW_all_mean_runtimes_WO_baseline_dict[r]
    df_sds = RW_all_sd_runtimes_WO_baseline_dict[r]
    df_props = RW_WO_proportions_b[r]
    for c in bs_cols:
        if c == 0:
            mu = round(df_means.loc[1,0], 3)
            sd = round(df_sds.loc[1,0], 3)
            prop = df_props.loc[1,0]
            relevant_bs.loc[r,c] = (mu, sd, prop)
        elif c == 'Mean':
           mu = round(RW_overall_means_WO_dict[r]['b'],3)
           sd = round(RW_overall_sd_WO_dict[r]['b'],3)
           prop = round(np.mean(df_props),3)
           relevant_bs.loc[r,c] = (mu, sd, prop)
relevant_bs.rename(columns={0: (1,0)}, inplace=True)


# Bars T
Tdf_idx = graph_names
Tdf2_cols = ['Mean', 'Largest']
Tdf2 = pd.DataFrame(
    index=Tdf_idx,
    columns=Tdf2_cols
)
for r in Tdf2.index:
    df = RW_all_mean_runtimes_WSO_Tn_dict[r]
    for c in Tdf2_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1_WSO.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        Tdf2.loc[r,c] = round(v, 3)    
Tdf2.rename(columns={0: (1,0)}, inplace=True)

# Bars tot
totn_idx = graph_names
totn_cols = ['Mean', 'Largest']
totn = pd.DataFrame(
    index=totn_idx,
    columns=totn_cols
)
for r in totn.index:
    df = RW_all_mean_runtimes_WSO_tot_n_dict[r]
    for c in totn_cols:
        if c == 'Mean':
           v = float(np.mean(df))
        else:
            i = relevant_rts1_WSO.loc[r,'Largest'][1]
            v = df.loc[i[0], i[1]]
        totn.loc[r,c] = round(v, 3)    
totn.rename(columns={0: (1,0)}, inplace=True)


list_pairs = [f"{relevant_rts1_WSO.loc[dag,'Largest'][1]}" for dag in graph_names]

xti = [f"{dag}\n {pair}" for dag, pair in zip(graph_names, list_pairs)]


df1 = totn
df2 = Tdf2

for col in df1.columns:
    plt.figure()
    if col == 'Largest':
        x = xti
    else:
        x = [rf"$\mathbf{{{idx}}}$" for idx in df1.index]
    plt.bar(x, df1[col], label = r"$\mathbf{Identification \; of \; d-separated \; nodes}$")
    plt.bar(x, df2[col], label = r"$\mathbf{d-collision \; graph \; generation}$")
    plt.xticks(rotation=45, fontsize=10, fontweight = 'bold')
    plt.yticks(fontsize=10, fontweight='bold')   
    plt.tight_layout()
    if col == "Mean":
        tit = r"$\mathbf{Overall \; mean  \; runtimes}$"
    else:
        tit = r"$\mathbf{Largest \; mean \; runtimes}$"
    plt.title(tit, fontsize=16)

    plt.ylabel(r"$\mathbf{Mean \; runtime \; in  \; seconds}$", fontsize=14)

    plt.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 0.85, 0.2), reverse=True, fontsize=13)
    plt.show()


###BARS

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

ax1, ax2 = axes

df1 = totn
df2 = Tdf2

# x = [rf"$\mathbf{{{idx}}}$" for idx in df1.index]
x = df1.index
ax1.bar(x, df1['Mean'], label = r"Identification of d-separated nodes")
ax1.bar(x, df2['Mean'], label = r"d-collision graph generation")
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.set_title(r"Overall mean runtimes", fontsize=14)
ax1.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 0.86, 0.2), reverse=True, fontsize=10.7)

x = [rf"$\mathbf{{{idx}}}$" for idx in df1.index]
ax2.bar(xti, df1['Largest'], label = r"Identification of d-separated nodes")
ax2.bar(xti, df2['Largest'], label = r"d-collision graph generation")
ax2.tick_params(axis='x', rotation=45, labelsize=9.5)
ax2.set_title(r"Largest mean runtimes", fontsize=14)
ax2.legend(loc='upper center', bbox_to_anchor=(0, 0.8, 0.86, 0.2), reverse=True, fontsize=10.7)
ax2.tick_params(axis='y', labelleft=True)


# for ax in (ax1, ax2):
#     for lbl in ax.get_xticklabels() + ax.get_yticklabels():
#         lbl.set_fontweight('bold')

fig.supylabel("Mean runtime in sec.", y=0.59, fontsize=15)
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "RW_bars.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
img = PIL.Image.open(out_dir / "RW_bars.png")
img.show()



####LCS

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax1, ax2 = axes

## Mean runtimes T WO

df = RW_means_over_dim_Zfix_WSO
df.rename(columns=new_cols, inplace=True)
x = df.columns.astype(str)   # convert tuple → string
for name in graph_names:
    y = df.loc[name].astype(float)
    ax1.plot(x, y, marker='o', label=name)
ax1.set_title(r"d-collision graph generation", fontsize=14)
ax1.set_xlabel(r"Proportion of nodes in the input Z", fontsize=16)
ax1.tick_params(axis='x', labelsize=11)
#ax1.legend(loc='upper left', ncol=3)


## Mean runtimes Qn WO

df = RW_means_over_dim_Qn_WSO
df.rename(columns=new_cols2, inplace=True)
x = df.columns.astype(str)   # convert tuple → string

for name in graph_names:
    y = df.loc[name].astype(float)
    ax2.plot(x, y, marker='o', label=name)

ax2.set_title(r"Identification of d-separated nodes", fontsize=14)
ax2.set_xlabel(r"Proportion of nodes  in the input $X \cup Z$", fontsize=16)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelleft=True)


#ax2.legend(loc='upper left', ncol=3)

for ax in (ax1, ax2):
    # for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    #     lbl.set_fontweight('bold')
    ax.grid(True, axis='y', alpha=0.3)
        
fig.supylabel("Mean runtime in sec.", y=0.46, fontsize=15)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=5,
    frameon=False,
    prop={'size': 12},
)

fig.tight_layout(rect=[0, 0, 1, 0.82])

fig.savefig(os.path.join(out_dir, "RW_lcs.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
img = PIL.Image.open(out_dir / "RW_lcs.png")
img.show()