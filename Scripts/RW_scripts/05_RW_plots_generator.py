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

###----------------------###

# Base path
BASE = Path(__file__).resolve().parent.parent.parent

# Path to subfolders
runtimes_dir = BASE / "Results/Runtimes"
all_runtimes_dir = runtimes_dir / "All_runtimes"
mean_runtimes_dir = runtimes_dir / "Mean_runtimes"
var_runtimes_dir = runtimes_dir / "Variances_of_the_runtimes"

########--------------- Load runtimes' files ---------------########

# ### Read all runtimes
    
# ## Read transformation runtimes

# with open(all_runtimes_dir / "RW_all_runtimes_T_dict.pkl", "rb") as f:
#     RW_all_runtimes_T_dict = pickle.load(f)

# ## Read query runtimes

# # Native
# with open(all_runtimes_dir / "RW_all_runtimes_Qn_dict.pkl", "rb") as f:
#     RW_all_runtimes_Qn_dict = pickle.load(f)
# # APOC
# with open(all_runtimes_dir / "RW_all_runtimes_Qa_dict.pkl", "rb") as f:
#     RW_all_runtimes_Qa_dict = pickle.load(f)
    
# ## Read total runtimes

# # Native
# with open(all_runtimes_dir / "RW_all_runtimes_tot_n_dict.pkl", "rb") as f:
#     RW_all_runtimes_tot_n_dict = pickle.load(f)
# # APOC
# with open(all_runtimes_dir / "RW_all_runtimes_tot_a_dict.pkl", "rb") as f:
#     RW_all_runtimes_tot_a_dict = pickle.load(f)

# ## Baseline runtimes
# with open(all_runtimes_dir / "RW_all_runtimes_baseline_dict.pkl", "rb") as f:
#     RW_all_runtimes_baseline_dict = pickle.load(f)

### Read mean runtimes

## Read mean transformation runtimes

# All
with open(mean_runtimes_dir / "RW_all_mean_runtimes_T_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_T_dict = pickle.load(f)
    
# |Z| fixed
with open(mean_runtimes_dir / "RW_all_mean_runtimes_T_Zfix.pkl", "rb") as f:
    RW_all_mean_runtimes_T_Zfix = pickle.load(f)

## Read mean query runtimes

# Native
with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qn_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_Qn_dict = pickle.load(f)

# APOC
with open(mean_runtimes_dir / "RW_all_mean_runtimes_Qa_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_Qa_dict = pickle.load(f)

## Read mean total runtimes

# Native
with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_n_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_tot_n_dict  = pickle.load(f)

# APOC
with open(mean_runtimes_dir / "RW_all_mean_runtimes_tot_a_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_tot_a_dict  = pickle.load(f)
    
# Baseline
with open(mean_runtimes_dir / "RW_all_mean_runtimes_baseline_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_baseline_dict  = pickle.load(f) 


# ### Read variances of the runtimes

# ## Read variances of transformation runtimes

# # All
# with open(var_runtimes_dir / "RW_all_var_runtimes_T_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_T_dict = pickle.load(f)
    
# # |Z| fixed
# with open(var_runtimes_dir / "RW_all_var_runtimes_T_Zfix.pkl", "rb") as f:
#     RW_all_var_runtimes_T_Zfix = pickle.load(f)

# ## Read variances of query runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_var_runtimes_Qn_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_Qn_dict = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_var_runtimes_Qa_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_Qa_dict = pickle.load(f)

# ## Read variances of total runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_var_runtimes_tot_n_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_tot_n_dict  = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_var_runtimes_tot_a_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_tot_a_dict  = pickle.load(f)
    
# # Baseline
# with open(var_runtimes_dir / "RW_all_var_runtimes_baseline_dict.pkl", "rb") as f:
#     RW_all_var_runtimes_baseline_dict  = pickle.load(f) 

    
##########========================================================############

### Graphs' names

graph_names = ['SACHS', 'CHILD', 'BARLEY', 'WIN95PTS', 'LINK', 'MUNIN', 'SMALLCOVID', 'REDUCEDCOVID', 'COVID', 'CNSAMPLEDAG']
baseline_graph_names = ['SACHS', 'CHILD', 'SMALLCOVID', 'REDUCEDCOVID']

########--------------- PLOTS ---------------########

########--------------- Native VS Baseline ---------------########

for name in baseline_graph_names:
    
    df1 = RW_all_mean_runtimes_tot_n_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_baseline_dict[name]     # second dataframe
    
    print(name)

    for row_idx in df1.index:
        
        y1 = df1.loc[row_idx]
        y2 = df2.loc[row_idx]
    
        mask = ~df1.loc[row_idx].isna()   # use original mask
        x = df1.columns[mask].astype(float)
    
        plt.figure()
        plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
        plt.plot(x, y2[mask], 'o--', color='red', label='Baseline')
    
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
    
        plt.title(name + f" |Z| = {col}")
        plt.xlabel("|X|")
        plt.ylabel("mean t")
        plt.xlim(left=0, right=1.05*x.max())
        plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
        plt.legend()
        plt.grid(True)
        plt.show()
    
########------––-- Native VS Baseline with row/column fixed ----––----########
    
for name in baseline_graph_names:
    
    df1 = RW_all_mean_runtimes_tot_n_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_baseline_dict[name]     # second dataframe
    
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
            x = df.index[mask]
            y = y_full[mask]
        
            plt.plot(x, y, label=f"{col}", color=colors[idx])
        
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
            x = df.columns[mask].astype(float)
            y = y_full[mask]
        
            plt.plot(x, y, label=f"{row}", color=colors[idx])

        
        plt.xlabel("|Z|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |X| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        

########------––-- Native - Transformation only ----––----########

RW_mean_T_Zfix_interpolated = RW_all_mean_runtimes_T_Zfix.interpolate(method='index', limit_area='inside')

### |Z| fixed
for col in graph_names:
    plt.figure()
    plt.plot(RW_mean_T_Zfix_interpolated.index, RW_mean_T_Zfix_interpolated[col], marker='o')
    plt.title(f'Transformation runtime for {col}')
    plt.xlabel('|Z|')
    plt.ylabel('mean rt')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()
        
# ### PLOTS FIND Y_ALL QUERY

# ### LINEAR INTERPOLATION

# ## LI Native and APOC comparison

# for name in graph_names:
    
#     df1 = RW_all_mean_runtimes_Qn_dict[name]     # first dataframe
#     df2 = RW_all_mean_runtimes_Qa_dict[name]     # second dataframe

    
#     for row_idx in df1.index:
        
#         y1 = df1.loc[row_idx]
#         y2 = df2.loc[row_idx]
    
#         mask = ~df1.loc[row_idx].isna()   # use original mask
#         x = df1.columns[mask].astype(float)
    
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
    
    df1 = RW_all_mean_runtimes_Qn_dict[name]     # first dataframe
    df2 = RW_all_mean_runtimes_Qa_dict[name]     # second dataframe
    
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
            x = df.index[mask]
            y = y_full[mask]
        
            plt.plot(x, y, label=f"{col}", color=colors[idx])
        
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
            x = df.columns[mask].astype(float)
            y = y_full[mask]
        
            plt.plot(x, y, label=f"{row}", color=colors[idx])

        
        plt.xlabel("|Z|")
        plt.ylabel("mean t")
        plt.title(k + ' ' + name + " |X| fixed")
        plt.grid(True)
        plt.legend()
        plt.show()
        
### Main comparisons

dict_sachs = {'|X| fixed' : [1,6,2], '|Z| fixed' : [0,2]}
dict_child = {'|X| fixed' : [1,2], '|Z| fixed' : [1,7]}
dict_smallcovid = {'|X| fixed' : [1,9], '|Z| fixed': [0,1,2]}
dict_reducedcovid = {'|X| fixed' : [1,9], '|Z| fixed': [0,1,2]}
dict_covid = {'|X| fixed' : [1,10,22], '|Z| fixed' : [0,2]}
dict_barley = {'|X| fixed' : [1,9,33], '|Z| fixed' : [0,9,28]}
dict_win95pt = {'|X| fixed' : [1,2,45,53], '|Z| fixed' : [0,15]}
dict_cnsampledag = {'|X| fixed' : [1,16,83,116], '|Z| fixed' : [0,16,116]}
dict_link = {'|X| fixed' : [1,71,642], '|Z| fixed' : [0,71,642]}
dict_munin = {'|X| fixed' : [1,104,936], '|Z| fixed' : [0,104,936]}

dict_fixed = {}
dict_fixed['SACHS'] = dict_sachs
dict_fixed['CHILD'] = dict_child
dict_fixed['SMALLCOVID'] = dict_smallcovid
dict_fixed['REDUCEDCOVID'] = dict_reducedcovid
dict_fixed['COVID'] = dict_covid
dict_fixed['BARLEY'] = dict_barley
dict_fixed['WIN95PTS'] = dict_win95pt
dict_fixed['CNSAMPLEDAG'] = dict_cnsampledag
dict_fixed['LINK'] = dict_link
dict_fixed['MUNIN'] = dict_munin


for name in graph_names:
    
    df1 = RW_all_mean_runtimes_Qn_dict[name]
    df2 = RW_all_mean_runtimes_Qa_dict[name]
    
    rows_to_keep = [r for r in dict_fixed[name]['|X| fixed'] if r in df1.index]
    cols = df1.columns  # full set of Z values

    # Interpolate along columns (axis=1)
    norm_interp = df1.loc[rows_to_keep].interpolate(axis=1, limit_area='inside')
    apoc_interp = df2.loc[rows_to_keep].interpolate(axis=1, limit_area='inside')

    plt.figure(figsize=(8, 5))

    for r in rows_to_keep:
        plt.plot(cols, norm_interp.loc[r], marker='o', label=f"|X|={r} Native")
        plt.plot(cols, apoc_interp.loc[r], marker='s', linestyle='--', label=f"|X|={r} APOC")

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

    plt.figure(figsize=(8, 5))

    for c in cols_to_keep:
        plt.plot(rows, norm_interp[c], marker='o', label=f"|Z|={c} Native")
        plt.plot(rows, apoc_interp[c], marker='s', linestyle='--', label=f"|Z|={c} APOC")

    plt.title(f"{name}: Runtime Comparison (|X| varies)")
    plt.xlabel("|X|")
    plt.ylabel("runtime")
    plt.grid(True)
    plt.legend()
    plt.show()


