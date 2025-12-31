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

# with open(all_runtimes_dir / "RW_all_runtimes_WO_T_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_T_dict = pickle.load(f)

# ## Read query runtimes

# # Native
# with open(all_runtimes_dir / "RW_all_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_Qn_dict = pickle.load(f)
# # APOC
# with open(all_runtimes_dir / "RW_all_runtimes_WO_Qa_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_Qa_dict = pickle.load(f)
    
# ## Read total runtimes

# # Native
# with open(all_runtimes_dir / "RW_all_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_tot_n_dict = pickle.load(f)
# # APOC
# with open(all_runtimes_dir / "RW_all_runtimes_WO_tot_a_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_tot_a_dict = pickle.load(f)

# ## Baseline runtimes
# with open(all_runtimes_dir / "RW_all_runtimes_WO_baseline_dict.pkl", "rb") as f:
#     RW_all_runtimes_WO_baseline_dict = pickle.load(f)

### Read mean runtimes

## Read mean transformation runtimes

# All
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Tn_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_T_dict = pickle.load(f)
    
# |Z| fixed
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Tn_Zfix.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_T_Zfix = pickle.load(f)

## Read mean query runtimes

# Native
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qn_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_Qn_dict = pickle.load(f)

# APOC
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_Qa_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_Qa_dict = pickle.load(f)

## Read mean total runtimes

# Native
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_n_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_tot_n_dict  = pickle.load(f)

# APOC
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_tot_a_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_tot_a_dict  = pickle.load(f)
    
# Baseline
with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_baseline_dict.pkl", "rb") as f:
    RW_all_mean_runtimes_WO_baseline_dict  = pickle.load(f) 


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

"""
with open(BASE / "Results/dim_dict.pkl", "rb") as f:
    dim_dict = pickle.load(f)
    

rrts_idx = RW_all_mean_runtimes_WO_T_Zfix.columns
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



rrts_cols1 = [0,'Mean','Slowest']
relevant_rts1 = pd.DataFrame(
    index=rrts_idx,
    columns=rrts_cols1
)
for r in rrts_idx:
    df = RW_all_mean_runtimes_WO_tot_n_dict[r]
    N = dim_dict[r]['V']
    for c in rrts_cols1:
        if c == 0:
            relevant_rts1.loc[r,c] = df.loc[1,0]  
        elif c == 'Mean':
           mu = float(np.mean(df))
           relevant_rts1.loc[r,c] = mu
        else:
            w = float(np.max(df))
            matches = df.where(df == w).stack()
            i = list(matches.index)[0]
            relevant_rts1.loc[r,c] = (w, i)
            
relevant_rts1.rename(columns={0: (1,0)}, inplace=True)

df = relevant_rts0
x = df.columns.astype(str)   # convert tuple → string

plt.figure(figsize= (8,6))
for name in graph_names:
    y = df.loc[name].astype(float)
    plt.plot(x, y, marker='o', label=name)

plt.title("Trend of the mean total runtime")
plt.xlabel("Sizes of X and Z")
plt.grid(True)
plt.legend()
plt.show()


bs_cols = [0,'Mean']
relevant_bs = pd.DataFrame(
    index=baseline_graph_names,
    columns=bs_cols
)
for r in baseline_graph_names:
    df = RW_all_mean_runtimes_WO_baseline_dict[r]
    for c in bs_cols:
        if c == 0:
            relevant_bs.loc[r,c] = round(df.loc[1,0], 3)
        elif c == 'Mean':
           mu = float(np.mean(df))
           relevant_bs.loc[r,c] = round(mu, 3)
relevant_bs.rename(columns={0: (1,0)}, inplace=True)


Tdf_idx = graph_names
Tdf_cols = [0, 5, 8, 'Mean']
Tdf = pd.DataFrame(
    index=Tdf_idx,
    columns=Tdf_cols
)
for r in Tdf.index:
    df = RW_all_mean_runtimes_WO_T_dict[r]
    for c in Tdf_cols:
        if c == 0:
            mu = float(np.mean(df.loc[:, 0]))
        elif c == 5 or c==8:
            n = dim_dict[r][c][0]
            mu = float(np.mean(df.loc[:, n]))
        else:
            mu = float(np.mean(df))
        Tdf.loc[r,c] = round(mu, 3)    
Tdf.rename(columns={0: '0%', 5: '50%', 8: '80%'}, inplace=True)


Tdf_idx = graph_names
Tdf2_cols = [0, 'Mean', 'Slowest']
Tdf2 = pd.DataFrame(
    index=Tdf_idx,
    columns=Tdf2_cols
)
for r in Tdf2.index:
    df = RW_all_mean_runtimes_WO_T_dict[r]
    for c in Tdf2_cols:
        if c == 0:
            v = df.loc[1,0]
        elif c == 'Mean':
            v = float(np.mean(df))
        else:
            i = relevant_rts1.loc[r,'Slowest'][1]
            v = df.loc[i[0], i[1]]
        Tdf2.loc[r,c] = round(v, 3)    
Tdf2.rename(columns={0: (1,0)}, inplace=True)


totn_idx = graph_names
totn_cols = [0, 'Mean', 'Slowest']
totn = pd.DataFrame(
    index=totn_idx,
    columns=totn_cols
)
for r in totn.index:
    df = RW_all_mean_runtimes_WO_tot_n_dict[r]
    for c in totn_cols:
        if c == 0:
            v = df.loc[1,0]
        elif c == 'Mean':
            v = float(np.mean(df))
        else:
            i = relevant_rts1.loc[r,'Slowest'][1]
            v = df.loc[i[0], i[1]]
        totn.loc[r,c] = round(v, 3)    
totn.rename(columns={0: (1,0)}, inplace=True)

xti = [f"{dag}\n {relevant_rts1.loc[dag,'Slowest'][1]}" for dag in graph_names]
graph_names = ['SACHS', 'C01', 'C02', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']

df1 = totn
df2 = Tdf2

for col in df1.columns:
    plt.figure()
    if col == 'Slowest':
        x = xti
    else:
        x = df1.index
    plt.bar(x, df1[col], label = 'Identification')
    plt.bar(x, df2[col], label = 'Transformation')
    plt.xticks(rotation=45)   # 45 degrees
    plt.tight_layout()
    if col == (1,0):
        tit = "Mean runtimes for (|X|,|Z|)=(1,0)"
    elif col == "Mean":
        tit = "Overall mean runtimes"
    else:
        tit = "Slowest mean runtimes"
    plt.title(tit)
    plt.legend()
    plt.show()