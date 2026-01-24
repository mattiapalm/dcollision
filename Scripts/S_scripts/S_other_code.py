import networkx as nx
import json

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
Graph = require("py2neo", "py2neo").Graph
Node = require("py2neo", "py2neo").Node
Relationship = require("py2neo", "py2neo").Relationship
time = require("time")
pickle = require("pickle")



########------––-- Native - Transformation only ----––----########

"""S_mean_T_Zfix_interpolated = S_all_mean_runtimes_WO_T_Zfix.interpolate(method='index', limit_area='inside')

### |Z| fixed
for col in current_run_names:
    plt.figure()
    plt.plot(S_mean_T_Zfix_interpolated.index, S_mean_T_Zfix_interpolated[col], marker='o')
    plt.title(f'Transformation runtime for {col}')
    plt.xlabel('|Z|')
    plt.ylabel('mean rt')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()
        
# ### PLOTS FIND Y_ALL QUERY

# ### LINEAR INTERPOLATION

# for name in current_run_names:
    
#     df1 = S_all_mean_runtimes_WO_Qn_dict[name]     # first dataframe

#     for row_idx in df1.index:
        
#         y1 = df1.loc[row_idx]
#         y2 = df2.loc[row_idx]
    
#         mask = ~df1.loc[row_idx].isna()   # use original mask
#         x = df1.columns[mask].astype(float)
    
#         plt.figure()
#         plt.plot(x, y1[mask], 'o-', color='blue', label='Native')
    
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
    
#         plt.title(name + f" |Z| = {col}")
#         plt.xlabel("|X|")
#         plt.ylabel("mean t")
#         plt.xlim(left=0, right=1.05*x.max())
#         plt.ylim(bottom=0, top=1.15*max(y1.max(), y2.max()))
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
        

########------––-- Native  with row/column fixed ----––----########
    
for name in current_run_names:
    
    df = S_all_mean_runtimes_WO_Qn_dict[name]

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
    plt.title(name + " |Z| fixed")
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
    plt.title(name + " |X| fixed")
    plt.grid(True)
    plt.legend()
    plt.show()


for name in current_run_names:
    
    df1 = S_all_mean_runtimes_WO_Qn_dict[name]
    df2 = S_all_mean_runtimes_Qa_dict[name]
    
    rows_to_keep = [r for r in dict_fixed[name]['|X| fixed'] if r in df1.index]
    cols = df1.columns  # full set of Z values

    # Interpolate along columns (axis=1)
    norm_interp = df1.loc[rows_to_keep].interpolate(axis=1, limit_area='inside')

    plt.figure(figsize=(8, 5))

    for r in rows_to_keep:
        plt.plot(cols, norm_interp.loc[r], marker='o', label=f"|X|={r} Native")

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

    plt.figure(figsize=(8, 5))

    for c in cols_to_keep:
        plt.plot(rows, norm_interp[c], marker='o', label=f"|Z|={c} Native")

    plt.title(f"{name}: Runtime Comparison (|X| varies)")
    plt.xlabel("|X|")
    plt.ylabel("runtime")
    plt.grid(True)
    plt.legend()
    plt.show()

"""