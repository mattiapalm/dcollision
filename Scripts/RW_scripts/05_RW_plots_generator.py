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
outliers_dir = runtimes_dir / "Outliers"
out_dir = BASE / "Results/Plots"
os.makedirs(out_dir, exist_ok=True)


########--------------- Load runtimes' files ---------------########


### d-collision Graph Method VS Baseline
with open(mean_runtimes_dir / "DCGM_VS_Baseline", "rb") as f:
    DCGM_VS_Baseline = pickle.load(f)

### Read mean runtimes

## Read mean d-collision graph generation runtimes

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

## Read mean Identification of d-separate nodes runtimes

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
    
# # Baseline
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_baseline_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_baseline_dict  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_all_mean_runtimes_WO_baseline_dict.pkl", "rb") as f:
#     RW_all_mean_runtimes_WO_baseline_dict  = pickle.load(f)
    
# Overall means
# with open(mean_runtimes_dir / "RW_overall_means_dict.pkl", "rb") as f:
#     RW_overall_means_dict  = pickle.load(f)
# with open(mean_runtimes_dir / "RW_overall_means_WO_dict.pkl", "rb") as f:
#     RW_overall_means_WO_dict  = pickle.load(f)
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

### Read standard deviations of the runtimes

## Load standard deviations of d-collision graph generation runtimes

# # All
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Tn_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_Tn_dict = pickle.load(f)
    
# # |Z| fixed
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Tn_Zfix.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_T_Zfix = pickle.load(f)

## Load standard deviations of Identification of d-separate nodes runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_Qn_dict = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_Qa_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_Qa_dict = pickle.load(f)

## Load standard deviations of total runtimes

# # Native
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_tot_n_dict  = pickle.load(f)

# # APOC
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_tot_a_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_tot_a_dict  = pickle.load(f)
    
# # Baseline
# with open(var_runtimes_dir / "RW_all_sd_runtimes_baseline_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_baseline_dict  = pickle.load(f)
# with open(var_runtimes_dir / "RW_all_sd_runtimes_WO_baseline_dict.pkl", "rb") as f:
#     RW_all_sd_runtimes_WO_baseline_dict  = pickle.load(f)
    
# # Overall standard deviations
# with open(var_runtimes_dir / "RW_overall_sd_dict.pkl", "rb") as f:
#     RW_overall_sd_dict  = pickle.load(f)   
# with open(var_runtimes_dir / "RW_overall_sd_WO_dict.pkl", "rb") as f:
#     RW_overall_sd_WO_dict  = pickle.load(f)


## Load proportions of WO runtimes

# # Native
# with open(outliers_dir / "RW_WO_proportions_n.pkl", "rb") as f:
#     RW_WO_proportions_n  = pickle.load(f)

# # APOC
# with open(outliers_dir / "RW_WO_proportions_a.pkl", "rb") as f:
#     RW_WO_proportions_a  = pickle.load(f)
    
# Baseline
with open(outliers_dir / "RW_WO_proportions_b.pkl", "rb") as f:
    RW_WO_proportions_b  = pickle.load(f)
    
##########========================================================############

### Graphs' names

graph_names = ['SACHS', 'C01', 'C02', 'COVID', 'BARLEY', 'WIN95PTS', 'CNSDAG', 'LINK', 'MUNIN']
baseline_graph_names = ['SACHS', 'C01', 'C02']


for key in DCGM_VS_Baseline.keys():
    if key == "DCGM":
        to_be_printed = "Performance with outliers of the d-collision Graph Method:"
    elif key == "DCGM_WO":
        to_be_printed = "Performance without outliers of the d-collision Graph Method:"
    elif key == "Baseline":
        to_be_printed = "Performance with outliers of the Baseline:"
    else:
        to_be_printed = "Performance without outliers of the Baseline:"
    print(to_be_printed)
    print(DCGM_VS_Baseline[key], "\n")
    


with open(BASE / "Results/dim_dict.pkl", "rb") as f:
    dim_dict = pickle.load(f)


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



###BARS

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

ax1, ax2 = axes

df1 = totn
df2 = Tdf2

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


for ax in (ax1, ax2):
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