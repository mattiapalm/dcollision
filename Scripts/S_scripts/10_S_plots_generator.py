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
with open(mean_runtimes_dir / "S_all_mean_runtimes_T_dict.pkl", "rb") as f:
    S_all_mean_runtimes_T_dict = pickle.load(f)
# with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_T_dict.pkl", "rb") as f:
#     S_all_mean_runtimes_WO_T_dict = pickle.load(f)

    
# All
with open(mean_runtimes_dir / "S_all_mean_runtimes_T_dict.pkl", "rb") as f:
    S_all_mean_runtimes_T_dict = pickle.load(f)
# with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_T_dict.pkl", "rb") as f:
#     S_all_mean_runtimes_WO_T_dict = pickle.load(f)


## Read mean query runtimes

# Native
with open(mean_runtimes_dir / "S_all_mean_runtimes_Qn_dict.pkl", "rb") as f:
    S_all_mean_runtimes_Qn_dict = pickle.load(f)
# with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     S_all_mean_runtimes_WO_Qn_dict = pickle.load(f)

## Read mean total runtimes

# Native
with open(mean_runtimes_dir / "S_all_mean_runtimes_tot_n_dict.pkl", "rb") as f:
    S_all_mean_runtimes_tot_n_dict  = pickle.load(f)
# with open(mean_runtimes_dir / "S_all_mean_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     S_all_mean_runtimes_WO_tot_n_dict  = pickle.load(f)
    
    
# # Overall means

# with open(mean_runtimes_dir / "S_overall_means_WO_dict.pkl", "rb") as f:
#     S_overall_means_WO_dict  = pickle.load(f)
    
# # Means over dimension

# with open(mean_runtimes_dir / "S_means_over_dim_WO.pkl", "rb") as f:
#     S_means_over_dim_WO  = pickle.load(f)
    
# with open(mean_runtimes_dir / "S_means_over_dim_Qn_WO.pkl", "rb") as f:
#     S_means_over_dim_Qn_WO  = pickle.load(f)

with open(mean_runtimes_dir / "S_means_over_dim_Zfix.pkl", "rb") as f:
    S_means_over_dim_Zfix  = pickle.load(f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1.pkl", "rb") as f:
    S_means_Qn_over_Z_X1  = pickle.load(f)
    
with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1perc.pkl", "rb") as f:
    S_means_Qn_over_Z_X1perc  = pickle.load(f)

with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1perc.pkl", "rb") as f:
    S_means_Qn_over_X_Z1perc  = pickle.load(f)
    
# with open(mean_runtimes_dir / "S_means_over_dim_Zfix_WO.pkl", "rb") as f:
#     S_means_over_dim_Zfix_WO  = pickle.load(f)
    
# with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1_WO.pkl", "rb") as f:
#     S_means_Qn_over_Z_X1_WO  = pickle.load(f)
    
# with open(mean_runtimes_dir / "S_means_Qn_over_Z_X1perc_WO.pkl", "rb") as f:
#     S_means_Qn_over_Z_X1perc_WO  = pickle.load(f)

# with open(mean_runtimes_dir / "S_means_Qn_over_X_Z1perc_WO.pkl", "rb") as f:
#     S_means_Qn_over_X_Z1perc_WO  = pickle.load(f)
    

# ### Read variances of the runtimes

# ## Read variances of transformation runtimes

# # All
# with open(var_runtimes_dir / "S_all_var_runtimes_WO_T_dict.pkl", "rb") as f:
#     S_all_var_runtimes_WO_T_dict = pickle.load(f)
    
# # |Z| fixed
# with open(var_runtimes_dir / "S_all_var_runtimes_WO_T_Zfix.pkl", "rb") as f:
#     S_all_var_runtimes_WO_T_Zfix = pickle.load(f)

# ## Read variances of query runtimes

# # Native
# with open(var_runtimes_dir / "S_all_var_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     S_all_var_runtimes_WO_Qn_dict = pickle.load(f)

# ## Read variances of total runtimes

# # Native
# with open(var_runtimes_dir / "S_all_var_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     S_all_var_runtimes_WO_tot_n_dict  = pickle.load(f)


### Read standard deviations of the runtimes

## Load standard deviations of transformation runtimes

# # All
# with open(var_runtimes_dir / "S_all_sd_runtimes_T_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_T_dict = pickle.load(f)
# with open(var_runtimes_dir / "S_all_sd_runtimes_WO_T_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_WO_T_dict = pickle.load(f)
    
# # |Z| fixed
# with open(var_runtimes_dir / "S_all_sd_runtimes_T_Zfix.pkl", "rb") as f:
#     S_all_sd_runtimes_T_Zfix = pickle.load(f)
# with open(var_runtimes_dir / "S_all_sd_runtimes_WO_T_Zfix.pkl", "rb") as f:
#     S_all_sd_runtimes_WO_T_Zfix = pickle.load(f)

# ## Load standard deviations of query runtimes

# # Native
# with open(var_runtimes_dir / "S_all_sd_runtimes_Qn_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_Qn_dict = pickle.load(f)
# with open(var_runtimes_dir / "S_all_sd_runtimes_WO_Qn_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_WO_Qn_dict = pickle.load(f)

# ## Load standard deviations of total runtimes

# # Native
# with open(var_runtimes_dir / "S_all_sd_runtimes_tot_n_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_tot_n_dict  = pickle.load(f)
# with open(var_runtimes_dir / "S_all_sd_runtimes_WO_tot_n_dict.pkl", "rb") as f:
#     S_all_sd_runtimes_WO_tot_n_dict  = pickle.load(f)
    
# # Overall standard deviations
    
# with open(var_runtimes_dir / "S_overall_sd_WO_dict.pkl", "rb") as f:
#     S_overall_sd_WO_dict  = pickle.load(f)


## Load proportions of WO runtimes

# Native
with open(proportions_dir / "S_WO_proportions_n.pkl", "rb") as f:
    S_WO_proportions_n  = pickle.load(f)


    
##########========================================================############

### Graphs' names

graph_types = ['BA', 'ER', 'LF', 'TR']
current_run_types = ['BA', 'ER', 'LF', 'TR']
current_run_dim = ['0', '1', '2']
GR0, GR1, GR2 = ['ER0', 'BA0', 'LF0', 'TR0'], ['ER1', 'BA1', 'LF1', 'TR1'], ['ER2', 'BA2', 'LF2', 'TR2']
GER, GBA, GLF, GTR = ['ER0','ER1','ER2'], ['BA0','BA1','BA2'], ['LF0','LF1','LF2'], ['TR0','TR1','TR2']

current_run_names = []
for t in current_run_types:
    for d in current_run_dim:
        name = t+d
        current_run_names.append(name)

########--------------- PLOTS ---------------########


colors20 = plt.cm.tab20c
blues = tuple(reversed(colors20.colors[:3]))
oranges = tuple(reversed(colors20.colors[4:7]))
greens = tuple(reversed(colors20.colors[8:11]))
greys = tuple(reversed(colors20.colors[16:19]))
colors = blues + oranges + greens + greys
dict_colors = {}
i = 0
for GXX in GER, GBA, GLF, GTR:
    for dag in GXX:
        dict_colors[dag] = colors[i]
        i+=1
        
markers = ("o", "^", "s", "P")
dict_markers = {'ER':"o", 'BA':"^", 'LF':"s", 'TR':"P"}

DIC_tot = S_all_mean_runtimes_tot_n_dict
DIC_T = S_all_mean_runtimes_T_dict
# DIC_tot = S_all_mean_runtimes_WO_tot_n_dict
# DIC_T = S_all_mean_runtimes_WO_T_dict
df_largest_tot = pd.DataFrame(
    index=graph_types,
    columns=current_run_dim
)
df_largest_T = pd.DataFrame(
    index=graph_types,
    columns=current_run_dim
)
for t in graph_types:
    for d in current_run_dim:
        dag = t+d
        
        df = DIC_tot[dag]
        w = float(np.max(df))
        matches = df.where(df == w).stack()
        i = list(matches.index)[0] 
        df_largest_tot.loc[t,d] = (round(w,3), i)
        
        df = DIC_T[dag]
        w = float(np.max(df))
        matches = df.where(df == w).stack()
        i = list(matches.index)[0] 
        df_largest_T.loc[t,d] = (round(w,3), i)
    
width=0.25
step = 0.3
x_pos = np.arange(4) * step
for GRD in [GR0, GR1, GR2]:
    d = GRD[0][2]
    labels = [f"{dag}\n {df_largest_tot.loc[dag[:2],d][1]}" for dag in GRD]
    y_tot = [df_largest_tot.loc[dag[:2], d][0] for dag in GRD]
    y_T   = [df_largest_T.loc[dag[:2], d][0] for dag in GRD]
    plt.figure(figsize=(max(3, 4 * 1.2), 4))
    plt.bar(x_pos, y_tot, width=width, label='Id. of d-sep. nodes')
    plt.bar(x_pos, y_T,   width=width, label='d-coll. gr. gen.')
    plt.xticks(
            x_pos,
            labels,
            rotation=45,)
    plt.tight_layout()
    plt.ylabel("Mean runtime in seconds")
    tit = f"Largest runtimes in {GRD}"
    plt.title(tit)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.8, 0.6, 0.2), reverse=True)
    plt.savefig(os.path.join(out_dir, f"S_BC_largest_{d}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    
df_zfix = S_means_over_dim_Zfix
#df_zfix = S_means_over_dim_Zfix_WO
x = df_zfix.columns.astype(str)   # convert tuple → string

for GRD in [GR0, GR1, GR2]:
    d = GRD[0][2]
    plt.figure(figsize= (6,4))
    for dag in GRD:
        y = df_zfix.loc[dag].astype(float)
        plt.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
    plt.title(f"Mean rt for d-col. gr. gen. in {GRD}")
    plt.xlabel("Proportion of nodes in the input $Z$")
    plt.ylabel("Mean runtime in seconds")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"S_LC_GEN_{d}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
df_x1 = S_means_Qn_over_Z_X1
#df_x1 = S_means_Qn_over_Z_X1_WO
x = df_x1.columns.astype(str)   # convert tuple → string

for GRD in [GR0, GR1, GR2]:
    d = GRD[0][2]
    plt.figure(figsize= (6,4))
    for dag in GRD:
        y = df_x1.loc[dag].astype(float)
        plt.plot(x, y, marker=dict_markers[dag[:2]],  color=dict_colors[dag], label=dag)
    plt.title(f"Mean rt for Id. of d-sep. nodes |X|=1 in {GRD}")
    plt.xlabel("Proportion of nodes in the input $Z$")
    plt.ylabel("Mean runtime in seconds")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"S_LC_ID_X1_{d}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

all_dags = GR0 + GR1 + GR2
plt.figure(figsize= (7,4))
for GRD in [GR2, GR1, GR0]:
    d = GRD[0][2]
    for dag in GRD:
        y = df_x1.loc[dag].astype(float)
        y = np.log(y)
        plt.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
plt.title(f"Mean rt for Id. of d-sep. nodes |X|=1 in {GRD}")
plt.xlabel("Proportion of nodes in the input $Z$")
plt.ylabel("Mean runtime in log(seconds)")
plt.grid(True)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(out_dir, "S_LC_ID_X1_LOG.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()
    

df_x1perc = S_means_Qn_over_Z_X1perc
#df_x1 = S_means_Qn_over_Z_X1_WO
x = df_x1perc.columns.astype(str)   # convert tuple → string

plt.figure(figsize=(7,4))
all_dags = GR0 + GR1 + GR2
i = 0
for GRD in [GR2, GR1, GR0]:
    for dag in GRD:
        y = df_x1perc.loc[dag].astype(float)
        plt.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        i += 1

plt.title("Mean rt for Id. of d-sep. nodes |X|= 0.01|V|")
plt.xlabel("Proportion of nodes in the input $Z$")
plt.ylabel("Mean runtime in seconds")
plt.grid(True)

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(out_dir, "S_LC_ID_X1perc.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


df_z1perc = S_means_Qn_over_X_Z1perc
#df_z1perc = S_means_Qn_over_X_Z1perc_WO
x = df_z1perc.columns.astype(str)   # convert tuple → string

plt.figure(figsize=(7, 4))
i = 0
for GRD in [GR2, GR1, GR0]:
    for dag in GRD:
        y = df_z1perc.loc[dag].astype(float)
        plt.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        i += 1

plt.title("Mean rt for Id. of d-sep. nodes |Z|= 0.01|V|")
plt.xlabel("Proportion of nodes in the input $X$")
plt.ylabel("Mean runtime in seconds")
plt.grid(True)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(out_dir, "S_LC_ID_Z1perc_LOG.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
i = 0
for GRD in [GR2, GR1, GR0]:
    for dag in GRD:
        y = df_z1perc.loc[dag].astype(float)
        y = np.log(y)
        plt.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        i += 1

plt.title("Mean rt for Id. of d-sep. nodes |Z|= 0.01|V|")
plt.xlabel("Proportion of nodes in the input $X$")
plt.ylabel("Mean runtime in seconds")
plt.grid(True)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(os.path.join(out_dir, "S_LC_ID_Z1perc_LOG.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

df_zfix = S_means_over_dim_Zfix
#df_zfix = S_means_over_dim_Zfix_WO
x = df_zfix.columns.astype(str)   # convert tuple → string

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)

i=0
for GRD in [GR0, GR1, GR2]:
    ax = axes[i]
    d = GRD[0][2]
    for dag in GRD:
        y = df_zfix.loc[dag].astype(float)
        ax.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        
    #ax.set_title(f"{GRD}")
    ax.set_xlabel("Proportion of nodes in the input $Z$")
    ax.set_ylabel("Mean runtime in seconds")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    i+=1
    
fig.suptitle("Mean runtime for d-collision graph generation", y = 0.82)

# 👇 collect legend handles from ONE axis
handles, labels = axes[0].get_legend_handles_labels()

# leave space for legend
fig.tight_layout(rect=[0, 0, 1, 0.88])

fig.savefig(os.path.join(out_dir, "S_LC_GEN_3in1.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
img = PIL.Image.open(out_dir / "S_LC_GEN_3in1.png")
img.show()



fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

ax1, ax2, ax3 = axes


df_x1perc = S_means_Qn_over_Z_X1perc
#df_x1 = S_means_Qn_over_Z_X1_WO
x = df_x1perc.columns.astype(str)   # convert tuple → string
i = 0
for GRD in [GR2, GR1, GR0]:
    for dag in GRD:
        y = df_x1perc.loc[dag].astype(float)
        ax1.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        i += 1

ax1.set_title("|X|= 0.01|V|")
ax1.set_xlabel("Proportion of nodes in the input $Z$")
ax1.set_ylabel("Mean runtime in seconds")
ax1.grid(True)
#ax1.tight_layout(rect=[0, 0, 0.75, 1])

for GRD in [GR2, GR1, GR0]:
    d = GRD[0][2]
    for dag in GRD:
        y = df_x1.loc[dag].astype(float)
        y = np.log(y)
        ax2.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
ax2.set_title("|X|=1")
ax2.set_xlabel("Proportion of nodes in the input $Z$")
ax2.set_ylabel("Mean runtime in log(seconds)")
ax2.grid(True)
#ax2.tight_layout(rect=[0, 0, 0.75, 1])

df_z1perc = S_means_Qn_over_X_Z1perc
#df_z1perc = S_means_Qn_over_X_Z1perc_WO
x = df_z1perc.columns.astype(str)   # convert tuple → string

i = 0
for GRD in [GR2, GR1, GR0]:
    for dag in GRD:
        y = df_z1perc.loc[dag].astype(float)
        y = np.log(y)
        ax3.plot(x, y, marker=dict_markers[dag[:2]], color=dict_colors[dag], label=dag)
        i += 1

ax3.set_title("|Z|= 0.01|V|")
ax3.set_xlabel("Proportion of nodes in the input $X$")
ax3.set_ylabel("Mean runtime in log(seconds)")
ax3.grid(True)
#ax3.tight_layout(rect=[0, 0, 0.75, 1])


fig.suptitle("Mean runtime for Identification of d-separated nodes", y = 0.90)

# 👇 collect legend handles from ONE axis
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.85),   # 👈 BELOW suptitle
    ncol=len(labels),
    frameon=False,
    fontsize=8
)

# leave space for legend
fig.tight_layout(rect=[0, 0, 1, 0.88])

fig.savefig(os.path.join(out_dir, "S_LC_ID_3in1.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
img = PIL.Image.open(out_dir / "S_LC_ID_3in1.png")
img.show()


        
