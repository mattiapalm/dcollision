
# **d-collision Graph Method for d-separation**

This repository contains code and resources for the research paper "Enabling d-separation for Graph Databases" by Mattia Palmiotto (Lyon 1 University & CNRS Liris), Angela Bonifati (Lyon 1 University, CNRS Liris & IUF), Andrea Mauri (Lyon 1 University & CNRS Liris).

---

## **Description**

The d-separation is a graph-theoretic operator that is fundamental to many tasks of causal analysis. It consists in a criterion to derive the conditional independences of causal variables from a causal DAG (directed acyclic graph).

This project addresses the problem of declaratively computing d-separation in order to empower graph databases, property graph models and queries with causal analysis capabilities.

This repository includes:
- Cypher queries implementing the d-collision Graph Method and the baseline approach `Scripts\queries.py`
- Code to reproduce the experimental evaluation
- Supplemental material associated with the paper

---
## Experimental setup and requirements

The experiments were run on a KVM-based virtual machine with 12 vCPUs, 125 GB RAM, and 1 TB of dedicated storage, running Ubuntu 24.04 LTS. Graph processing used Neo4j 5.26.17 on OpenJDK 17. Neo4j was configured with a 32 GB maximum Java heap and an 80 GB page cache. The experiments were implemented in Python 3.12.3 using a dedicated virtual environment.

The file requirements.txt lists all Python dependencies required to run this project. The environment is configured for GPU-accelerated computation using NVIDIA CUDA libraries. You can install these requirements after cloning the repository:

```
### Clone the Repository
git clone https://github.com/mattiapalm/dcollision/tree/master

### Install requirements
cd dcollision
pip install -r requirements.txt
```


---
## Utilized DAGs

Experiments were conducted on both real-world and synthetic DAGs (see `DAGs` folder).

Real-world DAGs were obtained from:
- [bnlearn](https://www.bnlearn.com/bnrepository/) repository.
- [causalcovid19](https://github.com/zidatalab/causalcovid19) repository.
- [CauseNet](https://causenet.org) benchmark.

The synthetic DAGs are produced with both ad-hoc and `NetworkX` methods.

---
## Reproduction

Scripts are organized as follows:
- `Scripts/RW_script/` → real-world DAG experiments
- `Scripts/S_scripts/` → synthetic DAG experiments

Scripts that execute queries (`RW_02`, `RW_03`, `S_08`) require manual configuration of Neo4j credentials:

```
# Neo4j connection settings
host = "bolt://localhost:7687"
username = "neo4j"
neo4j_psw = "password"
```
Replace these values with your own Neo4j instance settings.

### Full reproduction (from scratch)
If you want to reproduce all experiments entirely from scratch, thus using different randomly generated inputs and synthetic DAGs, you must sequentially run the scripts for the real-world and synthetic DAGs in the order indicated at prefix of the name of the script, that is:

```
01_RW_inputs_generator.py
02_RW_dcollision_experiments.py
03_RW_baseline_experiments.py
04_RW_runtimes_statistics.py
05_RW_plots_generator.py

06_S_DAG_generation.py
07_S_input_generator.py
08_S_dcollision_experiments.py
09_S_runtimes_statistics.py
10_S_plots_generator.py

```

### Reproducing Original Experiments

To reproduce the exact experiments reported in the paper, skip:

```
01_RW_inputs_generator.py
06_S_DAG_generation.py
07_S_input_generator.py
```

---



