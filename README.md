# D-Collision Experiments

This repository contains the implementation and experimental evaluation of the d-collision Graph Method, together with baseline comparisons and runtime analysis.

The experiments are designed to run on a Linux-based virtual machine with Neo4j and Python 3.12.

---

##  System Requirements

The experiments were tested on:

- **OS:** Ubuntu 24.04 LTS  
- **Kernel:** 6.8.x  
- **CPU:** 12-core virtual CPU (KVM, AMD-V)  
- **RAM:** 128 GB  
- **Python:** 3.12.3  
- **Neo4j:** 5.26.x  
- **Java:** OpenJDK 17  

Neo4j was configured with increased memory allocation for large-scale experiments.


## ⚙️ Setup Instructions


```bash

### Clone the Repository
git clone <your-repo-url>
cd <repo-folder>/Scripts

### Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate

### Install requirements
pip install -r requirements.txt

### Start Neo4j
sudo systemctl start neo4j

### Verify
sudo systemctl status neo4j

### Run experimentd
python3 RW_scripts/02_RW_dcollision_experiments.py
```


