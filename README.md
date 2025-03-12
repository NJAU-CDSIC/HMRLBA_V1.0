# **HMRLBA** Code Repository

This is a repository to deposit the code and data for **HMRLBA** model. **HMRLBA** is a hierarchical multi-scale representation learning model for predicting protein-ligand binding affinity.

---

The folders in the HMRLBA repository:

- **configs**: Parameters for data preprocessing and model training.

- **Datasets**: 

  a. **Raw_data**: Three PDBbind v2019 benchmark  dataset and Enzyme classification dataset.

  b. **Hard_samples**: 21 hard samples.

  c. **Virtual screening**: SMILES strings of 2616 FDA-approved drugs and 18 EGFR inhibitors.

  d. **PDB_id_list**: The protein list of different dataset split.

- **hmrlba_code**: Main code file for the HMRLBA model.

- **PLMs**: Three protein language models - ESM-1b, Ankh, ProtTrans.

- **scripts**: Scripts for data preprocessing, graph generation, model training and testing.

- **Experiments:** Model training result save file.

- **Supplementary Files**: The detailed results for our HMRLBA manuscript.

- **SOTA**: Comparative methods used in the contrast experiments:

  ​	DeepDTA: https://github.com/hkmztrk/DeepDTA

  ​	MGraphDTA: https://github.com/guaguabujianle/MGraphDTA

  ​	IEConv: https://github.com/phermosilla/IEConv_proteins

  ​	RF-score: https://github.com/guaguabujianle/GIGN/tree/main/RF-Score

  ​	PSICHIC: https://github.com/huankoh/PSICHIC

  ​	HoloProt: https://github.com/vsomnath/holoprot

  ​	MaSIF: https://github.com/LPDI-EPFL/masif
  
  ​	HaPPy: https://github.com/Jthy-af/HaPPy

---



### **Step-by-step Running:**

## 1. Environment Installation

It is recommended to use the conda environment (python 3.7), mainly installing the following dependencies:

- [ ] ​		**pytorch (1.9.0)、torch-geometric (1.7.1)、dgl-cu111 (0.6.1)、cudatoolkit (11.1.74)**

- [ ] ​		**[msms](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1)、[dssp](https://swift.cmbi.umcn.nl/gv/dssp/) (3.0.0)、[blender](https://www.blender.org/) (3.5.1)、pdb2pqr (2.1.1) 、biopython (1.79)、rdkit (2023.3.1)、transformers (4.24.0)、**

  ​		**wandb (0.15.4)、pymesh2 (0.3)、pdbfixer (1.6)**

See environment.yaml for details.




## 2. Environment Variables

You need to change these environment variables according to your installation path.

```
export PROT="/mnt/disk/hzy/HMRLBA" (project root)
export DSSP_BIN="dssp"
export MSMS_BIN="/home/ubuntu/anaconda3/envs/pyg/bin/msms"
export BLENDER_BIN="/home/ubuntu/anaconda3/envs/pyg/lib/python3.7/site-packages/blender-3.5.1-linux-x64/blender"
export PATH="/home/ubuntu/anaconda3/bin:$PATH"
export PATH="/home/ubuntu/anaconda3/envs/pyg/lib/python3.7/site-packages/blender-3.5.1-linux-x64:$PATH"
export PYTHONPATH="${PYTHONPATH}:/mnt/disk/hzy/HMRLBA"
```



## 3. Datasets

Download the datasets from the following links:

-  /Datasets/Raw_data:  https://zenodo.org/records/15005823 or https://doi.org/10.6084/m9.figshare.27644664

  

## 4. PLMs

Download the protein language models from the following links:

- ​		/PLMs /ankh:  https://huggingface.co/ElnaggarLab/ankh-large/tree/main
- ​		/PLMs /esm1b:  https://huggingface.co/facebook/esm1b_t33_650M_UR50S/tree/main
- ​		/PLMs /prottrans:  https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main



## 5. Data Preprocessing

Calculate protein secondary structure (dssp), and generate protein surface mesh, for subsequent **graph** **generation**.

```
python -W ignore scripts/preprocess/run_binaries.py --dataset pdbbind --tasks all
```

The generated files are stored in the original dataset path：`./Datasets/Raw_data/pdbbind/pdb_files/...`



## 6.  Graph Generation

Graph reprensentation for protein-ligand pairs.

```
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm esm1b
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm ankh
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm prottrans
```

The graph data is stored in： `./Datasets/processed/pdbbind/surface2backbone/...`



## 7.  Training and Testing

We use wandb to track out experiments. Please make sure to have the [setup](https://docs.wandb.ai/quickstart) complete before doing that.

Modify the following content in `scripts/train/run_model.py` and set it to your wandb account:

```
wandb.init(project='HMRLBA', dir=args.out_dir,
           entity='xxx', ## change here
           config=args.config_file)
```

The experiment config files are organized as `configs/Model_training/pdbbind/SPLIT.yaml` where `SPLIT` is one of `{identity30, identity60, scaffold, casf}`.

- Training model, taking identity30 dataset as an example:


```
python scripts/train/run_model.py --config_file configs/Model_training/pdbbind/identity30.yaml
```

- Testing model:

```
python scripts/eval/eval_model.py --exp_name run-20241124_204606-r94ymd7y
```

The exp_name **run-20241124_204606-r94ymd7y** is a model that has been trained on identity30 dataset.

If you want to test your trained model, change exp_name to the name of the model training result folder in `/Experiments/wandb/...`



## Enzyme Classification Experiment

Similar to the binding affinity prediction task.

```
# Data Preprocessing
python -W ignore scripts/preprocess/run_binaries.py --dataset enzyme --tasks all

# Graph Generation
python -W ignore scripts/preprocess/prepare_graphs.py --dataset enzyme --prot_mode surface2backbone --plm esm1b
python -W ignore scripts/preprocess/prepare_graphs.py --dataset enzyme --prot_mode surface2backbone --plm ankh
python -W ignore scripts/preprocess/prepare_graphs.py --dataset enzyme --prot_mode surface2backbone --plm prottrans

# Training model
python scripts/train/run_model.py --config_file configs/Model_training/enzyme/default_config.yaml
```


