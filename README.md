# **HMRLBA** Code Repository

This is the code repository of **HMRLBA**, it is a novel hierarchical protein multi-scale representation learning model for protein-ligand binding affinity prediction.

---

The folders in the HMRLBA repository:

- **configs**: Parameters for data preprocessing and model training.

- **Datasets**: Dataset used (PDBbind) and the three split configurations; Hard samples; Drugs used in virtual screening. (see **3. Dataset and PLMs Download**)

- **hmrlba_code**: Main code file for the HMRLBA model.

- **PLMs**: Storage for three different PLMs.

- **scripts**: Scripts for data preprocessing, graph generation, model training, and testing**.**

- **Experiments:** Model training result save file.

- **SOTA**: Comparative methods used in the contrast experiments:

  ​	DeepDTA: https://github.com/hkmztrk/DeepDTA
  
  ​	MGraphDTA: https://github.com/guaguabujianle/MGraphDTA
  
  ​	IEConv: https://github.com/phermosilla/IEConv_proteins
  
  ​	PSICHIC: https://github.com/huankoh/PSICHIC
  
  ​	HoloProt: https://github.com/vsomnath/holoprot
  
  ​	MaSIF: https://github.com/LPDI-EPFL/masif
  
  ​	HaPPy: https://github.com/Jthy-af/HaPPy

---



### **Step-by-step Running:**

## 1. Environment Install

It is recommended to use the conda environment (python 3.7), mainly installing the following dependencies:

- [ ] ​		**pytorch (1.9.0)、torch-geometric (2.0.4)、dgl-cu111 (0.6.1)、cudatoolkit (11.1.74)**

- [ ] ​		**[msms](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1)、[dssp](https://swift.cmbi.umcn.nl/gv/dssp/) (3.0.0)、[blender](https://www.blender.org/) (3.5.1)、pdb2pqr (2.1.1) 、biopython (1.79)、rdkit (2023.3.1)、transformers (4.24.0)、**

  ​		**wandb (0.15.4)、pymesh2 (0.3)、pdbfixer (1.6)**

See environment.yaml for details.




## 2. Environment Variables

You need to change these environment variables according to your installation path.

```
export PROT=/mnt/disk/hzy/HMRLBA (project root)
export DSSP_BIN=dssp
export MSMS_BIN=/home/ubuntu/anaconda3/envs/pyg/bin/msms
export BLENDER_BIN=/home/ubuntu/anaconda3/envs/pyg/lib/python3.7/site-packages/blender-3.5.1-linux-x64/blender
export PATH="/home/ubuntu/anaconda3/bin:$PATH"
export PATH="/home/ubuntu/anaconda3/envs/pyg/lib/python3.7/site-packages/blender-3.5.1-linux-x64:$PATH"
export PYTHONPATH="${PYTHONPATH}:/mnt/disk/hzy/HMRLBA"
```



## 3. Dataset and PLMs Download

Download the dataset and unzip it to the corresponding folder:

- ​		/Datasets/Raw_data:  https://figshare.com/articles/dataset/HMRLBA-dataset-pdbbind_tar_gz/27644664?file=50343855 

  ​											  or  https://zenodo.org/records/14061991

Download PLMs to the corresponding folder:

- ​		/PLMs /ankh:  https://huggingface.co/ElnaggarLab/ankh-large/tree/main
- ​		/PLMs /esm1b:  https://huggingface.co/facebook/esm1b_t33_650M_UR50S/tree/main
- ​		/PLMs /prottrans:  https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main



## 4. Data Preprocessing

Calculate protein secondary structure (dssp), and generate protein surface mesh, for subsequent **graph** **generation**.

```
python -W ignore scripts/preprocess/run_binaries.py --dataset pdbbind --tasks all
```

The generated files are stored in the original dataset path：`./Datasets/Raw_data/pdbbind/pdb_files/...`



## 5.  Graph Generation

Construct graphs using three different PLMs

```
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm esm1b
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm ankh
python -W ignore scripts/preprocess/prepare_graphs.py --dataset pdbbind --prot_mode surface2backbone --plm prottrans
```

The graph data is stored in： `./Datasets/processed/pdbbind/surface2backbone/...`



## 6.  Training and Testing

We use wandb to track out experiments. Please make sure to have the [setup](https://docs.wandb.ai/quickstart) complete before doing that.

Modify the following content in `scripts/train/run_model.py` and set it to your wandb account:

```
wandb.init(project='HMRLBA', dir=args.out_dir,
           entity='xxx', ## change here
           config=args.config_file)
```

The experiment config files are organized as `configs/Model_training/pdbbind/SPLIT.yaml` where `SPLIT` is one of `{identity30, identity60, scaffold}`.

- Training model, taking identity30 dataset as an example:


```
python scripts/train/run_model.py --config_file configs/Model_training/pdbbind/identity30.yaml
```

- Testing model:

```
python scripts/eval/eval_model.py --exp_name run-20240516_224208-z1bjc3ku   
```

**exp_name** Change to the name of the model training result folder in `/Experiments/...`























