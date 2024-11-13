import gc
import os
from typing import List, Dict, Tuple, Callable
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, Sampler
from itertools import product
import json

from hmrlba_code.data.base import HierData, ComplexData
from hmrlba_code.data.batching import ProtBatch, ComplexBatch
from sklearn.decomposition import PCA


class BaseDataset(Dataset):
    """
    BaseDataset it the dataset class. By default, we first prepare 
    the graphs for each protein and save them to disk. This is then 
    loaded using BaseDataset and combined with torch DataLoader to 
    generate batches. Each subclass of BaseDataset must implement 
    its own load_ids() method, that loads all the pdb_ids associated 
    with training / validation / testing. These ids are then used to 
    load the corresponding graphs. One passes the directory containing 
    the raw and processed data, and the hierarchy being used. Additional
    arguments such as splits can also be passed.
    """

    def __init__(self,
                 mode: str = 'train',
                 raw_dir: str = None,
                 prot_mode: str = None,
                 processed_dir: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        mode: str, (default train)
            Whether the dataset should load train / valid or test proteins
        raw_dir: str,
            Directory where the raw data and target information is stored.
        prot_mode: str,
            Encodes the hierarchy.
        processed_dir: str,
            Directory where processed data is stored.
        """
        self.mode = mode
        self.raw_dir = raw_dir
        self.prot_mode = prot_mode
        self.data_dir = f"{processed_dir}/processed/pdbbind/{self.prot_mode}/ankh"
        self.data_dir2 = f"{processed_dir}/processed/pdbbind/{self.prot_mode}/esm1b"
        self.data_dir3 = f"{processed_dir}/processed/pdbbind/{self.prot_mode}/prottrans"

        self.__dict__.update(**kwargs)
        self.load_ids()

    def load_ids(self):
        raise NotImplementedError("Subclasses must implement for themselves.")

    def add_or_update_target(self, data, idx):
        """Functionality to add a target y or any transformation to it."""
        return data

    def __getitem__(self, idx):
        class_name = self.__class__.__name__.lower()
        item = self.ids[idx]
        if isinstance(item, str):
            pdb_id = item
        else:
            pdb_id, lig_id = item

        if not os.path.exists(f"{self.data_dir}/{pdb_id.upper()}.pth"):
            return None
        if not os.path.exists(f"{self.data_dir2}/{pdb_id.upper()}.pth"):
            return None
        if not os.path.exists(f"{self.data_dir3}/{pdb_id.upper()}.pth"):
            return None

        target_dict = torch.load(f"{self.data_dir}/{pdb_id.upper()}.pth")
        target_dict2 = torch.load(f"{self.data_dir2}/{pdb_id.upper()}.pth")
        target_dict3 = torch.load(f"{self.data_dir3}/{pdb_id.upper()}.pth")

        if 'prot' not in target_dict:
            return None
        if 'prot' not in target_dict2:
            return None
        if 'prot' not in target_dict3:
            return None

        data = target_dict['prot']
        data2 = target_dict2['prot']
        data3 = target_dict3['prot']

        if "pdbbind" in class_name:
            if lig_id not in target_dict:
                return None

            ligand_data = target_dict[lig_id]
            data = ComplexData(protein=data, protein2=data2, protein3=data3, ligand=ligand_data)
        data = self.add_or_update_target(data, idx)
        return data

    def collate_fn(self, data_list):
        data_list = [data for data in data_list if data is not None]
        if not len(data_list):
            return None

        if isinstance(data_list[0], ComplexData):
            return ComplexBatch.from_data_list(data_list, prot_mode=self.prot_mode)

        elif isinstance(data_list[0], HierData):
            return ProtBatch.from_data_list(data_list, prot_mode=self.prot_mode)

    def create_loader(self,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = False,
                      sampler: Sampler = None) -> torch.utils.data.DataLoader:
        if shuffle and sampler is not None:
            print("Overwriting shuffle since sampler is not None")
            shuffle = False
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn
        )

    def __len__(self):
        return len(self.ids)


class PDBBindDataset(BaseDataset):

    def load_ids(self):
        with open(f"{self.raw_dir}/metadata/{self.split}_split.json", "r") as f:
            splits = json.load(f)
        self.ids = splits[self.mode]

        lig_ids = [pdb_id + "_ligand" for pdb_id in self.ids]
        self.ids = list(zip(*(self.ids, lig_ids)))


