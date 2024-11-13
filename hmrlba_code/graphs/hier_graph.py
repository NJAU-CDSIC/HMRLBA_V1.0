import os
import pickle
import re

from numpy.core.fromnumeric import amin
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.subgraph import subgraph as subgraph_util
from scipy.spatial import KDTree
from typing import List, Dict, Union

from hmrlba_code.data.base import HierData
from hmrlba_code.utils.surface import get_surface
from hmrlba_code.utils.tensor import create_pad_tensor
from hmrlba_code.graphs import Backbone, Surface

from Bio import PDB
from Bio.PDB import PDBParser, Polypeptide

from transformers import logging

logging.set_verbosity_warning()

MSMS_BIN = os.environ['MSMS_BIN']

parser = PDBParser()
pp_builder = Polypeptide.PPBuilder()


def get_resid_from_name(name: str):
    """Converts name of node in mesh to residue_id."""
    entities = name.split("_")
    chain_id = entities[0]

    res_pos = int(entities[1])
    insertion = entities[2]

    if insertion == "x":
        insertion = " "

    res_id = (" ", res_pos, insertion)
    return (chain_id, res_id)


class Surface2Backbone:

    def __init__(self,
                 max_num_neighbors: int = 128,
                 mode: str = 'ca',
                 radius: float = 12.0,
                 sigma: float = 0.01,
                 msms_bin: str = MSMS_BIN):
        self.backbone_builder = Backbone(max_num_neighbors=max_num_neighbors,
                                         mode=mode, radius=radius, sigma=sigma)
        self.surface_builder = Surface(sigma=sigma, msms_bin=msms_bin)
        self.msms_bin = msms_bin

    @staticmethod
    def get_hier_map(names: List[str], resid_to_idx: Dict) -> torch.Tensor:
        """
        resid_to_idx is a dictionary mapping residue id to its idx. Each node in
        the surface graph maps to a residue, which is captured in names.
        get_hier_map returns this mapping of idxs of nodes from the surface
        mapping to corresponding residue indices.
        """
        node_to_resid = [get_resid_from_name(name) for name in names]
        hier_mappings = [[] for i in range(len(resid_to_idx))]
        for node_idx, resid in enumerate(node_to_resid):
            if resid in resid_to_idx:
                residx = resid_to_idx[resid]
                hier_mappings[residx].append(node_idx + 1)  # +1 so that the padded elements are 0
        mapping_lens = [len(res) for res in hier_mappings]
        hier_mappings = create_pad_tensor(hier_mappings)
        return hier_mappings

    def __call__(self, pdb_file: str, target: Union[np.ndarray, torch.Tensor] = None, **kwargs) -> Data:
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str,
              target: Union[float, np.ndarray] = None, **kwargs) -> Data:
        import pymesh
        if 'mesh_file' not in kwargs:
            raise ValueError('mesh file not found.')
        mesh_file = kwargs['mesh_file']
        vertices, faces, normals, names, _ = get_surface(pdb_file=pdb_file,
                                                         msms_bin=self.msms_bin)

        tree = KDTree(data=vertices)
        new_mesh = pymesh.load_mesh(mesh_file)
        closest_idxs = tree.query(new_mesh.vertices)[1]
        node_names = [names[idx] for idx in closest_idxs]

        hier_data = HierData()

        pdb_file = os.path.abspath(pdb_file)
        pdb_base = pdb_file.split("/")[-1]
        pdb_id = ".".join(pdb_base.split(".")[:-1])
        if "fixed" in pdb_file:
            pdb_id = "_".join(pdb_id.split("_")[:-1])
        dssp_file = "/".join(pdb_file.split("/")[:-1] + [f"{pdb_id}.dssp"])

        if not os.path.exists(dssp_file):
            print(f"{pdb_id}: dssp file not found. Returning None")
            return None

        with open(dssp_file, "rb") as f:
            dssp_dict = pickle.load(f)

        struct = parser.get_structure(pdb_id, file=pdb_file)
        polypeptides = pp_builder.build_peptides(struct)
        residues = [res for pp in polypeptides for res in pp]
        # Select only residues that have secondary structure information
        residues = [res for res in residues if res.get_full_id()[2:] in dssp_dict]

        residue_coordinates = []
        # 提取所有残基的三维坐标
        for residue in residues:
            atom_coordinates = np.array([atom.get_coord() for atom in residue.get_atoms()])
            residue_avg_coord = np.mean(atom_coordinates, axis=0)
            residue_coordinates.append(residue_avg_coord)
        # 建立 KD 树
        kdtree = KDTree(residue_coordinates)

        protein_sequence = ""
        for residue in residues:
            # 获取残基的三字母氨基酸代码
            if PDB.is_aa(residue):
                aa_code = PDB.Polypeptide.three_to_one(residue.get_resname())
                protein_sequence += aa_code
            else:
                protein_sequence += '<unk>'

        model = kwargs['prot_model']
        tokenizer = kwargs['prot_tokenizer']
        plm = kwargs['plm']

        emb = []
        if plm == 'esm1b':
            max_input_length = 1022
            for i in range(0, len(protein_sequence), max_input_length):
                sub_sequence = protein_sequence[i:i + max_input_length]
                inputs = tokenizer(sub_sequence, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                sub_esm_emb = last_hidden_states.squeeze().tolist()[1:-1]
                emb += sub_esm_emb
        elif plm == 'prottrans':
            protein_sequence = [protein_sequence]
            protein_sequence = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_sequence]
            inputs = tokenizer(protein_sequence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            emb = last_hidden_states.squeeze().tolist()[0:-1]
        elif plm == 'ankh':
            inputs = tokenizer(protein_sequence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            emb = last_hidden_states.squeeze().tolist()[0:-1]

        surface_data = self.surface_builder(pdb_file=pdb_file, mesh_file=mesh_file, emb=emb, kdtree=kdtree)
        amino_data, residues = self.backbone_builder(pdb_file=pdb_file, emb=emb, return_res=True)
        resid_to_idx = {res.get_full_id()[2:]: idx
                        for idx, res in enumerate(residues)}
        mappings = Surface2Backbone.get_hier_map(node_names, resid_to_idx)

        hier_data.surface = surface_data
        hier_data.backbone = amino_data
        hier_data.mapping = mappings

        def finite_check(x):
            return torch.isfinite(x).all().item()

        checks = [hier_data.backbone is not None, hier_data.surface is not None,
                  finite_check(hier_data.mapping)]
        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            hier_data.y = target
            checks += [finite_check(hier_data.y)]

        if not all(checks):
            print(f"Nan checks failed for hierarchical protein: {pdb_file}", flush=True)
            return None

        return hier_data
