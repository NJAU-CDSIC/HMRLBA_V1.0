"""
Preprocess PDBbind 2019 data for MEGDTA.

Input can be either an extracted PDBbind directory or the
`pdbbind_dataset.tar.gz` archive. The script writes the same files used by
graph_loader.py:

    data/pdbbind/updated_full.csv
    data/pdbbind/ligand_to_graph.pkl
    data/pdbbind/ligand_to_ecfp.pkl
    data/pdbbind/protein_to_graph.pkl
    data/pdbbind/folds/train_fold_setting1.txt
    data/pdbbind/folds/test_fold_setting1.txt

RDKit is required for ligand graphs and ECFP fingerprints.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from params import (
    N_CHEM_ECFP,
    N_CHEM_EDGE_FEAT,
    N_CHEM_NODE_FEAT,
    N_PROT_EDGE_FEAT,
    N_PROT_NODE_FEAT,
    SEED,
)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdPartialCharges
except ImportError as exc:
    raise SystemExit(
        "RDKit is required for PDBbind preprocessing. Install it with:\n"
        "  conda install -c conda-forge rdkit\n"
        "or:\n"
        "  pip install rdkit"
    ) from exc


RDLogger.DisableLog("rdApp.*")

AA_CODES = OrderedDict(
    [
        ("ALA", "A"),
        ("ARG", "R"),
        ("ASN", "N"),
        ("ASP", "D"),
        ("CYS", "C"),
        ("GLN", "Q"),
        ("GLU", "E"),
        ("GLY", "G"),
        ("HIS", "H"),
        ("ILE", "I"),
        ("MET", "M"),
        ("LEU", "L"),
        ("LYS", "K"),
        ("PHE", "F"),
        ("PRO", "P"),
        ("SER", "S"),
        ("THR", "T"),
        ("TRP", "W"),
        ("TYR", "Y"),
        ("VAL", "V"),
    ]
)
AA_LIST = list(AA_CODES.values()) + ["X"]
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_LIST[:-1])}
AA_TO_INT["X"] = 0

HYDROPATHY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "M": 1.9,
    "L": 3.8,
    "K": -3.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "X": 0.0,
}
AA_GROUPS = {
    "hydrophobic": set("AVLIMFWY"),
    "polar": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWYH"),
    "small": set("AGSTCP"),
    "tiny": set("AGS"),
    "sulfur": set("CM"),
    "proline": set("P"),
    "charged": set("KRHDE"),
    "aliphatic": set("AVIL"),
    "hbond": set("STNQCYHRKDEW"),
}
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def atom_features(atom):
    rdPartialCharges.ComputeGasteigerCharges(atom.GetOwningMol(), throwOnParamFailure=False)
    charge = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
    try:
        charge = float(charge)
    except ValueError:
        charge = 0.0
    if not math.isfinite(charge):
        charge = 0.0

    features = []
    features.extend(one_hot(atom.GetSymbol(), ATOM_SYMBOLS))
    features.extend(
        [
            float(atom.GetAtomicNum()),
            float(charge),
            float(atom.GetMass()),
        ]
    )
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(
        [
            float(atom.GetTotalValence()),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
        ]
    )
    assert len(features) == N_CHEM_NODE_FEAT
    return features


def bond_features(bond):
    bond_type = bond.GetBondType()
    stereo = bond.GetStereo()
    features = [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(stereo == Chem.rdchem.BondStereo.STEREONONE),
        float(stereo == Chem.rdchem.BondStereo.STEREOANY),
        float(stereo == Chem.rdchem.BondStereo.STEREOZ),
        float(stereo == Chem.rdchem.BondStereo.STEREOE),
        float(bond.GetBeginAtom().GetIsAromatic()),
        float(bond.GetEndAtom().GetIsAromatic()),
    ]
    assert len(features) == N_CHEM_EDGE_FEAT
    return features


def mol_to_graph(mol):
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() < 2:
        raise ValueError("molecule has fewer than 2 atoms")

    nodes = [atom_features(atom) for atom in mol.GetAtoms()]
    edges, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if not edges:
        raise ValueError("molecule has no bonds")
    return [nodes, edges, edge_attr]


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


def mol_to_ecfp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_CHEM_ECFP)
    return [int(x) for x in fp.ToBitString()]


def clean_residue_name(resname):
    return AA_CODES.get(resname.upper(), "X")


def residue_features(aa, residue_index, n_residues):
    aa = aa if aa in AA_LIST else "X"
    rel_pos = residue_index / max(n_residues - 1, 1)
    features = []
    features.extend(one_hot(aa, AA_LIST))
    features.append(float(HYDROPATHY[aa]))
    features.extend(float(aa in group) for group in AA_GROUPS.values())
    features.append(float(rel_pos))
    assert len(features) == N_PROT_NODE_FEAT
    return features


def pdb_ca_records(pdb_path):
    records = []
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            try:
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            records.append((clean_residue_name(resname), coord))
    return records


def pdb_to_protein_graph(pdb_path, contact_threshold=8.0, max_seq_distance_edges=1):
    records = pdb_ca_records(pdb_path)
    if len(records) < 2:
        raise ValueError(f"fewer than 2 CA residues in {pdb_path}")

    sequence = "".join(aa for aa, _ in records)
    coords = np.asarray([coord for _, coord in records], dtype=np.float32)
    distances = squareform(pdist(coords))

    nodes = [residue_features(aa, i, len(records)) for i, (aa, _) in enumerate(records)]
    edges, edge_attr = [], []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            seq_dist = j - i
            dist = float(distances[i, j])
            is_contact = dist <= contact_threshold
            is_backbone = seq_dist <= max_seq_distance_edges
            if not (is_contact or is_backbone):
                continue
            feat = [
                float(seq_dist == 1),
                float(seq_dist <= 5),
                float(5 < seq_dist <= 10),
                float(seq_dist > 10),
                float(dist),
                float(1.0 / max(dist, 1e-6)),
                float(is_contact),
            ]
            edges.extend([[i, j], [j, i]])
            edge_attr.extend([feat, feat])

    if not edges:
        raise ValueError(f"no protein edges built for {pdb_path}")
    return sequence, [nodes, edges, edge_attr]


def sequence_to_encoding(sequence):
    return [AA_TO_INT.get(aa, 0) for aa in sequence]


def find_pdbbind_root(path):
    path = Path(path)
    candidates = [
        path,
        path / "pdbbind",
        path / "Raw_data" / "pdbbind",
        path / "HMRLBA_Datasets" / "Raw_data" / "pdbbind",
    ]
    for candidate in candidates:
        if (candidate / "metadata").is_dir() and (candidate / "pdb_files").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find PDBbind metadata/pdb_files under {path}")


def prepare_input(args):
    if args.pdbbind_dir:
        return find_pdbbind_root(args.pdbbind_dir), None

    archive = Path(args.pdbbind_tar)
    if not archive.is_file():
        raise FileNotFoundError(archive)

    if args.keep_extracted:
        extract_dir = Path(args.keep_extracted)
        extract_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="megdata_pdbbind_")
        extract_dir = Path(cleanup.name)

    print(f"Extracting {archive} to {extract_dir} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    return find_pdbbind_root(extract_dir), cleanup


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_split_ids(split):
    return {part: [str(item).lower() for item in ids] for part, ids in split.items()}


def make_random_split(ids, train_frac=0.8, valid_frac=0.1):
    ids = list(ids)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_frac)
    n_valid = int(len(ids) * valid_frac)
    return {
        "train": ids[:n_train],
        "valid": ids[n_train : n_train + n_valid],
        "test": ids[n_train + n_valid :],
    }


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def preprocess_pdbbind(args):
    pdbbind_root, cleanup = prepare_input(args)
    try:
        metadata = pdbbind_root / "metadata"
        pdb_files = pdbbind_root / "pdb_files"
        affinities = {str(k).lower(): float(v) for k, v in load_json(metadata / "affinities.json").items()}
        smiles_map = {
            str(k).replace("_ligand", "").lower(): v
            for k, v in load_json(metadata / "lig_smiles.json").items()
        }

        split_path = metadata / f"{args.split}_split.json"
        if split_path.exists():
            split = normalize_split_ids(load_json(split_path))
        else:
            split = make_random_split(affinities.keys())

        candidate_ids = [pdb_id for pdb_id in affinities if pdb_id in smiles_map]
        if args.max_complexes:
            candidate_ids = candidate_ids[: args.max_complexes]

        output_dir = Path(args.output_dir)
        folds_dir = output_dir / "folds"
        output_dir.mkdir(parents=True, exist_ok=True)
        folds_dir.mkdir(exist_ok=True)

        rows = []
        ligand_to_graph, ligand_to_ecfp, protein_to_graph = {}, {}, {}
        proteins_txt, ligands_txt = {}, {}
        skipped = []

        for pdb_id in tqdm(candidate_ids, desc="PDBbind complexes"):
            complex_dir = pdb_files / pdb_id
            pdb_path = complex_dir / f"{pdb_id}_fixed.pdb"
            if not pdb_path.exists():
                pdb_path = complex_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                skipped.append((pdb_id, "missing pdb"))
                continue

            smiles = smiles_map[pdb_id]
            try:
                mol = smiles_to_mol(smiles)
                if smiles not in ligand_to_graph:
                    ligand_to_graph[smiles] = mol_to_graph(mol)
                    ligand_to_ecfp[smiles] = mol_to_ecfp(mol)
                sequence, protein_graph = pdb_to_protein_graph(
                    pdb_path,
                    contact_threshold=args.contact_threshold,
                    max_seq_distance_edges=args.max_seq_distance_edges,
                )
            except Exception as exc:
                skipped.append((pdb_id, str(exc)))
                continue

            protein_to_graph[pdb_id] = protein_graph
            ligands_txt[smiles] = smiles
            proteins_txt[pdb_id] = sequence
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "ligand": smiles,
                    "protein": pdb_id,
                    "series": sequence_to_encoding(sequence),
                    "label": affinities[pdb_id],
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No PDBbind complexes were processed successfully.")

        df.to_csv(output_dir / "updated_full.csv", index=False)
        save_pickle(ligand_to_graph, output_dir / "ligand_to_graph.pkl")
        save_pickle(ligand_to_ecfp, output_dir / "ligand_to_ecfp.pkl")
        save_pickle(protein_to_graph, output_dir / "protein_to_graph.pkl")
        (output_dir / "ligands.txt").write_text(json.dumps(ligands_txt, indent=2), encoding="utf-8")
        (output_dir / "proteins.txt").write_text(json.dumps(proteins_txt, indent=2), encoding="utf-8")

        id_to_index = {pdb_id: i for i, pdb_id in enumerate(df["pdb_id"])}
        valid_indices = [id_to_index[x] for x in split.get("valid", []) if x in id_to_index]
        test_indices = [id_to_index[x] for x in split.get("test", []) if x in id_to_index]
        if not valid_indices or not test_indices:
            random_split = make_random_split(df["pdb_id"])
            valid_indices = [id_to_index[x] for x in random_split["valid"] if x in id_to_index]
            test_indices = [id_to_index[x] for x in random_split["test"] if x in id_to_index]

        (folds_dir / "train_fold_setting1.txt").write_text(json.dumps([valid_indices]), encoding="utf-8")
        (folds_dir / "test_fold_setting1.txt").write_text(json.dumps(test_indices), encoding="utf-8")

        report = {
            "pdbbind_root": str(pdbbind_root),
            "processed_complexes": int(len(df)),
            "unique_ligands": int(len(ligand_to_graph)),
            "unique_proteins": int(len(protein_to_graph)),
            "valid_size": int(len(valid_indices)),
            "test_size": int(len(test_indices)),
            "skipped_count": int(len(skipped)),
            "skipped_examples": skipped[:20],
            "feature_dims": {
                "chem_node": N_CHEM_NODE_FEAT,
                "chem_edge": N_CHEM_EDGE_FEAT,
                "chem_ecfp": N_CHEM_ECFP,
                "protein_node": N_PROT_NODE_FEAT,
                "protein_edge": N_PROT_EDGE_FEAT,
            },
        }
        (output_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(json.dumps(report, indent=2))
        print(f"\nDone. Train with: python test.py --datasets {output_dir.name} --folds 0")
    finally:
        if cleanup is not None:
            cleanup.cleanup()
        elif args.remove_extracted_after and args.keep_extracted:
            shutil.rmtree(args.keep_extracted, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBbind for MEGDTA")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdbbind_tar", type=str, help="Path to pdbbind_dataset.tar.gz")
    source.add_argument("--pdbbind_dir", type=str, help="Path to an extracted PDBbind directory")
    parser.add_argument("--output_dir", type=str, default="data/pdbbind")
    parser.add_argument("--split", choices=["scaffold", "identity30", "identity60"], default="scaffold")
    parser.add_argument("--contact_threshold", type=float, default=8.0)
    parser.add_argument("--max_seq_distance_edges", type=int, default=1)
    parser.add_argument("--max_complexes", type=int, default=None, help="Debug option for a small subset")
    parser.add_argument("--keep_extracted", type=str, default=None, help="Directory for extracted archive reuse")
    parser.add_argument("--remove_extracted_after", action="store_true")
    args = parser.parse_args()
    preprocess_pdbbind(args)


if __name__ == "__main__":
    main()
