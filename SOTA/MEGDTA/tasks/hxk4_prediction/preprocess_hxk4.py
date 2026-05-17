"""
Preprocess HXK4 virtual screening dataset for MEGDTA.

This script processes actives and decoys from the DUD-E format and creates
the files needed by MEGDTA for prediction.
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit is required")
    print("Install: conda install -c conda-forge rdkit")
    exit(1)

from preprocess_data import (
    smiles_to_graph,
    smiles_to_ecfp,
    sequence_to_graph,
    sequence_to_encoding
)


def read_ism_file(ism_path):
    """Read SMILES from ISM file (SMILES format)."""
    smiles_list = []

    with open(ism_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # ISM format: SMILES ID [other_info]
                parts = line.split()
                if parts:
                    smiles = parts[0]
                    smiles_list.append(smiles)

    return smiles_list


def read_protein_sequence_from_pdb(pdb_path):
    """Extract protein sequence from PDB file."""
    from collections import OrderedDict

    # Standard 3-letter to 1-letter amino acid code
    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    residues = OrderedDict()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()

                key = (chain, res_num)
                if key not in residues and res_name in aa_3to1:
                    residues[key] = aa_3to1[res_name]

    sequence = ''.join(residues.values())
    return sequence


def preprocess_hxk4(data_dir, output_dir):
    """Preprocess HXK4 virtual screening dataset."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "folds").mkdir(exist_ok=True)

    print("=" * 60)
    print("Preprocessing HXK4 Virtual Screening Dataset")
    print("=" * 60)

    # Read protein sequence from PDB
    print("\n1. Reading protein sequence from PDB...")
    pdb_path = data_path / "receptor.pdb"
    protein_sequence = read_protein_sequence_from_pdb(pdb_path)
    print(f"   Protein length: {len(protein_sequence)} residues")
    print(f"   Sequence: {protein_sequence[:50]}...")

    # Read actives
    print("\n2. Reading active compounds...")
    actives_path = data_path / "actives_final.ism"
    actives_smiles = read_ism_file(actives_path)
    print(f"   Found {len(actives_smiles)} actives")

    # Read decoys
    print("\n3. Reading decoy compounds...")
    decoys_path = data_path / "decoys_final.ism"
    decoys_smiles = read_ism_file(decoys_path)
    print(f"   Found {len(decoys_smiles)} decoys")

    # Create dataset
    print("\n4. Creating dataset...")
    data = []

    # Add actives (label = 1)
    for smiles in actives_smiles:
        data.append({
            'ligand_smiles': smiles,
            'protein_sequence': protein_sequence,
            'label': 1  # Active
        })

    # Add decoys (label = 0)
    for smiles in decoys_smiles:
        data.append({
            'ligand_smiles': smiles,
            'protein_sequence': protein_sequence,
            'label': 0  # Decoy
        })

    df = pd.DataFrame(data)
    print(f"   Total samples: {len(df)}")
    print(f"   Actives: {(df['label'] == 1).sum()}")
    print(f"   Decoys: {(df['label'] == 0).sum()}")

    # Get unique ligands and proteins
    unique_ligands = df['ligand_smiles'].unique()
    unique_proteins = df['protein_sequence'].unique()
    print(f"   Unique ligands: {len(unique_ligands)}")
    print(f"   Unique proteins: {len(unique_proteins)}")

    # Generate ligand graphs and ECFP
    print("\n5. Generating ligand graphs and ECFP fingerprints...")
    ligand_to_graph = {}
    ligand_to_ecfp = {}

    failed_ligands = []
    for smiles in tqdm(unique_ligands, desc="   Processing ligands"):
        try:
            ligand_to_graph[smiles] = smiles_to_graph(smiles)
            ligand_to_ecfp[smiles] = smiles_to_ecfp(smiles)
        except Exception as e:
            failed_ligands.append(smiles)
            continue

    print(f"   Successfully processed: {len(ligand_to_graph)} ligands")
    if failed_ligands:
        print(f"   Failed: {len(failed_ligands)} ligands")

    # Remove failed ligands from dataset
    df = df[df['ligand_smiles'].isin(ligand_to_graph.keys())]
    print(f"   Final dataset size: {len(df)}")

    # Generate protein graphs
    print("\n6. Generating protein graphs...")
    protein_to_graph = {}

    for seq in tqdm(unique_proteins, desc="   Processing proteins"):
        try:
            protein_to_graph[seq] = sequence_to_graph(seq)
        except Exception as e:
            print(f"\n   Warning: Protein processing failed: {e}")
            continue

    print(f"   Successfully processed: {len(protein_to_graph)} proteins")

    # Create updated_full.csv
    print("\n7. Creating updated_full.csv...")
    df_output = pd.DataFrame()
    df_output['ligand'] = df['ligand_smiles']
    df_output['protein'] = df['protein_sequence']
    df_output['series'] = df['protein_sequence'].apply(sequence_to_encoding)
    df_output['label'] = df['label']

    df_output.to_csv(output_path / "updated_full.csv", index=False)
    print(f"   Saved: {output_path / 'updated_full.csv'}")

    # Save ligands.txt and proteins.txt
    print("\n8. Saving ligands.txt and proteins.txt...")
    ligands_dict = {smiles: smiles for smiles in unique_ligands if smiles in ligand_to_graph}
    proteins_dict = {seq: seq for seq in unique_proteins}

    with open(output_path / "ligands.txt", 'w') as f:
        json.dump(ligands_dict, f)

    with open(output_path / "proteins.txt", 'w') as f:
        json.dump(proteins_dict, f)

    # Save pkl files
    print("\n9. Saving pkl files...")
    with open(output_path / "ligand_to_graph.pkl", 'wb') as f:
        pickle.dump(ligand_to_graph, f)
    print(f"   Saved: ligand_to_graph.pkl ({len(ligand_to_graph)} entries)")

    with open(output_path / "ligand_to_ecfp.pkl", 'wb') as f:
        pickle.dump(ligand_to_ecfp, f)
    print(f"   Saved: ligand_to_ecfp.pkl ({len(ligand_to_ecfp)} entries)")

    with open(output_path / "protein_to_graph.pkl", 'wb') as f:
        pickle.dump(protein_to_graph, f)
    print(f"   Saved: protein_to_graph.pkl ({len(protein_to_graph)} entries)")

    # Create fold files (use all data as test set for prediction)
    print("\n10. Creating fold files...")
    n_samples = len(df_output)
    test_indices = list(range(n_samples))
    train_indices = []  # Empty train set for prediction only

    with open(output_path / "folds" / "train_fold_setting1.txt", 'w') as f:
        json.dump([train_indices], f)

    with open(output_path / "folds" / "test_fold_setting1.txt", 'w') as f:
        json.dump(test_indices, f)

    print(f"   Test set: {len(test_indices)} samples")

    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_path / 'updated_full.csv'}")
    print(f"  - {output_path / 'ligands.txt'}")
    print(f"  - {output_path / 'proteins.txt'}")
    print(f"  - {output_path / 'ligand_to_graph.pkl'}")
    print(f"  - {output_path / 'ligand_to_ecfp.pkl'}")
    print(f"  - {output_path / 'protein_to_graph.pkl'}")
    print(f"  - {output_path / 'folds' / 'train_fold_setting1.txt'}")
    print(f"  - {output_path / 'folds' / 'test_fold_setting1.txt'}")

    return df_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess HXK4 virtual screening dataset')
    parser.add_argument('--data_dir', type=str, default='data/hxk4',
                       help='Input directory containing HXK4 data')
    parser.add_argument('--output_dir', type=str, default='data/hxk4',
                       help='Output directory')

    args = parser.parse_args()

    preprocess_hxk4(args.data_dir, args.output_dir)
