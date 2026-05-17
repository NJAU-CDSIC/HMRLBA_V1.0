#!/usr/bin/env python3
"""Convert one DUD-E target into the raw-data layout expected by HMRLBA.

Input target directory example:
    Datasets/Virtual screening/DUD-E/hxk4/

Output directory example:
    Datasets/Raw_data/dude_hxk4/

The script reads DUD-E actives/decoys .ism files and writes:
    metadata/lig_smiles.json
    metadata/affinities.json
    pdb_files/<target>/<target>.pdb
"""

import argparse
import json
import shutil
from pathlib import Path


def read_ism(path, target, label, start_index=0):
    smiles = {}
    affinities = {}
    details = []
    with open(path, "r") as handle:
        for offset, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("Invalid DUD-E .ism line in {}: {}".format(path, line))
            smi = parts[0]
            source_id = parts[1]
            lig_id = "{}_{}".format(target, source_id)
            if lig_id in smiles:
                lig_id = "{}_{}_{}".format(target, source_id, start_index + offset)
            smiles[lig_id] = smi
            affinities[lig_id] = float(label)
            details.append({
                "ligand_id": lig_id,
                "source_id": source_id,
                "source_name": parts[2] if len(parts) > 2 else None,
                "label": float(label),
            })
    return smiles, affinities, details


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare an HMRLBA raw dataset from one DUD-E target.")
    parser.add_argument("--target-dir", required=True,
                        help="DUD-E target directory, e.g. Datasets/Virtual screening/DUD-E/hxk4")
    parser.add_argument("--target", default=None, help="Target name. Defaults to target-dir basename.")
    parser.add_argument("--out-root", required=True, help="Output Raw_data root or dataset directory.")
    parser.add_argument("--dataset-name", default=None, help="Dataset directory name. Defaults to dude_<target>.")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    target = (args.target or target_dir.name).lower()
    dataset_name = args.dataset_name or "dude_{}".format(target)
    out_root = Path(args.out_root).resolve()
    out_dir = out_root if out_root.name == dataset_name else out_root / dataset_name

    active_smiles, active_aff, active_details = read_ism(
        target_dir / "actives_final.ism", target, 1.0
    )
    decoy_smiles, decoy_aff, decoy_details = read_ism(
        target_dir / "decoys_final.ism", target, 0.0, start_index=len(active_smiles)
    )

    lig_smiles = {}
    lig_smiles.update(active_smiles)
    lig_smiles.update(decoy_smiles)
    affinities = {}
    affinities.update(active_aff)
    affinities.update(decoy_aff)

    receptor_src = target_dir / "receptor.pdb"
    receptor_dst = out_dir / "pdb_files" / target / "{}.pdb".format(target)
    receptor_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(receptor_src), str(receptor_dst))

    write_json(out_dir / "metadata" / "lig_smiles.json", lig_smiles)
    write_json(out_dir / "metadata" / "affinities.json", affinities)
    write_json(out_dir / "metadata" / "dude_source_records.json", active_details + decoy_details)

    print("Prepared {}".format(out_dir))
    print("Active ligands: {}".format(len(active_smiles)))
    print("Decoy ligands: {}".format(len(decoy_smiles)))
    print("Total ligands: {}".format(len(lig_smiles)))


if __name__ == "__main__":
    main()
