"""
Contains preprocessing tasks that utilize different binaries. Tasks include adding missing atoms/residues, preparing the surface mesh
"""

import os
from subprocess import Popen, PIPE
import pickle
import numpy as np
import argparse
import pandas as pd
import json
import pymesh
from typing import List
import multiprocessing

from hmrlba_code.feat.complex import get_secondary_struct_features
from hmrlba_code.utils.surface import get_surface, prepare_mesh

DATA_DIR = os.path.join(os.environ["PROT"], "Datasets")
DSSP_BIN = os.environ['DSSP_BIN']
BLENDER_BIN = os.environ['BLENDER_BIN']
MSMS_BIN = os.environ['MSMS_BIN']
BLENDER_SCRIPT = 'scripts/preprocess/resize_meshes.py'


def load_ids(args: argparse.Namespace) -> List[str]:
    """Load pdb ids for each dataset."""
    if args.dataset == "pdbbind":
        with open(f"{args.data_dir}/Raw_data/pdbbind/metadata/affinities.json", 'r') as f:
            affinity_dict = json.load(f)
            pdb_ids = list(affinity_dict.keys())

    elif args.dataset == "enzyme":
        base_dir = f"{args.data_dir}/raw/enzyme"
        pdb_ids = os.listdir(f"{base_dir}/pdb_files/")

    else:
        raise ValueError()

    return pdb_ids


def apply_pdbfixer(base_dir: str, pdb_id: str, **kwargs) -> str:
    """
    Fix corrupt pdb files by handling missing atoms/residues using pdbfixer. 
    More details regarding usage at https://github.com/openmm/pdbfixer. 

    Parameters
    ----------
    base_dir: str,
        Directory to load pdb files from
    pdb_id: str,
        PDB ID of the protein of interest
    """
    orig_file = f"{base_dir}/{pdb_id}/{pdb_id}.pdb"
    fixed_file = f"{base_dir}/{pdb_id}/{pdb_id}_fixed.pdb"
    dirname = os.path.dirname(os.path.abspath(orig_file))
    args = ["pdbfixer", orig_file, f"--output={fixed_file}", "--add-atoms=heavy"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=dirname)
    stdout, stderr = p2.communicate()
    if not os.path.exists(fixed_file):
        return None

    if stderr.decode("utf-8") != "":
        print(f"{pdb_id}: Could not process with pdb_fixer because of {stderr}.")
        return None
    else:
        return pdb_id


def prepare_dssp(base_dir: str, pdb_id: str, **kwargs):
    """
    Computes secondary structure features using DSSP. More details regarding usage
    can be found at https://swift.cmbi.umcn.nl/gv/dssp/DSSP_3.html

    Parameters
    ----------
    base_dir: str,
        Directory to load pdb files from
    pdb_id: str,
        PDB ID of the protein of interest
    """
    orig_file = f"{base_dir}/{pdb_id}/{pdb_id}.pdb"
    fixed_file = f"{base_dir}/{pdb_id}/{pdb_id}_fixed.pdb"
    dssp_file = f"{base_dir}/{pdb_id}/{pdb_id}.dssp"

    try:
        dssp_dict = get_secondary_struct_features(orig_file, dssp_bin=DSSP_BIN)
        with open(dssp_file, "wb") as f:
            pickle.dump(dssp_dict, f)
        return pdb_id

    except Exception as e:
        print(f"{pdb_id}: Failed to produce DSSP output because of {e}. Trying the fixed file. ", flush=True)
        pass

    try:
        if not os.path.exists(fixed_file):
            print(f"{pdb_id}: Fixed file not found for preparing DSSP output. Returning None.")
            return None
        dssp_dict = get_secondary_struct_features(fixed_file, dssp_bin=DSSP_BIN)
        with open(dssp_file, "wb") as f:
            pickle.dump(dssp_dict, f)
        return pdb_id

    except Exception as e:
        print(f"PDBID: {pdb_id}: Failed to produce DSSP output with fixed file because of{e}. Returning None. ",
              flush=True)
        return None


def prepare_surface_mesh(base_dir: str, pdb_id: str, num_faces: int = 2600, **kwargs):
    """Prepares surface mesh and normalizes it to specified number of edges using BLENDER.
    
    Parameters
    ----------
    base_dir: str,
        Directory to load pdb files from
    pdb_id: str,
        PDB ID of the protein of interest
    save_dir: str, default None
        Optional directory to save the dssp file. If None, then base_dir is used
    num_faces: int, default 2600
        Number of faces to compress the mesh to
    """
    orig_file = f"{base_dir}/{pdb_id}/{pdb_id}.pdb"
    fixed_file = f"{base_dir}/{pdb_id}/{pdb_id}_fixed.pdb"
    mesh_file = f"{base_dir}/{pdb_id}/{pdb_id}_base.obj"
    export_file = f"{base_dir}/{pdb_id}/{pdb_id}.obj"

    if os.path.exists(export_file):
        return pdb_id

    def build_and_normalize_surface(pdb_file: str):
        surface = get_surface(pdb_file=pdb_file, msms_bin=MSMS_BIN)
        mesh = prepare_mesh(vertices=surface[0], faces=surface[1], normals=surface[2],
                            resolution=1.5, apply_fixes=True)
        pymesh.save_mesh(filename=mesh_file, mesh=mesh)
        blender_args = [BLENDER_BIN, "--background", "-noaudio", "--python",
                        BLENDER_SCRIPT, mesh_file, f"{num_faces}", export_file]
        p2 = Popen(blender_args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        if stderr.decode("utf-8") != "":
            raise ValueError(f"Blender could not normalize mesh because of {stderr}.")

    try:
        build_and_normalize_surface(pdb_file=orig_file)
        return pdb_id
    except Exception as e:
        print(f"{pdb_id}: Failed to produce mesh because of {e}. Retrying with fixed file.")
        pass

    try:
        build_and_normalize_surface(pdb_file=fixed_file)
        return pdb_id
    except Exception as e:
        print(f"{pdb_id}: Failed to produce mesh because of {e}, Returning None", flush=True)
        return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument("--num_faces", type=int, default=2600)
    parser.add_argument("--pdb_ids", nargs='+', default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default='all',
                        choices=['pdbfixer', 'surface', 'dssp', 'all'])
    args = parser.parse_args()
    return args


def run_tasks(tasks, base_dir, pdb_id, **kwargs):
    print_str = f"{pdb_id}: "
    for task_name, task_fn in tasks:
        task_output = task_fn(base_dir=base_dir, pdb_id=pdb_id, **kwargs)
        if task_output is not None:
            print_str += f"{task_name},"
    print_str = print_str.rstrip(",")
    return print_str


def main():
    args = get_args()
    pdb_ids = args.pdb_ids
    if pdb_ids is None:
        pdb_ids = load_ids(args)
    base_dir = f"{args.data_dir}/Raw_data/{args.dataset}/pdb_files"
    save_dir = base_dir

    TASK_FNS = {'pdbfixer': apply_pdbfixer, 'surface': prepare_surface_mesh, 'dssp': prepare_dssp}

    if args.tasks == ["all"]:
        tasks = [('pdbfixer', apply_pdbfixer),
                 ('dssp', prepare_dssp),
                 ('surface', prepare_surface_mesh)]
    else:
        tasks = []
        for task in args.tasks:
            tasks.append((task, TASK_FNS.get(task)))

    kwargs = {'num_faces': args.num_faces}

    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 5, maxtasksperchild=1)
    results = []
    for pdb_id in pdb_ids:
        results.append(pool.apply_async(run_tasks, (tasks, base_dir, pdb_id), kwargs))

    for result in results:
        try:
            task_output = result.get(timeout=600)
            if task_output is not None:
                print(f"{task_output}", flush=True)
            else:
                continue
        except multiprocessing.context.TimeoutError as e:
            continue


if __name__ == "__main__":
    main()
