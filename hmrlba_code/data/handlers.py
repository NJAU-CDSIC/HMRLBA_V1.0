import os
import torch
import json
from typing import Dict, List, Union
import sys
import signal
import traceback

from transformers import AutoTokenizer, EsmModel, logging
from transformers import T5Tokenizer, T5EncoderModel

from hmrlba_code.graphs import (Backbone, Surface,
                                Surface2Backbone, Complex)

Builder = Union[Backbone, Surface, Surface2Backbone, Complex]

PROT_BUILDERS = {'backbone': Backbone,
                 'surface': Surface,
                 'surface2backbone': Surface2Backbone,
                 }


def get_builder_from_config(config: Dict, prot_mode: str) -> Builder:
    builder_class = PROT_BUILDERS.get(prot_mode, None)
    if builder_class is not None:
        builder_fn = builder_class(**config)
        return builder_fn
    else:
        raise ValueError(f"Graph type {prot_mode} not supported.")


def process_id_callback(handler, pdb_id, **kwargs):
    return handler.process_id(pdb_id, **kwargs)


class DataHandler:

    def __init__(self,
                 dataset: str,
                 prot_mode: str,
                 plm: str,
                 data_dir: str = None,
                 use_mp: bool = True,
                 **kwargs):
        self.dataset = dataset
        self.prot_mode = prot_mode
        self.plm = plm
        self.data_dir = data_dir

        self.builder_fn = self._load_builder()
        self._setup_directories()
        self.load_ids()
        self.__dict__.update(**kwargs)

    def _setup_directories(self):
        self.base_dir = f"{self.data_dir}/Raw_data/{self.dataset}"
        self.save_dir = f"{self.data_dir}/processed/{self.dataset}/{self.prot_mode}/{self.plm}"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _load_builder(self):
        with open(f"configs/Preprocessing/{self.dataset}/{self.prot_mode}.json", "r") as f:
            config = json.load(f)
        builder_fn = get_builder_from_config(config, prot_mode=self.prot_mode)
        if self.dataset in ['pdbbind']:
            builder_fn = Complex(prot_builder=builder_fn)
        return builder_fn

    def load_ids(self):
        raise NotImplementedError('Subclasses must implement for themselves.')

    @staticmethod
    def run_builder(builder_fn, *args, **kwargs):
        return builder_fn(*args, **kwargs)

    def process_ids(self, pdb_ids: List[str] = None):
        if pdb_ids is None:
            pdb_ids = self.pdb_ids

        def handler(signum, frame):
            raise Exception("PDB file could not be processed within given time.")

        # 获取模型路径
        # plm_dir = os.path.join(os.environ['PROT'], "PLMs")
        plm_dir = "/mnt/disk/hzy/pyg/plms"
        model_checkpoint = os.path.join(plm_dir, self.plm)

        # 根据模型类型加载相应的模型和tokenizer
        if self.plm == "ankh":
            prot_model = T5EncoderModel.from_pretrained(model_checkpoint)
            prot_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        elif self.plm == "esm1b":
            prot_model = EsmModel.from_pretrained(model_checkpoint)
            prot_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        elif self.plm == 'prottrans':
            prot_model = T5EncoderModel.from_pretrained(model_checkpoint)
            prot_tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        else:
            raise ValueError(f"PLM '{self.plm}' not supported.")  # 提示具体的PLM名称，增加错误信息的清晰度

        prot_model.eval()

        for pdb_id in pdb_ids:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(500)

            try:
                target = self.process_id(pdb_id, prot_model, prot_tokenizer)
                if target is not None:
                    save_file = f"{self.save_dir}/{target['pdb_id'].upper()}.pth"
                    torch.save(target, save_file)
                    print(f"{target['pdb_id']} processed.", flush=True, file=sys.stdout)
            except Exception as e:
                print(f"{pdb_id}: {e}")
                traceback.print_exc()
                signal.alarm(0)
                continue

    def process_id(self, pdb_id: str, prot_model, prot_tokenizer):
        if os.path.exists(f"{self.save_dir}/{pdb_id.upper()}.pth"):
            target = torch.load(f"{self.save_dir}/{pdb_id.upper()}.pth", map_location='cpu')
            return target

        orig_file = f"{self.base_dir}/pdb_files/{pdb_id}/{pdb_id}.pdb"
        fixed_file = f"{self.base_dir}/pdb_files/{pdb_id}/{pdb_id}_fixed.pdb"

        def run_builder(pdb_file, prot_model, prot_tokenizer):
            target = {}
            target['pdb_id'] = pdb_id
            base_inputs = (self.builder_fn, pdb_file)
            kwargs = {}

            if self.prot_mode in ['surface', 'surface2backbone']:
                mesh_file = f"{self.data_dir}/Raw_data/{self.dataset}/pdb_files/{pdb_id}/{pdb_id}.obj"
                kwargs['mesh_file'] = mesh_file

            if self.dataset == "pdbbind":
                lig_smi = self.lig_smiles[f"{pdb_id}_ligand"]
                activity = self.affinity_dict[pdb_id]
                base_inputs += (lig_smi, float(activity))
                kwargs['build_prot'] = True

            kwargs['prot_model'] = prot_model
            kwargs['prot_tokenizer'] = prot_tokenizer
            kwargs['plm'] = self.plm
            graph = DataHandler.run_builder(*base_inputs, **kwargs)
            if graph is not None:
                if isinstance(graph, tuple):
                    protein, ligand = graph
                    target['prot'] = protein
                    target[f"{pdb_id}_ligand"] = ligand
                else:
                    target['prot'] = graph
                return target

        try:
            return run_builder(orig_file, prot_model, prot_tokenizer)
        except Exception as e:
            msg = f"Failed to generate graph with base file due to {e}. Trying with fixed pdb file."
            print(f"{pdb_id}: {msg}", flush=True)
            traceback.print_exc()
            pass

        try:
            return run_builder(fixed_file, prot_model, prot_tokenizer)
        except Exception as e:
            msg = f"Failed to generate graph with fixed pdb file due to {e}. Returning None."
            print(f"{pdb_id}: {msg}", flush=True)
            traceback.print_exc()
            return None


class PDBBind(DataHandler):

    def load_ids(self):
        with open(os.path.join(self.base_dir, "metadata/affinities.json"), "r") as f:
            self.affinity_dict = json.load(f)
        with open(os.path.join(self.base_dir, "metadata/lig_smiles.json"), "r") as f:
            self.lig_smiles = json.load(f)
        self.pdb_ids = list(self.affinity_dict.keys())
