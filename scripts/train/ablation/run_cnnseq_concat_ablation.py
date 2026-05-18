#!/usr/bin/env python3
"""
Train/evaluate a non-invasive CNN-Seq + sequence-vector ablation for HMRLBA.

This script does two things without editing the original HMRLBA source tree:

1. Replaces PLM node features with residue-level CNN sequence features.
2. Mean-pools the CNN residue features into a sequence vector and concatenates
   it with the three graph vectors before the final MLP:

      [backbone_graph_vec, surface_graph_vec, ligand_graph_vec, cnn_seq_vec]

The graph construction, edges, ligand features, MPNN layers, losses, optimizer,
and train/valid/test split are still taken from the original project.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


DEFAULT_PROT = "/root/HMRLBA_V1.0"
AA_TO_IDX = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
    "X": 21,
}
PAD_IDX = 0
UNK_IDX = AA_TO_IDX["X"]


def bootstrap_project(prot_root):
    prot_root = str(Path(prot_root).resolve())
    os.environ.setdefault("PROT", prot_root)
    if prot_root not in sys.path:
        sys.path.insert(0, prot_root)
    return prot_root


def import_runtime():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from Bio import PDB
    from Bio.PDB import PDBParser, Polypeptide
    from torch_geometric.nn import global_add_pool, global_mean_pool
    import yaml
    import wandb

    from hmrlba_code.data import DATASETS
    from hmrlba_code.data.base import ComplexData
    from hmrlba_code.data.batching import ComplexBatch
    from hmrlba_code.models import Trainer
    from hmrlba_code.models.HMRLBA import HMRLBA
    from hmrlba_code.models.model_builder import affinity_pred_config
    from hmrlba_code.utils.metrics import DATASET_METRICS, METRICS
    from hmrlba_code.utils.tensor import build_mlp

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "PDB": PDB,
        "PDBParser": PDBParser,
        "Polypeptide": Polypeptide,
        "global_add_pool": global_add_pool,
        "global_mean_pool": global_mean_pool,
        "yaml": yaml,
        "wandb": wandb,
        "DATASETS": DATASETS,
        "ComplexData": ComplexData,
        "ComplexBatch": ComplexBatch,
        "Trainer": Trainer,
        "HMRLBA": HMRLBA,
        "affinity_pred_config": affinity_pred_config,
        "DATASET_METRICS": DATASET_METRICS,
        "METRICS": METRICS,
        "build_mlp": build_mlp,
    }


def seed_everything(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flatten_wandb_config(raw_config):
    config = {}
    for key, value in raw_config.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    return config


def load_yaml_config(path, yaml):
    with open(path, "r") as handle:
        return flatten_wandb_config(yaml.load(handle, Loader=yaml.FullLoader))


def normalize_config(config):
    defaults = {
        "num_workers": 0,
        "batch_size": 1,
        "print_every": 500,
        "eval_every": None,
        "epochs": 100,
        "lr": 0.001,
        "scheduler_type": "plateau",
        "step_after": 50,
        "anneal_rate": 0.9,
        "patience": 5,
        "metric_thresh": 0.01,
        "accum_every": None,
        "clip_norm": 10.0,
        "weight_decay": 0.0,
        "use_doubles": False,
    }
    out = dict(defaults)
    out.update(config)
    return out


def seq_to_tensor(sequence, torch):
    ids = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence]
    if not ids:
        ids = [UNK_IDX]
    return torch.tensor(ids, dtype=torch.long)


def three_to_one(resname, rt):
    Polypeptide = rt["Polypeptide"]
    if hasattr(Polypeptide, "three_to_one"):
        try:
            return Polypeptide.three_to_one(resname)
        except KeyError:
            return "X"
    table = getattr(Polypeptide, "protein_letters_3to1", {})
    return table.get(resname.capitalize(), table.get(resname.upper(), "X"))


def candidate_structure_files(raw_dir, dataset, pdb_id):
    raw_dir = Path(raw_dir)
    ids = []
    if dataset.startswith("dude_"):
        ids.append(pdb_id.split("_")[0])
    ids.extend([pdb_id, pdb_id.lower(), pdb_id.upper()])

    seen = set()
    candidates = []
    for item in ids:
        if not item or item in seen:
            continue
        seen.add(item)
        candidates.extend([
            raw_dir / "pdb_files" / item / "{}.pdb".format(item),
            raw_dir / "pdb_files" / item.lower() / "{}.pdb".format(item.lower()),
            raw_dir / "pdb_files" / item.upper() / "{}.pdb".format(item.upper()),
        ])
    return candidates


def sequence_from_pdb(pdb_file, rt):
    parser = rt["PDBParser"](QUIET=True)
    pp_builder = rt["Polypeptide"].PPBuilder()
    struct = parser.get_structure(Path(pdb_file).stem, str(pdb_file))
    polypeptides = pp_builder.build_peptides(struct)
    residues = [res for pp in polypeptides for res in pp]

    dssp_file = Path(pdb_file).with_suffix(".dssp")
    if dssp_file.exists():
        import pickle
        with open(dssp_file, "rb") as handle:
            dssp_dict = pickle.load(handle)
        residues = [res for res in residues if res.get_full_id()[2:] in dssp_dict]

    chars = []
    for residue in residues:
        if rt["PDB"].is_aa(residue):
            chars.append(three_to_one(residue.get_resname(), rt))
        else:
            chars.append("X")
    return "".join(chars)


def make_sequence_dataset_class(base_class, rt):
    torch = rt["torch"]

    class SequenceDataset(base_class):
        def __init__(self, *args, **kwargs):
            self._sequence_cache = {}
            super().__init__(*args, **kwargs)

        def _pdb_id_for_idx(self, idx):
            item = self.ids[idx]
            if isinstance(item, str):
                return item
            return item[0]

        def _load_sequence(self, pdb_id, fallback_len):
            cache_key = (self.dataset, pdb_id)
            if cache_key in self._sequence_cache:
                return self._sequence_cache[cache_key]

            sequence = None
            for pdb_file in candidate_structure_files(self.raw_dir, self.dataset, pdb_id):
                if pdb_file.exists():
                    try:
                        sequence = sequence_from_pdb(pdb_file, rt)
                        break
                    except Exception as exc:
                        print("Warning: failed to parse sequence from {}: {}".format(pdb_file, exc), flush=True)

            if not sequence:
                sequence = "X" * max(1, int(fallback_len))
                print("Warning: using X fallback sequence for {} length={}".format(pdb_id, len(sequence)),
                      flush=True)

            self._sequence_cache[cache_key] = sequence
            return sequence

        def _file_pdb_id(self, pdb_id):
            class_name = self.__class__.__mro__[1].__name__.lower()
            return pdb_id if "dude" in class_name else pdb_id.upper()

        def _load_single_graph_cache(self, pdb_id, lig_id):
            file_pdb_id = self._file_pdb_id(pdb_id)
            graph_file = Path(self.graph_source_dir) / "{}.pth".format(file_pdb_id)
            if not graph_file.exists():
                return None
            try:
                target_dict = torch.load(str(graph_file), map_location="cpu")
            except (EOFError, RuntimeError, OSError) as exc:
                print("Warning: failed to load {}: {}".format(graph_file, exc), flush=True)
                return None

            if "prot" not in target_dict or lig_id not in target_dict:
                return None

            prot = target_dict["prot"]
            ligand = target_dict[lig_id]
            # protein2/protein3 are placeholders only. The CNN model reads protein.
            return rt["ComplexData"](protein=prot, protein2=prot, protein3=prot, ligand=ligand)

        def __getitem__(self, idx):
            item = self.ids[idx]
            if isinstance(item, str):
                pdb_id = item
                lig_id = item
            else:
                pdb_id, lig_id = item

            data = self._load_single_graph_cache(pdb_id, lig_id)
            if data is None:
                return None

            data = self.add_or_update_target(data, idx)
            fallback_len = data.protein.backbone.num_nodes
            sequence = self._load_sequence(pdb_id, fallback_len)
            data.seq_idx = seq_to_tensor(sequence, torch)
            data.sequence = sequence
            return data

        def collate_fn(self, data_list):
            data_list = [data for data in data_list if data is not None]
            if not data_list:
                return None

            seqs = [data.seq_idx for data in data_list]
            max_len = max(seq.size(0) for seq in seqs)
            seq_idx = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
            seq_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
            for row, seq in enumerate(seqs):
                seq_idx[row, :seq.size(0)] = seq
                seq_mask[row, :seq.size(0)] = True

            batch = rt["ComplexBatch"].from_data_list(data_list, prot_mode=self.prot_mode)
            if batch is not None:
                batch.seq_idx = seq_idx
                batch.seq_mask = seq_mask
            return batch

    return SequenceDataset


class CNNSequenceEncoderModule:
    @staticmethod
    def build(rt, vocab_size=22, embed_dim=128, channels=128, out_dim=512, dropout=0.15):
        nn = rt["nn"]
        F = rt["F"]

        class CNNSequenceEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
                self.conv3 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
                self.conv5 = nn.Conv1d(embed_dim, channels, kernel_size=5, padding=2)
                self.conv7 = nn.Conv1d(embed_dim, channels, kernel_size=7, padding=3)
                self.proj = nn.Linear(channels * 3, out_dim)
                self.norm = nn.LayerNorm(out_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, seq_idx, seq_mask=None):
                x = self.embedding(seq_idx)
                x = x.transpose(1, 2)
                h3 = F.relu(self.conv3(x))
                h5 = F.relu(self.conv5(x))
                h7 = F.relu(self.conv7(x))
                h = rt["torch"].cat([h3, h5, h7], dim=1).transpose(1, 2)
                h = self.dropout(self.norm(self.proj(h)))
                if seq_mask is not None:
                    h = h * seq_mask.unsqueeze(-1).to(h.dtype)
                return h

        return CNNSequenceEncoder()


class CNNSeqConcatModelBuilder:
    @staticmethod
    def build(config, rt, device):
        torch = rt["torch"]
        nn = rt["nn"]
        build_mlp = rt["build_mlp"]

        class CNNSeqConcatModel(nn.Module):
            def __init__(self, loaded_config):
                super().__init__()
                self.loaded_config = dict(loaded_config)
                self.device = device
                dataset_metrics = rt["DATASET_METRICS"].get(loaded_config["dataset"], [])
                if not isinstance(dataset_metrics, list):
                    dataset_metrics = [dataset_metrics]
                self.metrics = {}
                for metric in dataset_metrics:
                    self.metrics[metric], _, _ = rt["METRICS"].get(metric)

                node_dim = loaded_config["cnn_out_dim"]
                model_config = rt["affinity_pred_config"](loaded_config)
                mpn_config = model_config["config"]["mpn_config"]
                mpn_config["bconfig"]["node_fdim"] = node_dim
                mpn_config["sconfig"]["node_fdim"] = node_dim

                self.hmrlba = rt["HMRLBA"](**model_config, device=device)
                old_in_dim = self.hmrlba.activity_mlp[0].in_features
                self.hmrlba.activity_mlp = build_mlp(
                    in_dim=old_in_dim + loaded_config["seq_vec_dim"],
                    h_dim=self.hmrlba.config["hsize"],
                    out_dim=1,
                    dropout_p=self.hmrlba.config["dropout_mlp"],
                    activation=self.hmrlba.config["activation"],
                )

                self.seq_encoder = CNNSequenceEncoderModule.build(
                    rt,
                    vocab_size=22,
                    embed_dim=loaded_config["cnn_embed_dim"],
                    channels=loaded_config["cnn_channels"],
                    out_dim=loaded_config["cnn_out_dim"],
                    dropout=loaded_config["cnn_dropout"],
                )
                if loaded_config["seq_vec_dim"] == loaded_config["cnn_out_dim"]:
                    self.seq_vec_proj = nn.Identity()
                else:
                    self.seq_vec_proj = nn.Linear(loaded_config["cnn_out_dim"],
                                                  loaded_config["seq_vec_dim"])
                self.loss_fn = nn.MSELoss(reduction="none")

            def get_saveables(self):
                return {"loaded_config": self.loaded_config}

            def _node_features_from_sequence(self, prot, residue_h):
                torch = rt["torch"]
                backbone = prot.backbone
                surface = prot.surface
                batch_index = backbone.batch
                node_dim = residue_h.size(-1)
                backbone_x = torch.zeros((backbone.num_nodes, node_dim), device=residue_h.device,
                                         dtype=residue_h.dtype)

                for batch_id in range(residue_h.size(0)):
                    node_idx = (batch_index == batch_id).nonzero(as_tuple=False).view(-1)
                    if node_idx.numel() == 0:
                        continue
                    take = min(node_idx.numel(), residue_h.size(1))
                    backbone_x[node_idx[:take]] = residue_h[batch_id, :take]
                    if take < node_idx.numel():
                        backbone_x[node_idx[take:]] = residue_h.new_zeros((node_idx.numel() - take, node_dim))

                surface_x = torch.zeros((surface.num_nodes, node_dim), device=residue_h.device,
                                        dtype=residue_h.dtype)
                mapping = prot.mapping
                for res_idx in range(mapping.size(0)):
                    surf_idx = mapping[res_idx]
                    surf_idx = surf_idx[surf_idx > 0].long() - 1
                    if surf_idx.numel() > 0:
                        surface_x[surf_idx] = backbone_x[res_idx]

                return backbone_x, surface_x

            def _sequence_vector(self, residue_h, seq_mask):
                if seq_mask is None:
                    seq_vec = residue_h.mean(dim=1)
                else:
                    mask = seq_mask.unsqueeze(-1).to(residue_h.dtype)
                    denom = mask.sum(dim=1).clamp_min(1.0)
                    seq_vec = (residue_h * mask).sum(dim=1) / denom
                return self.seq_vec_proj(seq_vec)

            def forward(self, complex_data):
                torch = rt["torch"]
                global_add_pool = rt["global_add_pool"]
                global_mean_pool = rt["global_mean_pool"]

                complex_data = complex_data.to(self.device)
                seq_idx = complex_data.seq_idx.to(self.device)
                seq_mask = complex_data.seq_mask.to(self.device)
                residue_h = self.seq_encoder(seq_idx, seq_mask)
                seq_vec = self._sequence_vector(residue_h, seq_mask)

                prot = complex_data.protein
                backbone_x, surface_x = self._node_features_from_sequence(prot, residue_h)
                prot.backbone.x = backbone_x
                prot.surface.x = surface_x

                lig = complex_data.ligand
                lig_node_emb = self.hmrlba.lig_mpn(lig.x, lig.edge_index, lig.edge_attr)
                if self.hmrlba.config.get("graph_pool", "sum_pool") == "sum_pool":
                    lig_emb = global_add_pool(lig_node_emb, lig.batch)
                elif self.hmrlba.config.get("graph_pool", "sum_pool") == "mean_pool":
                    lig_emb = global_mean_pool(lig_node_emb, lig.batch)
                else:
                    raise ValueError("Unsupported graph_pool")

                top_graph_emb, bottom_graph_emb = self.hmrlba.prot_mpn(prot)
                complex_vec = torch.cat([top_graph_emb, bottom_graph_emb, lig_emb, seq_vec], dim=-1)
                return self.hmrlba.activity_mlp(complex_vec)

            def train_step(self, complex_data):
                pred = self(complex_data)
                loss = self.loss_fn(pred.squeeze(-1), complex_data.y).mean()
                return loss, {"loss": loss.item()}

            def eval_step(self, eval_data):
                if eval_data is None:
                    return {}
                torch = rt["torch"]
                np = rt["np"]
                eval_pred, eval_labels, eval_loss = [], [], []
                self.eval()
                with torch.no_grad():
                    for inputs in eval_data:
                        if inputs is None:
                            continue
                        pred = self(inputs)
                        loss = self.loss_fn(pred.squeeze(-1), inputs.y).mean()
                        eval_pred.append(pred.item())
                        eval_labels.append(inputs.y.item())
                        eval_loss.append(loss.item())
                eval_labels = np.array(eval_labels).flatten()
                eval_pred = np.array(eval_pred).flatten()
                out = {}
                for metric, metric_fn in self.metrics.items():
                    out[metric] = np.round(metric_fn(eval_labels, eval_pred), 4)
                out["loss"] = np.round(np.mean(eval_loss), 4) if eval_loss else None
                self.train()
                return out

            def predict(self, complex_data):
                with rt["torch"].no_grad():
                    return self(complex_data)

        return CNNSeqConcatModel(config)


def infer_graph_source_dir(data_dir, dataset, prot_mode, requested=None):
    if requested:
        return requested

    base = Path(data_dir) / "processed" / dataset / prot_mode
    for name in ["cnn_base", "esm1b", "ankh", "prottrans"]:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return str(base / "cnn_base")


def build_loader(config, mode, rt, data_dir, num_workers=None, shuffle=False):
    dataset_class = rt["DATASETS"].get(config["dataset"])
    dataset_class = make_sequence_dataset_class(dataset_class, rt)
    raw_dir = os.path.abspath("{}/Raw_data/{}".format(data_dir, config["dataset"]))
    processed_dir = os.path.abspath(data_dir)
    dataset = dataset_class(
        mode=mode,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prot_mode=config["prot_mode"],
        dataset=config["dataset"],
        split=config.get("split", "identity30"),
    )
    dataset.graph_source_dir = infer_graph_source_dir(
        data_dir, config["dataset"], config["prot_mode"], config.get("graph_source_dir")
    )
    if config.get("limit_ids") is not None:
        dataset.ids = dataset.ids[:int(config["limit_ids"])]
    print("{} loader graph_source_dir={}".format(mode, dataset.graph_source_dir), flush=True)
    workers = config["num_workers"] if num_workers is None else num_workers
    batch_size = config["batch_size"] if mode == "train" else 1
    return dataset.create_loader(batch_size=batch_size, num_workers=workers, shuffle=shuffle)


def run_train(args, rt, device):
    torch = rt["torch"]
    wandb = rt["wandb"]
    Trainer = rt["Trainer"]
    config = load_yaml_config(args.config_file, rt["yaml"])
    config = normalize_config(config)
    config.update({
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "cnn_embed_dim": args.cnn_embed_dim,
        "cnn_channels": args.cnn_channels,
        "cnn_out_dim": args.cnn_out_dim,
        "cnn_dropout": args.cnn_dropout,
        "seq_vec_dim": args.seq_vec_dim,
        "graph_source_dir": args.graph_source,
        "limit_ids": args.limit_ids,
    })
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.disable_eval:
        config["eval_every"] = None
    elif args.eval_every_set:
        config["eval_every"] = args.eval_every
    if args.print_every is not None:
        config["print_every"] = args.print_every
    if args.lr is not None:
        config["lr"] = args.lr

    if not args.wandb_online:
        os.environ.setdefault("WANDB_MODE", "offline")

    for seed in args.seeds:
        seed_everything(seed, torch, rt["np"])
        run_name = "cnnseq_concat_seed{}".format(seed)
        print("\n[train] {}".format(run_name), flush=True)
        run_config = dict(config)
        run_config["seed"] = seed
        wandb.init(project=args.wandb_project, dir=args.out_dir,
                   entity=args.wandb_entity, name=run_name, config=run_config,
                   reinit=True)
        try:
            torch.set_default_dtype(torch.float64 if run_config["use_doubles"] else torch.float32)
            model = CNNSeqConcatModelBuilder.build(run_config, rt, device).to(device)
            train_loader = build_loader(run_config, "train", rt, args.data_dir,
                                        num_workers=args.num_workers, shuffle=True)
            valid_loader = None
            if run_config["eval_every"] is not None:
                valid_loader = build_loader(run_config, "valid", rt, args.data_dir,
                                            num_workers=args.num_workers)
            trainer = Trainer(model=model, dataset=run_config["dataset"],
                              task_type=run_config["prot_mode"],
                              print_every=run_config["print_every"],
                              eval_every=run_config["eval_every"])
            trainer.build_optimizer(learning_rate=run_config["lr"],
                                    weight_decay=run_config.get("weight_decay", 0.0))
            trainer.build_scheduler(type=run_config["scheduler_type"],
                                    step_after=run_config["step_after"],
                                    anneal_rate=run_config["anneal_rate"],
                                    patience=run_config["patience"],
                                    thresh=run_config["metric_thresh"])
            trainer.train_epochs(train_loader, valid_loader, run_config["epochs"],
                                 **{"accum_every": run_config["accum_every"],
                                    "clip_norm": run_config["clip_norm"]})
        finally:
            wandb.finish()


def compute_metrics(y_true, y_pred, rt):
    out = {}
    for metric in ["rmse", "mae", "pearsonr", "spearmanr", "r2"]:
        metric_fn, _, _ = rt["METRICS"].get(metric)
        out[metric] = float(metric_fn(y_true, y_pred))
    return out


def resolve_checkpoint_path(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    wandb_root = Path(args.out_dir) / "wandb"
    candidates = sorted(wandb_root.glob("*/files/best_ckpt.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No best_ckpt.pt found under {}".format(wandb_root))
    return candidates[-1]


def run_eval(args, rt, device):
    torch = rt["torch"]
    np = rt["np"]
    checkpoint_path = resolve_checkpoint_path(args)
    print("[eval] checkpoint={}".format(checkpoint_path), flush=True)
    loaded = torch.load(str(checkpoint_path), map_location=device)
    if "saveables" not in loaded or "loaded_config" not in loaded["saveables"]:
        raise ValueError("Checkpoint is not from run_cnnseq_concat_ablation.py")

    config = normalize_config(loaded["saveables"]["loaded_config"])
    if args.graph_source is not None:
        config["graph_source_dir"] = args.graph_source
    if args.limit_ids is not None:
        config["limit_ids"] = args.limit_ids

    model = CNNSeqConcatModelBuilder.build(config, rt, device).to(device)
    model.load_state_dict(loaded["state"])
    model.eval()

    loader = build_loader(config, args.eval_mode, rt, args.data_dir,
                          num_workers=args.num_workers, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx, inputs in enumerate(loader, start=1):
            if inputs is None:
                continue
            pred = model.predict(inputs).item()
            y_pred.append(pred)
            y_true.append(inputs.y.item())
            if idx % 500 == 0:
                print("[eval] scanned {} batches, usable={}".format(idx, len(y_true)), flush=True)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics = compute_metrics(y_true, y_pred, rt)
    metrics = {k: round(v, 6) for k, v in metrics.items()}
    metrics["n"] = int(len(y_true))
    print("[eval] metrics={}".format(metrics), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cnnseq_concat_{}_metrics.json".format(args.eval_mode)
    with open(out_file, "w") as handle:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "mode": args.eval_mode,
            "metrics": metrics,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }, handle, indent=2)
    print("[eval] saved {}".format(out_file), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HMRLBA CNN-Seq node replacement plus sequence-vector concat ablation."
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--prot-root", default=DEFAULT_PROT)
    parser.add_argument("--data-dir", default=None,
                        help="Dataset root. Defaults to $PROT/Datasets.")
    parser.add_argument("--out-dir", default="/data/HMRLBA_cnnseq_concat_runs")
    parser.add_argument("--config-file",
                        default="/root/HMRLBA_V1.0/configs/Model_training/pdbbind/identity30.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--graph-source", default=None,
                        help="Directory containing one set of cached .pth graphs; PLM features are ignored.")
    parser.add_argument("--limit-ids", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-every-set", action="store_true",
                        help="Apply --eval-every, including setting it to null/None when omitted.")
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--cnn-embed-dim", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--cnn-out-dim", type=int, default=512)
    parser.add_argument("--cnn-dropout", type=float, default=0.15)
    parser.add_argument("--seq-vec-dim", type=int, default=512)

    parser.add_argument("--wandb-project", default="HMRLBA-cnnseq-concat-ablation")
    parser.add_argument("--wandb-entity", default="hzy-team")
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-mode", choices=["train", "valid", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    bootstrap_project(args.prot_root)
    if args.data_dir is None:
        args.data_dir = str(Path(args.prot_root) / "Datasets")
    rt = import_runtime()
    torch = rt["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("Using device: {}".format(device), flush=True)
    if args.mode == "train":
        run_train(args, rt, device)
    else:
        run_eval(args, rt, device)


if __name__ == "__main__":
    main()
