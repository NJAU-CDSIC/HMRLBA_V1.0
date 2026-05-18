#!/usr/bin/env python3
"""
Run non-invasive cascade ablations for HMRLBA.

The original HMRLBA forward pass concatenates three processed protein feature
streams: Ankh, ESM-1b, and ProtTrans.  This script patches a model instance at
runtime so selected streams are zeroed before the original MPNN/MLP stack is
used.  No source file under /root/HMRLBA_V1.0 is modified.
"""

import argparse
import csv
import json
import os
import random
import sys
import types
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_PROT = "/root/HMRLBA_V1.0"
DEFAULT_VARIANTS = [
    "full",
    "drop_prottrans",
    "drop_esm1b_prottrans",
    "drop_all_plm",
]

VARIANT_KEEP = {
    "full": ("ankh", "esm1b", "prottrans"),
    "drop_prottrans": ("ankh", "esm1b"),
    "drop_esm1b_prottrans": ("ankh",),
    "drop_all_plm": (),
    "ankh_only": ("ankh",),
    "esm1b_only": ("esm1b",),
    "prottrans_only": ("prottrans",),
    "no_ankh": ("esm1b", "prottrans"),
    "no_esm1b": ("ankh", "prottrans"),
    "no_prottrans": ("ankh", "esm1b"),
}


def bootstrap_project(prot_root):
    prot_root = str(Path(prot_root).resolve())
    os.environ.setdefault("PROT", prot_root)
    if prot_root not in sys.path:
        sys.path.insert(0, prot_root)
    return prot_root


def import_runtime():
    global np
    if np is None:
        import numpy as _np
        np = _np

    import torch
    from torch_geometric.nn import global_add_pool, global_mean_pool
    import yaml
    import wandb

    from hmrlba_code.data import DATASETS
    from hmrlba_code.models import Trainer
    from hmrlba_code.models.model_builder import build_model, MODEL_CLASSES
    from hmrlba_code.utils.metrics import DATASET_METRICS, METRICS

    return {
        "torch": torch,
        "global_add_pool": global_add_pool,
        "global_mean_pool": global_mean_pool,
        "yaml": yaml,
        "wandb": wandb,
        "DATASETS": DATASETS,
        "Trainer": Trainer,
        "build_model": build_model,
        "MODEL_CLASSES": MODEL_CLASSES,
        "DATASET_METRICS": DATASET_METRICS,
        "METRICS": METRICS,
    }


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def zero_unless(name, keep, tensor, torch):
    if name in keep:
        return tensor
    return torch.zeros_like(tensor)


def install_cascade_forward(model, keep_streams, rt):
    """Patch one model instance with a masked version of HMRLBA.forward."""
    torch = rt["torch"]
    global_add_pool = rt["global_add_pool"]
    global_mean_pool = rt["global_mean_pool"]
    keep = set(keep_streams)

    def cascade_forward(self, complex_data):
        complex_data = complex_data.to(self.device)
        prot = complex_data.protein
        prot2 = complex_data.protein2
        prot3 = complex_data.protein3

        s1_x = zero_unless("ankh", keep, prot.surface.x, torch)
        s2_x = zero_unless("esm1b", keep, prot2.surface.x, torch)
        s3_x = zero_unless("prottrans", keep, prot3.surface.x, torch)
        prot.surface.x = torch.cat((s1_x, s2_x, s3_x), dim=-1)

        b1_x = zero_unless("ankh", keep, prot.backbone.x, torch)
        b2_x = zero_unless("esm1b", keep, prot2.backbone.x, torch)
        b3_x = zero_unless("prottrans", keep, prot3.backbone.x, torch)
        prot.backbone.x = torch.cat((b1_x, b2_x, b3_x), dim=-1)

        lig = complex_data.ligand
        lig_node_emb = self.lig_mpn(lig.x, lig.edge_index, lig.edge_attr)

        if self.config.get("graph_pool", "sum_pool") == "sum_pool":
            lig_emb = global_add_pool(lig_node_emb, lig.batch)
        elif self.config.get("graph_pool", "sum_pool") == "mean_pool":
            lig_emb = global_mean_pool(lig_node_emb, lig.batch)
        else:
            raise ValueError("Unsupported graph_pool: {}".format(self.config.get("graph_pool")))

        top_graph_emb, bottom_graph_emb = self.prot_mpn(prot)
        complex_vec = torch.cat([top_graph_emb, bottom_graph_emb, lig_emb], dim=-1)
        return self.activity_mlp(complex_vec)

    model.cascade_keep_streams = tuple(keep_streams)
    model.forward = types.MethodType(cascade_forward, model)
    return model


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


def load_checkpoint_and_config(exp_dir, exp_name, rt, device):
    torch = rt["torch"]
    yaml = rt["yaml"]
    if "run" in exp_name:
        run_dir = Path(exp_dir) / "wandb" / exp_name / "files"
        ckpt_path = run_dir / "best_ckpt.pt"
        config_path = run_dir / "config.yaml"
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        train_config = load_yaml_config(config_path, yaml)
    else:
        run_dir = Path(exp_dir) / exp_name
        ckpt_path = run_dir / "checkpoints" / "best_ckpt.pt"
        config_path = run_dir / "args.json"
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        with open(config_path, "r") as handle:
            train_config = json.load(handle)
    return checkpoint, train_config


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
    merged = dict(defaults)
    merged.update(config)
    return merged


def build_loader(config, mode, rt, data_dir, eval_dataset=None, shuffle=False, num_workers=None):
    dataset_name = eval_dataset or config["dataset"]
    dataset_class = rt["DATASETS"].get(dataset_name)
    if dataset_class is None:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    kwargs = {"split": config.get("split", "identity30")}
    raw_dir = os.path.abspath("{}/Raw_data/{}".format(data_dir, dataset_name))
    processed_dir = os.path.abspath(data_dir)
    dataset = dataset_class(
        mode=mode,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prot_mode=config["prot_mode"],
        dataset=dataset_name,
        **kwargs,
    )
    workers = config["num_workers"] if num_workers is None else num_workers
    return dataset.create_loader(batch_size=1 if mode != "train" else config["batch_size"],
                                 num_workers=workers,
                                 shuffle=shuffle)


def build_eval_model(checkpoint, train_config, variant, rt, device):
    model_class = rt["MODEL_CLASSES"].get(train_config["dataset"])
    if model_class is None:
        raise ValueError("No model class for dataset: {}".format(train_config["dataset"]))
    model = model_class(**checkpoint["saveables"], device=device)
    model.load_state_dict(checkpoint["state"])
    model.to(device)
    model.eval()
    return install_cascade_forward(model, VARIANT_KEEP[variant], rt)


def compute_regression_metrics(y_true, y_pred, dataset, rt):
    metrics = rt["DATASET_METRICS"].get(dataset)
    if metrics is None:
        metrics = ["rmse", "pearsonr", "mae", "r2", "spearmanr"]
    if not isinstance(metrics, list):
        metrics = [metrics]

    out = {}
    for metric in metrics:
        metric_fn, _, _ = rt["METRICS"].get(metric)
        out[metric] = float(metric_fn(y_true, y_pred))
    return out


def enrichment_factor(y_true, scores, fraction):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    n_total = len(y_true)
    n_active = int(y_true.sum())
    if n_total == 0 or n_active == 0:
        return 0.0
    top_k = max(1, int(np.ceil(n_total * fraction)))
    order = np.argsort(scores)[::-1][:top_k]
    top_active = int(y_true[order].sum())
    return float((top_active / float(top_k)) / (n_active / float(n_total)))


def compute_vs_metrics(y_true, y_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    out = {
        "total": int(len(y_true)),
        "actives": int(y_true.sum()),
        "decoys": int(len(y_true) - y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true.tolist())) == 2 else float("nan"),
        "aupr": float(average_precision_score(y_true, y_pred)) if y_true.sum() > 0 else float("nan"),
        "ef1": enrichment_factor(y_true, y_pred, 0.01),
        "ef5": enrichment_factor(y_true, y_pred, 0.05),
        "ef10": enrichment_factor(y_true, y_pred, 0.10),
    }
    return out


def evaluate_model(model, loader):
    y_true = []
    y_pred = []
    for idx, inputs in enumerate(loader):
        if inputs is None:
            continue
        pred = model.predict(inputs).item()
        y_pred.append(pred)
        y_true.append(inputs.y.item())
        if (idx + 1) % 500 == 0:
            print("  evaluated {} batches".format(idx + 1), flush=True)
    return np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def append_csv(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_eval(args, rt, device):
    checkpoint, train_config = load_checkpoint_and_config(args.exp_dir, args.exp_name, rt, device)
    train_config = normalize_config(train_config)
    eval_dataset = args.eval_dataset or train_config["dataset"]
    mode = args.eval_mode
    loader = build_loader(train_config, mode, rt, args.data_dir, eval_dataset=eval_dataset,
                          num_workers=args.num_workers)

    rows = []
    for variant in args.variants:
        print("\n[eval] variant={} keep={}".format(variant, ",".join(VARIANT_KEEP[variant]) or "none"), flush=True)
        model = build_eval_model(checkpoint, train_config, variant, rt, device)
        y_true, y_pred = evaluate_model(model, loader)

        if args.virtual_screening or eval_dataset.startswith("dude_"):
            metrics = compute_vs_metrics(y_true, y_pred)
        else:
            metrics = compute_regression_metrics(y_true, y_pred, train_config["dataset"], rt)

        row = {"variant": variant, "dataset": eval_dataset, "mode": mode}
        row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()})
        rows.append(row)
        append_csv(Path(args.out_dir) / "cascade_ablation_eval.csv", row)
        write_json(Path(args.out_dir) / "{}_{}_metrics.json".format(eval_dataset, variant),
                   {"row": row, "y_true": y_true.tolist(), "y_pred": y_pred.tolist()})
        print("  metrics: {}".format(row), flush=True)

    write_json(Path(args.out_dir) / "cascade_ablation_eval_summary.json", rows)


def run_train(args, rt, device):
    torch = rt["torch"]
    yaml = rt["yaml"]
    wandb = rt["wandb"]
    Trainer = rt["Trainer"]
    build_model = rt["build_model"]

    config = load_yaml_config(args.config_file, yaml)
    config = normalize_config(config)
    config["data_dir"] = args.data_dir
    config["out_dir"] = args.out_dir

    if not args.wandb_online:
        os.environ.setdefault("WANDB_MODE", "offline")

    for seed in args.seeds:
        seed_everything(seed, torch)
        for variant in args.variants:
            run_name = "cascade_{}_seed{}".format(variant, seed)
            print("\n[train] {} keep={}".format(run_name, ",".join(VARIANT_KEEP[variant]) or "none"), flush=True)

            run_config = dict(config)
            run_config["seed"] = seed
            run_config["cascade_variant"] = variant
            run_config["cascade_keep_streams"] = list(VARIANT_KEEP[variant])

            wandb.init(project=args.wandb_project, dir=args.out_dir,
                       entity=args.wandb_entity, name=run_name, config=run_config,
                       reinit=True)
            try:
                if run_config["use_doubles"]:
                    torch.set_default_dtype(torch.float64)
                else:
                    torch.set_default_dtype(torch.float32)

                model = build_model(run_config, device=device)
                model = install_cascade_forward(model, VARIANT_KEEP[variant], rt)
                model.to(device)

                train_loader = build_loader(run_config, "train", rt, args.data_dir, shuffle=True,
                                            num_workers=args.num_workers)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HMRLBA PLM-stream cascade ablations without modifying original source."
    )
    parser.add_argument("--prot-root", default=DEFAULT_PROT)
    parser.add_argument("--data-dir", default=None,
                        help="Dataset root. Defaults to $PROT/Datasets.")
    parser.add_argument("--out-dir", default="/data/HMRLBA_ablation_runs")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=sorted(VARIANT_KEEP))
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--config-file",
                        default="/root/HMRLBA_V1.0/configs/Model_training/pdbbind/identity30.yaml",
                        help="Training config YAML, used by --mode train.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--wandb-project", default="HMRLBA-cascade-ablation")
    parser.add_argument("--wandb-entity", default="hzy-team")
    parser.add_argument("--wandb-online", action="store_true")

    parser.add_argument("--exp-dir", default="/root/HMRLBA_V1.0/Experiments")
    parser.add_argument("--exp-name", default="run-20241124_204606-r94ymd7y")
    parser.add_argument("--eval-dataset", default=None,
                        help="Optional dataset for evaluation, e.g. dude_pygm.")
    parser.add_argument("--eval-mode", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--virtual-screening", action="store_true",
                        help="Compute AUROC/AUPR/EF metrics instead of regression metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    bootstrap_project(args.prot_root)
    if args.data_dir is None:
        args.data_dir = str(Path(args.prot_root) / "Datasets")
    rt = import_runtime()
    torch = rt["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device), flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        run_eval(args, rt, device)
    else:
        run_train(args, rt, device)


if __name__ == "__main__":
    main()
