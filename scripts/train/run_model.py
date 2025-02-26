import torch
import os
import argparse
from datetime import datetime as dt
from rdkit import RDLogger
import json
import wandb
import yaml
import multiprocessing

from hmrlba_code.models.model_builder import build_model, MODEL_CLASSES
from hmrlba_code.models import Trainer
from hmrlba_code.data import DATASETS

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
    ROOT_DIR = os.environ["PROT"]
    # DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "Datasets")
    DATA_DIR = os.path.join(ROOT_DIR, "Datasets")
    EXP_DIR = os.path.join(ROOT_DIR, "Experiments")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "Datasets")
    EXP_DIR = os.path.join(ROOT_DIR, "local_experiments")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_model(config):
    # set default type
    if config['use_doubles']:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    # prepare model
    model = build_model(config, device=DEVICE)
    print(f"Converting model to device: {DEVICE}", flush=True)
    model.to(DEVICE)

    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10 ** 6, "M", flush=True)
    print(flush=True)

    print(f"Device used: {DEVICE}", flush=True)

    dataset_class = DATASETS.get(config['dataset'])
    kwargs = {'split': config['split']}
    if config['dataset'] in ['scope', 'enzyme']:
        kwargs['add_target'] = True

    raw_dir = os.path.abspath(f"{config['data_dir']}/Raw_data/{config['dataset']}")

    processed_dir = os.path.abspath(f"{config['data_dir']}")

    train_dataset = dataset_class(mode='train', raw_dir=raw_dir,
                                  processed_dir=processed_dir,
                                  prot_mode=config['prot_mode'], dataset=config['dataset'], **kwargs)

    train_data = train_dataset.create_loader(batch_size=config['batch_size'], num_workers=config['num_workers'],
                                             shuffle=True)
    eval_data = None

    if config['eval_every'] is not None:
        eval_dataset = dataset_class(mode='valid', raw_dir=raw_dir,
                                     processed_dir=processed_dir,
                                     prot_mode=config['prot_mode'], dataset=config['dataset'], **kwargs)
        eval_data = eval_dataset.create_loader(batch_size=1,
                                               num_workers=config['num_workers'])

    trainer = Trainer(model=model, dataset=config['dataset'], task_type=config['prot_mode'],
                      print_every=config['print_every'], eval_every=config['eval_every'])
    trainer.build_optimizer(learning_rate=config['lr'])
    trainer.build_scheduler(type=config['scheduler_type'], step_after=config['step_after'],
                            anneal_rate=config['anneal_rate'], patience=config['patience'],
                            thresh=config['metric_thresh'])
    trainer.train_epochs(train_data, eval_data, config['epochs'],
                         **{"accum_every": config['accum_every'],
                            "clip_norm": config['clip_norm']})


def main(args):
    # initialize wandb
    wandb.init(project='HMRLBA', dir=args.out_dir,
               entity='hzy-team',
               config=args.config_file)
    config = wandb.config
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        config[key] = value

    print(config)
    # start run
    run_model(config)


if __name__ == "__main__":
    def get_argparse():
        parser = argparse.ArgumentParser()

        # Logging and setup args
        parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
        parser.add_argument("--out_dir", default=EXP_DIR, help="Experiments directory")
        parser.add_argument("--config_file", default=None)

        args = parser.parse_args()
        return args


    # get args
    args = get_argparse()

    main(args)
