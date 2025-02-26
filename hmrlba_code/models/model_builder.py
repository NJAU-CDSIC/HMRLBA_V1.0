"""
Functions that take in command line arguments and build the model.
"""
import json
import numpy as np
from hmrlba_code.models.affinity_pred import AffinityPred
from hmrlba_code.feat import (AMINO_ACIDS, SECONDARY_STRUCTS, ATOM_FDIM,
                              BOND_FDIM, CONTACT_FDIM, SURFACE_NODE_FDIM, SURFACE_EDGE_FDIM,
                              PATCH_NODE_FDIM, PATCH_EDGE_FDIM)
from hmrlba_code.models.prot_classifier import ProtClassifier
from hmrlba_code.utils.metrics import EVAL_METRICS, DATASET_METRICS, METRICS

MODEL_CLASSES = {
    'pdbbind': AffinityPred,
    'enzyme': ProtClassifier
}


def build_mpn_config(config):
    mpn_config = {}

    bconfig, sconfig = {}, {}
    # bconfig['node_fdim'] = len(AMINO_ACIDS) + len(SECONDARY_STRUCTS) + 2
    bconfig['node_fdim'] = 1280 * 3
    bconfig['edge_fdim'] = CONTACT_FDIM
    bconfig['hsize'] = config['bhsize']
    bconfig['depth'] = config['bdepth']
    bconfig['dropout_p'] = config['dropout_mpn']
    bconfig['activation'] = config.get('activation', 'relu')
    bconfig['jk_pool'] = config.get("jk_pool", None)  # default

    sconfig['hsize'] = config['shsize']
    sconfig['depth'] = config['sdepth']
    sconfig['dropout_p'] = config['dropout_mpn']
    sconfig['activation'] = config.get('activation', 'relu')
    sconfig['jk_pool'] = config.get("jk_pool", None)  # default

    if config['prot_mode'] == 'surface2backbone':
        sconfig['node_fdim'] = 1280 * 3
        sconfig['edge_fdim'] = SURFACE_EDGE_FDIM + CONTACT_FDIM

    if config.get("use_mpn_in_patch", False):
        sp_config = {}
        sp_config['node_fdim'] = SURFACE_NODE_FDIM
        sp_config['edge_fdim'] = SURFACE_EDGE_FDIM + CONTACT_FDIM
        sp_config['hsize'] = config['shsize']  # Use the same hsize as patch
        sp_config['depth'] = config['sdepth']  # Use the same depth as patch
        sp_config['activation'] = config.get('activation', 'relu')
        sp_config['dropout_p'] = config['dropout_mpn']
        sp_config['jk_pool'] = config.get("jk_pool", None)  # default
        mpn_config['sp_config'] = sp_config

    mpn_config['bconfig'] = bconfig
    mpn_config['sconfig'] = sconfig
    mpn_config['dropout_mlp'] = config['dropout_mlp']  # Check this part

    mpn_config['graph_pool'] = config.get("graph_pool", "sum_pool")
    return mpn_config


def affinity_pred_config(loaded_config):
    model_config = {}
    config = {}
    toggles = {}
    mpn_config = build_mpn_config(loaded_config)

    config['mpn_config'] = mpn_config
    config['atom_fdim'] = ATOM_FDIM
    config['bond_fdim'] = BOND_FDIM
    config['lig_hsize'] = loaded_config['lig_hsize']
    if isinstance(loaded_config['hsize'], int):
        config['hsize'] = [loaded_config['hsize']]
    else:
        config['hsize'] = loaded_config['hsize']
    config['encoder'] = loaded_config['encoder']
    config['lig_depth'] = loaded_config['lig_depth']
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['dropout_mpn'] = loaded_config['dropout_mpn']
    config['prot_mode'] = loaded_config['prot_mode']
    config['activation'] = loaded_config.get("activation", "relu")
    config['jk_pool'] = loaded_config.get("jk_pool", None)
    config['graph_pool'] = loaded_config.get("graph_pool", "sum_pool")

    toggles['ligand_only'] = loaded_config.get('ligand_only', False)
    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_mpn_in_patch'] = loaded_config.get('use_mpn_in_patch', False)

    if config['prot_mode'] in ['backbone', 'surface']:
        toggles['use_attn'] = False

    model_config['config'] = config
    model_config['toggles'] = toggles
    return model_config


def prot_class_config(loaded_config):
    model_config = {}
    config = {}
    toggles = {}

    mpn_config = build_mpn_config(loaded_config)
    config['mpn_config'] = mpn_config
    config['n_classes'] = loaded_config['n_classes']
    if isinstance(loaded_config['hsize'], int):
        config['hsize'] = [loaded_config['hsize']]
    else:
        config['hsize'] = loaded_config['hsize']
    config['encoder'] = loaded_config['encoder']
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['prot_mode'] = loaded_config['prot_mode']
    config['activation'] = loaded_config.get("activation", "relu")
    config['jk_pool'] = loaded_config.get("jk_pool", None)

    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_mpn_in_patch'] = loaded_config.get('use_mpn_in_patch', False)

    if config['prot_mode'] in ['backbone', 'surface']:
        toggles['use_attn'] = False
    model_config['config'] = config
    model_config['toggles'] = toggles
    return model_config


CONFIG_FNS = {
    'pdbbind': affinity_pred_config,
    'enzyme': prot_class_config
}


def compute_function_weights(config):
    raw_dir = f"{config['data_dir']}/raw/enzyme/metadata"
    labels_to_idx_file = f"{raw_dir}/labels_to_idx.json"
    labels_file = f"{raw_dir}/function_labels.json"
    with open(labels_to_idx_file, "r") as f:
        labels_to_idx = json.load(f)

    with open(labels_file, "r") as f:
        function_labels = json.load(f)

    weights = np.full((len(labels_to_idx)), 0, dtype=np.int32)
    for pdb_id in function_labels:
        weights[labels_to_idx[function_labels[pdb_id]]] += 1
    weights = weights.astype(np.float32) / (float(len(function_labels)) * 0.5)

    print("Min occurence: ", np.amin(weights))
    print("Max occurence: ", np.amax(weights))

    weights_log = 1.0 / np.log(1.2 + weights)
    return weights_log


def build_model(loaded_config, device='cpu'):
    config_fn = CONFIG_FNS.get(loaded_config['dataset'])
    dataset_metrics = DATASET_METRICS.get(loaded_config['dataset'])
    if not isinstance(dataset_metrics, list):
        dataset_metrics = [dataset_metrics]
    model_config = config_fn(loaded_config)

    metrics = {}
    for metric in dataset_metrics:
        metrics[metric], _, _ = METRICS.get(metric)

    if loaded_config['dataset'] == 'enzyme':
        weights = compute_function_weights(loaded_config)
        model_config['class_weights'] = weights

    model_config['metrics'] = metrics
    model_class = MODEL_CLASSES.get(loaded_config['dataset'])
    model = model_class(**model_config, device=device)
    return model
