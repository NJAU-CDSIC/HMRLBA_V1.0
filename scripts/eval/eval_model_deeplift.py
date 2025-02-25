import json
import os
import argparse
import torch
import numpy as np
import wandb
import yaml
from scipy import stats
import pandas as pd

from hmrlba_code.data import DATASETS
from hmrlba_code.models.model_builder import MODEL_CLASSES
from hmrlba_code.utils.metrics import DATASET_METRICS, METRICS

# 导入 Captum 中的 DeepLift
from captum.attr import DeepLift

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(os.environ['PROT'], "Datasets")
EXP_DIR = os.path.join(os.environ['PROT'], "Experiments")


def get_model_class(dataset):
    model_class = MODEL_CLASSES.get(dataset)
    return model_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--exp_dir", default=EXP_DIR)
    parser.add_argument("--exp_name", nargs="+")
    args = parser.parse_args()

    metrics_all = {}

    for exp_name in args.exp_name:
        if "run" in exp_name:
            # wandb specific loading
            loaded = torch.load(f"{args.exp_dir}/wandb/{exp_name}/files/best_ckpt.pt", map_location=DEVICE)

            with open(f"{args.exp_dir}/wandb/{exp_name}/files/config.yaml", "r") as f:
                loaded_train_config = yaml.load(f, Loader=yaml.FullLoader)

            train_args = {}
            for key in loaded_train_config:
                if isinstance(loaded_train_config[key], dict):
                    if 'value' in loaded_train_config[key]:
                        train_args[key] = loaded_train_config[key]['value']

        else:
            loaded = torch.load(os.path.join(args.exp_dir, exp_name,
                                             "checkpoints", "best_ckpt.pt"), map_location=DEVICE)

            with open(f"{args.exp_dir}/{exp_name}/args.json", "r") as f:
                train_args = json.load(f)

        dataset = train_args['dataset']
        dataset_class = DATASETS.get(dataset)
        prot_mode = train_args['prot_mode']
        split = train_args['split']
        num_workers = train_args['num_workers']

        raw_dir = f"{args.data_dir}/Raw_data/{dataset}"
        processed_dir = f"{args.data_dir}"

        config = loaded['saveables']
        model_class = get_model_class(dataset)
        model = model_class(**config, device=DEVICE)
        model.load_state_dict(loaded['state'])
        model.to(DEVICE)
        model.eval()

        # =========================
        # 注册 hook 捕获进入 activity_mlp 的融合向量
        fusion_vectors = {}
        def capture_fusion(module, input, output):
            # input 是元组，取第一个元素即为 fusion_vector
            fusion_vectors['fusion'] = input[0]
        hook_handle = model.activity_mlp.register_forward_hook(capture_fusion)
        # =========================

        # 初始化 DeepLIFT，作用于最后的 MLP 层
        deeplift = DeepLift(model.activity_mlp)

        base_dataset = dataset_class(mode='test', raw_dir=raw_dir,
                                     processed_dir=processed_dir,
                                     prot_mode=prot_mode, split=split)
        data_loader = base_dataset.create_loader(batch_size=1, num_workers=num_workers)

        activity_true_all = []
        activity_pred_all = []
        deepLIFT_all = []  # 用于存储每个样本的 DeepLIFT attribution

        for idx, inputs in enumerate(data_loader):
            if inputs is None:
                continue
            else:
                # 预测时会触发 hook，捕获融合向量
                activity_pred = model.predict(inputs)
                activity_pred_item = activity_pred.item()
                activity_true = inputs.y.item()
                activity_true_all.append(activity_true)
                activity_pred_all.append(activity_pred_item)

                # 获取刚刚捕获的融合向量
                fusion_vector = fusion_vectors.get('fusion', None)
                if fusion_vector is None:
                    print("未捕获到融合向量！")
                    continue

                # 确保融合向量可以计算梯度
                fusion_vector = fusion_vector.detach().clone().requires_grad_()
                # 设置基线为全 0 向量（形状与 fusion_vector 相同）
                baseline = torch.zeros_like(fusion_vector)

                # 计算 DeepLIFT attribution，输出与 fusion_vector 同形状
                attribution = deeplift.attribute(fusion_vector, baseline)
                # 可以根据需要保存 attribution，例如转换为 numpy 数组
                deepLIFT_all.append(attribution.detach().cpu().numpy())

                # 如果需要，可打印当前样本 attribution 的统计信息
                print(f"Sample {idx} DeepLIFT attribution: mean={attribution.mean().item():.4f}, "
                      f"std={attribution.std().item():.4f}")

        # 移除 hook，避免影响后续操作
        hook_handle.remove()

        # 计算指标
        activity_true_all = np.array(activity_true_all).flatten()
        activity_pred_all = np.array(activity_pred_all).flatten()

        metrics = DATASET_METRICS.get(dataset)
        print_msgs = []

        for metric in metrics:
            metric_fn, _, _ = METRICS.get(metric)
            metric_val = metric_fn(activity_true_all, activity_pred_all)
            if metric not in metrics_all:
                metrics_all[metric] = [metric_val]
            else:
                metrics_all[metric].append(metric_val)
            print_msgs.append(f"{metric}: {np.round(metric_val, 4)}")

        print_msg = ", ".join(msg for msg in print_msgs)
        print(print_msg, flush=True)

    final_metrics = {}
    for metric, metric_vals in metrics_all.items():
        mean = np.mean(metric_vals)
        std = np.std(metric_vals)
        final_metrics[metric] = (np.round(mean, 4), np.round(std, 4))

    print(f"Final Metrics: {final_metrics}", flush=True)

    # 如果需要，将 deepLIFT_all 保存或后续分析
    np.save("deepLIFT_attributions.npy", np.array(deepLIFT_all))


if __name__ == "__main__":
    main()
