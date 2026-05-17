#!/usr/bin/env python3
"""
预处理补充数据并重新测试五折交叉验证
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import json

# 导入项目模块
import sys
sys.path.append('/root/MEGDTA')
from preprocess_data import smiles_to_graph, smiles_to_ecfp, sequence_to_graph
from models import DTIProtGraphChemGraphECFP
from utils.metrics import get_cindex, get_rm2, get_metrics_reg
from scipy.stats import pearsonr, spearmanr

def extract_smiles_from_sdf(sdf_file):
    """从SDF或MOL2文件提取SMILES"""
    pdb_dir = sdf_file.parent
    pdb_id = pdb_dir.name

    # 尝试SDF文件
    try:
        mol = Chem.SDMolSupplier(str(sdf_file), sanitize=True, removeHs=True)[0]
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass

    # 尝试MOL2文件
    mol2_file = pdb_dir / f"{pdb_id}_ligand.mol2"
    if mol2_file.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_file), sanitize=True, removeHs=True)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except:
            pass

    # 尝试不sanitize的MOL2
    if mol2_file.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_file), sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    return Chem.MolToSmiles(mol)
                except:
                    # 即使sanitize失败，也尝试生成SMILES
                    try:
                        return Chem.MolToSmiles(mol, sanitize=False)
                    except:
                        pass
        except:
            pass

    return None

def extract_sequence_from_pdb(pdb_file):
    """从PDB文件提取蛋白质序列（简单文本解析）"""
    # 标准氨基酸三字母到单字母的映射
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    sequence = []
    seen_residues = set()

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # PDB格式：ATOM行包含残基名称和残基编号
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                chain_id = line[21]

                # 创建唯一标识符
                res_id = f"{chain_id}_{res_num}"

                # 如果是标准氨基酸且未见过
                if res_name in aa_map and res_id not in seen_residues:
                    sequence.append(aa_map[res_name])
                    seen_residues.add(res_id)

    return ''.join(sequence)

def load_casf_affinity_data():
    """加载CASF-2016的亲和力数据"""
    affinity_file = Path('/root/MEGDTA/supplement_data/CASF-2016_PDB_IDs.csv')

    if not affinity_file.exists():
        print(f"警告: 未找到亲和力数据文件 {affinity_file}")
        print("将使用默认值0.0")
        return {}

    # 读取CSV文件，跳过注释行
    df = pd.read_csv(affinity_file)

    # 检查列名
    print(f"CSV列名: {df.columns.tolist()}")

    affinity_dict = {}
    for _, row in df.iterrows():
        pdb_id = str(row['#code']).lower().strip()  # 注意列名是'#code'
        logKa = float(row['logKa'])
        affinity_dict[pdb_id] = logKa

    print(f"加载了 {len(affinity_dict)} 个样本的亲和力数据")

    return affinity_dict

def preprocess_supplement_data():
    """预处理补充数据"""
    supplement_dir = Path('/root/MEGDTA/supplement_data')
    pdb_dirs = sorted([d for d in supplement_dir.iterdir() if d.is_dir()])

    print(f"找到 {len(pdb_dirs)} 个补充样本")
    print("="*80)

    # 加载亲和力数据
    affinity_dict = load_casf_affinity_data()

    # 存储提取的数据
    supplement_data = []
    ligand_to_graph = {}
    ligand_to_ecfp = {}
    protein_to_graph = {}

    for pdb_dir in tqdm(pdb_dirs, desc="预处理补充数据"):
        pdb_id = pdb_dir.name.lower()

        # 查找文件
        sdf_file = pdb_dir / f"{pdb_id}_ligand.sdf"
        pdb_file = pdb_dir / f"{pdb_id}_protein.pdb"

        if not sdf_file.exists() or not pdb_file.exists():
            print(f"警告: {pdb_id} 缺少必要文件")
            continue

        try:
            # 提取SMILES
            smiles = extract_smiles_from_sdf(sdf_file)
            if smiles is None:
                print(f"警告: {pdb_id} 无法提取SMILES")
                continue

            # 提取蛋白质序列
            sequence = extract_sequence_from_pdb(pdb_file)
            if not sequence:
                print(f"警告: {pdb_id} 无法提取蛋白质序列")
                continue

            # 获取亲和力值
            affinity = affinity_dict.get(pdb_id, 0.0)

            # 生成配体图
            ligand_graph = smiles_to_graph(smiles)
            if ligand_graph is None:
                print(f"警告: {pdb_id} 无法生成配体图")
                continue

            # 生成配体ECFP
            ecfp_array = np.array(smiles_to_ecfp(smiles, radius=2, n_bits=2048))

            # 生成蛋白质图
            protein_graph = sequence_to_graph(sequence)
            if protein_graph is None:
                print(f"警告: {pdb_id} 无法生成蛋白质图")
                continue

            # 存储数据
            supplement_data.append({
                'pdb_id': pdb_id,
                'ligand': smiles,
                'protein': pdb_id,  # 使用pdb_id作为蛋白质标识
                'label': affinity
            })

            ligand_to_graph[smiles] = ligand_graph
            ligand_to_ecfp[smiles] = ecfp_array
            protein_to_graph[pdb_id] = protein_graph

            print(f"✓ {pdb_id}: SMILES长度={len(smiles)}, 序列长度={len(sequence)}, 亲和力={affinity}")

        except Exception as e:
            print(f"错误: {pdb_id} 处理失败: {e}")
            continue

    print(f"\n成功预处理 {len(supplement_data)} 个样本")

    return supplement_data, ligand_to_graph, ligand_to_ecfp, protein_to_graph

def merge_with_test_set(supplement_data, ligand_to_graph, ligand_to_ecfp, protein_to_graph):
    """将补充数据与原测试集合并"""
    casf_dir = Path('/root/MEGDTA/data/casf')

    print("加载原始数据...")
    # 加载原始数据
    original_df = pd.read_csv(casf_dir / 'updated_full.csv')

    print("加载原始图数据...")
    with open(casf_dir / 'ligand_to_graph.pkl', 'rb') as f:
        original_ligand_graph = pickle.load(f)
    with open(casf_dir / 'ligand_to_ecfp.pkl', 'rb') as f:
        original_ligand_ecfp = pickle.load(f)
    with open(casf_dir / 'protein_to_graph.pkl', 'rb') as f:
        original_protein_graph = pickle.load(f)

    # 读取原测试集索引
    with open(casf_dir / 'folds' / 'test_fold_all.txt', 'r') as f:
        original_test_indices = [int(line.strip()) for line in f if line.strip()]

    print(f"原测试集样本数: {len(original_test_indices)}")
    print(f"补充样本数: {len(supplement_data)}")

    # 创建补充数据的DataFrame
    supplement_df = pd.DataFrame(supplement_data)

    # 为补充数据添加series列（使用空列表占位）
    if 'series' in original_df.columns and 'series' not in supplement_df.columns:
        print("为补充数据添加series列...")
        # 使用空字符串占位，后续会被pad_sequences处理
        supplement_df['series'] = [[] for _ in range(len(supplement_df))]

    print("合并DataFrame...")
    # 合并数据
    merged_df = pd.concat([original_df, supplement_df], ignore_index=True)

    print("合并图数据...")
    # 更新图数据
    merged_ligand_graph = {**original_ligand_graph, **ligand_to_graph}
    merged_ligand_ecfp = {**original_ligand_ecfp, **ligand_to_ecfp}
    merged_protein_graph = {**original_protein_graph, **protein_to_graph}

    # 新测试集索引 = 原测试集索引 + 新增样本的索引
    new_test_indices = original_test_indices + list(range(len(original_df), len(merged_df)))

    print(f"合并后总样本数: {len(merged_df)}")
    print(f"新测试集样本数: {len(new_test_indices)}")

    # 保存合并后的数据
    output_dir = Path('/root/MEGDTA/data/casf_extended')
    output_dir.mkdir(exist_ok=True)

    print("保存合并后的数据...")
    merged_df.to_csv(output_dir / 'updated_full.csv', index=False)

    print("保存图数据...")
    with open(output_dir / 'ligand_to_graph.pkl', 'wb') as f:
        pickle.dump(merged_ligand_graph, f)
    with open(output_dir / 'ligand_to_ecfp.pkl', 'wb') as f:
        pickle.dump(merged_ligand_ecfp, f)
    with open(output_dir / 'protein_to_graph.pkl', 'wb') as f:
        pickle.dump(merged_protein_graph, f)

    # 保存新测试集索引
    folds_dir = output_dir / 'folds'
    folds_dir.mkdir(exist_ok=True)

    with open(folds_dir / 'test_fold_all.txt', 'w') as f:
        for idx in new_test_indices:
            f.write(f"{idx}\n")

    # 复制训练集和验证集索引
    print("复制fold索引文件...")
    for fold in range(1, 6):
        for split in ['train', 'val']:
            src = casf_dir / 'folds' / f'{split}_fold{fold}.txt'
            dst = folds_dir / f'{split}_fold{fold}.txt'
            with open(src, 'r') as f_in, open(dst, 'w') as f_out:
                f_out.write(f_in.read())

    print(f"\n合并后的数据已保存到: {output_dir}")

    return output_dir, new_test_indices

def evaluate_model_on_test_set(model_path, data_dir, test_indices, device):
    """在测试集上评估模型"""
    from graph_loader import load_fold_data

    # 加载数据
    _, _, test_loader = load_fold_data(
        data_dir=str(data_dir),
        fold=1,  # fold参数不影响测试集
        batch_size=128
    )

    # 加载模型
    model = DTIProtGraphChemGraphECFP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预测
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            # 解包batch数据
            ligand_graph, ligand_ecfp, protein_graph, protein_seq, label = batch

            # 移动到设备
            ligand_graph = ligand_graph.to(device)
            ligand_ecfp = ligand_ecfp.to(device)
            protein_graph = protein_graph.to(device)
            protein_seq = protein_seq.to(device)
            label = label.to(device)

            # 预测
            output = model(ligand_graph, ligand_ecfp, protein_graph, protein_seq)

            predictions.extend(output.cpu().numpy().flatten())
            labels.extend(label.cpu().numpy().flatten())

    predictions = np.array(predictions)
    labels = np.array(labels)

    # 计算指标
    rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    mae = np.mean(np.abs(labels - predictions))
    pearson, _ = pearsonr(labels, predictions)
    spearman, _ = spearmanr(labels, predictions)
    r2 = pearson ** 2

    return {
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'spearman': spearman,
        'r2': r2,
        'predictions': predictions,
        'labels': labels
    }

def main():
    print("="*80)
    print("预处理补充数据并重新测试五折交叉验证")
    print("="*80)
    print()

    # 1. 预处理补充数据
    print("步骤 1: 预处理补充数据")
    print("-"*80)
    supplement_data, ligand_to_graph, ligand_to_ecfp, protein_to_graph = preprocess_supplement_data()
    print()

    # 2. 合并数据
    print("步骤 2: 合并补充数据与原测试集")
    print("-"*80)
    data_dir, new_test_indices = merge_with_test_set(
        supplement_data, ligand_to_graph, ligand_to_ecfp, protein_to_graph
    )
    print()

    # 3. 在新测试集上评估5个模型
    print("步骤 3: 在新测试集上评估5个模型")
    print("-"*80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()

    results_dir = Path('/root/MEGDTA/results_casf')
    all_results = []

    for fold in range(1, 6):
        print(f"\nFold {fold}:")
        print("-"*40)

        model_path = results_dir / f'best_model_fold{fold}.pth'

        if not model_path.exists():
            print(f"警告: 模型文件不存在 {model_path}")
            continue

        try:
            result = evaluate_model_on_test_set(model_path, data_dir, new_test_indices, device)

            print(f"  RMSE:     {result['rmse']:.4f}")
            print(f"  MAE:      {result['mae']:.4f}")
            print(f"  Pearson:  {result['pearson']:.4f}")
            print(f"  Spearman: {result['spearman']:.4f}")
            print(f"  R²:       {result['r2']:.4f}")

            all_results.append({
                'fold': fold,
                **{k: v for k, v in result.items() if k not in ['predictions', 'labels']}
            })

            # 保存预测结果
            pred_df = pd.DataFrame({
                'label': result['labels'],
                'prediction': result['predictions']
            })
            pred_df.to_csv(results_dir / f'predictions_fold{fold}_extended.csv', index=False)

        except Exception as e:
            print(f"错误: Fold {fold} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. 生成汇总报告
    print("\n" + "="*80)
    print("汇总结果")
    print("="*80)

    if all_results:
        results_df = pd.DataFrame(all_results)

        print(f"\n测试集样本数: {len(new_test_indices)}")
        print(f"  原始样本: 260")
        print(f"  补充样本: {len(new_test_indices) - 260}")
        print()

        print("各Fold结果:")
        print(results_df.to_string(index=False))
        print()

        print("平均结果:")
        for metric in ['rmse', 'mae', 'pearson', 'spearman', 'r2']:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")

        # 保存结果
        results_df.to_csv(results_dir / 'extended_test_results.csv', index=False)

        # 保存详细报告
        with open(results_dir / 'extended_test_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("CASF-2016扩展测试集评估报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"测试集样本数: {len(new_test_indices)}\n")
            f.write(f"  原始样本: 260\n")
            f.write(f"  补充样本: {len(new_test_indices) - 260}\n\n")
            f.write("各Fold结果:\n")
            f.write(results_df.to_string(index=False) + "\n\n")
            f.write("平均结果:\n")
            for metric in ['rmse', 'mae', 'pearson', 'spearman', 'r2']:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                f.write(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}\n")

        print(f"\n结果已保存到: {results_dir}")
    else:
        print("没有成功评估任何模型")

if __name__ == '__main__':
    main()
