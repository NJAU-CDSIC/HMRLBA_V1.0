"""
MEGDTA 数据预处理脚本

功能：
1. 从SMILES生成化合物图 (ligand_to_graph.pkl)
2. 从SMILES生成ECFP指纹 (ligand_to_ecfp.pkl)
3. 从蛋白质序列生成蛋白质图 (protein_to_graph.pkl)
4. 生成蛋白质序列编码 (series)

使用方法：
python preprocess_data.py --input your_data.csv --output_dir data/mydataset/

输入CSV格式：
ligand_smiles,protein_sequence,affinity
CC1=C2C=C...,MKKFFDSRREQGG...,7.366
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    print("错误: 需要安装RDKit")
    print("安装命令: conda install -c conda-forge rdkit")
    exit(1)


# ==================== 化合物图生成 ====================

def one_of_k_encoding(x, allowable_set):
    """One-hot编码"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    """
    提取原子特征 (22维) - 与MEGDTA原始实现一致

    特征包括：
    - 原子类型 one-hot (9维): C, N, O, S, F, P, Cl, Br, I
    - 原子序数 (1维)
    - Gasteiger电荷 (1维)
    - 原子质量 (1维)
    - 杂化类型 one-hot (5维): SP, SP2, SP3, SP3D, SP3D2
    - 总价 (1维)
    - 形式电荷 (1维)
    - 自由基电子数 (1维)
    - 总氢数 (1维)
    - 是否芳香性 (1维)
    """
    from rdkit.Chem import rdPartialCharges
    import math

    # 计算Gasteiger电荷
    rdPartialCharges.ComputeGasteigerCharges(atom.GetOwningMol(), throwOnParamFailure=False)
    charge = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
    try:
        charge = float(charge)
    except ValueError:
        charge = 0.0
    if not math.isfinite(charge):
        charge = 0.0

    # 原子符号
    ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]

    # 杂化类型
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    features = []
    features.extend(one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOLS))
    features.extend([
        float(atom.GetAtomicNum()),
        float(charge),
        float(atom.GetMass()),
    ])
    features.extend(one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend([
        float(atom.GetTotalValence()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
    ])

    assert len(features) == 22, f"Expected 22 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def bond_features(bond):
    """
    提取化学键特征 (12维)

    特征包括：
    - 键类型 (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    - 是否在环中
    - 是否共轭
    """
    bt = bond.GetBondType()
    return np.array([
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ] + [0] * 6).astype(np.float32)  # 补齐到12维


def smiles_to_graph(smiles):
    """
    从SMILES生成分子图

    返回: (nodes, edges, edge_attr)
    - nodes: [num_atoms, 22] 原子特征矩阵
    - edges: [num_bonds, 2] 边索引 (无向图，每条边存两次)
    - edge_attr: [num_bonds, 12] 边特征矩阵
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES: {smiles}")

    # 提取原子特征
    atoms = mol.GetAtoms()
    nodes = np.array([atom_features(atom) for atom in atoms])

    # 提取边和边特征
    edges = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 无向图：添加两个方向
        edges.append([i, j])
        edges.append([j, i])

        # 边特征
        bf = bond_features(bond)
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    edges = np.array(edges, dtype=np.int64)
    edge_attrs = np.array(edge_attrs, dtype=np.float32)

    return [nodes.tolist(), edges.tolist(), edge_attrs.tolist()]


def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    """
    从SMILES生成ECFP指纹 (Extended Connectivity Fingerprint)

    参数:
    - radius: 半径 (默认2，即ECFP4)
    - n_bits: 指纹长度 (默认2048)

    返回: [2048] 的0/1列表
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fp.ToList()


# ==================== 蛋白质图生成 ====================

# 氨基酸字母表
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

# 氨基酸理化性质 (简化版)
AA_PROPERTIES = {
    'A': [1, 0, 0, 0, 0],  # 疏水性, 极性, 正电, 负电, 芳香性
    'C': [1, 0, 0, 0, 0],
    'D': [0, 1, 0, 1, 0],
    'E': [0, 1, 0, 1, 0],
    'F': [1, 0, 0, 0, 1],
    'G': [0, 0, 0, 0, 0],
    'H': [0, 1, 1, 0, 1],
    'I': [1, 0, 0, 0, 0],
    'K': [0, 1, 1, 0, 0],
    'L': [1, 0, 0, 0, 0],
    'M': [1, 0, 0, 0, 0],
    'N': [0, 1, 0, 0, 0],
    'P': [0, 0, 0, 0, 0],
    'Q': [0, 1, 0, 0, 0],
    'R': [0, 1, 1, 0, 0],
    'S': [0, 1, 0, 0, 0],
    'T': [0, 1, 0, 0, 0],
    'V': [1, 0, 0, 0, 0],
    'W': [1, 0, 0, 0, 1],
    'Y': [0, 1, 0, 0, 1],
    'X': [0, 0, 0, 0, 0],  # 未知氨基酸
}


def residue_features(residue):
    """
    提取氨基酸残基特征 (35维) - 与PDBbind数据集一致

    特征包括：
    - 氨基酸类型 one-hot (21维)
    - 理化性质 (5维)
    - 位置编码 (9维，预留)
    """
    if residue not in AMINO_ACIDS:
        residue = 'X'

    # One-hot编码
    aa_onehot = one_of_k_encoding(residue, AMINO_ACIDS)

    # 理化性质
    properties = AA_PROPERTIES[residue]

    # 位置编码 (简化版，全0) - 9维以达到总共35维
    position = [0] * 9

    return np.array(aa_onehot + properties + position, dtype=np.float32)


def sequence_to_graph(sequence, contact_threshold=8.0):
    """
    从蛋白质序列生成接触图

    注意: 这是简化版本，使用序列距离作为接触判断
    真实版本需要3D结构 (AlphaFold2预测)

    参数:
    - sequence: 蛋白质序列字符串
    - contact_threshold: 接触阈值 (序列距离)

    返回: (nodes, edges, edge_attr)
    - nodes: [seq_len, 34] 残基特征矩阵
    - edges: [num_contacts, 2] 边索引
    - edge_attr: [num_contacts, 7] 边特征矩阵
    """
    seq_len = len(sequence)

    # 提取残基特征
    nodes = np.array([residue_features(aa) for aa in sequence])

    # 构建接触图 (简化版: 序列距离 < threshold)
    edges = []
    edge_attrs = []

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            seq_dist = j - i

            # 简化规则: 相邻残基 + 每隔5个残基的长程接触
            if seq_dist <= 1 or (seq_dist >= 5 and seq_dist % 5 == 0 and seq_dist <= 50):
                # 添加双向边
                edges.append([i, j])
                edges.append([j, i])

                # 边特征 (7维): 序列距离的one-hot编码
                edge_feat = [
                    int(seq_dist == 1),      # 相邻
                    int(seq_dist <= 5),      # 短程
                    int(5 < seq_dist <= 10), # 中程
                    int(seq_dist > 10),      # 长程
                    0, 0, 0                  # 预留
                ]
                edge_attrs.append(edge_feat)
                edge_attrs.append(edge_feat)

    edges = np.array(edges, dtype=np.int64)
    edge_attrs = np.array(edge_attrs, dtype=np.float32)

    return [nodes.tolist(), edges.tolist(), edge_attrs.tolist()]


def sequence_to_encoding(sequence, vocab=None):
    """
    将蛋白质序列编码为整数列表

    参数:
    - sequence: 蛋白质序列
    - vocab: 氨基酸到整数的映射 (默认使用标准映射)

    返回: 整数列表
    """
    if vocab is None:
        vocab = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
        vocab['X'] = 0  # 未知氨基酸

    encoding = []
    for aa in sequence:
        if aa in vocab:
            encoding.append(vocab[aa])
        else:
            encoding.append(vocab['X'])

    return encoding


# ==================== 主处理流程 ====================

def preprocess_dataset(input_csv, output_dir, dataset_name="mydataset"):
    """
    预处理完整数据集

    参数:
    - input_csv: 输入CSV文件路径
    - output_dir: 输出目录
    - dataset_name: 数据集名称
    """
    print(f"开始预处理数据集: {dataset_name}")
    print(f"输入文件: {input_csv}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "folds").mkdir(exist_ok=True)

    # 读取数据
    print("\n1. 读取数据...")
    df = pd.read_csv(input_csv)
    print(f"   总样本数: {len(df)}")

    required_cols = ['ligand_smiles', 'protein_sequence', 'affinity']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV必须包含列: {required_cols}")

    # 提取唯一的化合物和蛋白质
    unique_ligands = df['ligand_smiles'].unique()
    unique_proteins = df['protein_sequence'].unique()
    print(f"   唯一化合物数: {len(unique_ligands)}")
    print(f"   唯一蛋白质数: {len(unique_proteins)}")

    # 生成化合物图
    print("\n2. 生成化合物图...")
    ligand_to_graph = {}
    ligand_to_ecfp = {}

    for smiles in tqdm(unique_ligands, desc="   处理化合物"):
        try:
            ligand_to_graph[smiles] = smiles_to_graph(smiles)
            ligand_to_ecfp[smiles] = smiles_to_ecfp(smiles)
        except Exception as e:
            print(f"\n   警告: 化合物 {smiles[:50]}... 处理失败: {e}")
            continue

    print(f"   成功处理: {len(ligand_to_graph)} 个化合物")

    # 生成蛋白质图
    print("\n3. 生成蛋白质图...")
    protein_to_graph = {}

    for seq in tqdm(unique_proteins, desc="   处理蛋白质"):
        try:
            protein_to_graph[seq] = sequence_to_graph(seq)
        except Exception as e:
            print(f"\n   警告: 蛋白质 {seq[:50]}... 处理失败: {e}")
            continue

    print(f"   成功处理: {len(protein_to_graph)} 个蛋白质")

    # 生成updated_full.csv
    print("\n4. 生成updated_full.csv...")
    df_output = pd.DataFrame()
    df_output['ligand'] = df['ligand_smiles']
    df_output['protein'] = df['protein_sequence']
    df_output['series'] = df['protein_sequence'].apply(sequence_to_encoding)
    df_output['label'] = df['affinity']

    df_output.to_csv(output_path / "updated_full.csv", index=False)
    print(f"   保存到: {output_path / 'updated_full.csv'}")

    # 生成ligands.txt和proteins.txt
    print("\n5. 生成ligands.txt和proteins.txt...")
    ligands_dict = {smiles: smiles for smiles in unique_ligands}
    proteins_dict = {seq: seq for seq in unique_proteins}

    with open(output_path / "ligands.txt", 'w') as f:
        json.dump(ligands_dict, f)

    with open(output_path / "proteins.txt", 'w') as f:
        json.dump(proteins_dict, f)

    # 保存pkl文件
    print("\n6. 保存pkl文件...")
    with open(output_path / "ligand_to_graph.pkl", 'wb') as f:
        pickle.dump(ligand_to_graph, f)
    print(f"   保存: ligand_to_graph.pkl ({len(ligand_to_graph)} 个)")

    with open(output_path / "ligand_to_ecfp.pkl", 'wb') as f:
        pickle.dump(ligand_to_ecfp, f)
    print(f"   保存: ligand_to_ecfp.pkl ({len(ligand_to_ecfp)} 个)")

    with open(output_path / "protein_to_graph.pkl", 'wb') as f:
        pickle.dump(protein_to_graph, f)
    print(f"   保存: protein_to_graph.pkl ({len(protein_to_graph)} 个)")

    # 生成fold文件 (默认80/20划分)
    print("\n7. 生成fold文件...")
    n_samples = len(df_output)
    train_size = int(0.8 * n_samples)

    train_indices = list(range(train_size))
    test_indices = list(range(train_size, n_samples))

    with open(output_path / "folds" / "train_fold_setting1.txt", 'w') as f:
        json.dump([train_indices], f)

    with open(output_path / "folds" / "test_fold_setting1.txt", 'w') as f:
        json.dump(test_indices, f)

    print(f"   训练集: {len(train_indices)} 个样本")
    print(f"   测试集: {len(test_indices)} 个样本")

    print("\n✅ 预处理完成!")
    print(f"\n输出文件:")
    print(f"  - {output_path / 'updated_full.csv'}")
    print(f"  - {output_path / 'ligands.txt'}")
    print(f"  - {output_path / 'proteins.txt'}")
    print(f"  - {output_path / 'ligand_to_graph.pkl'}")
    print(f"  - {output_path / 'ligand_to_ecfp.pkl'}")
    print(f"  - {output_path / 'protein_to_graph.pkl'}")
    print(f"  - {output_path / 'folds' / 'train_fold_setting1.txt'}")
    print(f"  - {output_path / 'folds' / 'test_fold_setting1.txt'}")

    print(f"\n运行训练:")
    print(f"  python test.py --datasets {dataset_name} --folds 0 --gpu 0")


def main():
    parser = argparse.ArgumentParser(description='MEGDTA数据预处理脚本')
    parser.add_argument('--input', type=str, required=True,
                       help='输入CSV文件路径 (必须包含: ligand_smiles, protein_sequence, affinity)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录 (例如: data/mydataset/)')
    parser.add_argument('--dataset_name', type=str, default='mydataset',
                       help='数据集名称 (默认: mydataset)')

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
