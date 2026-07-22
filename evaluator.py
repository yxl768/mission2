"""分子表示评估器 - 比较学习到的编码器嵌入与RDKit描述符和Morgan指纹基线.

通过下游性质预测任务评估不同分子表示的质量，使用5-fold交叉验证确保评估的稳健性。
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from seq2seq_mol.data_utils import IUPAC_COLUMN, SMILES_COLUMN, enrich_with_descriptors, load_dataframe


DEFAULT_DESCRIPTORS = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]
MORGAN_RADIUS = 2
MORGAN_BITS = 1024

PROPERTY_ALIASES = {
    "logp": "logP",
    "qed": "qed",
    "sas": "SAS",
}


def resolve_property_name(frame, property_name):
    """解析属性名称 - 支持大小写不敏感匹配和别名映射.
    
    Args:
        frame: 包含属性列的DataFrame
        property_name: 属性名称（可能是别名或不同大小写）
    
    Returns:
        str: DataFrame中实际的列名
    
    Raises:
        ValueError: 属性名称未找到
    """
    columns = {c.lower(): c for c in frame.columns}
    
    # 直接匹配
    if property_name in frame.columns:
        return property_name
    
    # 大小写不敏感匹配
    if property_name.lower() in columns:
        return columns[property_name.lower()]
    
    # 别名匹配
    alias = PROPERTY_ALIASES.get(property_name.lower())
    if alias and alias in frame.columns:
        return alias
    
    raise ValueError(
        f"属性 '{property_name}' 未在以下列中找到: {list(frame.columns)}"
    )


def deduplicate_by_iupac(frame, embeddings):
    """按IUPAC名称去重 - 防止数据泄漏.
    
    增强数据集中包含同一分子的多个SMILES变体（相同IUPAC名称）。
    随机划分训练/测试集会导致同一分子泄漏到两个集合中，使评估指标虚高。
    由于编码器接收IUPAC名称作为输入，同一分子的所有变体共享相同的嵌入，
    因此我们只保留每个唯一IUPAC名称的第一个出现。
    
    Args:
        frame: 原始DataFrame
        embeddings: 嵌入矩阵，形状 (num_samples, embedding_dim)
    
    Returns:
        tuple: (去重后的DataFrame, 去重后的嵌入矩阵)
    """
    if IUPAC_COLUMN not in frame.columns:
        return frame, embeddings
    
    # 创建去重掩码
    unique_mask = ~frame[IUPAC_COLUMN].astype(str).duplicated(keep="first")
    n_before = len(frame)
    
    # 应用去重
    frame = frame.loc[unique_mask].reset_index(drop=True)
    embeddings = embeddings[unique_mask.to_numpy()]
    
    n_after = len(frame)
    if n_before != n_after:
        print(f"按IUPAC去重: {n_before} -> {n_after} 个唯一分子")
    
    return frame, embeddings


def _require_rdkit():
    """检查RDKit是否可用."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except ImportError as error:
        raise RuntimeError("RDKit是计算Morgan指纹所必需的.") from error


def compute_morgan_fingerprints(smiles_list, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """计算Morgan指纹（ECFP4）.
    
    Morgan指纹是一种基于圆形拓扑的分子指纹，广泛用于QSAR/QSPR研究。
    
    Args:
        smiles_list: SMILES字符串列表
        radius: 指纹半径（默认为2，即ECFP4）
        n_bits: 指纹位数（默认为1024）
    
    Returns:
        np.ndarray: 指纹矩阵，形状 (num_samples, n_bits)
    """
    Chem, AllChem = _require_rdkit()
    fingerprints = np.zeros((len(smiles_list), n_bits), dtype=np.float64)
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # 计算Morgan指纹并转换为位向量
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprints[idx] = np.array(bit_vector)
    
    return fingerprints


def fit_and_score(features, target, train_indices, test_indices, alpha, pca_components=None):
    """训练模型并计算评估指标.
    
    使用StandardScaler + Ridge回归的Pipeline进行性质预测。
    可选使用PCA降维处理高维嵌入。
    
    Args:
        features: 特征矩阵，形状 (num_samples, feature_dim)
        target: 目标向量，形状 (num_samples,)
        train_indices: 训练集索引
        test_indices: 测试集索引
        alpha: Ridge回归的正则化参数
        pca_components: PCA降维后的维度（可选）
    
    Returns:
        tuple: (指标字典, 真实值, 预测值)
    """
    steps = []
    
    # 可选的PCA降维
    if pca_components is not None and pca_components < features.shape[1]:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))
    
    # 标准化 + Ridge回归
    steps.extend([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    
    # 创建Pipeline
    model = Pipeline(steps)
    
    # 训练模型
    model.fit(features[train_indices], target[train_indices])
    
    # 预测
    predicted = model.predict(features[test_indices])
    truth = target[test_indices]
    
    # 计算评估指标
    return {
        "r2": float(r2_score(truth, predicted)),
        "rmse": float(mean_squared_error(truth, predicted) ** 0.5),
        "mae": float(mean_absolute_error(truth, predicted)),
    }, truth, predicted


def kfold_eval(features, target, alpha, n_splits=5, seed=42, pca_components=None):
    """运行k-fold交叉验证.
    
    Args:
        features: 特征矩阵
        target: 目标向量
        alpha: Ridge回归正则化参数
        n_splits: 折数（默认为5）
        seed: 随机种子
        pca_components: PCA降维维度（可选）
    
    Returns:
        tuple: (均值指标, 标准差指标, 各折指标列表)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics_list = []
    
    for train_idx, test_idx in kf.split(features):
        metrics, _, _ = fit_and_score(features, target, train_idx, test_idx, alpha, pca_components)
        metrics_list.append(metrics)
    
    # 计算均值和标准差
    mean_metrics = {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
    }
    std_metrics = {
        key: float(np.std([m[key] for m in metrics_list]))
        for key in metrics_list[0]
    }
    
    return mean_metrics, std_metrics, metrics_list


def plot_projection(features, target, title, path, sample_indices):
    """绘制PCA降维可视化图.
    
    将特征通过PCA降至2维，并按目标属性值着色，展示不同表示空间的聚类效果。
    
    Args:
        features: 特征矩阵
        target: 目标属性值
        title: 图标题
        path: 保存路径
        sample_indices: 采样索引（用于控制可视化样本数量）
    """
    # 标准化并PCA降维到2维
    projected = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(features[sample_indices])
    )
    
    # 绘制散点图
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        projected[:, 0], projected[:, 1], 
        c=target[sample_indices], cmap="viridis", s=10, alpha=0.75
    )
    
    # 添加颜色条和标签
    figure.colorbar(scatter, ax=axis, label="属性值")
    axis.set(title=title, xlabel="PC1", ylabel="PC2")
    
    # 保存图像
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_predictions(results, path):
    """绘制预测值与真实值的散点图.
    
    比较不同表示方法在性质预测任务上的表现。
    
    Args:
        results: 字典，键为表示方法名称，值为 (真实值, 预测值, 指标)
        path: 保存路径
    """
    n_panels = len(results)
    figure, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    
    if n_panels == 1:
        axes = [axes]
    
    for axis, (name, (truth, prediction, metrics)) in zip(axes, results.items()):
        # 计算坐标轴范围
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        
        # 绘制散点图
        axis.scatter(truth, prediction, s=14, alpha=0.7, color="#267e8c")
        
        # 绘制对角线（完美预测线）
        axis.plot([lower, upper], [lower, upper], color="#bf4d35", linewidth=1)
        axis.set(title=f"{name}: R2={metrics['r2']:.3f}", xlabel="真实值", ylabel="预测值")
    
    # 保存图像
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def compare_representations(data_path, embeddings_path, output_dir, property_name="LogP",
                            test_fraction=0.2, seed=42, alpha=1.0, n_folds=5, deduplicate=True, pca_components=None):
    """比较不同分子表示方法的性质预测性能.
    
    评估Seq2Seq编码器嵌入、RDKit描述符和Morgan指纹在下游性质预测任务上的表现。
    
    Args:
        data_path: 训练时使用的处理后CSV文件
        embeddings_path: 编码器嵌入文件（encoder_embeddings.npy）
        output_dir: 输出目录
        property_name: 要预测的属性名称
        test_fraction: 单分割测试集比例
        seed: 随机种子
        alpha: Ridge回归正则化参数
        n_folds: k-fold交叉验证的折数
        deduplicate: 是否按IUPAC名称去重
        pca_components: PCA降维维度（可选）
    """
    # 参数验证
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction必须在0和1之间.")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    frame = load_dataframe(data_path)
    property_name = resolve_property_name(frame, property_name)
    
    # 如果缺少属性列，计算描述符
    if property_name not in frame.columns or frame[property_name].isna().any():
        frame = enrich_with_descriptors(frame)
        property_name = resolve_property_name(frame, property_name)
    
    # 检查必需列
    required = DEFAULT_DESCRIPTORS + [property_name, SMILES_COLUMN]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"无法获取所需的属性/描述符列: {sorted(missing)}")
    
    # 移除缺失值
    frame = frame.dropna(subset=required).reset_index(drop=True)
    
    # 加载嵌入
    embeddings = np.load(embeddings_path)
    
    # 检查嵌入和数据行数是否匹配
    if len(embeddings) != len(frame):
        raise ValueError(
            f"嵌入/数据行数不匹配: {len(embeddings)} 个嵌入对应 {len(frame)} 个可用分子. "
            "请使用训练时提供的处理后数据集."
        )
    
    # 按IUPAC去重（可选）
    if deduplicate:
        frame, embeddings = deduplicate_by_iupac(frame, embeddings)

    # 准备特征
    descriptor_names = [name for name in DEFAULT_DESCRIPTORS if name != property_name]
    descriptors = frame[descriptor_names].to_numpy(dtype=np.float64)
    smiles_list = frame[SMILES_COLUMN].astype(str).tolist()
    morgan_fps = compute_morgan_fingerprints(smiles_list)
    target = frame[property_name].to_numpy(dtype=np.float64)

    # 单分割评估（用于可视化）
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)), test_size=test_fraction, random_state=seed
    )
    
    # 评估三种表示方法
    desc_metrics, desc_truth, desc_prediction = fit_and_score(descriptors, target, train_indices, test_indices, alpha)
    morgan_metrics, morgan_truth, morgan_prediction = fit_and_score(morgan_fps, target, train_indices, test_indices, alpha)
    emb_metrics, emb_truth, emb_prediction = fit_and_score(embeddings, target, train_indices, test_indices, alpha, pca_components)

    # k-fold交叉验证
    print(f"\n=== {n_folds}-折交叉验证 ({property_name}) ===")
    desc_mean, desc_std, _ = kfold_eval(descriptors, target, alpha, n_folds, seed)
    morgan_mean, morgan_std, _ = kfold_eval(morgan_fps, target, alpha, n_folds, seed)
    emb_mean, emb_std, _ = kfold_eval(embeddings, target, alpha, n_folds, seed, pca_components)

    # 构建结果摘要
    summary = pd.DataFrame([
        {"representation": "RDKit描述符", **desc_metrics,
         "kfold_r2_mean": desc_mean["r2"], "kfold_r2_std": desc_std["r2"],
         "kfold_rmse_mean": desc_mean["rmse"], "kfold_rmse_std": desc_std["rmse"]},
        {"representation": "Morgan指纹 (ECFP4)", **morgan_metrics,
         "kfold_r2_mean": morgan_mean["r2"], "kfold_r2_std": morgan_std["r2"],
         "kfold_rmse_mean": morgan_mean["rmse"], "kfold_rmse_std": morgan_std["rmse"]},
        {"representation": "Seq2Seq编码器嵌入", **emb_metrics,
         "kfold_r2_mean": emb_mean["r2"], "kfold_r2_std": emb_std["r2"],
         "kfold_rmse_mean": emb_mean["rmse"], "kfold_rmse_std": emb_std["rmse"]},
    ])
    
    # 保存结果
    summary.to_csv(os.path.join(output_dir, "property_prediction_summary.csv"), index=False)
    with open(os.path.join(output_dir, "property_prediction_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)

    # 打印结果
    print("\n=== 单分割结果 ===")
    print(summary[["representation", "r2", "rmse", "mae"]].to_string(index=False))
    
    print(f"\n=== {n_folds}-折交叉验证 (均值 ± 标准差) ===")
    for _, row in summary.iterrows():
        print(f"{row['representation']}: R2={row['kfold_r2_mean']:.4f}±{row['kfold_r2_std']:.4f}, "
              f"RMSE={row['kfold_rmse_mean']:.4f}±{row['kfold_rmse_std']:.4f}")

    # 可视化
    rng = np.random.default_rng(seed)
    visual_indices = rng.choice(len(frame), size=min(len(frame), 5000), replace=False)
    
    # 绘制PCA可视化
    plot_projection(descriptors, target, "RDKit描述符空间", os.path.join(output_dir, "descriptor_pca.png"), visual_indices)
    plot_projection(morgan_fps, target, "Morgan指纹空间", os.path.join(output_dir, "morgan_pca.png"), visual_indices)
    plot_projection(embeddings, target, "Seq2Seq编码器嵌入空间", os.path.join(output_dir, "encoder_embedding_pca.png"), visual_indices)
    
    # 绘制预测散点图
    plot_predictions({
        "RDKit描述符": (desc_truth, desc_prediction, desc_metrics),
        "Morgan指纹": (morgan_truth, morgan_prediction, morgan_metrics),
        "编码器嵌入": (emb_truth, emb_prediction, emb_metrics),
    }, os.path.join(output_dir, "predicted_vs_observed.png"))


def parse_args():
    """解析命令行参数.
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="评估学习到的分子表示")
    parser.add_argument("--data", required=True, help="Seq2Seq训练器使用的处理后CSV文件")
    parser.add_argument("--embeddings", required=True, help="训练器输出的encoder_embeddings.npy")
    parser.add_argument("--output", default="eval_outputs")
    parser.add_argument("--property", default="LogP", help="要预测的分子属性")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="单分割测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge回归正则化参数")
    parser.add_argument("--n-folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--no-deduplicate", action="store_true",
                        help="禁用按IUPAC名称去重（保留增强数据原样）")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="回归前将嵌入降维到的PCA分量数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_representations(
        args.data, args.embeddings, args.output, args.property,
        args.test_fraction, args.seed, args.ridge_alpha, args.n_folds,
        deduplicate=not args.no_deduplicate,
        pca_components=args.pca_components,
    )
