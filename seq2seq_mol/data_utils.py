"""数据处理工具 - 处理分子数据的加载、转换和增强.

提供SMILES到SELFIES的转换、分子描述符计算、数据集构建等功能。
"""

import os
from typing import Dict, Iterable, List, Optional

import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import selfies
except ImportError as error:
    selfies = None
    SELFIES_IMPORT_ERROR = error
else:
    SELFIES_IMPORT_ERROR = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError as error:
    Chem = None
    Descriptors = None
    RDKIT_IMPORT_ERROR = error
else:
    RDKIT_IMPORT_ERROR = None


SELFIES_COLUMN = "selfies"
IUPAC_COLUMN = "iupac"
SMILES_COLUMN = "smiles"


def _require_selfies() -> None:
    """检查selfies库是否可用，不可用则抛出异常."""
    if SELFIES_IMPORT_ERROR is not None:
        raise RuntimeError("selfies库是数据集构建所必需的.") from SELFIES_IMPORT_ERROR


def _require_rdkit() -> None:
    """检查RDKit库是否可用，不可用则抛出异常."""
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError("RDKit是数据集构建和描述符计算所必需的.") from RDKIT_IMPORT_ERROR


def smiles_to_selfies(smiles: str) -> Optional[str]:
    """将SMILES字符串转换为SELFIES表示.
    
    SELFIES是一种基于语义的分子表示，保证生成的序列都是有效的分子。
    
    Args:
        smiles: SMILES格式的分子字符串
    
    Returns:
        Optional[str]: SELFIES格式的分子字符串，转换失败返回None
    """
    _require_selfies()
    try:
        selfies_seq = selfies.encoder(smiles)
        return selfies_seq
    except Exception:
        return None


def get_mol_from_smiles(smiles: str):
    """从SMILES字符串创建RDKit分子对象.
    
    Args:
        smiles: SMILES格式的分子字符串
    
    Returns:
        Chem.Mol: RDKit分子对象，解析失败返回None
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """计算分子的RDKit描述符.
    
    计算6种常用的分子描述符：分子量、LogP、TPSA、氢键供体数、氢键受体数、可旋转键数。
    
    Args:
        smiles: SMILES格式的分子字符串
    
    Returns:
        Optional[Dict[str, float]]: 描述符字典，计算失败返回None
    """
    _require_rdkit()
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),           # 分子量
        "LogP": Descriptors.MolLogP(mol),          # 脂水分配系数
        "TPSA": Descriptors.TPSA(mol),             # 拓扑极性表面积
        "NumHDonors": Descriptors.NumHDonors(mol), # 氢键供体数
        "NumHAcceptors": Descriptors.NumHAcceptors(mol), # 氢键受体数
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol), # 可旋转键数
    }


def load_smiles_source(
    input_path: str,
    smiles_column: str = SMILES_COLUMN,
    n_samples: Optional[int] = None,
) -> List[str]:
    """从文件加载SMILES列表.
    
    支持CSV/TSV文件和纯文本文件（每行一个SMILES）。
    
    Args:
        input_path: 输入文件路径
        smiles_column: CSV/TSV中的SMILES列名
        n_samples: 最大加载样本数，None表示全部加载
    
    Returns:
        List[str]: SMILES字符串列表
    """
    ext = os.path.splitext(input_path)[1].lower()
    smiles: List[str] = []

    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        frame = pd.read_csv(input_path, sep=sep, usecols=[smiles_column])
        smiles = frame[smiles_column].dropna().astype(str).tolist()
    else:
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip().split()
                if not row:
                    continue
                smiles.append(row[0])
    
    if n_samples is not None:
        smiles = smiles[: n_samples]
    
    return smiles


def load_translation_source(
    input_path: str,
    smiles_column: str = SMILES_COLUMN,
    iupac_column: str = IUPAC_COLUMN,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """加载配对的SMILES/IUPAC数据.
    
    RDKit不提供SMILES到IUPAC命名的功能，因此需要预先准备包含IUPAC名称的配对数据。
    
    Args:
        input_path: 输入文件路径（CSV/TSV）
        smiles_column: SMILES列名
        iupac_column: IUPAC列名
        n_samples: 最大加载样本数
    
    Returns:
        pd.DataFrame: 包含smiles和iupac列的DataFrame
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".tsv"}:
        raise ValueError("IUPAC翻译数据必须是CSV/TSV格式，包含SMILES和IUPAC列.")
    
    sep = "," if ext == ".csv" else "\t"
    frame = pd.read_csv(input_path, sep=sep)
    
    # 检查必需列是否存在
    missing = {smiles_column, iupac_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"输入文件缺少必需列: {', '.join(sorted(missing))}")
    
    # 重命名列并清理数据
    frame = frame[[smiles_column, iupac_column]].rename(
        columns={smiles_column: SMILES_COLUMN, iupac_column: IUPAC_COLUMN}
    )
    frame = frame.dropna().astype(str)
    
    if n_samples is not None:
        frame = frame.head(n_samples)
    
    return frame.reset_index(drop=True)


def build_translation_dataset(
    molecules: Iterable,
    save_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """构建翻译数据集 - 将SMILES/IUPAC配对转换为包含SELFIES的数据集.
    
    Args:
        molecules: 分子数据迭代器，可以是dict、tuple或list
        save_path: 可选的保存路径
        max_samples: 最大样本数
        progress: 是否显示进度条
    
    Returns:
        pd.DataFrame: 包含smiles、iupac、selfies三列的数据集
    """
    records: List[Dict[str, str]] = []
    iterator = list(molecules)
    
    if max_samples is not None:
        iterator = iterator[:max_samples]

    for item in tqdm(iterator, desc="转换分子", disable=not progress):
        # 支持多种输入格式
        if isinstance(item, dict):
            smiles, iupac = item.get(SMILES_COLUMN), item.get(IUPAC_COLUMN)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            smiles, iupac = item[0], item[1]
        else:
            raise ValueError("每个分子必须同时提供SMILES和IUPAC名称.")
        
        # 跳过无效数据
        if not isinstance(smiles, str) or not isinstance(iupac, str):
            continue
        
        smiles, iupac = smiles.strip(), iupac.strip()
        # 转换为SELFIES
        selfies_seq = smiles_to_selfies(smiles)
        
        # 跳过转换失败的数据
        if not smiles or not iupac or selfies_seq is None:
            continue
        
        records.append({SMILES_COLUMN: smiles, IUPAC_COLUMN: iupac, SELFIES_COLUMN: selfies_seq})

    frame = pd.DataFrame(records)
    
    # 保存数据集
    if save_path is not None:
        frame.to_csv(save_path, index=False)
    
    return frame


def enrich_with_descriptors(frame: pd.DataFrame, smiles_column: str = SMILES_COLUMN) -> pd.DataFrame:
    """为数据集添加分子描述符.
    
    对数据集中的每个分子计算RDKit描述符，并添加到DataFrame中。
    
    Args:
        frame: 包含SMILES列的DataFrame
        smiles_column: SMILES列名
    
    Returns:
        pd.DataFrame: 添加了描述符列的DataFrame
    """
    records: List[Dict[str, float]] = []
    for smiles in tqdm(frame[smiles_column].tolist(), desc="计算描述符"):
        desc = compute_descriptors(smiles)
        if desc is None:
            # 计算失败时填充NaN
            desc = {
                "MolWt": float("nan"), "LogP": float("nan"), "TPSA": float("nan"),
                "NumHDonors": float("nan"), "NumHAcceptors": float("nan"), 
                "NumRotatableBonds": float("nan")
            }
        records.append(desc)
    
    desc_frame = pd.DataFrame(records)
    # 拼接原始数据和描述符
    result = pd.concat([frame.reset_index(drop=True), desc_frame.reset_index(drop=True)], axis=1)
    return result


def save_dataframe(frame: pd.DataFrame, path: str) -> None:
    """保存DataFrame到CSV文件.
    
    Args:
        frame: 要保存的DataFrame
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    """从CSV文件加载DataFrame.
    
    Args:
        path: 文件路径
    
    Returns:
        pd.DataFrame: 加载的DataFrame
    """
    return pd.read_csv(path)
