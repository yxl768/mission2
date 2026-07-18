import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class MolecularDataPreprocessor:
    """分子数据预处理类"""
    def __init__(self, data_path, max_seq_len=128):
        """
        初始化数据预处理器
        
        参数:
            data_path: ZINC 250K数据集路径
            max_seq_len: 最大序列长度
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.src_vocab = None  # IUPAC词汇表
        self.tgt_vocab = None  # SELFIES词汇表
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        
    def load_data(self):
        """加载ZINC 250K数据集"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集文件 {self.data_path} 不存在")
        
        # 读取CSV文件
        df = pd.read_csv(self.data_path)
        print(f"成功加载数据集，共 {len(df)} 个分子")
        
        # 提取IUPAC名称和分子性质
        self.data = df[['smiles', 'logP', 'qed', 'SAS']]
        self.data.columns = ['smiles', 'logP', 'qed', 'synthetic_accessibility']
        
        return self.data
    
    def smiles_to_selfies(self, smiles):
        """将SMILES转换为SELFIES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return sf.encoder(smiles)
        except:
            return None
    
    def preprocess(self):
        """数据预处理主函数"""
        # 加载数据
        data = self.load_data()
        
        # 过滤无效分子
        data = data.dropna(subset=['smiles'])
        
        # 转换为SELFIES
        print("开始将SMILES转换为SELFIES...")
        data['selfies'] = data['smiles'].apply(self.smiles_to_selfies)
        data = data.dropna(subset=['selfies'])
        print(f"转换完成，有效分子数: {len(data)}")
        
        # 构建词汇表
        self._build_vocab(data['smiles'], data['selfies'])
        
        # 分割数据集
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
        
        print(f"数据集分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size
        }
    
    def _build_vocab(self, smiles_list, selfies_list):
        """构建IUPAC和SELFIES词汇表"""
        # 构建IUPAC词汇表
        src_chars = set()
        for smiles in smiles_list:
            src_chars.update(list(smiles))
        self.src_vocab = {char: idx+2 for idx, char in enumerate(sorted(src_chars))}
        self.src_vocab['<pad>'] = 0
        self.src_vocab['<unk>'] = 1
        self.src_vocab_size = len(self.src_vocab)
        
        # 构建SELFIES词汇表
        tgt_chars = set()
        for selfies in selfies_list:
            tgt_chars.update(sf.split_selfies(selfies))
        self.tgt_vocab = {char: idx+2 for idx, char in enumerate(sorted(tgt_chars))}
        self.tgt_vocab['<pad>'] = 0
        self.tgt_vocab['<sos>'] = 1  # 序列开始标记
        self.tgt_vocab['<eos>'] = 2  # 序列结束标记
        self.tgt_vocab['<unk>'] = 3
        self.tgt_vocab_size = len(self.tgt_vocab)
        
        print(f"IUPAC词汇表大小: {self.src_vocab_size}")
        print(f"SELFIES词汇表大小: {self.tgt_vocab_size}")
    
    def encode_sequence(self, sequence, vocab, is_selfies=False):
        """将序列编码为索引"""
        if is_selfies:
            tokens = sf.split_selfies(sequence)
        else:
            tokens = list(sequence)
        
        encoded = [vocab.get(token, vocab['<unk>']) for token in tokens]
        
        # 添加开始和结束标记 (仅对目标序列)
        if is_selfies:
            encoded = [vocab['<sos>']] + encoded + [vocab['<eos>']]
        
        # 填充序列
        if len(encoded) < self.max_seq_len:
            encoded += [vocab['<pad>']] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[:self.max_seq_len]
        
        return encoded

class MolecularDataset(Dataset):
    """分子数据集类"""
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_seq = self.preprocessor.encode_sequence(row['smiles'], self.preprocessor.src_vocab)
        tgt_seq = self.preprocessor.encode_sequence(row['selfies'], self.preprocessor.tgt_vocab, is_selfies=True)
        
        properties = torch.tensor([
            row['logP'],
            row['qed'],
            row['synthetic_accessibility']
        ], dtype=torch.float)
        
        return {
            'src_seq': torch.tensor(src_seq, dtype=torch.long),
            'tgt_seq': torch.tensor(tgt_seq, dtype=torch.long),
            'properties': properties
        }

# 示例用法
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = MolecularDataPreprocessor('data/zinc250k.csv')
    
    # 预处理数据
    processed_data = preprocessor.preprocess()
    
    # 创建数据集
    train_dataset = MolecularDataset(processed_data['train'], preprocessor)
    val_dataset = MolecularDataset(processed_data['val'], preprocessor)
    test_dataset = MolecularDataset(processed_data['test'], preprocessor)
    
    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print(f"源序列形状: {batch['src_seq'].shape}")
    print(f"目标序列形状: {batch['tgt_seq'].shape}")
    print(f"性质形状: {batch['properties'].shape}")