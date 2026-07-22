"""Tokenizer工具 - 将分子序列转换为模型可处理的token ID.

提供IUPAC名称和SELFIES序列的tokenization功能，支持字符级和符号级分词。
"""

import re
from typing import Callable, Dict, Iterable, List, Optional

import selfies


class Tokenizer:
    """通用Tokenizer类 - 支持编码和解码操作.
    
    将序列转换为token ID列表，或将token ID列表转换回序列。
    支持特殊token（padding、开始符、结束符、未知token）。
    """

    def __init__(
        self,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pre_tokenize: Optional[Callable[[str], List[str]]] = None,
    ):
        """初始化Tokenizer.
        
        Args:
            pad_token: padding token，用于填充变长序列
            bos_token: 开始符token，标记序列开始
            eos_token: 结束符token，标记序列结束
            unk_token: 未知token，处理不在词汇表中的字符
            pre_tokenize: 预分词函数，自定义token切分方式
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pre_tokenize = pre_tokenize

        # 词汇表映射
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab: List[str] = []

    def _default_pre_tokenize(self, sequence: str) -> List[str]:
        """默认预分词函数 - 字符级切分.
        
        将序列按字符逐个切分。
        
        Args:
            sequence: 输入序列
        
        Returns:
            List[str]: 字符列表
        """
        return list(sequence)

    def _safe_tokenize(self, sequence: str) -> List[str]:
        """安全分词 - 根据配置选择预分词方式.
        
        Args:
            sequence: 输入序列
        
        Returns:
            List[str]: token列表
        """
        if self.pre_tokenize is not None:
            return list(self.pre_tokenize(sequence))
        return self._default_pre_tokenize(sequence)

    def build_vocab(self, sequences: Iterable[str]) -> None:
        """构建词汇表 - 从训练数据中学习词汇表.
        
        Args:
            sequences: 序列迭代器，用于构建词汇表
        """
        tokens = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        for sequence in sequences:
            tokens.update(self._safe_tokenize(sequence))
        
        # 排序词汇表，保证一致性
        self.vocab = sorted(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, sequence: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """编码 - 将序列转换为token ID列表.
        
        Args:
            sequence: 输入序列
            add_bos: 是否添加开始符
            add_eos: 是否添加结束符
        
        Returns:
            List[int]: token ID列表
        """
        token_list = self._safe_tokenize(sequence)
        
        # 添加特殊token
        if add_bos:
            token_list = [self.bos_token] + token_list
        if add_eos:
            token_list = token_list + [self.eos_token]
        
        # 转换为ID，未知token使用unk_token的ID
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in token_list]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """解码 - 将token ID列表转换回序列.
        
        Args:
            ids: token ID列表
            skip_special_tokens: 是否跳过特殊token（pad、bos、eos）
        
        Returns:
            str: 解码后的序列
        """
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        
        # 跳过特殊token
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in {self.pad_token, self.bos_token, self.eos_token}]
        
        # 拼接token得到序列
        return "".join(tokens)

    @property
    def pad_id(self) -> int:
        """padding token的ID."""
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        """开始符token的ID."""
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        """结束符token的ID."""
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        """未知token的ID."""
        return self.token_to_id[self.unk_token]


def build_iupac_tokenizer(iupac_strings: Iterable[str]) -> Tokenizer:
    """构建IUPAC名称的Tokenizer.
    
    IUPAC名称使用字符级分词，每个字符作为一个token。
    
    Args:
        iupac_strings: IUPAC名称迭代器
    
    Returns:
        Tokenizer: 配置好的Tokenizer实例
    """
    tokenizer = Tokenizer(pre_tokenize=lambda s: list(s))
    tokenizer.build_vocab(iupac_strings)
    return tokenizer


def build_selfies_tokenizer(selfies_strings: Iterable[str]) -> Tokenizer:
    """构建SELFIES序列的Tokenizer.
    
    SELFIES使用符号级分词，使用selfies库的split_selfies函数切分。
    
    Args:
        selfies_strings: SELFIES序列迭代器
    
    Returns:
        Tokenizer: 配置好的Tokenizer实例
    """
    tokenizer = Tokenizer(pre_tokenize=lambda s: selfies.split_selfies(s))
    tokenizer.build_vocab(selfies_strings)
    return tokenizer
