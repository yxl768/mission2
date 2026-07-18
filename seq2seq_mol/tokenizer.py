import re
from typing import Callable, Dict, Iterable, List, Optional

import selfies


class Tokenizer:
    def __init__(
        self,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pre_tokenize: Optional[Callable[[str], List[str]]] = None,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pre_tokenize = pre_tokenize

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab: List[str] = []

    def _default_pre_tokenize(self, sequence: str) -> List[str]:
        return list(sequence)

    def _safe_tokenize(self, sequence: str) -> List[str]:
        if self.pre_tokenize is not None:
            return list(self.pre_tokenize(sequence))
        return self._default_pre_tokenize(sequence)

    def build_vocab(self, sequences: Iterable[str]) -> None:
        tokens = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        for sequence in sequences:
            tokens.update(self._safe_tokenize(sequence))
        self.vocab = sorted(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, sequence: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        token_list = self._safe_tokenize(sequence)
        if add_bos:
            token_list = [self.bos_token] + token_list
        if add_eos:
            token_list = token_list + [self.eos_token]
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in token_list]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in {self.pad_token, self.bos_token, self.eos_token}]
        return "".join(tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]


def build_iupac_tokenizer(iupac_strings: Iterable[str]) -> Tokenizer:
    tokenizer = Tokenizer(pre_tokenize=lambda s: list(s))
    tokenizer.build_vocab(iupac_strings)
    return tokenizer


def build_selfies_tokenizer(selfies_strings: Iterable[str]) -> Tokenizer:
    tokenizer = Tokenizer(pre_tokenize=lambda s: selfies.split_selfies(s))
    tokenizer.build_vocab(selfies_strings)
    return tokenizer
