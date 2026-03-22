from abc import ABC, abstractmethod
from typing import List, Dict
import torch


class BaseTokenizer(ABC):
    """Base class for all tokenizers"""

    def __init__(self):
        self.vocab = None
        self.vocab_size = None
        self.max_len = None
        self.token2idx = None
        self.idx2token = None

        # Special tokens that all tokenizers should have
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'

    @abstractmethod
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts"""
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Encode a text string into token indices"""
        pass

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> str:
        """Decode token indices back to text"""
        pass

    def get_special_token_indices(self) -> Dict[str, int]:
        """Get indices of special tokens"""
        return {
            'sos': self.token2idx[self.sos_token],
            'eos': self.token2idx[self.eos_token],
            'pad': self.token2idx[self.pad_token]
        }

    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
