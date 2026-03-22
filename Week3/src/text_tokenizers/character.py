from typing import List
import torch
from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer - maintains original behavior"""

    def __init__(self):
        super().__init__()
        # Initialize with predefined character set (same as original)
        self.chars = [
            '<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')',
            ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z'
        ]
        self.vocab = self.chars
        self.token2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2token = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.max_len = 201  # Default, will be updated by build_vocab

    def build_vocab(self, texts: List[str]) -> None:
        """
        For character tokenizer, vocab is predefined.
        This method just calculates the max_len from the training data.
        """
        if not texts:
            return

        # Calculate maximum length needed (+ 2 for SOS/EOS)
        max_chars = max(len(text) for text in texts)
        self.max_len = max_chars + 2  # +2 for <SOS> and <EOS>

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to character indices with padding"""
        encoded = [self.token2idx[self.sos_token]]

        for char in text:
            if char in self.token2idx:
                encoded.append(self.token2idx[char])
            # If char not in vocab, skip it (original behavior)

        encoded.append(self.token2idx[self.eos_token])

        # Pad to max_len
        while len(encoded) < self.max_len:
            encoded.append(self.token2idx[self.pad_token])

        # Truncate if necessary
        encoded = encoded[:self.max_len]

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """Decode character indices back to text"""
        chars = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val == self.token2idx[self.eos_token]:
                break
            if idx_val not in (self.token2idx[self.sos_token], self.token2idx[self.pad_token]):
                chars.append(self.idx2token[idx_val])
        return ''.join(chars)
