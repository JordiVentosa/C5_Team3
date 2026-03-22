from typing import List
import torch
from .base import BaseTokenizer
from collections import Counter


class WordTokenizer(BaseTokenizer):
    """Word-level tokenizer with UNK token for unknown words"""

    def __init__(self, min_freq: int = 1):
        super().__init__()
        self.min_freq = min_freq
        self.unk_token = '<UNK>'
        self.max_len = None

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        if not texts:
            raise ValueError("Cannot build vocabulary from empty text list")

        # Tokenize all texts and count word frequencies
        word_counter = Counter()
        all_lengths = []

        for text in texts:
            words = text.lower().split()
            word_counter.update(words)
            # +2 for SOS and EOS tokens
            all_lengths.append(len(words) + 2)

        # Set max_len to the maximum sequence length in training data
        self.max_len = max(all_lengths)

        # Build vocabulary: special tokens + words that meet min_freq threshold
        vocab = [self.sos_token, self.eos_token, self.pad_token, self.unk_token]

        # Add words that appear at least min_freq times
        for word, freq in word_counter.items():
            if freq >= self.min_freq:
                vocab.append(word)

        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to word indices with padding"""
        if self.vocab is None:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        words = text.lower().split()
        encoded = [self.token2idx[self.sos_token]]

        for word in words:
            if word in self.token2idx:
                encoded.append(self.token2idx[word])
            else:
                # Use UNK token for unknown words
                encoded.append(self.token2idx[self.unk_token])

        encoded.append(self.token2idx[self.eos_token])

        # Pad to max_len
        while len(encoded) < self.max_len:
            encoded.append(self.token2idx[self.pad_token])

        # Truncate if necessary
        encoded = encoded[:self.max_len]

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """Decode word indices back to text"""
        words = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val == self.token2idx[self.eos_token]:
                break
            if idx_val not in (self.token2idx[self.sos_token], self.token2idx[self.pad_token]):
                words.append(self.idx2token[idx_val])
        return ' '.join(words)
