from typing import List
import torch
from .base import BaseTokenizer
from transformers import BertTokenizer


class SubWordTokenizer(BaseTokenizer):
    """SubWord tokenizer using BERT's WordPiece tokenizer"""

    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)

        # Map special tokens to BERT's tokens
        self.sos_token = self.bert_tokenizer.cls_token  # [CLS]
        self.eos_token = self.bert_tokenizer.sep_token  # [SEP]
        self.pad_token = self.bert_tokenizer.pad_token  # [PAD]

        self.vocab_size = len(self.bert_tokenizer)
        self.max_len = None

    def build_vocab(self, texts: List[str]) -> None:
        """
        For BERT tokenizer, vocab is predefined.
        This method calculates the max_len from the training data.
        """
        if not texts:
            return

        # Calculate maximum length needed
        max_tokens = 0
        for text in texts:
            # Tokenize and count tokens (including special tokens)
            tokens = self.bert_tokenizer.tokenize(text)
            # +2 for CLS and SEP tokens
            max_tokens = max(max_tokens, len(tokens) + 2)

        self.max_len = max_tokens

    @property
    def token2idx(self):
        """Map tokens to indices using BERT's vocab"""
        return self.bert_tokenizer.get_vocab()

    @property
    def idx2token(self):
        """Map indices to tokens using BERT's vocab"""
        return {idx: token for token, idx in self.bert_tokenizer.get_vocab().items()}

    def encode(self, text: str) -> torch.Tensor:
        """Encode text using BERT tokenizer with padding"""
        if self.max_len is None:
            raise RuntimeError("max_len not set. Call build_vocab() first.")

        # Use BERT's encode method
        # add_special_tokens=True adds [CLS] and [SEP]
        encoded = self.bert_tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded.squeeze(0)  # Remove batch dimension

    def decode(self, indices: torch.Tensor) -> str:
        """Decode token indices back to text"""
        # Use BERT's decode method, skip_special_tokens=True removes [CLS], [SEP], [PAD]
        text = self.bert_tokenizer.decode(indices, skip_special_tokens=True)
        return text

    def get_special_token_indices(self):
        """Get indices of special tokens"""
        return {
            'sos': self.bert_tokenizer.cls_token_id,
            'eos': self.bert_tokenizer.sep_token_id,
            'pad': self.bert_tokenizer.pad_token_id
        }
