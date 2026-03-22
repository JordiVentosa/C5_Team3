from .base import BaseTokenizer
from .character import CharacterTokenizer
from .word import WordTokenizer
from .subword import SubWordTokenizer


def get_tokenizer(tokenizer_type: str = 'character', **kwargs):
    """
    Factory function to get the appropriate tokenizer.

    Args:
        tokenizer_type: Type of tokenizer ('character', 'word', 'subword')
        **kwargs: Additional arguments to pass to the tokenizer constructor

    Returns:
        An instance of the requested tokenizer
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == 'character':
        return CharacterTokenizer(**kwargs)
    elif tokenizer_type == 'word':
        return WordTokenizer(**kwargs)
    elif tokenizer_type == 'subword':
        return SubWordTokenizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Choose from: 'character', 'word', 'subword'"
        )


__all__ = [
    'BaseTokenizer',
    'CharacterTokenizer',
    'WordTokenizer',
    'SubWordTokenizer',
    'get_tokenizer'
]
