"""
src/metrics.py
Calculate BLEU-1, BLEU-2, ROUGE-L & METEOR
"""

import os
import nltk
nltk.data.path.insert(0, os.path.expanduser('~/nltk_data'))

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def compute_metrics(predictions: list[str], references: list[list[str]]):
    assert len(predictions) == len(references)

    smooth = SmoothingFunction().method1
    refs_tok  = [[r.split() for r in refs] for refs in references]
    preds_tok = [p.split() for p in predictions]

    bleu1 = corpus_bleu(refs_tok, preds_tok,
                        weights=(1, 0, 0, 0),
                        smoothing_function=smooth)
    bleu2 = corpus_bleu(refs_tok, preds_tok,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = sum(
        max(scorer.score(ref, pred)['rougeL'].fmeasure for ref in refs)
        for pred, refs in zip(predictions, references)
    ) / len(predictions)

    meteor = sum(
        meteor_score([r.split() for r in refs], pred.split())
        for pred, refs in zip(predictions, references)
    ) / len(predictions)

    return {
        'bleu-1':  bleu1,
        'bleu-2':  bleu2,
        'rouge-l': rouge_l,
        'meteor':  meteor,
    }


def print_metrics(metrics: dict, title: str = "Metrics"):
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")
    print(f"{'='*40}\n")