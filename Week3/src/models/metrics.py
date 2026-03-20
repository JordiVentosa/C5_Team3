import evaluate
from typing import List, Dict


class Metric:
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu1 = self.bleu.compute(predictions=predictions, references=references, max_order=1)
        bleu2 = self.bleu.compute(predictions=predictions, references=references, max_order=2)
        rouge_scores = self.rouge.compute(predictions=predictions, references=references)
        meteor_score = self.meteor.compute(predictions=predictions, references=references)
        
        return {
            'bleu1': bleu1['bleu'],
            'bleu2': bleu2['bleu'],
            'rougeL': rouge_scores['rougeL'],
            'meteor': meteor_score['meteor']
        }
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        return (f"BLEU-1:{metrics['bleu1']*100:.1f}%, "
                f"BLEU-2:{metrics['bleu2']*100:.1f}%, "
                f"ROUGE-L:{metrics['rougeL']*100:.1f}%, "
                f"METEOR:{metrics['meteor']*100:.1f}%")
