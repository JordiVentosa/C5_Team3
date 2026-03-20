import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .baseline import Baseline, char2idx, idx2char, TEXT_MAX_LEN
from .metrics import Metric


class CaptioningModule:
    def __init__(self, model: Baseline, learning_rate: float = 1e-4, device: str = 'cpu', teacher_forcing_ratio: float = 0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.metric = Metric()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        images, captions = batch
        images = images.to(self.device)
        captions = captions.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(images, target=captions, teacher_forcing_ratio=self.teacher_forcing_ratio)
        loss = self.criterion(logits, captions)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        images, captions = batch
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images)
            loss = self.criterion(logits, captions)
        
        return {'loss': loss.item()}
    
    def predict(self, images: torch.Tensor) -> list:
        self.model.eval()
        images = images.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images)
            predictions = torch.argmax(logits, dim=1)
        
        captions = []
        for pred in predictions:
            caption = self._decode_caption(pred)
            captions.append(caption)
        
        return captions
    
    def _decode_caption(self, encoded: torch.Tensor) -> str:
        chars = []
        for idx in encoded:
            idx = idx.item()
            if idx == char2idx['<EOS>']:
                break
            if idx != char2idx['<SOS>'] and idx != char2idx['<PAD>']:
                chars.append(idx2char[idx])
        return ''.join(chars)
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return self.metric.compute(predictions, references)
    
    def configure_optimizer(self, optimizer: Optional[torch.optim.Optimizer] = None):
        if optimizer is not None:
            self.optimizer = optimizer
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])