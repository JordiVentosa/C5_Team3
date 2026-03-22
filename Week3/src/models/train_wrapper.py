import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .baseline import Baseline
from .metrics import Metric
import lightning as L
from text_tokenizers import BaseTokenizer


class CaptioningModule:
    def __init__(
        self,
        model: Baseline,
        tokenizer: BaseTokenizer,
        learning_rate: float = 1e-4,
        device: str = 'cpu',
        teacher_forcing_ratio: float = 0.0,
        optimizer_type: str = "Adam"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.optimizer_type = optimizer_type
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_special_token_indices()['pad'])
        self.optimizer = self._get_optimizer()
        self.metric = Metric()

    def _get_optimizer(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

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
            caption = self.tokenizer.decode(pred)
            captions.append(caption)

        return captions

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


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        model: Baseline,
        tokenizer: BaseTokenizer,
        learning_rate: float = 1e-4,
        teacher_forcing_ratio: float = 0.0,
        batch_size: int = 128,
        optimizer_type: str = "Adam"
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.optimizer_type = optimizer_type

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_special_token_indices()['pad'])
        self.metric = Metric()

        self.save_hyperparameters(ignore=['model', 'tokenizer', 'criterion', 'metric'])

    # ------------------------------------------------------------------ #
    #  Core steps                                                        #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, captions, caption_text = batch
        logits = self.model(images, target=captions, teacher_forcing_ratio=self.teacher_forcing_ratio)
        loss = self.criterion(logits, captions)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)

        predictions = [self.tokenizer.decode(pred) for pred in torch.argmax(logits, dim=1)]
        self._train_predictions.extend(predictions)
        self._train_references.extend(caption_text)

        return loss

    def on_train_epoch_start(self):
        self._train_predictions = []
        self._train_references = []

    def on_train_epoch_end(self):
        metrics = self.compute_metrics(self._train_predictions, self._train_references)
        for name, value in metrics.items():
            self.log(f'train/{name}', value, prog_bar=True, on_epoch=True)
        self._train_predictions.clear()
        self._train_references.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, captions, caption_text = batch
        logits = self.model(images)
        loss = self.criterion(logits, captions)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)

        predictions = [self.tokenizer.decode(pred) for pred in torch.argmax(logits, dim=1)]
        self._val_predictions.extend(predictions)
        self._val_references.extend(caption_text)

        return loss

    def on_validation_epoch_start(self):
        self._val_predictions = []
        self._val_references = []

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self._val_predictions, self._val_references)
        for name, value in metrics.items():
            self.log(f'val/{name}', value, prog_bar=True, on_epoch=True)
        self._val_predictions.clear()
        self._val_references.clear()

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> list:
        images, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        logits = self.model(images)
        predictions = torch.argmax(logits, dim=1)
        return [self.tokenizer.decode(pred) for pred in predictions]

    # ------------------------------------------------------------------ #
    #  Optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return self.metric.compute(predictions, references)

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
