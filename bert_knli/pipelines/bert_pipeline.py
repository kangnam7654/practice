from typing import Any
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

class BertPipeline(pl.LightningModule):
    def __init__(self, model, lr=1e-5):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.valid_acc = Accuracy(task="multiclass", num_classes=3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        train_loss = self.loop(batch, step="train")
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.loop(batch, step="valid")
        return valid_loss

    def loop(self, batch, step="train"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, label)
        acc = self.train_acc if step == "train" else self.valid_acc
        acc(logits, label)

        self.log(f"{step}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{step}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer