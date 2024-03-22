import torch
import torch.nn as nn
import pytorch_lightning as pl

import wandb


class BartPipeline(pl.LightningModule):
    def __init__(self, model, lr, tokenizer):
        super().__init__()
        self.model = model
        self.lr = lr
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.global_step % 100 == 0:
            self.text_logging(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loop(batch)
        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def loop(self, batch):
        ko_ids, ko_att, label = batch
        output = self.model(ko_ids, attention_mask=ko_att, labels=label)
        loss = output.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def text_logging(self, batch):
        ko, _, en = batch
        gen_ids = self.model.generate(ko, max_length=50, num_beams=4)
        gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        ko_text = self.tokenizer.decode(ko[0], skip_special_tokens=True)
        en_text = self.tokenizer.decode(en[0], skip_special_tokens=True)
        # ko_text, en_text = ko_text[0], en_text[0]

        to_log = ko_text + "<br><br>" + en_text + "<br><br>" + gen_text

        self.logger.experiment.log({"gen": wandb.Html(to_log)}, step=self.global_step)
