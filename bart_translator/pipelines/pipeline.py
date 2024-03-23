import torch
import torch.nn as nn
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu

import wandb


class BartPipeline(pl.LightningModule):
    def __init__(self, model, lr, tokenizer):
        super().__init__()
        self.model = model
        self.lr = lr
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self._inner_counter = 0

    def training_step(self, batch, batch_idx):
        ko_ids, ko_att, en_ids = batch
        loss = self.loop(ko_ids, ko_att, en_ids)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.global_step % 1000 == 0:
            self.text_logging(ko_ids, en_ids)
        return loss

    def validation_step(self, batch, batch_idx):
        ko_ids, ko_att, en_ids = batch
        loss = self.loop(ko_ids, ko_att, en_ids)
        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self._inner_counter % 1000 == 0:
            self.text_logging(ko_ids, en_ids, step="valid")
        self._inner_counter += 1
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self._inner_counter = 0

    def loop(self, ko_ids, ko_att, en_ids):
        output = self.model(ko_ids, attention_mask=ko_att, labels=en_ids)
        loss = output.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def text_logging(self, ko_ids, en_ids, step="train"):
        gen_ids = self.model.generate(ko_ids, max_length=50, num_beams=4)
        gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        ko_text = self.tokenizer.decode(ko_ids[0], skip_special_tokens=True)
        en_text = self.tokenizer.decode(en_ids[0], skip_special_tokens=True)

        to_log = ko_text + "<br><br>" + en_text + "<br><br>" + gen_text
        self.logger.experiment.log({f"{step} gen": wandb.Html(to_log)}, step=self.global_step)

        # score = sentence_bleu(en_text.split(), gen_text.split())
        # self.log(f"{step} BLEU", score, prog_bar=True, on_epoch=True)

    def compute_blue(self, ko_ids, en_ids):
        gen_ids = self.model.generate(ko_ids, max_length=50, num_beams=4)
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        en_text = self.tokenizer.decode(en_ids, skip_special_tokens=True)

        score = sentence_bleu(en_text, gen_text)

        self.log("BLEU", score, prog_bar=True, on_epoch=True)
