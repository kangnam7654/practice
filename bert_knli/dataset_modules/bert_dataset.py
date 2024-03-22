import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset


class BertDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        self.data = load_dataset("klue", "nli")
        self.df = pd.DataFrame(self.data["train" if train else "validation"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        premise = self.df.loc[idx, "premise"]
        hypothesis = self.df.loc[idx, "hypothesis"]

        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = self.df.loc[idx, "label"]
        label = torch.tensor(label)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label,
        }
