import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BartDataset(Dataset):
    def __init__(self, file_path, tokenizer, leash=True):
        super().__init__()
        self.data = (
            self.load_json(file_path)["data"] if leash else self.load_json(file_path)
        )
        self.tokenizer = tokenizer

    def __len__(self):
        # return len(self.data)
        return 10000

    def __getitem__(self, idx):
        data = self.data[idx]
        # | 한글, 영어 분리 |
        ko = data["ko"]
        en = data["en"]
        label = en
        
        ko_out = self.tokenizer.encode_plus(
            ko,
            max_length=512,
            add_special_tokens=True,
            padding="max_length",
            truncation=False,
            return_tensors = "pt"
        )
        label_out = self.tokenizer.encode_plus(
            label,
            max_length=512,
            add_special_tokens=False,
            padding="max_length",
            truncation=False,
            return_tensors = "pt"
        )["input_ids"].squeeze(0)

        ko_ids = ko_out["input_ids"].squeeze(0)
        ko_att = ko_out["attention_mask"].squeeze(0)
        
        return ko_ids, ko_att, label_out

    def load_json(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data


def main():
    from transformers import BartTokenizer
    word = "<s>I am a cat."
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenized = tokenizer.encode_plus(word, max_length=30, padding="max_length", add_special_tokens=False)
    pass
    

if __name__ == "__main__":
    main()
