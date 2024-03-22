import torch
from torch import nn as nn
from transformers import BertModel, BertConfig
from transformers import BertTokenizerFast


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.load_bert()
        self.classifier = nn.Linear(768, 3)

    def load_bert(self, pretrained=True):
        bert_config = BertConfig()
        if pretrained:
            bert_model = BertModel.from_pretrained(
                "klue/bert-base"
            )
        else:
            bert_model = BertModel(config=bert_config)
        return bert_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def main():
    model = BertClassifier()
    test_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    tokened = test_tokenizer("I am a cat.", return_tensors="pt")["input_ids"]
    output = model(tokened)


if __name__ == "__main__":
    main()
