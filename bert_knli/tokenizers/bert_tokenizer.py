from transformers import BertTokenizer


def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
