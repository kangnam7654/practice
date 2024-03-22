from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def main():
    vocab_size = 50000  # 어휘 크기 명확히 설정
    files = ["/home/kangnam/project/practice/bart_translator/words.txt"]  # 훈련 데이터 파일

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens=["[SOS]", "[EOS]", "[PAD]", "[SEP]", "[MASK]", "[UNK]"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    
    tokenizer.train(files=files, trainer=trainer)
    
    tokenizer.save("./ko_en_tokenizer.json")

if __name__ == "__main__":
    main()
