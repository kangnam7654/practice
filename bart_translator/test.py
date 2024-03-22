import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"  # 예: "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 번역하고자 하는 문장 예시
input_sentence = "The input sentence to be translated."
input_tokens = tokenizer.encode(input_sentence, return_tensors="pt")

# 모델을 사용하여 번역(또는 생성) 수행
output_tokens = model.generate(input_tokens, max_length=50, num_beams=5)

# 생성된 토큰을 실제 텍스트로 변환
translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(translated_text)
