import torch
import torch.nn as nn
from transformers import BartModel, BartForConditionalGeneration, BartConfig


class RandomEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self._build_model()

    def forward(self, input_ids):
        out = self.model(input_ids)
        return out.last_hidden_state

    def _build_model(self):
        bart = BartModel.from_pretrained("facebook/bart-base")
        model = bart.get_encoder()
        return model


class ProjectionLayer(nn.Module):
    def __init__(self, h_dim=768, len_vocab=50000):
        super().__init__()
        self.proj1 = nn.Linear(h_dim, h_dim * 4)
        self.proj2 = nn.Linear(h_dim * 4, len_vocab)

    def forwrd(self, h):
        h = self.proj1(h)
        h = self.proj2(h)
        return h


class BartTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder, self.proj_layer = self._build_model()
        self.embedding = self.decoder.get_input_embeddings()

    def forward(self, encoder_ids, decoder_ids, encoder_mask=None):
        h = self.encoder(encoder_ids, attention_mask=encoder_mask)
        output = self.decoder(decoder_ids, encoder_hidden_states=h.last_hidden_state)

        self.proj_layer
        return output.last_hidden_state

    def _build_model(self):
        config = BartConfig(
            vocab_size=50000,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            decoder_start_token_id=1,
            forced_eos_token_id=1,
        )
        model = BartForConditionalGeneration(config)
        encoder = model.get_encoder()
        decoder = model.get_decoder()
        # encoder.embed_tokens = RandomEncoder()
        proj_layer = ProjectionLayer()
        return encoder, decoder, proj_layer

    # def generate(self, input_ids, length=50):
    #     encoder_outputs = self.encoder(input_ids)
    #     decoder_input_ids = torch.tensor([0])

    #     for _ in range(length):
    #         # 디코더를 통해 다음 토큰의 숨겨진 상태 예측
    #         decoder_outputs = self.decoder(
    #             decoder_input_ids,
    #             encoder_hidden_states=encoder_outputs.last_hidden_state,
    #         )
    #         hidden_states = decoder_outputs.last_hidden_state

    #         # 숨겨진 상태를 로짓으로 변환
    #         logits = self.proj_layer(
    #             hidden_states[:, -1, :]
    #         )  # 마지막 타임스텝에 대해서만 처리

    #         # 다음 토큰의 ID 예측
    #         next_token_id = logits.argmax(-1).unsqueeze(-1)

    #         # 생성된 토큰이 <eos> 토큰이면 종료
    #         if torch.eq(next_token_id, 1).all():
    #             break

    #         # 디코더 입력 업데이트
    #         decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

    #     return decoder_input_ids


if __name__ == "__main__":
    x = torch.ones(1, 512).to(torch.long)
    x2 = x
    model = BartTranslator()

    output = model(x, x2)
    pass
