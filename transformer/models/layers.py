import numpy as np
import torch
import torch.nn as nn


def positional_encoding(max_length, d_model):
    """포지셔널 인코딩 생성"""
    pos_enc = np.zeros((max_length, d_model))
    for pos in range(max_length):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))

    return torch.tensor(pos_enc, dtype=torch.float32)


def scaled_dot_product_attention(q, k, v, mask=None):
    qk = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size(-1)
    scaled = qk / torch.sqrt(torch.tensor(d_k))

    if mask is not None:
        scaled += mask * -1e-9
    attention_score = torch.softmax(scaled, -1)
    out = torch.matmul(attention_score, v)
    return out, attention_score


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = int(self.d_model / self.num_heads)
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)

        self.dense = nn.Linear(self.d_model, self.d_model)

    def split_head(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        attention_output, attention_score = scaled_dot_product_attention(q, k, v, mask)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        concatenated = attention_output.view(batch_size, -1, self.d_model)
        output = self.dense(concatenated)
        return output, attention_score


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output, _ = self.multi_head_attention(q=x, k=x, v=x, mask=mask)
        out1 = self.layer_norm1(self.dropout1(attention_output) + x)

        ffn_output = self.ffn(out1)

        out2 = self.layer_norm2(self.dropout2(ffn_output) + out1)
        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        max_length,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length=max_length, d_model=d_model)
        self.enc_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.enc_layers.append(
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_length = x.size(1)

        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))
        x += self.pos_encoding[:seq_length, :]

        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, drop_out=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.masked_attention1 = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.enc_dec_attention2 = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention1, _ = self.masked_attention1(x, x, x, tgt_mask)
        x = self.layer_norm1(self.dropout1(attention1) + x)
        attention2, _ = self.enc_dec_attention2(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(self.dropout2(attention2) + x)
        ffn_out = self.ffn(x)
        x = self.layer_norm3(self.dropout3(ffn_out) + x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        max_length,
        d_ff,
        num_heads,
        input_vocab_size,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_enc = positional_encoding(
            max_length=max_length, d_model=d_model
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(
                DecoderLayer(num_heads=num_heads, d_model=d_model, d_ff=d_ff)
            )

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        seq_len = x.size(1)

        x = self.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        x += self.positional_enc[:seq_len]

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


if __name__ == "__main__":
    model = Decoder(
        num_layers=2,
        num_heads=8,
        d_model=512,
        d_ff=512,
        input_vocab_size=100000,
        max_length=1024,
    )
    x = torch.ones(1, 100).int()
    enc = torch.randn(1, 512)
    y = model(x, enc)
    pass
