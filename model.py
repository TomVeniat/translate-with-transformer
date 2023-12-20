import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, tensor, Tensor


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, voc_size: int):
        super().__init__()
        self.d_model = d_model
        self.voc_size = voc_size
        self.embedding = nn.Embedding(voc_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, seq_len: int, dropout: float):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.arange(seq_len, dtype=torch.float).unsqueeze(1).repeat(1, d)

        i = torch.arange(d)
        idx = (i / 2).int().float() * 2
        div_term = idx * (-math.log(10000.0) / d)
        div_term = torch.exp(div_term)

        # print(div_term)
        pe *= div_term
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        pe = pe.unsqueeze(0)  # (1, seq_len, d)

        self.register_buffer("pe", pe)

    def forward(self, x):
        y = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(y)


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.tensor):
        mu = x.mean(-1, keepdim=True)
        sigma = x.std(-1, keepdim=True)
        # todo, different from the base implem
        norm_x = (x - mu) / torch.sqrt(sigma ** 2 + self.eps)
        return self.gamma * norm_x + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: tensor):
        return self.layer2(self.dropout(F.relu(self.layer1(x))))


def attention(q: tensor, k: Tensor, v: Tensor, mask: Optional[Tensor], dropout: Optional[nn.Dropout]):
    # assert q.dim() == 3
    # assert (q.shape == k.shape == v.shape)
    dk = k.shape[2]

    scores = q @ k.permute((0, 2, 1)) / math.sqrt(dk)  # (batch, seq_len, d_att)
    if mask is not None:
        scores.masked_fill_(mask == 0, -torch.inf)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return scores @ v, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_att: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        self.w_q = nn.Linear(d_model, d_att)
        self.w_k = nn.Linear(d_model, d_att)
        self.w_v = nn.Linear(d_model, d_att)

        self.w_o = nn.Linear(d_att, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        x_q = self.w_q(q)  # (batch, seq len, d_att)
        x_k = self.w_k(k)
        x_v = self.w_v(v)

        xq_split = x_q.tensor_split(self.h, dim=-1)  # h x (batch, seq_len, d_att//h)
        xk_split = x_k.tensor_split(self.h, dim=-1)
        xv_split = x_v.tensor_split(self.h, dim=-1)

        attention_res = [attention(q, k, v, mask, self.dropout) for q, k, v in zip(xq_split, xk_split, xv_split)]
        heads, attention_scores = list(zip(*attention_res))

        res = torch.cat(heads, -1)
        # print(res.shape)
        return self.w_o(res)


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])

    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, source_mask))
        return self.residual_connections[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoded, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoded, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, voc_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, voc_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, target_embed: InputEmbeddings,
                 src_pos_embed: PositionalEncoding, target_pos_embed: PositionalEncoding, out_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos_embed = src_pos_embed
        self.target_pos_embed = target_pos_embed
        self.out_layer = out_layer

    def encode(self, x, src_mask):
        x = self.src_pos_embed(self.src_embed(x))
        return self.encoder(x, src_mask)

    def decode(self, encoder_out: Tensor, src_mask: Tensor, target: Tensor, target_mask: Tensor):
        target = self.target_pos_embed(self.target_embed(target))
        target = self.decoder(target, encoder_out, src_mask, target_mask)
        return target

    def project(self, x):
        return self.out_layer(x)


def build_transformer(src_voc_size: int, target_voc_size: int, src_seq_len: int, target_seq_len: int,
                      d_model: int = 512,
                      n_encoder_block: int = 6, n_decoder_blocks: int = 6, n_heads: int = 4, d_ff: int = 2048,
                      dropout: float = 0.2) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_voc_size)
    target_embed = InputEmbeddings(d_model, target_voc_size)

    pos_embed = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder = Encoder(
        nn.ModuleList(
            [EncoderBlock(MultiHeadAttention(d_model, d_model, n_heads, dropout),
                          FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(n_encoder_block)]
        ),

    )
    decoder = Decoder(
        nn.ModuleList(
            [DecoderBlock(MultiHeadAttention(d_model, d_model, n_heads, dropout),
                          MultiHeadAttention(d_model, d_model, n_heads, dropout),
                          FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(n_decoder_blocks)]
        )
    )

    proj = ProjectionLayer(d_model, target_voc_size)

    transformer = Transformer(encoder, decoder, src_embed, target_embed, pos_embed, pos_embed, proj)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


if __name__ == '__main__':
    b = 3
    seq_len = 5
    d_model = 15
    t1 = torch.randn(b, seq_len, d_model)  # (batch, seq len, d_model)

    n_heads = 4
    model = MultiHeadAttention(d_model, 511, n_heads, 0)

    r = model(t1)
    print(r.shape)
