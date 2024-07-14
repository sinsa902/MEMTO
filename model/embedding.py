import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty

# %%
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import lib as lb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        pad = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=pad,
            padding_mode="circular",
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = self.bias
            x = x + bias[None]
        return x


class InputEmbedding(nn.Module):
    def __init__(self, d_numerical, categories, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.tokenizer = Tokenizer(d_numerical, categories, d_model, True)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_num, x_cat):
        concat_x = self.tokenizer(x_num, x_cat)
        position_embed = self.pos_embedding(concat_x).to(self.device)
        return self.dropout(concat_x + position_embed)
