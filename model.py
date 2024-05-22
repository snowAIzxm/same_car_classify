import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.out = nn.Linear(embed_dim, input_dim)

    def forward(self, x, y):
        # x: (batch_size, seq_len_x, embed_dim)
        # y: (batch_size, seq_len_y, embed_dim)

        batch_size = x.size(0)

        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len_x, embed_dim)
        K = self.key(y)  # (batch_size, seq_len_y, embed_dim)
        V = self.value(y)  # (batch_size, seq_len_y, embed_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(attn_output)
        return output


class SameCarModel(nn.Module):
    def __init__(self, input_size=768,
                 attention_layer: int = 3,
                 embed_dim: int = 32 * 8,
                 num_heads: int = 8,
                 out_class: int = 1):
        super().__init__()
        self.cross_attention_list = [
            CrossAttention(
                input_dim=input_size,
                embed_dim=embed_dim,
                num_heads=num_heads

            ) for _ in range(attention_layer)
        ]
        self.classify_linear = nn.Linear(input_size, out_class)

    def forward(self, x, y):
        for cross_attention in self.cross_attention_list:
            x = cross_attention(x, y)
        return self.classify_linear(x)
