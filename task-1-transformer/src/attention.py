import torch
import torch.nn as nn

import math

# (B, H, T, D) 分别对应batch_size, num_heads, seq_len, d_k

def scaled_dot_product_attention(Q, K, V, mask = None): # (B, H, T, D)
    D = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D) # 除以sqrt(D)，消除方差
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, -1)
    return torch.matmul(attn, V)

def apply_rope(x, base: float = 10000.0):
    """旋转位置编码（RoPE），作用在 Q/K 上，形状 (B, H, T, D)。

    做法：把最后一维的相邻两维看成复数 (x1 + i·x2)，乘上 e^{i·m·θ}，
    其中 m 是位置、θ_j = base^(-2j/D)。旋转后 Q·K 的内积只依赖相对位置差。

    实现上采用"相邻两维一组"的配对方式（0-1, 2-3, ...）；
    另一种常见写法是"前后折半"（j 配 j+D/2），两者数学等价，但**必须与解码端一致**。
    """
    T, D = x.size(-2), x.size(-1)
    # θ_j = base^(-2j/D)，j = 0..D/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    freqs = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), inv_freq)  # (T, D/2)
    cos = freqs.cos()[None, None, :, :]     # (1, 1, T, D/2)
    sin = freqs.sin()[None, None, :, :]

    x1 = x[..., 0::2]                        # 偶数维
    x2 = x[..., 1::2]                        # 奇数维
    # 复数乘法 (x1 + i·x2)(cos + i·sin)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)               # 把最后一对两维拼回去


class MultiHeadAttention(nn.Module): # (B, T, D_m=H*D)
    def __init__(self, H, D_m, use_rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.H = H
        self.D_m = D_m
        self.D = D_m // H
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.W_Q = nn.Linear(D_m, D_m, bias=False)
        self.W_K = nn.Linear(D_m, D_m, bias=False)
        self.W_V = nn.Linear(D_m, D_m, bias=False)
        self.W_out = nn.Linear(D_m, D_m, bias=False)

    def split_heads(self, x): # (B, T, D_m=H*D)
        B, T, _ = x.size()
        x = x.view(B, T, self.H, self.D)
        x = x.transpose(1, 2).contiguous() # (B, H, T, D)
        return x

    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.W_Q(Q)) # (B, H, T_q, D)
        K = self.split_heads(self.W_K(K)) # (B, H, T_k, D)
        V = self.split_heads(self.W_V(V)) # (B, H, T_v=T_k, D)

        # RoPE 只加在 Q/K 上，绝不加 V（V 不参与位置相关打分）
        if self.use_rope:
            Q = apply_rope(Q, self.rope_base)
            K = apply_rope(K, self.rope_base)

        # scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)
        # if mask is not None:
        #     scores = scores.masked_fill(mask, float('-inf'))
        # attn = torch.softmax(scores, -1)
        # attn = torch.matmul(attn, V) # (B, H, T_q, D)
        attn = scaled_dot_product_attention(Q, K, V, mask = mask)

        B, _, T_q, _ =attn.shape
        attn = attn.transpose(1,2).contiguous()
        attn = attn.view(B, T_q, self.D_m)
        attn = self.W_out(attn)

        return attn