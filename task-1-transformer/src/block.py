import torch
import torch.nn as nn

from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """Pre-LN 的 encoder block。

    use_residual / use_ln 两个开关只给消融实验用（S2）：
        use_residual=False  去掉 x = x + ... 里的 x，子层输出直接往下传
        use_ln=False        两个 LayerNorm 换成 nn.Identity()
    默认都是 True，行为与标准实现完全一致。
    """

    def __init__(self, H, D_m, D_ff, dropout=0.1, use_residual=True, use_ln=True,
                 use_rope=False, rope_base=10000.0):
        super().__init__()
        self.use_residual = use_residual
        self.attn = MultiHeadAttention(H, D_m, use_rope=use_rope, rope_base=rope_base)
        self.ln1 = nn.LayerNorm(D_m) if use_ln else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(D_m, D_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(D_ff, D_m)
        )
        self.ln2 = nn.LayerNorm(D_m) if use_ln else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None): # (B, T, D_m)
        h = self.ln1(x)
        attn_out = self.drop(self.attn(h, h, h, mask))
        x = x + attn_out if self.use_residual else attn_out

        ffn_out = self.drop(self.ffn(self.ln2(x)))
        x = x + ffn_out if self.use_residual else ffn_out
        return x
