import torch
import torch.nn as nn

from dataclasses import asdict
from pathlib import Path

from .block import TransformerBlock
from .config import ModelConfig
from .tokenizer import CharTokenizer, WordTokenizer

# ids (B,T) → embedding + 位置编码 → dropout → N × TransformerBlock → ln_f → pooling → Linear → (B,2)
class TransformerClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.D_m, padding_idx=cfg.pad_id)
        # RoPE 自己编码位置，不需要可学习的位置嵌入表
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.D_m) if cfg.pos_encoding == "abs" else None
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.H, cfg.D_m, cfg.D_ff, cfg.dropout,
                             use_residual=cfg.use_residual, use_ln=cfg.use_ln,
                             use_rope=(cfg.pos_encoding == "rope"), rope_base=cfg.rope_base)
            for _ in range(cfg.n_layers)
        ])
        # 去掉 LN 的消融实验里，ln_f 也要一起去掉，否则模型里还留着一层归一化
        self.ln_f = nn.LayerNorm(cfg.D_m) if cfg.use_ln else nn.Identity()
        self.head = nn.Linear(cfg.D_m, cfg.num_classes)

    def forward(self, ids, mask=None):
        if mask is None:
            mask = (ids == self.cfg.pad_id)[:, None, None, :] # (B,1,1,T) True=屏蔽, 后续广播到(B, H, T_q, T_k)
        tokens = self.tok_emb(ids) # (B, T, D_m)
        if self.pos_emb is not None:  # abs 位置编码；RoPE 在 attention 内部完成
            tokens = tokens + self.pos_emb(torch.arange(ids.size(1),device=ids.device))
        tokens = self.dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens, mask)
        tokens = self.ln_f(tokens)
        m = (ids != self.cfg.pad_id).unsqueeze(-1).float()
        tokens = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1) # mean 池化：只对真token求平均
        tokens = self.dropout(tokens)
        tokens = self.head(tokens)
        return tokens

def load_for_eval(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location = 'cpu', weights_only=True) # 读
    cfg = ModelConfig(**ckpt['cfg']) # 还原结构
    model = TransformerClassifier(cfg) # 建模型
    model.load_state_dict(ckpt['state_dict'])
    model.eval() # 关掉 dropout
    # 词级模型必须用 WordTokenizer 还原，否则切分方式跟训练时不一致
    tok_cls = WordTokenizer if cfg.tokenizer == "jieba" else CharTokenizer
    tokenizer = tok_cls.from_vocab(ckpt['vocab']) # 还原词表
    def tokenize_fn(text: str) -> torch.Tensor:
        return tokenizer.encode(text, max_len = cfg.max_len)
    return model, tokenize_fn

def save_checkpoint(model, tokenizer, path):
    """保存成 load_for_eval 能直接读的格式。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'cfg': asdict(model.cfg), # 必须是普通 dict，不能是 ModelConfig 实例
        'vocab': tokenizer.vocab, # dict[str, int]
        'state_dict': model.state_dict() # 权重
    }, path)
