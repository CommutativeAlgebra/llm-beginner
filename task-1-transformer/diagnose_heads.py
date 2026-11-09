"""head 冗余诊断：为什么减少 head 数不掉点。

三个实验（对应 report.md 第 4 节）：
    ① 各头输出投影的行范数        —— 是否某个头"嗓门大"
    ② 头↔头 / 头↔均匀的 JS 散度 + 归一化熵 —— 头有没有分化
    ③ 剪枝：只保留一个头时的 dev acc        —— 删掉头到底损失多少

用法：python diagnose_heads.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import load_for_eval
from src.tokenizer import CharTokenizer, pad_batch

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "ckpt" / "best.pt"
DATA = ROOT / "data" / "validation.parquet"

N_EVAL = 600        # 剪枝实验用前 600 条（完整模型在该子集上 ≈0.838；全量 1200 条是 0.8492）
N_JS = 32           # JS / 熵统计用前 32 条即可
BATCH = 64


# --------------------------------------------------------------------------- 工具
def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q)，0·log0 按 0 处理。"""
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    """JS 散度：对称，取值 [0, ln2]。"""
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_mean(p: np.ndarray, q: np.ndarray) -> float:
    """逐行（逐 query）算 JS 再取平均。p/q 形状 (T_q, T_k)。"""
    m = 0.5 * (p + q)
    rows = 0.5 * _kl_rows(p, m) + 0.5 * _kl_rows(q, m)
    return float(rows.mean())


def _kl_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = a > 0
    safe_a = np.where(mask, a, 1.0)
    return np.where(mask, a * np.log(safe_a / b), 0.0).sum(-1)


def ent_mean(p: np.ndarray) -> float:
    """逐行算熵再取平均。"""
    mask = p > 0
    safe = np.where(mask, p, 1.0)
    return float(np.where(mask, -p * np.log(safe), 0.0).sum(-1).mean())


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def attention_of(model, ids, mask, layer: int) -> torch.Tensor:
    """取第 layer 层的注意力权重 (H, T, T)（batch=1）。"""
    cfg = model.cfg
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.size(1), device=ids.device))
    for i in range(layer):
        x = model.blocks[i](x, mask)
    blk = model.blocks[layer]
    h = blk.ln1(x)
    q = blk.attn.split_heads(blk.attn.W_Q(h))
    k = blk.attn.split_heads(blk.attn.W_K(h))
    scores = q @ k.transpose(-2, -1) / (cfg.D_m // cfg.H) ** 0.5
    return torch.softmax(scores.masked_fill(mask, float("-inf")), -1)[0]


def main() -> None:
    model, tokenize_fn = load_for_eval(str(CKPT))
    model.eval()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    tok = CharTokenizer.from_vocab(ckpt["vocab"])
    H, D = model.cfg.H, model.cfg.D_m
    d_head = D // H

    dev = pd.read_parquet(DATA).head(N_EVAL)

    # ---- ① 各头输出投影的行范数 ----
    print("=" * 72)
    print("① W_out 中每个头对应的行块范数（判断是否有某个头嗓门更大）")
    for L, blk in enumerate(model.blocks):
        W = blk.attn.W_out.weight.data
        norms = [W[h * d_head:(h + 1) * d_head].norm().item() for h in range(H)]
        share = [n / sum(norms) for n in norms]
        print(f"   layer {L}: 范数={[round(n, 2) for n in norms]}  "
              f"占比={[f'{s:.0%}' for s in share]}")

    # ---- ② 头↔头 / 头↔均匀 的 JS + 归一化熵 ----
    print("=" * 72)
    print("② 头↔头 JS、头↔均匀 JS、归一化熵（0=只盯一个位置，1=完全平均）")
    print(f"   {'layer':<7}{'头↔头':>9}{'头↔均匀':>10}{'熵/lnT':>9}")

    ids, mask = pad_batch([tokenize_fn(t) for t in dev.head(N_JS)["text"].tolist()])
    for L in range(model.cfg.n_layers):
        # batch=1 取第一条样本，保证第 L 层的输入是"真正经过了前 L 层"的
        p = attention_of(model, ids[:1], mask[:1], layer=L).numpy()
        valid = ~mask[0, 0, 0].numpy()
        sub = p[:, valid][:, :, valid]
        T = sub.shape[-1]
        uni = np.full_like(sub, 1.0 / T)
        hh = np.mean([js_mean(sub[a], sub[b]) for a, b in itertools.combinations(range(H), 2)])
        hu = np.mean([js_mean(sub[a], uni[a]) for a in range(H)])
        ent = np.mean([ent_mean(sub[a]) / np.log(T) for a in range(H)])
        print(f"   layer {L:<3} {hh:>9.4f}{hu:>10.4f}{ent:>9.3f}")

    # ---- ③ 剪枝：只保留一个头 ----
    print("=" * 72)
    print(f"③ 剪枝实验（前 {N_EVAL} 条 dev）")

    def acc() -> float:
        correct = total = 0
        for i in range(0, len(dev), BATCH):
            chunk = dev.iloc[i:i + BATCH]
            ids, mask = pad_batch([tokenize_fn(t) for t in chunk["text"].tolist()])
            lab = torch.tensor(chunk["label"].tolist())
            pred = model(ids, mask).argmax(-1).cpu()
            correct += (pred == lab).sum().item()
            total += lab.numel()
        return correct / total

    W = model.blocks[0].attn.W_out.weight
    orig = W.data.clone()
    base = acc()
    print(f"   完整 {H} 头:      acc = {base:.4f}")

    for keep in range(H):
        for h in range(H):
            if h != keep:
                W.data[h * d_head:(h + 1) * d_head, :] = 0
        a = acc()
        W.data.copy_(orig)
        print(f"   只保留 head {keep}: acc = {a:.4f}   (Δ {a - base:+.4f})")

    if H >= 2:
        for h in range(2, H):
            W.data[h * d_head:(h + 1) * d_head, :] = 0
        print(f"   保留 head 0+1:  acc = {acc():.4f}   (Δ {acc() - base:+.4f})")
        W.data.copy_(orig)

    print(f"   恢复后核对:     acc = {acc():.4f}")


if __name__ == "__main__":
    main()
