"""M5：注意力热图。用法：python visualize.py"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from src.model import load_for_eval
from src.tokenizer import CharTokenizer

# 中文字体：不加这两行，坐标轴上的汉字会变成方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- ① 路径常量 ----
ROOT = Path(__file__).resolve().parent
# 注意：这里刻意指向**基线模型**（字级 + 绝对位置编码），不是 ckpt/best.pt。
# 原因：本脚本的 get_attention 直接复现 "tok_emb + pos_emb → ln1 → Q/K" 这段，
# 而最终模型（第 6 节的 jieba + RoPE 版本）既没有 pos_emb、又要在 attention 内部
# 额外调用 apply_rope，这条路径对不上。基线模型结构完全匹配，且它是报告 §8 的分析对象。
CKPT = ROOT / 'ckpt' / 'best_baseline.pt'
DATA = ROOT / 'data' / 'validation.parquet'   # 样本从验证集挑（没参与训练）
FIG_DIR = ROOT / 'figures'

SHORT_CHARS = 40          # "短句"阈值：验证集 p25≈46，取 40 保证够短
LONG_CHARS = 150          # "长句"阈值：验证集 p90≈191、p95≈248
BEST_HEAD_ONLY = True     # True 每类只画最集中的 1 个头（4 张图，够交付）

# 刻度上显示用的短符号：<unk> 有 5 个字符，会把一格撑得很宽，看着更乱
DISPLAY = {'<cls>': '[C]', '<unk>': '?', '<pad>': '_'}

# ---- ② 加载模型 + 词表（tokenize_fn 是函数，没有 decode，得自己重建）----
def load_all():
    model, tokenize_fn = load_for_eval(str(CKPT))
    model.eval()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    tokenizer = CharTokenizer.from_vocab(ckpt['vocab'])
    return model, tokenize_fn, tokenizer

# ---- ③ 核心：取某一层的注意力权重（就是那 7 行，包成函数）----
@torch.no_grad()
def get_attention(model, ids, layer = 0):
    """ids: (1,T) -> (H, T, T)"""
    cfg = model.cfg
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.size(1), device=ids.device))
    blk = model.blocks[layer]
    h = blk.ln1(x)
    q = blk.attn.split_heads(blk.attn.W_Q(h))
    k = blk.attn.split_heads(blk.attn.W_K(h))
    scores = q @ k.transpose(-2, -1) / (cfg.D_m // cfg.H) ** 0.5
    mask = (ids == cfg.pad_id)[:, None, None, :]
    p = torch.softmax(scores.masked_fill(mask, float('-inf')), -1)
    return p[0] # (H, T, T)

# ---- ④ 画图 ----
def plot_heatmap(chars, weights, title, out_path):
    n = len(chars)
    # 句子越长画布越大，保证 176 个刻度标签也放得下
    side = max(6, n * 0.16)
    fig, ax = plt.subplots(figsize = (side + 1.5, side))

    # 关键：按「均匀分布」归一化。原始权重平均只有 1/T（T=172 时是 0.006），
    # 直接画出来整张图几乎全白。乘上 T 之后：1.0 = 完全均匀，越大越集中。
    scaled = weights * n
    import numpy as np
    vmax = max(3.0, float(np.percentile(scaled, 99.9)))
    im = ax.imshow(scaled, cmap = 'Blues', vmin = 0, vmax = vmax)

    # 每个词元都标出来。之前抽稀成「每 n//40 个标一个」，坐标轴读出来是
    # 第 1、5、9… 个字，句子被跳着念，自然就不通顺了 —— 这就是那个现象的来源。
    # 字多时靠放大画布 + 缩小字号来装，标签一个都不能丢。
    fs = 7 if n <= 80 else 6
    ax.set_xticks(range(n)); ax.set_xticklabels(chars, rotation=90, fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(chars, fontsize=fs)
    ax.set_xlabel("key（被看的词）"); ax.set_ylabel("query（在看的词）")
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label('注意力 ÷ 均匀分布（1=平均，越大越集中）', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---- ⑤ 四条样本：从验证集里按「第一个」规则挑 ----
def pick_samples(df, short_chars = SHORT_CHARS, long_chars = LONG_CHARS):
    """返回 [(tag, label, text)]：

    第一个 ≤40 字的正面 / 第一个 ≤40 字的负面 /
    第一个 ≥150 字的正面 / 第一个 ≥150 字的负面。
    不硬编码文本，换数据集也能跑。
    """
    df = df.copy()
    df['n_chars'] = df['text'].str.len()

    def first(label, min_chars = 0, max_chars = 10 ** 6):
        sub = df[(df['label'] == label)
                 & (df['n_chars'] >= min_chars)
                 & (df['n_chars'] <= max_chars)]
        if sub.empty:
            raise ValueError(f'没有 label={label} 且长度在 [{min_chars},{max_chars}] 的样本')
        row = sub.iloc[0]
        return int(row['label']), row['text']

    return [
        ('01_短正面', *first(1, 0, short_chars)),
        ('02_短负面', *first(0, 0, short_chars)),
        ('03_长正面', *first(1, long_chars)),
        ('04_长负面', *first(0, long_chars)),
    ]

def main():
    model, tokenize_fn, tokenizer = load_all()
    df = pd.read_parquet(DATA)
    SAMPLES = pick_samples(df)
    FIG_DIR.mkdir(parents = True, exist_ok = True)

    print(f'模型: d_model={model.cfg.D_m} heads={model.cfg.H} layers={model.cfg.n_layers}')

    for tag, label, text in SAMPLES:
        ids = tokenize_fn(text).unsqueeze(0)
        raw = [tokenizer.decode([i]) for i in ids[0].tolist()]
        chars = [DISPLAY.get(c, c) for c in raw]     # 特殊符号换短标记
        n_unk = raw.count('<unk>')

        # 扫所有层，找出权重最集中的那个头（平均分布的头发出来是灰糊糊一张图）
        best = None                     # (集中度, layer, head, p)
        for layer in range(model.cfg.n_layers):
            p = get_attention(model, ids, layer)
            for h in range(p.shape[0]):
                c = float(p[h].max(dim = -1).values.mean())
                if best is None or c > best[0]:
                    best = (c, layer, h, p[h])

        c, layer, head, p = best
        print(f'  [{tag}] label={label} T={len(chars)} unk={n_unk} '
              f'最佳 layer={layer} head={head} 集中度={c:.3f}')
        print(f'       原文: {text}')

        if BEST_HEAD_ONLY:
            plot_heatmap(chars, p.numpy(),
                         f'{tag}  label={label}  T={len(chars)}\n'
                         f'layer {layer} · head {head} · 集中度={c:.3f}',
                         FIG_DIR / f'attn_{tag}.png')
            print(f'       -> figures/attn_{tag}.png')
        else:
            # 每个头各画一张，方便横向比较（4 条样本 × 4 头 = 16 张）
            for h in range(p.shape[0]):
                plot_heatmap(chars, get_attention(model, ids, layer)[h].numpy(),
                             f'{tag} · layer {layer} · head {h}',
                             FIG_DIR / f'attn_{tag}_L{layer}H{h}.png')
            print(f'       -> 按 head 共出 4 张')

if __name__ == "__main__":
    main()
