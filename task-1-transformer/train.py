"""任务一 · 训练入口：ChnSentiCorp 情感分类 + 手写 Transformer。

================================ 结构 ================================
    ① 常量与路径      ② SentimentDataset   ③ collate
    ④ tokenizer/loader ⑤ evaluate           ⑥ 实验记录
    ⑦ plot_history    ⑧ parse_args          ⑨ main

================================ 用法 ================================
    python train.py                              # 默认超参，5 个 epoch
    python train.py --limit 512 --epochs 1       # 冒烟，几十秒
    python train.py --d_model 192 --n_layers 6   # 消融实验

================================ 产物 ================================
    ckpt/best.pt      dev acc 最高的一轮，eval/run.py 读它
    ckpt/last.pt      最后一轮（想看"训到最后"什么水平时用）
    ckpt/history.json 曲线数据（写报告时比截图方便）
    figures/loss.png  loss / acc 曲线

================================ 超参来源 ============================
    d_model=128, n_heads=4, n_layers=4, lr=3e-4, batch=32, epochs=5
    是 README「建议超参起点」里的值，不是抄来的：4 层 / 128 维
    在这个 9600 条的数据集上刚好够用，8.5 秒一个 epoch，五分钟能跑 6 组消融。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd          # 读 parquet（9500 条文本用 pandas 最省事）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm        # 进度条；不装也能跑，把 tqdm(...) 那行删掉即可

from src.config import ModelConfig
from src.model import TransformerClassifier, save_checkpoint
from src.tokenizer import CharTokenizer, WordTokenizer, pad_batch

# --------------------------------------------------------------------------- ①
# 用 __file__ 反推绝对路径，而不是写 "data/train.parquet"：
# 相对路径是相对「当前工作目录」解析的，你在 task 目录和仓库根目录跑会指向两处。
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpt"
FIG_DIR = ROOT / "figures"


# --------------------------------------------------------------------------- ②
class SentimentDataset(Dataset):
    """把 dataframe 里的 text/label 预先编码成 id 序列。

    设计要点：
    - 编码放在 __init__：9600 条只要 ~0.1 秒。若放在 __getitem__ 里，
      DataLoader 每个 epoch 都会重新编码一遍，纯浪费。
    - 只存 list[Tensor] 和 list[int]，不抱着 dataframe：DataLoader 会做
      多进程/随机采样，持有原表反而增加内存和序列化负担。
    """

    def __init__(self, df, tokenizer, max_len):
        # encode 返回 (T,) 的 int64 张量，长度已经按 max_len 截断
        self.ids = [tokenizer.encode(t, max_len=max_len) for t in df["text"]]
        # 标签是 numpy.int64，显式转成 python int，避免 collate 里 dtype 混乱
        self.labels = [int(y) for y in df["label"].tolist()]

    def __len__(self):
        # 必须实现：DataLoader 靠它算 len(loader) 和最后一个 batch 的大小
        return len(self.labels)

    def __getitem__(self, i):
        # 这里必须是 O(1) 的列表索引，不要写 iterrows / iloc
        return self.ids[i], self.labels[i]


# --------------------------------------------------------------------------- ③
def collate(batch):
    """list[(ids (T_i,), label)] -> (ids (B,T), mask (B,1,1,T), labels (B,))。

    为什么需要 collate：一个 batch 里句子长短不一，没法直接 stack，
    必须先 pad 到同一长度。pad_batch 顺手把 padding mask 也算出来了，
    mask 形状 (B,1,1,T) 可以直接广播到 attention 的 (B,H,T,T) 分数上。
    """
    # batch 是 [(ids, label), ...]，zip(*batch) 把它拆成两个 tuple
    seqs, labels = zip(*batch)
    ids, mask = pad_batch(list(seqs))
    return ids, mask, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------- ④
def build_tokenizer(train_df, args):
    """只在训练集上建词表（不能用 dev/test，那是数据泄漏）。

    min_freq=2：只出现一次的生僻字直接丢掉，词表从 4248 压到 3654，
    覆盖率仍有 99.94%，验证集 OOV 约 1.18%，是划算的交换。
    """
    cls = WordTokenizer if args.tokenizer == "jieba" else CharTokenizer
    return cls.build(
        train_df["text"].tolist(), min_freq=args.min_freq, max_size=args.vocab_size
    )


def build_loaders(train_df, dev_df, tokenizer, args):
    train_ds = SentimentDataset(train_df, tokenizer, args.max_len)
    dev_ds = SentimentDataset(dev_df, tokenizer, args.max_len)

    # shuffle=True 只给训练集：dev 参不参与梯度，顺序无关紧要，
    # 但固定顺序能让每次 evaluate 的中间结果完全可复现。
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate,
        # Windows 下 num_workers>0 要走 spawn，容易卡死或报 pickle 错；
        # 数据已经全在内存里，主进程加载完全够快（~40 it/s）。
        num_workers=0, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    return train_loader, dev_loader


# --------------------------------------------------------------------------- ⑤
@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (准确率, 平均 loss)。

    三个细节：
    - @torch.no_grad()：不建计算图，省显存也快 2~3 倍
    - model.eval()：关掉 dropout，否则每次评估结果都在抖
    - loss 按样本数加权：最后一个 batch 可能不满，直接平均 batch loss 会偏
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0

    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        # .item() 拿出来是 python float，不会累积计算图
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    # max(total, 1) 防 dev 集为空时除零
    return correct / max(total, 1), loss_sum / max(total, 1)


# --------------------------------------------------------------------------- ⑥
def save_history(history, path):
    """把每个 epoch 的指标落成 json。

    为什么单独存：checkpoint 的格式要跟 load_for_eval 对齐（cfg/vocab/state_dict），
    往里塞训练指标会把两件事耦合在一起。曲线数据单独一个文件更干净，
    写报告时也能直接读数字，不用从终端日志里抄。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- ⑦
def plot_history(history, path):
    """画 loss / dev acc 双曲线，存成 png（报告里直接用）。"""
    try:
        import matplotlib

        # Agg = 无界面后端，只往文件里写。
        # 必须在 import pyplot 之前设置，否则某些环境会尝试弹窗然后卡住。
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # 画不了图不该让训练失败
        print(f"[提示] 跳过画图：{exc}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    # 左图：train / dev loss。两条线的间距就是过拟合程度
    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["dev_loss"], marker="s", label="dev")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("loss"); ax1.legend()

    # 右图：dev acc，把通过线（0.80）和参考基线（0.85）画成参考线
    ax2.plot(epochs, history["dev_acc"], marker="o", color="tab:green")
    ax2.axhline(0.80, ls="--", lw=1, color="tab:red", label="pass line 0.80")
    ax2.axhline(0.85, ls=":", lw=1, color="tab:orange", label="baseline 0.85")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("dev accuracy")
    ax2.set_title("dev accuracy"); ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # 不关会泄漏 figure，循环里画多张图时会吃满内存
    print(f"曲线已保存：{path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- ⑧
def parse_args():
    """命令行参数。默认值 = README 建议超参，能直接 python train.py 跑。"""
    p = argparse.ArgumentParser(description="ChnSentiCorp + 手写 Transformer 分类")

    # ---- 模型结构（改这些会改变 checkpoint 的形状，旧 ckpt 会加载失败）----
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)      # 必须整除 d_model
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=None)      # None -> 4 * d_model
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)    # p95≈249，能盖住 95% 的样本
    p.add_argument("--vocab_size", type=int, default=4000)  # 词表上限（含 3 个特殊符）
    p.add_argument("--min_freq", type=int, default=2)     # 低于此频次的字进 <unk>

    # ---- 训练 ----
    p.add_argument("--epochs", type=int, default=5)       # 也是 cosine 的 T_max
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)      # AdamW 的峰值学习率
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)     # 梯度裁剪阈值
    p.add_argument("--patience", type=int, default=3)     # 早停耐心值（按 dev acc）

    # ---- 调试 / 环境 ----
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条训练，0 表示全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log_every", type=int, default=100)  # 每 N 步打一行 loss，0 表示不打
    p.add_argument("--run_name", type=str, default="",
                   help="消融实验用：输出 best_{run_name}.pt，留空则用 best.pt")
    p.add_argument("--no_plot", action="store_true",
                   help="不画损失曲线（消融时不需要，也避免覆盖 figures/loss.png）")
    p.add_argument("--no_residual", action="store_true", help="消融 S2：去掉两条残差连接")
    p.add_argument("--no_ln", action="store_true", help="消融 S2：去掉所有 LayerNorm")
    # 分词与位置编码（S4）
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "jieba"],
                   help="char=字级；jieba=词级（需要 pip install jieba）")
    p.add_argument("--pos", type=str, default="abs", choices=["abs", "rope"],
                   help="abs=可学习绝对位置嵌入；rope=旋转位置编码")
    return p.parse_args()


# --------------------------------------------------------------------------- ⑨
def main():
    args = parse_args()

    # 输出路径：默认 best.pt / last.pt / history.json / loss.png；
    # 加了 --run_name 就变成 best_H4L2.pt 之类，消融实验之间不会互相覆盖
    suffix = f"_{args.run_name}" if args.run_name else ""
    best_path = CKPT_DIR / f"best{suffix}.pt"
    last_path = CKPT_DIR / f"last{suffix}.pt"
    hist_path = CKPT_DIR / f"history{suffix}.json"
    plot_path = FIG_DIR / f"loss{suffix}.png"

    # ---- 1) 可复现：四个随机源都要锁 ----
    # python 的 random、numpy 被库内部用、torch 的 CPU/GPU 各一套
    random.seed(args.seed)
    torch.manual_seed(args.seed)          # CPU 与所有 GPU 的默认 generator
    torch.cuda.manual_seed_all(args.seed) # 多卡时逐卡设（单卡这行是冗余但无害）

    # ---- 2) 设备：auto 时优先 GPU ----
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- 3) 数据 ----
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    dev_df = pd.read_parquet(DATA_DIR / "validation.parquet")

    # --limit：冒烟用，只取一小部分，几十秒就能验证整条链路通不通
    if args.limit > 0:
        train_df = train_df.sample(
            n=min(args.limit, len(train_df)), random_state=args.seed
        ).reset_index(drop=True)      # sample 后索引是乱的，reset 一下更整齐
        dev_df = dev_df.sample(
            n=min(args.limit // 4 + 1, len(dev_df)), random_state=args.seed
        ).reset_index(drop=True)

    # ---- 4) 词表 → 配置 → loader ----
    tokenizer = build_tokenizer(train_df, args)

    # ModelConfig 的字段名是 D_m / H / D_ff，和命令行参数 d_model / n_heads / d_ff 对应
    cfg = ModelConfig(
        vocab_size=len(tokenizer),      # 必须用实际词表大小，不能用 args.vocab_size
        D_m=args.d_model,
        H=args.n_heads,
        n_layers=args.n_layers,
        D_ff=args.d_ff,
        max_len=args.max_len,
        num_classes=2,
        dropout=args.dropout,
        pad_id=0,
        use_residual=not args.no_residual,   # S2 消融：拆掉 residual
        use_ln=not args.no_ln,               # S2 消融：拆掉 LayerNorm
        tokenizer=args.tokenizer,            # char / jieba
        pos_encoding=args.pos,               # abs / rope
    )
    train_loader, dev_loader = build_loaders(train_df, dev_df, tokenizer, args)

    print(f"device = {device}")
    print(f"词表 = {len(tokenizer)} | 训练 {len(train_loader.dataset)} 条 | "
          f"dev {len(dev_loader.dataset)} 条")

    # ---- 5) 模型 ----
    model = TransformerClassifier(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params / 1e6:.2f}M")

    # ---- 6) 优化器：AdamW + 参数分组 ----
    # bias 和 LayerNorm 的 γ/β 不参与权重衰减：
    # 它们不是"特征权重"，压小只会让模型变僵，GPT-3 起是标准做法。
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2 or name.endswith(".bias") or ".ln" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.98),          # Transformer 惯例，比默认 0.999 响应更快
    )

    # cosine 从 lr 平滑降到 0；T_max 必须等于总 epoch 数，
    # 否则早停时会停在"lr 还没降下来"的位置。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CrossEntropyLoss 内部自带 log_softmax，模型输出必须是 logits，不能先 softmax
    criterion = nn.CrossEntropyLoss()

    # ---- 7) 训练循环 ----
    history = {"train_loss": [], "dev_loss": [], "dev_acc": [], "lr": []}
    best_acc, bad_epochs = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()               # 打开 dropout（evaluate 里会切回 eval）
        epoch_loss = 0.0
        started = time.time()

        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for step, (ids, mask, labels) in enumerate(bar, start=1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

            logits = model(ids, mask)          # mask 一定要传，否则 PAD 会参与注意力
            loss = criterion(logits, labels)

            # set_to_none=True 比清零更快：直接把 .grad 置空，省一次写内存
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 裁剪梯度范数：小模型偶尔会遇到 loss spike，一步大更新就毁了
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
            if args.log_every > 0 and step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        # ---- 每个 epoch 收尾：先算指标，再调 scheduler ----
        train_loss = epoch_loss / len(train_loader.dataset)
        dev_acc, dev_loss = evaluate(model, dev_loader, device)

        lr_now = scheduler.get_last_lr()[0]    # 先记下本 epoch 用的 lr
        scheduler.step()                       # 再前进（必须在 optimizer.step 之后）

        for key, value in [("train_loss", train_loss), ("dev_loss", dev_loss),
                           ("dev_acc", dev_acc), ("lr", lr_now)]:
            history[key].append(value)

        # ---- early stopping：按 dev 准确率，不按 loss ----
        # 理由：loss 可能继续降（模型在拟合噪声），而 acc 已经见顶回落
        improved = dev_acc > best_acc
        print(
            f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} "
            f"lr={lr_now:.2e} | {time.time() - started:.1f}s"
            + ("  <- best" if improved else "")
        )

        # last 每轮都存（想看最终状态），best 只在创新高时存（提交用这个）
        save_checkpoint(model, tokenizer, last_path)
        if improved:
            best_acc, bad_epochs = dev_acc, 0
            save_checkpoint(model, tokenizer, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"dev acc 连续 {args.patience} 轮没提升，early stop。")
                break

    # ---- 8) 收尾：写实验记录 + 画曲线 ----
    history["best_dev_acc"] = best_acc
    save_history(history, hist_path)
    if not args.no_plot:
        plot_history(history, plot_path)

    print(f"\n最佳 dev acc = {best_acc:.4f}  (通过线 0.80 / 参考基线 0.85)")
    print("下一步：python eval/run.py")


if __name__ == "__main__":
    main()
