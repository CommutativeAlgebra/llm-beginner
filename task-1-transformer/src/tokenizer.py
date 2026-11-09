"""字级 tokenizer + padding 工具（可直接用，不必自己写）。

为什么用字级：中文情感分类用「一个字一个 id」就够了，本数据集训练集一共
4248 个不同字符，min_freq=2 建表后 3654 个，覆盖率 99.94%。不需要 BPE，
也不需要联网下载别人的词表。

特殊符号占前三个 id：
    <pad>=0（padding_idx）, <unk>=1, <cls>=2（句首，供 [CLS] 池化用）
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>"]


class CharTokenizer:
    """字 -> id 映射。词表就是个普通 dict，能直接塞进 checkpoint。

    ``_split`` 是"怎么把一句话切成 token"的那一步：基类按字切，
    ``WordTokenizer`` 覆盖它改成 jieba 词级切分，其余逻辑完全复用。
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = dict(vocab)
        self.itos = {i: t for t, i in self.vocab.items()}

    @staticmethod
    def _split(text: str) -> list[str]:
        """默认：按字切，丢掉空白。子类可覆盖。"""
        return [ch for ch in text if not ch.isspace()]

    # ---------------------------------------------------------------- 构建
    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> "CharTokenizer":
        """按 token 频率建表。min_freq=2 会丢掉只出现一次的生僻字（进 <unk>）。"""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls._split(text))

        chars = [c for c, n in counter.most_common() if n >= min_freq]
        if max_size is not None:
            chars = chars[: max(0, max_size - len(SPECIAL_TOKENS))]

        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in chars:
            vocab.setdefault(ch, len(vocab))
        return cls(vocab)

    @classmethod
    def from_vocab(cls, vocab: dict[str, int]) -> "CharTokenizer":
        return cls(vocab)

    # ---------------------------------------------------------------- 编解码
    def encode(
        self,
        text: str,
        max_len: int | None = None,
        add_cls: bool = True,
    ) -> torch.LongTensor:
        """单条文本 -> ``(T,)`` LongTensor。超过 max_len 从尾部截断。"""
        ids = [self.vocab.get(tok, UNK_ID) for tok in self._split(text)]
        if add_cls:
            ids = [CLS_ID] + ids
        if max_len is not None:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "<unk>") for i in ids)

    def __len__(self) -> int:
        return len(self.vocab)


class WordTokenizer(CharTokenizer):
    """jieba 词级 tokenizer，接口与 CharTokenizer 完全一致。

    唯一的差别是 ``_split``：把"按字切"换成"jieba 分词"。
    好处是"不错""太差""服务员"这类词成为整体，热图上的标签更可读；
    代价是词表更大、OOV 更多（生僻词会掉进 <unk>）。
    """

    @staticmethod
    def _split(text: str) -> list[str]:
        import jieba  # 延迟导入：不用词级 tokenizer 时不依赖 jieba

        return [w for w in jieba.lcut(text) if not w.isspace()]


def pad_batch(
    seqs: list[torch.LongTensor],
    pad_id: int = PAD_ID,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """把一批变长 id 序列 pad 成方阵。

    返回
    ----
    ids  : ``(B, T)``，右侧补 pad_id
    mask : ``(B, 1, 1, T)`` bool，``True = PAD，要屏蔽``，可直接广播进 attention
    """
    if not seqs:
        raise ValueError("pad_batch 收到空列表")

    length = max(int(s.numel()) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : int(seq.numel())] = seq

    mask = (ids == pad_id)[:, None, None, :]
    return ids, mask
