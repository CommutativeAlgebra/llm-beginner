from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    vocab_size: int
    D_m: int = 128
    H: int = 4
    n_layers: int = 4
    D_ff: int | None = None
    max_len: int = 256
    num_classes: int = 2
    dropout: float = 0.1
    pad_id: int = 0
    # 消融开关（S2）：默认都开着，和标准 Transformer 一致
    use_residual: bool = True     # False = 去掉两条残差连接
    use_ln: bool = True           # False = 去掉所有 LayerNorm（含 ln_f）
    # 分词与位置编码（S4）
    tokenizer: str = "char"       # "char"（字级）| "jieba"（词级）
    pos_encoding: str = "abs"     # "abs"（可学习绝对位置嵌入）| "rope"（旋转位置编码）
    rope_base: float = 10000.0    # 仅 pos_encoding="rope" 时生效

    def __post_init__(self):
        if self.D_ff is None:
            self.D_ff = 4 * self.D_m