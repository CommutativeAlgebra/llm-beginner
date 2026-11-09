"""下载 ChnSentiCorp 中文情感分类数据集到当前目录（适配 datasets 4.x / 5.x）。

为什么改：Hugging Face 从 datasets 4.0 起移除了「数据集加载脚本」支持，
而 seamew/ChnSentiCorp 是 script-based dataset，所以原来的
    load_dataset("seamew/ChnSentiCorp")
会直接报
    RuntimeError: Dataset scripts are no longer supported, but found ChnSentiCorp.py
那个脚本本身还是从 Google Drive 拉 CSV，国内基本拉不动。

本脚本改为直接下载仓库里现成的 Arrow 数据文件（chn_senti_corp-*.arrow），
用 pyarrow 读出再存成 parquet，绕开 datasets 的加载逻辑，与 datasets 版本无关。

依赖：pip install pyarrow
国内加速（PowerShell）：$env:HF_ENDPOINT = "https://hf-mirror.com"
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "cache"
REPO = "seamew/ChnSentiCorp"
SPLITS = ("train", "validation", "test")


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(url: str, dst: Path) -> Path:
    """下载到 dst，已存在则复用（避免重复拉 3 MB 文件）。"""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"    命中缓存 {dst.name}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(dst)
    return dst


def arrow_to_parquet(arrow_path: Path, out_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    with pa.memory_map(str(arrow_path), "r") as src:
        table = ipc.open_stream(src).read_all()
    table = table.select(["text", "label"])  # 原文件列序是 (label, text)，统一一下
    pq.write_table(table, out_path)
    return table.num_rows


def main() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("[错误] 缺少依赖：pip install pyarrow")

    if "HF_ENDPOINT" not in os.environ:
        print('[提示] 如下载缓慢，可先 $env:HF_ENDPOINT="https://hf-mirror.com" 再重试\n')

    print(f"正在下载 {REPO} ...")
    for split in SPLITS:
        name = f"chn_senti_corp-{split}.arrow"
        url = f"{endpoint()}/datasets/{REPO}/resolve/main/{name}"
        try:
            arrow_path = fetch(url, CACHE_DIR / name)
        except Exception as e:  # noqa: BLE001
            sys.exit(
                f"[错误] 下载 {name} 失败：{type(e).__name__}: {e}\n"
                '       若是网络问题，试试 $env:HF_ENDPOINT="https://hf-mirror.com"'
            )
        out = DATA_DIR / f"{split}.parquet"
        n = arrow_to_parquet(arrow_path, out)
        print(f"  {split}: {n} 条 -> {out.name}")

    print(f"\n完成。数据保存在 {DATA_DIR}")
    print("\n--- ModelScope 备选下载 ---")
    print("如果你完全无法访问 HF：")
    print("  pip install modelscope")
    print("  python -c \"from modelscope.msdatasets import MsDataset; "
          "ds = MsDataset.load('ChnSentiCorp'); print(ds)\"")


if __name__ == "__main__":
    main()
