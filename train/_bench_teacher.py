# -*- coding: utf-8 -*-
"""教师 vs 学生 延迟实测：决定要不要蒸馏

测四组：
  1. bge-small 单条查询延迟（GPU / CPU）
  2. bge-small 批量吞吐（200 条）
  3. 字节 CNN 单条延迟（GPU / CPU）
  4. 字节 CNN 批量吞吐
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from memory_core import CNNEncoder  # noqa: E402

D, C, L = 128, 512, 64
CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"
REPS_SINGLE = 50
N_BATCH = 200


def to_tensor(text):
    return torch.tensor((list(text.encode("utf-8", "ignore")) + [0] * L)[:L], dtype=torch.long)


def load_sents(n=250):
    raw = open(CORPUS, "rb").read()[20_000_000:24_000_000]
    d3 = [b"\xe3\x80\x82", b"\xef\xbc\x9f", b"\xef\xbc\x81"]
    pos = [0]
    for i in range(len(raw) - 3):
        if raw[i:i + 3] in d3 or raw[i:i + 1] == b"\n":
            pos.append(i + 3)
    out = []
    for j in range(len(pos) - 1):
        s = raw[pos[j]:pos[j + 1]].strip()
        if 8 <= len(s) <= 64:
            out.append(s.decode("utf-8", "ignore"))
        if len(out) >= n:
            break
    return out


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, reps):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    sents = load_sents()
    print(f"样本 {len(sents)} 条\n")

    # --- 教师 ---
    from sentence_transformers import SentenceTransformer
    for tag, dev in (("GPU", "cuda"), ("CPU", "cpu")):
        if tag == "GPU" and not torch.cuda.is_available():
            continue
        m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
        single = timeit(lambda: m.encode([sents[0]], normalize_embeddings=True,
                                         show_progress_bar=False), REPS_SINGLE)
        t0 = time.perf_counter()
        m.encode(sents[:N_BATCH], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        batch = (time.perf_counter() - t0) / N_BATCH * 1000
        print(f"  教师 bge-small [{tag}]   单条 {single:7.2f} ms   批量 {batch:6.2f} ms/条")

    # --- 学生 ---
    encoder = CNNEncoder(D)
    st = torch.load(os.path.join(BASE, "coord.pt"), map_location="cpu")
    if isinstance(st.get("encoder"), dict):
        encoder.load_state_dict(st["encoder"], strict=False)
    encoder.eval()

    for tag, dev in (("GPU", "cuda"), ("CPU", "cpu")):
        if tag == "GPU" and not torch.cuda.is_available():
            continue
        enc = encoder.to(dev)
        x1 = to_tensor(sents[0]).unsqueeze(0).to(dev)
        xb = torch.stack([to_tensor(s) for s in sents[:N_BATCH]]).to(dev)

        def one():
            with torch.no_grad():
                enc(x1)

        def many():
            with torch.no_grad():
                enc(xb)

        single = timeit(one, REPS_SINGLE)
        t0 = time.perf_counter()
        many()
        sync()
        batch = (time.perf_counter() - t0) / N_BATCH * 1000
        print(f"  学生 字节CNN  [{tag}]   单条 {single:7.2f} ms   批量 {batch:6.2f} ms/条")


if __name__ == "__main__":
    main()
