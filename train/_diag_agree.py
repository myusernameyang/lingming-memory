# -*- coding: utf-8 -*-
"""教师一致率评测：廉价管线能复现 bge 的排序到多少

做法（不需要人工标注）：
  池子 P = 1000 条语料句
  查询 Q = 200 条**不在池子里**的句
  参考答案 = bge 在 P 上的 top-1
  指标 = 各管线 top-1 与参考答案的一致率

这直接回答：坐标量化 + 字节 CNN 这套廉价管线，相对强编码器损失了多少。
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from memory_core import CNNEncoder, LingmingMemory  # noqa: E402

D, C, L = 128, 512, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"
N_POOL, N_QUERY = 1000, 200


def to_tensor(text):
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    return torch.tensor((list(text) + [0] * L)[:L], dtype=torch.long)


def split_sents(raw):
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
    return out


def teacher(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    e = m.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    return F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1)


def student(encoder, texts):
    with torch.no_grad():
        X = torch.stack([to_tensor(t) for t in texts])
        return F.normalize(encoder(X), dim=-1)


def load_student():
    encoder = CNNEncoder(D)
    W_coord = nn.Linear(D, C, bias=False)
    st = torch.load(os.path.join(BASE, "coord.pt"), map_location="cpu")
    w = st.get("W_coord", st.get("W", None))
    if isinstance(w, dict):
        encoder.load_state_dict(st.get("encoder", {}), strict=False)
        W_coord.load_state_dict(w, strict=False)
    encoder.eval(); W_coord.eval()
    return encoder, W_coord


def main():
    sents = split_sents(open(CORPUS, "rb").read()[20_000_000:40_000_000])
    pool, queries = sents[:N_POOL], sents[N_POOL:N_POOL + N_QUERY]
    print(f"池 {len(pool)} / 查询 {len(queries)}", flush=True)

    Pe, Qe = teacher(pool), teacher(queries)
    gold = (Qe @ Pe.T).argmax(-1)
    print(f"  A. 教师 bge（参考答案本身）            一致率 100.0%（基准）", flush=True)

    encoder, W_coord = load_student()
    Sp, Sq = student(encoder, pool), student(encoder, queries)
    agree_b = ((Sq @ Sp.T).argmax(-1) == gold).float().mean().item()
    print(f"  B. 学生特征 float cosine（无坐标）    一致率 {agree_b*100:5.1f}%", flush=True)

    for k in (8, 16):
        mem = LingmingMemory(encoder, W_coord, k=k, L=L, capacity=100000)
        for t in pool:
            mem.add(t, "user")
        text2mid = {v["text"]: mid for mid, v in mem.store_.items()}
        c_ok = d_ok = 0
        for q, g in zip(queries, gold.tolist()):
            gtext = pool[g]
            votes = {}
            for bar in mem.coord(q):
                for mid in mem.index.get(bar, []):
                    votes[mid] = votes.get(mid, 0) + 1
            if votes:
                best = max(votes.items(), key=lambda kv: kv[1])[0]
                if mem.store_[best]["text"] == gtext:
                    c_ok += 1
            got = mem.retrieve(q, 1)
            if got and got[0]["text"] == gtext:
                d_ok += 1
        n = len(queries)
        print(f"  C. 桶投票 k={k:<2d}（不细排）          一致率 {c_ok/n*100:5.1f}%", flush=True)
        print(f"  D. 桶投票 k={k:<2d} + cosine 细排      一致率 {d_ok/n*100:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
