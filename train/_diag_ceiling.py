# -*- coding: utf-8 -*-
"""天花板诊断（修正版）：用真实语料的纠正对做评测集，带 bge 天花板

评测集：从语料中「下一句以 不对/错了/应该是... 开头」挖出的强纠正对。
用法：记忆侧存"原句"，查询侧用"纠正句"（或反之），测 recall@1。

对比四档：
  A. 教师 bge float cosine（天花板，无量化）
  B. 学生特征 float cosine（无坐标量化）
  C. 学生 + 坐标桶投票（k 扫描，不细排）
  D. 学生 + 桶投票 + cosine 细排（插件当前管线）
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
MARKERS = ["不对", "错了", "应该是", "我说的是", "不是这样", "纠正一下", "其实是", "搞错了"]


def to_tensor(text):
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    return torch.tensor((list(text) + [0] * L)[:L], dtype=torch.long)


def split_sents(raw):
    delims3 = [b"\xe3\x80\x82", b"\xef\xbc\x9f", b"\xef\xbc\x81"]
    pos = [0]
    for i in range(len(raw) - 3):
        if raw[i:i + 3] in delims3 or raw[i:i + 1] == b"\n":
            pos.append(i + 3)
    out = []
    for j in range(len(pos) - 1):
        s = raw[pos[j]:pos[j + 1]].strip()
        if 8 <= len(s) <= 64:
            out.append(s)
    return out


def mine_correction(raw):
    sents = split_sents(raw)
    pairs = []
    for i in range(len(sents) - 1):
        bt = sents[i + 1].decode("utf-8", errors="ignore")
        if any(bt.startswith(m) for m in MARKERS):
            pairs.append((sents[i], sents[i + 1]))
    return pairs


def teacher_feats(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    e = m.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    return F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1)


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


def student_feats(encoder, texts):
    with torch.no_grad():
        X = torch.stack([to_tensor(t) for t in texts])
        return F.normalize(encoder(X), dim=-1)


def top1_self(M, Q):
    pred = (Q @ M.T).argmax(-1)
    return (pred == torch.arange(len(pred))).float().mean().item()


def main():
    raw_train = open(CORPUS, "rb").read(20_000_000)
    raw_test = open(CORPUS, "rb").read()[20_000_000:40_000_000]
    test_pairs = mine_correction(raw_test)
    print(f"评测集（留存纠正对）：{len(test_pairs)} 对", flush=True)
    mems = [a.decode("utf-8", "ignore") for a, _ in test_pairs]
    qs = [b.decode("utf-8", "ignore") for _, b in test_pairs]

    Ze = teacher_feats(mems); Zq = teacher_feats(qs)
    print(f"  A. 教师 bge float cosine（天花板）     recall@1 {top1_self(Ze, Zq)*100:5.1f}%", flush=True)

    encoder, W_coord = load_student()
    Se = student_feats(encoder, mems); Sq = student_feats(encoder, qs)
    print(f"  B. 学生特征 float cosine（无坐标）     recall@1 {top1_self(Se, Sq)*100:5.1f}%", flush=True)

    for k in (8, 16):
        mem = LingmingMemory(encoder, W_coord, k=k, L=L, capacity=100000)
        for t in mems:
            mem.add(t, "user")
        text2mid = {v["text"]: mid for mid, v in mem.store_.items()}
        c1 = d1 = 0
        for mem_text, query in zip(mems, qs):
            target = text2mid[mem_text]
            votes = {}
            for bar in mem.coord(query):
                for mid in mem.index.get(bar, []):
                    votes[mid] = votes.get(mid, 0) + 1
            if votes and max(votes.items(), key=lambda kv: kv[1])[0] == target:
                c1 += 1
            got = [text2mid[it["text"]] for it in mem.retrieve(query, 1)]
            if got and got[0] == target:
                d1 += 1
        n = len(test_pairs)
        print(f"  C. 桶投票 k={k:<2d}（不细排）         recall@1 {c1/n*100:5.1f}%", flush=True)
        print(f"  D. 桶投票 k={k:<2d} + cosine 细排     recall@1 {d1/n*100:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
