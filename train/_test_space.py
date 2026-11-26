# -*- coding: utf-8 -*-
"""内容空间对齐实验：数据库存什么、查询用什么

灵元子里写入的是 w = normalize(h + content)（输入+召回），读出来是 content。
落到"数据库检索"上，可选三种对齐方式：
  1. 库存 w，查询用 h       （当前实现）
  2. 库存 w，查询用 w_q     （对称：输入+召回，两边同空间）
  3. 库存 h，查询用 h       （纯语义库）
比较三者 end-to-end 命中率。
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from memory_core import LingmingMemory  # noqa: E402

CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_POOL, N_QUERY = 1000, 200


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


def main():
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    D = bge.get_embedding_dimension()

    def encode(text):
        e = bge.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return torch.tensor(e[0], dtype=torch.float32, device=DEV)

    sents = split_sents(open(CORPUS, "rb").read()[20_000_000:40_000_000])
    pool, queries = sents[:N_POOL], sents[N_POOL:N_POOL + N_QUERY]

    e = bge.encode(pool, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    Hp = F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1).to(DEV)
    e = bge.encode(queries, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    Hq = F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1).to(DEV)
    gold = (Hq @ Hp.T).argmax(-1)
    gold_text = [pool[i] for i in gold.tolist()]

    mem = LingmingMemory(encode, d=D, c_mult=4, k=8, capacity=100000, device=DEV)
    for t in pool:
        mem.add(t, "user")
    print(f"已写入 {mem.n} 条，桶 {len(mem.index)} 个，C={mem.C}\n", flush=True)

    stats = {name: 0 for name in ("库存w/查h", "库存w/查w_q", "库存h/查h")}
    for j, q in enumerate(queries):
        h = encode(q)
        _, bars = mem._coord(h)
        cand = mem._candidates(bars)
        if not cand:
            continue
        mids = sorted(cand)
        rows = torch.tensor([mem.slot_of[m] for m in mids], device=DEV)

        content_q = mem._read(h, bars)
        w_q = F.normalize(h + content_q, dim=-1)

        def hit(space, query_vec):
            sim = space[rows] @ query_vec
            row = rows[sim.argmax()].item()
            return mem.texts[row] == gold_text[j]

        if hit(mem.contents, h):
            stats["库存w/查h"] += 1
        if hit(mem.contents, w_q):
            stats["库存w/查w_q"] += 1
        if hit(Hp, h):
            stats["库存h/查h"] += 1

    n = len(queries)
    for name, v in stats.items():
        print(f"  {name:14s} end-to-end {v/n*100:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
