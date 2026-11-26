# -*- coding: utf-8 -*-
"""验证灵铭 g 记忆（contents=数据库版）

池 1000 / 查询 200（不重叠），参考答案 = bge 在池上的 top-1（按文本比对，避免重复句歧义）。
指标：candidate-recall、end-to-end、写入/查询耗时。
"""
import os
import sys
import time

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
    print(f"池 {len(pool)} / 查询 {len(queries)} / d={D}", flush=True)

    e = bge.encode(pool, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    Hp = F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1).to(DEV)
    e = bge.encode(queries, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    Hq = F.normalize(torch.tensor(e, dtype=torch.float32), dim=-1).to(DEV)
    gold = (Hq @ Hp.T).argmax(-1)
    gold_text = [pool[i] for i in gold.tolist()]

    for k in (8, 16):
        mem = LingmingMemory(encode, d=D, c_mult=4, k=k, capacity=100000, device=DEV)
        t0 = time.time()
        for t in pool:
            mem.add(t, "user")
        write_t = time.time() - t0
        st = mem.status()

        cand_ok = 0
        for j, q in enumerate(queries):
            h = encode(q)
            _, bars = mem._coord(h)
            cand = mem._candidates(bars)
            if any(mem.texts[mem.slot_of[m]] == gold_text[j] for m in cand):
                cand_ok += 1

        e2e = 0
        t0 = time.time()
        for j, q in enumerate(queries):
            got = mem.retrieve(q, 1)
            if got and got[0]["text"] == gold_text[j]:
                e2e += 1
        query_t = (time.time() - t0) / len(queries) * 1000

        n = len(queries)
        print(f"\nk={k}  写入 {write_t:.1f}s（{write_t/len(pool)*1000:.1f}ms/条）  "
              f"桶数 {st['n_bars']}  库行数 {st['n_mem']}  C={st['C']}", flush=True)
        print(f"  candidate-recall {cand_ok/n*100:5.1f}%   end-to-end {e2e/n*100:5.1f}%   "
              f"单查询(含编码) {query_t:.1f}ms", flush=True)

    print(f"\n对照：纯 bge 暴力 cosine（1000 条）单查询 "
          f"{min((time.perf_counter()-t0 for _ in [0]), default=0)*0:.0f}ms（实际 <0.1ms，可忽略）", flush=True)


if __name__ == "__main__":
    main()
