# -*- coding: utf-8 -*-
"""原型：双 query 并集（伪相关反馈扩张候选集）

问题：new 集上 hit@1 = 0.967，剩 2 条（KX025/KX033）是**根本没进坐标桶**。
boost/余弦排序救不了没进候选的。

做法（不改 memory_core，包装一层）：
  1. bars1 = coord(h(q))                     原始 query 的桶
  2. cand1 = 并集(bars1)，取 sim 最高的那条文本 t1
  3. q2 = q + " " + t1，bars2 = coord(h(q2))  用"猜出来的上下文"再算一组桶
  4. 候选 = 并集(bars1, bars2)；**排序仍用原始 h**（不改 query 语义）
  5. 可选多轮

判据：hit@1 是否从 0.967 上去；同时看平均候选大小涨了多少（噪音代价）。

用法：python _exp_prf.py [facts] [rounds]
"""
import sys
import re
import torch
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory
from bench import make_facts

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"

_enc = None


def encode(text):
    global _enc
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(e[0], dtype=torch.float32, device=DEV)


def code_of(text):
    m = re.search(r"KX\d{3}", text)
    return m.group(0) if m else ""


def _rank(mem, h, bars, topK, scope=""):
    cand = mem._candidates(bars)
    if not cand:
        return [], 0
    mids = [m for m in sorted(cand) if m not in mem.deleted]
    if not mids:
        return [], 0
    rows = torch.tensor([mem.slot_of[m] for m in mids], device=mem.dev)
    sim = mem.contents[rows] @ h
    order = sim.argsort(descending=True)[:topK]
    out = [mem.texts[rows[i].item()] for i in order]
    return out, len(mids)


def retrieve_prf(mem, query, topK=3, rounds=1):
    """双 query 并集。排序用原始 h，只用扩张的桶扩候选。"""
    h = mem.encode(query)
    _, bars = mem._coord(h)
    bars = set(bars)
    cand_size = 0
    for _ in range(rounds):
        items, cand_size = _rank(mem, h, bars, 1)
        if not items:
            break
        t1 = items[0]
        h2 = mem.encode(query + " " + t1)
        _, bars2 = mem._coord(h2)
        bars = bars | set(bars2)
    out, cand_size = _rank(mem, h, bars, topK)
    return out, cand_size


def evaluate(mem, facts, prf_rounds):
    hit1 = hit3 = 0
    sizes = []
    for f in facts:
        if prf_rounds == 0:
            items = mem.retrieve(f["query"], topK=3)
            codes = [code_of(it["text"]) for it in items]
            sizes.append(0)
        else:
            texts, sz = retrieve_prf(mem, f["query"], topK=3, rounds=prf_rounds)
            codes = [code_of(t) for t in texts]
            sizes.append(sz)
        if f["code"] in codes:
            r = codes.index(f["code"]) + 1
            if r == 1:
                hit1 += 1
            if r <= 3:
                hit3 += 1
    n = len(facts)
    avg = (sum(sizes) / n) if sizes and any(sizes) else 0.0
    return hit1 / n, hit3 / n, avg


def run(n_facts=60):
    facts = make_facts(n_facts, qmode="new")
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n_facts * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")

    print(f"\n=== 双 query 并集 (facts={n_facts}, C={mem.C}) ===")
    print(f"{'配置':<16}{'hit@1':>8}{'hit@3':>8}{'平均候选':>10}")
    for r in (0, 1, 2):
        label = "baseline" if r == 0 else f"PRF x{r}"
        h1, h3, avg = evaluate(mem, facts, r)
        print(f"{label:<16}{h1:8.3f}{h3:8.3f}{avg:10.1f}")

    # 哪几条还是错的
    miss = []
    for f in facts:
        texts, _ = retrieve_prf(mem, f["query"], topK=3, rounds=1)
        if f["code"] not in [code_of(t) for t in texts]:
            miss.append(f["code"])
    print(f"\nPRF x1 仍未召回：{miss}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(n)
