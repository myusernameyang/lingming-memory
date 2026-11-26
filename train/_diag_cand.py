# -*- coding: utf-8 -*-
"""诊断：候选集是不是 hit@1 的天花板

量三件事，不改任何机制：
  1. 候选召回率 —— 正确答案的 mid 有没有进候选集（进不去就永远排不第一）
  2. 平均候选大小 —— 每个 query 的候选有多少条
  3. hit@1 / hit@3 —— 和 bench 一致

判据：如果候选召回率 ≈ hit@1，说明瓶颈在坐标桶，不在余弦排序。

用法：python _diag_cand.py [facts] [k_cand]
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


def run(n_facts=60, k_cand=16):
    facts = make_facts(n_facts)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=k_cand,
                         capacity=max(10000, n_facts * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)

    # 写入，同时记录每条事实的 mid -> code
    mid_code = {}
    for f in facts:
        mid = mem.add(f["text"], "user")
        mid_code[mid] = f["code"]

    # 段级：一条消息可能切多段，父 msg 也能反查。这里用 text 里的 code 对齐。
    row_code = {}
    for mid, code in mid_code.items():
        row_code[mid] = code
    # 补：按段文本判断（split 后每段都带 code 才准）
    for i, t in enumerate(mem.texts):
        c = code_of(t)
        if c:
            mid = None
            for m_, r_ in mem.slot_of.items():
                if r_ == i:
                    mid = m_
                    break
            if mid is not None:
                row_code[mid] = c

    in_cand = 0
    hit1 = hit3 = 0
    cand_sizes = []
    miss = []

    for f in facts:
        h = mem.encode(f["query"])
        _, bars = mem._coord(h)
        cand = mem._candidates(bars)
        mids = [m for m in sorted(cand) if m not in mem.deleted]
        cand_sizes.append(len(mids))

        target_in = any(row_code.get(m) == f["code"] for m in mids)
        if target_in:
            in_cand += 1
        else:
            miss.append(f["code"])

        items = mem.retrieve(f["query"], topK=3)
        codes = [code_of(it["text"]) for it in items]
        if f["code"] in codes:
            rank = codes.index(f["code"]) + 1
            if rank == 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1

    n = len(facts)
    avg_cand = sum(cand_sizes) / n
    print(f"\n=== 诊断 (facts={n}, k_cand={k_cand}, C={mem.C}) ===")
    print(f"候选召回率    {in_cand / n:7.3f}   <- 正确答案进候选的比例（hit@1 的天花板）")
    print(f"hit@1        {hit1 / n:7.3f}")
    print(f"hit@3        {hit3 / n:7.3f}")
    print(f"平均候选大小  {avg_cand:7.1f} 条")
    print(f"候选为空的比例 {sum(1 for c in cand_sizes if c == 0) / n:7.3f}")
    if miss:
        print(f"\n没进候选的 {len(miss)} 条：{miss[:12]}")
    print("\n结论：" + (
        "瓶颈在候选集（召回率≈hit@1），余弦排序不是问题"
        if abs(in_cand / n - hit1 / n) < 0.15 else
        "候选召回率明显高于 hit@1，瓶颈在排序/相似度"
    ))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    run(n, k)
