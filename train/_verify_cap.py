# -*- coding: utf-8 -*-
"""验证 memory_core 里的 per_bar 限流是否生效

量的是 _cap_candidates 之后的候选数（不是限流前），
并对比 per_bar=0 / 3 / 6 的 hit@1。

用法：python _verify_cap.py
"""
import torch

from memory_core import LingmingMemory
from _exp_capacity import make_facts, code_of, encode


def run(n=1000):
    facts = make_facts(n)
    print(f"\n=== per_bar 验证 (N={n}, C=2048, k=16) ===")
    print(f"{'per_bar':>8}{'hit@1':>9}{'候选(限流后)':>14}{'候选(限流前)':>14}")
    for pb in (0, 3, 6):
        mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                             capacity=max(10000, n * 4), center=False,
                             write_scale=0.1, gate_bias=0.2, gate_temp=0.05,
                             read_top=3, per_bar=pb)
        for f in facts:
            mem.add(f["text"], "user")

        hit1 = 0
        tot_after = tot_before = 0
        for f in facts:
            h = mem.encode(f["query"])
            _, bars = mem._coord(h)
            cand = mem._candidates(bars)
            mids = [m for m in sorted(cand) if m not in mem.deleted]
            rows = torch.tensor([mem.slot_of[m] for m in mids], device=mem.dev)
            sim = mem.contents[rows] @ h
            tot_before += len(mids)
            mids2, _ = mem._cap_candidates(mids, sim, bars)
            tot_after += len(mids2)

            items = mem.retrieve(f["query"], topK=3)
            codes = [code_of(it["text"]) for it in items]
            if f["code"] in codes and codes.index(f["code"]) == 0:
                hit1 += 1
        print(f"{pb:>8}{hit1/n:>9.3f}{tot_after/n:>14.0f}{tot_before/n:>14.0f}")


if __name__ == "__main__":
    run(1000)
