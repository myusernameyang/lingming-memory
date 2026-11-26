# -*- coding: utf-8 -*-
"""倒排（关键词/标识符）那一路：向量对随机标识符不敏感，倒排精确

场景：真实排障记忆里大量是标识符——错误码、驱动版本、寄存器名、符号名
（rebar、CE/UE、535、ERR-7f3a9c）。句向量对这类随机串几乎无区分力。

测试集：N 条事实，每条带一个唯一随机错误码，描述各不相同。
query 只给错误码（不给描述）—— 这是在考"能不能按标识符精确命中"。

三路对比：
  vector    —— 纯坐标桶 + 余弦
  inverted  —— 只用标识符倒排
  hybrid    —— 倒排命中的并进候选，再按余弦排（预取）

用法：python _exp_inverted.py [n]
"""
import sys
import re
import random
import string
import torch

from memory_core import LingmingMemory
from _exp_capacity import encode

SUBSYS = ["NCCL", "CUDA", "驱动", "算子", "集合通信", "内存", "调度", "日志"]
PROB = ["超时", "宕机", "精度下降", "hang 住", "OOM", "校验失败",
        "版本不兼容", "初始化失败", "性能回退", "丢包"]


def rand_code(rng):
    return "ERR-" + "".join(rng.choice(string.ascii_lowercase + string.digits)
                            for _ in range(6))


def make_facts(n, seed=7):
    rng = random.Random(seed)
    facts = []
    for i in range(n):
        code = rand_code(rng)
        s = SUBSYS[i % len(SUBSYS)]
        p = PROB[(i // len(SUBSYS)) % len(PROB)]
        facts.append({
            "code": code,
            "text": f"{s}出现{p}，错误码 {code}，已在集群 A{i:04d} 复现。",
            "query": f"{code} 是什么问题",
        })
    return facts


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*|\d+")


def tokens(text):
    out = set()
    for t in TOKEN_RE.findall(text.lower()):
        out.add(t)
    # 错误码整体也算一个 token（上面正则已覆盖 err-xxx）
    return out


def build_inverted(texts):
    inv = {}
    for i, t in enumerate(texts):
        for tok in tokens(t):
            inv.setdefault(tok, set()).add(i)
    return inv


def run(n=300):
    facts = make_facts(n)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n * 4), center=False,
                         write_scale=0.1, gate_bias=0.2, gate_temp=0.05,
                         read_top=3, per_bar=3)
    for f in facts:
        mem.add(f["text"], "user")

    # 段文本 -> 行号（slot_of 反查）
    slot_to_mid = {}
    for mid, row in mem.slot_of.items():
        slot_to_mid[row] = mid
    texts = [mem.texts[mem.slot_of[m]] for m in sorted(mem.slot_of)]
    inv = build_inverted(texts)
    mids_sorted = sorted(mem.slot_of)

    def hit_at1(text_list):
        return text_list

    res = {}
    for mode in ("vector", "inverted", "hybrid"):
        h1 = 0
        cand_sizes = []
        for f in facts:
            h = mem.encode(f["query"])
            # 倒排候选
            toks = tokens(f["query"])
            inv_rows = set()
            for tok in toks:
                inv_rows |= inv.get(tok, set())
            inv_rows = {slot_to_mid[r] for r in inv_rows if r in slot_to_mid}

            if mode == "inverted":
                if not inv_rows:
                    cand_sizes.append(0)
                    continue
                cand = sorted(inv_rows)
            else:
                _, bars = mem._coord(h)
                if mode == "hybrid" and inv_rows:
                    for mid in inv_rows:
                        bars = set(bars) | set(mem.bars.get(mid, ()))
                cand = [m for m in sorted(mem._candidates(bars)) if m not in mem.deleted]
            cand_sizes.append(len(cand))
            if not cand:
                continue
            rows = torch.tensor([mem.slot_of[m] for m in cand], device=mem.dev)
            sim = mem.contents[rows] @ h
            # 倒排那一路要有优先权：标识符全中 > 余弦
            if mode in ("inverted", "hybrid"):
                rows_inv = set()
                for m in cand:
                    if m in inv_rows:
                        rows_inv.add(m)
                if rows_inv:
                    boost = torch.tensor(
                        [0.5 if m in rows_inv else 0.0 for m in cand], device=mem.dev)
                    sim = sim + boost
            top = sim.argsort(descending=True)[:1]
            mid = cand[top[0].item()]
            if f["code"] in mem.texts[mem.slot_of[mid]]:
                h1 += 1
        res[mode] = (h1 / n, sum(cand_sizes) / len(cand_sizes))

    print(f"\n=== 倒排 vs 向量 (N={n}, query 只给错误码) ===")
    print(f"{'路径':<10}{'hit@1':>8}{'平均候选':>10}")
    for k, (a, b) in res.items():
        print(f"{k:<10}{a:>8.3f}{b:>10.1f}")
    print("\n判据：inverted/hybrid 明显高于 vector => 标识符那一路值得做")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(n)
