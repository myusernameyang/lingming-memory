# -*- coding: utf-8 -*-
"""对照：bench 的 query 缺 attr，导致 6 选 1 的天花板

同一批事实，两种 query：
  A. 旧 query（只含 subject，同 subject 有 6 条，天花板 1/6）
  B. 新 query（含 attr 关键词，唯一指向一条）

如果 B 的 hit@1 远高于 A，说明 0.167 是测试集造成的，不是系统差。
顺便跑一次 C：旧 query + 双 query 并集（前瞻打分原型），看能否超过 1/6。

用法：python _diag_cap.py [facts]
"""
import sys
import re
import torch
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory
from bench import make_facts, SUBJECTS, ATTRS

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


def queries_old(f):
    return f["query"]


def queries_new(f):
    # 从事实文本里取 attr 名，拼进 query
    for attr, _ in ATTRS:
        if attr in f["text"]:
            subj = f["text"].split("的")[0]
            return f"{subj}的{attr}是什么，代号是什么"
    return f["query"]


def evaluate(mem, facts, qfn):
    hit1 = hit3 = 0
    for f in facts:
        items = mem.retrieve(qfn(f), topK=3)
        codes = [code_of(it["text"]) for it in items]
        if f["code"] in codes:
            r = codes.index(f["code"]) + 1
            if r == 1:
                hit1 += 1
            if r <= 3:
                hit3 += 1
    n = len(facts)
    return hit1 / n, hit3 / n


def run(n_facts=60):
    facts = make_facts(n_facts)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n_facts * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")

    print(f"\n=== 天花板对照 (facts={n_facts}, C={mem.C}) ===")
    print(f"同 subject 的事实条数 = {n_facts // len(SUBJECTS)}  -> 每 query 候选里有这么多条无法区分\n")

    h1, h3 = evaluate(mem, facts, queries_old)
    print(f"A. 旧 query（只含 subject）   hit@1 {h1:6.3f}   hit@3 {h3:6.3f}")

    h1, h3 = evaluate(mem, facts, queries_new)
    print(f"B. 新 query（含 attr）        hit@1 {h1:6.3f}   hit@3 {h3:6.3f}")

    print("\n判据：B 明显高于 A => 0.167 是测试集造成的，不是系统差")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(n)
