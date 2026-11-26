# -*- coding: utf-8 -*-
"""验证 query→memory 关联表：先问清楚的，再问模糊的

场景模拟真实使用：
  第 1 轮 —— 用户问得很具体（"杭州服务器的核数是什么，代号是什么"），命中率高，
             把 (query 向量, 命中的 mid) 记进关联表。
  第 2 轮 —— 用户问得模糊（"我们的杭州服务器配置怎么样，代号是什么"），
             同一 subject 有 6 条事实，自身信息不足以区分（天花板 1/6）。
             此时靠关联表：相似的完整 query 用过哪条，就预取那条的桶。

判据：模糊轮 hit@1 是否从 1/6 明显上升。
如果 query 相似度够高（>= q_thresh），机制就成立。

用法：python _exp_qtable.py [facts]
"""
import sys
import torch

from memory_core import LingmingMemory
from bench import make_facts
from _exp_capacity import code_of, encode


def run(n=60):
    # 同一批事实，两种问法
    specs = make_facts(n, qmode="new")   # 具体
    vague = make_facts(n, qmode="old")   # 模糊

    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n * 4), center=False,
                         write_scale=0.1, gate_bias=0.2, gate_temp=0.05,
                         read_top=3, per_bar=3)
    for f in specs:
        mem.add(f["text"], "user")

    def ev(facts, use_qtable):
        old = mem.q_embs if not use_qtable else None
        if not use_qtable:
            mem.q_embs = None
        h1 = 0
        for f in facts:
            items = mem.retrieve(f["query"], topK=3)
            codes = [code_of(it["text"]) for it in items]
            if f["code"] in codes and codes.index(f["code"]) == 0:
                h1 += 1
        if not use_qtable:
            mem.q_embs = old
        return h1 / len(facts)

    # 第 0 轮：模糊 query，无关联表（基线，应 ~1/6）
    base = ev(vague, use_qtable=False)
    # 第 1 轮：具体 query，学进关联表
    for f in specs:
        items = mem.retrieve(f["query"], topK=3)
        if not items:
            continue
        codes = [code_of(it["text"]) for it in items]
        if f["code"] in codes and codes.index(f["code"]) == 0:
            mem.add_query_feedback(mem.encode(f["query"]), [items[0]["mid"]])
    # 第 2 轮：模糊 query，开关联表
    after = ev(vague, use_qtable=True)

    # 顺便量一下"具体 query"与"模糊 query"的余弦相似度够不够门槛
    sims = []
    for a, b in zip(specs, vague):
        sims.append(float((mem.encode(a["query"]) * mem.encode(b["query"])).sum()))
    sims.sort()
    med = sims[len(sims) // 2]

    print(f"\n=== query→memory 关联表 (N={n}, q_thresh={mem.q_thresh}) ===")
    print(f"模糊 query hit@1（无表）  {base:6.3f}   <- 自身信息不足，天花板 1/6")
    print(f"模糊 query hit@1（有表）  {after:6.3f}")
    print(f"关联表条目数              {len(mem.q_mids):6d}")
    print(f"具体 vs 模糊 query 余弦   中位数 {med:.3f}  "
          f"范围 {sims[0]:.3f}~{sims[-1]:.3f}")
    print(f"\n判据：有表 > 无表 => 机制成立；余弦中位数需 >= {mem.q_thresh}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(n)
