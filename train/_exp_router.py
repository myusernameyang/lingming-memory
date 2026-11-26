# -*- coding: utf-8 -*-
"""分流效果：同一个库里混两种查询，看分流能不能同时照顾两边

库 = 一半语义事实（杭州服务器的核数…）+ 一半标识符事实（NCCL 超时，错误码 ERR-xxx）

两种查询：
  语义型  "杭州服务器的核数是什么，代号是什么"     -> 走向量 + per_bar
  标识符型 "ERR-7f3a9c 是什么问题"                -> 走倒排

策略：
  all_vector  —— 全走向量（现状）
  router      —— 有标识符走倒排，没有走向量（分流）
  all_invert  —— 有词就全走倒排（拿来做对照，语义型会废）

判据：router 在两类查询上都不低于各自的最优单路。

用法：python _exp_router.py [n_per_type]
"""
import sys
import re
import random
import string
import torch

from memory_core import LingmingMemory
from _exp_capacity import encode
from _exp_inverted import tokens, build_inverted, PROB, SUBSYS

PREFIX = ["杭州", "北京", "上海", "深圳", "广州", "成都", "武汉", "西安",
          "南京", "苏州", "郑州", "长沙", "重庆", "天津", "青岛"]
ROLE = ["服务器", "数据库", "缓存", "备份", "监控", "网关", "队列", "日志"]
ATTRS = [("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T NVMe"),
         ("带宽", "500Mbps"), ("系统", "Ubuntu 22.04"), ("端口", "8443")]

IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]{2,}|\d{3,}")


def has_ident(q):
    return bool(IDENT_RE.search(q))


def make_mixed(n):
    """n 条语义事实 + n 条标识符事实"""
    rng = random.Random(11)
    facts = []
    for i in range(n):
        p = PREFIX[i % len(PREFIX)]
        r = ROLE[(i // len(PREFIX)) % len(ROLE)]
        a, v = ATTRS[(i // (len(PREFIX) * len(ROLE))) % len(ATTRS)]
        code = f"QX{i:04d}"
        subj = f"{p}{r}"
        facts.append({
            "kind": "sem",
            "code": code,
            "text": f"{subj}的{a}是 {v}，代号 {code}。",
            "query": f"{subj}的{a}是什么，代号是什么",
        })
    for i in range(n):
        code = "ERR-" + "".join(rng.choice(string.ascii_lowercase + string.digits)
                                for _ in range(6))
        s = SUBSYS[i % len(SUBSYS)]
        p = PROB[(i // len(SUBSYS)) % len(PROB)]
        facts.append({
            "kind": "ident",
            "code": code,
            "text": f"{s}出现{p}，错误码 {code}，已在集群 B{i:04d} 复现。",
            "query": f"{code} 是什么问题",
        })
    return facts


def run(n=120):
    facts = make_mixed(n)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(20000, len(facts) * 4), center=False,
                         write_scale=0.1, gate_bias=0.2, gate_temp=0.05,
                         read_top=3, per_bar=3)
    for f in facts:
        mem.add(f["text"], "user")

    texts = [mem.texts[mem.slot_of[m]] for m in sorted(mem.slot_of)]
    inv = build_inverted(texts)
    slot_to_mid = {r: m for m, r in mem.slot_of.items()}

    def vector_hits(f, topk=1):
        h = mem.encode(f["query"])
        _, bars = mem._coord(h)
        cand = [m for m in sorted(mem._candidates(bars)) if m not in mem.deleted]
        if not cand:
            return [], 0
        rows = torch.tensor([mem.slot_of[m] for m in cand], device=mem.dev)
        sim = mem.contents[rows] @ h
        mids, sim = mem._cap_candidates(cand, sim, bars)
        order = sim.argsort(descending=True)[:topk]
        return [mids[i.item()] for i in order], len(mids)

    def inverted_hits(f, topk=1):
        out = set()
        for tok in tokens(f["query"]):
            for r in inv.get(tok, ()):
                if r in slot_to_mid:
                    out.add(slot_to_mid[r])
        if not out:
            return [], 0
        return sorted(out)[:topk], len(out)

    print(f"\n=== 分流 (库={len(facts)} 条: {n} 语义 + {n} 标识符) ===")
    print(f"{'策略':<12}{'语义 hit@1':>12}{'标识符 hit@1':>14}{'总 hit@1':>10}{'平均候选':>10}")

    strategies = {
        "all_vector": lambda f: vector_hits(f),
        "router": lambda f: inverted_hits(f) if has_ident(f["query"]) else vector_hits(f),
        "all_invert_first": lambda f: (inverted_hits(f) if inverted_hits(f)[0] else vector_hits(f)),
    }
    for name, fn in strategies.items():
        sem = ident = 0
        ns = ni = 0
        tot = 0
        totc = 0
        for f in facts:
            mids, csz = fn(f)
            totc += csz
            ok = bool(mids) and (f["code"] in mem.texts[mem.slot_of[mids[0]]])
            if f["kind"] == "sem":
                ns += 1
                sem += ok
            else:
                ni += 1
                ident += ok
            tot += ok
        print(f"{name:<12}{sem/ns:>12.3f}{ident/ni:>14.3f}"
              f"{tot/len(facts):>10.3f}{totc/len(facts):>10.1f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run(n)
