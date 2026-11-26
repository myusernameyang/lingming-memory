# -*- coding: utf-8 -*-
"""C5 容量曲线：条目数从 60 加到 2000，hit@1 怎么变

测试集必须保证每条事实语义唯一，否则衰减测的是测试集歧义而不是容量。
做法：subject = 前缀 × 城市（500 种），(subject, attr) 在全集内唯一。

两条曲线：baseline / PRF x1（双 query 并集）
右轴：平均候选大小

用法：python _exp_capacity.py
输出：capacity_curve.png
"""
import time
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"

PREFIX = ["服务器", "数据库", "缓存", "备份", "监控", "网关", "队列",
          "日志", "镜像仓库", "构建机", "训练机", "推理机", "调度器",
          "消息总线", "对象存储", "配置中心", "注册中心", "链路追踪",
          "特征库", "标注平台"]
CITY = ["杭州", "北京", "上海", "深圳", "广州", "成都", "武汉", "西安",
        "南京", "苏州", "郑州", "长沙", "重庆", "天津", "青岛", "厦门",
        "合肥", "福州", "济南", "昆明", "沈阳", "大连", "无锡", "宁波",
        "东莞"]
ATTRS = [("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T NVMe"),
         ("带宽", "500Mbps"), ("系统", "Ubuntu 22.04"), ("端口", "8443")]

_enc = None


def encode(text):
    global _enc
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(e[0], dtype=torch.float32, device=DEV)


def code_of(text):
    import re
    m = re.search(r"KX(\d{3,4})", text)
    return m.group(0) if m else ""


def make_facts(n):
    facts = []
    for i in range(n):
        p = PREFIX[i % len(PREFIX)]
        c = CITY[(i // len(PREFIX)) % len(CITY)]
        a, v = ATTRS[(i // (len(PREFIX) * len(CITY))) % len(ATTRS)]
        subj = f"{c}{p}"
        code = f"KX{i:04d}"
        facts.append({
            "code": code,
            "text": f"{subj}的{a}是 {v}，代号 {code}。",
            "query": f"{subj}的{a}是什么，代号是什么",
        })
    return facts


def _rank(mem, h, bars, topK):
    cand = mem._candidates(bars)
    if not cand:
        return [], 0
    mids = [m for m in sorted(cand) if m not in mem.deleted]
    if not mids:
        return [], 0
    rows = torch.tensor([mem.slot_of[m] for m in mids], device=mem.dev)
    sim = mem.contents[rows] @ h
    order = sim.argsort(descending=True)[:topK]
    return [mem.texts[rows[i].item()] for i in order], len(mids)


def retrieve_prf(mem, query, topK=3, rounds=1):
    h = mem.encode(query)
    _, b0 = mem._coord(h)
    bars = set(b0)
    for _ in range(rounds):
        items, _ = _rank(mem, h, bars, 1)
        if not items:
            break
        h2 = mem.encode(query + " " + items[0])
        _, b2 = mem._coord(h2)
        bars = bars | set(b2)
    return _rank(mem, h, bars, topK)


def run(n):
    facts = make_facts(n)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")

    res = {"n": n}
    for mode in ("base", "prf"):
        hit1 = hit3 = 0
        sizes = []
        for f in facts:
            if mode == "base":
                items = mem.retrieve(f["query"], topK=3)
                texts = [it["text"] for it in items]
                sizes.append(0)
            else:
                texts, sz = retrieve_prf(mem, f["query"], topK=3, rounds=1)
                sizes.append(sz)
            codes = [code_of(t) for t in texts]
            if f["code"] in codes:
                r = codes.index(f["code"]) + 1
                if r == 1:
                    hit1 += 1
                if r <= 3:
                    hit3 += 1
        res[mode + "_hit1"] = hit1 / n
        res[mode + "_hit3"] = hit3 / n
        if mode == "prf":
            res["avg_cand"] = sum(sizes) / n
    print(f"N={n:5d}  base hit@1={res['base_hit1']:.3f}  "
          f"prf hit@1={res['prf_hit1']:.3f}  cand={res['avg_cand']:.0f}", flush=True)
    return res


if __name__ == "__main__":
    NS = [60, 120, 250, 500, 1000, 2000]
    rows = []
    t0 = time.time()
    for n in NS:
        rows.append(run(n))
    print(f"\n总用时 {time.time()-t0:.0f}s", flush=True)

    import json
    with open("capacity_curve.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["n"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ns, [r["base_hit1"] for r in rows], "o-", label="baseline hit@1")
        ax1.plot(ns, [r["prf_hit1"] for r in rows], "s-", label="PRF x1 hit@1")
        ax1.plot(ns, [r["base_hit3"] for r in rows], "o--", alpha=.45, label="baseline hit@3")
        ax1.set_xscale("log")
        ax1.set_xlabel("memory items (N, log scale)")
        ax1.set_ylabel("hit rate")
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=.3)

        ax2 = ax1.twinx()
        ax2.plot(ns, [r["avg_cand"] for r in rows], "^:", color="gray",
                 label="avg candidate size")
        ax2.set_ylabel("avg candidate size")

        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower left", fontsize=8)
        plt.title("Lingming memory: capacity curve (C=2048, k=16)")
        plt.tight_layout()
        plt.savefig("capacity_curve.png", dpi=150)
        print("saved capacity_curve.png")
    except Exception as e:
        print(f"plot failed: {e}")
