# -*- coding: utf-8 -*-
"""E5 实验：用 g 做桶级粗筛，看候选集能不能不随 N 线性膨胀

背景：容量曲线显示 hit@1 不降，但候选数 ≈ N（N=2000 时 1809/2000），
      坐标桶索引在 N 大时退化成全表扫描。
原因：query 直接取 top-k 坐标桶，热桶撑满并集。

E5 做法：
  现在的桶分 = query 的坐标值（谁被点亮就取谁）
  E5 的桶分 = g.T @ h  —— 该桶累计写入的内容与 query 有多像
  按 E5 分数取同样数量的桶，再取并集做候选，最后仍用 contents 细排。

三条曲线：
  baseline  —— 现在的方式
  PRF x1    —— 双 query 并集（扩候选）
  E5        —— g 粗筛选桶
看：hit@1 有没有掉、候选数有没有被压住。

用法：python _exp_e5.py
输出：e5_curve.png
"""
import time
import torch
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory
from _exp_capacity import make_facts, code_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"
_enc = None


def encode(text):
    global _enc
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(e[0], dtype=torch.float32, device=DEV)


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


def bars_baseline(mem, h):
    _, bars = mem._coord(h)
    return set(bars)


def bars_prf(mem, h, query, rounds=1):
    bars = bars_baseline(mem, h)
    for _ in range(rounds):
        items, _ = _rank(mem, h, bars, 1)
        if not items:
            break
        h2 = mem.encode(query + " " + items[0])
        _, b2 = mem._coord(h2)
        bars = bars | set(b2)
    return bars


def bars_e5(mem, h, k_bars=16):
    """E5：桶分 = g.T @ h（该桶累计写入的内容与 query 的相似度）"""
    if mem.g is None:
        return bars_baseline(mem, h)
    scores = mem.g.t() @ h          # (C,)
    top = scores.argsort(descending=True)[:k_bars]
    return set(top.tolist())


def run(n):
    facts = make_facts(n)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")

    res = {"n": n}
    for mode in ("base", "prf", "e5"):
        hit1 = hit3 = 0
        tot_cand = 0
        for f in facts:
            h = mem.encode(f["query"])
            if mode == "base":
                bars = bars_baseline(mem, h)
            elif mode == "prf":
                bars = bars_prf(mem, h, f["query"], rounds=1)
            else:
                bars = bars_e5(mem, h, k_bars=16)
            texts, csz = _rank(mem, h, bars, 3)
            tot_cand += csz
            codes = [code_of(t) for t in texts]
            if f["code"] in codes:
                r = codes.index(f["code"]) + 1
                if r == 1:
                    hit1 += 1
                if r <= 3:
                    hit3 += 1
        res[mode + "_hit1"] = hit1 / n
        res[mode + "_hit3"] = hit3 / n
        res[mode + "_cand"] = tot_cand / n
    print(f"N={n:5d}  base h1={res['base_hit1']:.3f} cand={res['base_cand']:.0f} | "
          f"prf h1={res['prf_hit1']:.3f} cand={res['prf_cand']:.0f} | "
          f"e5 h1={res['e5_hit1']:.3f} cand={res['e5_cand']:.0f}", flush=True)
    return res


if __name__ == "__main__":
    NS = [60, 250, 1000, 2000]
    rows = []
    t0 = time.time()
    for n in NS:
        rows.append(run(n))
    print(f"\n总用时 {time.time()-t0:.0f}s", flush=True)

    import json
    with open("e5_curve.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for m, style in (("base", "o-"), ("prf", "s-"), ("e5", "^-")):
        ax1.plot(ns, [r[m + "_hit1"] for r in rows], style, label=f"{m} hit@1")
    ax1.set_xscale("log")
    ax1.set_xlabel("memory items (N, log)")
    ax1.set_ylabel("hit@1")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=.3)
    ax1.legend()
    ax1.set_title("hit@1 vs N")

    for m, style in (("base", "o-"), ("prf", "s--"), ("e5", "^-")):
        ax2.plot(ns, [r[m + "_cand"] for r in rows], style, label=f"{m} avg candidate")
    ax2.plot(ns, ns, "k:", alpha=.4, label="candidate = N (全表)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("memory items (N, log)")
    ax2.set_ylabel("avg candidate size (log)")
    ax2.grid(alpha=.3)
    ax2.legend()
    ax2.set_title("candidate size vs N")

    plt.suptitle("E5: g-based bucket prefilter (C=2048, k_bars=16)")
    plt.tight_layout()
    plt.savefig("e5_curve.png", dpi=150)
    print("saved e5_curve.png")
