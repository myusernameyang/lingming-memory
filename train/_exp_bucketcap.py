# -*- coding: utf-8 -*-
"""桶内限流：每桶只取 sim 最高的 M 条，候选上限 = k_bars * M，与 N 无关

动机：容量曲线显示候选 ≈ N。根源不是"选错桶"，是"热桶里的条目全进了候选"。
      换桶（E5/g 粗筛）会更散，已证失败。
      正解是桶内限流：同一条记忆可能占多个桶，按桶各取 top-M 再并。

判据：
  - 候选数是否被压成常数（不随 N 涨）
  - hit@1 是否掉

用法：python _exp_bucketcap.py
输出：bucketcap_curve.png
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


def retrieve_cap(mem, query, topK=3, per_bar=3):
    """坐标桶并集 -> 每桶取 top-M -> 全局细排"""
    h = mem.encode(query)
    _, bars = mem._coord(h)
    cand = mem._candidates(bars)
    if not cand:
        return [], 0
    mids = [m for m in sorted(cand) if m not in mem.deleted]
    if not mids:
        return [], 0
    rows = torch.tensor([mem.slot_of[m] for m in mids], device=mem.dev)
    sim = mem.contents[rows] @ h
    sim_l = sim.tolist()

    # 反转索引：只对 query 命中的那些 bar 限流，每桶取 sim 前 M
    bars_set = set(bars)
    bar2 = {}
    for idx, mid in enumerate(mids):
        for b in set(mem.bars.get(mid, ())) & bars_set:
            bar2.setdefault(b, []).append((sim_l[idx], mid))
    keep = set()
    for b, lst in bar2.items():
        lst.sort(reverse=True)
        for _, m in lst[:per_bar]:
            keep.add(m)

    keep = sorted(keep)
    krows = torch.tensor([mem.slot_of[m] for m in keep], device=mem.dev)
    ksim = mem.contents[krows] @ h
    order = ksim.argsort(descending=True)[:topK]
    return [mem.texts[krows[i].item()] for i in order], len(keep)


def run(n):
    facts = make_facts(n)
    mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                         capacity=max(10000, n * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")

    res = {"n": n}
    for pb in (0, 1, 3, 6):
        hit1 = hit3 = 0
        tot = 0
        for f in facts:
            if pb == 0:
                items = mem.retrieve(f["query"], topK=3)
                texts = [it["text"] for it in items]
                # 同样算一遍未限流的候选数，用于画 base 曲线
                h = mem.encode(f["query"])
                _, bars = mem._coord(h)
                csz = len([m for m in mem._candidates(bars) if m not in mem.deleted])
            else:
                texts, csz = retrieve_cap(mem, f["query"], topK=3, per_bar=pb)
            tot += csz
            codes = [code_of(t) for t in texts]
            if f["code"] in codes:
                r = codes.index(f["code"]) + 1
                if r == 1:
                    hit1 += 1
                if r <= 3:
                    hit3 += 1
        key = "base" if pb == 0 else f"m{pb}"
        res[key + "_hit1"] = hit1 / n
        res[key + "_hit3"] = hit3 / n
        res[key + "_cand"] = tot / n
    print(f"N={n:5d}  " + "  ".join(
        f"{k}:h1={res[k+'_hit1']:.3f},c={res[k+'_cand']:.0f}"
        for k in ("base", "m1", "m3", "m6")), flush=True)
    return res


if __name__ == "__main__":
    NS = [60, 250, 1000, 2000]
    rows = []
    t0 = time.time()
    for n in NS:
        rows.append(run(n))
    print(f"\n总用时 {time.time()-t0:.0f}s", flush=True)

    import json
    with open("bucketcap_curve.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    keys = [("base", "o-", "no cap"), ("m1", "^-", "per-bucket top-1"),
            ("m3", "s-", "per-bucket top-3"), ("m6", "d-", "per-bucket top-6")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for k, st, lab in keys:
        ax1.plot(ns, [r[k + "_hit1"] for r in rows], st, label=lab)
        ax2.plot(ns, [r[k + "_cand"] for r in rows], st, label=lab)
    ax1.set_xscale("log"); ax1.set_xlabel("N (log)"); ax1.set_ylabel("hit@1")
    ax1.set_ylim(0, 1.05); ax1.grid(alpha=.3); ax1.legend(); ax1.set_title("hit@1 vs N")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("N (log)"); ax2.set_ylabel("avg candidate (log)")
    ax2.plot(ns, ns, "k:", alpha=.35, label="N (brute force)")
    ax2.grid(alpha=.3); ax2.legend(); ax2.set_title("candidate size vs N")
    plt.suptitle("Per-bucket top-M cap (C=2048, k_bars=16)")
    plt.tight_layout()
    plt.savefig("bucketcap_curve.png", dpi=150)
    print("saved bucketcap_curve.png")
