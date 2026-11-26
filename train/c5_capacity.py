# -*- coding: utf-8 -*-
"""C5 · 容量曲线：条目数增大时，检索质量什么时候崩

同一批合成事实，从 N=50 加到 N=3200，每档重建记忆库后测：
  hit@1 / hit@5 / MRR / 空结果率 / 桶占用
"""
import re
import time
import torch
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"

SUBJECTS = [
    ("服务器", "杭州机房"), ("数据库", "北京亦庄"), ("缓存", "阿里云上海区"),
    ("备份", "成都双流"), ("监控", "深圳南山"), ("网关", "广州天河"),
    ("队列", "南京江宁"), ("日志", "武汉光谷"), ("镜像仓库", "西安高新"),
    ("构建机", "苏州工业园"),
]
ATTRS = [
    ("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T NVMe"),
    ("带宽", "500Mbps"), ("系统", "Ubuntu 22.04"), ("端口", "8443"),
]

_enc = None
_cache = {}

def encode(text):
    """带缓存的编码：同一条文本只编码一次，跨 N 复用"""
    global _enc
    if text in _cache:
        return _cache[text]
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    v = torch.tensor(e[0], dtype=torch.float32, device=DEV)
    _cache[text] = v
    return v


def make_facts(n):
    facts = []
    for i in range(n):
        subj, loc = SUBJECTS[i % len(SUBJECTS)]
        attr, val = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
        code = f"KX{i:05d}"
        facts.append({
            "code": code,
            "text": f"{subj}的{attr}是 {val}，代号 {code}，位于{loc}。",
            "query": f"{subj}的{attr}代号是什么",
        })
    return facts


def run(sizes=(50, 100, 200, 400, 800, 1600, 3200), probe=60):
    facts = make_facts(max(sizes))
    probes = facts[:probe]          # 固定探针，保证各档可比

    print(f"{'N':>6} {'hit@1':>7} {'hit@5':>7} {'MRR':>7} {'空结果':>7} {'桶数':>7} {'次/桶':>7} {'耗时':>6}")
    print("-" * 68)
    rows = []
    for N in sizes:
        t0 = time.time()
        mem = LingmingMemory(encode, d=512, c_mult=4, k=16,
                             capacity=max(10000, N * 2), device=DEV,
                             center=False, write_scale=0.1,
                             gate_bias=0.2, gate_temp=0.05, read_top=3)
        for f in facts[:N]:
            mem.add(f["text"], "user")

        h1 = h5 = mr = empty = 0
        for p in probes:
            items = mem.retrieve(p["query"], topK=5)
            if not items:
                empty += 1
                continue
            codes = []
            for it in items:
                m = re.search(r"KX\d{5}", it["text"])
                codes.append(m.group(0) if m else "")
            if p["code"] in codes:
                rank = codes.index(p["code"]) + 1
                mr += 1.0 / rank
                if rank == 1:
                    h1 += 1
                if rank <= 5:
                    h5 += 1
        n = len(probes)
        bars = len(mem.index)
        occ = (N * mem.k) / max(1, bars)
        dt = time.time() - t0
        print(f"{N:>6} {h1/n:7.3f} {h5/n:7.3f} {mr/n:7.3f} {empty/n:7.3f} {bars:>7} {occ:7.1f} {dt:5.1f}s")
        rows.append((N, h1/n, h5/n, mr/n, empty/n, bars, occ))
    return rows


if __name__ == "__main__":
    run()
