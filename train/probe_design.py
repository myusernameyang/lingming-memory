# -*- coding: utf-8 -*-
"""探针设计对比：随机 / 去均值 / PCA / 白化

坐标 = 探针方向上的投影；这里比较四种"探针组"下的容量曲线。
注意：PCA 用全体样本估计方向（相当于"知道语料是什么"的探针），属于上界参考。
"""
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory, sparse_topk

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"
D, C, K = 512, 2048, 16

SUBJECTS = [("服务器", "杭州"), ("数据库", "北京"), ("缓存", "上海"), ("备份", "成都"),
            ("监控", "深圳"), ("网关", "广州"), ("队列", "南京"), ("日志", "武汉"),
            ("镜像", "西安"), ("构建机", "苏州")]
ATTRS = [("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T"), ("带宽", "500Mbps"),
         ("系统", "Ubuntu 22.04"), ("端口", "8443")]

_cache = {}
_enc = None
def encode(text):
    global _enc
    if text in _cache:
        return _cache[text]
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    v = torch.tensor(e[0], dtype=torch.float32, device=DEV)
    _cache[text] = v
    return v


def facts(n):
    out = []
    for i in range(n):
        s, loc = SUBJECTS[i % len(SUBJECTS)]
        a, v = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
        out.append({"code": f"KX{i:05d}",
                    "text": f"{s}的{a}是 {v}，代号 KX{i:05d}，位于{loc}。",
                    "query": f"{s}的{a}代号是什么"})
    return out


def patched_coord(W, mean=None):
    """构造一个替代 _coord 的函数：投影 W 作用在 (h - mean) 上"""
    def _coord(h):
        hh = h - mean if mean is not None else h
        coord = W @ hh
        q = sparse_topk(coord.unsqueeze(0), min(K, C))[0]
        bars = q.nonzero(as_tuple=False).flatten().tolist()
        return q, bars
    return _coord


def evaluate(mem, probes):
    h1 = h5 = mr = 0
    for p in probes:
        items = mem.retrieve(p["query"], topK=5)
        codes = []
        for it in items:
            m = re.search(r"KX\d{5}", it["text"])
            codes.append(m.group(0) if m else "")
        if p["code"] in codes:
            r = codes.index(p["code"]) + 1
            mr += 1.0 / r
            h1 += r == 1
            h5 += r <= 5
    n = len(probes)
    return h1 / n, h5 / n, mr / n


def build_mem(N, W, mu):
    """按探针数建记忆库：C 必须与 W 的行数一致"""
    Cw = W.shape[0]
    mem = LingmingMemory(encode, d=D, c_mult=max(1, Cw // D), k=K,
                         capacity=max(10000, N * 2), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    if mem.C != Cw:
        mem.C = Cw
        mem.g = torch.zeros(D, Cw, device=DEV)
        mem.t_g = torch.zeros(Cw, device=DEV)
    mem.W_coord = W
    mem._coord = patched_coord(W, mu)
    return mem


def main():
    ALL = facts(3200)
    probes = ALL[:60]
    H = torch.stack([encode(f["text"]) for f in ALL])
    mean = H.mean(0, keepdim=True)          # [1, D]
    mean_vec = mean.squeeze(0)              # [D]，广播时用这个
    Hc = H - mean

    gen = torch.Generator(device="cpu").manual_seed(12345)
    W_rand = F.normalize(torch.randn(C, D, generator=gen), dim=-1).to(DEV)

    # PCA：协方差的前 C 个主方向（用全量样本估计，属于上界参考）
    U, S, V = torch.linalg.svd(Hc, full_matrices=False)   # V: [r, D]
    W_pca = V[:C]                                          # [C, D]
    lam = (S[:C] ** 2 / max(1, Hc.shape[0] - 1)).clamp(min=1e-8)
    W_white = W_pca / lam.sqrt().unsqueeze(1)              # 白化：按特征值归一

    W_rand512 = F.normalize(torch.randn(512, D, generator=torch.Generator(device="cpu").manual_seed(999)), dim=-1).to(DEV)

    modes = [
        ("随机 2048",      W_rand,    None),
        ("随机 512",       W_rand512, None),
        ("随机+去均值",    W_rand,    mean_vec),
        ("PCA 探针",       W_pca,     mean_vec),
        ("PCA+白化",       W_white,   mean_vec),
    ]
    sizes = [200, 400, 800, 1600, 3200]

    print(f"{'探针':14s} " + " ".join(f"N={n:<5d}" for n in sizes) + "  用桶/2048")
    print("-" * 74)
    for name, W, mu in modes:
        row = []
        bars_used = 0
        for N in sizes:
            mem = build_mem(N, W, mu)
            for f in ALL[:N]:
                mem.add(f["text"], "user")
            h1, h5, mr = evaluate(mem, probes)
            row.append(h1)
            bars_used = len(mem.index)
        print(f"{name:14s} " + " ".join(f"{v:6.3f}" for v in row) + f"  {bars_used:4d}")


if __name__ == "__main__":
    main()
