# -*- coding: utf-8 -*-
"""验证「raw = g·h，top-k 当向量库键」这条路

h = 冻结 bge 特征（384 维，语义现成，不蒸馏）
g = 384 → C 的线性投影（唯一要训的东西）
top-k(g·h) = 桶键 → 向量库候选过滤 → 精确 cosine 排序

对比：g 随机初始化 vs g 训练后
指标：candidate-recall（正确项有没有进候选集）+ end-to-end 一致率
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DH, C_MULT = 384, 4          # bge-small 隐藏维 / 桶数倍数
C = C_MULT * DH
N_TRAIN, N_POOL, N_QUERY = 5000, 1000, 200
STEPS, BATCH, LR = 2000, 256, 1e-3


def split_sents(raw, limit=20_000_000):
    d3 = [b"\xe3\x80\x82", b"\xef\xbc\x9f", b"\xef\xbc\x81"]
    pos = [0]
    for i in range(len(raw) - 3):
        if raw[i:i + 3] in d3 or raw[i:i + 1] == b"\n":
            pos.append(i + 3)
    out = []
    for j in range(len(pos) - 1):
        s = raw[pos[j]:pos[j + 1]].strip()
        if 8 <= len(s) <= 64:
            out.append(s.decode("utf-8", "ignore"))
    return out


def bge_encode(texts, batch=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    out = []
    for i in range(0, len(texts), batch):
        e = m.encode(texts[i:i + batch], normalize_embeddings=True,
                     batch_size=batch, show_progress_bar=False)
        out.append(torch.tensor(e, dtype=torch.float32))
    return F.normalize(torch.cat(out).to(DEV), dim=-1)


def build_candidates(keys, g, k):
    """keys: [N,C] 连续坐标；返回每个查询的候选下标集合（桶内并集）"""
    pass


def eval_pipeline(W, Hp, Hq, gold, k, C_):
    """W: [C,dh]；Hp: 池特征；Hq: 查询特征；gold: 查询对应池内正确下标"""
    with torch.no_grad():
        kp = (W @ Hp.T).T.topk(k, dim=-1).indices      # [Np,k]
        kq = (W @ Hq.T).T.topk(k, dim=-1).indices      # [Nq,k]
    # 建倒排：bar -> [pool idx]
    index = {}
    for i in range(kp.shape[0]):
        for b in kp[i].tolist():
            index.setdefault(b, []).append(i)

    cand_recall = 0
    e2e = 0
    for j in range(kq.shape[0]):
        cands = set()
        for b in kq[j].tolist():
            cands.update(index.get(b, ()))
        if gold[j].item() in cands:
            cand_recall += 1
        if cands:
            idx = torch.tensor(sorted(cands), device=Hq.device)
            sim = Hq[j] @ Hp[idx].T
            best = idx[sim.argmax()]
            if best.item() == gold[j].item():
                e2e += 1
    n = kq.shape[0]
    return cand_recall / n, e2e / n


def main():
    sents = split_sents(open(CORPUS, "rb").read(20_000_000))
    need = N_TRAIN + N_POOL + N_QUERY
    sents = sents[:need]
    print(f"句子 {len(sents)} 条", flush=True)

    t0 = time.time()
    H = bge_encode(sents)
    print(f"bge 特征 {tuple(H.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    Htr = H[:N_TRAIN]
    Hp = H[N_TRAIN:N_TRAIN + N_POOL]
    Hq = H[N_TRAIN + N_POOL:need]

    # 用 bge 近邻造正例（自监督，不需要人工标注）
    with torch.no_grad():
        sim = Htr @ Htr.T
        sim.fill_diagonal_(-2.0)
        nn_idx = sim.argmax(-1)
    print(f"bge 近邻正例已构建（每句配最近邻）", flush=True)

    gold = (Hq @ Hp.T).argmax(-1)      # 参考答案 = bge 在池上的 top-1

    torch.manual_seed(0)
    W_rand = nn.Linear(DH, C, bias=False).to(DEV)

    W = nn.Linear(DH, C, bias=False).to(DEV)
    opt = torch.optim.Adam(W.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    print(f"\n训练 g（{DH} -> {C}），{STEPS} 步", flush=True)
    t0 = time.time()
    for st in range(STEPS):
        idx = torch.randint(0, N_TRAIN, (BATCH,))
        pos = nn_idx[idx]
        a = F.normalize(W(Htr[idx]), dim=-1)
        b = F.normalize(W(Htr[pos]), dim=-1)
        loss = F.cross_entropy((a @ b.T) / 0.05, torch.arange(BATCH, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (st + 1) % 500 == 0:
            acc = ((a @ b.T).argmax(-1) == torch.arange(BATCH, device=DEV)).float().mean().item()
            print(f"  step {st+1:4d}  loss {loss.item():.4f}  batch_acc {acc*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'g':10s} {'k':>4s}  {'cand-recall':>12s}  {'end-to-end':>11s}", flush=True)
    for name, Wm in (("随机", W_rand), ("训练后", W)):
        for k in (8, 16, 32):
            cr, e2 = eval_pipeline(Wm.weight, Hp, Hq, gold, k, C)
            print(f"{name:10s} {k:4d}  {cr*100:11.1f}%  {e2*100:10.1f}%", flush=True)


if __name__ == "__main__":
    main()
