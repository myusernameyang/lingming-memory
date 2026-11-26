# -*- coding: utf-8 -*-
"""灵铭冷启动 v2：从 bge 蒸馏语义（关系蒸馏，不需要维度对齐）

思路：
  教师 z = bge-small(句子) → 归一化，得到"谁跟谁像"的相似度图
  学生 s = normalize(W_coord(CNN(字节)))
  损失 = MSE(s @ s.T, z @ z.T)   —— 让学生复现教师的相似度结构

为什么不直接用蒸馏成回归：学生维度 C=512 与教师 384 不同，关系蒸馏不需要对齐维度。
"""
import os
import sys
import time
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from memory_core import CNNEncoder, LingmingMemory  # noqa: E402
from _test_recall import PAIRS  # noqa: E402

D, C, K, L = 128, 512, 8, 64
BATCH = 128
STEPS = 3000
LR = 1e-3
N_SENTS = 30000

DEFAULT_CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(text):
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    b = (list(text) + [0] * L)[:L]
    return torch.tensor(b, dtype=torch.long)


def split_sents(path, max_bytes=20_000_000):
    raw = open(path, "rb").read(max_bytes)
    delims3 = [b"\xe3\x80\x82", b"\xef\xbc\x9f", b"\xef\xbc\x81"]
    pos = [0]
    for i in range(len(raw) - 3):
        if raw[i:i + 3] in delims3 or raw[i:i + 1] == b"\n":
            pos.append(i + 3)
    out = []
    for j in range(len(pos) - 1):
        s = raw[pos[j]:pos[j + 1]].strip()
        if 8 <= len(s) <= 64:
            out.append(s)
    return out


def teacher_embed(texts, batch=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    out = []
    for i in range(0, len(texts), batch):
        e = m.encode(texts[i:i + batch], normalize_embeddings=True,
                     batch_size=batch, show_progress_bar=False)
        out.append(torch.tensor(e, dtype=torch.float32))
    return torch.cat(out)


def evaluate(encoder, W_coord, label):
    mem = LingmingMemory(encoder, W_coord, k=K, L=L, capacity=10000)
    for mem_text, _ in PAIRS:
        mem.add(mem_text, "user")
    text2mid = {v["text"]: mid for mid, v in mem.store_.items()}
    r1 = r3 = 0
    for mem_text, query in PAIRS:
        target = text2mid[mem_text]
        got = [text2mid[it["text"]] for it in mem.retrieve(query, 3)]
        if got and got[0] == target:
            r1 += 1
        if target in got:
            r3 += 1
    n = len(PAIRS)
    print(f"  {label:24s} recall@1 {r1/n*100:5.1f}%   recall@3 {r3/n*100:5.1f}%", flush=True)
    return r1 / n


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    ckpt = os.path.join(BASE, "coord.pt")
    bak = os.path.join(BASE, "coord.pt.bak")
    if os.path.exists(ckpt) and not os.path.exists(bak):
        shutil.copy2(ckpt, bak)
        print(f"已备份 -> {bak}", flush=True)

    sents_b = split_sents(corpus)
    step = max(1, len(sents_b) // N_SENTS)
    sents_b = sents_b[::step][:N_SENTS]
    sents_s = [s.decode("utf-8", errors="ignore") for s in sents_b]
    print(f"样本句 {len(sents_b)} 条（从 {len(sents_s)} 抽稀）", flush=True)

    print("编码教师 bge ...", flush=True)
    t0 = time.time()
    Z = teacher_embed(sents_s).to(DEV)
    print(f"教师特征 {tuple(Z.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    X = torch.stack([to_tensor(s) for s in sents_b])
    n = X.shape[0]

    torch.manual_seed(0)
    encoder = CNNEncoder(D).to(DEV)
    W_coord = nn.Linear(D, C, bias=False).to(DEV)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(W_coord.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    print(f"\n关系蒸馏训练：{STEPS} 步，batch {BATCH}", flush=True)
    t0 = time.time()
    encoder.train(); W_coord.train()
    for st in range(STEPS):
        idx = torch.randint(0, n, (BATCH,))
        xb = X[idx].to(DEV)
        zb = Z[idx]
        s = F.normalize(W_coord(encoder(xb)), dim=-1)
        st_sim = s @ s.T
        te_sim = zb @ zb.T
        loss = F.mse_loss(st_sim, te_sim)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (st + 1) % 500 == 0:
            with torch.no_grad():
                corr = torch.corrcoef(torch.stack([st_sim.flatten(), te_sim.flatten()]))[0, 1].item()
            print(f"  step {st+1:5d}  mse {loss.item():.5f}  corr {corr:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    encoder.eval(); W_coord.eval()
    torch.save({"encoder": encoder.state_dict(), "W_coord": W_coord.state_dict()}, ckpt)
    print(f"\n已保存 -> {ckpt}", flush=True)

    print("\n【冷启动 v2 评估】", flush=True)
    evaluate(encoder, W_coord, "bge 蒸馏")


if __name__ == "__main__":
    main()
