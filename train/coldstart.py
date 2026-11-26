# -*- coding: utf-8 -*-
"""灵铭冷启动：从语料挖相似对 → 预训 coord.pt → 前后对比

配对来源（两类）：
  1) 纠正对：下一句以「不对/错了/应该是/我说的是」开头 → (原句, 纠正句) 强正例
  2) 相邻对：同一段对话的相邻两句 → 弱正例，量大

用法：
  python coldstart.py [语料路径] [相邻对上限]
"""
import os
import sys
import json
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
from _test_recall import PAIRS  # 标注集（20 组）  # noqa: E402

D, C, K, L = 128, 512, 8, 64
BATCH = 128
STEPS = 4000
LR = 1e-3

DEFAULT_CORPUS = r"C:\Users\yangtianpeng\Documents\ChatGPT\swam_model\gm_ablation\zh_dialogue.txt"

MARKERS = [
    "不对", "错了", "应该是", "我说的是", "不是这样", "纠正一下",
    "准确地说", "更准确", "其实是", "搞错了",
]


def to_tensor(text):
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    b = (list(text) + [0] * L)[:L]
    return torch.tensor(b, dtype=torch.long)


def split_sents(path, max_bytes=20_000_000):
    raw = open(path, "rb").read(max_bytes)
    delims3 = [b"\xe3\x80\x82", b"\xef\xbc\x9f", b"\xef\xbc\x81"]  # 。？！ (3 bytes each)
    pos = [0]
    for i in range(len(raw) - 3):
        if raw[i:i + 3] in delims3 or raw[i:i + 1] == b"\n":
            pos.append(i + 3)
    out = []
    for j in range(len(pos) - 1):
        s = raw[pos[j]:pos[j + 1]].strip()
        if 6 <= len(s) <= 64:
            out.append(s)
    return out


def mine(corpus, max_adjacent=200_000):
    sents = split_sents(corpus)
    print(f"切出句子 {len(sents)} 条", flush=True)
    strong, weak = [], []
    for i in range(len(sents) - 1):
        a, b = sents[i], sents[i + 1]
        if a == b:
            continue
        bt = b.decode("utf-8", errors="ignore")
        if any(bt.startswith(m) for m in MARKERS):
            strong.append((a, b))
        else:
            weak.append((a, b))
    # 相邻对降采样，避免完全压过纠正对
    if len(weak) > max_adjacent:
        step = len(weak) / max_adjacent
        weak = [weak[int(i * step)] for i in range(max_adjacent)]
    pairs = strong + weak
    print(f"纠正对 {len(strong)} 对 + 相邻对 {len(weak)} 对 = {len(pairs)} 对", flush=True)
    return pairs


def evaluate(encoder, W_coord, label):
    """在 20 组标注集上测 recall@1 / @3（含细排）"""
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
    print(f"  {label:22s} recall@1 {r1/n*100:5.1f}%   recall@3 {r3/n*100:5.1f}%", flush=True)
    return r1 / n, r3 / n


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    max_adj = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

    ckpt = os.path.join(BASE, "coord.pt")
    bak = os.path.join(BASE, "coord.pt.bak")
    if os.path.exists(ckpt) and not os.path.exists(bak):
        shutil.copy2(ckpt, bak)
        print(f"已备份原 coord.pt -> {bak}", flush=True)

    # 旧模型基线
    enc_old = CNNEncoder(D)
    w_old = nn.Linear(D, C, bias=False)
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu")
        w = st.get("W_coord", st.get("W", None))
        if isinstance(w, dict):
            enc_old.load_state_dict(st.get("encoder", {}), strict=False)
            w_old.load_state_dict(w, strict=False)
    enc_old.eval(); w_old.eval()
    print("\n【冷启动前】", flush=True)
    evaluate(enc_old, w_old, "旧 coord.pt（5 对训练）")

    pairs = mine(corpus, max_adj)
    A = torch.stack([to_tensor(a) for a, _ in pairs])
    B = torch.stack([to_tensor(b) for _, b in pairs])
    n = len(pairs)

    torch.manual_seed(0)
    encoder = CNNEncoder(D)
    W_coord = nn.Linear(D, C, bias=False)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(W_coord.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    print(f"\n开始训练：{n} 对，batch {BATCH}，{STEPS} 步", flush=True)
    t0 = time.time()
    encoder.train(); W_coord.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,))
        ca = F.normalize(W_coord(encoder(A[idx])), dim=-1)
        cb = F.normalize(W_coord(encoder(B[idx])), dim=-1)
        sim = ca @ cb.T
        loss = F.cross_entropy(sim / 0.07, torch.arange(BATCH))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            acc = (sim.argmax(-1) == torch.arange(BATCH)).float().mean().item()
            print(f"  step {step+1:5d}  loss {loss.item():.4f}  batch_acc {acc*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    encoder.eval(); W_coord.eval()
    torch.save({"encoder": encoder.state_dict(), "W_coord": W_coord.state_dict()}, ckpt)
    print(f"\n已保存新 coord.pt -> {ckpt}", flush=True)

    print("\n【冷启动后】", flush=True)
    old1, _ = evaluate(enc_old, w_old, "旧模型（对照）")
    new1, _ = evaluate(encoder, W_coord, "新模型（冷启动）")
    print(f"\nrecall@1: {old1*100:.1f}%  ->  {new1*100:.1f}%   ({'+' if new1>=old1 else ''}{(new1-old1)*100:.1f})")


if __name__ == "__main__":
    main()
