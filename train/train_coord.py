# -*- coding: utf-8 -*-
"""灵铭记忆 · 坐标训练脚本
用「相似对」训练坐标网络：语义相似的文本 → 相同的稀疏 top-k 坐标（桶交集命中）

用法：
  python train_coord.py data.json

data.json 格式（{"pairs": [["文本A", "文本B"], ...]}，A/B 语义相似）：
  {"pairs": [["我叫小明", "我的名字是小明"], ["喜欢篮球", "爱打篮球"]]}

原理：对比损失（InfoNCE）——同一对的两句坐标接近、不同对的远离，
训练 CNN 语义特征 + W_coord 坐标投影，让"相似 → 相同坐标"。

产物：coord.pt（encoder + W_coord 权重，embed_server 加载）
"""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_core import CNNEncoder

torch.manual_seed(0)

D, C, K, L = 128, 512, 8, 64
BATCH = 32
EPOCHS = 300


def to_tensor(text):
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    b = (list(text) + [0] * L)[:L]
    return torch.tensor(b, dtype=torch.long)


def main(data_path):
    data = json.load(open(data_path, encoding="utf-8-sig"))
    pairs = data.get("pairs", [])
    if len(pairs) < 2:
        print("data.json 需要至少 2 对 pairs（相似对）")
        sys.exit(1)

    A = torch.stack([to_tensor(a) for a, _ in pairs])
    B = torch.stack([to_tensor(b) for _, b in pairs])
    n = len(pairs)
    print(f"相似对 {n} 对，训练 CNN({D}) + W_coord({D}->{C}) 稀疏 top-k({K})")

    encoder = CNNEncoder(D)
    W_coord = nn.Linear(D, C, bias=False)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(W_coord.parameters()), lr=1e-3)

    for epoch in range(EPOCHS):
        idx = torch.randint(0, n, (BATCH,))
        ca = F.normalize(W_coord(encoder(A[idx])), dim=-1)
        cb = F.normalize(W_coord(encoder(B[idx])), dim=-1)
        sim = ca @ cb.T  # [B,B]，对角线 = 正确对
        loss = F.cross_entropy(sim / 0.1, torch.arange(BATCH))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            acc = (sim.argmax(-1) == torch.arange(BATCH)).float().mean().item()
            print(f"epoch {epoch+1}: loss {loss.item():.4f} 检索准确率 {acc*100:.1f}%")

    torch.save({"encoder": encoder.state_dict(), "W_coord": W_coord.state_dict()}, "coord.pt")
    print(f"\n已保存 coord.pt（{n} 对训练完成）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.json")
