# -*- coding: utf-8 -*-
"""灵铭记忆核心：坐标寻址多级向量库
存：文本 → CNN 语义特征 → 稀疏 top-k 坐标（k 个 bar = k 级地址）→ 分桶索引
查：查询 → 坐标 top-k → 桶交集 → 命中最相关记忆（O(k)，不随记忆量涨）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_topk(coord, k):
    """返回 top-k 的 index（地址），不返回值"""
    return coord.topk(k, dim=-1).indices


class CNNEncoder(nn.Module):
    """语义压缩：字节序列 → 特征向量（可外挂冻结 embedding + CNN 适配）"""
    def __init__(self, d=128, freeze_embed=False):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        if freeze_embed:
            self.embed.weight.requires_grad = False
        self.conv1 = nn.Conv1d(d, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, padding=1)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)
        h = F.relu(self.conv2(F.relu(self.conv1(e))))
        return h.max(dim=-1).values  # [B, d]


class LingmingMemory:
    """坐标寻址多级索引：稀疏 top-k 坐标当多级地址，桶交集寻址"""
    def __init__(self, encoder, W_coord, k=8, L=32):
        self.encoder = encoder
        self.W_coord = W_coord
        self.k = k
        self.L = L
        self.index = {}      # bar index -> [mem_id]（多级索引）
        self.store_ = {}     # mem_id -> {"text": ..., "role": ...}
        self.next_id = 0
        self.total_writes = 0

    def _tensor(self, text):
        if isinstance(text, str):
            text = text.encode("utf-8", errors="ignore")
        b = (list(text) + [0] * self.L)[:self.L]
        return torch.tensor(b, dtype=torch.long).unsqueeze(0)

    def coord(self, text):
        """文本 → 坐标 top-k（k 个 bar index）"""
        with torch.no_grad():
            x = self._tensor(text).to(next(self.encoder.parameters()).device)
            h = self.encoder(x)
            return sparse_topk(self.W_coord(h), self.k)[0].tolist()

    def add(self, text, role="user"):
        """存记忆：坐标 top-k → 每个 bar 桶挂上这个记忆"""
        if not text:
            return -1
        mid = self.next_id
        self.store_[mid] = {"text": text, "role": role}
        for bar in self.coord(text):
            self.index.setdefault(bar, []).append(mid)
        self.next_id += 1
        self.total_writes += 1
        return mid

    def retrieve(self, query, topK=3):
        """查：坐标 top-k → 桶交集 → 命中次数最多的记忆"""
        votes = {}
        for bar in self.coord(query):
            for mid in self.index.get(bar, []):
                votes[mid] = votes.get(mid, 0) + 1
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        return [self.store_[mid] for mid, _ in ranked[:topK]]

    def status(self):
        return {"total_writes": self.total_writes, "n_mem": self.next_id, "n_bars": len(self.index)}
