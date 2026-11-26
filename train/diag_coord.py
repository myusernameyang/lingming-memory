# -*- coding: utf-8 -*-
"""诊断：为什么桶数不增长？—— 坐标投影是否退化

对比三种坐标计算方式下，3200 条事实实际用到多少个不同的桶：
  A 原样：        W_coord @ h
  B 去均值：      W_coord @ (h - h_mean)
  C 去均值+白化： W_coord @ (h - h_mean) / std
"""
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from memory_core import sparse_topk

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"
D, C, K = 512, 2048, 16

SUBJECTS = [("服务器", "杭州"), ("数据库", "北京"), ("缓存", "上海"), ("备份", "成都"),
            ("监控", "深圳"), ("网关", "广州"), ("队列", "南京"), ("日志", "武汉"),
            ("镜像", "西安"), ("构建机", "苏州")]
ATTRS = [("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T"), ("带宽", "500Mbps"),
         ("系统", "Ubuntu 22.04"), ("端口", "8443")]

texts = []
for i in range(3200):
    s, loc = SUBJECTS[i % len(SUBJECTS)]
    a, v = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
    texts.append(f"{s}的{a}是 {v}，代号 KX{i:05d}，位于{loc}。")

m = SentenceTransformer(BGE, device=DEV)
H = torch.tensor(m.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                 dtype=torch.float32, device=DEV)

gen = torch.Generator(device="cpu").manual_seed(12345)
W = F.normalize(torch.randn(C, D, generator=gen), dim=-1).to(DEV)

def usage(Hin, label):
    coord = Hin @ W.T                      # [n, C]
    val, idx = coord.topk(K, dim=-1)
    used = torch.unique(idx)
    # 全局：这些桶一共被点亮多少次
    total = idx.numel()
    occ = total / used.numel()
    # 退化程度：最热的 10 个桶吃掉了多少比例的点亮次数
    flat = idx.flatten()
    counts = torch.bincount(flat, minlength=C)
    top10 = counts.topk(10).values.sum().item() / total
    print(f"{label:16s} 用到桶 {used.numel():4d}/{C}  平均 {occ:6.1f} 次/桶  "
          f"最热 10 桶占比 {top10:5.1%}")
    return used.numel()

h_mean = H.mean(0, keepdim=True)
std = H.std(0, keepdim=True).clamp(min=1e-6)

print(f"样本 {H.shape[0]} 条，d={D}, C={C}, k={K}\n")
uA = usage(H, "A 原样")
uB = usage(H - h_mean, "B 去均值")
uC = usage((H - h_mean) / std, "C 白化")

# 追一层：数据本身的主方向占多少
Hc = H - h_mean
U, S, V = torch.linalg.svd(Hc, full_matrices=False)
ratio = (S ** 2) / (S ** 2).sum()
print(f"\n前 1 个主成分占方差 {ratio[0]:.1%}，前 5 个占 {ratio[:5].sum():.1%}")
print(f"参与度（有效维）: {(ratio.sum()**2)/(ratio**2).sum():.1f}")
