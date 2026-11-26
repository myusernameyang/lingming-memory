# -*- coding: utf-8 -*-
"""修正后的测试：query 带唯一代号，任务不再有歧义

然后测：学一个低秩度量 M（h' = normalize(h @ M)），
能不能把"只差代号"的两条记忆拉开？——这就是 XWY 里的 W。
"""
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"
SUBJ = ["服务器", "数据库", "缓存", "备份", "监控", "网关", "队列", "日志", "镜像", "构建机"]
ATTR = ["核数", "内存", "磁盘", "带宽", "系统", "端口"]
VAL = ["32 核", "128G", "4T", "500Mbps", "Ubuntu", "8443"]
LOC = ["杭州", "北京", "上海", "成都", "深圳", "广州", "南京", "武汉", "西安", "苏州"]

N_TRAIN, N_TEST, N_DB = 2000, 300, 3200


def make(n):
    facts, queries = [], []
    for i in range(n):
        s, a, v, l = SUBJ[i % 10], ATTR[(i // 10) % 6], VAL[(i // 10) % 6], LOC[i % 10]
        code = f"KX{i:05d}"
        facts.append(f"{s}的{a}是 {v}，代号 {code}，位于{l}。")
        queries.append(f"代号 {code} 那台{s}的{a}是什么")
    return facts, queries


def hit(Fx, Qsub, idx, M=None, k=5):
    Fx2, Qx2 = Fx, Qsub
    if M is not None:
        Fx2 = F.normalize(Fx @ M, dim=-1)
        Qx2 = F.normalize(Qsub @ M, dim=-1)
    S = Qx2 @ Fx2.T                   # [nq, ndb]
    top = S.topk(k, dim=1).indices
    h1 = (top[:, 0] == idx).float().mean().item()
    hk = (top == idx.unsqueeze(1)).any(dim=1).float().mean().item()
    return h1, hk


def main():
    enc = SentenceTransformer(BGE, device=DEV)
    facts, queries = make(N_DB)
    Fx = torch.tensor(enc.encode(facts, normalize_embeddings=True, show_progress_bar=False),
                      dtype=torch.float32, device=DEV)
    Qx = torch.tensor(enc.encode(queries, normalize_embeddings=True, show_progress_bar=False),
                      dtype=torch.float32, device=DEV)

    # 基线：只用 bge 余弦
    tr_idx = torch.arange(N_TRAIN, device=DEV)
    te_idx = torch.arange(N_DB - N_TEST, N_DB, device=DEV)
    h1, hk = hit(Fx, Qx[te_idx], te_idx)
    print(f"基线（bge 余弦）    hit@1={h1:.3f}  hit@5={hk:.3f}")

    # 只差代号的两条事实，原始余弦
    c0 = float(Fx[0] @ Fx[1])
    c1 = float(Qx[0] @ Fx[0])       # 正确配对
    c2 = float(Qx[0] @ Fx[1])       # 错误配对（只差代号）
    print(f"事实0 vs 事实1（只差代号） 余弦 {c0:.4f}")
    print(f"查询0 vs 正确事实          余弦 {c1:.4f}")
    print(f"查询0 vs 错误事实          余弦 {c2:.4f}   差值 {c1-c2:+.4f}")

    # 学一个低秩度量 M：[d, r]，用 infoNCE 把 query_i 拉向 fact_i
    for r in (32, 128):
        M = torch.nn.Parameter(torch.eye(Fx.shape[1], r, device=DEV) * 0.1)
        opt = torch.optim.Adam([M], lr=1e-3)
        for step in range(400):
            q = F.normalize(Qx[tr_idx] @ M, dim=-1)
            f = F.normalize(Fx[tr_idx] @ M, dim=-1)
            logits = q @ f.T / 0.1
            loss = F.cross_entropy(logits, torch.arange(len(tr_idx), device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Mv = M.detach()
            h1m, hkm = hit(Fx, Qx[te_idx], te_idx, M=Mv)
            q0 = F.normalize(Qx[0:1] @ Mv, dim=-1)
            f0 = F.normalize(Fx[0:1] @ Mv, dim=-1)
            f1 = F.normalize(Fx[1:2] @ Mv, dim=-1)
            d_right = float(q0 @ f0.T)
            d_wrong = float(q0 @ f1.T)
            sep = float(Fx[0:1] @ Mv @ (Fx[1:2] @ Mv).T) if False else None
        print(f"\n学习 M (r={r})   hit@1={h1m:.3f}  hit@5={hkm:.3f}")
        print(f"  查询0 vs 正确 {d_right:.4f}  vs 错误(只差代号) {d_wrong:.4f}  差值 {d_right-d_wrong:+.4f}")


if __name__ == "__main__":
    main()
