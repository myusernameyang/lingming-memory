# -*- coding: utf-8 -*-
"""日常语言的"有效维度"测量

从真实会话日志里抽取文本 → bge 编码 → 中心化后看谱
回答：512 维的嵌入空间里，真实语言实际占几个独立方向？
"""
import json
import re
import sys

import torch
import zstandard as zstd
from sentence_transformers import SentenceTransformer

SESSION = (r"C:\Users\yangtianpeng\.dsh\sessions"
           r"\--C-Users-yangtianpeng-Documents-deepseek-harness-default-workspace--"
           r"\session-e3fc92dc-f029-4caf-8067-38fbf5b58f35\session.v4.jsonl.zstd")


def collect_texts(limit=500):
    """从会话日志里抽 user/assistant 的文本块"""
    texts = []
    def walk(o, out):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, out)
        elif isinstance(o, list):
            for v in o:
                walk(v, out)

    with open(SESSION, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as r:
            buf = b''
            while True:
                ch = r.read(1 << 20)
                if not ch:
                    break
                buf += ch
                *lines, buf = buf.split(b'\n')
                for ln in lines:
                    if len(texts) >= limit:
                        break
                    if not (b'"user/message"' in ln or b'"assistant/message"' in ln):
                        continue
                    try:
                        obj = json.loads(ln.decode('utf-8', 'ignore'))
                    except Exception:
                        continue
                    out = []
                    walk(obj.get('data', obj), out)
                    for t in out:
                        t = t.strip()
                        # 过滤：太短、工具输出、代码块、乱码
                        if len(t) < 30 or len(t) > 400:
                            continue
                        if t.startswith('{') or '```' in t or 'C:\\' in t:
                            continue
                        if not re.search(r'[\u4e00-\u9fff]', t):
                            continue
                        texts.append(t)
                        break
                if len(texts) >= limit:
                    break
    # 去重
    seen, uniq = set(), []
    for t in texts:
        k = t[:40]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def spectrum_report(X):
    """X: [n, d] 已归一化；中心化后做 SVD"""
    Xc = X - X.mean(0, keepdim=True)
    # Gram 矩阵的特征值（n < d 时更省）
    G = Xc @ Xc.T / max(1, X.shape[0] - 1)
    lam = torch.linalg.eigvalsh(G).clamp(min=0).flip(0)
    lam = lam[lam > 1e-8]
    total = lam.sum()
    cum = torch.cumsum(lam, 0) / total
    # 参与度：有效维度
    pr = (total ** 2) / (lam ** 2).sum()
    out = {'n': X.shape[0], 'd': X.shape[1], 'rank': int(lam.numel()), 'participation_ratio': float(pr)}
    for th in (0.5, 0.9, 0.95, 0.99):
        idx = int((cum < th).sum().item()) + 1
        out[f'dim_{int(th*100)}'] = min(idx, int(lam.numel()))
    return out


def main():
    texts = collect_texts(500)
    print(f"抽取文本 {len(texts)} 条，平均长度 {sum(len(t) for t in texts)/max(1,len(texts)):.0f} 字")
    if len(texts) < 30:
        print("样本太少"); return

    m = SentenceTransformer('BAAI/bge-small-zh-v1.5',
                            device='cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=torch.float32)

    rep = spectrum_report(X)
    print("\n=== 嵌入空间的谱 ===")
    print(f"样本数 {rep['n']}  维度 {rep['d']}  秩上界 {rep['rank']}")
    print(f"参与度（有效维度）：{rep['participation_ratio']:.1f}")
    print(f"解释 50% 方差需要 {rep['dim_50']} 维")
    print(f"解释 90% 方差需要 {rep['dim_90']} 维")
    print(f"解释 95% 方差需要 {rep['dim_95']} 维")
    print(f"解释 99% 方差需要 {rep['dim_99']} 维")

    # 两两余弦分布（不中心化，看原始相似度有多"挤"）
    S = X @ X.T
    n = S.shape[0]
    off = S[~torch.eye(n, dtype=torch.bool)]
    print(f"\n=== 两两余弦（原始空间）===")
    print(f"均值 {off.mean():.3f}  中位数 {off.median():.3f}  "
          f"5% 分位 {torch.quantile(off, 0.05):.3f}  95% 分位 {torch.quantile(off, 0.95):.3f}")

    # 中心化后的余弦（去掉共模，"真正"的方向差异）
    Xc = X - X.mean(0, keepdim=True)
    Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    S2 = Xc @ Xc.T
    off2 = S2[~torch.eye(n, dtype=torch.bool)]
    print(f"中心化后：均值 {off2.mean():.3f}  95% 分位 {torch.quantile(off2, 0.95):.3f}")


if __name__ == '__main__':
    main()
