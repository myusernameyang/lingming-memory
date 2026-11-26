# -*- coding: utf-8 -*-
"""用嵌入谱/近邻分析量三件事：

C2 近邻距离分布   —— 数据彼此有多挤
C3 干扰统计       —— 多少对文本余弦 > 0.9
中观基线          —— 用户重复说了同一件事多少次（记忆失败的可见信号）
"""
import json
import re
import torch
import zstandard as zstd
from sentence_transformers import SentenceTransformer

SESSION = (r"C:\Users\yangtianpeng\.dsh\sessions"
           r"\--C-Users-yangtianpeng-Documents-deepseek-harness-default-workspace--"
           r"\session-e3fc92dc-f029-4caf-8067-38fbf5b58f35\session.v4.jsonl.zstd")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def walk(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)


def collect():
    user_texts, all_texts = [], []
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
                    is_user = b'"user/message"' in ln
                    is_asst = b'"assistant/message"' in ln
                    if not (is_user or is_asst):
                        continue
                    try:
                        obj = json.loads(ln.decode('utf-8', 'ignore'))
                    except Exception:
                        continue
                    out = []
                    walk(obj.get('data', obj), out)
                    for t in out:
                        t = (t or '').strip()
                        if len(t) < 8 or len(t) > 300:
                            continue
                        if t.startswith('{') or '```' in t:
                            continue
                        if '（灵铭记忆' in t or 'Your parent agent id' in t:
                            continue
                        if not re.search(r'[\u4e00-\u9fff]', t):
                            continue
                        all_texts.append(t)
                        if is_user:
                            user_texts.append(t)
                        break
    def dedupe(xs):
        seen, out = set(), []
        for t in xs:
            k = t[:30]
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out
    return dedupe(user_texts), dedupe(all_texts)


def stats(name, X, thr_pairs=(0.9,)):
    S = X @ X.T
    n = S.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    off = S[~eye]
    print(f"\n=== {name}（{n} 条）===")
    print(f"两两余弦：均值 {off.mean():.3f}  中位 {off.median():.3f}  "
          f"5% {torch.quantile(off,0.05):.3f}  95% {torch.quantile(off,0.95):.3f}")
    for th in thr_pairs:
        mask = (S > th) & ~eye
        pairs = int(mask.sum().item()) // 2
        print(f"余弦 > {th} 的对数：{pairs}")
    # 每条文本的最近邻相似度
    S2 = S.clone(); S2.fill_diagonal_(-1)
    nn = S2.max(dim=1).values
    print(f"最近邻相似度：中位 {nn.median():.3f}  90% 分位 {torch.quantile(nn,0.9):.3f}  "
          f"最大 {nn.max():.3f}")
    return S, nn


def main():
    user_texts, all_texts = collect()
    print(f"用户消息 {len(user_texts)} 条，全部文本 {len(all_texts)} 条")

    m = SentenceTransformer('BAAI/bge-small-zh-v1.5', device=DEV)

    Xu = torch.tensor(m.encode(user_texts, normalize_embeddings=True, show_progress_bar=False),
                      dtype=torch.float32)
    S, nn = stats("用户消息", Xu)

    # 重复提问：最近邻 > 0.88 视为"同一件事说过多次"
    dup_idx = (nn > 0.88).nonzero(as_tuple=True)[0].tolist()
    print(f"\n疑似重复表述的用户消息：{len(dup_idx)} / {len(user_texts)}")

    Xa = torch.tensor(m.encode(all_texts, normalize_embeddings=True, show_progress_bar=False),
                      dtype=torch.float32)
    stats("全部文本", Xa)


if __name__ == '__main__':
    main()
