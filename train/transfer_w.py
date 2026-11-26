# -*- coding: utf-8 -*-
"""迁移性测试：在合成模板上学的度量 M，能不能用在别处？

训练：模板 A（服务器的核数是 32 核，代号 KX00001，位于杭州。 / 查询带代号）
测试三档：
  T1 同模板（对照）
  T2 换表述的模板（同一批实体，不同句式）
  T3 真实语料（会话日志里的真实段落，用唯一 token 当查询）
"""
import json
import re
import torch
import torch.nn.functional as F
import zstandard as zstd
from sentence_transformers import SentenceTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"
SESSION = (r"C:\Users\yangtianpeng\.dsh\sessions"
           r"\--C-Users-yangtianpeng-Documents-deepseek-harness-default-workspace--"
           r"\session-e3fc92dc-f029-4caf-8067-38fbf5b58f35\session.v4.jsonl.zstd")

SUBJ = ["服务器", "数据库", "缓存", "备份", "监控", "网关", "队列", "日志", "镜像", "构建机"]
ATTR = ["核数", "内存", "磁盘", "带宽", "系统", "端口"]
VAL = ["32 核", "128G", "4T", "500Mbps", "Ubuntu", "8443"]
LOC = ["杭州", "北京", "上海", "成都", "深圳", "广州", "南京", "武汉", "西安", "苏州"]


def make_A(n):
    f, q = [], []
    for i in range(n):
        s, a, v, l = SUBJ[i % 10], ATTR[(i // 10) % 6], VAL[(i // 10) % 6], LOC[i % 10]
        c = f"KX{i:05d}"
        f.append(f"{s}的{a}是 {v}，代号 {c}，位于{l}。")
        q.append(f"代号 {c} 那台{s}的{a}是什么")
    return f, q


def make_B(n):
    """同实体、换句式（更接近口语）"""
    f, q = [], []
    for i in range(n):
        s, a, v, l = SUBJ[i % 10], ATTR[(i // 10) % 6], VAL[(i // 10) % 6], LOC[i % 10]
        c = f"KX{i:05d}"
        f.append(f"编号 {c} 的那台机器放在{l}，是{s}，{a}给到了 {v}。")
        q.append(f"{c} 的配置是啥")
    return f, q


def collect_real(limit=400):
    def walk(o, out):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, out)
        elif isinstance(o, list):
            for v in o:
                walk(v, out)
    texts = []
    with open(SESSION, 'rb') as fh:
        d = zstd.ZstdDecompressor()
        with d.stream_reader(fh) as r:
            buf = b''
            while True:
                ch = r.read(1 << 20)
                if not ch:
                    break
                buf += ch
                *lines, buf = buf.split(b'\n')
                for ln in lines:
                    if b'"assistant/message"' not in ln and b'"user/message"' not in ln:
                        continue
                    try:
                        obj = json.loads(ln.decode('utf-8', 'ignore'))
                    except Exception:
                        continue
                    out = []
                    walk(obj.get('data', obj), out)
                    for t in out:
                        t = (t or '').strip()
                        if 60 <= len(t) <= 220 and '```' not in t and '（灵铭记忆' not in t:
                            if re.search(r'[\u4e00-\u9fff]', t):
                                texts.append(t)
                                break
    # 去重后取前 limit 条
    seen, uniq = set(), []
    for t in texts:
        k = t[:30]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq[:limit]


def hit(Fx, Qx, idx, M=None, k=5):
    if M is not None:
        Fx = F.normalize(Fx @ M, dim=-1)
        Qx = F.normalize(Qx @ M, dim=-1)
    else:
        Fx = F.normalize(Fx, dim=-1)
        Qx = F.normalize(Qx, dim=-1)
    S = Qx @ Fx.T
    top = S.topk(k, dim=1).indices
    h1 = (top[:, 0] == idx).float().mean().item()
    hk = (top == idx.unsqueeze(1)).any(1).float().mean().item()
    return h1, hk


def main():
    enc = SentenceTransformer(BGE, device=DEV)
    E = lambda xs: torch.tensor(enc.encode(xs, normalize_embeddings=True, show_progress_bar=False),
                                dtype=torch.float32, device=DEV)

    # 训练集：模板 A
    fA, qA = make_A(2000)
    Fa, Qa = E(fA), E(qA)

    M = torch.nn.Parameter(torch.eye(512, 32, device=DEV) * 0.1)
    opt = torch.optim.Adam([M], lr=1e-3)
    for _ in range(400):
        q = F.normalize(Qa @ M, dim=-1)
        f = F.normalize(Fa @ M, dim=-1)
        loss = F.cross_entropy(q @ f.T / 0.1, torch.arange(len(fA), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    Mv = M.detach()

    print("训练完成（2000 对，r=32）\n")
    print(f"{'测试集':10s} {'基线 hit@1':>10s} {'学M hit@1':>10s} {'基线 hit@5':>10s} {'学M hit@5':>10s}")

    # T1 同模板（后 200 条，未见）
    f, q = make_A(2000 + 200)
    Fx, Qx = E(f), E(q)
    idx = torch.arange(2000, 2200, device=DEV)
    b1, b5 = hit(Fx, Qx[idx], idx)
    m1, m5 = hit(Fx, Qx[idx], idx, M=Mv)
    print(f"{'T1 同模板':10s} {b1:10.3f} {m1:10.3f} {b5:10.3f} {m5:10.3f}")

    # T2 换表述
    f2, q2 = make_B(400)
    Fx2, Qx2 = E(f2), E(q2)
    idx2 = torch.arange(400, device=DEV)
    b1, b5 = hit(Fx2, Qx2, idx2)
    m1, m5 = hit(Fx2, Qx2, idx2, M=Mv)
    print(f"{'T2 换表述':10s} {b1:10.3f} {m1:10.3f} {b5:10.3f} {m5:10.3f}")

    # T3 真实语料：用唯一 token 当查询
    real = collect_real(400)
    tok_re = re.compile(r'[A-Za-z][A-Za-z0-9_.\-/]{3,}')
    cands = []
    for t in real:
        for tok in set(tok_re.findall(t)):
            cands.append((tok, t))
    seen = {}
    for tok, t in cands:
        seen.setdefault(tok, set()).add(t)
    pairs = [(tok, list(ts)[0]) for tok, ts in seen.items() if len(ts) == 1]
    pairs = pairs[:200]
    print(f"\n真实语料：{len(real)} 段，可用唯一 token 查询 {len(pairs)} 个")
    if pairs:
        docs = real
        didx = {t: i for i, t in enumerate(docs)}
        Fx3 = E(docs)
        Qx3 = E([f"关于 {tok} 的记录是什么" for tok, _ in pairs])
        idx3 = torch.tensor([didx[t] for _, t in pairs], device=DEV)
        b1, b5 = hit(Fx3, Qx3, idx3)
        m1, m5 = hit(Fx3, Qx3, idx3, M=Mv)
        print(f"{'T3 真实语料':10s} {b1:10.3f} {m1:10.3f} {b5:10.3f} {m5:10.3f}")


if __name__ == "__main__":
    main()
