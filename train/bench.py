# -*- coding: utf-8 -*-
"""灵铭召回基准：测"连续性"与"越用越精准"

不经过 DSH / 插件 / HTTP，直接用 LingmingMemory：
  1. 灌入 N 条带唯一标识的事实（记忆库）
  2. 用 M 个改写过的问句去召回，统计 hit@1 / hit@3 / MRR
  3. 每轮把"召回正确的"标记为 used、"召回错误的"标记为 unused，喂 feedback
  4. 重复 R 轮，看指标是否随使用提升

用法：
  python bench.py [facts] [rounds]
"""
import sys, random, time
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from memory_core import LingmingMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BGE = "BAAI/bge-small-zh-v1.5"

_enc = None
def encode(text):
    global _enc
    if _enc is None:
        _enc = SentenceTransformer(BGE, device=DEV)
    e = _enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(e[0], dtype=torch.float32, device=DEV)

# ---- 合成事实：每条有一个唯一代号，问句不出现代号 ----
SUBJECTS = [
    ("服务器", "杭州机房"), ("数据库", "北京亦庄"), ("缓存", "阿里云上海区"),
    ("备份", "成都双流"), ("监控", "深圳南山"), ("网关", "广州天河"),
    ("队列", "南京江宁"), ("日志", "武汉光谷"), ("镜像仓库", "西安高新"),
    ("构建机", "苏州工业园"),
]
ATTRS = [
    ("核数", "32 核"), ("内存", "128G"), ("磁盘", "4T NVMe"),
    ("带宽", "500Mbps"), ("系统", "Ubuntu 22.04"), ("端口", "8443"),
]


def make_facts(n, qmode="old"):
    """生成 n 条事实：主谓+唯一代号，保证互相可区分

    qmode:
      "old" —— query 只含 subject。同一 subject 有 len(ATTRS) 条事实,
               但 query 没有任何信息能区分它们 -> hit@1 天花板 = 1/6。
               （_diag_cap.py 实测: hit@1 恒为 0.167, hit@3 恒为 0.500）
      "new" —— query 含 attr,唯一指向一条事实。用它才能测机制。
    """
    facts = []
    for i in range(n):
        subj, loc = SUBJECTS[i % len(SUBJECTS)]
        attr, val = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
        code = f"KX{i:03d}"
        if qmode == "new":
            query = f"{subj}的{attr}是什么，代号是什么"
        else:
            query = f"我们的{subj}配置怎么样，代号是什么"
        facts.append({
            "code": code,
            "text": f"{subj}的{attr}是 {val}，代号 {code}，位于{loc}。",
            "query": query,
        })
    return facts


def run(n_facts=60, rounds=5, k_cand=16, qmode="old"):
    random.seed(0)
    facts = make_facts(n_facts, qmode=qmode)
    enc_t0 = time.time()
    mem = LingmingMemory(encode, d=512, c_mult=4, k=k_cand,
                         capacity=max(10000, n_facts * 4), device=DEV,
                         center=False, write_scale=0.1,
                         gate_bias=0.2, gate_temp=0.05, read_top=3)
    for f in facts:
        mem.add(f["text"], "user")
    print(f"灌入 {len(facts)} 条事实，用时 {time.time()-enc_t0:.1f}s，库内 {mem.n} 段\n")

    def evaluate():
        hit1 = hit3 = 0
        mrr = 0.0
        wrong = []
        for f in facts:
            items = mem.retrieve(f["query"], topK=3)
            codes = []
            for it in items:
                # 从召回文本里找回代号
                import re
                m = re.search(r"KX\d{3}", it["text"])
                codes.append(m.group(0) if m else "")
            if f["code"] in codes:
                rank = codes.index(f["code"]) + 1
                mrr += 1.0 / rank
                if rank == 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
            else:
                wrong.append(f["code"])
        n = len(facts)
        return hit1 / n, hit3 / n, mrr / n, wrong

    h1, h3, mrr, _ = evaluate()
    print(f"{'轮次':>4}  {'hit@1':>7}  {'hit@3':>7}  {'MRR':>7}")
    print(f"{'初始':>4}  {h1:7.3f}  {h3:7.3f}  {mrr:7.3f}")

    for r in range(1, rounds + 1):
        # 模拟"用后反馈"：正确命中的记忆 +，未命中的也按真实使用给负反馈
        for f in facts:
            items = mem.retrieve(f["query"], topK=3)
            import re
            for it in items:
                m = re.search(r"KX\d{3}", it["text"])
                code = m.group(0) if m else ""
                used = (code == f["code"])
                mem.feedback([(it["mid"], used)])
        h1, h3, mrr, wrong = evaluate()
        print(f"{r:>4}  {h1:7.3f}  {h3:7.3f}  {mrr:7.3f}")
        if r == rounds and wrong:
            print(f"\n仍未召回的 {len(wrong)} 条：{wrong[:10]}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    r = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    q = sys.argv[3] if len(sys.argv) > 3 else "new"
    print(f"[qmode={q}]  (old=旧 query, hit@1 天花板 1/6; new=含 attr，可测机制)")
    run(n, r, qmode=q)
