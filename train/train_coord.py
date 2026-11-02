# -*- coding: utf-8 -*-
"""
灵铭记忆 · 坐标训练脚本
用法：
  export LINGMING_API_KEY="sk-xxx"        # DeepSeek API key（打分用）
  python train_coord.py data.json          # data.json 里是 {"facts": [...], "queries": [...]}

原理：用 API 打分做 RL（REINFORCE + softmax 探索），训练一个「坐标网络」——
它学会把语义相关的内容放到相近的坐标，替代固定规则的 bigram，这就是"模型打分"。

产物：
  coord.pt    坐标网络权重（T_net + W_expand）
  vocab.json  bigram 词表（embed 服务复现 embedding 用）

我们只验证能跑；正式数据由使用者自己喂。
"""
import os, sys, re, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

torch.manual_seed(0)

API_KEY = os.environ.get("LINGMING_API_KEY", "")
MODEL = "deepseek-chat"
ENDPOINT = "https://api.deepseek.com/chat/completions"
OUT_DIM = 64

# ============ bigram embedding（输入特征，固定） ============
def bigrams(t): return [t[i:i+2] for i in range(len(t)-1)]
def build_embedder(texts):
    vocab = sorted({bg for t in texts for bg in bigrams(t)})
    idx = {bg: i for i, bg in enumerate(vocab)}
    def embed(t):
        v = np.zeros(len(vocab), np.float32)
        for bg in bigrams(t):
            if bg in idx:
                v[idx[bg]] += 1.0
        n = np.linalg.norm(v)
        return torch.tensor(v/n if n > 0 else v, dtype=torch.float32)
    return embed, vocab

# ============ 坐标网络（模型打分：text -> 坐标 k） ============
class CoordNet(nn.Module):
    def __init__(self, dim, out_dim=OUT_DIM):
        super().__init__()
        self.T_net = nn.Linear(dim, 3)
        self.W_expand = nn.Parameter(torch.randn(3, out_dim) * 0.02)
    def forward(self, h):
        rtp = self.T_net(h).clamp(-5, 5)
        return F.normalize(rtp @ self.W_expand, dim=-1)

# ============ API 批量打分 ============
def batch_score(qlist, fact_idx_list, facts):
    lines = [f"{i+1}. 问题「{qlist[i]}」检索到「{facts[fact_idx_list[i]]}」" for i in range(len(qlist))]
    prompt = (
        "下面有若干（问题，检索内容）对。请判断每对里检索内容能否回答该问题，"
        f"按顺序只输出 {len(qlist)} 个 0 到 1 之间的分数，用逗号分隔，例如：" + ",".join(["0.8"]*len(qlist)) + "\n" + "\n".join(lines)
    )
    for attempt in range(3):
        try:
            r = requests.post(ENDPOINT,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, timeout=60)
            txt = r.json()["choices"][0]["message"]["content"]
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
            return (nums + [0.0]*len(qlist))[:len(qlist)]
        except Exception as e:
            if attempt == 2:
                print(f"  [打分失败] {str(e)[:60]}")
            time.sleep(2)
    return [0.0]*len(qlist)

# ============ 主流程 ============
def main(data_path):
    if not API_KEY:
        print("请先设置 LINGMING_API_KEY（DeepSeek API key，打分用）")
        sys.exit(1)
    data = json.load(open(data_path, encoding="utf-8"))
    facts = data["facts"]; queries = data["queries"]
    embed, vocab = build_embedder(facts + queries)
    dim = len(vocab)
    print(f"facts {len(facts)} 条、queries {len(queries)} 条、bigram 维度 {dim}")

    net = CoordNet(dim)
    opt = torch.optim.Adam(net.parameters(), lr=2e-2)
    fc = torch.stack([net(embed(f)) for f in facts])

    def acc():
        with torch.no_grad():
            fc2 = torch.stack([net(embed(f)) for f in facts])
            qc = torch.stack([net(embed(q)) for q in queries])
            return (qc @ fc2.T).argmax(1).tolist()

    for epoch in range(40):
        fc = torch.stack([net(embed(f)) for f in facts])
        qc = torch.stack([net(embed(q)) for q in queries])
        sims = qc @ fc.T
        temp = max(0.3, 1.0 - epoch * 0.02)
        probs = F.softmax(sims / temp, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        scores = batch_score(queries, sampled.tolist(), facts)
        reward = torch.tensor(scores, dtype=torch.float32)
        log_probs = F.log_softmax(sims / temp, dim=-1)
        picked = log_probs[range(len(queries)), sampled]
        loss = -(reward * picked).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 8 == 0:
            print(f"epoch {epoch+1}: reward均值 {reward.mean().item():.2f}")

    # 保存产物
    torch.save(net.state_dict(), "coord.pt")
    json.dump(vocab, open("vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n已保存 coord.pt + vocab.json")
    # 验证：检索准确率（需要数据里有 correct 映射时打印）
    if "correct" in data:
        pred = acc()
        hit = sum(1 for p, c in zip(pred, data["correct"]) if p == c)
        print(f"检索命中 {hit}/{len(queries)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.json")
