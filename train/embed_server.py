# -*- coding: utf-8 -*-
"""
灵铭记忆 · 坐标 embed 服务（含初始化 + 实时 RL）
接口：
  POST /embed    {"text":"..."} -> {"k":[...]}               坐标
  POST /init     {"apiKey":"sk-xxx"} -> 一键训练初始坐标网络
  POST /rl-step  {"query":"...","fact":"...","apiKey":"..."} -> 实时 RL 一步（对话微调坐标）

插件自动拉起本服务；用户不直接跑脚本，初始化/微调都走 HTTP。
"""
import os, sys, re, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from flask import Flask, request, jsonify

OUT_DIM = 64
MODEL = "deepseek-chat"
ENDPOINT = "https://api.deepseek.com/chat/completions"
app = Flask(__name__)

# ============ 坐标网络 ============
class CoordNet(nn.Module):
    def __init__(self, dim, out_dim=OUT_DIM):
        super().__init__()
        self.T_net = nn.Linear(dim, 3)
        self.W_expand = nn.Parameter(torch.randn(3, out_dim) * 0.02)
    def forward(self, h):
        rtp = self.T_net(h).clamp(-5, 5)
        return F.normalize(rtp @ self.W_expand, dim=-1)

# ============ bigram ============
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

# ============ 全局状态 ============
net = None
embed = None
vocab = []

def try_load():
    global net, embed, vocab
    if os.path.exists("coord.pt") and os.path.exists("vocab.json"):
        vocab = json.load(open("vocab.json", encoding="utf-8"))
        embed, _ = build_embedder([])  # 占位，下面重建
        IDX = {bg: i for i, bg in enumerate(vocab)}
        def _embed(t):
            v = np.zeros(len(vocab), np.float32)
            for bg in bigrams(t):
                if bg in IDX:
                    v[IDX[bg]] += 1.0
            n = np.linalg.norm(v)
            return torch.tensor(v/n if n > 0 else v, dtype=torch.float32)
        embed = _embed
        net = CoordNet(len(vocab))
        net.load_state_dict(torch.load("coord.pt", map_location="cpu"))
        net.eval()
        return True
    return False

def api_score(api_key, query, fact):
    prompt = (f"问题是：{query}\n检索到的内容：{fact}\n"
              f"请判断该内容能否回答这个问题。只回答一个 0 到 1 之间的数，不要解释。")
    for _ in range(3):
        try:
            r = requests.post(ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, timeout=60)
            txt = r.json()["choices"][0]["message"]["content"]
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
            return (nums + [0.0])[0]
        except Exception:
            time.sleep(1)
    return 0.0

@app.post("/embed")
def emb():
    global net, embed
    if net is None:
        return jsonify({"k": [], "error": "未初始化"})
    text = request.json.get("text", "")
    if not text:
        return jsonify({"k": []})
    with torch.no_grad():
        k = net(embed(text))
    return jsonify({"k": k.tolist()})

@app.post("/init")
def init():
    """一键训练初始坐标网络（用内置 demo 数据 + API 打分）"""
    global net, embed, vocab
    api_key = request.json.get("apiKey", "")
    if not api_key:
        return jsonify({"error": "缺 apiKey"}), 400
    data = json.load(open("data.json", encoding="utf-8"))
    facts, queries = data["facts"], data["queries"]
    embed, vocab = build_embedder(facts + queries)
    net = CoordNet(len(vocab))
    opt = torch.optim.Adam(net.parameters(), lr=2e-2)
    for epoch in range(30):
        fc = torch.stack([net(embed(f)) for f in facts])
        qc = torch.stack([net(embed(q)) for q in queries])
        sims = qc @ fc.T
        temp = max(0.3, 1.0 - epoch * 0.02)
        probs = F.softmax(sims / temp, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        scores = [api_score(api_key, queries[i], facts[sampled[i].item()]) for i in range(len(queries))]
        reward = torch.tensor(scores, dtype=torch.float32)
        log_probs = F.log_softmax(sims / temp, dim=-1)
        picked = log_probs[range(len(queries)), sampled]
        loss = -(reward * picked).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(net.state_dict(), "coord.pt")
    json.dump(vocab, open("vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
    net.eval()
    return jsonify({"ok": True, "facts": len(facts)})

@app.post("/rl-step")
def rl_step():
    """实时 RL 一步：query 对 fact 打分，微调坐标网络"""
    global net, embed
    if net is None:
        return jsonify({"error": "未初始化"}), 400
    q = request.json.get("query", "")
    fact = request.json.get("fact", "")
    api_key = request.json.get("apiKey", "")
    if not q or not fact or not api_key:
        return jsonify({"error": "缺参数"}), 400
    score = api_score(api_key, q, fact)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    qc = net(embed(q))
    fc = net(embed(fact))
    sim = torch.dot(qc, fc)
    # REINFORCE：score 高 → 拉近；score 低 → 推远
    loss = -score * sim
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return jsonify({"ok": True, "score": score})

if __name__ == "__main__":
    print("加载坐标...", "成功" if try_load() else "未初始化（POST /init 生成）", flush=True)
    print("坐标 embed 服务启动：http://127.0.0.1:9000", flush=True)
    app.run(host="127.0.0.1", port=9000, debug=False)
