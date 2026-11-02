# -*- coding: utf-8 -*-
"""
灵铭记忆 · g 快权重服务（误差累加 + 碎片重建 + 匹配）
接口：
  POST /init      {"apiKey":"sk-xxx"} 训练坐标（对比学习，让 query 坐标靠近正确 fact 坐标）
  POST /write     {"text":"...","role":"user"} 误差累加进 g + 存记忆
  POST /retrieve  {"query":"...","topK":3} -> g@k 重建碎片 → 向量匹配 → {"items":[{text,role}]}
  POST /rl-step   {"query","fact","apiKey"} 实时 RL 微调坐标（可选）
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

D = 128        # 特征/坐标维度
LAMBDA = 0.9   # g 遗忘率
MODEL = "deepseek-chat"
ENDPOINT = "https://api.deepseek.com/chat/completions"
app = Flask(__name__)

# ============ 特征 embed（bigram hash 到 D 维） ============
def bigrams(t): return [t[i:i+2] for i in range(len(t)-1)]
def fnv(s):
    h = 0x811c9dc5
    for c in s.encode():
        h ^= c; h = (h * 0x01000193) & 0xffffffff
    return h
def embed(text):
    v = np.zeros(D, np.float32)
    for bg in bigrams(text):
        v[fnv(bg) % D] += 1.0
    n = np.linalg.norm(v)
    return torch.tensor(v/n if n > 0 else v, dtype=torch.float32)

# ============ 坐标网络（学出来） ============
class Coord(nn.Module):
    def __init__(self):
        super().__init__()
        self.T_net = nn.Linear(D, 3)
        self.W_expand = nn.Parameter(torch.randn(3, D) * 0.02)
    def forward(self, h):
        rtp = self.T_net(h).clamp(-5, 5)
        return F.normalize(rtp @ self.W_expand, dim=-1)

# ============ 全局状态 ============
coord = Coord()
g = torch.zeros(D, D)          # g 快权重矩阵
memory = []                    # [{text, role, h}]
g_file = "g.pt"; coord_file = "coord.pt"

def try_load():
    global g, coord
    if os.path.exists(coord_file):
        coord.load_state_dict(torch.load(coord_file, map_location="cpu"))
    if os.path.exists(g_file):
        g = torch.load(g_file, map_location="cpu")
        return True
    return False

def save():
    torch.save(coord.state_dict(), coord_file)
    torch.save(g, g_file)

def write_g(text):
    """误差累加进 g"""
    global g
    h = embed(text)
    k = coord(h)
    content = g @ k
    err = (h - h.mean()) - content
    err = F.normalize(err, dim=-1)
    g = LAMBDA * g + torch.outer(err, k)
    return h

@app.post("/write")
def write():
    text = request.json.get("text", "")
    role = request.json.get("role", "user")
    if not text:
        return jsonify({"ok": False})
    h = write_g(text)
    memory.append({"text": text, "role": role, "h": h})
    return jsonify({"ok": True, "size": len(memory)})

@app.post("/retrieve")
def retrieve():
    query = request.json.get("query", "")
    topK = int(request.json.get("topK", 3))
    if not query or not memory:
        return jsonify({"items": []})
    k_q = coord(embed(query))
    fragment = g @ k_q                 # g@k 重建碎片 [D]
    scores = [(i, torch.dot(fragment, m["h"]).item()) for i, m in enumerate(memory)]
    scores.sort(key=lambda x: -x[1])
    items = [{"text": memory[i]["text"], "role": memory[i]["role"]} for i, s in scores[:topK] if s > 0]
    return jsonify({"items": items})

@app.post("/init")
def init():
    """训练坐标网络（demo 数据，对比学习）"""
    data = json.load(open("data.json", encoding="utf-8"))
    facts, queries = data["facts"], data["queries"]
    correct = data.get("correct", list(range(len(queries))))
    hs = [embed(f) for f in facts]
    hqs = [embed(q) for q in queries]
    opt = torch.optim.Adam(coord.parameters(), lr=1e-2)
    for epoch in range(200):
        fc = torch.stack([coord(h) for h in hs])
        qc = torch.stack([coord(h) for h in hqs])
        sims = qc @ fc.T
        loss = F.cross_entropy(sims, torch.tensor(correct))
        opt.zero_grad(); loss.backward(); opt.step()
    save()
    return jsonify({"ok": True, "facts": len(facts)})

@app.post("/rl-step")
def rl_step():
    q = request.json.get("query", "")
    fact = request.json.get("fact", "")
    api_key = request.json.get("apiKey", "")
    if not q or not fact or not api_key:
        return jsonify({"error": "缺参数"}), 400
    score = api_score(api_key, q, fact)
    coord.train()
    opt = torch.optim.Adam(coord.parameters(), lr=1e-3)
    kq = coord(embed(q)); kf = coord(embed(fact))
    sim = torch.dot(kq, kf)
    loss = -score * sim
    opt.zero_grad(); loss.backward(); opt.step()
    coord.eval()
    return jsonify({"ok": True, "score": score})

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

if __name__ == "__main__":
    print("加载 g/坐标...", "成功" if try_load() else "未初始化（POST /init 生成坐标）", flush=True)
    print("g 快权重服务启动：http://127.0.0.1:9000", flush=True)
    app.run(host="127.0.0.1", port=9000, debug=False)
