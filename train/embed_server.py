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
    global g, coord, memory
    if os.path.exists(coord_file):
        coord.load_state_dict(torch.load(coord_file, map_location="cpu"))
    if os.path.exists(g_file):
        g = torch.load(g_file, map_location="cpu")
    if os.path.exists("memory.json"):
        data = json.load(open("memory.json", encoding="utf-8"))
        memory = [{"text": d["text"], "role": d.get("role", "user"), "h": embed(d["text"])} for d in data]
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
    try:
        json.dump([{"text": m["text"], "role": m["role"]} for m in memory],
                  open("memory.json", "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return jsonify({"ok": True, "size": len(memory)})

@app.post("/retrieve")
def retrieve():
    query = request.json.get("query", "")
    topK = int(request.json.get("topK", 3))
    if not query or not memory:
        return jsonify({"items": []})
    h_q = embed(query)
    k_q = coord(h_q)
    fragment = g @ k_q                          # g@k 读出的叠加碎片 [D]
    completed = F.normalize(h_q + fragment, dim=-1)   # 碎片完善上下文（不是精确匹配一句）
    scores = [(i, torch.dot(completed, m["h"]).item()) for i, m in enumerate(memory)]
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

@app.post("/reinit")
def reinit():
    """用当前 memory（用户真实对话）自监督重训坐标：相邻消息是正样本，隔开的是负样本"""
    global coord, g
    texts = [m["text"] for m in memory]
    if len(texts) < 3:
        return jsonify({"error": "记忆太少（<3 条）"}), 400
    hs = [m["h"] for m in memory]
    opt = torch.optim.Adam(coord.parameters(), lr=2e-2)
    for epoch in range(150):
        loss = torch.tensor(0.0)
        for i in range(len(hs) - 1):
            pos = torch.dot(coord(hs[i]), coord(hs[i + 1]))          # 相邻 = 正样本
            neg_idx = (i + len(hs) // 2) % len(hs)                    # 隔开 = 负样本
            neg = torch.dot(coord(hs[i]), coord(hs[neg_idx]))
            loss = loss - (pos - neg)
        opt.zero_grad(); loss.backward(); opt.step()
    coord.eval()
    # 用新坐标重写 g
    g = torch.zeros(D, D)
    for m in memory:
        k = coord(m["h"])
        content = g @ k
        err = (m["h"] - m["h"].mean()) - content
        err = F.normalize(err, dim=-1)
        g = LAMBDA * g + torch.outer(err, k)
    save()
    return jsonify({"ok": True, "n": len(texts)})

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

def teacher_rank(api_key, query, history):
    """老师（带完整上下文的 agent）给每条历史打相关分 0~1，形成打分规则"""
    prompt = ("你是一个记忆检索老师。下面是对话历史（用户之前说过的话，每条一行）：\n"
              + "\n".join(f"{i}. {h}" for i, h in enumerate(history))
              + f"\n\n当前用户问题：{query}\n\n"
              "请给每条历史打一个 0~1 的相关分（表示它对回答当前问题有多相关），"
              f"按顺序输出 {len(history)} 个分数，用逗号分隔，只输出数字，不要解释。例如：0.9,0.0,0.3")
    for _ in range(3):
        try:
            r = requests.post(ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, timeout=120)
            txt = r.json()["choices"][0]["message"]["content"]
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
            if len(nums) >= len(history):
                return nums[:len(history)]
            if len(nums) >= 2:   # 数量不够就截断/补零
                return nums + [0.0] * (len(history) - len(nums))
        except Exception:
            time.sleep(1)
    return [0.0] * len(history)

@app.post("/distill")
def distill():
    """老师（全上下文）打分 → 坐标学习这套打分规则（蒸馏）"""
    query = request.json.get("query", "")
    api_key = request.json.get("apiKey", "")
    if not query or not api_key or len(memory) < 2:
        return jsonify({"error": "缺参数或记忆太少"}), 400
    history = [m["text"] for m in memory]
    scores = teacher_rank(api_key, query, history)          # 老师打分规则
    coord.train()
    opt = torch.optim.Adam(coord.parameters(), lr=2e-3)
    k_q = coord(embed(query))
    loss = torch.tensor(0.0)
    for m, s in zip(memory, scores):
        sim = torch.dot(k_q, coord(m["h"]))
        loss = loss + (sim - s) ** 2                          # 让坐标 sim 逼近老师分数
    opt.zero_grad(); loss.backward(); opt.step()
    coord.eval()
    return jsonify({"ok": True, "scores": [round(s, 2) for s in scores]})

if __name__ == "__main__":
    print("加载 g/坐标...", "成功" if try_load() else "未初始化（POST /init 生成坐标）", flush=True)
    print("g 快权重服务启动：http://127.0.0.1:9000", flush=True)
    app.run(host="127.0.0.1", port=9000, debug=False)
