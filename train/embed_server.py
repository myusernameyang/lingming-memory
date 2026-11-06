# -*- coding: utf-8 -*-
"""灵铭记忆服务：坐标寻址多级向量库
接口：
  POST /write    {"text":"...","role":"user"}   存记忆 → 坐标 top-k 分桶
  POST /retrieve {"query":"...","topK":3}       查 → 坐标桶交集 → 命中最相关
  POST /save                                     保存坐标权重（coord.pt）
  GET  /status                                   状态（记忆数 / 桶数）
"""
import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from memory_core import CNNEncoder, LingmingMemory

D = int(os.environ.get("LINGMING_D", "128"))          # 模型维度
C = int(os.environ.get("LINGMING_C_MULT", "4")) * D   # 坐标 bar 数 = 横向倍数 × d
K = int(os.environ.get("LINGMING_K", "8"))            # 稀疏 top-k
CAPACITY = int(os.environ.get("LINGMING_CAPACITY", "10000"))  # 向量库容量（记忆条数上限）
L = 64                                                # 单条文本截断长度（字节）
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coord.pt")

encoder = CNNEncoder(D)
W_coord = nn.Linear(D, C, bias=False)
if os.path.exists(CKPT):
    state = torch.load(CKPT, map_location="cpu")
    enc = state.get("encoder", {})
    w = state.get("W_coord", state.get("W", None))
    if isinstance(w, dict):
        encoder.load_state_dict(enc, strict=False)
        W_coord.load_state_dict(w, strict=False)
    else:
        # 兼容旧格式：直接是 coord.pt 存 W_coord
        W_coord.load_state_dict({"weight": state}, strict=False)
    print("已加载坐标权重 coord.pt", flush=True)
else:
    print("未找到 coord.pt，用随机坐标（先用 train_coord.py 训练）", flush=True)

mem = LingmingMemory(encoder, W_coord, k=K, L=L, capacity=CAPACITY)

app = Flask(__name__)


@app.post("/write")
def write():
    d = request.get_json(force=True, silent=True) or {}
    text = str(d.get("text", "")).strip()
    role = str(d.get("role", "user"))
    if not text:
        return jsonify({"ok": False})
    mid = mem.add(text, role)
    return jsonify({"ok": True, "id": mid})


@app.post("/retrieve")
def retrieve():
    d = request.get_json(force=True, silent=True) or {}
    query = str(d.get("query", "")).strip()
    topK = int(d.get("topK", 3))
    if not query:
        return jsonify({"items": []})
    items = mem.retrieve(query, topK)
    return jsonify({"items": items})


@app.post("/save")
def save():
    torch.save({"encoder": encoder.state_dict(), "W_coord": W_coord.state_dict()}, CKPT)
    return jsonify({"ok": True, "path": CKPT})


@app.get("/status")
def status():
    return jsonify(mem.status())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000)
