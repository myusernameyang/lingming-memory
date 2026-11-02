# -*- coding: utf-8 -*-
import sys, requests, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:9000"

facts = [
    "灵元子用连续坐标场快权重做持久记忆，替代 KV 缓存。",
    "GPU 性能测试包括 p2p、all_reduce、gemm 基准。",
    "大模型 KV 缓存随序列长度线性增长，长上下文内存开销大。",
    "互补注意力用 attention 管现在、快权重 g 管过去。",
    "强化学习通过奖励和惩罚引导模型行为。",
    "向量数据库存 embedding 并做相似度检索，是 RAG 核心。",
]
queries = [
    "什么是灵元子？", "GPU 性能怎么测？", "KV 缓存有什么问题？",
    "什么是互补注意力？", "强化学习怎么引导模型？", "向量数据库是干什么的？",
]
correct = [0,1,2,3,4,5]

print("1. /init 训练坐标...")
r = requests.post(f"{BASE}/init", json={}, timeout=120).json()
print("   ", r)

print("2. /write 写入 facts（误差累加进 g）...")
for f in facts:
    requests.post(f"{BASE}/write", json={"text": f, "role": "user"}, timeout=15)

print("3. /retrieve 检索（g@k 碎片匹配）...")
hit = 0
for q, c in zip(queries, correct):
    r = requests.post(f"{BASE}/retrieve", json={"query": q, "topK": 3}, timeout=15).json()
    items = r.get("items", [])
    top = items[0]["text"] if items else "(空)"
    ok = items and items[0]["text"] == facts[c]
    hit += 1 if ok else 0
    print(f"   {q} -> {'✅' if ok else '❌'} {top[:30]}")

print(f"\n命中 {hit}/{len(queries)}")
