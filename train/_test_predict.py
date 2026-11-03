# -*- coding: utf-8 -*-
"""测试新 embed_server：喂对话（write）→ 学 → 生成理解（retrieve）"""
import os, sys, time, subprocess, requests, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE = "http://127.0.0.1:9000"
server = subprocess.Popen(
    [sys.executable, "embed_server.py"],
    cwd=r"C:\Users\yangtianpeng\Documents\ChatGPT\lingming-memory\train",
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("起服务中...", flush=True)
for _ in range(20):
    try:
        requests.get(BASE + "/", timeout=1); break
    except Exception:
        time.sleep(1)

# 喂对话（外部锚）
conversation = "你好呀，我今天想聊聊篮球，你喜欢打篮球吗？"
print(f"喂对话：{conversation}", flush=True)
for i in range(10):
    r = requests.post(BASE + "/write", json={"text": conversation}, timeout=30)
    if i % 2 == 0:
        print(f"  第{i}次喂 → loss={r.json().get('loss')}", flush=True)

# 生成理解
print("\n生成（吐它的理解）：", flush=True)
r = requests.post(BASE + "/retrieve", json={"query": "你好"}, timeout=30)
print(f"  它吐的：{r.json().get('text')}", flush=True)

server.kill()
print("done", flush=True)
