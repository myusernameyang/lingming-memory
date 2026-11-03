# -*- coding: utf-8 -*-
"""
灵铭记忆 · 新机制：灵元子预测对话（外部锚）+ 误差驱动学习
接口：
  POST /write    {"text":"..."}   喂对话 → CE 预测 → 反向更新（学）
  POST /retrieve {"query":"..."}  喂 query → 生成 → 吐出它的理解
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from remote_think import Model

# 4 层本地灵元子（试用版 d=128；正式版 d=768 训练好的 / seq=2048）
D, N_LAYERS, H, SEQ = 128, 4, 8, 256
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingyuanzi_mem.pt")

torch.manual_seed(0)
m = Model(d=D, c=32, M=64, n_layers=N_LAYERS, H=H, seq=SEQ, paper_len=0, causal=True)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

if os.path.exists(CKPT):
    try:
        m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
        print("已加载记忆模型", flush=True)
    except Exception:
        pass


def decode(bs):
    return bytes(bs).decode("utf-8", errors="replace")


@torch.no_grad()
def generate(prompt, length=40):
    out = list(prompt)
    for _ in range(length):
        x = torch.tensor([out[-SEQ:]], dtype=torch.long)
        logits, g, t_g = m(x)
        p = torch.softmax(logits[0, -1, 0] / 0.8, dim=-1)
        out.append(int(torch.multinomial(p, 1).item()))
    return out


app = Flask(__name__)


@app.post("/write")
def write():
    """喂对话 → CE 预测下一字节 → 误差反向更新（学）"""
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"ok": False})
    bs = list(text.encode())[:SEQ]
    x = torch.tensor([bs], dtype=torch.long)
    y = torch.tensor([bs[1:] + [0]], dtype=torch.long)
    opt.zero_grad()
    logits, g, t_g = m(x)
    loss = F.cross_entropy(logits[:, :-1, 0, :].reshape(-1, 256), y[:, :-1].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.g_init.data = g.mean(dim=0).detach().float()
    m.t_g_init.data = t_g.detach().float()
    return jsonify({"ok": True, "loss": round(loss.item(), 4)})


@app.post("/retrieve")
def retrieve():
    """喂 query → 生成 → 吐出它的理解"""
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"text": ""})
    seed = list(query.encode())[:SEQ]
    out = generate(seed)
    return jsonify({"text": decode(out)})


@app.post("/save")
def save():
    torch.save(m.state_dict(), CKPT)
    return jsonify({"ok": True})


HTML = """
<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>灵元子记忆 · 预测对话</title>
<style>
body{font-family:sans-serif;max-width:720px;margin:24px auto;padding:0 16px;background:#111;color:#eee}
h2{color:#7ec8ff}
.card{background:#1a1a2e;border-radius:10px;padding:14px;margin:12px 0}
input{width:70%;padding:10px;font-size:15px;border-radius:6px;border:1px solid #444;background:#222;color:#eee}
button{padding:10px 16px;margin-left:6px;border-radius:6px;border:none;background:#2a7fff;color:#fff;cursor:pointer;font-size:15px}
button:hover{background:#1e6ae0}
.mono{font-family:monospace;white-space:pre-wrap;word-break:break-all}
#lossList{max-height:180px;overflow:auto;font-family:monospace;font-size:12px;color:#9f9}
#output{font-size:16px;color:#ffd98a}
.lbl{color:#888;font-size:13px;margin:6px 0}
</style></head><body>
<h2>灵元子记忆 · 预测对话（外部锚）</h2>
<div class="card">
  <div class="lbl">喂对话（它学）——模型预测这句话的下一字节，误差反向更新</div>
  <input id="text" placeholder="喂一句对话，比如：你好呀，今天聊聊篮球">
  <button onclick="write()">喂</button>
  <div class="lbl">最近 loss：<span id="loss">-</span>（越低=学得越准）</div>
  <div id="lossList"></div>
</div>
<div class="card">
  <div class="lbl">给它一个开头（它接）——看它"说"出来的是什么</div>
  <input id="query" placeholder="比如：你好">
  <button onclick="retrieve()">生成</button>
  <div class="lbl">它吐的：</div>
  <div id="output" class="mono">-</div>
</div>
<script>
async function write(){
  const t=document.getElementById('text').value.trim(); if(!t)return;
  const r=await fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  const j=await r.json();
  document.getElementById('loss').textContent=j.loss;
  document.getElementById('lossList').innerHTML+='<div>'+j.loss+'</div>';
  document.getElementById('text').value='';
}
async function retrieve(){
  const q=document.getElementById('query').value.trim(); if(!q)return;
  const r=await fetch('/retrieve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
  const j=await r.json();
  document.getElementById('output').textContent=j.text;
}
</script></body></html>
"""


@app.get("/")
def index():
    return HTML


if __name__ == "__main__":
    print(f"灵元子记忆服务（预测对话）启动：http://127.0.0.1:9000  d={D} {N_LAYERS}层 seq={SEQ}", flush=True)
    app.run(host="127.0.0.1", port=9000, debug=False)
