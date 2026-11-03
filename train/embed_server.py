# -*- coding: utf-8 -*-
"""
灵铭记忆 · 新机制：灵元子预测对话（外部锚）+ 误差驱动学习
接口：
  POST /write    {"text":"..."}   喂对话 → 切段 CE 预测 → 反向更新（学），记入历史
  POST /retrieve {"query":"..."}  喂 query → 生成 → 吐出它的理解
  GET  /history                   最近的喂入历史（可视化窗口自动刷新用）
  GET  /status                    状态（喂入次数 + g 范数）
  POST /save                      保存（权重 + g 状态）
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

history = []   # 喂入历史：{text, loss, chunks, ts}（可视化窗口展示用）
stats = {"total_writes": 0}   # 总喂入次数（历史会裁剪，这个不裁）


def record(text, loss, chunks):
    history.append({"text": text[:80], "loss": loss, "chunks": chunks, "ts": time.time()})
    if len(history) > 200:
        del history[: len(history) - 200]


@app.post("/write")
def write():
    """喂对话 → 切段 CE 预测下一字节 → 误差反向更新（学）"""
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"ok": False})
    bs = list(text.encode())
    total_loss = 0.0
    n_chunks = 0
    for i in range(0, len(bs), SEQ):          # 切成 SEQ 一段段喂，不截断、不浪费
        chunk = bs[i:i + SEQ]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk], dtype=torch.long)
        y = torch.tensor([chunk[1:] + [0]], dtype=torch.long)
        opt.zero_grad()
        logits, g, t_g = m(x)
        loss = F.cross_entropy(logits[:, :-1, 0, :].reshape(-1, 256), y[:, :-1].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        m.g_init.data = g.mean(dim=0).detach().float()
        m.t_g_init.data = t_g.detach().float()
        total_loss += loss.item()
        n_chunks += 1
    avg_loss = round(total_loss / max(n_chunks, 1), 4)
    record(text, avg_loss, n_chunks)
    stats["total_writes"] += 1
    return jsonify({"ok": True, "loss": avg_loss, "chunks": n_chunks})


@app.post("/retrieve")
def retrieve():
    """喂 query → 生成 → 吐出它的理解"""
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"text": ""})
    seed = list(query.encode())[:SEQ]
    out = generate(seed)
    return jsonify({"text": decode(out)})


@app.get("/history")
def get_history():
    return jsonify({"history": history[-50:]})


@app.get("/status")
def status():
    """状态：总喂入次数 + g 快权重范数（g 是持久记忆，state_dict 里就带着它）"""
    return jsonify({
        "total_writes": stats["total_writes"],
        "g_norm": round(m.g_init.norm().item(), 2),
        "t_g_norm": round(m.t_g_init.norm().item(), 2),
        "d": D, "layers": N_LAYERS, "seq": SEQ,
    })


@app.post("/save")
def save():
    """保存：m.state_dict() 已含 g_init / t_g_init（快权重记忆），一起持久化"""
    torch.save(m.state_dict(), CKPT)
    return jsonify({"ok": True, "g_norm": round(m.g_init.norm().item(), 2)})


HTML = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>灵元子记忆 · 预测对话</title>
<style>
body{font-family:sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#111;color:#eee}
h2{color:#7ec8ff}
.card{background:#1a1a2e;border-radius:10px;padding:14px;margin:12px 0}
input[type=text],input[type=file]{width:70%;padding:10px;font-size:14px;border-radius:6px;border:1px solid #444;background:#222;color:#eee;margin-bottom:6px}
button{display:inline-block;padding:9px 16px;margin:4px 6px 4px 0;border-radius:6px;border:none;background:#2a7fff;color:#fff;cursor:pointer;font-size:14px}
button:hover{background:#1e6ae0}
.mono{font-family:monospace;white-space:pre-wrap;word-break:break-all}
#feed{max-height:280px;overflow:auto;font-size:13px}
.feed-item{padding:4px 0;border-bottom:1px solid #2a2a3e}
.feed-loss{color:#9f9;font-family:monospace}
.feed-text{color:#ccc}
#output{font-size:16px;color:#ffd98a}
.lbl{color:#888;font-size:13px;margin:6px 0}
</style></head><body>
<h2>灵元子记忆 · 预测对话（外部锚）</h2>
<div class="card">
  <div class="lbl">后台喂入（DSH 聊的都会流进来，2 秒自动刷新）</div>
  <div id="feed">加载中...</div>
</div>
<div class="card">
  <div class="lbl">上传文件喂（JSON / txt / md，自动解析 + 切段）</div>
  <input type="file" id="file" accept=".json,.txt,.jsonl,.md">
  <button type="button" onclick="uploadFile()">上传并喂</button>
  <div class="lbl" id="uploadStatus">-</div>
</div>
<div class="card">
  <div class="lbl">手动喂一段（它学）——长文本自动切段，不浪费</div>
  <input type="text" id="text" placeholder="喂一段对话">
  <button type="button" onclick="feedText()">喂</button>
  <div class="lbl">最近 loss：<span id="loss">-</span>（越低 = 学得越准）</div>
</div>
<div class="card">
  <div class="lbl">给它一个开头（它接）——看它说出来的</div>
  <input type="text" id="query" placeholder="比如：你好">
  <button type="button" onclick="retrieve()">生成</button>
  <div class="lbl">它吐的：</div>
  <div id="output" class="mono">-</div>
</div>
<script>
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function refreshFeed(){
  try{
    var r=await fetch('/history');
    var j=await r.json();
    var feed=document.getElementById('feed');
    var items=j.history.slice().reverse();
    if(!items.length){feed.innerHTML='<div class="lbl">还没有喂入，去 DSH 聊天或上传文件</div>';return;}
    feed.innerHTML=items.map(function(h){
      return '<div class="feed-item"><span class="feed-loss">loss='+h.loss+'</span> x'+h.chunks+'段 <span class="feed-text">'+esc(h.text)+'</span></div>';
    }).join('');
  }catch(e){}
}
async function feedText(){
  try{
    var t=document.getElementById('text').value.trim(); if(!t)return;
    var r=await fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
    var j=await r.json();
    document.getElementById('loss').textContent=j.loss;
    document.getElementById('text').value='';
    refreshFeed();
  }catch(e){}
}
async function uploadFile(){
  try{
    var f=document.getElementById('file').files[0];
    if(!f){document.getElementById('uploadStatus').textContent='请先选文件';return;}
    var text=await f.text();
    var content=text;
    try{
      var obj=JSON.parse(text);
      if(Array.isArray(obj)){
        content=obj.map(function(x){
          if(typeof x==='string')return x;
          if(x&&typeof x==='object')return x.content||x.text||x.message||'';
          return '';
        }).filter(function(t){return t;}).join('\\n');
      }else if(obj&&typeof obj==='object'){
        if(obj.text||obj.content){content=obj.text||obj.content;}
        else if(Array.isArray(obj.messages)){content=obj.messages.map(function(x){return x.content||x.text||'';}).join('\\n');}
      }
    }catch(e){}
    content=content.trim();
    if(!content){document.getElementById('uploadStatus').textContent='解析后内容为空';return;}
    document.getElementById('uploadStatus').textContent='正在喂 '+content.length+' 字节...';
    var r=await fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:content})});
    var j=await r.json();
    document.getElementById('uploadStatus').textContent='喂完：'+j.chunks+' 段，平均 loss='+j.loss;
    refreshFeed();
  }catch(e){document.getElementById('uploadStatus').textContent='出错：'+e.message;}
}
async function retrieve(){
  try{
    var q=document.getElementById('query').value.trim(); if(!q)return;
    var r=await fetch('/retrieve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
    var j=await r.json();
    document.getElementById('output').textContent=j.text;
  }catch(e){}
}
setInterval(refreshFeed,2000);
refreshFeed();
</script></body></html>"""


@app.get("/")
def index():
    return HTML


if __name__ == "__main__":
    print("灵元子记忆服务启动：http://127.0.0.1:9000  d=%d %d层 seq=%d" % (D, N_LAYERS, SEQ), flush=True)
    app.run(host="127.0.0.1", port=9000, debug=False)
