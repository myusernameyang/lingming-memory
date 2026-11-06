# -*- coding: utf-8 -*-
"""灵铭记忆服务：坐标寻址多级向量库
接口：
  POST /write    {"text":"...","role":"user"}   存记忆 → 坐标 top-k 分桶
  POST /retrieve {"query":"...","topK":3}       查 → 坐标桶交集 → 命中最相关
  POST /save                                     保存坐标权重（coord.pt）
  GET  /status                                   状态（记忆数 / 桶数）
  GET  /config                                   可视化配置页（HTML）
  GET  /api/config                               配置 JSON
  POST /api/config                               保存配置 + 重建（实时生效）
"""
import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, Response
from memory_core import CNNEncoder, LingmingMemory

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
CKPT = os.path.join(BASE, "coord.pt")
L = 64

DEFAULT_CONFIG = {
    "d": 128,
    "cMult": 4,
    "k": 8,
    "capacity": 10000,
    "topK": 3,
}

# 字段定义（HTML 页和校验共用）
FIELDS = [
    {"key": "topK", "label": "召回条数", "hint": "每轮召回的历史记忆条数", "type": "number", "min": 1, "max": 20},
    {"key": "d", "label": "维度 d", "hint": "模型维度（改了要重训坐标）", "type": "number", "min": 32},
    {"key": "cMult", "label": "横向倍数", "hint": "坐标 bar 数 C = 横向倍数 × d", "type": "number", "min": 1},
    {"key": "k", "label": "稀疏 top-k", "hint": "每条记忆点亮几个坐标 bar", "type": "number", "min": 1},
    {"key": "capacity", "label": "向量库容量", "hint": "最多存多少条记忆（FIFO 淘汰）", "type": "number", "min": 1},
]


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


CONFIG = load_config()


def build_memory():
    """按当前配置重建 encoder + W_coord + 记忆库"""
    d = int(CONFIG["d"])
    c = max(1, int(CONFIG["cMult"])) * d
    k = max(1, int(CONFIG["k"]))
    cap = max(1, int(CONFIG["capacity"]))
    encoder = CNNEncoder(d)
    W_coord = nn.Linear(d, c, bias=False)
    if os.path.exists(CKPT):
        state = torch.load(CKPT, map_location="cpu")
        enc = state.get("encoder", {})
        w = state.get("W_coord", state.get("W", None))
        if isinstance(w, dict):
            encoder.load_state_dict(enc, strict=False)
            W_coord.load_state_dict(w, strict=False)
        else:
            W_coord.load_state_dict({"weight": state}, strict=False)
        print("已加载坐标权重 coord.pt", flush=True)
    else:
        print("未找到 coord.pt，用随机坐标（先用 train_coord.py 训练）", flush=True)
    return LingmingMemory(encoder, W_coord, k=k, L=L, capacity=cap)


app = Flask(__name__)
mem = build_memory()


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
    topK = int(d.get("topK", CONFIG.get("topK", 3)))
    if not query:
        return jsonify({"items": []})
    items = mem.retrieve(query, topK)
    return jsonify({"items": items})


@app.post("/save")
def save():
    torch.save({"encoder": mem.encoder.state_dict(), "W_coord": mem.W_coord.state_dict()}, CKPT)
    return jsonify({"ok": True, "path": CKPT})


@app.get("/status")
def status():
    st = mem.status()
    st["config"] = CONFIG
    return jsonify(st)


@app.get("/api/config")
def api_get_config():
    return jsonify({"ok": True, "config": CONFIG, "fields": FIELDS})


@app.post("/api/config")
def api_set_config():
    global CONFIG, mem
    body = request.get_json(force=True, silent=True) or {}
    patch = body.get("config") or body
    try:
        for f in FIELDS:
            if f["key"] in patch:
                v = int(patch[f["key"]])
                if "min" in f and v < f["min"]:
                    v = f["min"]
                if "max" in f and v > f["max"]:
                    v = f["max"]
                CONFIG[f["key"]] = v
        save_config(CONFIG)
        mem = build_memory()
        return jsonify({"ok": True, "config": CONFIG})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.get("/config")
def config_page():
    fields_json = json.dumps(FIELDS, ensure_ascii=False)
    cfg_json = json.dumps(CONFIG, ensure_ascii=False)
    html = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>灵铭记忆 · 配置</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:640px;margin:40px auto;padding:0 20px;color:#222;background:#fff}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:#888;font-size:13px;margin:0 0 24px}
  .row{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid #eee}
  .lab{display:flex;flex-direction:column;gap:2px}
  .lab b{font-size:14px;font-weight:500}
  .lab span{font-size:12px;color:#999}
  input{width:180px;padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:14px;text-align:right}
  .bar{margin-top:24px;display:flex;gap:12px;align-items:center}
  button{padding:8px 20px;border:0;border-radius:6px;background:#247bbf;color:#fff;font-size:14px;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  #msg{font-size:13px;color:#2a8;display:none}
  #err{font-size:13px;color:#c33;display:none}
</style></head><body>
<h1>灵铭记忆 · 配置</h1>
<p class="sub">改完点保存，立即生效（写 config.json 并重建记忆库）</p>
<div id="form"></div>
<div class="bar">
  <button id="btn" onclick="save()">保存配置</button>
  <span id="msg">已保存 ✅</span>
  <span id="err"></span>
</div>
<script>
const FIELDS = __FIELDS__;
let CFG = __CFG__;
function render(){
  const box = document.getElementById('form');
  box.innerHTML = '';
  for (const f of FIELDS) {
    const row = document.createElement('div'); row.className='row';
    const lab = document.createElement('div'); lab.className='lab';
    lab.innerHTML = '<b>'+f.label+'</b><span>'+f.hint+'</span>';
    const inp = document.createElement('input');
    inp.type = f.type; inp.id = 'f_'+f.key; inp.value = CFG[f.key];
    if (f.min !== undefined) inp.min = f.min;
    if (f.max !== undefined) inp.max = f.max;
    row.appendChild(lab); row.appendChild(inp);
    box.appendChild(row);
  }
}
function save(){
  const btn = document.getElementById('btn');
  btn.disabled = true;
  const patch = {};
  for (const f of FIELDS) patch[f.key] = Number(document.getElementById('f_'+f.key).value);
  fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({config: patch})})
    .then(r => r.json())
    .then(j => {
      btn.disabled = false;
      if (j.ok) { CFG = j.config; document.getElementById('msg').style.display='inline'; document.getElementById('err').style.display='none'; }
      else { document.getElementById('err').textContent = '保存失败: ' + j.error; document.getElementById('err').style.display='inline'; }
    })
    .catch(e => { btn.disabled = false; document.getElementById('err').textContent = '保存失败: ' + e; document.getElementById('err').style.display='inline'; });
}
render();
</script>
</body></html>"""
    html = html.replace("__FIELDS__", fields_json).replace("__CFG__", cfg_json)
    return Response(html, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000)
