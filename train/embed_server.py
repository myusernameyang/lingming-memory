# -*- coding: utf-8 -*-
"""灵铭记忆服务：外积 g 写入 + 稀疏插值坐标 + 向量库

架构（照灵元子 FastWeight 写法）：
    coord   = W_coord @ h                     # 固定随机投影
    q       = sparse_topk(coord, k)           # 插值坐标（保权重），下标 = 数据库键
    content = normalize(contents[cand] @ h)   # 数据库召回
    w       = normalize(h + content)          # 输入 + 召回
    g       = tanh(lam*g + writeScale*gate*(w ⊗ q))
    contents[row] = h                         # 库里存纯语义向量

接口：
  POST /write    {"text","role","scope"}      存记忆（scope 隔离聊天室）
  POST /retrieve {"query","topK","scope"}     检索（只返回同 scope / 共享条目）
  GET  /scope                                 各 scope 条数
  POST /save                          持久化记忆库
  POST /reset                         清空记忆
  GET  /status                        状态
  GET  /config                        可视化配置页
  GET  /api/config                    配置 JSON
  POST /api/config                    保存配置（改结构参数会重建并清空记忆）
"""
import os
import json
import shutil
import time
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from flask import Flask, request, jsonify, Response

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
DB_PATH = os.path.join(BASE, "memory_db.pt")

D_DEFAULT = 384          # bge-small-en-v1.5 的维度（不再可配，编码器决定）
BGE_MODEL = os.environ.get("LINGMING_BGE", "BAAI/bge-small-zh-v1.5")

DEFAULT_CONFIG = {
    "topK": 3,
    "cMult": 4,
    "k": 8,
    "capacity": 10000,
    "writeScale": 0.1,
    "gateBias": 0.2,
    "gateTemp": 0.05,
    "readTop": 3,
    "perBar": 3,
    "center": False,
    "contextMode": "augment",
    "keepMessages": 6,
}

FIELDS = [
    {"key": "topK", "label": "召回条数", "hint": "每轮拼进上下文的历史记忆条数", "type": "number", "min": 1, "max": 20},
    {"key": "cMult", "label": "坐标桶倍数", "hint": "桶数 C = 倍数 × 384（越大分桶越细）", "type": "number", "min": 1, "max": 16},
    {"key": "k", "label": "稀疏 top-k", "hint": "每条记忆点亮几个桶（数据库键数）", "type": "number", "min": 1, "max": 64},
    {"key": "capacity", "label": "记忆容量", "hint": "最多存多少条（超过自动扩容）", "type": "number", "min": 100, "max": 2000000},
    {"key": "writeScale", "label": "写入强度", "hint": "外积写 g 的步长（越大记忆越快饱和）", "type": "number", "min": 0.01, "max": 2, "step": 0.01},
    {"key": "gateBias", "label": "写门控偏置", "hint": "越小越倾向写入", "type": "number", "min": -1, "max": 1, "step": 0.05},
    {"key": "gateTemp", "label": "写门控温度", "hint": "越小门控越硬", "type": "number", "min": 0.01, "max": 1, "step": 0.01},
    {"key": "readTop", "label": "读取聚合数", "hint": "召回时取最相关几条求平均作为 content", "type": "number", "min": 1, "max": 10},
    {"key": "perBar", "label": "桶内限流 (perBar)", "hint": "每个坐标桶只留余弦前 N 条进候选；越大越稳越慢，0=不限流（候选随库大小涨）", "type": "number", "min": 0, "max": 32},
    {"key": "center", "label": "减共模", "hint": "1=写入前减运行均值，0=关闭", "type": "number", "min": 0, "max": 1},
    {"key": "contextMode", "label": "上下文模式", "hint": "augment=历史+记忆 / window=最近N条+记忆 / replace=仅当前+记忆", "type": "text"},
    {"key": "keepMessages", "label": "保留原始条数", "hint": "window 模式下保留最近几条原始消息", "type": "number", "min": 1, "max": 40},
]

REBUILD_KEYS = {"cMult", "k", "capacity", "writeScale", "gateBias", "gateTemp", "readTop", "perBar", "center"}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8-sig") as f:
                raw = json.load(f)
            # 只接受已知键（清掉历史遗留的 d 等）
            cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG})
        except Exception:
            pass
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


CONFIG = load_config()

# 插件设置面板通过环境变量下发参数；收到就落盘，保证面板与 /config 网页一致
ENV_MAP = {
    "LINGMING_TOPK": "topK",
    "LINGMING_C_MULT": "cMult",
    "LINGMING_K": "k",
    "LINGMING_CAPACITY": "capacity",
    "LINGMING_WRITE_SCALE": "writeScale",
    "LINGMING_GATE_BIAS": "gateBias",
    "LINGMING_GATE_TEMP": "gateTemp",
    "LINGMING_READ_TOP": "readTop",
    # center 不从这里下发：它由 /config 网页独占，避免插件面板与网页两处打架
}


def _coerce(key, raw):
    if key == "center":
        return bool(int(float(raw)))
    return float(raw) if "." in raw else int(float(raw))


def apply_env(cfg):
    changed = False
    for env, key in ENV_MAP.items():
        if env in os.environ:
            try:
                cfg[key] = _coerce(key, os.environ[env])
                changed = True
            except Exception:
                pass
    return changed


if apply_env(CONFIG):
    save_config(CONFIG)
    print("已从插件设置同步参数到 config.json", flush=True)

# ---------------- 编码器 ----------------
_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = SentenceTransformer(BGE_MODEL, device=dev)
        print(f"bge 编码器已加载：{BGE_MODEL} ({dev})", flush=True)
    return _encoder


def encode(text):
    e = get_encoder().encode([text], normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(e[0], dtype=torch.float32, device=DEVICE)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 记忆库 ----------------
def get_dim():
    """编码维度从模型实际值取（bge-small-en 是 384，bge-small-zh 是 512）"""
    try:
        return int(get_encoder().get_sentence_embedding_dimension())
    except Exception:
        return D_DEFAULT


def build_memory():
    from memory_core import LingmingMemory
    cfg = CONFIG
    mem = LingmingMemory(
        encode,
        d=get_dim(),
        c_mult=int(cfg["cMult"]),
        k=int(cfg["k"]),
        capacity=int(cfg["capacity"]),
        device=DEVICE,
        center=bool(cfg["center"]),
        write_scale=float(cfg["writeScale"]),
        gate_bias=float(cfg["gateBias"]),
        gate_temp=float(cfg["gateTemp"]),
        read_top=int(cfg["readTop"]),
        per_bar=int(cfg.get("perBar", 3)),
    )
    return mem


app = Flask(__name__)
app.json.ensure_ascii = False          # 接口直接返回中文，而不是 \uXXXX 转义
app.json.sort_keys = False
mem = build_memory()


def _load_db(m):
    if not os.path.exists(DB_PATH):
        return m
    try:
        st = torch.load(DB_PATH, map_location=DEVICE, weights_only=False)
        if st.get("d") != m.d or st.get("C") != m.C:
            print("持久化记忆库维度不匹配，跳过加载", flush=True)
            return m
        saved_model = st.get("encoder_name")
        if saved_model and saved_model != BGE_MODEL:
            print(f"编码器不一致（库来自 {saved_model}，当前 {BGE_MODEL}），跳过加载以免向量空间混用", flush=True)
            return m
        n = int(st["n"])
        if n > m.contents.shape[0]:
            m.contents = torch.zeros(n, m.d, device=m.dev)
        m.contents[:n] = st["contents"][:n]
        m.n = n
        m.texts = list(st["texts"])
        m.roles = list(st["roles"])
        m.slot_of = dict(st["slot_of"])
        m.bars = dict(st["bars"])
        m.index = {k: list(v) for k, v in st["index"].items()}
        m.next_id = int(st["next_id"])
        m.total_writes = int(st["total_writes"])
        m.msg_of = dict(st.get("msg_of", {}))
        m.msg_text = dict(st.get("msg_text", {}))
        m.boost = dict(st.get("boost", {}))
        m.deleted = set(st.get("deleted", []))
        if hasattr(m, "_load_scopes"):
            m._load_scopes(st.get("scopes", {}))
        m.g = st["g"].to(m.dev)
        m.t_g = st["t_g"].to(m.dev)
        print(f"已恢复记忆库 {m.n} 条", flush=True)
    except Exception as e:
        print(f"记忆库加载失败：{e}", flush=True)
    return m


mem = _load_db(mem)
_DB_LOADED_N = mem.n
print(f"[persist] loaded memories n={mem.n}", flush=True)


# ---------------- 接口 ----------------
_AUTOSAVE_EVERY = 20
_writes_since_save = 0


def _persist(force=False):
    """原子写盘 + 空库保护。

    2026-11-15 事故：torch>=2.6 的 weights_only 默认 True 让旧库加载失败，
    服务以空库启动后把空库保存回去，覆盖了 555 条记忆。
    现在：1) 加载用 weights_only=False；2) 空库默认拒绝覆盖非空库；3) tmp+replace 原子写。
    """
    global _DB_LOADED_N
    if not force and mem.n == 0 and _DB_LOADED_N > 0:
        print(f"[persist] refuse: store empty but loaded {_DB_LOADED_N} memories before", flush=True)
        return False
    payload = {
        "encoder_name": BGE_MODEL,
        "d": mem.d, "C": mem.C, "n": mem.n,
        "contents": mem.contents[:mem.n].cpu(),
        "texts": mem.texts, "roles": mem.roles,
        "slot_of": mem.slot_of, "bars": mem.bars, "index": mem.index,
        "next_id": mem.next_id, "total_writes": mem.total_writes,
        "msg_of": mem.msg_of, "msg_text": mem.msg_text,
        "boost": mem.boost,
        "scopes": getattr(mem, "scopes", {}),
        "deleted": sorted(mem.deleted),
        "g": mem.g.cpu(), "t_g": mem.t_g.cpu(),
    }
    tmp = DB_PATH + ".tmp"
    if os.path.exists(DB_PATH):
        try:
            shutil.copyfile(DB_PATH, DB_PATH + ".bak1")
        except Exception as e:
            print(f"[persist] backup failed: {e}", flush=True)
    torch.save(payload, tmp)
    os.replace(tmp, DB_PATH)
    if mem.n > _DB_LOADED_N:
        _DB_LOADED_N = mem.n
    return True


@app.post("/write")
def write():
    global _writes_since_save
    d = request.get_json(force=True, silent=True) or {}
    text = str(d.get("text", "")).strip()
    role = str(d.get("role", "user"))
    if not text:
        return jsonify({"ok": False, "error": "empty text"})
    scope = str(d.get("scope", "") or "").strip()
    mid = mem.add(text, role, scope=scope)
    # 自动落盘：插件重启会重建服务端，不落盘就丢记忆
    _writes_since_save += 1
    if _writes_since_save >= _AUTOSAVE_EVERY:
        _writes_since_save = 0
        try:
            _persist()
        except Exception as e:
            print(f"自动保存失败：{e}", flush=True)
    return jsonify({"ok": True, "id": mid, "n_mem": mem.n})


@app.post("/retrieve")
def retrieve():
    d = request.get_json(force=True, silent=True) or {}
    query = str(d.get("query", "")).strip()
    topK = int(d.get("topK", CONFIG.get("topK", 3)))
    scope = str(d.get("scope", "") or "").strip()
    if not query:
        return jsonify({"items": []})
    return jsonify({"items": mem.retrieve(query, topK, scope=scope)})


_INJECTIONS = deque(maxlen=60)     # 最近注入记录（插件每轮上报）


@app.post("/log_injection")
def log_injection():
    """插件上报本轮到底往上下文里注入了什么（用于可视化观察效果）"""
    d = request.get_json(force=True, silent=True) or {}
    items = d.get("items") or []
    _INJECTIONS.append({
        "ts": time.strftime("%H:%M:%S"),
        "agent": str(d.get("agent", ""))[:8],
        "mode": d.get("mode", ""),
        "msgsIn": d.get("msgsIn"),
        "msgsKept": d.get("msgsKept"),
        "query": str(d.get("query", ""))[:160],
        "items": [
            {
                "score": round(float(x.get("score", 0)), 4),
                "role": x.get("role"),
                "text": str(x.get("text", ""))[:120],
            }
            for x in items
        ],
    })
    return jsonify({"ok": True})


@app.get("/injections")
def injections():
    """看最近的注入记录：?format=text 直接人读"""
    if request.args.get("format") == "text":
        if not _INJECTIONS:
            return Response("暂无注入记录", mimetype="text/plain; charset=utf-8")
        lines = []
        for e in reversed(_INJECTIONS):
            lines.append(f"[{e['ts']}] agent={e['agent']}  mode={e.get('mode')}  "
                         f"上下文 {e.get('msgsIn')}→{e.get('msgsKept')} 条  注入 {len(e['items'])} 条  查询: {e['query']}")
            for it in e["items"]:
                lines.append(f"    {it['score']:.3f} [{it['role']}] {it['text']}")
            lines.append("")
        return Response("\n".join(lines), mimetype="text/plain; charset=utf-8")
    return jsonify({"n": len(_INJECTIONS), "items": list(reversed(_INJECTIONS))})


@app.post("/repair")
def repair():
    """按当前过滤规则清洗已有记忆：重新切分 + 丢弃低价值片段（内容不丢，垃圾被清掉）"""
    global mem
    saved = [(mem.texts[r], mem.roles[r])
             for mid, r in sorted(mem.slot_of.items(), key=lambda kv: kv[1])]
    before = len(saved)
    mem = build_memory()
    for text, role in saved:
        try:
            mem.add(text, role)
        except Exception:
            continue
    _persist()
    return jsonify({"ok": True, "before": before, "after": mem.n})


@app.post("/score")
def score():
    """两条文本的余弦相似度（冲突判定用完整原文时，需要重新算分）"""
    d = request.get_json(force=True, silent=True) or {}
    a = str(d.get("a", "")).strip()
    b = str(d.get("b", "")).strip()
    if not a or not b:
        return jsonify({"ok": False, "score": 0.0})
    ha, hb = encode(a), encode(b)
    return jsonify({"ok": True, "score": float((ha @ hb).item())})


@app.get("/msg")
def msg_text():
    """取某条记忆所属的完整原始消息（冲突判定需要完整语义，不能只看分段）"""
    try:
        mid = int(request.args.get("id", "-1"))
    except Exception:
        return jsonify({"ok": False})
    return jsonify({"ok": True, "id": mid, "text": mem.msg_text.get(mid, "")})


@app.post("/delete")
def delete_memory():
    """删除一条记忆（被删除的条目不参与召回）"""
    d = request.get_json(force=True, silent=True) or {}
    mid = d.get("mid")
    ok = mem.delete(mid) if mid is not None else False
    if ok:
        try:
            _persist()
        except Exception:
            pass
    return jsonify({"ok": ok, "mid": mid, "n_mem": mem.n - len(mem.deleted)})


@app.get("/dump")
def dump():
    """调试用：看最近存进库里的记忆（含角色 / 学习权重 / 所属消息）"""
    n = int(request.args.get("n", 20))
    live = [(mid, row) for mid, row in mem.slot_of.items() if mid not in mem.deleted]
    pairs = sorted(live, key=lambda kv: kv[1])[-n:]
    items = []
    for mid, row in pairs:
        items.append({
            "mid": mid,
            "role": mem.roles[row],
            "text": mem.texts[row],
            "boost": round(float(mem.boost.get(mid, 0.0)), 4),
            "msg": mem.msg_of.get(mid, -1),
        })
    # ?format=text 直接输出人可读列表
    if request.args.get("format") == "text":
        body = [f"库内 {len(live)} 条（显示最近 {len(items)} 条）"]
        for it in items:
            body.append(f"[{it['role']}]{'+' + str(it['boost']) if it['boost'] else ''} {it['text']}")
        return Response("\n".join(body), mimetype="text/plain; charset=utf-8")
    return jsonify({"n_mem": len(live), "items": items})


@app.post("/feedback")
def feedback():
    """用后反馈：告诉记忆库哪些召回真的被用上了（在线学习权重）"""
    d = request.get_json(force=True, silent=True) or {}
    pairs = []
    for x in (d.get("items") or []):
        try:
            pairs.append((int(x.get("mid")), bool(x.get("used"))))
        except Exception:
            continue
    if pairs:
        mem.feedback(pairs)
    return jsonify({"ok": True, "n": len(pairs)})


@app.post("/save")
def save():
    _persist()
    return jsonify({"ok": True, "path": DB_PATH, "n_mem": mem.n})


@app.post("/reset")
def reset():
    global mem
    mem = build_memory()
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.get("/scope")
def scope_stats():
    """各 scope 的条目数：核对"每个聊天室各有多少条记忆"。"""
    out = getattr(mem, "scope_stats", None)
    return jsonify({"ok": True, "scopes": out() if callable(out) else {}, "n_mem": mem.n})


@app.get("/status")
def status():
    st = mem.status()
    st["config"] = CONFIG
    st["encoder"] = BGE_MODEL
    st["device"] = DEVICE
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
        need_rebuild = any(k in patch and patch[k] != CONFIG.get(k) for k in REBUILD_KEYS)
        for f in FIELDS:
            key = f["key"]
            if key in patch:
                v = patch[key]
                if f.get("type") == "text":
                    v = str(v)                      # 字符串类参数（如 contextMode）不转数字
                elif key == "center":
                    v = bool(int(float(v)))
                else:
                    v = float(v) if isinstance(v, float) else int(float(v))
                    if "min" in f and v < f["min"]:
                        v = f["min"]
                    if "max" in f and v > f["max"]:
                        v = f["max"]
                CONFIG[key] = v
        save_config(CONFIG)
        if need_rebuild:
            # 重建会换掉向量空间参数，但已有记忆不该丢：先快照文本，重建后重放
            saved = [(mem.texts[r], mem.roles[r])
                     for mid, r in sorted(mem.slot_of.items(), key=lambda kv: kv[1])]
            mem = build_memory()
            for text, role in saved:
                try:
                    mem.add(text, role)
                except Exception:
                    continue
            print(f"配置变更后已重放 {len(saved)} 条记忆", flush=True)
        return jsonify({"ok": True, "config": CONFIG, "rebuilt": need_rebuild})
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
  body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:680px;margin:40px auto;padding:0 20px;color:#e6e6e6;background:#0f1116}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:#8a8f98;font-size:13px;margin:0 0 24px}
  .row{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid #23262e}
  .lab{display:flex;flex-direction:column;gap:2px}
  .lab b{font-size:14px;font-weight:500}
  .lab span{font-size:12px;color:#8a8f98}
  input{width:180px;padding:6px 8px;border:1px solid #333a45;border-radius:6px;font-size:14px;text-align:right;background:#171a21;color:#e6e6e6}
  .bar{margin-top:24px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  button{padding:8px 20px;border:0;border-radius:6px;background:#247bbf;color:#fff;font-size:14px;cursor:pointer}
  button.ghost{background:#2a3140}
  button:disabled{opacity:.5;cursor:default}
  #msg{font-size:13px;color:#2a8;display:none}
  #err{font-size:13px;color:#c33;display:none}
  #info{margin-top:14px;font-size:12px;color:#8a8f98;white-space:pre-wrap}
</style></head><body>
<h1>灵铭记忆 · 配置</h1>
<p class="sub">bge 语义编码 + 稀疏坐标分桶 + 外积 g 写入。改完点保存立即生效。</p>
<div id="form"></div>
<div class="bar">
  <button id="btn" onclick="save()">保存配置</button>
  <button class="ghost" onclick="refresh()">刷新状态</button>
  <button class="ghost" onclick="resetDb()">清空记忆</button>
  <span id="msg">已保存 ✅</span>
  <span id="err"></span>
</div>
<div id="info"></div>
<script>
const FIELDS = __FIELDS__;
let CFG = __CFG__;
function render(){
  const box = document.getElementById('form'); box.innerHTML='';
  for (const f of FIELDS){
    const row=document.createElement('div'); row.className='row';
    const lab=document.createElement('div'); lab.className='lab';
    lab.innerHTML='<b>'+f.label+'</b><span>'+f.hint+'</span>';
    const inp=document.createElement('input');
    inp.type = (f.type==='text') ? 'text' : 'number';
    inp.id='f_'+f.key;
    inp.value=(typeof CFG[f.key]==='boolean') ? (CFG[f.key]?1:0) : CFG[f.key];
    if(f.min!==undefined) inp.min=f.min;
    if(f.max!==undefined) inp.max=f.max;
    if(f.step!==undefined) inp.step=f.step;
    row.appendChild(lab); row.appendChild(inp); box.appendChild(row);
  }
}
function refresh(){
  fetch('/status').then(r=>r.json()).then(j=>{
    document.getElementById('info').textContent =
      '编码器 '+j.encoder+' ('+j.device+')   已存 '+j.n_mem+' 条   桶 '+j.n_bars+
      '   写入次数 '+j.total_writes;
  }).catch(()=>{ document.getElementById('info').textContent='服务未响应'; });
}
function save(){
  const btn=document.getElementById('btn'); btn.disabled=true;
  const patch={};
  for(const f of FIELDS){
    const el=document.getElementById('f_'+f.key);
    patch[f.key] = (f.type==='text') ? el.value : Number(el.value);
  }
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:patch})})
    .then(r=>r.json()).then(j=>{
      btn.disabled=false;
      if(j.ok){ CFG=j.config;
        document.getElementById('msg').style.display='inline';
        document.getElementById('err').style.display='none';
        if(j.rebuilt) setTimeout(refresh,300);
      } else {
        document.getElementById('err').textContent='保存失败: '+j.error;
        document.getElementById('err').style.display='inline';
      }
    }).catch(e=>{ btn.disabled=false;
      document.getElementById('err').textContent='保存失败: '+e;
      document.getElementById('err').style.display='inline'; });
}
function resetDb(){
  if(!confirm('确定清空全部记忆？此操作不可恢复')) return;
  fetch('/reset',{method:'POST'}).then(r=>r.json()).then(()=>{
    document.getElementById('msg').style.display='none';
    refresh();
  });
}
render(); refresh();
</script>
</body></html>"""
    html = html.replace("__FIELDS__", fields_json).replace("__CFG__", cfg_json)
    return Response(html, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("LINGMING_PORT", "9000"))
    app.run(host="127.0.0.1", port=port)
