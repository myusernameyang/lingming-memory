# -*- coding: utf-8 -*-
"""重建灵铭记忆库：清空后只写入"重要且需要长期记住"的内容。

用 /write 逐条写入，再 /save 落盘。
"""
import json
import urllib.request

BASE = "http://127.0.0.1:9000"

FACTS = [
    # ---- 项目状态 ----
    "灵铭记忆的编码器是 BAAI/bge-small-zh-v1.5，输出 512 维；坐标桶 C=2048（cMult=4），稀疏 k=16，center=false，相关性阈值 0.45。",
    "灵铭从 bge-small-en 换成 bge-small-zh 的原因是：英文模型在中文上排序错误，无关句 0.8199 高于正确答案 0.7749；换 zh 后正确项 0.6596 排第一，且区分区间从 0.72~0.82 拓宽到 0.34~0.66。",
    "坐标桶数 C=2048 时 k=8 太稀疏会漏召回（查询可能返回空），实测拐点在 k=16。",
    "记忆库的 contents 存的是纯语义向量 h；写入向量 w（输入+召回）只用于外积写 g，不存进库，实测存 w 会让端到端从 95.5% 掉到 57.5%。",
    "坐标投影 W_coord 用固定随机投影即可，训练过的投影更差（候选召回 99.5% vs 88.5%）。",
    "记忆写入前按句切分成多段，各自独立入库；纯 URL、markdown 标题、代码围栏、表格线、框架包装文本会被过滤掉。",
    # ---- DSH 机制约束 ----
    "DSH 的 agent/pre-step 事件里，payload.messages 只是进入这一步的新消息，不是完整历史；返回的 messages 会替换掉它。",
    "DSH 的会话历史由内核按会话日志自行重放，插件无法删改；agent/request 事件明确写着不能修改 messages。",
    "DSH 的压缩摘要走独立通道（单条约 7600 字符，实测24条），pre-step 碰不到；要完全没有摘要只能开新会话并把 compaction 的 auto 设为 false。",
    "因此灵铭无法替代 DSH 上下文，正确用法是让智能体按需调用工具检索，而不是每轮注入。",
    # ---- 踩过的坑 ----
    "不能用 role:'user' 把记忆注入对话：它会显示成用户自己说的话，还会被当成用户记忆再存一遍。",
    "JS 里用 let base 做局部变量会遮蔽外层服务地址，导致 fetch 请求地址变成 [object Object] 而静默失败。",
    "JS 在 if 块内声明 const、却在块外使用会抛 ReferenceError，若被外层 catch 吞掉，功能会完全失效且没有任何报错。",
    "memory.js 只挑 text 和 role 返回，会丢掉 score 和 mid，导致相关性阈值失效、反馈学习拿不到 id。",
    "召回条数 topK 太小的话，正确答案会被近似的旧条目挤掉；灵铭默认用 5。",
    "配置页曾把字符串参数渲染成 number 输入框并做 Number() 转换，导致 contextMode 变成 NaN 无法保存。",
    # ---- 用户要求 ----
    "用户要求结论必须有实测依据，不许夸大或凭印象下结论；不确定时先查证再回答。",
    "用户最反感的是同一类问题要重复说好几遍；遇到类似场景应先检索记忆，再动手。",
    # ---- 关键路径 ----
    "灵铭插件目录是 C:\\Users\\yangtianpeng\\Documents\\ChatGPT\\lingming-memory，服务端在 train\\embed_server.py，记忆库在 train\\memory_db.pt，配置在 train\\config.json。",
    "桌面 profile 补丁在 C:\\Users\\yangtianpeng\\.dsh\\profiles\\desktop\\cordis.patch.yml，compaction 的 auto:false 就写在这里。",
]


def post(path, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


post("/reset", {})
n = 0
for f in FACTS:
    try:
        post("/write", {"text": f, "role": "user"})
        n += 1
    except Exception as e:
        print("写入失败:", f[:30], e)
print("已写入", n, "条")
print(post("/save", {}))
st = json.loads(urllib.request.urlopen(BASE + "/status", timeout=30).read().decode("utf-8"))
print("库内", st["n_mem"], "条 |", st["encoder"], "| C =", st["C"], "| k =", st["k"])
