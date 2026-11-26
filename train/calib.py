# -*- coding: utf-8 -*-
"""冲突阈值标定：用已知样本量出"重复 / 同话题新内容"的分数分布

label:
  dup      = 同一事实换个说法（应该拦下来）
  update   = 同一对象的新值（应该拦下来，提示覆盖）
  related  = 同话题但不同信息（应该放行，最多提示）
  distinct = 无关（应该放行且不提示）
"""
import torch
from sentence_transformers import SentenceTransformer

BGE = "BAAI/bge-small-zh-v1.5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

A = "灵铭记忆的编码器是 BAAI/bge-small-zh-v1.5，输出 512 维；坐标桶 C=2048（cMult=4），稀疏 k=16，center=false，相关性阈值 0.45。"
B = "坐标桶数 C=2048 时 k=8 太稀疏会漏召回，实测拐点在 k=16。"
C = "记忆库的 contents 存的是纯语义向量 h；写入向量 w 只用于外积写 g，不存进库。"
D = "灵铭从 bge-small-en 换成 bge-small-zh 的原因是英文模型在中文上排序错误。"
E = "灵铭插件目录是 C:\\Users\\yangtianpeng\\Documents\\ChatGPT\\lingming-memory，服务端在 train\\embed_server.py。"
F = "DSH 的会话历史由内核按会话日志自行重放，插件无法删改。"
G = "用户要求结论必须有实测依据，不许夸大或凭印象下结论。"
H = "灵铭是外部记忆系统（DSH 插件），不是语言模型，contents 是数据库表。"

PAIRS = [
    # --- 同一事实换说法：应拦 ---
    ("灵铭的编码器用的是 bge-small-zh-v1.5，512 维向量。", A, "dup"),
    ("配置是 C=2048 个桶、k=16、阈值 0.45、center 关闭。", A, "dup"),
    ("写记忆时数据库里存的是 h，w 不落库。", C, "dup"),
    ("用 w 当数据库内容会让效果变差，所以只存 h。", C, "dup"),
    ("英文模型编码中文排序不准，所以换了中文模型。", D, "dup"),
    ("历史是 DSH 自己按会话日志重放的，插件改不了。", F, "dup"),
    ("结论要有实测依据，不能夸大。", G, "dup"),
    # --- 同一对象的新值：应拦并提示覆盖 ---
    ("灵铭的 k 现在改成 32 了。", A, "update"),
    ("灵铭的坐标桶倍数改成 8，C=4096。", A, "update"),
    ("相关性阈值调整为 0.6。", A, "update"),
    # --- 同话题不同信息：应放行 ---
    ("灵铭的空桶比例过高时说明 W_coord 需要重新生成。", A, "related"),
    ("bge 模型在 GPU 上单条编码约 7ms，CPU 约 13ms。", D, "related"),
    ("Embed server 默认监听 127.0.0.1:9000，端口可以在插件配置里改。", E, "related"),
    ("DSH 的 compaction 摘要单条约 7600 字符。", F, "related"),
    ("用户偏好中文回答，不喜欢过度使用英文术语。", G, "related"),
    ("灵元子是 GPT 式语言模型，记忆在内部快权重 g 里。", H, "related"),
    # --- 无关：应放行且不提示 ---
    ("今天的服务器负载比较高。", A, "distinct"),
    ("我想买个新的机械键盘。", G, "distinct"),
    ("明天上午十点开会。", F, "distinct"),
    ("Python 的 GIL 限制了多线程并行。", C, "distinct"),
]

m = SentenceTransformer(BGE, device=DEV)


def cos(x, y):
    e = m.encode([x, y], normalize_embeddings=True, show_progress_bar=False)
    return float(torch.tensor(e[0]) @ torch.tensor(e[1]))


rows = [(label, cos(new, old), new, old) for new, old, label in PAIRS]
rows.sort(key=lambda r: -r[1])

print(f"{'label':9s} {'score':>7s}  新内容")
print("-" * 78)
for label, s, new, _ in rows:
    print(f"{label:9s} {s:7.3f}  {new[:46]}")

print("\n--- 按类别统计 ---")
for lab in ("dup", "update", "related", "distinct"):
    vals = [r[1] for r in rows if r[0] == lab]
    if vals:
        print(f"{lab:9s} n={len(vals):2d}  min={min(vals):.3f}  max={max(vals):.3f}  均值={sum(vals)/len(vals):.3f}")
