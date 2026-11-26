# -*- coding: utf-8 -*-
"""冲突判定规则验证：结构化（参数键+数值）优先，余弦兜底"""
import re
import torch
from sentence_transformers import SentenceTransformer

BGE = "BAAI/bge-small-zh-v1.5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

NUM = re.compile(r"\d+(?:\.\d+)?")
# 参数键：ASCII 标识符或常见中文参数名
KEY = re.compile(
    r"(cMult|topK|readTop|capacity|writeScale|gateBias|gateTemp|center|"
    r"\bk\b|\bC\b|阈值|端口|容量|维度)\D{0,12}(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,}")


def numbers(t):
    return set(NUM.findall(t))


def params(t):
    out = {}
    for k, v in KEY.findall(t):
        out[k.lower()] = v
    return out


def decide(new, old, cos):
    """返回 (verdict, reason)；verdict ∈ duplicate/conflict/allow"""
    pn, po = params(new), params(old)
    n_new, n_old = numbers(new), numbers(old)

    # 1) 同参数、不同值 → 冲突（最该提醒的情况）
    for k, v in pn.items():
        if k in po and po[k] != v:
            return "conflict", f"参数 {k}：{po[k]} → {v}"

    # 2) 新内容里的数字多数已在旧条目里 → 重复
    if n_new and n_old:
        shared = n_new & n_old
        if len(shared) >= max(1, (len(n_new) + 1) // 2):
            return "duplicate", f"共有的数值 {sorted(shared)}"

    # 3) 无数字可比时：标识符重叠优先，余弦兜底
    if not n_new or not n_old:
        shared_ids = {x.lower() for x in ID.findall(new)} & {x.lower() for x in ID.findall(old)}
        if len(shared_ids) >= 2:
            return "duplicate", f"共有的标识符 {sorted(shared_ids)[:4]}"
        if cos >= 0.75:
            return "duplicate", f"文本高度相似 cos={cos:.3f}"
    return "allow", ""


A = "灵铭记忆的编码器是 BAAI/bge-small-zh-v1.5，输出 512 维；坐标桶 C=2048（cMult=4），稀疏 k=16，center=false，相关性阈值 0.45。"
B = "坐标桶数 C=2048 时 k=8 太稀疏会漏召回，实测拐点在 k=16。"
C = "记忆库的 contents 存的是纯语义向量 h；写入向量 w 只用于外积写 g，不存进库。"
D = "灵铭从 bge-small-en 换成 bge-small-zh 的原因是英文模型在中文上排序错误。"
E = "灵铭插件目录是 C:\\Users\\yangtianpeng\\Documents\\ChatGPT\\lingming-memory，服务端在 train\\embed_server.py。"
F = "DSH 的会话历史由内核按会话日志自行重放，插件无法删改。"
G = "用户要求结论必须有实测依据，不许夸大或凭印象下结论。"
H = "灵铭是外部记忆系统（DSH 插件），不是语言模型，contents 是数据库表。"

PAIRS = [
    ("灵铭的编码器用的是 bge-small-zh-v1.5，512 维向量。", A, "dup"),
    ("配置是 C=2048 个桶、k=16、阈值 0.45、center 关闭。", A, "dup"),
    ("写记忆时数据库里存的是 h，w 不落库。", C, "dup"),
    ("用 w 当数据库内容会让效果变差，所以只存 h。", C, "dup"),
    ("英文模型编码中文排序不准，所以换了中文模型。", D, "dup"),
    ("历史是 DSH 自己按会话日志重放的，插件改不了。", F, "dup"),
    ("结论要有实测依据，不能夸大。", G, "dup"),
    ("灵铭的 k 现在改成 32 了。", A, "update"),
    ("灵铭的坐标桶倍数改成 8，C=4096。", A, "update"),
    ("相关性阈值调整为 0.6。", A, "update"),
    ("灵铭的空桶比例过高时说明 W_coord 需要重新生成。", A, "related"),
    ("bge 模型在 GPU 上单条编码约 7ms，CPU 约 13ms。", D, "related"),
    ("Embed server 默认监听 127.0.0.1:9000，端口可以在插件配置里改。", E, "related"),
    ("DSH 的 compaction 摘要单条约 7600 字符。", F, "related"),
    ("用户偏好中文回答，不喜欢过度使用英文术语。", G, "related"),
    ("灵元子是 GPT 式语言模型，记忆在内部快权重 g 里。", H, "related"),
    ("今天的服务器负载比较高。", A, "distinct"),
    ("我想买个新的机械键盘。", G, "distinct"),
    ("明天上午十点开会。", F, "distinct"),
    ("Python 的 GIL 限制了多线程并行。", C, "distinct"),
]

m = SentenceTransformer(BGE, device=DEV)
enc = m.encode([p[0] for p in PAIRS] + [p[1] for p in PAIRS],
               normalize_embeddings=True, show_progress_bar=False)
t = torch.tensor(enc)

print(f"{'label':9s} {'verdict':10s} {'cos':>6s}  说明")
print("-" * 84)
stats = {}
for i, (new, old, label) in enumerate(PAIRS):
    cos = float(t[i] @ t[len(PAIRS) + i])
    verdict, why = decide(new, old, cos)
    print(f"{label:9s} {verdict:10s} {cos:6.3f}  {why[:40]}")
    stats.setdefault(label, []).append(verdict)

print("\n--- 期望 vs 实际 ---")
expect = {"dup": {"duplicate"}, "update": {"conflict"}, "related": {"allow"}, "distinct": {"allow"}}
ok = 0
for lab, verdicts in stats.items():
    hit = sum(1 for v in verdicts if v in expect[lab])
    ok += hit
    print(f"{lab:9s} {hit}/{len(verdicts)} 正确   {verdicts}")
print(f"\n总计 {ok}/{len(PAIRS)} 正确")
