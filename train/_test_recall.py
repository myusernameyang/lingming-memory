# -*- coding: utf-8 -*-
"""灵铭插件记忆召回诊断：候选覆盖 + L/k 扫描

关键指标 candidate-recall：正确答案是否进入了粗排候选集。
如果没进，细排再强也救不回来 —— 那就是坐标/截断的问题。
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DIR = r"C:\Users\yangtianpeng\Documents\ChatGPT\lingming-memory\train"
sys.path.insert(0, TRAIN_DIR)
from memory_core import CNNEncoder, LingmingMemory, sparse_topk  # noqa: E402

CKPT = os.path.join(TRAIN_DIR, "coord.pt")
D, CMULT, CAP = 128, 4, 10000

PAIRS = [
    ("我更喜欢用 Python 而不是 JavaScript 写后端", "服务端语言选哪个好"),
    ("我的项目部署在阿里云上海区", "服务器放在哪里"),
    ("我们约定接口返回字段统一用 snake_case", "字段命名风格怎么定"),
    ("上周讨论过把缓存层换成 Redis", "缓存组件怎么选"),
    ("团队规定所有提交必须走 code review", "代码提交要走什么流程"),
    ("测试环境数据库用的是 PostgreSQL 15", "测试库用什么版本的数据库"),
    ("前端框架定了 Next.js 做 SSR", "前端用什么框架做服务端渲染"),
    ("日志统一打到一个 ELK 集群", "日志收集方案是什么"),
    ("CI 用 GitHub Actions 跑单测和构建", "持续集成用什么"),
    ("上线时间定在下周四晚上八点", "什么时候发版"),
    ("移动端只做 iOS，暂不做安卓", "客户端覆盖哪些平台"),
    ("鉴权用 JWT，有效期两小时", "用户认证方案"),
    ("图片走 CDN，源站是 OSS", "静态资源怎么分发"),
    ("监控用 Prometheus 加 Grafana", "系统监控用什么"),
    ("数据库主从延迟要求低于 100ms", "主从同步的延迟指标"),
    ("接口限流按 IP 每分钟 600 次", "限流策略怎么配"),
    ("灰度发布先放 5% 流量", "灰度量级多少"),
    ("密钥统一放 Vault 管理", "敏感配置存哪里"),
    ("错误码规范用 5 位数字", "错误码怎么编号"),
    ("压测目标是单机 3000 QPS", "性能压测指标"),
]


def build(L, k):
    encoder = CNNEncoder(D)
    W_coord = nn.Linear(D, CMULT * D, bias=False)
    state = torch.load(CKPT, map_location="cpu")
    w = state.get("W_coord", state.get("W", None))
    if isinstance(w, dict):
        encoder.load_state_dict(state.get("encoder", {}), strict=False)
        W_coord.load_state_dict(w, strict=False)
    encoder.eval()
    W_coord.eval()
    return LingmingMemory(encoder, W_coord, k=k, L=L, capacity=CAP)


def run(L, k):
    mem = build(L, k)
    for mem_text, _ in PAIRS:
        mem.add(mem_text, "user")
    text2mid = {v["text"]: mid for mid, v in mem.store_.items()}

    cand_hit = r1 = r3 = 0
    for mem_text, query in PAIRS:
        target = text2mid[mem_text]
        # 粗排候选
        votes = {}
        for bar in mem.coord(query):
            for mid in mem.index.get(bar, []):
                votes[mid] = votes.get(mid, 0) + 1
        if target in votes:
            cand_hit += 1
        got = [text2mid[it["text"]] for it in mem.retrieve(query, 3)]
        if got and got[0] == target:
            r1 += 1
        if target in got:
            r3 += 1

    n = len(PAIRS)
    print(f"  L={L:3d} k={k:2d}   cand-recall {cand_hit/n*100:5.1f}%   "
          f"recall@1 {r1/n*100:5.1f}%   recall@3 {r3/n*100:5.1f}%")
    return cand_hit / n, r1 / n, r3 / n


if __name__ == "__main__":
    print(f"记忆 {len(PAIRS)} 条 / 查询 {len(PAIRS)} 条（同义改写，不共享原句）\n")
    print("扫描 L（输入字节上限）与 k（点亮 bar 数）：")
    for L in (64, 128, 192, 256):
        for k in (8, 16):
            run(L, k)
