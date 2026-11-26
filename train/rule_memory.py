# -*- coding: utf-8 -*-
"""规则记忆层（规律层）——灵铭记忆的「高效记忆」补充。

主人 2026-11-15 的原话：「记忆不单单是记住，怎么更高效的记住才是真的记住……
我很少去死记硬背，我记得都是规律」。

设计（与原文记忆分离、可独立评测）：
  · 一条规则 = {id, when, then, scope, confidence, hits, evidence[]}
  · 写：从对话/任务里提炼 when→then（可以由 agent 在复盘时调用 add），同 when 的规则做合并/升降置信度
  · 读：按「当前情境」匹配 when，返回 top-k 规则（可按 scope 过滤）
  · 反馈：规则被采纳 +，被纠正 -；低置信度规则自动衰减
  · 召回先用字符 bigram TF-IDF（零依赖，先跑通）；生产可换成坐标寻址 encode(when)

用法：
  python rule_memory.py add "when..." "then..." --scope experiment --conf 0.9
  python rule_memory.py probe "这次实验差了0.15是不是真的"
  python rule_memory.py list
"""
from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("rules.jsonl")


def _bigrams(text: str) -> Counter:
    t = re.sub(r"\s+", "", str(text))
    return Counter(t[i:i + 2] for i in range(max(len(t) - 1, 1)))


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-9)


class RuleMemory:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.rules: list[dict] = []
        self.load()

    # ---------- 持久化 ----------
    def load(self):
        self.rules = []
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.rules.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for r in self.rules:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 写 ----------
    def add(self, when: str, then: str, scope: str = "general",
            confidence: float = 0.7, evidence: list[str] | None = None) -> dict:
        """同 when 的规则合并：then 相同则升置信度，不同则以高置信度者为准。"""
        for r in self.rules:
            if _cos(_bigrams(r["when"]), _bigrams(when)) > 0.85:
                if _cos(_bigrams(r["then"]), _bigrams(then)) > 0.8:
                    r["confidence"] = min(1.0, float(r.get("confidence", 0.5)) + 0.05)
                    r["hits"] = int(r.get("hits", 0)) + 1
                else:
                    r["confidence"] = max(0.3, float(r.get("confidence", 0.5)) - 0.1)
                    r["evidence"] = (r.get("evidence") or []) + [then]
                self.save()
                return r
        rule = {
            "id": "R" + uuid.uuid4().hex[:6],
            "when": when,
            "then": then,
            "scope": scope,
            "confidence": float(confidence),
            "hits": 1,
            "evidence": evidence or [],
        }
        self.rules.append(rule)
        self.save()
        return rule

    # ---------- 读 ----------
    def retrieve(self, query: str, k: int = 3, scope: str | None = None):
        q = _bigrams(query)
        scored = []
        for r in self.rules:
            if scope and r.get("scope") != scope:
                continue
            s = _cos(q, _bigrams(r["when"] + " " + r["then"]))
            s *= 0.5 + 0.5 * float(r.get("confidence", 0.5))
            scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    # ---------- 反馈 ----------
    def feedback(self, rule_id: str, delta: float = 0.05) -> dict | None:
        for r in self.rules:
            if r["id"] == rule_id:
                r["confidence"] = max(0.05, min(1.0, float(r.get("confidence", 0.5)) + delta))
                r["hits"] = int(r.get("hits", 0)) + (1 if delta > 0 else 0)
                self.save()
                return r
        return None

    # ---------- 维护 ----------
    def prune(self, min_conf: float = 0.15):
        before = len(self.rules)
        self.rules = [r for r in self.rules if float(r.get("confidence", 0)) >= min_conf]
        self.save()
        return before - len(self.rules)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("when")
    a.add_argument("then")
    a.add_argument("--scope", default="general")
    a.add_argument("--conf", type=float, default=0.7)

    p = sub.add_parser("probe")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=3)

    sub.add_parser("list")
    sub.add_parser("prune")

    args = ap.parse_args()
    rm = RuleMemory()

    if args.cmd == "add":
        r = rm.add(args.when, args.then, scope=args.scope, confidence=args.conf)
        print(json.dumps(r, ensure_ascii=False))
    elif args.cmd == "probe":
        for s, r in rm.retrieve(args.query, k=args.k):
            print(f"{s:.3f}  {r['id']} [{r['scope']}] conf={r['confidence']}")
            print(f"   when: {r['when']}")
            print(f"   then: {r['then']}")
    elif args.cmd == "list":
        for r in rm.rules:
            print(f"{r['id']} [{r['scope']}] conf={r['confidence']} hits={r.get('hits')}  {r['when']} → {r['then']}")
    elif args.cmd == "prune":
        print(f"pruned {rm.prune()}")


if __name__ == "__main__":
    main()
