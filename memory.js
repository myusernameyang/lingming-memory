// 灵铭记忆核心：委托给 g 快权重服务（误差累加 + 碎片匹配）
// add -> POST /write（误差累加进 g + 存记忆）
// retrieve -> POST /retrieve（g@k 重建碎片 → 匹配 → 文本）
export function createMemory(options = {}) {
  const base = options.base;
  const topK = options.topK ?? 3;

  return {
    async add(text, role = 'user') {
      if (!base) return -1;
      const r = await fetch(`${base}/write`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, role }),
      });
      return r.ok ? 0 : -1;
    },
    async retrieve(query, k = topK) {
      if (!base) return [];
      const r = await fetch(`${base}/retrieve`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, topK: k }),
      });
      if (!r.ok) return [];
      const j = await r.json();
      return (j.items || []).map((x) => ({ text: x.text, role: x.role }));
    },
    clear() {},
    size() { return 0; },
  };
}
