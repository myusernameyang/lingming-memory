// 灵铭记忆核心：委托给坐标寻址服务
// add -> POST /write（存记忆 → 坐标 top-k 分桶）
// retrieve -> POST /retrieve（坐标桶交集 → 命中最相关记忆）
export function createMemory(options = {}) {
  const base = options.base;
  const topK = options.topK ?? 3;
  const apiKey = options.apiKey ?? '';

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
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, topK: k, apiKey }),
      });
      if (!r.ok) return [];
      const j = await r.json();
      return (j.items || []).map((it) => ({ text: it.text, role: it.role || 'user' }));
    },
  };
}
