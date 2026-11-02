// 灵铭记忆核心：纯坐标记忆，每条带角色标签（用户/助手），检索时带上标签防"谁说的"干扰
// coordEmbed 必须提供：async (text) => number[]，由 embed 服务返回坐标 k
export function createMemory(options = {}) {
  const topK = options.topK ?? 3;
  const coordEmbed = options.coordEmbed || null;
  const records = []; // { text, role, embedding }

  function cosine(a, b) {
    const n = Math.min(a.length, b.length);
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  return {
    async add(text, role = 'user') {
      if (!coordEmbed) return -1;
      const k = await coordEmbed(text);
      if (k && k.length > 0) records.push({ text, role, embedding: k });
      return records.length - 1;
    },
    async retrieve(query, k = topK) {
      if (!coordEmbed || records.length === 0) return [];
      const q = await coordEmbed(query);
      if (!q || q.length === 0) return [];
      const scored = records.map((r, i) => ({ i, text: r.text, role: r.role, score: cosine(q, r.embedding) }));
      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, k).filter((r) => r.score > 0).map((r) => ({ text: r.text, role: r.role }));
    },
    clear() { records.length = 0; },
    size() { return records.length; },
  };
}
