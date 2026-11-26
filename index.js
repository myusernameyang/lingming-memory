/** 灵铭记忆 Host 半：坐标寻址多级向量库（存用户&助手的话带角色标签）+ 自动拉起 + 上下文注入

事件契约（来自 Host Event Inspect）：
  agent/assistant-stream  emit    { agent, frame }                     采集助手输出
  agent/status            emit    { agent, status }                    running/idle，idle 落库
  agent/created           serial  { agent, source, signal }            新 agent，挂 per-agent 钩子
  agent/pre-step          waterfall(this=Scoped<Agent>)
        payload { agent, messages, turn, step, signal }
        next() -> PreStepDecision = {kind:'reject'} | {kind:'enter', messages, startsRequestSeries?}
        作用域过滤：只有挂在该 agent scope 上的监听器能收到
        注入方式：await next() 之后返回 {...decision, messages: [...history, ...decision.messages]}
*/
import { createMemory } from './memory.js';
import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import z from '@deepseek-ai/schemastery';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRAIN_DIR = join(__dirname, 'train');
const DEFAULT_PYTHON = process.env.LINGMING_PYTHON || 'D:\\conda\\python.exe';
let serverProc = null;
let serverEnv = {};
let enabled = true;
// 代际编号：重载会留下解不掉的旧监听器，只有最新一代允许干活
let activeGeneration = 0;

/** 取纯文本：content 可能是 string，也可能是 [{type:'text',text}] 块数组 */
function textOf(message) {
  if (!message) return '';
  const c = message.content;
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return c.map((b) => (b && typeof b.text === 'string' ? b.text : '')).join('');
  return '';
}

/** 合法 UserMessage：必须有 id / 块数组 content / source（缺 source 会让 agent 崩溃） */
function userMessage(text) {
  return {
    id: randomUUID(),
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  };
}

export const Config = z.object({
  enabled: z.boolean().default(true).description('总开关：是否启用灵铭记忆（也可用 /lingming 随时切换）').volatile(),
  embedServerUrl: z.string().default('http://127.0.0.1:9000').description('本地记忆服务地址（embed_server.py 的端口）').volatile(),
  python: z.string().default('D:\\conda\\python.exe').description('带 torch + sentence-transformers 的 Python 路径').volatile(),
  topK: z.number().min(1).max(20).step(1).default(5).description('每轮召回的历史记忆条数').volatile(),
  inject: z.boolean().default(false).description('是否自动注入召回（默认关：改用 memory_search 工具按需检索）').volatile(),
  autoCapture: z.boolean().default(false).description('是否自动把每轮对话写入记忆（默认关：改用 memory_write 主动保存）').volatile(),
  minScore: z.number().min(0).max(1).step(0.01).default(0.45).description('相关性阈值：低于它不注入（实测无关 0.30~0.37，相关 0.56~0.72）').volatile(),
  historyTurns: z.number().min(1).max(8).step(1).default(3).description('短查询补语义：拼接最近几轮当检索词').volatile(),
  contextMode: z.union(['augment', 'window', 'replace']).default('augment')
    .description('上下文模式：augment=保留原始历史+注入记忆 / window=只保留最近若干条+记忆 / replace=只留当前消息+记忆').volatile(),
  keepMessages: z.number().min(1).max(40).step(1).default(6)
    .description('window 模式下保留最近几条原始消息').volatile(),
  cMult: z.number().min(1).max(16).step(1).default(4).description('坐标桶倍数：桶数 C = 倍数 × 384').volatile(),
  k: z.number().min(1).max(64).step(1).default(16).description('稀疏 top-k：每条记忆点亮几个桶（C=2048 时 k≥16 才够共享）').volatile(),
  capacity: z.number().min(100).step(1).default(10000).description('记忆容量（超过自动扩容）').volatile(),
  writeScale: z.number().min(0.01).max(2).step(0.01).default(0.1).description('外积写 g 的强度').volatile(),
  gateBias: z.number().min(-1).max(1).step(0.05).default(0.2).description('写门控偏置（越小越倾向写入）').volatile(),
  gateTemp: z.number().min(0.01).max(1).step(0.01).default(0.05).description('写门控温度（越小越硬）').volatile(),
  readTop: z.number().min(1).max(10).step(1).default(3).description('召回聚合条数（取最相关几条求均值）').volatile(),
});

const PY_ENV = { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE', HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' };

function ensureServer(python) {
  if (serverProc) return;
  try {
    serverProc = spawn(python, ['embed_server.py'], { cwd: TRAIN_DIR, stdio: 'ignore', env: { ...PY_ENV, ...serverEnv } });
    serverProc.on('exit', () => { serverProc = null; });
    serverProc.on('error', () => { serverProc = null; });
  } catch (_) { serverProc = null; }
}

function ensurePython(python) {
  const check = spawn(python, ['-c', 'import torch, flask, sentence_transformers'], { cwd: TRAIN_DIR, stdio: 'ignore', env: PY_ENV });
  check.on('error', () => {
    for (const alt of ['py', 'python3']) {
      const c2 = spawn(alt, ['-c', 'import torch, flask, sentence_transformers'], { cwd: TRAIN_DIR, stdio: 'ignore', env: PY_ENV });
      c2.on('exit', (code) => { if (code === 0) ensureServer(alt); else pipInstall(alt); });
    }
  });
  check.on('exit', (code) => {
    if (code === 0) ensureServer(python);
    else pipInstall(python);
  });
}

function pipInstall(python) {
  const install = spawn(python, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: TRAIN_DIR, stdio: 'ignore', env: PY_ENV });
  install.on('exit', (code) => { if (code === 0) ensureServer(python); });
  install.on('error', () => {});
}

function frameText(frame) {
  if (!frame) return '';
  if (frame.type === 'chunk' && frame.chunk && frame.chunk.type === 'text-delta') return frame.chunk.text ?? '';
  const t = frame.text ?? frame.delta ?? frame.content ?? '';
  return typeof t === 'string' ? t : '';
}

function isVolatileRef(value) {
  return typeof value === 'object' && value !== null && typeof value.get === 'function';
}

function derefVolatileConfig(config) {
  const out = {};
  for (const [key, value] of Object.entries(config || {})) {
    out[key] = isVolatileRef(value) ? value.get() : value;
  }
  return out;
}

export function apply(ctx, config = {}) {
  const generation = ++activeGeneration;
  const isCurrent = () => generation === activeGeneration;
  const cfg = derefVolatileConfig(config);
  const topK = cfg.topK ?? 5;
  const base = cfg.embedServerUrl ?? 'http://127.0.0.1:9000';
  const python = cfg.python ?? DEFAULT_PYTHON;
  const injectEnabled = cfg.inject ?? false;
  const autoCapture = cfg.autoCapture ?? false;
  const minScore = cfg.minScore ?? 0.45;
  const historyTurns = cfg.historyTurns ?? 3;
  let contextMode = cfg.contextMode ?? 'augment';
  const keepMessages = cfg.keepMessages ?? 6;
  // 上下文模式以服务端 /api/config 为准（可随时改，即时生效，不依赖命令注册）
  let runtimeMode = contextMode;
  let runtimeKeep = keepMessages;
  let cfgTs = 0;
  async function refreshRuntimeCfg() {
    if (Date.now() - cfgTs < 3000) return;
    cfgTs = Date.now();
    try {
      const r = await fetch(`${base}/api/config`, { signal: AbortSignal.timeout(1500) });
      if (!r.ok) return;
      const j = await r.json();
      const c = (j && j.config) || {};
      if (typeof c.contextMode === 'string') runtimeMode = c.contextMode;
      if (typeof c.keepMessages === 'number') runtimeKeep = c.keepMessages;
    } catch (_) {}
  }
  const maxQueryChars = 600;
  enabled = cfg.enabled ?? true;
  serverEnv = {
    LINGMING_TOPK: String(topK),
    LINGMING_C_MULT: String(cfg.cMult ?? 4),
    LINGMING_K: String(cfg.k ?? 16),
    LINGMING_CAPACITY: String(cfg.capacity ?? 10000),
    LINGMING_WRITE_SCALE: String(cfg.writeScale ?? 0.1),
    LINGMING_GATE_BIAS: String(cfg.gateBias ?? 0.2),
    LINGMING_GATE_TEMP: String(cfg.gateTemp ?? 0.05),
    LINGMING_READ_TOP: String(cfg.readTop ?? 3),
    // center 不下发：由 /config 网页独占，避免两处设置打架
  };
  const commands = typeof ctx.get === 'function' ? ctx.get('commands') : undefined;
  const tools = typeof ctx.get === 'function' ? ctx.get('tools') : undefined;

  ensurePython(python);

  // 健康检查：能连上就不重复拉起；连不上就补一个（避免重启时端口竞争导致永久挂掉）
  async function probe() {
    try {
      const r = await fetch(`${base}/status`, { signal: AbortSignal.timeout(1500) });
      return r.ok;
    } catch (_) { return false; }
  }
  (async () => {
    if (!(await probe())) ensureServer(python);
  })();
  const watchdog = setInterval(async () => {
    if (!enabled) return;
    if (!(await probe()) && !serverProc) ensureServer(python);
  }, 30000);
  if (watchdog.unref) watchdog.unref();

  const mem = createMemory({ base, topK });
  const buffers = new Map();     // agent.id -> 本轮 assistant 累积文本
  const wrote = new Map();       // agent.id -> 已写的 assistant 文本
  const lastUser = new Map();    // agent.id -> 已写的最后一条用户消息
  const recent = new Map();      // agent.id -> 最近几轮用户消息（短查询补语义）
  const agentScope = new Map();  // agent.id -> 记忆作用域（隔离聊天室，2026-11-15）
  const injected = new Map();    // agent.id -> 已注入过的记忆文本（防重复灌同一条）
  const pending = new Map();     // agent.id -> 本轮注入的 [{mid, text}]（等回复后算反馈）
  const disposers = new Map();   // agent.id -> 上一个 pre-step 监听器的解绑函数（防重载累加）
  const turnStart = new Map();   // agent.id -> {turn, idx}：replace 模式下本轮起点

  // 记忆作用域：QQ 聊天室各自独立，别的会话（GUI 主窗口/普通子会话）归 main。
  // 规则：qq- 开头的会话（qq-private-<号> / qq-group-<号>）用 "qq:<完整会话id>"，其余用 main。
  const scopeOfAgent = (agent) => {
    const id = String((agent && agent.id) || '');
    if (!id) return 'main';
    const hit = agentScope.get(id);
    if (hit) return hit;
    const s = id.startsWith('qq-') ? `qq:${id}` : 'main';
    agentScope.set(id, s);
    return s;
  };


  // 冲突判定：结构化（参数键+数值）优先，标识符/余弦兜底
  // 实测依据：单用余弦无法区分"同事实换说法"(0.67~0.91) 与 "同话题新信息"(0.53~0.72)
  const NUM_RE = /\d+(?:\.\d+)?/g;
  const KEY_RE = /(cMult|topK|readTop|capacity|writeScale|gateBias|gateTemp|center|\bk\b|\bC\b|阈值|端口|容量|维度)\D{0,12}(\d+(?:\.\d+)?)/gi;
  const ID_RE = /[A-Za-z][A-Za-z0-9_]{1,}/g;   // 至少两个字符，排除 c / x 这类单字母
  // 文件扩展名、常见英文虚词不算"标识符"，否则路径一撞就误报重复
  const ID_STOP = new Set([
    'md', 'py', 'js', 'ts', 'json', 'txt', 'yml', 'yaml', 'png', 'jpg', 'csv',
    'url', 'http', 'https', 'www', 'com', 'the', 'and', 'for', 'with', 'ion',
    // 通用技术词：撞上它们不代表内容重复（实测会大量误报）
    'memory', 'train', 'model', 'data', 'config', 'plugin', 'server', 'code',
    'file', 'test', 'task', 'run', 'loss', 'gate', 'content', 'coord', 'layer',
    'token', 'tokens', 'value', 'key', 'keys', 'true', 'false', 'none', 'self',
  ]);
  // 年份与个位数不参与"数值重合"判定：2026 / 11 / 1 这类撞上不代表重复
  const isTrivialNum = (v) => /^(19|20)\d{2}$/.test(v) || v.length === 1;

  function judgeConflict(newText, oldText, cos) {
    const params = (t) => {
      const out = {};
      KEY_RE.lastIndex = 0;
      let m;
      while ((m = KEY_RE.exec(t)) !== null) out[m[1].toLowerCase()] = m[2];
      return out;
    };
    const pn = params(newText);
    const po = params(oldText);
    for (const k of Object.keys(pn)) {
      if (k in po && po[k] !== pn[k]) {
        return { level: 'conflict', reason: `参数 ${k}：${po[k]} → ${pn[k]}` };
      }
    }
    const nums = (t) => new Set((t.match(NUM_RE) || []).filter((v) => !isTrivialNum(v)));
    const nn = nums(newText);
    const no = nums(oldText);
    if (nn.size && no.size) {
      const shared = [...nn].filter((v) => no.has(v));
      if (shared.length >= Math.max(1, Math.ceil(nn.size / 2))) {
        return { level: 'duplicate', reason: `共有数值 ${shared.slice(0, 4).join(',')}` };
      }
    }
    if (!nn.size || !no.size) {
      const ids = (t) => new Set(
        (t.match(ID_RE) || []).map((s) => s.toLowerCase()).filter((s) => !ID_STOP.has(s)));
      const a = ids(newText);
      const b = ids(oldText);
      const shared = [...a].filter((v) => b.has(v));
      if (shared.length >= 3) {
        return { level: 'duplicate', reason: `共有标识符 ${shared.slice(0, 4).join(',')}` };
      }
      if (cos >= 0.75) {
        return { level: 'duplicate', reason: `文本高度相似 ${cos.toFixed(3)}` };
      }
    }
    return null;
  }

  function textOut(lines) {
    return [{ type: 'text', text: lines.filter(Boolean).join('\n') }];
  }

  if (tools && typeof tools.register === 'function') {
    tools.register({
      name: 'memory_search',
      description:
        '检索长期记忆库：按关键词或语义查找以前保存的条目（配置、决定、偏好、约定等）。返回按相关度排序的片段及分数；没有匹配时返回空结果。适合回答“上次怎么定的/之前说过什么/某配置是多少”这类问题。',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: '检索词，可以是关键词组合或一句自然语言问句' },
          top_k: { type: 'number', description: '返回条数，默认 5，最大 20' },
          scope: { type: 'string', description: '记忆作用域（可选）：不填=全局共享；"main"=主窗口；"qq:<会话id>"=某个聊天室' },
        },
        required: ['query'],
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            items: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  text: { type: 'string' },
                  role: { type: 'string' },
                  scope: { type: 'string' },
                  score: { type: 'number' },
                  mid: { type: 'number' },
                },
              },
            },
          },
        },
        render(args, value) {
          const v = value ?? {};
          const items = v.items ?? [];
          if (!items.length) return textOut([`没有找到与「${v.query ?? args?.query ?? ''}」相关的记忆`]);
          return textOut([
            `查到 ${items.length} 条：`,
            ...items.map((it, i) => `${i + 1}. [${it.score?.toFixed(3)}] mid=${it.mid} (${it.role}) ${it.text}`),
          ]);
        },
      },
      async execute(args, exec) {
        const query = String(args?.query ?? '').trim();
        const memoryScope = String(args?.scope ?? '').trim() || scopeOfAgent(exec?.agent ?? {});
        if (!query) return { query, items: [] };
        const k = Math.min(Math.max(Number(args?.top_k) || 5, 1), 20);
        let items = [];
        try { items = await mem.retrieve(query, k, memoryScope); } catch (_) { items = []; }
        return {
          query,
          items: items.map((it) => ({
            text: it.text,
            role: it.role,
            score: typeof it.score === 'number' ? Math.round(it.score * 1000) / 1000 : undefined,
            mid: it.mid,
          })),
        };
      },
    });

    tools.register({
      name: 'memory_write',
      description:
        '把一条值得长期保留的信息写入记忆库（配置数值、环境、决定与原因、偏好、约定、踩过的坑）。写入前会自动检查库里是否已有相近条目：若存在重复或可能冲突的记录，不会直接写入，而是把候选返回给你，由你判断该覆盖、合并还是放弃。默认不要在寒暄、临时过程、一次性调试信息上调用。',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: '要保存的内容，写成一句自洽的事实或结论，脱离上下文也能看懂' },
          force: { type: 'boolean', description: '确认为新信息、需与相似条目并存时才设为 true' },
          scope: { type: 'string', description: '记忆作用域（可选）：不填=全局共享（谁都能看到）；"qq:<会话id>"=只给那个聊天室' },
        },
        required: ['text'],
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            written: { type: 'boolean' },
            reason: { type: 'string' },
            conflicts: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  text: { type: 'string' },
                  score: { type: 'number' },
                  level: { type: 'string' },
                },
              },
            },
          },
        },
        render(args, value) {
          const v = value ?? {};
          if (v.written) {
            return textOut([
              '已写入记忆库。',
              ...(v.conflicts?.length
                ? ['另有相近条目（按 force 并存）：',
                   ...v.conflicts.map((c) => `- [${c.level}] ${c.text}`)]
                : []),
            ]);
          }
          if (v.reason === 'duplicate' || v.reason === 'conflict') {
            return textOut([
              v.reason === 'conflict'
                ? '未写入：与已有记忆冲突（同一参数出现了不同取值）。'
                : '未写入：库里已有重复记录。',
              ...(v.conflicts ?? []).map((c) =>
                `- [${c.level}]${c.why ? ' ' + c.why : ''}：${c.text}`),
              '确认是新的、需要并存时加 force: true；若是要更新旧值，请先 memory_forget 删掉旧条目再写。',
            ]);
          }
          return textOut(['未写入：内容为空。']);
        },
      },
      async execute(args, exec) {
        const text = String(args?.text ?? '').trim();
        const memoryScope = String(args?.scope ?? '').trim() || scopeOfAgent(exec?.agent ?? {});
        const force = args?.force === true;
        if (!text) return { written: false, reason: 'empty', conflicts: [] };

        let similar = [];
        try { similar = await mem.retrieve(text, 6, memoryScope); } catch (_) { similar = []; }

        const conflicts = [];
        const related = [];
        const parentCache = new Map();
        async function parentOf(mid, fallback) {
          if (typeof mid !== 'number' || mid < 0) return fallback;
          if (parentCache.has(mid)) return parentCache.get(mid);
          let t = fallback;
          try {
            const r = await fetch(`${base}/msg?id=${mid}`, { signal: AbortSignal.timeout(2000) });
            const j = await r.json();
            if (j && j.text) t = j.text;
          } catch (_) {}
          parentCache.set(mid, t);
          return t;
        }

        for (const s of similar) {
          const cos = typeof s.score === 'number' ? s.score : 0;
          // 分段会切断完整语义，冲突判定用所属原始消息，并重算与原文的余弦
          const full = await parentOf(s.msg, s.text);
          let cosFull = cos;
          if (full !== s.text) {
            try {
              const r = await fetch(`${base}/score`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ a: text, b: full }),
                signal: AbortSignal.timeout(2000),
              });
              const j = await r.json();
              if (j && typeof j.score === 'number') cosFull = j.score;
            } catch (_) {}
          }
          const verdict = judgeConflict(text, full, cosFull);
          if (verdict) {
            conflicts.push({
              text: full,
              score: Math.round(cos * 1000) / 1000,
              level: verdict.level,
              why: verdict.reason,
            });
          } else if (cos >= 0.6) {
            related.push({ text: s.text, score: Math.round(cos * 1000) / 1000 });
          }
        }

        if (conflicts.length && !force) {
          return { written: false, reason: conflicts[0].level, conflicts, related };
        }
        const ok = await mem.add(text, 'user', memoryScope);
        if (ok >= 0) fetch(`${base}/save`, { method: 'POST' }).catch(() => {});
        return { written: ok >= 0, conflicts, related };
      },
    });

    tools.register({
      name: 'memory_forget',
      description:
        '删除一条过时或错误的记忆，避免旧信息继续被检索到。用 memory_search 返回的 mid 指定要删的条目；不确定是哪条时先检索再删。',
      parameters: {
        type: 'object',
        properties: {
          mid: { type: 'number', description: 'memory_search 结果里的 mid' },
        },
        required: ['mid'],
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            ok: { type: 'boolean' },
            mid: { type: 'number' },
            n_mem: { type: 'number' },
          },
        },
        render(args, value) {
          const v = value ?? {};
          return textOut([
            v.ok ? `已删除 mid=${v.mid}，剩余 ${v.n_mem} 条` : `未找到 mid=${v.mid} 或已删除`,
          ]);
        },
      },
      async execute(args) {
        const mid = Number(args?.mid);
        if (!Number.isFinite(mid)) return { ok: false };
        try {
          const r = await fetch(`${base}/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mid }),
          });
          return await r.json();
        } catch (_) {
          return { ok: false, mid };
        }
      },
    });
  }
  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    try {
      if (!enabled || !agent || !isCurrent()) return;
      const t = frameText(frame);
      if (t) buffers.set(agent.id, (buffers.get(agent.id) ?? '') + t);
    } catch (_) {}
  });

  // 2) 一轮结束落库（全局事件）
  ctx.on('agent/status', async ({ agent, status }) => {
    try {
      if (!enabled || !agent || status !== 'idle' || !isCurrent()) return;
      const text = (buffers.get(agent.id) ?? '').trim();
      buffers.delete(agent.id);

      // 用后反馈：看这轮回复里有没有真的用上注入的记忆（灵铭据此学会该召回什么）
      const pend = pending.get(agent.id) || [];
      pending.delete(agent.id);
      if (pend.length && text) {
        const norm = (s) => String(s || '').replace(/[\s，。！？；、,.!?;:「」“”"'']/g, '');
        const reply = norm(text);
        const items = pend
          .filter((p) => Number.isInteger(p.mid))
          .map((p) => {
            const t = norm(p.text);
            const probe = t.slice(0, Math.min(8, t.length));
            return { mid: p.mid, used: !!probe && reply.includes(probe) };
          });
        if (items.length) {
          fetch(`${base}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items }),
          }).catch(() => {});
        }
      }

      if (!text || wrote.get(agent.id) === text) return;
      if (!autoCapture) return;     // 默认不自动采集，靠 memory_write 主动保存
      wrote.set(agent.id, text);
      await mem.add(text, 'assistant', scopeOfAgent(agent)).catch(() => {});
      // 回合边界即存盘：避免重启丢掉未到自动保存阈值的记忆
      fetch(`${base}/save`, { method: 'POST' }).catch(() => {});
    } catch (_) {}
  });

  // 3) 每个 agent 的 pre-step：写用户消息 + 注入召回（必须挂在 agent scope）
  const setupAgent = (agent) => {
    try {
      const agentCtx = agent && agent.ctx;
      if (!agentCtx || typeof agentCtx.on !== 'function') return;
      // 重载时先解绑旧的，否则每重载一次就多注入一份
      const prev = disposers.get(agent.id);
      if (typeof prev === 'function') { try { prev(); } catch (_) {} }
      const off = agentCtx.on('agent/pre-step', async (payload, next) => {
        const decision = await next();
        if (!isCurrent()) return decision;   // 旧代际监听器一律不做任何事
        try {
          if (!enabled || !decision || decision.kind !== 'enter') return decision;

          // 取最新的上下文模式（服务端配置优先）
          await refreshRuntimeCfg();

          const msgs = (payload && payload.messages) || decision.messages || [];
          const current = msgs[msgs.length - 1];
          const currentText = textOf(current).trim();
          const isNew = !!currentText && lastUser.get(agent.id) !== currentText;

          // 先检索、再写入：否则刚存进去的这条会把自己召回（自我注入）
          // 归一化工具提到外面：后面幂等过滤也要用，放 if 块里会 ReferenceError
          const norm = (s) => String(s || '').replace(/[\s，。！？；、,.!?;:「」“”"'']/g, '');
          let keep = [];
          let query = '';
          if (injectEnabled && currentText) {
            const win = recent.get(agent.id) || [];
            const parts = (win.length > 1 ? win.slice(0, -1) : []).concat([currentText]);
            query = parts.join(' ').trim();
            if (query.length > maxQueryChars) query = query.slice(-maxQueryChars);

            let retrieved = [];
            try { retrieved = await mem.retrieve(query, topK, scopeOfAgent(agent)); } catch (_) { retrieved = []; }

            // 相关性闸 + 批内/跨轮去重 + 排除助手回声 + 排除当前消息自身
            const seen = injected.get(agent.id) || new Set();
            const usedMsg = new Set();      // 同一条消息只取一段，避免相近副本占满名额
            const cur = norm(currentText);
            keep = [];
            for (const r of retrieved) {
              if (typeof r.score === 'number' && r.score < minScore) continue;
              const t = norm(r.text);
              if (!t || seen.has(t)) continue;                     // 批内 + 跨轮去重
              if (cur && (cur.includes(t) || t.includes(cur))) continue;  // 当前消息自身
              if (typeof r.msg === 'number' && r.msg >= 0) {
                if (usedMsg.has(r.msg)) continue;
                usedMsg.add(r.msg);
              }
              seen.add(t);
              keep.push(r);
            }
            injected.set(agent.id, seen);
            if (keep.length) {
              const pend = pending.get(agent.id) || [];
              pend.push(...keep.map((r) => ({ mid: r.mid, text: r.text })));
              pending.set(agent.id, pend);
            }
          }

          // 再写入当前消息（只有开启自动采集才落库）
          if (isNew) {
            lastUser.set(agent.id, currentText);
            if (autoCapture) await mem.add(currentText, 'user', scopeOfAgent(agent)).catch(() => {});
            const r = recent.get(agent.id) || [];
            r.push(currentText);
            while (r.length > historyTurns) r.shift();
            recent.set(agent.id, r);
          }

          if (!keep.length && runtimeMode === 'augment') return decision;
          // 幂等保护：上下文里已有的内容不再重复注入（防止残留监听器造成多份）
          const exist = new Set((decision.messages || []).map((m) => norm(textOf(m))));
          keep = keep.filter((r) => !exist.has(norm(r.text)));

          // 上下文接管：与"是否有召回"解耦——window/replace 即使没召回也要裁剪
          // 注意：变量名不能叫 base，否则会遮蔽外层的服务地址
          let keptMsgs = decision.messages || [];
          const msgsIn = keptMsgs.length;
          if (runtimeMode === 'replace') {
            // 只保留"本轮开始之后"的消息：跨轮不记忆，但同一轮的多步工具调用保留
            const rec = turnStart.get(agent.id);
            if (!rec || rec.turn !== payload.turn) {
              turnStart.set(agent.id, { turn: payload.turn, idx: Math.max(0, keptMsgs.length - 1) });
            }
            keptMsgs = keptMsgs.slice(turnStart.get(agent.id).idx);
          } else if (runtimeMode === 'window') {
            keptMsgs = keptMsgs.slice(-runtimeKeep);
          }
          const msgsKept = keptMsgs.length;

          const history = keep.map((r) =>
            userMessage(`（灵铭记忆·${r.role === 'assistant' ? '我曾说过' : '你曾说过'}，非本轮输入）${r.text}`));

          // 上报本轮情况，便于在 /injections 观察
          fetch(`${base}/log_injection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              agent: agent.id,
              mode: runtimeMode,
              msgsIn,
              msgsKept,
              query,
              items: keep.map((r) => ({ score: r.score ?? 0, role: r.role, text: r.text })),
            }),
          }).catch(() => {});

          return { ...decision, messages: [...history, ...keptMsgs] };
        } catch (_) {
          return decision;   // 记忆层任何异常都不能影响对话
        }
      });
      if (typeof off === 'function') disposers.set(agent.id, off);
    } catch (_) {}
  };

  ctx.on('agent/created', (payload) => {
    const agent = (payload && payload.agent) || payload;
    if (agent) setupAgent(agent);
  });

  // 插件启用前就已存在的 agent 也挂上
  try {
    const agents = typeof ctx.get === 'function' ? ctx.get('agents') : undefined;
    const list = agents && typeof agents.list === 'function' ? agents.list() : [];
    for (const a of list) setupAgent(a);
  } catch (_) {}

  if (commands && typeof commands.register === 'function') {
    commands.register({
      name: 'lingming',
      description: '切换灵铭记忆（开启后本地记忆边聊边存、并注入召回）',
      handler: () => { enabled = !enabled; return { kind: 'success', text: `灵铭记忆已${enabled ? '开启' : '关闭'}` }; },
    });
    commands.register({
      name: 'lingming-save',
      description: '保存灵铭记忆（持久化记忆库）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/save`, { method: 'POST' });
          const j = await r.json();
          return r.ok ? { kind: 'success', text: `记忆已保存（${j.n_mem} 条）` } : { kind: 'error', text: '保存失败' };
        } catch (_) { return { kind: 'error', text: '服务未启动，保存失败' }; }
      },
    });
    commands.register({
      name: 'lingming-context',
      description: '切换上下文模式：augment / window / replace（replace 只用记忆替代历史）',
      handler: async (arg) => {
        const m = String(arg || '').trim();
        if (['augment', 'window', 'replace'].includes(m)) {
          runtimeMode = m;
          contextMode = m;
          cfgTs = 0;
          try {
            await fetch(`${base}/api/config`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ config: { contextMode: m } }),
            });
          } catch (_) {}
        }
        return { kind: 'success', text: `上下文模式：${runtimeMode}（augment=历史+记忆 / window=最近${runtimeKeep}条+记忆 / replace=仅当前+记忆）` };
      },
    });
    commands.register({
      name: 'lingming-status',
      description: '查看灵铭记忆状态（开/关 + 记忆数 + 坐标桶数）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/status`);
          const j = await r.json();
          return { kind: 'success', text: `灵铭记忆：${enabled ? '开' : '关'} | 注入 ${injectEnabled ? '开' : '关'} | 上下文 ${runtimeMode}（保留 ${runtimeKeep} 条）| 已存 ${j.n_mem} 条 | ${j.n_bars} 桶 | ${j.encoder}` };
        } catch (_) { return { kind: 'success', text: `灵铭记忆：${enabled ? '开' : '关'}（服务未启动）` }; }
      },
    });
    commands.register({
      name: 'lingming-config',
      description: '打开灵铭记忆可视化配置页',
      handler: () => ({ kind: 'success', text: `灵铭记忆配置页：${base}/config （浏览器打开即可改参数 + 保存）` }),
    });
  }

  ctx.effect(() => () => {
    buffers.clear();
    clearInterval(watchdog);
    for (const d of disposers.values()) { try { d(); } catch (_) {} }
    disposers.clear();
    if (serverProc) { try { serverProc.kill(); } catch (_) {} serverProc = null; }
  });
}
