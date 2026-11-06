/** 灵铭记忆 Host 半：坐标寻址多级向量库（存用户&助手的话带角色标签）+ 自动拉起 + 初始化 */
import { createMemory } from './memory.js';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import z from '@deepseek-ai/schemastery';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRAIN_DIR = join(__dirname, 'train');
const DEFAULT_PYTHON = process.env.LINGMING_PYTHON || 'D:\\conda\\python.exe';   // 有 torch 的 Python（系统 python 是商店假 stub）
let serverProc = null;
let serverEnv = {};
let enabled = true;   // 默认开（边聊边学），/lingming 可关

/** 配置面板（schemastery schema，字段都标 .volatile() 才会出现在设置表单里） */
export const Config = z.object({
  topK: z.number().min(1).max(20).step(1).default(3).description('每轮召回的历史记忆条数').volatile(),
  embedServerUrl: z.string().default('http://127.0.0.1:9000').description('本地记忆服务地址（embed_server.py 的端口）').volatile(),
  python: z.string().default('D:\\conda\\python.exe').description('带 torch 的 Python 可执行文件路径').volatile(),
  enabled: z.boolean().default(true).description('默认是否开启（可随时用 /lingming 切换）').volatile(),
  d: z.number().min(32).step(1).default(128).description('模型维度（embedding 维度，改后需重新训练坐标 coord.pt）').volatile(),
  cMult: z.number().min(1).step(1).default(4).description('横向倍数：坐标 bar 数 = cMult × d（容量更大、寻址更细）').volatile(),
  k: z.number().min(1).step(1).default(8).description('稀疏 top-k：每条记忆点亮几个坐标 bar（越大召回越稳）').volatile(),
  capacity: z.number().min(1).step(1).default(10000).description('向量库容量：最多存多少条记忆（超过按最早写入淘汰）').volatile(),
});

function ensureServer(python) {
  if (serverProc) return;
  try {
    serverProc = spawn(python, ['embed_server.py'], { cwd: TRAIN_DIR, stdio: 'ignore', env: { ...process.env, ...serverEnv } });
    serverProc.on('exit', () => { serverProc = null; });
    serverProc.on('error', () => { serverProc = null; });
  } catch (_) { serverProc = null; }
}

// 检测 Python 环境 + 依赖，缺了就自动 pip install，再拉起 embed 服务
function ensurePython(python) {
  const check = spawn(python, ['-c', 'import torch, flask'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  check.on('error', () => {
    // 指定的 Python 没装：尝试 py / python3
    for (const alt of ['py', 'python3']) {
      const c2 = spawn(alt, ['-c', 'import torch, flask'], { cwd: TRAIN_DIR, stdio: 'ignore' });
      c2.on('exit', (code) => { if (code === 0) ensureServer(alt); else pipInstall(alt); });
    }
  });
  check.on('exit', (code) => {
    if (code === 0) ensureServer(python);
    else pipInstall(python);
  });
}

function pipInstall(python) {
  const install = spawn(python, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  install.on('exit', (code) => { if (code === 0) ensureServer(python); });
  install.on('error', () => {});
}

function frameText(frame) {
  if (!frame) return '';
  const t = frame.text ?? frame.delta ?? frame.content ?? '';
  return typeof t === 'string' ? t : '';
}

export function apply(ctx, config = {}) {
  const topK = config.topK ?? 3;
  const base = config.embedServerUrl ?? 'http://127.0.0.1:9000';
  const python = config.python ?? DEFAULT_PYTHON;
  enabled = config.enabled ?? true;
  const d = config.d ?? 128;
  const cMult = config.cMult ?? 4;
  const k = config.k ?? 8;
  const capacity = config.capacity ?? 10000;
  serverEnv = {
    LINGMING_D: String(d),
    LINGMING_C_MULT: String(cMult),
    LINGMING_K: String(k),
    LINGMING_CAPACITY: String(capacity),
  };
  const agentDisposers = new Map();
  const commands = typeof ctx.get === 'function' ? ctx.get('commands') : undefined;

  ensurePython(python);

  ctx.on('agent/created', (payload) => {
    const { agent } = payload;
    const agentCtx = agent && agent.ctx;   // 实践文档：per-agent 行为注册在 agent.ctx 上
    if (!agentCtx || typeof agentCtx.on !== 'function') return;

    const mem = createMemory({ base, topK });
    const assistantBuf = { text: '' };

    // 累积助手（模型）的流式回复
    agentCtx.on('agent/assistant-stream', (sp) => {
      const t = frameText(sp && sp.frame);
      if (t) assistantBuf.text += t;
    });

    const dispose = agentCtx.on('agent/pre-step', async (p, next) => {
      const decision = await next();
      if (!enabled || decision.kind !== 'enter') return decision;

      const msgs = p.messages || [];

      // 把上一轮助手回复写进记忆（角色 assistant），再清空缓冲
      if (assistantBuf.text.trim()) {
        await mem.add(assistantBuf.text.trim(), 'assistant').catch(() => {});
        assistantBuf.text = '';
      }

      // 用户消息写进记忆（角色 user）
      const texts = msgs.map((m) => (typeof m.content === 'string' ? m.content : '')).filter((t) => t.length > 0);
      const current = msgs[msgs.length - 1];
      const currentText = current && typeof current.content === 'string' ? current.content : '';
      for (const t of texts.slice(0, -1)) await mem.add(t, 'user').catch(() => {});

      // 检索相关记忆（用户 + 助手都检索，带角色标签）
      let retrieved = [];
      try {
        retrieved = await mem.retrieve(currentText, topK);
      } catch (_) { retrieved = []; }

      const history = retrieved.map((r) => ({
        role: 'user',
        content: `[${r.role === 'assistant' ? '助手' : '用户'}] ${r.text}`,
      }));
      return { ...decision, messages: [...history, current] };
    });

    agentDisposers.set(agent.id, dispose);
  });

  if (commands && typeof commands.register === 'function') {
    commands.register({
      name: 'lingming',
      description: '切换灵铭记忆（开启后本地坐标寻址记忆边聊边学）',
      handler: () => { enabled = !enabled; return { kind: 'success', text: `灵铭记忆已${enabled ? '开启' : '关闭'}` }; },
    });
    commands.register({
      name: 'lingming-save',
      description: '保存灵铭记忆（坐标权重持久化）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/save`, { method: 'POST' });
          const j = await r.json();
          return r.ok
            ? { kind: 'success', text: `记忆已保存（坐标权重已持久化到 coord.pt）` }
            : { kind: 'error', text: '保存失败' };
        } catch (_) { return { kind: 'error', text: '服务未启动，保存失败' }; }
      },
    });
    commands.register({
      name: 'lingming-status',
      description: '查看灵铭记忆状态（开/关 + 记忆数 + 坐标桶数）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/status`);
          const j = await r.json();
          return { kind: 'success', text: `灵铭记忆：${enabled ? '开' : '关'} | 已存 ${j.n_mem} 条记忆 | ${j.n_bars} 个坐标桶` };
        } catch (_) { return { kind: 'success', text: `灵铭记忆：${enabled ? '开' : '关'}（服务未启动）` }; }
      },
    });
  }

  ctx.effect(() => () => {
    for (const d of agentDisposers.values()) d();
    agentDisposers.clear();
    if (serverProc) { try { serverProc.kill(); } catch (_) {} serverProc = null; }
  });
}
