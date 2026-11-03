/** 灵铭记忆 Host 半：纯坐标 + 存用户&助手的话（带角色标签）+ 自动拉起 + 初始化 + 实时 RL */
import { createMemory } from './memory.js';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRAIN_DIR = join(__dirname, 'train');
const PYTHON = process.env.LINGMING_PYTHON || 'D:\\conda\\python.exe';   // 有 torch 的 Python（系统 python 是商店假 stub）
let serverProc = null;
let enabled = true;   // 默认开（边聊边学），/lingming 可关

function ensureServer() {
  if (serverProc) return;
  try {
    serverProc = spawn(PYTHON, ['embed_server.py'], { cwd: TRAIN_DIR, stdio: 'ignore' });
    serverProc.on('exit', () => { serverProc = null; });
    serverProc.on('error', () => { serverProc = null; });
  } catch (_) { serverProc = null; }
}

// 检测 Python 环境 + 依赖，缺了就自动 pip install，再拉起 embed 服务
function ensurePython() {
  const check = spawn(PYTHON, ['-c', 'import torch, flask, numpy, requests'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  check.on('error', () => {
    // 指定的 Python 没装：尝试 py / python3
    for (const alt of ['py', 'python3']) {
      const c2 = spawn(alt, ['-c', 'import torch, flask, numpy, requests'], { cwd: TRAIN_DIR, stdio: 'ignore' });
      c2.on('exit', (code) => { if (code === 0) ensureServer(); else pipInstall(alt); });
    }
  });
  check.on('exit', (code) => {
    if (code === 0) ensureServer();
    else pipInstall(PYTHON);
  });
}

function pipInstall(python) {
  const install = spawn(python, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  install.on('exit', (code) => { if (code === 0) ensureServer(); });
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
  const apiKey = config.apiKey ?? '';
  const agentDisposers = new Map();
  const commands = typeof ctx.get === 'function' ? ctx.get('commands') : undefined;

  ensurePython();

  ctx.on('agent/created', (payload) => {
    const { agent } = payload;
    const agentCtx = agent && agent.ctx;   // 实践文档：per-agent 行为注册在 agent.ctx 上
    if (!agentCtx || typeof agentCtx.on !== 'function') return;

    const mem = createMemory({ base, topK, apiKey });
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
      description: '切换灵铭记忆（开启后灵元子预测对话、边聊边学）',
      handler: () => { enabled = !enabled; return { kind: 'success', text: `灵铭记忆已${enabled ? '开启' : '关闭'}` }; },
    });
    commands.register({
      name: 'lingming-save',
      description: '保存灵铭记忆（权重 + g 快权重状态）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/save`, { method: 'POST' });
          const j = await r.json();
          return r.ok
            ? { kind: 'success', text: `记忆已保存（g 范数 ${j.g_norm}，g 状态已持久化）` }
            : { kind: 'error', text: '保存失败' };
        } catch (_) { return { kind: 'error', text: '服务未启动，保存失败' }; }
      },
    });
    commands.register({
      name: 'lingming-status',
      description: '查看灵铭记忆状态（开/关 + 喂入次数 + g 状态）',
      handler: async () => {
        try {
          const r = await fetch(`${base}/status`);
          const j = await r.json();
          return { kind: 'success', text: `灵铭记忆：${enabled ? '开' : '关'} | 已喂 ${j.total_writes} 次 | g 范数 ${j.g_norm} | ${j.d}维 ${j.layers}层` };
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
