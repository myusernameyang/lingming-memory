/** 灵铭记忆 Host 半：纯坐标 + 存用户&助手的话（带角色标签）+ 自动拉起 + 初始化 + 实时 RL */
import { createMemory } from './memory.js';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRAIN_DIR = join(__dirname, 'train');
let serverProc = null;
let enabled = false;

function ensureServer() {
  if (serverProc) return;
  try {
    serverProc = spawn('python', ['embed_server.py'], { cwd: TRAIN_DIR, stdio: 'ignore' });
    serverProc.on('exit', () => { serverProc = null; });
    serverProc.on('error', () => { serverProc = null; });
  } catch (_) { serverProc = null; }
}

// 检测 Python 环境 + 依赖，缺了就自动 pip install，再拉起 embed 服务
function ensurePython() {
  const python = process.env.LINGMING_PYTHON || 'python';
  const check = spawn(python, ['-c', 'import torch, flask, numpy, requests'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  check.on('error', () => {
    // python 没装：尝试 py / python3
    for (const alt of ['py', 'python3']) {
      const c2 = spawn(alt, ['-c', 'import torch, flask, numpy, requests'], { cwd: TRAIN_DIR, stdio: 'ignore' });
      c2.on('exit', (code) => { if (code === 0) ensureServer(); else pipInstall(alt); });
    }
  });
  check.on('exit', (code) => {
    if (code === 0) ensureServer();
    else pipInstall(python);
  });
}

function pipInstall(python) {
  const install = spawn(python, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: TRAIN_DIR, stdio: 'ignore' });
  install.on('exit', (code) => { if (code === 0) ensureServer(); });
  install.on('error', () => {});
}

async function coordEmbed(text, base) {
  const r = await fetch(`${base}/embed`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`embed 服务返回 ${r.status}`);
  const j = await r.json();
  if (!j.k || j.k.length === 0) throw new Error('embed 服务未初始化');
  return j.k;
}

async function initCoord(base, apiKey) {
  const r = await fetch(`${base}/init`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ apiKey }),
  });
  return r.ok;
}

function rlStep(query, fact, apiKey, base) {
  if (!apiKey || !fact) return;
  fetch(`${base}/rl-step`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, fact, apiKey }),
  }).catch(() => {});
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
    const agentCtx = agent && agent.ctx;
    if (!agentCtx || typeof agentCtx.on !== 'function') return;

    const mem = createMemory({ topK, coordEmbed: (text) => coordEmbed(text, base) });
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

      if (retrieved.length > 0) rlStep(currentText, retrieved[0].text, apiKey, base);

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
      description: '切换灵铭记忆机制（开启后用坐标检索替代长上下文）',
      handler: () => { enabled = !enabled; return { kind: 'success', text: `灵铭记忆已${enabled ? '开启' : '关闭'}` }; },
    });
    commands.register({
      name: 'lingming-init',
      description: '一键初始化坐标网络',
      handler: async () => {
        if (!apiKey) return { kind: 'error', text: '未配置 apiKey' };
        const ok = await initCoord(base, apiKey);
        return ok ? { kind: 'success', text: '坐标网络已初始化' } : { kind: 'error', text: '初始化失败' };
      },
    });
  }

  ctx.effect(() => () => {
    for (const d of agentDisposers.values()) d();
    agentDisposers.clear();
    if (serverProc) { try { serverProc.kill(); } catch (_) {} serverProc = null; }
  });
}
