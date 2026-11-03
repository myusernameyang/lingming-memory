"""600M 'point cloud retrieves + attention integrates + FFN predicts' model,
self-contained for the remote A800.  NO explicit self-state.

Point cloud = trainable landmarks; recall by RBF direction lookup (no softmax).
Attention integrates the recalled clue; FFN predicts.  Byte-level LM.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

DATA = "/root/autodl-tmp/data20g.txt"
OUTDIR = "/root/autodl-tmp/think600m_mgran_v69"
MAX_BYTES = 20_000_000_000

H_G = 1   # 单头 g：一个方向贯穿全程（content = query * contents 逐元素门控，不用多头）

USE_CORRECTION = True   # 快权重 g 纠偏分支开（配合 warmup LR：基线已确认是 LR 问题，g 照常加回来）

READ_DEPTH = 1         # 深度回忆跳数：1 = 单跳，>1 = content 递归查（深度回忆）

EF_GAMMA = 1.0   # entropy feedback: flatten over-confident landmark selection

G_LAMBDA = 0.99   # g 写：指数衰减 g = λ·g + delta（λ<1 收敛到稳态 g*=delta/(1-λ)，旧记忆 fade；0.99 让记忆能跨 ~100 forward 存住，帧存得住）

C_COORD = 256   # 时空坐标场：g 是 [C,d] 坐标场（C 个坐标槽，每个存一个 d 维内容向量），读=坐标寻址查表，写=误差写回坐标
C_V, C_TH, C_PH = 4, 8, 8   # 3D 球坐标 (r,θ,φ)：速度(4) × 方位(8) × 仰角(8)，C = 4*8*8 = 256；文本只用速度维，θ/φ 留空

K_HEAD = 4      # 多预测（K 步前看）：一个位置同时预测未来 K 个字节（像先想词不是字母），训练信号 ×K、生成投机解码 ×K 提速


def load_bytes(path, max_bytes=MAX_BYTES):
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    return torch.frombuffer(raw, dtype=torch.uint8)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6) * self.w


def precompute_rope(head_dim, max_seq_len, base=10000.0, device="cpu"):
    """Standard RoPE: order encoding for Q/K (fixed multi-frequency rotation)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)   # [max_seq_len, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE to a [B,H,S,dh] tensor. cos/sin: [S, dh//2]."""
    d = x.shape[-1]
    S = x.shape[-2]
    cos = cos[:S, : d // 2].view(1, 1, S, d // 2)
    sin = sin[:S, : d // 2].view(1, 1, S, d // 2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class PointCloud(nn.Module):
    def __init__(self, d=1280, c=64, M=256, H_g=8):
        super().__init__()
        self.H_g = H_g
        self.dh_g = d // H_g
        self.C = C_COORD                                                # 保留（历史），连续坐标场不再用离散槽
        self.R = nn.Parameter(torch.randn(H_g, d // H_g, d // H_g) * 0.05)  # [H_g,dh_g,dh_g] 选择头（Rh）
        self.c_scale2 = nn.Parameter(torch.tensor(0.3))         # 插值系数 λ：h += λ·content（注入强度）
        self.T_net = nn.Linear(d, 3)                            # T(d)：h → (r,θ,φ) 3D 坐标（学出来）
        self.W_expand = nn.Parameter(torch.randn(3, d) * 0.02)  # [3,d]：3D 坐标 → d 维嵌入（学出来的展开，前后统一，不是固定 RoPE）
        self.contents = nn.Parameter(torch.randn(d, d) * (1.0 / math.sqrt(d)))  # 知识矩阵：把坐标处读到的原始记忆投影成内容


class Block(nn.Module):
    def __init__(self, d=1280, H=16, M=256, err_decay=1.0):
        super().__init__()
        self.err_decay = err_decay                              # 误差逐层衰减（按层数固定：浅层误差大、深层误差归零）
        self.inhibit_beta = 1.0                                 # 放大独特系数 β：query_li = query + β·(query-mean)，保留共有+放大独特
        self.H = H
        self.dh = d // H
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)                                    # residual norm（每层末尾压回单位，防 h 跨层无界增长 → inf）
        self.n3 = RMSNorm(d)                                    # FFN Pre-LN（对齐 n1）：x_mid 进 W_up/W_gm 前归一化，压住注入/attention 对 FFN 的幅度影响
        self.content_norm = RMSNorm(d)                          # 检索结果归一化：content → |·|=sqrt(d)，对齐 h，切断 content↔contents 幅度正反馈
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        # G_nav: sigmoid(W_delta(x)) gates Q/K/V (drops unneeded dims)
        self.W_delta = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.W_delta.weight, std=0.001)
        self.pc = PointCloud(d, M, H_g=H_G) if USE_CORRECTION else None   # per-layer correction (纠偏, multi-head g)
        # Closure FFN（原始 feature-level gating, v34 消融）:
        #   W_up (D→2D) --ReLU--> Gm 门控 --> W_inner (2D→2D) --ReLU--> W_down (2D→D)
        self.W_up = nn.Linear(d, 2 * d, bias=False)               # 升维: D→2D（原子特征字典）
        self.W_inner = nn.Linear(2 * d, 2 * d, bias=False)        # 子空间混合: 2D→2D
        self.W_down = nn.Linear(2 * d, d, bias=False)             # 降维: 2D→D
        self.W_gm = nn.Linear(d, 2 * d, bias=False)               # router: d→2D (input-dependent)
        self.gm_temp = 0.2
        nn.init.normal_(self.W_up.weight, std=0.02)
        nn.init.normal_(self.W_inner.weight, std=0.02)
        nn.init.normal_(self.W_down.weight, std=0.02)
        nn.init.zeros_(self.W_gm.weight)
        self.W_gate = nn.Linear(d, M, bias=False)               # contents 门控（对齐 Gm router，按内容槽 M 选择）
        nn.init.zeros_(self.W_gate.weight)
        self.last_read_ent = 0.0                               # 实时诊断：本层 read 的注意力熵
        self.expl_err = 0.0                                   # 解释误差（先验 g·h 没解释掉的部分）
        self.orth_err = 0.0                                   # 正交限制（h_norm 和 g 的点积平方）
        self.read_ent = 0.0                                   # read 熵（tensor，负熵惩罚用）
        self.W_seg = nn.Linear(d, 1, bias=False)            # learned segment boundary (segmentation by attention)
        nn.init.normal_(self.W_seg.weight, std=0.02)
        self.q_pool = nn.Parameter(torch.randn(H, self.dh) * 0.02)   # shared pooling query for coarse summary
        self.max_seg = 128                                           # cap on number of soft segments

    def forward(self, h, g, t_g, mask, pad_mask=None):
        B, S, D = h.shape
        x_in = h
        q, k, v = self.qkv(self.n1(h)).chunk(3, -1)      # [B,S,D]
        Gh = torch.sigmoid(self.W_delta(x_in))           # G_nav gate from X
        q, k, v = q * Gh, k * Gh, v * Gh
        q = q.view(B, S, self.H, self.dh).transpose(1, 2)
        k = k.view(B, S, self.H, self.dh).transpose(1, 2)
        v = v.view(B, S, self.H, self.dh).transpose(1, 2)
        q = q - q.mean(dim=-1, keepdim=True)             # lateral inhibition (remove common-mode)
        k = k - k.mean(dim=-1, keepdim=True)
        # Q/K order encoding (standard RoPE), separate from content spacetime encoding.
        cos, sin = precompute_rope(self.dh, S, base=10000.0, device=h.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # standard causal token attention (RoPE + G_nav gate)，去掉了分组/分段 attention
        scale = 1.0 / math.sqrt(self.dh)
        m = mask                                                  # causal & padding
        sc = torch.einsum("bhid,bhjd->bhij", q, k) * scale          # [B,H,S,S]
        sc = sc.masked_fill(~m, float("-inf"))
        sc = sc.clamp(max=30.0)
        w = F.softmax(sc, dim=-1)
        self.last_w_tok = w                                       # 探针：token 注意力权重
        o = torch.einsum("bhij,bhjd->bhid", w, v).transpose(1, 2).reshape(B, S, D)
        h = h + self.o(o)
        # correction (纠偏, multi-head g 拆维度): 每头负责 dh_g 维切片，独立偏转 + 读，concat 拼回
        if USE_CORRECTION:
            # 级数（连续坐标场）：g = Σ 特征 ⊗ k(坐标)，k(坐标)=RoPE 核
            # T(d)：h → (r,θ,φ) 连续坐标（非整数），写读同址（同一个 T(d) 算出的坐标），全局记忆
            dh_g = self.pc.dh_g
            h_slices = h.view(B, S, H_G, dh_g)                      # [B,S,1,d]
            self.last_h = h                                         # 探针：h（纠偏前）
            h_mean = h.mean(dim=1, keepdim=True)                     # [B,1,d] 结构（各位置共有方向）
            # T(d)：h → (r,θ,φ) 连续坐标（学出来，跟 X 无关，是存储位置；不 tanh 避免量化，clip 防发散，溢出惩罚学尺度）
            rtp = self.pc.T_net(h).clamp(-5.0, 5.0)                  # [B,S,3] 连续坐标（clip 硬边界兜底）
            # 学出来的展开：[3,d] 矩阵把 3D 坐标展开成 d 维嵌入（前后统一，不是固定 RoPE）
            k = rtp @ self.pc.W_expand                               # [B,S,3] @ [3,d] = [B,S,d]
            k = F.normalize(k, dim=-1, eps=1e-6)                     # [B,S,d] 归一化
            # 时间快权重：t_g 跟 g 一样的算法（λ·t_g + "1"⊗k 外积写，λ 逐层衰减）
            t_delta = k.reshape(B * S, D).sum(dim=0) / (B * S)       # [D] "1" 权重的核求和
            t_g = G_LAMBDA * t_g + t_delta                           # [D] 时间快权重（跟 g 一样的 λ·t_g + delta）
            t_raw = torch.einsum("d,bsd->bs", t_g, k)                # [B,S] 最近活动量（1=刚写，λ 老化）
            self.last_t_raw = t_raw                                          # 探针：最近性（非 detach，供泡内惩罚用梯度）
            # 读：raw = g @ k（级数在坐标处求值），content = RMSNorm(contents @ raw)（知识投影从混叠里挑相关）
            raw = torch.einsum("bde,bse->bsd", g, k)                   # [B,S,d] 级数求值（per-batch g：每样本读自己的记忆）
            content = self.content_norm(torch.einsum("de,bse->bsd", self.pc.contents, raw))  # [B,S,d] 知识投影 → RMSNorm
            # 写：g = λ·g + err ⊗ k（误差 = 突出信号 − 已记住的，有符号 + 带 -content 覆盖；自纠错，λ 高也不饱和）
            error = (h - h_mean) - content                            # [B,S,d] 误差（有符号，-content 覆盖旧记忆）
            err = F.normalize(error, dim=-1, eps=1e-6)                # [B,S,d]
            delta = (0.5 / S) * torch.einsum("bsd,bse->bde", err, k)   # [B,d,d] 每样本各自 errᵀ·k（不跨 batch 混）
            g = G_LAMBDA * g + delta                                  # [B,d,d] 每样本独立衰减累积
            self.last_content_h_probe = content.reshape(B, S, D)      # 探针：content
            self.last_coord = (rtp[..., 0].mean().detach(), rtp[..., 0].std().detach(),
                               rtp[..., 1].mean().detach(), rtp[..., 1].std().detach(),
                               rtp[..., 2].mean().detach(), rtp[..., 2].std().detach())  # 探针：3D 坐标（均值+方差，看是否连续分散）
            self.last_rtp = rtp                                       # 探针：完整坐标 [B,S,3]（非 detach，供 overflow 惩罚用梯度）
            # 方向变化：h 注入前后的方向差，判断注入是否真的改了 h（不是单纯注入、不改变方向）
            injection = self.pc.c_scale2 * content                    # [B,S,d]
            h_after = h + injection                                   # [B,S,d]
            cos_sim = F.cosine_similarity(h, h_after, dim=-1, eps=1e-6)  # [B,S] ∈ [-1,1]
            self.last_dir_change = 1.0 - cos_sim.abs()                   # [B,S] 方向变化（1-|cos|：反转/同向=0，垂直=1，防"反转 h"刷分）
            h = h_after                                               # 注入（幅度 + 强度可学 c_scale2）
        x_mid = h
        x_n = self.n3(x_mid)                               # Pre-LN：FFN 输入归一化（对齐 n1），注入/attention 的幅度不再直接灌进 FFN
        f = F.relu(self.W_up(x_n))                         # [B,S,2D] 升维 + 第一层 relu
        gate = torch.sigmoid(self.W_gm(x_n) / self.gm_temp)  # [B,S,2D] 第二层选择 (input-dependent 路由)
        f = f * gate                                       # 门控特征
        f = F.relu(self.W_inner(f))                        # [B,S,2D] 子空间混合 + 第三层 relu
        h = h + self.W_down(f)                             # [B,S,D] 降维
        h = self.n2(h)                                       # residual norm（防 h 跨层无界增长）
        return h, g, t_g


class Model(nn.Module):
    def __init__(self, d=1280, c=64, M=256, n_layers=28, H=16, seq=512, paper_len=64, causal=True):
        super().__init__()
        self.seq = seq
        self.paper_len = paper_len
        self.embed = nn.Embedding(256, d)
        self.blocks = nn.ModuleList([Block(d, H, M, err_decay=1.0) for i in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, K_HEAD * 256, bias=False)   # K 步前看：K 个头，每个预测一个未来字节
        self.register_buffer("g_init", torch.randn(d, d) * 0.02)  # 持久状态（g 连续坐标场 [d,d] 快权重，读=g@k(x) 位置寻址，就地覆盖）
        self.register_buffer("t_g_init", torch.zeros(d))          # 时间快权重 t_g（[d] 向量，记录"最近性"：写"1"、λ 老化，读=t_g·k）
        self.register_buffer("paper", torch.zeros(paper_len, dtype=torch.long))  # 自述纸：模型自己说过的话（前向，detach，只走前向）
        if causal:
            mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))   # 因果（预测下一字节）
        else:
            mask = torch.ones(seq, seq, dtype=torch.bool)               # 双向（感受误差/背题，不预测）
        mask = mask.view(1, 1, seq, seq)
        self.register_buffer("mask", mask)
        self.ent_alpha = 0.0                                # entropy penalty weight (0 = disabled)

    def forward(self, x, pad_mask=None):
        B, S = x.shape
        P = min(self.paper_len, S)
        # 自述前缀（只走前向）：把上一轮模型自己生成的字节拼到输入前面，模型先"读自己说过的话"再读数据
        paper = self.paper[:P].unsqueeze(0).expand(B, -1).to(x.device)   # [B, P]
        x_full = torch.cat([paper, x[:, :S - P]], dim=1)                 # [B, S]（前 P 自述 + 后 S-P 数据）
        h = self.embed(x_full)
        # position: RoPE on Q/K inside Block only (spacetime content encoding removed, reverted to RoPE)
        mask = self.mask[:, :, :S, :S]
        if pad_mask is not None:
            mask = mask & pad_mask[:, None, None, :]        # 结合 padding mask（忽略填充位置）
        g = self.g_init.unsqueeze(0).expand(B, -1, -1)                 # [B,d,d] 记忆矩阵（per-batch：每样本一个，不串）
        t_g = self.t_g_init                                            # [d] 时间快权重（最近性）
        for b in self.blocks:
            h, g, t_g = b(h, g, t_g, mask, pad_mask)        # segment attention + g 纠偏 + FFN
        self.last_h_final = h                                # 探针：FFN 过后的最终表征（生成）
        logits = self.head(self.norm(h)).view(B, S, K_HEAD, 256)   # [B,S,K,256] K 步前看
        # 更新自述纸（detach，只走前向）：模型对数据部分的"下一字节"预测 = 它自己说的话
        with torch.no_grad():
            if S > P:
                pred = logits[:, P - 1:S - 1, 0, :].argmax(-1)   # [B, S-P] 模型对数据的下一字节选择
                n = min(self.paper_len, S - P)
                self.paper[:n] = pred[0, :n].detach()            # 取第一条样本的前 n 位作为新的自述
        return logits, g, t_g

    @torch.no_grad()
    def generate(self, prompt, length=200, dev="cuda", temp=0.8, top_p=0.95, freq_pen=0.5):
        """decode 端采样（工作空间）：temperature + top-p + 全局频率惩罚，破重复。"""
        out = list(prompt.tolist())
        for _ in range(length):
            x = torch.tensor(out[-(self.seq - self.paper_len):], dtype=torch.long, device=dev).unsqueeze(0)
            logits, g, t_g = self.forward(x)
            self.g_init.data = g.mean(dim=0)                 # [B,d,d] 合并回 [d,d] 持久状态
            self.t_g_init.data = t_g                          # t_g 时间快权重写回
            logits = logits[0, -1, 0].clone()                 # [K,256] -> 取第 1 个头（单步生成；多步投机解码后续加）
            # 全局频率惩罚：按整段出现次数压 logits
            for tok in set(out):
                logits[tok] -= freq_pen * out.count(tok)
            # temperature
            logits = logits / temp
            # top-p (nucleus) 采样
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=0)
            keep = cum <= top_p
            keep[0] = True  # 至少保留一个
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = sorted_probs[keep]
            filtered = filtered / (filtered.sum() + 1e-12)
            out.append(int(torch.multinomial(filtered, 1).item()))
        return torch.tensor(out, dtype=torch.long)


RL_LAMBDA = 0.05   # 词级 RL 奖励的权重（REINFORCE，联合 CE 的 RL 版）


def word_rl_loss(logits, y):
    """GRPO-lite：组内相对优势（组 = 一个 batch 的所有词）。
    reward = 正确字节数 / 词长（partial credit）。
    advantage_i = (r_i - mean(r)) / (std(r) + eps)，比 REINFORCE 全局 baseline 方差低。
    返回 (loss, 平均奖励, 词级全对准确率)。"""
    B, S, V = logits.shape
    logp = F.log_softmax(logits, dim=-1)                       # [B,S,V]
    pred = logits.argmax(dim=-1)                               # [B,S]
    boundary = ((y == 32) | (y == 10) | (y == 46) | (y == 44) | (y == 33) |
                (y == 63) | (y == 59) | (y == 58))             # 空格/换行/.,!?;:
    word_logps = []
    rewards = []
    full = []
    for b in range(B):
        wlp = 0.0
        ncorr = 0
        wlen = 0
        for s in range(S):
            byte = int(y[b, s])
            wlp = wlp + logp[b, s, byte]
            if int(pred[b, s]) == byte:
                ncorr += 1
            wlen += 1
            if boundary[b, s]:
                rewards.append(ncorr / max(wlen, 1))           # partial credit
                word_logps.append(wlp)
                full.append(1.0 if ncorr == wlen else 0.0)     # 全对（wacc 用）
                wlp = 0.0
                ncorr = 0
                wlen = 0
    if len(rewards) == 0:
        return 0.0, 0.0, 0.0
    r = torch.tensor(rewards, device=logits.device)
    wlp = torch.stack(word_logps)
    adv = (r - r.mean()) / (r.std() + 1e-8)                    # 组内相对优势（GRPO）
    loss = -(wlp * adv).mean()
    return loss, r.mean().item(), sum(full) / len(full)


def main():
    dev = "cuda"
    data = load_bytes(DATA)
    S, bs = 512, 8
    m = Model(d=2880, c=64, M=256, n_layers=6, H=16, seq=S).to(dev)
    n_params = sum(p.numel() for p in m.parameters())
    peak_lr = 1e-4
    warmup = 1000
    opt = torch.optim.AdamW(m.parameters(), lr=peak_lr, weight_decay=0.1)
    n = len(data)
    steps = 60000
    os.makedirs(OUTDIR, exist_ok=True)
    start_step = 0
    # resume from the newest checkpoint whose state_dict exactly matches this architecture
    model_keys = set(m.state_dict().keys())
    best_path, best_mtime = None, -1.0
    for fname in os.listdir(OUTDIR):
        if not (fname.startswith("ckpt_") and fname.endswith(".pt")):
            continue
        p = os.path.join(OUTDIR, fname)
        try:
            sd = torch.load(p, map_location="cpu")
            if set(sd.get("model", {}).keys()) == model_keys:
                mt = os.path.getmtime(p)
                if mt > best_mtime:
                    best_mtime, best_path = mt, p
        except Exception:
            continue
    if best_path is not None:
        sd = torch.load(best_path, map_location=dev)
        m.load_state_dict(sd["model"])
        start_step = int(sd.get("step", 0))
        print(f"resumed {best_path} step={start_step}", flush=True)
    print(f"params={n_params/1e6:.1f}M  data={len(data)} bytes  seq={S} bs={bs} start={start_step}", flush=True)
    ema_ce = 0.0   # CE 滑动平均（自适应 LR 用）
    rl_baseline = 0.0   # 词级 RL 的 baseline（词准确率滑动平均）
    ce_prev = 0.0
    lr = peak_lr
    t0 = time.time()
    for step in range(start_step, steps):
        if start_step > 0:
            lr = 1e-4  # 续训：词级 RL 用较大 LR（让奖励信号够强）
        elif step < warmup:
            lr = peak_lr * (step + 1) / warmup
        else:
            # 自适应 LR：CE 相对滑动平均的位置决定（平台期提 LR 逃离，下降期微降）
            if ce_prev > 0.0 and ema_ce > 0.0:
                if ce_prev < ema_ce * 0.999:
                    lr = max(lr * 0.999, 1e-5)       # 在降：温和降
                else:
                    lr = min(lr * 1.003, 1.5e-4)      # 平台：温和提，上限 1.5e-4（避免忽高忽低）
        for g in opt.param_groups:
            g["lr"] = lr
        # 整个数据集从前往后线性走，25% 重叠（stride = 3/4 span）
        span = S * bs
        stride = span * 3 // 4
        base = (step * stride) % (n - span - 1)
        st = [base + i * S for i in range(bs)]
        x = torch.stack([data[i:i + S] for i in st]).to(dev).long()
        y = torch.stack([data[i + 1:i + S + 1] for i in st]).to(dev).long()
        g_prev = m.g_init.clone()                             # 输入的 g（持久状态）
        logits, g, t_g = m(x)                               # forward 用持久状态 g_init
        g_cos = F.cosine_similarity(g_prev.reshape(1, -1), g[:, -1].mean(0).reshape(1, -1), dim=-1).mean().item()  # g 变化量
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        ce_prev = ce.item()
        if step >= warmup:
            ema_ce = ce.item() if ema_ce == 0.0 else 0.99 * ema_ce + 0.01 * ce.item()
        p = F.softmax(logits, dim=-1)                        # [B,S,V]
        ent = -(p * (p + 1e-9).log()).sum(-1) / math.log(256)  # [B,S] normalized entropy in [0,1]
        loss = ce - m.ent_alpha * ent.mean()  # 纯 CE（正交限制已取消，预测误差通过 content→h 参与 CE）
        rl, r_avg, wacc = word_rl_loss(logits, y)                       # 词级 GRPO-lite（组内相对优势）
        loss = loss + RL_LAMBDA * rl                                    # 联合 CE：CE 底 + RL 词级奖励
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: ce={ce.item():.3f}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        m.g_init.data = g[:, -1].mean(0).detach()   # 累乘已累积全部 token，取最后位置 = 完整全局帧
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            gb = torch.cuda.max_memory_allocated() / 1e9
            read_ent = m.blocks[-1].last_read_ent   # 最后一层 read 注意力熵
            c2 = m.blocks[-1].pc.c_scale2.item()    # content 直接注入强度（最后一层）
            print(f"step {step+1}/{steps} ce={ce.item():.3f} ent={ent.mean().item():.4f} readent={read_ent:.3f} gcos={g_cos:.4f} expl={m.total_expl_err.item():.4f} c2={c2:.3f} wacc={wacc:.3f} lr={lr:.2e} {el/(step+1)*1000:.0f}ms/step mem={gb:.1f}GB", flush=True)
        if (step + 1) % 10000 == 0:
            torch.save({"model": m.state_dict(), "step": step + 1}, os.path.join(OUTDIR, f"ckpt_{step+1}.pt"))
            print(f"saved ckpt_{step+1}.pt", flush=True)

    torch.save({"model": m.state_dict(), "step": steps}, os.path.join(OUTDIR, "final.pt"))
    ce = 0.0
    with torch.no_grad():
        for _ in range(100):
            st = torch.randint(0, n - S - 1, (2,))
            x = torch.stack([data[i:i + S] for i in st]).to(dev).long()
            y = torch.stack([data[i + 1:i + S + 1] for i in st]).to(dev).long()
            ce += F.cross_entropy(m(x)[0].reshape(-1, 256), y.reshape(-1)).item()
    print(f"PPL={math.exp(ce / 100):.3f}", flush=True)

    prompt = data[1000:1024]
    gen = m.generate(prompt, length=240, dev=dev)
    print("GEN:", flush=True)
    print(bytes(gen.tolist()).decode("utf-8", errors="replace"), flush=True)


if __name__ == "__main__":
    main()
