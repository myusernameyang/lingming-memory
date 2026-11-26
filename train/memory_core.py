# -*- coding: utf-8 -*-
"""灵铭记忆核心：照搬灵元子 FastWeight，但 contents = 数据库本体

灵元子 memory.py 原文：
    q       = sparse_topk(W_coord(h), k)          # 稀疏插值坐标（保权重值）
    raw     = g @ q                                # 快权重稀疏寻址读
    memory  = contents @ raw
    content = content_norm(memory)                 # 召回内容
    gate    = sigmoid((t_raw - gate_bias)/gate_temp)
    w       = normalize((h - h_mean) + content)    # 输入 + 召回
    g       = lam*g + (0.5/S)*gate*(w ⊗ q)
    g       = tanh(g)

本实现的映射（与灵元子的差异，诚实标注）：
  · contents：灵元子里是 [d,d] 可学习语义投影；这里是 [capacity, d] **数据库表本体**，
    每行是一条记忆的向量 w。读就是 contents[cand] @ 查询向量。
  · h：用 bge 语义向量（不训练），单位范数。
  · W_coord：固定随机投影（实测随机已够用，训练反而变差）。
  · h − h_mean：灵元子沿序列位置减共模；单条记忆无序列维，默认不减（center 可开）。
  · content 量级：灵元子对齐 √d（残差流量级）；这里 h 是单位范数，故用单位归一化。

作用域（scope）——2026-11-15 增加：
  每条记忆带一个 scope，召回时只返回「同 scope」或「空 scope(共享)」的条目，
  写/读全部按 scope 隔开。目的：不同 QQ 聊天室不能互相串记忆。
  · scope=""             全局共享（人格、主人偏好这类对谁都能说的话）
  · scope="main"         GUI 主窗口
  · scope="qq:<会话id>"  某个 QQ 聊天室自己的记忆，别的聊天室看不到
  兼容旧库：旧条目没有 scope，加载时统一归到 LEGACY_SCOPE，只有 main 能看到
  （旧库里混着群里的话，不能再让 QQ 聊天室召回）。
"""
import re

import torch
import torch.nn.functional as F


# 中文句末 + 分号 + 换行 + 英文句末 之后切开
_SPLIT_RE = re.compile(r'(?<=[。！？!?；;])|(?<=\n)')
# 顿号/逗号也切（长句内部分句），但只在剩余片段仍过长时才用（见 split_text）
_SOFT_RE = re.compile(r'(?<=[，,、])')


# 智能体框架自己塞的包装文本，不是用户记忆
_HARNESS_NOISE = re.compile(
    r'Your parent agent id is|send_message\(\{|self-contained result|'
    r'does not automatically receive your transcript|sending a message does not end your turn')

LEGACY_SCOPE = 'main'        # 旧条目没有 scope → 归主窗口，QQ 聊天室读不到
SHARED_SCOPE = ''            # 空 scope = 谁都能召回


def norm_scope(scope):
    """scope 规范化：None/空 → 共享；其余裁掉首尾空白。"""
    return str(scope or '').strip()


def _low_value(s):
    """低信息片段：纯 URL / 纯 markdown 标题 / 纯代码围栏 / 纯表格线 / 只有符号 / 框架包装文本"""
    if _HARNESS_NOISE.search(s):
        return True
    if re.fullmatch(r'https?://\S+', s):
        return True
    if re.fullmatch(r'#{1,6}\s*\S*', s):
        return True
    if re.fullmatch(r'`{3,}[\w-]*', s):
        return True
    if re.fullmatch(r'\|[\s:|-]*\|?', s):
        return True
    return not re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s)


def split_text(text, max_len=200, min_len=6):
    """把一条消息切成检索粒度的片段

    规则：
      1. 先按硬标点（。！？；换行）切句
      2. 单句超过 max_len 时，再按软标点（，、）二次切
      3. 仍然超长就硬截断
      4. 去掉过短碎片（< min_len），除非整条都没了
    这样"一大段"会变成若干条独立记忆，召回粒度跟着消息的实际信息量走。
    """
    if not text:
        return []
    segs = []
    for chunk in _SPLIT_RE.split(text):
        chunk = (chunk or '').strip()
        if not chunk:
            continue
        if len(chunk) <= max_len:
            segs.append(chunk)
            continue
        for piece in _SOFT_RE.split(chunk):
            piece = (piece or '').strip()
            if not piece:
                continue
            while len(piece) > max_len:
                segs.append(piece[:max_len])
                piece = piece[max_len:]
            if piece:
                segs.append(piece)
    kept = [s for s in segs if len(s) >= min_len and not _low_value(s)]
    if not kept:
        whole = text.strip()
        kept = [whole[:max_len]] if whole and not _low_value(whole) else []
    return kept


def sparse_topk(coord, k):
    """硬 top-k + 直通：保留 top-k 的权重值（插值坐标），其余置 0"""
    val, idx = coord.topk(k, dim=-1)
    mask = torch.zeros_like(coord)
    mask.scatter_(-1, idx, 1.0)
    return coord * mask


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]{2,}|\d{3,}")


def ident_tokens(text):
    """提标识符：英文/型号/含数字的串（长度 >= 3）。中文不参与倒排。

    中文分词成本高，而且中文靠向量就够；倒排专治向量不敏感的随机串。
    """
    return set(TOKEN_RE.findall(text.lower()))


class LingmingMemory:
    def __init__(self, encode, d=384, c_mult=4, k=8, lam=0.99,
                 capacity=10000, device=None, center=False, write_scale=0.1,
                 gate_bias=0.2, gate_temp=0.05, read_top=3, per_bar=3):
        self.encode = encode
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.d, self.k, self.lam = d, k, lam
        self.C = c_mult * d
        self.capacity = int(capacity)
        self.center = center
        self.write_scale = write_scale
        self.gate_bias = gate_bias
        self.gate_temp = gate_temp
        self.read_top = read_top
        # 桶内限流：query 命中的每个桶里只保留 sim 前 per_bar 条进候选。
        # 候选上限 = k * per_bar，与库大小无关（实测 N=2000 时候选 1539 -> 7，
        # hit@1 保持 1.000；per_bar=0 退回旧行为=全取并集，候选随 N 线性涨）
        self.per_bar = int(per_bar)

        # 坐标投影（固定随机，行归一化 ≈ 保距投影）
        gen = torch.Generator(device="cpu").manual_seed(12345)
        self.W_coord = F.normalize(torch.randn(self.C, d, generator=gen), dim=-1).to(self.dev)

        # g 快权重 [d,C] + 时间 [C]
        self.g = torch.zeros(d, self.C, device=self.dev)
        self.t_g = torch.zeros(self.C, device=self.dev)

        # ★ 数据库本体 [capacity, d]
        self.contents = torch.zeros(self.capacity, d, device=self.dev)
        self.n = 0                    # 已用行数
        self.texts = []
        self.roles = []
        self.slot_of = {}             # mid -> 行号
        self.bars = {}                # mid -> [bar...]
        self.index = {}               # bar -> [mid...]
        self.next_id = 0
        self.total_writes = 0
        self.h_mean = torch.zeros(d, device=self.dev)
        self._seen = 0
        # 分段：一条消息切成多段各自入库，msg_of 记录段→原消息，便于回溯
        self.msg_of = {}              # mid -> 消息号
        self.msg_text = {}            # 消息号 -> 原始完整文本
        self.boost = {}               # mid -> 可学习加权（用后反馈，正=有用）
        self.deleted = set()          # 已删除的 mid（检索时过滤）
        # 作用域索引：mid -> scope 字符串；_scope_ids 把 scope 编码成 int 方便过滤
        self.scopes = {}              # mid -> scope（"" = 共享）
        self._scope_ids = {SHARED_SCOPE: 0}
        self._scope_names = [SHARED_SCOPE]
        # query→memory 关联表：分数必须是 (query, memory) 的函数，不是 (memory) 的。
        # boost 是 (memory) 的标量，所以它必然无效（实测 hit@1 不动）。
        # 这里存历史 query 向量 + 它用到的 mid；检索时找相似历史 query，
        # 把它们用过的 mid 所在的桶并进候选（预测性预取），不是只加 sim。
        self.q_embs = None            # (M, d)
        self.q_mids = []              # M x [mid...]
        self.q_topn = 8               # 取几条最相似的历史 query
        self.q_thresh = 0.75          # 相似度低于此不采纳
        self.q_alpha = 0.0            # 关联表加权强度。默认 0 = 关闭：
        #   实测在合成集上调不起来 —— "query 信息不足"时，模糊 query 与
        #   所有兄弟条目的具体 query 都一样近（cosine 全 > 0.75），加权会把
        #   6 条兄弟一起拉上来，仍然分不出。信息不足不能用回忆补。
        #   留着代码是为了接真实对话（跨会话的偏好/习惯才是它的地盘）。
        # 标识符倒排：token -> {mid}。句向量对随机串（错误码/版本号/符号名）
        # 几乎无区分力（实测 hit@1 只有 0.600），倒排是 1.000 且候选 1 条。
        # 分流规则：query 提得出标识符就走倒排，否则走坐标桶 + per_bar。
        self.inv = {}
        self.ident_route = True       # False = 关掉分流，全部走向量
        self.q_decay = 0.98           # 关联表条目的时间衰减（未实现，留作参数）

    def delete(self, mid):
        """删除一条记忆：标记删除并从倒排索引摘除"""
        try:
            mid = int(mid)
        except Exception:
            return False
        if mid not in self.slot_of or mid in self.deleted:
            return False
        self.deleted.add(mid)
        for b in self.bars.get(mid, ()):
            lst = self.index.get(b)
            if lst and mid in lst:
                lst.remove(mid)
        row = self.slot_of.get(mid)
        if row is not None and row < len(self.texts):
            for tok in ident_tokens(self.texts[row]):
                s = self.inv.get(tok)
                if s:
                    s.discard(mid)
        return True

    # ---------- 作用域 ----------
    def _sid_of(self, scope):
        s = norm_scope(scope)
        sid = self._scope_ids.get(s)
        if sid is None:
            sid = len(self._scope_names)
            self._scope_ids[s] = sid
            self._scope_names.append(s)
        return sid

    def scope_of(self, mid):
        return self.scopes.get(mid, LEGACY_SCOPE)

    def _load_scopes(self, scopes):
        """载入 mid -> scope 映射（旧库没有时为空）"""
        self.scopes = {}
        self._scope_ids = {SHARED_SCOPE: 0}
        self._scope_names = [SHARED_SCOPE]
        if isinstance(scopes, dict):
            for k, v in scopes.items():
                try:
                    mid = int(k)
                except Exception:
                    continue
                s = norm_scope(v)
                self.scopes[mid] = s
                self._sid_of(s)

    def _scope_mask(self, mids, scope):
        """返回哪些 mid 属于本 scope（同 scope 或共享），旧条目按主窗口算。"""
        s = norm_scope(scope)
        sid_q = self._sid_of(s)
        sid_legacy = self._sid_of(LEGACY_SCOPE)
        keep = []
        for m in mids:
            if m in self.scopes:
                sid = self._sid_of(self.scopes[m])
            else:
                sid = sid_legacy
            if sid == 0 or sid == sid_q:
                keep.append(m)
        return keep

    def assign_scope(self, mids, scope):
        """把若干条记忆改到指定 scope（迁移/纠错用）"""
        s = norm_scope(scope)
        n = 0
        for m in mids or []:
            try:
                mid = int(m)
            except Exception:
                continue
            if mid in self.slot_of:
                self.scopes[mid] = s
                self._sid_of(s)
                n += 1
        return n

    def scope_stats(self):
        """各 scope 的条目数，便于核对"谁看得到多少"。"""
        out = {}
        for mid in self.slot_of:
            if mid in self.deleted:
                continue
            s = self.scope_of(mid)
            out[s] = out.get(s, 0) + 1
        return out

    # ---------- 坐标 ----------
    def _coord(self, h):
        coord = self.W_coord @ h
        q = sparse_topk(coord.unsqueeze(0), min(self.k, self.C))[0]
        bars = q.nonzero(as_tuple=False).flatten().tolist()
        return q, bars

    def _candidates(self, bars):
        cand = set()
        for b in bars:
            cand.update(self.index.get(b, ()))
        return cand

    def _qtable_bars(self, h, bars):
        """query→memory 关联表：找相似历史 query，把它们用过的 mid 的桶并进候选。

        这一步是「预测性预取」——预测分的落脚点是**候选集**，不是 sim 排序。
        只加到 sim 上救不了「没进桶」的记忆（这是 boost 失败的根因）。
        """
        if self.q_embs is None or not self.q_mids:
            return bars
        qs = self.q_embs @ h
        k = min(self.q_topn, qs.numel())
        top = qs.topk(k).indices.tolist()
        out = set(bars)
        for i in top:
            if qs[i].item() < self.q_thresh:
                continue
            for mid in self.q_mids[i]:
                out |= set(self.bars.get(mid, ()))
        return out

    def _qtable_boost(self, h, mids):
        """关联表加权：相似历史 query 用过的 mid，按 query 相似度给 sim 加分。

        只做预取不够 —— 预取只把答案放进候选集（它本来就在），
        排序仍由原始 sim 决定，而模糊 query 的 sim 分不出兄弟条目。
        必须有这一路把正确条目顶上去。
        """
        if self.q_embs is None or not self.q_mids:
            return None
        qs = self.q_embs @ h
        adj = {}
        for i, q in enumerate(qs.tolist()):
            if q < self.q_thresh:
                continue
            for mid in self.q_mids[i]:
                if q > adj.get(mid, -1e9):
                    adj[mid] = q
        if not adj:
            return None
        return torch.tensor([adj.get(m, 0.0) for m in mids],
                            device=h.device, dtype=h.dtype)

    def add_query_feedback(self, h, mids):
        """记录：这条 query 向量 → 它实际用到的 mid 列表（供下次相似 query 预取）"""
        mids = [m for m in mids if m in self.slot_of and m not in self.deleted]
        if not mids:
            return
        hv = h.detach().clone().reshape(1, -1)
        self.q_embs = hv if self.q_embs is None else torch.cat([self.q_embs, hv], 0)
        self.q_mids.append(list(mids))

    def _cap_candidates(self, mids, sim, bars):
        """桶内限流：query 命中的每个桶里只保留 sim 前 per_bar 条。

        返回 (mids, sim)，顺序一致。per_bar<=0 时原样返回。
        动机：桶并集在 N 大时几乎覆盖全库（N=2000 时候选 1539），检索退化成
        全表扫描。限流后候选上限 = len(bars) * per_bar，与 N 无关。
        """
        m = self.per_bar
        if m <= 0 or len(mids) <= 2 * m:
            return mids, sim
        bars_set = set(bars)
        sim_l = sim.tolist()
        buckets = {}
        for i, mid in enumerate(mids):
            for b in set(self.bars.get(mid, ())) & bars_set:
                buckets.setdefault(b, []).append((sim_l[i], mid))
        keep = set()
        for lst in buckets.values():
            lst.sort(reverse=True)
            for _, mid in lst[:m]:
                keep.add(mid)
        if not keep:
            return mids, sim
        keep = sorted(keep)
        pos = {mid: i for i, mid in enumerate(mids)}
        idx = torch.tensor([pos[x] for x in keep], device=sim.device,
                           dtype=torch.long)
        return keep, sim[idx]

    def _read(self, h, bars, scope=""):
        """从数据库读回与 h 最相关的若干条，作为"召回内容"（无候选则零向量）"""
        cand = self._candidates(bars)
        if not cand:
            return torch.zeros(self.d, device=self.dev)
        mids = [m for m in sorted(cand) if m not in self.deleted]
        mids = self._scope_mask(mids, scope)
        if not mids:
            return torch.zeros(self.d, device=self.dev)
        rows = torch.tensor([self.slot_of[m] for m in mids], device=self.dev)
        sim = self.contents[rows] @ h
        mids2, sim = self._cap_candidates(mids, sim, bars)
        if len(mids2) != len(mids):
            rows = torch.tensor([self.slot_of[m] for m in mids2], device=self.dev)
        take = min(self.read_top, len(mids2))
        top = sim.argsort(descending=True)[:take]
        sel = rows[top]
        return F.normalize(self.contents[sel].mean(0), dim=-1)

    # ---------- 写 ----------
    def add(self, text, role="user", scope=""):
        """一条消息切成多段，每段独立入库（粒度细，召回才准）"""
        if not text:
            return -1
        scope = norm_scope(scope)
        msg_id = self.total_writes
        self.msg_text[msg_id] = text
        first = -1
        for seg in split_text(text):
            mid = self._add_one(seg, role, msg_id, scope)
            if first < 0:
                first = mid
        return first

    def _add_one(self, text, role, msg_id, scope=""):
        h = self.encode(text)
        hc = h - self.h_mean if (self.center and self._seen > 0) else h
        q, bars = self._coord(h)

        # 读：数据库召回（只召回同 scope / 共享，避免别的聊天室内容污染写入）
        content = self._read(h, bars, scope=scope)

        # 门控度量：t_g 是坐标空间上的累计寄存器（按 lam 衰减），t_g·q 衡量当前 h 点亮的
        # 那些坐标区域此前被重复写入的强度。q 由 h 决定，时间只体现为累积顺序，
        # 所以这是"内容×累计"的度量，不是时间近邻门控。
        t_raw = float(self.t_g @ q)
        gate = torch.sigmoid(torch.tensor((t_raw - self.gate_bias) / self.gate_temp, device=self.dev))

        # 写：输入 + 召回，外积进 g
        w = F.normalize(hc + content, dim=-1)
        self.g = torch.tanh(self.lam * self.g + self.write_scale * gate * torch.outer(w, q))
        self.t_g = self.lam * self.t_g + q

        # 写入数据库
        mid = self.next_id
        self.next_id += 1
        if self.n >= self.contents.shape[0]:
            grow = max(self.contents.shape[0], 1024)
            self.contents = torch.cat(
                [self.contents, torch.zeros(grow, self.d, device=self.dev)], dim=0)
        row = self.n
        self.n += 1
        # 库里存 h（纯语义向量）。w 只用于外积写 g，不进数据库：
        # 实测存 w 会让 end-to-end 从 95.5% 掉到 57.5%（w 混入了召回噪声，偏离自身语义）
        self.contents[row] = h
        self.texts.append(text)
        self.roles.append(role)
        self.slot_of[mid] = row
        self.bars[mid] = bars
        self.msg_of[mid] = msg_id
        self.scopes[mid] = scope
        self._sid_of(scope)
        for b in bars:
            self.index.setdefault(b, []).append(mid)
        for tok in ident_tokens(text):
            self.inv.setdefault(tok, set()).add(mid)
        self.total_writes += 1

        self.h_mean = (self.h_mean * self._seen + h) / (self._seen + 1)
        self._seen += 1
        return mid

    # ---------- 查 ----------
    @torch.no_grad()
    def retrieve(self, query, topK=3, scope=""):
        h = self.encode(query)
        # 分流：query 提得出标识符 -> 走倒排（精确、候选极少）
        bars = []
        inv_mids = set()
        if self.ident_route:
            for tok in ident_tokens(query):
                inv_mids |= self.inv.get(tok, set())
            inv_mids = {m for m in inv_mids if m not in self.deleted}
        if inv_mids:
            cand = sorted(inv_mids)
        else:
            _, bars = self._coord(h)
            # 预测性预取：相似历史 query 用过的记忆，其桶并入候选
            bars = self._qtable_bars(h, bars)
            cand = self._candidates(bars)
        if not cand:
            return []
        mids = [m for m in sorted(cand) if m not in self.deleted]
        if not mids:
            return []
        mids = self._scope_mask(mids, scope)
        if not mids:
            return []
        rows = torch.tensor([self.slot_of[m] for m in mids], device=self.dev)
        sim = self.contents[rows] @ h
        # 用后反馈加权：被用过的记忆下次更容易被召回（轻量在线学习）
        if self.boost:
            adj = torch.tensor([self.boost.get(m, 0.0) for m in mids], device=self.dev)
            sim = sim + adj
        # 关联表加权：相似历史 query 用过的 mid，按 query 相似度顶上去
        qb = self._qtable_boost(h, mids)
        if qb is not None:
            sim = sim + self.q_alpha * qb
        # 桶内限流：候选上限 = k * per_bar，不随库大小线性涨
        # 走倒排时候选本来就极少，这一步会直接跳过（len <= 2*per_bar）
        mids, sim = self._cap_candidates(mids, sim, bars if not inv_mids else [])
        rows = torch.tensor([self.slot_of[m] for m in mids], device=self.dev)
        order = sim.argsort(descending=True)[:topK]
        out = []
        for pos in order.tolist():
            mid = mids[pos]
            row = rows[pos].item()
            out.append({
                "mid": mid,
                "text": self.texts[row],
                "role": self.roles[row],
                "scope": self.scope_of(mid),
                "score": float(sim[pos]),     # cosine 细排分，供上层做相关性闸
                "msg": self.msg_of.get(mid, -1),   # 所属消息号（分段回溯）
            })
        return out

    def feedback(self, pairs, lr=0.05, clip=0.3):
        """用后反馈：pairs = [(mid, used)]，used 表示这条记忆在回复里被用到了"""
        for mid, used in pairs:
            b = self.boost.get(mid, 0.0)
            b += lr * (1.0 if used else -0.3)
            self.boost[mid] = max(-clip, min(clip, b))

    def status(self):
        return {
            "total_writes": self.total_writes,
            "n_mem": len(self.slot_of),
            "n_bars": len(self.index),
            "capacity": self.capacity,
            "C": self.C,
            "k": self.k,
            "scopes": self.scope_stats(),
        }
