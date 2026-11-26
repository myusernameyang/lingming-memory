<div align="center">

# 灵铭记忆 · Lingming Memory

**坐标寻址的向量记忆 —— 不喂长上下文，用稀疏坐标代替余弦全表扫描**

<p>
  <img src="https://img.shields.io/badge/DSH-Plugin-blue.svg" alt="DSH Plugin"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

[它是什么](#它是什么) · [检索流程](#检索流程) · [安装](#安装) · [使用](#使用) · [配置](#配置) · [接口](#接口) · [仓库结构](#仓库结构) · [已知局限](#已知局限)

</div>

---

## 它是什么

大模型聊得越久，上下文越长、越贵、越慢。**灵铭记忆**的做法：不喂完整历史，而是把每句对话用本地小模型编成一个语义向量，再用一个**固定的稀疏随机投影**把它压成 `k` 个坐标桶（`q`），按桶建倒排索引。查询时算出同一个 `q`，取那几个桶的并集做候选，再在这个小候选里按余弦细排。

一句话：**用"落点"代替"逐个比"**——相似输入落进同一批桶，候选集天然就小。

实际用的组件：

| 部件 | 现在是什么 |
|---|---|
| 语义编码器 | `BAAI/bge-small-zh-v1.5`（384 维，可换） |
| 坐标投影 | 固定随机投影（种子 12345，行归一化 ≈ 保距投影），**不训练** |
| 桶数 | `C = cMult × d`，默认 4 × 384 = 1536 |
| 稀疏度 | 每条记忆点亮 `k = 8` 个桶 |
| 候选 | 桶并集 → **桶内限流 `perBar = 3`** |

> 仓库里的 `train_coord.py` 和 `coord.pt` 是**历史路径**：早期试过训练坐标投影，
> 当前服务**不加载** `coord.pt`，用的是固定随机投影。留着是为了以后想试对比时能用。

## 检索流程

```
retrieve(query)
  ├─ ① query 提得出标识符（英文/型号/含数字，长度 ≥ 3）
  │     → 倒排精确命中（ERR-7f3a9c、rebar、535 这类）
  │       句向量对随机串几乎无区分力：实测 hit@1 仅 0.600，倒排 1.000
  └─ ② 提不出标识符
        → 坐标桶并集 → 桶内限流 perBar → 余弦细排
```

两个关键设计：

### 桶内限流（`perBar`）

坐标桶并集在库变大时会几乎覆盖全库——N=2000 时候选 1539 条，**检索退化成全表扫描**。

于是改成：query 命中的每个桶只保留余弦前 `perBar` 条进候选。候选上限 = `k × perBar`，**与库大小无关**。

| N | hit@1 | 不限流候选 | `perBar=3` 候选 |
|---|---|---|---|
| 60 | 1.000 | 56 | 5 |
| 250 | 1.000 | 233 | 6 |
| 1000 | 1.000 | 807 | 7 |
| 2000 | 1.000 | 1539 | 7 |

### 标识符分流

排障类记忆里大量是错误码、驱动版本、寄存器名、符号名。这类 token 走倒排，不走向量。

**两路必须分流，不能并集**——并集会让候选重新膨胀回库大小（实测混合库候选 299.7/240 条）。

混合库实测（120 条语义事实 + 120 条标识符事实）：

| 策略 | 语义 hit@1 | 标识符 hit@1 | 总 hit@1 | 平均候选 |
|---|---|---|---|---|
| 全走向量 | 1.000 | 0.750 | 0.875 | 19.2 |
| **分流** | 1.000 | **1.000** | **1.000** | **9.0** |

## 安装

```bash
dsh plugin --profile desktop add github:myusernameyang/lingming-memory
```

重启 DSH，聊天框会出现「灵铭记忆」开关。插件会自动检测 Python 环境、`pip install -r train/requirements.txt`、拉起本地服务。

## 使用

```
/lingming           开关（开/关记忆拦截）
/lingming-status    看已存多少条记忆 / 坐标桶数 / 编码器
/lingming-context   看本轮拼进上下文的记忆
/lingming-config    打开配置页
/lingming-save      保存记忆库到磁盘
```

开启后：

1. 你和 agent 的**每句话都会被写进记忆**（用户 + 助手，带角色标签）；
2. 每轮 `agent/pre-step` 会自动检索相关历史，拼进本轮上下文；
3. 召回结果会记录到注入日志，可用 `/lingming-context` 或 `GET /injections` 查看。

## 配置

`/lingming-config` 打开的页面里可调（也可 `POST /api/config`）：

| 键 | 默认 | 说明 |
|---|---|---|
| `topK` | 3 | 每轮拼进上下文几条记忆 |
| `cMult` | 4 | 桶数倍数，`C = cMult × 384` |
| `k` | 8 | 每条记忆点亮几个桶 |
| `capacity` | 10000 | 最大条数（超了自动扩容） |
| `writeScale` | 0.1 | 外积写 `g` 的步长 |
| `gateBias` / `gateTemp` | 0.2 / 0.05 | 写门控的偏置与温度 |
| `readTop` | 3 | 召回时取前几条求平均作为 content |
| **`perBar`** | **3** | 桶内限流，`0` = 不限流 |
| `center` | false | 写入前减运行均值 |
| `contextMode` | augment | `augment` / `window` / `replace` |
| `keepMessages` | 6 | `window` 模式保留几条最近消息 |

改 `REBUILD_KEYS` 里的项会重建记忆对象（不丢库，但要重新载入）。

## 接口

服务默认跑在本地，HTTP JSON：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/write` | 存一条记忆（自动分段） |
| `POST` | `/retrieve` | 检索（`query`、`topK`、`scope`） |
| `POST` | `/score` | 算两段文本的余弦（冲突判定用） |
| `GET` | `/msg` | 按 mid 取所属父消息原文 |
| `GET` | `/dump` | 导出全部记忆 |
| `POST` | `/delete` | 删除一条 |
| `POST` | `/feedback` | 用后反馈（`mid` + `used`） |
| `POST` | `/repair` | 修复索引 |
| `GET` | `/injections` · `POST /log_injection` | 注入日志 |
| `GET` | `/scope` | 各作用域条目数 |
| `GET` | `/status` | 状态（条数 / 桶数 / 编码器） |
| `GET`·`POST` | `/api/config` | 读写配置 |
| `POST` | `/save` · `/reset` | 存盘 / 清空 |
| `GET` | `/config` | 配置页 HTML |

## 仓库结构

```
index.js            Host 半：拉起服务 + agent/created 挂钩子 + agent/pre-step 写记忆/注入召回 + 5 个命令
client.js           Client 半：聊天框开关
memory.js           记忆客户端（fetch /write + /retrieve）
train/
  embed_server.py      HTTP 服务（Flask，上面那张接口表）
  memory_core.py       核心：坐标寻址 + 桶内限流 + 标识符倒排分流
  rule_memory.py      规则记忆（rules.jsonl）
  bench.py            召回基准（qmode=old/new）
  train_coord.py      坐标投影训练（历史路径，当前不加载）
  README.md           目录索引 + 实验脚本清单
cordis.patch.yml / locale / icon / package.json
最终方案-20261126.md   实验记录：诊断、证否、落地、验证
```

## 已知局限

**合成测试集测不出来的东西**（当前主要风险）：

1. **模糊提问**——“那个配置呢”这类。查询本身信息不足，实测用“相似历史 query 加权”也救不了：
   模糊查询与所有兄弟条目的具体查询余弦全在 0.75 以上，加权会把它们一起拉上来。
   **信息不足不能用回忆补。**
2. **重复提问**——用户第二次说同一件事。这是记忆失败最硬的信号，也是当前缺的基线。
3. **记重 / 记反 / 该忘没忘**——合成集里不存在。

**参数是猜的**：`perBar=3` 在合成集上无损，但真实记录的同桶相似度更高，可能需要 6 或自适应。没有真实数据前，这个值没有依据。

## 实验记录

所有机制改动都有可重跑的对照实验，见 [`最终方案-20261126.md`](最终方案-20261126.md)。

| 结论 | 状态 |
|---|---|
| 桶内限流 `perBar=3`：N=1000 候选 807 → 22，hit@1 不变 | ✅ 已落地 |
| 标识符倒排分流：总 hit@1 0.875 → 1.000，候选减半 | ✅ 已落地 |
| `hit@1=0.167` 是测试集 bug（查询缺属性词），实际 0.967 | ✅ 已修 |
| 容量曲线：hit@1 不随 N 掉，但候选 ≈ N | ✅ 已测 |
| 用 `g` 做桶级粗筛 | ❌ 证否，候选反而更大 |
| query→memory 关联表救模糊查询 | ❌ 证否，信息不足不能用回忆补 |

实验室脚本在 `train/_*.py`，索引见 [`train/README.md`](train/README.md)。

## 📄 License

[MIT](LICENSE)

---

<div align="center">
  <b>记忆 = 坐标寻址的先验习惯</b>
</div>
