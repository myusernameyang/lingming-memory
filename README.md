# 灵铭记忆（Lingming Memory）

> 用「模型打分的坐标检索」替代长上下文的记忆机制 —— DSH 插件

大模型聊得越久，上下文越长、越贵、越慢。灵铭记忆的思路：**不喂完整长上下文，只喂"检索到的相关历史 + 当前消息"**。

它的 embedding **不是固定规则、不是预训练模型**，而是**模型自己打分学出来的坐标**——用 API 打分做 RL，训练一个坐标网络，让它学会"语义相近 → 坐标相近"。

---

## 安装（傻瓜化，一条命令）

```bash
dsh plugin --profile desktop add github:myusernameyang/lingming-memory
```

重启 DSH，聊天框出现「灵铭记忆」开关。插件会**自动检测 Python 环境**（缺 torch/flask 等会自动 pip install），然后**自动拉起坐标 embed 服务**，点开关即切记忆机制（或输入 `/lingming`）。

## 初始化 + 实时微调（不用跑 Python 脚本）

在 Plugin Manager 配置界面填好 **apiKey**（DeepSeek API key），然后：

1. 聊天框点「**初始化坐标**」按钮 → 一键用 API 打分生成可用的坐标网络；
2. 之后直接对话——**每次对话自动做实时 RL，微调坐标**（模型边用边学）。

`train_coord.py` 是给想离线批量训练的人用的（`pip install -r requirements.txt` + 喂自己的 data.json），但日常使用**不碰 Python**：初始化、微调都走 HTTP，插件自动处理。

## 超参数（Plugin Manager 配置界面里可调）

| 参数 | 默认 | 说明 |
|---|---|---|
| `topK` | 3 | 每次检索返回的相关历史条数 |
| `embedServerUrl` | http://127.0.0.1:9000 | 坐标 embed 服务地址（根地址） |
| `apiKey` | （空） | DeepSeek API key（初始化 + 实时 RL 打分用） |

---

## 原理

```
对话历史 ──> 每条经「坐标网络」打分成坐标 k，写进向量库
新消息 ──> 同样打分成坐标 k ──> 余弦相似度 ──> 取 top-k 相关历史
        └─> 只把「检索到的相关历史 + 当前消息」喂给模型（不喂完整长上下文）
```

- **坐标打分**：`T_net`(文本) → (r,θ,φ) 连续坐标 → `W_expand` → k。RL 训练让语义相近的文本坐标相近。
- **拦截点**：`agent/pre-step` 瀑布监听器改写 messages。

---

## 仓库结构

```
index.js        Host 半：自动拉起 embed 服务 + agent/pre-step 拦截 + /lingming 开关命令
client.js       Client 半：聊天框开关
memory.js       记忆库（纯坐标，无兜底）
train/
  train_coord.py   RL + API 打分训练坐标网络
  embed_server.py  坐标 embed 服务（插件自动拉起）
  data.json         演示数据
  requirements.txt  Python 依赖
cordis.patch.yml / locale / icon
```

---

## 当前边界（诚实）

- 坐标训练是**无监督的**（API 打分，不依赖人工标注），demo 上 83% 命中，不是 100%；
- 坐标网络由使用者用自己的数据 + API key 训练，本仓库**只提供工具和验证，不提供训练好的模型**；
- 记忆在进程内（重启清空），后续可持久化。

## 关联

- 灵元子（Lingyuanzi）：自研的模型级隐式记忆架构（连续坐标场快权重），本插件的坐标打分是它"关键词解码 → 接检索"的轻量落地。
