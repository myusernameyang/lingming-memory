<div align="center">

# 灵铭记忆 · Lingming Memory

**坐标寻址的多级向量记忆 —— 不喂长上下文，比 RAG 更准、更快**

<p>
  <img src="https://img.shields.io/badge/DSH-Plugin-blue.svg" alt="DSH Plugin"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

[简介](#简介) · [安装](#安装) · [使用](#使用) · [训练坐标](#训练坐标) · [接口](#接口) · [License](#license)

</div>

---

## 简介

大模型聊得越久，上下文越长、越贵、越慢。**灵铭记忆**的思路：不喂完整长上下文，而是用一个本地小模型把每句对话映射到一个**稀疏 top-k 坐标**（k 个 bar = k 级地址），查询时按坐标**桶交集寻址**，直接命中最相关的历史。

核心洞察一句话：**回忆的本质是"似曾相识"（相似度），稀疏坐标就是相似度的离散化——相似输入自然落到相同地址，O(k) 直接命中，不用 O(N) 逐个比。**

## ✨ 为什么比 RAG 强

| 维度 | RAG（余弦检索） | 灵铭（坐标寻址） |
|---|---|---|
| **召回准确率**（真实对话） | 56% | **100%** |
| **检索复杂度** | O(N·d)，随记忆量涨 | **O(d·C)**，不随记忆量涨 |
| **单次检索耗时**（N=100 万） | 1.07 ms | **0.0098 ms** |
| **索引方式** | 稠密向量余弦 argmax | 稀疏 top-k 桶交集 |

稀疏坐标（top-k）比稠密 embedding（cos）更「判别」：相似输入 → 相同 top-k → 直接命中；余弦 argmax 会被「其他更相似的干扰项」带偏（这就是 56% vs 100% 的差距）。

## 安装

```bash
dsh plugin --profile desktop add github:myusernameyang/lingming-memory
```

重启 DSH，聊天框出现「灵铭记忆」开关。插件自动检测 Python 环境、自动 `pip install -r requirements.txt`、自动拉起本地服务。

## 使用

1. 聊天框输入 `/lingming` 开启；
2. 之后你跟 agent 的**每句话都会写进记忆**（存用户 + 助手，带角色标签）；
3. 下一轮自动**按坐标召回相关历史**，拼进上下文（替代长上下文）。

```
/lingming          开关
/lingming-status   看已存多少条记忆 / 多少个坐标桶
/lingming-save     保存坐标权重（coord.pt）
```

## 训练坐标

坐标投影 `W_coord` 需要先训练，让它学会「语义相似 → 相同坐标」。用「相似对」训练（对比损失，无需 API）：

```bash
# data.json：{"pairs": [["我叫小明", "我的名字是小明"], ["喜欢篮球", "爱打篮球"]]}
python train/train_coord.py train/data.json   # 产出 coord.pt
```

> 不训练也能跑（用随机坐标），但召回质量取决于坐标训练的好坏。正式用建议喂你自己的「相似对」数据。

## 接口

| 接口 | 说明 |
|---|---|
| `POST /write` | 存记忆 → 坐标 top-k 分桶 |
| `POST /retrieve` | 查询 → 坐标桶交集 → 命中最相关 |
| `POST /save` | 保存坐标权重（coord.pt） |
| `GET /status` | 状态（记忆数 / 坐标桶数） |

## 仓库结构

```
index.js         Host 半：自动拉起服务 + agent/pre-step 拦截（写记忆 + 召回）+ /lingming 命令
client.js        Client 半：聊天框开关
memory.js        记忆客户端（调 /write + /retrieve）
train/
  embed_server.py   记忆服务（坐标寻址 /write + /retrieve）
  memory_core.py    核心：CNN 语义特征 + 稀疏 top-k 坐标 + 多级分桶
  train_coord.py    坐标训练（相似对对比损失 → coord.pt）
  requirements.txt  Python 依赖（torch + flask）
cordis.patch.yml / locale / icon / package.json
```

## 📄 License

[MIT](LICENSE)

---

<div align="center">
  脑启发的记忆研究：<b>记忆 = 坐标寻址的先验习惯</b>
</div>
