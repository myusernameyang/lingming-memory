# train/ 目录索引

```
memory_core.py     核心：坐标寻址（稀疏 top-k 桶）+ 桶内限流 + 标识符倒排分流
embed_server.py    HTTP 服务（/write /retrieve /score /msg /status /save）
train_coord.py     坐标投影训练（相似对 → 对比损失 → coord.pt）
rule_memory.py     规则记忆（rules.jsonl）
```

## 实验脚本（`_` 开头，一次性诊断，不在主链路）

这些脚本当天跑完就有了结论，**留着是为了能重跑验证**，不是日常运行代码。

| 脚本 | 干什么 | 结论 |
|---|---|---|
| `_diag_cand.py` | 量候选召回率 vs hit@1 | 候选 91.7%，不是瓶颈 |
| `_diag_cap.py` | 旧/新 query 天花板对照 | 0.167 → 0.967（测试集 bug） |
| `_exp_prf.py` | 双 query 并集（伪相关反馈） | 0.967 → 1.000，未合入 |
| `_exp_capacity.py` | C5 容量曲线（N 60→2000） | hit@1 不掉，候选 ≈ N |
| `_exp_e5.py` | 用 `g` 做桶级粗筛 | ❌ 失败，候选反而更大 |
| `_exp_bucketcap.py` | 桶内限流（per-bucket top-M） | ✅ 候选 -99.5%，hit 不变 |
| `_verify_cap.py` | 验证已落地的 `per_bar` | ✅ 807 → 22 |
| `_exp_qtable.py` | query→memory 关联表 | ❌ 救不了"query 缺信息" |
| `_exp_inverted.py` | 倒排 vs 向量（标识符类查询） | ✅ 0.600 → 1.000 |
| `_exp_router.py` | 分流（混合库） | ✅ 0.875 → 1.000，候选减半 |

完整结论文档见仓库根目录 `最终方案-20261126.md`。

## 跑实验的环境

Windows 上 torch 会和 MKL 抢 OpenMP，必须先设：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:PYTHONIOENCODING="utf-8"
python _exp_capacity.py
```

不设 `KMP_DUPLICATE_LIB_OK` 会报 `OMP: Error #15`；不设 `PYTHONIOENCODING` 中文输出会乱码。

## 注意：这些脚本的基准集是合成的

合成集里的查询**一次就唯一可解**，测不出真实使用中的：

- 模糊提问（"那个配置呢"）
- 重复提问（记忆失败最硬的信号）
- 记重 / 记反 / 该忘没忘

真实数据要靠各聊天室记录（见 `任务-各群使用感受收集.md`，该文件不入库）。
