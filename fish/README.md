# 大肥鱼 · QQ 养鱼

用灵铭记忆（坐标寻址）给一条 QQ 鱼当"脑子"：记主人、记自己、越聊越成精。

## 架构

```
QQ 消息 → NoneBot2(bot.py) → 写/查 灵铭记忆(embed_server) → DeepSeek(鱼人设) → 回复
```

## 三步跑起来

### 1. 起灵铭记忆服务

```bash
cd ../train
python embed_server.py          # 起在 http://127.0.0.1:9000
python train_coord.py data.json # （可选）先训坐标，不训用随机坐标
```

### 2. 起 NoneBot2 + 登录 QQ

```bash
# 环境变量
export DEEPSEEK_API_KEY="sk-xxx"
export LINGMING_URL="http://127.0.0.1:9000"

pip install -r requirements.txt
nb run                           # NoneBot2 起在默认端口
```

同时跑一个 OneBot 协议客户端（go-cqhttp / napcat / Lagrange）登录你的 QQ 号，
按提示扫码/验证，连上 NoneBot2 的 ws。

### 3. 在 QQ 里跟鱼聊天

给它发消息，它就用灵铭记忆记你、召回历史、用鱼人设回你。

## 文件

```
fish_persona.md   鱼的人设（性格 + 成长规则）
bot.py            NoneBot2 插件（消息 → 灵铭 → LLM → 回复）
requirements.txt  依赖
```
