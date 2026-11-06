# -*- coding: utf-8 -*-
"""大肥鱼 · QQ 接入（NoneBot2）
流程：QQ 消息 → 写灵铭记忆 → 召回历史 → 鱼(LLM)回复 → 助手话写回记忆
"""
import os
import httpx
from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent

LINGMING = os.environ.get("LINGMING_URL", "http://127.0.0.1:9000")
LLM_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
_PERSONA = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish_persona.md"), encoding="utf-8").read()


async def _post(url, json):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json=json)
        return r.json() if r.status_code == 200 else {}


async def lingming_add(text, role):
    await _post(f"{LINGMING}/write", {"text": text, "role": role})


async def lingming_retrieve(query, topK=3):
    j = await _post(f"{LINGMING}/retrieve", {"query": query, "topK": topK})
    return j.get("items", [])


async def fish_reply(user_text, history):
    hist = "\n".join(f"[{'助手' if h['role'] == 'assistant' else '用户'}] {h['text']}" for h in history)
    msgs = [{"role": "system", "content": _PERSONA}]
    if hist:
        msgs.append({"role": "user", "content": f"（你记忆里相关的旧事，可以用）\n{hist}"})
    msgs.append({"role": "user", "content": user_text})
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "messages": msgs},
        )
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return "咕噜…（我卡住了）"


@on_message().handle()
async def handle(event: MessageEvent):
    text = str(event.get_message()).strip()
    if not text:
        return
    # 1. 用户话写进记忆（边聊边学）
    await lingming_add(text, "user")
    # 2. 召回相关历史（坐标寻址，不喂长上下文）
    history = await lingming_retrieve(text, 3)
    # 3. 鱼回复（人设 + 召回的历史）
    reply = await fish_reply(text, history)
    # 4. 助手话写回记忆
    await lingming_add(reply, "assistant")
    await event.reply(reply)
