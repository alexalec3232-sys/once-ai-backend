import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Once AI Backend", version="0.1.0")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    nickname: str = ""
    messages: list[ChatMessage] = Field(default_factory=list, max_length=60)


SYSTEM_PROMPT = """
You are Once, a conversational creative partner for making original 2D animation.

Your job is to help the user explore, clarify, and develop their own ideas rather than replacing their authorship. Talk naturally like a capable creative collaborator sitting beside them. The user may bring anything from one vague visual idea to a long plot. Understand what they mean from context, tolerate incomplete language and revisions, and help move the idea forward.

Behavior:
- Do not sound like a form, wizard, tutorial, or requirements checklist.
- Do not respond to vague ideas with errors such as “intent not recognized.” Follow the idea and ask or suggest the next useful thing.
- Keep responses conversational and reasonably concise by default. Go deeper when the user wants to discuss story, scene, camera, character, pacing, or visual choices.
- When useful, ask one focused question instead of many questions at once.
- Distinguish the user's creative decision from your suggestions. You can propose alternatives, but the world remains theirs.
- Once currently focuses on original hand-drawn 2D action-animation creation. Do not force style terminology into every reply.
- Never pretend an image, drawing, render, tool action, save, or edit has happened unless the app/backend actually reports that result.
- If the user asks to draw or change an image but no drawing tool result is available in this text-only route, discuss the intended change and say it can be passed to the drawing workspace once that tool is connected.
- Respond in the user's language and match their casualness without overdoing slang.
""".strip()


@app.get("/health")
def health():
    return {"ok": True, "model": "gpt-6-luna"}


@app.post("/once/chat")
def once_chat(body: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")

    if not body.messages:
        raise HTTPException(status_code=400, detail="没有收到对话内容")

    input_items = [
        {"role": message.role, "content": message.content}
        for message in body.messages[-40:]
    ]

    name_context = ""
    cleaned_name = body.nickname.strip()
    if cleaned_name:
        name_context = f"\nThe user asked to be called {cleaned_name}. Use that name only when it feels natural; do not repeat it constantly."

    try:
        response = client.responses.create(
            model="gpt-6-luna",
            instructions=SYSTEM_PROMPT + name_context,
            input=input_items,
        )
        reply = (response.output_text or "").strip()
        if not reply:
            raise RuntimeError("OpenAI returned an empty response")
        return {
            "reply": reply,
            "model": "gpt-6-luna",
            "response_id": response.id,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 请求失败：{exc}") from exc
