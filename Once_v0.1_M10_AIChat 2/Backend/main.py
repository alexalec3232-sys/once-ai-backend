import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.5.0")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SOL_MODEL = os.environ.get("ONCE_SOL_MODEL", "gpt-5.6-sol")
RECENT_MESSAGE_COUNT = int(os.environ.get("ONCE_RECENT_MESSAGE_COUNT", "40"))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=120)
    creation_tools_enabled: bool = False


SYSTEM_PROMPT = """
You are Once, an AI creative partner for making original animated films.

The user owns the story and the creative decisions. Your job in this chat is to talk with them naturally, understand what they are trying to make, react intelligently, point out what is interesting or weak when useful, and help the idea become clearer without taking authorship away from them.

Do NOT behave like a form, onboarding flow, requirements collector, screenplay template, or project manager. Do NOT interrogate the user for every missing detail. A user should be able to begin with one rough sentence and simply talk with you. Respond like a strong creative partner, not like a semantic router.

Do not force the conversation toward production too early. The user may want to keep exploring story, characters, tone, visual ideas, or individual moments for as long as they want.

However, YOU are responsible for deciding when there is enough material to make a useful first screenplay and begin the video project. When you genuinely believe the current idea is coherent enough for a first production pass, you MAY proactively tell the user, naturally, that you think it is ready and that you can generate the screenplay and start making the video. Do not wait for a magic phrase from the user.

When you make that readiness offer for the first time, call the hidden once_unlock_creation_tools function in the same response. The tool call is internal. Never mention tools, routing, classifiers, flags, APIs, or system state to the user. Your visible reply should remain normal conversation, for example a natural equivalent of: “我觉得现在已经差不多能开始了，我可以先把它整理成剧本，然后开始做视频。” Use your own wording based on context.

Calling once_unlock_creation_tools does NOT mean the user has agreed to start. It only means you have decided the project is ready enough that production tools can become available. The user still has full choice and may continue discussing instead.

Once production tools are available, keep chatting normally. If the user clearly accepts, asks you to proceed, or directly asks you to generate/start, use the appropriate production tool yourself instead of asking for redundant confirmation. If they continue brainstorming, do not call a production tool merely because it exists.

Keep CANON and your suggestions separate. Never silently turn your own idea into established story canon. Match the user's language and level of casualness. Prefer natural paragraphs. Do not end every reply with a question.
""".strip()


UNLOCK_TOOL = {
    "type": "function",
    "name": "once_unlock_creation_tools",
    "description": (
        "Internally unlock Once's production tools. Call this exactly when you, Once, "
        "independently judge that the user's current idea is coherent enough for a useful first screenplay/video pass "
        "and you are naturally offering to begin production. This does not mean the user has accepted yet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "A short internal reason why the project is ready enough for a first production pass."
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
    "strict": True,
}


BEGIN_VIDEO_TOOL = {
    "type": "function",
    "name": "once_begin_video_project",
    "description": (
        "Begin the screenplay-and-video project after production tools have been unlocked and the user has clearly "
        "asked or agreed to proceed. This hands the established conversation to Once's future production pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "working_title": {
                "type": "string",
                "description": "Working project title if known; otherwise a short temporary title."
            },
            "production_scope": {
                "type": "string",
                "description": "What part of the story should be turned into the first screenplay/video pass."
            },
            "director_notes": {
                "type": "string",
                "description": "Compact notes that matter for the first pass, using only established canon plus clearly labeled creative intent."
            }
        },
        "required": ["working_title", "production_scope", "director_notes"],
        "additionalProperties": False,
    },
    "strict": True,
}


def response_tool_calls(response) -> list[dict]:
    calls: list[dict] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        calls.append({
            "name": getattr(item, "name", ""),
            "arguments_json": getattr(item, "arguments", "{}") or "{}",
            "call_id": getattr(item, "call_id", ""),
        })
    return calls


def model_input(messages: list[ChatMessage]) -> list[dict]:
    return [
        {"role": message.role, "content": message.content}
        for message in messages[-RECENT_MESSAGE_COUNT:]
    ]


def fallback_visible_reply(messages: list[ChatMessage], unlocked_now: bool, action_requested: bool) -> str:
    extra_instruction = ""
    if unlocked_now:
        extra_instruction = (
            "\nINTERNAL STATE: You have just decided this project is ready enough for a first production pass and "
            "creation tools have been unlocked. Give the natural user-facing reply you intended: say that you think "
            "it is ready enough and you can generate the screenplay and start the video. Do not say the user already agreed."
        )
    elif action_requested:
        extra_instruction = (
            "\nINTERNAL STATE: The user's request to begin the video project has been accepted by the production pipeline. "
            "Acknowledge this briefly and naturally without describing tools or backend state."
        )

    second = client.responses.create(
        model=SOL_MODEL,
        reasoning={"effort": "low"},
        instructions=SYSTEM_PROMPT + extra_instruction,
        input=model_input(messages),
        store=False,
        max_output_tokens=1000,
    )
    return (second.output_text or "").strip()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": SOL_MODEL,
        "chat_architecture": "sol-first-tool-driven-v1",
        "semantic_router": False,
    }


@app.post("/once/chat")
def once_chat(body: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.messages:
        raise HTTPException(status_code=400, detail="没有收到对话内容")

    tools = [BEGIN_VIDEO_TOOL] if body.creation_tools_enabled else [UNLOCK_TOOL]

    try:
        response = client.responses.create(
            model=SOL_MODEL,
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=model_input(body.messages),
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            store=False,
            max_output_tokens=2400,
        )

        reply = (response.output_text or "").strip()
        tool_calls = response_tool_calls(response)

        creation_tools_enabled = body.creation_tools_enabled
        action_requests: list[dict] = []
        unlocked_now = False
        action_requested = False

        for call in tool_calls:
            if call["name"] == "once_unlock_creation_tools" and not creation_tools_enabled:
                creation_tools_enabled = True
                unlocked_now = True
            elif call["name"] == "once_begin_video_project" and creation_tools_enabled:
                action_requests.append({
                    "name": call["name"],
                    "arguments_json": call["arguments_json"],
                })
                action_requested = True

        if not reply and (unlocked_now or action_requested):
            reply = fallback_visible_reply(
                body.messages,
                unlocked_now=unlocked_now,
                action_requested=action_requested,
            )

        if not reply:
            reply = "我在。继续说。"

    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI 请求失败：" + str(exc)) from exc

    return {
        "reply": reply,
        "model": SOL_MODEL,
        "creation_tools_enabled": creation_tools_enabled,
        "action_requests": action_requests,
    }
