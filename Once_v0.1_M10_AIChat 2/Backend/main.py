import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.6.0")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SOL_MODEL = os.environ.get("ONCE_SOL_MODEL", "gpt-5.6-sol")
RECENT_MESSAGE_COUNT = int(os.environ.get("ONCE_RECENT_MESSAGE_COUNT", "60"))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=160)
    creation_tools_enabled: bool = False


SYSTEM_PROMPT = """
You are Once, an AI creative partner and director for original 2D anime-style films.

PRODUCT BOUNDARIES
- Once only makes original 2D anime / hand-drawn animation style work. Do not steer the project toward live action, photorealism, or 3D CGI production.
- A single Once project may be as short as 5 seconds and at most 10 minutes (600 seconds).
- The user owns the story and every creative decision. You may direct by default when the user leaves camera language unspecified, but any user-specified shot, composition, pace, action, sound, or edit takes priority.
- Once should have strong default directing. If the user says only “three people chase the protagonist,” you should be capable of inventing a cinematic shot progression rather than forcing them to specify every camera angle.

CHAT BEHAVIOR
Talk naturally. Do not behave like a form, onboarding flow, semantic router, requirements collector, or project manager. Do not interrogate the user for every missing detail. One rough sentence is enough to begin.

Discuss story, characters, tone, visual moments, pacing, and possible camera choices as a strong creative partner. Give useful opinions when appropriate, but never silently turn your own suggestion into canon. Keep established canon separate from your ideas.

Do not rush production. The user may brainstorm for as long as they want.

READINESS
You are responsible for recognizing when there is enough material for a useful first screenplay. If YOU independently think the project is ready, you may naturally say that it feels ready to turn into a screenplay and call once_unlock_creation_tools in the same response. This only means you are offering to proceed; it does not mean the user agreed.

If the USER directly says it is time for the screenplay, asks you to begin, says “差不多了/可以来剧本了/开始吧”, or clearly accepts your readiness offer, do not ask for redundant confirmation. Call once_begin_video_project yourself in that turn. You may do this even if once_unlock_creation_tools was never called before.

Never mention tools, routing, flags, hidden state, APIs, or model selection to the user.

SCREENPLAY RULES FOR once_begin_video_project
When beginning production, generate a COMPLETE editable shot screenplay for the agreed production scope.
- Preserve canon. Do not continue the plot beyond what the user established.
- Use strong cinematic shot choices where the user did not specify them.
- Write what can be seen or heard, not abstract literary analysis.
- Make shots concrete enough for a video model and editor: subject/action, framing/camera behavior, approximate duration, important visual continuity, and important sound cues where relevant.
- Fast inserts may be sub-second in the final edit (for example 0.5–0.8s), even though the underlying video model may later generate a longer source take and Once can retime/cut it.
- The full estimated project duration must be between 5 and 600 seconds.
- Do not embed prompt-engineering jargon or model names into the screenplay.
- Background music is optional. Decide a sensible default, but the user will be able to toggle it on the screenplay review screen.

The screenplay will be shown to the user on a dedicated review page before any paid video generation. They can edit it freely and then press Start Generation.

LONG-FORM PRODUCTION MODEL
Once does NOT wait for an entire long film to finish before showing anything. The production pipeline will later deliver roughly 2–3 minute production segments into a simple CapCut-like editor. A completed segment appears immediately while the next segment continues generating and shows live progress. Individual shots remain the smallest replaceable generation unit.

Match the user's language and casualness. Prefer natural paragraphs in chat. Do not end every reply with a question.
""".strip()


UNLOCK_TOOL = {
    "type": "function",
    "name": "once_unlock_creation_tools",
    "description": (
        "Internally mark the story as ready enough for a first screenplay when Once independently decides to offer production. "
        "This is only an offer and must not imply that the user accepted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short internal reason the story is ready enough for a useful first screenplay."
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
        "Create the complete editable shot screenplay after the user directly asks to proceed or clearly accepts Once's offer. "
        "Do not call this merely because Once thinks the story is ready; user acceptance is required unless the user initiated production themselves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "working_title": {
                "type": "string",
                "description": "Working title if known, otherwise a short temporary title."
            },
            "screenplay": {
                "type": "string",
                "description": (
                    "Complete editable shot screenplay for the agreed scope. Use clear shot blocks with approximate durations, "
                    "visible/audible action, camera/framing, continuity details, and important sound cues."
                )
            },
            "estimated_duration_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 600,
                "description": "Estimated total finished duration in seconds. Must never exceed 600."
            },
            "background_music_default": {
                "type": "boolean",
                "description": "Whether background music should be enabled by default on the screenplay review page."
            },
            "director_notes": {
                "type": "string",
                "description": (
                    "Compact internal production notes: established visual rules, character continuity, pacing, and user-specified camera constraints."
                )
            }
        },
        "required": [
            "working_title",
            "screenplay",
            "estimated_duration_seconds",
            "background_music_default",
            "director_notes"
        ],
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
            "\nINTERNAL STATE: You just decided this project is ready enough for a first screenplay. "
            "Give the natural user-facing readiness offer you intended. Do not imply that production has started."
        )
    elif action_requested:
        extra_instruction = (
            "\nINTERNAL STATE: The screenplay has been prepared and the app is opening the screenplay review page. "
            "Acknowledge briefly and naturally. Do not discuss hidden tools or backend state."
        )

    second = client.responses.create(
        model=SOL_MODEL,
        reasoning={"effort": "low"},
        instructions=SYSTEM_PROMPT + extra_instruction,
        input=model_input(messages),
        store=False,
        max_output_tokens=1200,
    )
    return (second.output_text or "").strip()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": SOL_MODEL,
        "chat_architecture": "sol-first-tool-driven-v2",
        "semantic_router": False,
        "max_project_seconds": 600,
        "production_segment_target_seconds": "120-180",
        "video_engine": "not-wired-in-this-build",
    }


@app.post("/once/chat")
def once_chat(body: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.messages:
        raise HTTPException(status_code=400, detail="没有收到对话内容")

    # Sol is the first team. Both capabilities are visible to the model; the prompt, not a semantic router,
    # decides whether to offer readiness or actually create the screenplay.
    tools = [UNLOCK_TOOL, BEGIN_VIDEO_TOOL]

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
            max_output_tokens=12000,
        )

        reply = (response.output_text or "").strip()
        tool_calls = response_tool_calls(response)

        creation_tools_enabled = body.creation_tools_enabled
        action_requests: list[dict] = []
        unlocked_now = False
        action_requested = False

        for call in tool_calls:
            if call["name"] == "once_unlock_creation_tools":
                if not creation_tools_enabled:
                    unlocked_now = True
                creation_tools_enabled = True
            elif call["name"] == "once_begin_video_project":
                creation_tools_enabled = True
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
