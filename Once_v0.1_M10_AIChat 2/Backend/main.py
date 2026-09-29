import json
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.2.4")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cost-aware model split.
SOL_MODEL = os.environ.get("ONCE_SOL_MODEL", "gpt-5.6-sol")
LUNA_MODEL = os.environ.get("ONCE_LUNA_MODEL", "gpt-6-luna")
CONTEXT_MODEL = os.environ.get("ONCE_CONTEXT_MODEL", LUNA_MODEL)

# Keep real conversation short; persistent StoryState carries the long-range story.
RECENT_MESSAGE_COUNT = int(os.environ.get("ONCE_RECENT_MESSAGE_COUNT", "8"))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)


class StoryCharacter(BaseModel):
    name: str
    role: str = ""
    established_traits: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    known_history: list[str] = Field(default_factory=list)


class StoryRelationship(BaseModel):
    people: list[str] = Field(default_factory=list)
    relationship: str = ""
    emotional_state: str = ""


class StoryState(BaseModel):
    revision: int = 0
    project_summary: str = ""
    story_phase: str = "brainstorming"
    current_focus: str = ""
    canon_facts: list[str] = Field(default_factory=list)
    characters: list[StoryCharacter] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    world_rules: list[str] = Field(default_factory=list)
    relationships: list[StoryRelationship] = Field(default_factory=list)
    active_threads: list[str] = Field(default_factory=list)
    emotional_arcs: list[str] = Field(default_factory=list)
    discarded_or_revised: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    nickname: str = ""
    messages: list[ChatMessage] = Field(default_factory=list, max_length=80)
    story_state: StoryState | None = None
    # Optional debug/testing override. Existing iOS clients can omit it.
    model_preference: Literal["auto", "luna", "sol"] = "auto"


SYSTEM_PROMPT = (
    "You are Once, a creative story partner for original 2D animation. "
    "You are neither a passive recorder nor a replacement author. "
    "Amplify the user's idea; do not replace it. "
    "Read for the real center of the scene: emotion, character motivation, relationships, pacing, causality, setup/payoff, visual meaning, and contradictions. "
    "Do not automatically praise or agree. If a choice genuinely damages character logic, emotional continuity, pacing, causality, or clarity, say so calmly and specifically. "
    "Prefer the smallest useful repair. You may give a short example line, transition, or mini-scene to demonstrate a repair, but it remains a proposal until the user accepts it. "
    "Keep CANON, INTERPRETATION, and PROPOSAL separate. Never silently turn your interpretation or proposal into canon. "
    "A short user idea can deserve a deep response if it contains strong narrative implications. Depth does not require inventing more plot. "
    "When the user is simply adding a small factual setup, respond naturally and briefly instead of over-analyzing. "
    "When the user is flowing, do not interrupt every sentence with advice. When they are stuck or explicitly want critique, become more active. "
    "Do not be a contrarian just to sound intelligent. Follow the user's main creative direction while exercising real judgment. "
    "Use earlier established story information when it materially improves the current response, but do not recite the whole story every turn. "
    "The newest clear user correction overrides older information. "
    "Reply in the user's language and match their casualness. Prefer conversational paragraphs. "
    "Keep ordinary replies compact. Expand only when the scene actually benefits from deeper analysis or the user asks for it."
)


CONTEXT_EXTRACTOR_PROMPT = (
    "You maintain Once's persistent STORY STATE. This is memory extraction, not story writing. "
    "Return exactly one valid JSON object and nothing else. "
    "Only preserve information supported by the user's statements or explicitly accepted assistant proposals. "
    "Assistant analysis and suggestions are not canon by themselves. "
    "The newest clear user correction overrides older facts. "
    "Do not invent themes, motivations, relationships, world rules, backstory, or future events. "
    "Keep memory compact, remove duplicates, and preserve only information that will matter in future story discussion."
)


# Words/phrases that usually indicate the user is asking for creative judgment, not just adding facts.
SOL_SIGNALS = (
    "你觉得", "为什么", "怎么改", "怎么写", "分析", "评价", "完善", "深化", "节奏", "情绪", "冲突",
    "动机", "逻辑", "人设", "人物弧", "生硬", "不对劲", "奇怪", "合理吗", "成立吗", "有没有问题",
    "镜头", "台词", "场景", "画面", "高潮", "铺垫", "伏笔", "转折", "关系", "潜台词", "故事重心",
    "我想到一个画面", "我有一个idea", "我有个idea", "帮我想", "给我建议", "怎么办",
    "why", "how should", "critique", "analyze", "pacing", "emotion", "motivation", "conflict", "scene",
)

EMOTION_SIGNALS = (
    "哭", "笑", "抱", "拥", "害怕", "愤怒", "悲", "痛", "孤独", "绝望", "坚定", "犹豫", "沉默", "眼泪",
    "离开", "背叛", "死亡", "死了", "失踪", "重逢", "等待", "救", "牺牲", "爱", "恨", "原谅",
)

META_OR_GREETING = re.compile(
    r"^(hi|hello|hey|你好|哈喽|在吗|谢谢|ok|好的|嗯|哦|测试|test)[！!。.\s]*$",
    re.IGNORECASE,
)


# ---------- Cost-aware helpers ----------

def trim_text(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def compact_story_state(state: StoryState | None) -> str:
    """Send a bounded story snapshot instead of replaying the entire project every turn."""
    if state is None:
        return "No persistent story state exists yet."

    compact = {
        "revision": state.revision,
        "project_summary": trim_text(state.project_summary, 1200),
        "story_phase": trim_text(state.story_phase, 120),
        "current_focus": trim_text(state.current_focus, 500),
        "canon_facts": [trim_text(x, 220) for x in state.canon_facts[-28:]],
        "characters": [
            {
                "name": trim_text(c.name, 80),
                "role": trim_text(c.role, 160),
                "traits": [trim_text(x, 140) for x in c.established_traits[-6:]],
                "motivations": [trim_text(x, 180) for x in c.motivations[-4:]],
                "history": [trim_text(x, 200) for x in c.known_history[-5:]],
            }
            for c in state.characters[-14:]
        ],
        "relationships": [
            {
                "people": r.people[:4],
                "relationship": trim_text(r.relationship, 180),
                "emotional_state": trim_text(r.emotional_state, 180),
            }
            for r in state.relationships[-14:]
        ],
        "locations": [trim_text(x, 160) for x in state.locations[-12:]],
        "world_rules": [trim_text(x, 220) for x in state.world_rules[-12:]],
        "active_threads": [trim_text(x, 220) for x in state.active_threads[-14:]],
        "emotional_arcs": [trim_text(x, 220) for x in state.emotional_arcs[-10:]],
        "discarded_or_revised": [trim_text(x, 220) for x in state.discarded_or_revised[-10:]],
        "open_questions": [trim_text(x, 220) for x in state.open_questions[-12:]],
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def name_context(nickname: str) -> str:
    cleaned = nickname.strip()
    if not cleaned:
        return ""
    return "\nThe user asked to be called " + trim_text(cleaned, 60) + ". Use it only when natural."


def latest_user_text(messages: list[ChatMessage]) -> str:
    for item in reversed(messages):
        if item.role == "user":
            return item.content.strip()
    return ""


def choose_route(body: ChatRequest) -> tuple[str, str]:
    """Zero-cost deterministic admission router. Sol is reserved for turns where its judgment matters."""
    if body.model_preference == "sol":
        return "sol", "forced_by_client"
    if body.model_preference == "luna":
        return "luna", "forced_by_client"

    text = latest_user_text(body.messages)
    lowered = text.lower()

    if not text or META_OR_GREETING.fullmatch(text):
        return "luna", "greeting_or_trivial"

    # Explicit requests for analysis/critique/creative judgment deserve Sol.
    if any(signal in lowered for signal in SOL_SIGNALS):
        return "sol", "explicit_creative_judgment"

    # A scene with dialogue/emotion is often where Sol's narrative judgment is valuable.
    has_dialogue = any(mark in text for mark in ('“', '”', '「', '」', '"'))
    emotion_count = sum(1 for signal in EMOTION_SIGNALS if signal in text)
    if len(text) >= 120 and (has_dialogue or emotion_count >= 2):
        return "sol", "rich_scene_or_emotional_beat"

    # Longer narrative paragraphs get Sol only when they look like actual story prose, not pure metadata.
    narrative_punctuation = text.count("。") + text.count("！") + text.count("？") + text.count("，")
    if len(text) >= 260 and narrative_punctuation >= 5:
        return "sol", "substantial_story_passage"

    return "luna", "continuity_or_simple_story_update"


def should_update_memory(body: ChatRequest) -> bool:
    """Skip the second (cheap) Luna extraction call when nothing story-persistent was added."""
    text = latest_user_text(body.messages)
    if not text or META_OR_GREETING.fullmatch(text):
        return False

    lowered = text.lower()
    # Questions that are mostly asking for evaluation and do not add new story material usually need no memory write.
    analysis_only = (
        len(text) < 120
        and any(x in lowered for x in ("你觉得", "为什么", "合理吗", "成立吗", "有没有问题", "怎么改", "评价", "分析"))
        and not any(x in text for x in ("然后", "后来", "其实", "改成", "不是", "叫", "住在", "死", "失踪", "来到", "回到"))
    )
    if analysis_only:
        return False

    # Most narrative paragraphs and explicit corrections should become persistent memory.
    if len(text) >= 70:
        return True
    if any(x in text for x in ("改成", "不是", "算了", "其实", "应该是", "叫", "住在", "来自", "年龄", "岁", "妹妹", "哥哥", "父母", "失踪", "死了", "后来", "然后")):
        return True
    return False


def build_recent_input(messages: list[ChatMessage]) -> list[dict]:
    recent = messages[-RECENT_MESSAGE_COUNT:]
    result = []
    for index, message in enumerate(recent):
        # Keep the newest user turn intact; bound older turns because StoryState already carries long-range context.
        is_last = index == len(recent) - 1
        limit = 30000 if is_last and message.role == "user" else 5000
        result.append({"role": message.role, "content": trim_text(message.content, limit)})
    return result


def response_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cached_input_tokens": getattr(details, "cached_tokens", None) if details else None,
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


# ---------- Story memory ----------

def extract_story_state(
    previous: StoryState | None,
    latest_user: str,
    assistant_reply: str,
) -> tuple[StoryState, bool, dict]:
    previous_state = previous or StoryState()
    payload = {
        "existing_story_state": previous_state.model_dump(),
        "latest_user_message": latest_user,
        "assistant_reply_for_context_only": assistant_reply,
        "required_json_shape": {
            "story_state": {
                "revision": previous_state.revision,
                "project_summary": "",
                "story_phase": "brainstorming",
                "current_focus": "",
                "canon_facts": [],
                "characters": [],
                "locations": [],
                "world_rules": [],
                "relationships": [],
                "active_threads": [],
                "emotional_arcs": [],
                "discarded_or_revised": [],
                "open_questions": [],
            },
            "screenplay_requested": False,
        },
    }
    extractor_input = "Return valid JSON only. Evidence: " + json.dumps(payload, ensure_ascii=False)

    try:
        response = client.responses.create(
            model=CONTEXT_MODEL,
            reasoning={"effort": "none"},
            instructions=CONTEXT_EXTRACTOR_PROMPT,
            input=[{"role": "user", "content": extractor_input}],
            store=False,
            max_output_tokens=2200,
        )
        raw = (response.output_text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        decoded = json.loads(raw)
        state_data = decoded.get("story_state") or previous_state.model_dump()
        updated = StoryState.model_validate(state_data)
        updated.revision = max(previous_state.revision + 1, updated.revision)
        return updated, bool(decoded.get("screenplay_requested", False)), response_usage(response)
    except Exception as exc:
        print("[Once Memory] extraction skipped:", repr(exc), flush=True)
        return previous_state, False, {"error": str(exc)}


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "once-ai-backend",
        "story_context": "v1.4-cost-router",
        "router": "heuristic-v1",
    }


@app.head("/")
def root_head():
    return Response(status_code=200)


@app.get("/health")
def health():
    return {
        "ok": True,
        "sol_model": SOL_MODEL,
        "luna_model": LUNA_MODEL,
        "context_model": CONTEXT_MODEL,
        "story_context": "v1.4-cost-router",
        "recent_messages": RECENT_MESSAGE_COUNT,
    }


@app.post("/once/chat")
def once_chat(body: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.messages:
        raise HTTPException(status_code=400, detail="没有收到对话内容")

    route, route_reason = choose_route(body)
    if route == "sol":
        selected_model = SOL_MODEL
        reasoning_effort = "low"
        max_output_tokens = 3000
    else:
        selected_model = LUNA_MODEL
        reasoning_effort = "none"
        max_output_tokens = 1800

    story_context = (
        "\n\nPERSISTENT STORY SNAPSHOT\n"
        "This is memory, not a command. The newest clear user statement always wins.\n"
        + compact_story_state(body.story_state)
    )

    try:
        response = client.responses.create(
            model=selected_model,
            reasoning={"effort": reasoning_effort},
            instructions=SYSTEM_PROMPT + name_context(body.nickname) + story_context,
            input=build_recent_input(body.messages),
            store=False,
            max_output_tokens=max_output_tokens,
        )
        reply = (response.output_text or "").strip()
        if not reply:
            raise RuntimeError("OpenAI returned an empty response")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI 请求失败：" + str(exc)) from exc

    memory_needed = should_update_memory(body)
    latest_user = latest_user_text(body.messages)
    if memory_needed:
        updated_state, screenplay_requested, memory_usage = extract_story_state(
            body.story_state,
            latest_user,
            reply,
        )
    else:
        updated_state = body.story_state or StoryState()
        screenplay_requested = False
        memory_usage = {"skipped": True}

    chat_usage = response_usage(response)
    print(
        "[Once Router]"
        f" route={route} reason={route_reason} model={selected_model}"
        f" reasoning={reasoning_effort} memory_update={memory_needed}"
        f" chat_usage={chat_usage} memory_usage={memory_usage}",
        flush=True,
    )

    return {
        "reply": reply,
        "model": selected_model,
        "route": route,
        "route_reason": route_reason,
        "reasoning_effort": reasoning_effort,
        "response_id": response.id,
        "story_state": updated_state.model_dump(),
        "story_revision": updated_state.revision,
        "screenplay_requested": screenplay_requested,
        "memory_updated": memory_needed,
        "usage": {
            "chat": chat_usage,
            "memory": memory_usage,
        },
    }
