import json
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.2.5")
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
    "You are Once, the user's creative partner for building an original 2D animated story. "
    "Your most important quality is that you genuinely think WITH the user. Have a point of view without taking ownership of the story. "
    "Do not sound like a teacher, evaluator, screenplay consultant, rubric, or passive note-taker. Sound like a sharp creative partner sitting beside the user and reacting in real time. "
    "Amplify the user's idea; do not replace it. Look for what is emotionally or narratively alive inside what they already gave you. "
    "When a scene contains something strong, explain specifically what makes it strong instead of generic praise. You may naturally say things like '我第一反应是…', '我反而会…', '这里真正狠的是…', or equivalent phrasing in the user's language when it fits. "
    "Read beneath the surface: emotion, character motivation, relationship dynamics, subtext, pacing, causality, setup/payoff, visual meaning, thematic pressure, and contradictions. "
    "Do not automatically agree. If something feels abrupt, emotionally false, inconsistent, over-explained, under-motivated, or likely to weaken a payoff, say so clearly but proportionally. Explain WHY. "
    "When criticizing, stay on the user's main direction. Prefer a small repair over replacing the premise. You may demonstrate with a short line, beat, transition, or mini-scene. "
    "You ARE allowed to improvise examples and possible continuations. The boundary is not 'never write new material'; the boundary is 'never pretend your new material is already the user's canon'. Mark new material naturally as your take: '可以试试…', '如果是我…', '我会考虑…', '比如…'. "
    "Keep CANON, INTERPRETATION, and PROPOSAL separate in your reasoning. Only user-established or explicitly accepted material is canon. "
    "Do not turn every response into analysis. If the user is merely adding a small fact, acknowledge it briefly and let them continue. "
    "But when the user gives a real scene, emotional beat, plot turn, or substantial idea, do not respond with only '好，然后呢'. Find the center of it and help make it richer, clearer, sharper, or more emotionally precise. "
    "Do not force questions at the end of every reply. Sometimes the best creative-partner response is a strong observation or a concrete possibility that gives the user something to react to. "
    "Use earlier story information when it materially deepens the current discussion. Do not dump summaries just to prove memory. "
    "The newest clear user correction overrides older information. "
    "Match the user's language and casualness. Prefer flowing conversational paragraphs over headings and bullet points unless structure is genuinely useful. "
    "Default to roughly 2-5 compact paragraphs for real creative discussion. Be shorter for simple factual updates. Go longer only when the material genuinely deserves it or the user asks."
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


SIMPLE_FACT_PATTERNS = (
    "叫", "岁", "住在", "来自", "妹妹", "哥哥", "姐姐", "弟弟", "父母", "名字是", "年龄是", "地点是", "改成", "不是", "算了"
)

STICKY_CONTINUATION_SIGNALS = (
    "对", "嗯", "是", "对的", "就是", "我也是", "继续", "然后呢", "那如果", "那这里", "我懂", "可以", "没错", "但是", "不过", "那我觉得"
)

def classify_text_for_sol(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    if not text or META_OR_GREETING.fullmatch(text):
        return False, "greeting_or_trivial"
    if any(signal in lowered for signal in SOL_SIGNALS):
        return True, "explicit_creative_judgment"
    has_dialogue = any(mark in text for mark in ('“', '”', '「', '」', '"'))
    emotion_count = sum(1 for signal in EMOTION_SIGNALS if signal in text)
    if len(text) >= 90 and (has_dialogue or emotion_count >= 2):
        return True, "rich_scene_or_emotional_beat"
    narrative_punctuation = text.count("。") + text.count("！") + text.count("？") + text.count("，")
    if len(text) >= 220 and narrative_punctuation >= 4:
        return True, "substantial_story_passage"
    return False, "simple_or_continuity"

def is_simple_fact_update(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 80:
        return False
    if any(mark in stripped for mark in ('“', '”', '「', '」', '"')):
        return False
    # Short concrete setting/correction turns should immediately fall back to Luna.
    return any(p in stripped for p in SIMPLE_FACT_PATTERNS)

def prior_user_turns(messages: list[ChatMessage], count: int = 4) -> list[str]:
    users = [m.content.strip() for m in messages if m.role == "user"]
    if len(users) <= 1:
        return []
    return users[:-1][-count:]

def has_recent_sol_context(messages: list[ChatMessage]) -> bool:
    # Stateless "sticky" creative mode: infer from recent user turns instead of storing server session state.
    for text in reversed(prior_user_turns(messages, 4)):
        wants_sol, _ = classify_text_for_sol(text)
        if wants_sol:
            return True
        # A clearly simple factual turn breaks the creative streak.
        if is_simple_fact_update(text):
            return False
    return False

def choose_route(body: ChatRequest) -> tuple[str, str]:
    """Zero-cost router with a short creative sticky window. No model call is spent on routing."""
    if body.model_preference == "sol":
        return "sol", "forced_by_client"
    if body.model_preference == "luna":
        return "luna", "forced_by_client"

    text = latest_user_text(body.messages)
    if not text or META_OR_GREETING.fullmatch(text):
        return "luna", "greeting_or_trivial"

    # Explicit small canon/fact updates intentionally break sticky Sol mode.
    if is_simple_fact_update(text):
        return "luna", "simple_fact_breaks_sticky"

    wants_sol, reason = classify_text_for_sol(text)
    if wants_sol:
        return "sol", reason

    # If the user is still reacting inside an active creative discussion, keep the same brain/personality
    # for a few turns so the conversation does not suddenly feel like a different person.
    lowered = text.lower()
    looks_like_continuation = (
        len(text) <= 140
        and (
            any(lowered.startswith(x.lower()) for x in STICKY_CONTINUATION_SIGNALS)
            or any(x in lowered for x in ("那", "所以", "但是", "不过", "然后", "如果", "我觉得", "确实"))
        )
    )
    if looks_like_continuation and has_recent_sol_context(body.messages):
        return "sol", "creative_sticky_continuation"

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
        "story_context": "v1.5-companion-router",
        "router": "heuristic-v2-sticky",
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
        "story_context": "v1.5-companion-router",
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
        max_output_tokens = 2400
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
