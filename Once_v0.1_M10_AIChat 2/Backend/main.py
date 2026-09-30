import asyncio
import base64
import json
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from openai import OpenAI
import httpx
import websockets
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.4.1")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cost-aware model split.
SOL_MODEL = os.environ.get("ONCE_SOL_MODEL", "gpt-5.6-sol")
LUNA_MODEL = os.environ.get("ONCE_LUNA_MODEL", "gpt-6-luna")
CONTEXT_MODEL = os.environ.get("ONCE_CONTEXT_MODEL", LUNA_MODEL)

# Keep real conversation short; persistent StoryState carries the long-range story.
RECENT_MESSAGE_COUNT = int(os.environ.get("ONCE_RECENT_MESSAGE_COUNT", "8"))

# Project/voice/image models. These can be overridden in Render without touching the app.
LIVE_MODEL = os.environ.get("ONCE_LIVE_MODEL", "gpt-live-1")
DRAFT_IMAGE_MODEL = os.environ.get("ONCE_DRAFT_IMAGE_MODEL", "gpt-image-2.5-flare")
RENDER_IMAGE_MODEL = os.environ.get("ONCE_RENDER_IMAGE_MODEL", "gpt-image-2.5-sunburst")

# Original 2D action-animation baseline for Once. Keep this descriptive rather than copying any named frame.
ONCE_VISUAL_BASELINE = (
    "MANDATORY Once visual language for generated artwork: original hand-drawn 2D action-animation frame, "
    "bold clean black contours with slightly rough expressive linework, simplified flat cel shading, high-contrast graphic shapes, "
    "limited controlled color, exaggerated readable action posing, strong foreshortening, speed lines and motion accents when appropriate. "
    "Avoid photorealism, 3D/CG rendering, glossy game-cinematic rendering, painterly concept-art texture, and generic high-detail illustration. "
    "The result should feel designed for an animated sequence and remain editable from pass to pass. "
    "Preserve character identity, staging, camera direction, and continuity once the user has established them. "
    "Do not copy existing copyrighted characters, logos, subtitles, or exact frames; create original characters while preserving these visual traits."
)


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
        "project_layer": "m14.19-progressive-cocreate",
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
        "live_model": LIVE_MODEL,
        "draft_image_model": DRAFT_IMAGE_MODEL,
        "render_image_model": RENDER_IMAGE_MODEL,
        "project_prepare": "v2.2-progressive-cocreate",
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

# ---------- Project preparation ----------

class ScreenplayShot(BaseModel):
    number: int = Field(ge=1, le=200)
    visual: str = Field(min_length=1, max_length=2400)


class ScreenplaySummary(BaseModel):
    title: str = Field(default="", max_length=200)
    shots: list[ScreenplayShot] = Field(default_factory=list, max_length=200)


class ProjectPrepareRequest(BaseModel):
    nickname: str = ""
    messages: list[ChatMessage] = Field(default_factory=list, max_length=120)
    story_state: StoryState | None = None
    previous_screenplay: ScreenplaySummary | None = None


class ProjectBrief(BaseModel):
    project_summary: str = ""
    creative_intent: str = ""
    current_story_position: str = ""
    visual_direction: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    scene_goals: list[str] = Field(default_factory=list)
    continuity_constraints: list[str] = Field(default_factory=list)
    emotional_targets: list[str] = Field(default_factory=list)
    open_decisions: list[str] = Field(default_factory=list)
    voice_context: str = ""
    image_context: str = ""


def _clean_json_text(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    return raw


def _project_history_for_prompt(messages: list[ChatMessage]) -> str:
    # This is a one-off user-triggered compilation, so it can read more history than ordinary chat,
    # but still caps payload size because StoryState carries long-range facts.
    pieces: list[str] = []
    total = 0
    for message in messages[-36:]:
        part = f"{message.role.upper()}: {trim_text(message.content, 9000)}"
        if total + len(part) > 70000:
            break
        pieces.append(part)
        total += len(part)
    return "\n\n".join(pieces)


@app.post("/once/project/prepare")
def prepare_project(body: ProjectPrepareRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.messages:
        raise HTTPException(status_code=400, detail="还没有可整理的故事内容")

    prompt = (
        "You are Once's SHOT SCREENPLAY COMPILER for an original 2D animation project. "
        "This is not a normal chat reply and not a story review. Convert the user's established story into a concrete, drawable sequence of shots. "
        "The visible screenplay must describe only what a camera can directly record: environment, characters, body movement, facial movement, props, spatial relationships, camera distance/movement, literal dialogue already supported by the story, and directly audible actions when necessary. "
        "Never write philosophy, metaphor, symbolism, themes, moral meaning, hidden psychology, abstract emotion labels, or invisible intent in the visible shots. "
        "Forbidden examples include: '他终于意识到自己逃不掉了', '这一幕象征孤独', '气氛充满宿命感'. "
        "Rewrite such ideas into observable behavior, for example: '男主停下，肩膀起伏，抬头看向前方三人'. "
        "One shot should contain one clear visual beat. Split the sequence whenever composition, action focus, location, or camera purpose materially changes. "
        "Do not compress several important actions into one giant shot. "
        "You may add neutral cinematic bridge shots that do NOT change canon, such as an insect in the desert, wind moving sand, a shoe striking the ground, dust crossing frame, or a wide establishing view. "
        "These additions are allowed only to make the existing action filmable; do not invent new plot events, powers, relationships, lore, injuries, dialogue, or outcomes. "
        "If the user's story stops, the screenplay stops there. Do not continue the plot beyond the established endpoint. "
        "Number shots sequentially starting at 1. Keep each visual description concrete and concise enough that an image model could draw it without interpreting abstract prose. "
        "If a previous screenplay is provided, treat this as UPDATE SCREENPLAY: preserve useful shots and revise them to match the newest conversation and canon. "
        "If the user clearly says they are switching to a new story, project, title, or scene unrelated to the prior screenplay, discard the old screenplay instead of blending the two stories. The newest explicit project switch always wins. "
        "In the same JSON, also produce a hidden project_brief for downstream realtime voice and image models. The hidden brief MAY contain narrative intent, emotional targets, continuity rules, and unresolved decisions because the user does not see that part. "
        "Return exactly one valid JSON object with this exact top-level shape: "
        "{project_brief:{project_summary:string,creative_intent:string,current_story_position:string,visual_direction:[string],characters:[string],scene_goals:[string],continuity_constraints:[string],emotional_targets:[string],open_decisions:[string],voice_context:string,image_context:string},"
        "screenplay:{title:string,shots:[{number:int,visual:string}]}}. "
        "The screenplay title should use the established project/story title when known; otherwise use '镜头剧本'. "
        "The hidden image_context and voice_context must understand the exact shot plan and must not silently rewrite the user's canon."
    )

    previous_text = "None"
    if body.previous_screenplay is not None:
        previous_text = json.dumps(body.previous_screenplay.model_dump(), ensure_ascii=False)

    evidence = (
        "PERSISTENT STORY STATE:\n" + compact_story_state(body.story_state) +
        "\n\nPREVIOUS SCREENPLAY IF ANY:\n" + trim_text(previous_text, 30000) +
        "\n\nPROJECT CONVERSATION:\n" + _project_history_for_prompt(body.messages)
    )

    try:
        response = client.responses.create(
            model=SOL_MODEL,
            reasoning={"effort": "low"},
            instructions=prompt,
            input=[{"role": "user", "content": "Return valid JSON only. Compile the shot screenplay now.\n" + evidence}],
            store=False,
            max_output_tokens=6000,
        )
        raw = _clean_json_text(response.output_text or "")
        decoded = json.loads(raw)
        brief = ProjectBrief.model_validate(decoded.get("project_brief", {}))
        screenplay = ScreenplaySummary.model_validate(decoded.get("screenplay", {}))
        if not screenplay.shots:
            raise ValueError("模型没有生成镜头")

        normalized_shots = [
            ScreenplayShot(number=index, visual=shot.visual.strip())
            for index, shot in enumerate(screenplay.shots, start=1)
            if shot.visual.strip()
        ]
        if not normalized_shots:
            raise ValueError("模型没有生成有效镜头")
        screenplay = ScreenplaySummary(title=screenplay.title.strip(), shots=normalized_shots)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="镜头剧本整理失败：" + str(exc)) from exc

    return {
        "ok": True,
        "model": SOL_MODEL,
        "project_brief": brief.model_dump(),
        "screenplay": screenplay.model_dump(),
        "story_revision": (body.story_state.revision if body.story_state else 0),
    }


class ProjectScreenplaySyncRequest(BaseModel):
    previous_screenplay: ScreenplaySummary
    edited_screenplay: ScreenplaySummary
    project_brief: ProjectBrief | None = None
    story_state: StoryState | None = None


@app.post("/once/project/sync-screenplay")
def sync_edited_screenplay(body: ProjectScreenplaySyncRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.edited_screenplay.shots:
        raise HTTPException(status_code=400, detail="编辑后的镜头剧本为空")

    # The user's edited screenplay is authoritative. This pass is intentionally cheap:
    # it does not rewrite the visible screenplay, it only reads what changed and refreshes
    # the hidden brief consumed by realtime voice and image models.
    sync_prompt = (
        "You are Once's PROJECT CONTEXT SYNCHRONIZER. "
        "The user manually edited a shot screenplay. Compare BEFORE and AFTER carefully. "
        "Treat AFTER as authoritative user intent. Do not rewrite, improve, critique, or add shots. "
        "Do not silently invent canon. Your only job is to understand the exact changes and update the hidden project brief so downstream voice and image models follow the edited screenplay. "
        "Preserve established story facts unless the user's manual edits clearly override the presentation of a shot. "
        "Return valid JSON only with exactly this shape: "
        "{project_brief:{project_summary:string,creative_intent:string,current_story_position:string,visual_direction:[string],characters:[string],scene_goals:[string],continuity_constraints:[string],emotional_targets:[string],open_decisions:[string],voice_context:string,image_context:string},change_summary:[string]}. "
        "change_summary should be short factual notes about what the user changed, with no advice."
    )

    payload = {
        "before": body.previous_screenplay.model_dump(),
        "after": body.edited_screenplay.model_dump(),
        "existing_project_brief": body.project_brief.model_dump() if body.project_brief else {},
        "story_state": body.story_state.model_dump() if body.story_state else {},
    }

    try:
        response = client.responses.create(
            model=CONTEXT_MODEL,
            reasoning={"effort": "none"},
            instructions=sync_prompt,
            input=[{
                "role": "user",
                "content": "Return valid JSON only. Read the user's screenplay edits and synchronize the hidden project context.\n" + json.dumps(payload, ensure_ascii=False),
            }],
            store=False,
            max_output_tokens=2200,
        )
        decoded = json.loads(_clean_json_text(response.output_text or ""))
        brief = ProjectBrief.model_validate(decoded.get("project_brief", {}))
        change_summary = decoded.get("change_summary") or []
        if not isinstance(change_summary, list):
            change_summary = []
        change_summary = [trim_text(str(item), 300) for item in change_summary[:20] if str(item).strip()]
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI 同步剧本修改失败：" + str(exc)) from exc

    # Return the user's screenplay unchanged; the backend never gets to 'correct' manual edits.
    return {
        "ok": True,
        "model": CONTEXT_MODEL,
        "screenplay": body.edited_screenplay.model_dump(),
        "project_brief": brief.model_dump(),
        "change_summary": change_summary,
    }


# ---------- GPT Image 2.5 ----------

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32000)
    project_brief: dict | None = None
    reference_image_base64: str | None = None
    mask_image_base64: str | None = None
    pass_type: Literal["blocking", "pose", "shape", "environment", "style_draft", "finish"] = "blocking"
    size: str = "1536x1024"
    quality: Literal["low", "medium", "high", "xhigh", "max", "auto"] = "auto"
    output_format: Literal["png", "jpeg", "webp"] = "png"


def _validated_image_size(size: str) -> str:
    if size == "auto":
        return size
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        raise HTTPException(status_code=400, detail="图片尺寸格式不正确")
    width, height = int(match.group(1)), int(match.group(2))
    if width % 16 or height % 16 or width > 3840 or height > 3840:
        raise HTTPException(status_code=400, detail="图片宽高需要是16的倍数且不超过3840")
    ratio = max(width / height, height / width)
    pixels = width * height
    if ratio > 3 or pixels < 655360 or pixels > 8294400:
        raise HTTPException(status_code=400, detail="图片比例或总像素超出模型范围")
    return size


def _decode_image(value: str | None, label: str) -> bytes | None:
    if not value:
        return None
    try:
        raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        return base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{label}不是有效的Base64图片") from exc


def _image_prompt(body: ImageRequest, stage: str) -> str:
    brief = ""
    if body.project_brief:
        brief = trim_text(json.dumps(body.project_brief, ensure_ascii=False), 14000)

    pass_rules = {
        "blocking": (
            "BLOCKING PASS ONLY. Produce an intentionally incomplete structural sketch, not a finished picture. "
            "Use sparse monochrome or near-monochrome lines, simple mannequin/stick-figure or silhouette placement, basic horizon/camera guides, and motion arrows if useful. "
            "Do NOT add a rendered background, detailed costume, polished face, lighting, materials, atmospheric effects, or full color unless the user explicitly asked for that exact element. "
            "The purpose is only to establish camera, framing, character count, relative positions, scale, facing direction, and movement."
        ),
        "pose": (
            "POSE PASS ONLY. Work from the existing reference and improve only gesture, body direction, weight, action clarity, spacing, and motion. "
            "Preserve camera and staging. Keep the image visibly rough and sketch-like. Do not add a finished background, detailed costume, or polish unless explicitly requested."
        ),
        "shape": (
            "CHARACTER SHAPE PASS ONLY. Preserve the locked camera, staging, and poses. Add only readable character silhouettes, clothing masses, hair shapes, props, and large design forms requested by the user. "
            "Keep linework draft-like; do not silently add environment detail or final lighting."
        ),
        "environment": (
            "ENVIRONMENT PASS ONLY. Preserve all established people, poses, camera, scale, and spatial relationships. Add only the requested environment/background elements around them. "
            "Do not redesign characters or move them unless the user explicitly asks."
        ),
        "style_draft": (
            "STYLE DRAFT PASS ONLY. Preserve all established structure. Translate the existing rough image into Once's mandatory 2D action-animation visual language while keeping it visibly unfinished and easy to revise. "
            "Do not turn it into a polished final illustration."
        ),
        "finish": (
            "FINISH PASS. This pass is allowed only after the user has explicitly approved the established shot and asked to refine/finish it. "
            "Preserve camera, composition, pose, character identity, scene layout, and all locked decisions. Improve linework, cel shading, color consistency, lighting and finish without redesigning or adding new story content."
        ),
    }
    pass_instruction = pass_rules.get(body.pass_type, pass_rules["blocking"])

    if stage == "draft":
        stage_instruction = (
            "This is Once's INCREMENTAL DRAFT ENGINE. Never try to complete the whole image in one pass. "
            "Only perform the current requested pass and leave everything else intentionally unresolved. "
            "If a reference image exists, treat it as the user's current working state: preserve all unmentioned parts and change only what the user asked to change. "
            + pass_instruction
        )
    else:
        stage_instruction = (
            "This is Once's FINAL/REFINEMENT ENGINE. It must not invent a new composition. "
            "Use the reference as the source of truth and preserve all previously approved decisions. "
            + pass_instruction
        )

    return (
        ONCE_VISUAL_BASELINE + "\n\n" + stage_instruction +
        "\n\nNON-NEGOTIABLE CO-CREATION RULE: user decisions outrank model taste. Do not improve, fill, beautify, or complete parts the user did not ask you to touch." +
        ("\n\nPROJECT BRIEF:\n" + brief if brief else "") +
        "\n\nCURRENT PASS: " + body.pass_type +
        "\n\nUSER IMAGE REQUEST:\n" + body.prompt
    )


def _openai_image_request(model: str, body: ImageRequest, stage: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    size = _validated_image_size(body.size)
    prompt = _image_prompt(body, stage)
    reference = _decode_image(body.reference_image_base64, "参考图")
    mask = _decode_image(body.mask_image_base64, "蒙版")
    headers = {"Authorization": "Bearer " + api_key}

    try:
        with httpx.Client(timeout=180.0) as http:
            if reference is None:
                response = http.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={**headers, "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "size": size,
                        "quality": body.quality,
                        "output_format": body.output_format,
                        "background": "auto",
                        "n": 1,
                    },
                )
            else:
                files: list[tuple[str, tuple[str, bytes, str]]] = [
                    ("image", ("reference.png", reference, "image/png")),
                ]
                if mask is not None:
                    files.append(("mask", ("mask.png", mask, "image/png")))
                response = http.post(
                    "https://api.openai.com/v1/images/edits",
                    headers=headers,
                    data={
                        "model": model,
                        "prompt": prompt,
                        "size": size,
                        "quality": body.quality,
                        "output_format": body.output_format,
                        "background": "auto",
                    },
                    files=files,
                )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="图像模型连接失败：" + str(exc)) from exc

    if response.status_code < 200 or response.status_code >= 300:
        try:
            message = response.json().get("error", {}).get("message", response.text)
        except Exception:
            message = response.text
        raise HTTPException(status_code=502, detail=f"图像模型请求失败：{message}")

    payload = response.json()
    data = payload.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise HTTPException(status_code=502, detail="图像模型没有返回图片")
    item = data[0]
    return {
        "ok": True,
        "model": model,
        "stage": stage,
        "image_base64": item["b64_json"],
        "revised_prompt": item.get("revised_prompt"),
        "output_format": body.output_format,
        "size": size,
    }


@app.post("/once/image/draft")
def image_draft(body: ImageRequest):
    # Flare is intentionally used for fast structural drafts.
    if body.quality == "auto":
        body.quality = "low"
    return _openai_image_request(DRAFT_IMAGE_MODEL, body, "draft")


@app.post("/once/image/render")
def image_render(body: ImageRequest):
    # Sunburst is reserved for precise editing / final-quality passes.
    if body.quality == "auto":
        body.quality = "high"
    return _openai_image_request(RENDER_IMAGE_MODEL, body, "render")


# ---------- GPT-Live 1 relay ----------

LIVE_PROMPT = (
    "You are Once in realtime voice mode, a calm and sharp creative partner physically working beside the user on an original 2D animation shot. "
    "The user is the director. The user owns camera structure, staging, character positions, action direction, pacing, and what gets added next. Your job is to help make those decisions visible step by step, not to finish the shot for them. "
    "Speak naturally and briefly. Listen before taking over. Have opinions when useful, but suggestions are only suggestions until the user accepts them. "
    "CO-CREATION DEFAULT: never jump from an idea to a complete image. Build in small passes: blocking/staging -> pose/action -> character shapes -> environment -> style draft -> finish. A pass may deliberately contain only lines, circles, stick figures, silhouettes, arrows, or three rough character positions with no background. Incompleteness is correct. "
    "Before the FIRST visual action in a shot, or before advancing to a new pass, briefly say exactly what you plan to add and ask for the user's confirmation. Do not call a visual tool in that same turn unless the user already explicitly approved that exact step (for example: '就这样画', '开始', '按这个来'). "
    "Within an already approved pass, direct corrections such as '男主再往右一点' or 'A再靠后' may be executed immediately without repeatedly asking. "
    "For instant structural marks such as a circle, line, arrow, box, rough person position, or simple staging layout, use the fast canvas blocking action instead of an image model. "
    "For image-model work, first passes must stay rough and incomplete. Never add a detailed background, full costume, polished lighting, or complete render unless the user explicitly asks for that stage. "
    "Never use the final render action for a blank shot. Only use it after a visual has already been built progressively and the user explicitly asks to finish/refine/render it. "
    "MANDATORY visual direction whenever an image model is used: original hand-drawn 2D action-animation look, bold black contours, slightly rough expressive lines, flat simplified cel shading, high-contrast graphic shapes, dynamic action posing and motion accents; no photorealism, no 3D/CG, no glossy cinematic concept-art look. "
    "When the user is actively drawing or revising a shot, talk about the current creative problem rather than lecturing about theory. "
    "Backchannel policy: short natural acknowledgements are allowed when useful; do not fill every silence. "
    "Interruption policy: stop cleanly when the user interrupts and follow the newest correction. "
    "Delegation policy: delegate when a visual action is actually needed. Never claim an image or canvas changed until the delegated tool result confirms it."
)

LIVE_BACKEND_PROMPT = (
    SYSTEM_PROMPT
    + "\n\nYou are the backend action planner for Once Live inside a 2D animation workspace. "
      "The user is the director; never seize authorship of composition or staging. The operating mode is progressive co-creation, not one-shot generation. "
      "The valid visual progression is blocking -> pose -> shape -> environment -> style_draft -> finish, but the user may pause, revise, or skip a stage explicitly. "
      "For circles, lines, arrows, boxes, rough character positions, or first-pass staging, prefer once_canvas_blocking because it is immediate and editable. "
      "Use once_generate_draft only for the single requested visual pass. Always set pass_type to the narrowest stage that matches the user's current approved request. "
      "Use once_render_image only for finish/refinement AFTER a prior visual exists AND the user's latest message explicitly asks to finish, polish, refine, or render. Never render a blank shot. "
      "Do not bundle multiple stages into one call. If the user asks only for three rough positions with no background, the tool prompt must explicitly say no background and no added detail. "
      "Preserve every unmentioned part of the current reference. Do not add background, costume detail, props, lighting, color, or new characters unless the user requested that exact addition. "
      "Before a first visual action or a stage advance, the realtime partner must have already obtained user confirmation. If that confirmation is absent or ambiguous, do not call a visual function; answer conversationally and ask the smallest useful confirmation question. "
      "The app supplies the current screenplay, selected shot, and current visual status as context. Keep every tool prompt concrete, literal, visual, and narrow. "
      "Never report success before the function result says it succeeded."
)

LIVE_VISUAL_TOOLS = [
    {
        "type": "function",
        "name": "once_canvas_blocking",
        "description": "Instantly place simple editable blocking marks on the current canvas. Use for circles, lines, arrows, boxes, rough stick-figure positions, movement paths, and first-pass staging. Prefer this over an image model whenever the user is still deciding structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "shot_number": {"type": "integer", "description": "Optional 1-based screenplay shot number to target."},
                "clear_existing": {"type": "boolean", "description": "Clear existing AI blocking marks before applying these marks. Use only when the user asks to restart/rebuild the blocking."},
                "marks": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["line", "arrow", "circle", "box", "stick_figure"]},
                            "x": {"type": "number", "minimum": 0, "maximum": 1, "description": "Normalized start/center X."},
                            "y": {"type": "number", "minimum": 0, "maximum": 1, "description": "Normalized start/center Y."},
                            "x2": {"type": "number", "minimum": 0, "maximum": 1, "description": "Normalized end X for line/arrow/box."},
                            "y2": {"type": "number", "minimum": 0, "maximum": 1, "description": "Normalized end Y for line/arrow/box."},
                            "size": {"type": "number", "minimum": 0.02, "maximum": 0.5, "description": "Normalized size/radius for circle or stick figure."}
                        },
                        "required": ["kind", "x", "y"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["marks"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "once_generate_draft",
        "description": "Make exactly one incremental rough visual pass while preserving all unmentioned structure. Never use as a one-shot finished-image generator.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Narrow literal instruction for only this pass. Explicitly state what must NOT be added when relevant."},
                "pass_type": {"type": "string", "enum": ["blocking", "pose", "shape", "environment", "style_draft"], "description": "The one incremental pass to perform."},
                "shot_number": {"type": "integer", "description": "Optional 1-based screenplay shot number to target."}
            },
            "required": ["prompt", "pass_type"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "once_render_image",
        "description": "Finish/refine an already established shot only after explicit user approval. Never use on a blank shot or to invent a new composition.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Concrete refinement instruction that preserves all approved structure."},
                "pass_type": {"type": "string", "enum": ["finish"], "description": "Must be finish."},
                "shot_number": {"type": "integer", "description": "Optional 1-based screenplay shot number to target."}
            },
            "required": ["prompt", "pass_type"],
            "additionalProperties": False
        }
    }
]


def _live_initial_input(project_context: str, history: list[dict]) -> list[dict]:
    items: list[dict] = []
    if project_context.strip():
        items.append({
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "CURRENT PROJECT CONTEXT:\n" + trim_text(project_context, 16000)}],
        })
    for item in history[-8:]:
        role = item.get("role")
        content = trim_text(str(item.get("content", "")), 2500)
        if role not in ("user", "assistant") or not content:
            continue
        content_type = "input_text" if role == "user" else "output_text"
        items.append({"type": "message", "role": role, "content": [{"type": content_type, "text": content}]})
    return items


@app.websocket("/once/live")
async def once_live(client_socket: WebSocket):
    await client_socket.accept()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        await client_socket.send_json({"type": "error", "message": "Render 缺少 OPENAI_API_KEY"})
        await client_socket.close(code=1011)
        return

    try:
        first = await client_socket.receive_text()
        start = json.loads(first)
        if start.get("type") != "start":
            raise ValueError("第一条消息必须是 start")
        project_context = str(start.get("project_context", ""))
        history = start.get("history") if isinstance(start.get("history"), list) else []
    except Exception as exc:
        await client_socket.send_json({"type": "error", "message": "语音启动参数错误：" + str(exc)})
        await client_socket.close(code=1003)
        return

    try:
        async with websockets.connect(
            "wss://api.openai.com/v1/live/sessions",
            additional_headers={"Authorization": "Bearer " + api_key},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as upstream:
            session_start = {
                "type": "session.start",
                "event_id": "once_live_start",
                "session": {
                    "model": LIVE_MODEL,
                    "instructions": LIVE_PROMPT,
                    "input": _live_initial_input(project_context, history),
                    "audio": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "output": {"voice": "marin"},
                    },
                    "delegation": {
                        "type": "responses",
                        "responses": {
                            "model": LUNA_MODEL,
                            "instructions": LIVE_BACKEND_PROMPT,
                            "tools": LIVE_VISUAL_TOOLS,
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                        },
                    },
                    "store": False,
                },
            }
            await upstream.send(json.dumps(session_start, ensure_ascii=False))

            # Do not start phone capture until OpenAI confirms the session is ready.
            while True:
                raw = await upstream.recv()
                event = json.loads(raw)
                event_type = event.get("type", "")
                if event_type == "session.started":
                    await client_socket.send_json({
                        "type": "ready",
                        "session_id": (event.get("session") or {}).get("id"),
                        "model": LIVE_MODEL,
                    })
                    break
                if event_type == "error":
                    await client_socket.send_json({"type": "error", "message": str(event.get("error") or event)})

            pending_visual_calls: dict[str, dict] = {}

            async def client_to_openai():
                try:
                    while True:
                        message = await client_socket.receive_text()
                        item = json.loads(message)
                        kind = item.get("type")
                        if kind == "audio" and item.get("data"):
                            await upstream.send(json.dumps({
                                "type": "session.input_audio.append",
                                "audio": item["data"],
                            }))
                        elif kind == "mute":
                            await upstream.send(json.dumps({"type": "session.input_audio.mute"}))
                        elif kind == "unmute":
                            await upstream.send(json.dumps({"type": "session.input_audio.unmute"}))
                        elif kind == "context" and item.get("content"):
                            await upstream.send(json.dumps({
                                "type": "session.thinking.append",
                                "event_id": "once_ui_context",
                                "content": trim_text(str(item["content"]), 12000),
                                "delegation_id": None,
                            }, ensure_ascii=False))
                        elif kind == "tool_result" and item.get("call_id"):
                            call_id = str(item["call_id"])
                            output = str(item.get("output") or '{"ok":false,"message":"missing tool result"}')
                            await upstream.send(json.dumps({
                                "type": "response.item.create",
                                "event_id": "once_tool_result_" + call_id[-12:],
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": output,
                                },
                            }, ensure_ascii=False))
                            pending_visual_calls.pop(call_id, None)
                            await client_socket.send_json({"type": "phase", "phase": "thinking"})
                            await upstream.send(json.dumps({
                                "type": "response.create",
                                "event_id": "once_continue_" + call_id[-12:],
                            }, ensure_ascii=False))
                        elif kind == "close":
                            await upstream.send(json.dumps({"type": "session.close"}))
                            return
                except (WebSocketDisconnect, asyncio.CancelledError):
                    return

            async def openai_to_client():
                try:
                    async for raw in upstream:
                        event = json.loads(raw)
                        kind = event.get("type", "")
                        if kind == "session.output_audio.delta":
                            await client_socket.send_json({"type": "audio", "data": event.get("delta", "")})
                        elif kind == "session.input_transcript.delta":
                            await client_socket.send_json({"type": "user_transcript", "delta": event.get("delta", "")})
                        elif kind == "session.output_transcript.delta":
                            await client_socket.send_json({"type": "assistant_transcript", "delta": event.get("delta", "")})
                        elif kind == "session.delegation.created":
                            await client_socket.send_json({"type": "phase", "phase": "thinking"})
                        elif kind == "response.event":
                            inner = event.get("event") if isinstance(event.get("event"), dict) else {}
                            inner_kind = inner.get("type", "")
                            delegation_id = str(event.get("delegation_id") or "")

                            if inner_kind == "response.output_item.done":
                                output_item = inner.get("item") if isinstance(inner.get("item"), dict) else {}
                                if output_item.get("type") == "function_call":
                                    name = str(output_item.get("name") or "")
                                    call_id = str(output_item.get("call_id") or "")
                                    if name in {"once_canvas_blocking", "once_generate_draft", "once_render_image"} and call_id:
                                        raw_arguments = output_item.get("arguments") or "{}"
                                        try:
                                            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                                        except Exception:
                                            arguments = {}
                                        if not isinstance(arguments, dict):
                                            arguments = {}
                                        pending_visual_calls[call_id] = {
                                            "delegation_id": delegation_id,
                                            "name": name,
                                        }
                                        await client_socket.send_json({"type": "phase", "phase": "acting"})
                                        await client_socket.send_json({
                                            "type": "tool_request",
                                            "call_id": call_id,
                                            "name": name,
                                            "arguments": arguments,
                                        })

                            elif inner_kind == "response.completed":
                                has_pending = any(
                                    meta.get("delegation_id") == delegation_id
                                    for meta in pending_visual_calls.values()
                                )
                                if not has_pending:
                                    await client_socket.send_json({"type": "phase", "phase": "listening"})
                        elif kind == "session.closed":
                            await client_socket.send_json({"type": "closed", "usage": event.get("usage")})
                            return
                        elif kind == "error":
                            await client_socket.send_json({"type": "error", "message": str(event.get("error") or event)})
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    return

            sender = asyncio.create_task(client_to_openai())
            receiver = asyncio.create_task(openai_to_client())
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception as exc:
        try:
            await client_socket.send_json({"type": "error", "message": "GPT-Live 连接失败：" + str(exc)})
        except Exception:
            pass
        try:
            await client_socket.close(code=1011)
        except Exception:
            pass
