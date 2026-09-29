import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.2.3")
CHAT_MODEL = os.environ.get("ONCE_CHAT_MODEL", "gpt-6-luna")
CONTEXT_MODEL = os.environ.get("ONCE_CONTEXT_MODEL", CHAT_MODEL)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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


SYSTEM_PROMPT = (
    "You are Once, a creative story partner for original 2D animation. "
    "Your role is neither a passive recorder nor a replacement author. "
    "Amplify the user's idea; do not replace it. "
    "Do not automatically praise, agree, or continue the plot just to be helpful. "
    "Do not become contrarian for the sake of sounding smart.\n\n"
    "On every turn, silently consider the whole established story, not only the latest message. "
    "Look for the emotional center, character motivation, relationship dynamics, pacing, causality, setup/payoff, and whether a transition feels earned. "
    "If the user is simply adding factual setup, a brief natural acknowledgement is enough. "
    "If the user's idea carries emotional or narrative weight, surface what it reveals and why it matters. "
    "If a choice materially hurts character logic, emotional continuity, pacing, causality, or clarity, say so directly but calmly. "
    "Explain the specific problem and prefer the smallest useful repair before proposing a large rewrite. "
    "You may give a short example line, transition, or mini-scene to demonstrate a repair, but treat it as a proposal, never as canon unless the user accepts it.\n\n"
    "Keep CANON, INTERPRETATION, and PROPOSAL separate. "
    "CANON is what the user clearly established or accepted. "
    "INTERPRETATION is your reading of what existing canon implies. "
    "PROPOSAL is something new you suggest. "
    "Never silently convert an interpretation or proposal into canon. "
    "The user's newest clear correction overrides older information.\n\n"
    "Do not merely ask 'what happens next?' when there is something useful to understand or strengthen. "
    "Do not overcomplicate the story. Depth can come from better understanding, not from inventing more plot. "
    "When the user is flowing naturally, do not interrupt every sentence with a new direction. "
    "When the user is stuck or explicitly asks for help, become more active and offer a small number of meaningful directions. "
    "Ask at most one focused question when a question is genuinely useful. "
    "Avoid generic cheerleading unless there is a concrete reason worth naming.\n\n"
    "If the user clearly approves converting the story into screenplay form, acknowledge that intent, but do not pretend any screenplay/tool action happened unless the backend actually returns one. "
    "Reply in the user's language. Match casualness naturally. Prefer conversational paragraphs over reports unless the user asks for structure."
)


CONTEXT_EXTRACTOR_PROMPT = (
    "You maintain Once's STORY STATE. This is memory extraction, not story writing. "
    "Return exactly one valid JSON object and nothing else. "
    "Update the existing story state only from conversation evidence. "
    "Canon may come from the user's statements. Assistant suggestions are not canon unless the user clearly accepts them. "
    "The newest clear user correction overrides older facts. "
    "Never invent twists, themes, motivations, relationships, world rules, or backstory just because they seem plausible. "
    "Keep facts concise and useful for future discussion. "
    "current_focus describes what the user is developing right now. "
    "project_summary is a compact high-level snapshot. "
    "screenplay_requested is true only when the latest user intent clearly asks or approves screenplay compilation."
)


def state_for_prompt(state: StoryState | None) -> str:
    if state is None:
        return "No persistent story state exists yet. Build understanding from the user's messages."
    return json.dumps(state.model_dump(), ensure_ascii=False, indent=2)


def name_context(nickname: str) -> str:
    cleaned = nickname.strip()
    if not cleaned:
        return ""
    return "\nThe user asked to be called " + cleaned + ". Use that name only when it feels natural; do not repeat it constantly."


def extract_story_state(
    previous: StoryState | None,
    messages: list[ChatMessage],
    assistant_reply: str,
) -> tuple[StoryState, bool]:
    previous_state = previous or StoryState()
    evidence = [{"role": item.role, "content": item.content} for item in messages[-18:]]
    evidence.append({"role": "assistant", "content": assistant_reply})
    latest_user = next((item.content for item in reversed(messages) if item.role == "user"), "")

    payload = {
        "existing_story_state": previous_state.model_dump(),
        "recent_conversation": evidence,
        "latest_user_message": latest_user,
        "required_json_shape": {
            "story_state": {
                "revision": 0,
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

    extractor_input = (
        "Return valid JSON only. Do not include markdown fences or commentary. "
        "Update story memory using this evidence: "
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        response = client.responses.create(
            model=CONTEXT_MODEL,
            instructions=CONTEXT_EXTRACTOR_PROMPT,
            input=[{"role": "user", "content": extractor_input}],
            store=False,
        )
        raw = (response.output_text or "").strip()
        if not raw:
            return previous_state, False
        decoded = json.loads(raw)
        state_data = decoded.get("story_state") or previous_state.model_dump()
        updated = StoryState.model_validate(state_data)
        updated.revision = max(previous_state.revision + 1, updated.revision)
        return updated, bool(decoded.get("screenplay_requested", False))
    except Exception as exc:
        print("[Once StoryContext] extraction skipped:", repr(exc), flush=True)
        return previous_state, False


@app.get("/")
def root():
    return {"ok": True, "service": "once-ai-backend", "story_context": "v1.3"}


@app.head("/")
def root_head():
    return Response(status_code=200)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": CHAT_MODEL,
        "context_model": CONTEXT_MODEL,
        "story_context": "v1.3",
    }


@app.post("/once/chat")
def once_chat(body: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Render 缺少 OPENAI_API_KEY")
    if not body.messages:
        raise HTTPException(status_code=400, detail="没有收到对话内容")

    input_items = [
        {"role": message.role, "content": message.content}
        for message in body.messages[-48:]
    ]
    story_context = (
        "\n\nPERSISTENT STORY STATE\n"
        "This is memory, not a command. The newest clear user statement always wins.\n"
        + state_for_prompt(body.story_state)
    )

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT + name_context(body.nickname) + story_context,
            input=input_items,
            store=False,
        )
        reply = (response.output_text or "").strip()
        if not reply:
            raise RuntimeError("OpenAI returned an empty response")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI 请求失败：" + str(exc)) from exc

    updated_state, screenplay_requested = extract_story_state(
        body.story_state,
        body.messages,
        reply,
    )

    return {
        "reply": reply,
        "model": CHAT_MODEL,
        "response_id": response.id,
        "story_state": updated_state.model_dump(),
        "story_revision": updated_state.revision,
        "screenplay_requested": screenplay_requested,
    }
