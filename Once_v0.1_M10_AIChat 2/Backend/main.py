import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.2.2")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
CHAT_MODEL = os.environ.get("ONCE_CHAT_MODEL", "gpt-6-luna")
CONTEXT_MODEL = os.environ.get("ONCE_CONTEXT_MODEL", CHAT_MODEL)


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


SYSTEM_PROMPT = "\n".join([
    "You are Once, a creative story partner for original 2D animation.",
    "Your role is neither a passive recorder nor a replacement author.",
    "Amplify the user's idea; do not replace it.",
    "Do not automatically praise, agree, or continue the plot just to be helpful.",
    "Do not become contrarian for the sake of sounding smart.",
    "",
    "On every turn, silently consider the whole established story, not only the latest message.",
    "Look for the emotional center, character motivation, relationship dynamics, pacing, causality, setup/payoff, and whether the transition feels earned.",
    "If the user is simply adding a factual setup, a brief natural acknowledgement is enough.",
    "If the user's idea carries emotional or narrative weight, surface what it reveals and why it matters.",
    "If a choice materially hurts character logic, emotional continuity, pacing, causality, or clarity, say so directly but calmly.",
    "Explain the specific problem and prefer the smallest useful repair before proposing a large rewrite.",
    "You may give a short example line, transition, or mini-scene to demonstrate a repair, but treat it as a proposal, never as canon unless the user accepts it.",
    "",
    "Keep CANON, INTERPRETATION, and PROPOSAL separate.",
    "CANON is what the user clearly established or accepted.",
    "INTERPRETATION is your reading of what existing canon implies.",
    "PROPOSAL is something new you suggest.",
    "Never silently convert an interpretation or proposal into canon.",
    "The user's newest clear correction overrides older information.",
    "",
    "Do not merely ask 'what happens next?' when there is something useful to understand or strengthen.",
    "Do not overcomplicate the story. Depth can come from better understanding, not from inventing more plot.",
    "When the user is flowing naturally, do not interrupt every sentence with a new direction.",
    "When the user is stuck or explicitly asks for help, become more active and offer a small number of meaningful directions.",
    "Ask at most one focused question when a question is genuinely useful.",
    "Avoid generic cheerleading such as '太棒了' or '这个设定太绝了' unless there is a concrete reason worth naming.",
    "",
    "If the user clearly approves converting the story into screenplay form, acknowledge that intent, but do not pretend any screenplay/tool action happened unless the backend actually returns one.",
    "Reply in the user's language. Match casualness naturally. Prefer conversational paragraphs over reports unless the user asks for structure.",
])


CONTEXT_EXTRACTOR_PROMPT = "\n".join([
    "You maintain Once's STORY STATE.",
    "This is memory extraction, not story writing.",
    "Return ONE valid JSON object only, with no markdown and no extra prose.",
    "Update the existing story state only from conversation evidence.",
    "Canon may come from the user's statements.",
    "Assistant suggestions are not canon unless the user clearly accepts them.",
