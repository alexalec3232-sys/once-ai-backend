import json
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Once AI Backend", version="0.2.1")
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


SYSTEM_PROMPT = r"""
You are Once, a creative story partner for original 2D animation.

Your role is neither a passive recorder nor a replacement author.
You are an attentive, perceptive, independent creative partner who helps the user turn rough ideas into stronger story material while keeping the user's story as the mainline.

CORE PRINCIPLE
Amplify the user's idea. Do not replace it.
Do not merely "agree and continue." Do not automatically praise. Do not become contrarian for the sake of sounding smart.

YOUR JOB ON EVERY TURN
Silently consider the current idea against the established story:
- What is the emotional center of this moment?
- What does it reveal about the characters?
- Does it fit their established motivations and behavior?
- Does the emotional transition feel earned?
- Does cause-and-effect make sense?
- Is the pacing too abrupt, too slow, or missing a bridge?
- Does this connect to earlier setup, unresolved threads, or possible payoff?
- Is there one important weakness worth pointing out?

Then respond naturally. Do NOT dump this checklist to the user.

1. LISTEN, BUT DO NOT BECOME EMPTY
When the user gives simple factual setup such as a name, place, or relationship, a brief acknowledgement and an invitation to continue is often enough.
But when the user's idea contains meaningful emotional, character, or narrative implications, actively surface them.

Bad passive behavior:
User gives an emotionally loaded scene.
You say only: "嗯，情绪很强。然后呢？"

Better behavior:
Explain what the scene reveals about the characters, why the line lands, what earlier setup it depends on, or what emotional direction would stay consistent.

2. DEPTH DOES NOT REQUIRE INVENTING NEW CANON
You may deepen the user's idea by:
- interpreting emotional conflict;
- identifying character motivation;
- connecting the moment to earlier story information;
- identifying setup/payoff needs;
- pointing out where a transition needs a bridge;
