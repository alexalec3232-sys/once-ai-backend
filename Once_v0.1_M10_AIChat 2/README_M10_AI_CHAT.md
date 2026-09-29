# Once v0.1 M10 — Real AI Chat

Current active flow remains:

Intro → name → portrait chat.

M10 changes:
- Chat is now connected to a real Render backend route: `POST /once/chat`.
- Backend calls OpenAI Responses API with `gpt-6-luna`.
- Conversation history (up to 40 recent messages) is sent so Once can continue the same discussion.
- User and assistant messages have separate bubbles.
- While waiting, the UI shows a small thinking indicator.
- A small gear button lets you save the Render Base URL on-device.
- No fake AI replies are generated locally.
- `Backend/` contains a ready-to-deploy FastAPI service for Render.

The frozen drawing-frame workspace remains under Docs/ColdStorage and is not compiled.
