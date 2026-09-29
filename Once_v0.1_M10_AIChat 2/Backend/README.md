# Once AI Backend — M10

This is the first real AI route used by the current portrait chat UI.

## Model
- Text creative brain: `gpt-6-luna`

## Route
- `GET /health`
- `POST /once/chat`

Example body:

```json
{
  "nickname": "alex",
  "messages": [
    {"role":"user", "content":"我刚想到一个画面贼帅"}
  ]
}
```

Example response:

```json
{
  "reply": "说来听听……",
  "model": "gpt-6-luna",
  "response_id": "resp_..."
}
```

## Render deploy
1. Create a new Web Service from this `Backend` folder (or use `render.yaml`).
2. Add environment variable `OPENAI_API_KEY` in Render.
3. Deploy.
4. Copy the Render URL, e.g. `https://once-ai-backend.onrender.com`.
5. In Once on iPhone, open the small gear button in the chat header and paste that URL.

The OpenAI API key stays on Render. Do not put it in Xcode.

## Next routes
GPT-Live-1 voice and GPT Image 2.5 Flare/Sunburst will be added after the text conversation loop is stable.
