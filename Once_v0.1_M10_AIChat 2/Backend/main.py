    previous: StoryState | None,
    messages: list[ChatMessage],
    assistant_reply: str,
) -> tuple[StoryState, bool]:
    previous_state = previous or StoryState()
    evidence = [
        {"role": m.role, "content": m.content}
        for m in messages[-18:]
    ]
    evidence.append({"role": "assistant", "content": assistant_reply})

    payload = {
        "existing_story_state": previous_state.model_dump(),
        "recent_conversation": evidence,
        "latest_user_message": next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        ),
    }

    # JSON mode requires an explicit JSON instruction in the actual input message.
    # Keep the entire extraction path fault-tolerant: chat must still succeed even if
    # the memory extractor fails for any reason.
    try:
        response = client.responses.create(
            model=CONTEXT_MODEL,
            instructions=CONTEXT_EXTRACTOR_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Return valid JSON only. Update the story state from the following JSON evidence.\n"
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                }
            ],
            text={"format": {"type": "json_object"}},
            store=False,
        )

        raw_text = (response.output_text or "").strip()
        if not raw_text:
            return previous_state, False

        decoded = json.loads(raw_text)
        updated = StoryState.model_validate(decoded.get("story_state", {}))
        updated.revision = max(previous_state.revision + 1, updated.revision)
        screenplay_requested = bool(decoded.get("screenplay_requested", False))
        return updated, screenplay_requested
    except Exception:
        # Story memory is secondary. Never turn a memory-extraction failure into
        # a failed user chat response.
        return previous_state, False


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": CHAT_MODEL,
        "context_model": CONTEXT_MODEL,
        "story_context": "v1.1",
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
        + _state_for_prompt(body.story_state)
    )

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT + _name_context(body.nickname) + story_context,
            input=input_items,
            store=False,
        )
        reply = (response.output_text or "").strip()
        if not reply:
            raise RuntimeError("OpenAI returned an empty response")

        updated_state, screenplay_requested = _extract_story_state(
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 请求失败：{exc}") from exc
