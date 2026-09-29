# Once v0.1 M11 — Story Brain

This build upgrades Once from a plain chat model into the first persistent story-discussion brain.

## What changed

### 1. Luna prompt rebuilt
Once is now instructed to:
- amplify the user's idea rather than replace it;
- avoid default praise / cheerleading;
- detect emotional center, character logic, pacing, causality, setup/payoff and unresolved threads;
- disagree when a real story problem exists;
- stay on the user's main story path when disagreeing;
- prefer the smallest useful repair over rewriting the story;
- provide short example dialogue / stage direction when helpful;
- keep examples as proposals, never silently convert them into canon;
- distinguish canon, interpretation and proposal;
- use the whole story when relevant, not only the latest sentence.

### 2. Story Context v1
Each `/once/chat` response now also updates a hidden `story_state` containing:
- project summary
- story phase
- current focus
- canon facts
- characters
- locations
- world rules
- relationships
- active story threads
- emotional arcs
- discarded / revised ideas
- open questions

The extractor has a hard rule: assistant suggestions are not canon unless the user clearly accepts them.

### 3. Corrections override older facts
If the user changes an established idea, the newest clear user statement wins.
Useful rejected/revised ideas can be kept in `discarded_or_revised` so they do not unexpectedly return later.

### 4. Memory survives Render sleep
The iOS app stores the returned story state locally in UserDefaults and sends it back with future turns.
Therefore Render's free-instance sleep/restart does not erase the story memory.

Current key:
`once.storyState.v1`

This is global for the current single-story MVP. When project files return, this should become per-project storage.

### 5. Screenplay intent detection is reserved
The extractor already returns `screenplay_requested=true` when the user clearly says things such as:
- OK，剧本OK了
- 就这样，变成剧本吧
- 可以开始写剧本了

M11 does not yet render the editable screenplay card. That is the next layer.

## Backend update
Your existing Render Root Directory is still:

`Once_v0.1_M10_AIChat 2/Backend`

Replace the deployed `Backend/main.py` with M11's `Backend/main.py` and commit to GitHub. Render should auto-deploy.

No new environment variable is required.

Optional:
- `ONCE_CHAT_MODEL=gpt-6-luna`
- `ONCE_CONTEXT_MODEL=gpt-6-luna`

## Important runtime detail
Each user turn currently performs:
1. one Luna call for the natural reply;
2. one Luna call for Story Context extraction.

This is intentionally simple and reliable for the MVP. Later we can optimize latency/cost after behavior is stable.
