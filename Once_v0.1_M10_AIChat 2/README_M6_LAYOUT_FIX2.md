# Once M6 Long Video Layout Fix 2

This pass fixes the long-video preparation page for iPhone landscape.

## Fixed
- The entire page no longer grows beyond the landscape viewport.
- Left `草稿剧本` header is pinned; only the script body scrolls.
- Draft script automatically returns to the top after regeneration.
- Right-side back button and intro remain visible.
- The story editor receives a dynamic height based on actual landscape height instead of a fixed 220–300pt minimum.
- The bottom `开始` button is permanently pinned inside the visible right panel.
- When the keyboard is visible, a floating `收起键盘 / 开始` bar sits just above it.
- The full page ignores keyboard-driven relayout so the screen itself does not get pulled vertically.

Bundle Identifier remains: `com.alex.once0929`
