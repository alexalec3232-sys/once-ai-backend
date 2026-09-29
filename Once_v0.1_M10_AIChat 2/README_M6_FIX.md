# Once v0.1 M6 Long Video Entry — Layout Fix

This patch fixes two usability problems in the long-video script entry screen:

- The keyboard no longer leaves the user trapped with the Start button hidden.
  - While typing, a compact floating bar appears above the keyboard with `收起键盘` and `开始`.
  - The editor compresses slightly while focused to preserve usable space.
  - Tapping Start dismisses the keyboard before transitioning.
- `草稿剧本` now always returns to its top when the generated draft changes, so `核心想法` and `当前整理` remain visible instead of leaving the panel scrolled into the middle.

No AI backend behavior was changed in this patch.
Bundle Identifier remains `com.alex.once0929`.
