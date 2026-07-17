---
name: golden-rules
description: Felipe's GOLDEN RULES — the engineering contract governing every coding task. Read before planning or touching any code, and when asked about working standards.
---

# Golden rules (advisory layer)

Read `~/.dasbrowcoder/GOLDEN-RULES.md` (deployed verbatim from git) and
follow it literally. Highlights you must never violate:

- Verify before you declare: label every claim [REAL] / [TEST] / [UNVERIFIED].
- TDD first, 80%+ coverage. File ceiling ~900 lines. API-first (SPEC.md before code).
- No in-memory repos in production paths. No metered API keys, ever.
- When stuck: official docs, then ask Felipe. No solo research. Surface every wall.

This skill is the ADVISORY layer for conversation. The deterministic layer is
`golden_guard.py`, which gates every delegated coding run mechanically — a red guard
blocks the verdict, and you never override it.
