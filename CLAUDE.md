@AGENTS.md

## Claude Code only
- Durable project knowledge (rules, decisions, current state) goes in AGENTS.md or docs/, not auto-memory, so every agent sees it. Keep auto-memory for notes that only matter to Claude Code.
- `.claude/settings.local.json` is gitignored and local; this repo is public, so never move its allow rules (some embed coordinates) into a tracked `.claude/settings.json`.
