# Lessons & Decisions Log

## 2026-02-23 — GenAI Portfolio Initial Build

### Decisions
- Used YAML-driven configuration across all three projects for consistency
- Chose pytest over unittest for cleaner test syntax and fixtures
- Used structlog over stdlib logging for structured JSON output
- Built all projects to run in demo mode with synthetic data (no external dependencies required)

### Lessons
- Always add `.gitignore` before first commit to avoid tracking __pycache__
- Deep glob patterns (`**/__pycache__`) don't always catch all nesting levels — use `git ls-files | grep` to verify
- Output artifacts from demo runs should be gitignored from the start

## 2026-02-23 — Repo Updates: README, CI/CD, Documentation

### Decisions
- Added GenAI portfolio section at the top of main README (above existing 33 projects) to highlight newer, more complex work
- CI/CD split into two workflows: `ci.yml` (genai tests + lint on every push) and `test-projects.yml` (classic projects, only on changes to `projects/`)
- Deep learning projects use import-check only in CI (TF/PyTorch too heavy for GitHub Actions runners)
- Kept `MPLBACKEND=Agg` in CI to prevent matplotlib display issues on headless runners
- Added `claude/**` branch pattern to CI triggers to cover current branching convention

### Lessons
- Matrix strategy with `fail-fast: false` gives better visibility — all three project test suites report independently
- Path-filtered workflows prevent unnecessary CI runs when only one area of the repo changes
- Always update project counts in README when adding new project categories (33 → 36)

## 2026-07-02 — Review of Shakeeb's pattern-strategy-bot bundle

### Decisions
- Keep his trading bot in a separate PRIVATE repo; do not merge into this public portfolio (live-ops state, secrets history, different lifecycle)
- Review was static-only: externally-shared code is not executed in the review environment
- Any auto-trading from Discord signals must route through the bot's existing risk gates (caps, drawdown halt, kill-switch), stay in paper mode, and use author allow-lists

### Lessons
- "Shared for the team" bundles can ship live credentials on purpose — always check config files AND docs (his secrets were duplicated in a markdown file) before pushing anywhere
- A commented-out .gitignore entry is as good as no entry: `git add -A` would have leaked three tokens
- Shadow-mode tracking (log what rejected signals would have done) is a cheap, powerful pattern for tuning filter thresholds with data instead of intuition
