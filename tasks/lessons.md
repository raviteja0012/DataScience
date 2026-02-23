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
