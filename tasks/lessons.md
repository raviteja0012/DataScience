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
