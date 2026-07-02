# TODO — Current Priorities

## Completed
- [x] Biller Integration Simulator (53 tests passing)
- [x] Payment Intelligence Agent (101 tests passing)
- [x] Multi-Source Data Integration Framework (86 tests passing)
- [x] Clean up __pycache__ and output artifacts
- [x] Push to claude/integrate-data-projects-ZBvV4
- [x] Add GitHub Actions CI/CD pipeline (ci.yml + test-projects.yml)
- [x] Add genai-portfolio/README.md with project summaries
- [x] Update main README.md with GenAI portfolio section (36 projects total)
- [x] Update CLAUDE.md project structure to reflect full repo layout
- [x] Verify all 240 tests pass (53 + 101 + 86)

## In Progress — Shakeeb's pattern-strategy-bot
- [x] Review the shared bundle (trading-bot / trading-dashboard / strategy-study) → `reviews/pattern-strategy-bot-review.md`
- [ ] BLOCKER (Shakeeb): rotate Discord webhook, Telegram token, X bearer token; remove secrets from tracked files
- [ ] Create private consolidated repo + CI (do NOT merge into this portfolio repo)
- [ ] Build report-upload pipeline (parse MT5 reports → track diffs → notify) — ~1–2 days
- [ ] Build Discord recommendation consumer (paper mode, allow-listed authors, via risk gates) — ~2–4 days

## Backlog
- [ ] Add architecture diagrams to each project (Mermaid or ASCII)
- [ ] Add coverage reporting to CI pipeline (pytest-cov + codecov)
- [ ] Set up pre-commit hooks (black, ruff, mypy)
