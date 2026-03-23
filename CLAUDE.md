# CLAUDE.md — Ravi's Data Science Portfolio

## Workflow Orchestration Rules

### Task Management
- Always start by reading `tasks/todo.md` for current priorities
- Update `tasks/todo.md` as tasks are completed
- Log decisions and learnings in `tasks/lessons.md`
- Break complex tasks into subtasks before starting

### Code Changes
- Read existing code before modifying
- Run tests after every change
- Commit after each logical unit of work
- Never push to main without explicit permission

### Research & Analysis
- Search codebase before asking questions
- Check existing patterns before introducing new ones
- Document assumptions in code comments

## Core Principles

### Quality Standards
- All code must have tests (target: 80%+ coverage)
- Use type hints on all function signatures
- Follow existing patterns in each project
- No hardcoded credentials or secrets

### Architecture Decisions
- Prefer composition over inheritance
- Keep modules focused and cohesive
- Use dependency injection for testability
- Configuration via YAML files, not code changes

### Communication
- Be concise in commit messages but descriptive
- Explain "why" not "what" in comments
- Flag risks and tradeoffs proactively

## Project Structure

```
DataScience/
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yml                # GenAI portfolio tests + lint
│   └── test-projects.yml     # Classic projects smoke tests
├── genai-portfolio/          # GenAI/enterprise data projects (240 tests)
│   ├── biller-integration-simulator/   # 53 tests
│   ├── payment-intelligence-agent/     # 101 tests
│   └── multi-source-data-integration/  # 86 tests
├── projects/                 # 33 classic DS/ML projects
│   ├── 01-machine-learning/  # 8 projects
│   ├── 02-deep-learning/     # 5 projects
│   ├── 03-data-analysis/     # 5 projects
│   ├── 04-nlp/               # 5 projects
│   ├── 05-data-engineering/  # 5 projects
│   └── 06-mlops-deployment/  # 5 projects
├── tasks/
│   ├── todo.md               # Current priorities
│   └── lessons.md            # Decisions & learnings
├── CLAUDE.md                 # This file
└── README.md                 # Portfolio overview
```

## Tech Stack Preferences
- Python 3.11+
- pytest for testing
- YAML for configuration
- Snowflake for cloud data warehouse
- Streamlit for data apps
- pandas/numpy for data manipulation
- structlog for logging
- GitHub Actions for CI/CD
