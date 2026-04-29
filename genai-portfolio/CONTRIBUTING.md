# Contributing to the GenAI Portfolio

Thanks for your interest in contributing. This document describes how to set up a local development environment, the code-quality standards we apply, and the workflow for proposing and merging changes.

Maintainer: Ravi Potluru

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Style](#code-style)
3. [Testing Requirements](#testing-requirements)
4. [Adding a New Project](#adding-a-new-project)
5. [Commit Message Conventions](#commit-message-conventions)
6. [Pull Request Process](#pull-request-process)

---

## Development Environment Setup

### Prerequisites

- Python 3.11 or newer
- `make` (GNU Make 3.81+)
- `git`
- Optional: Docker 24+ if you plan to work on the containerized stacks

### One-shot install

From the `genai-portfolio/` root:

```bash
# Create and activate a virtual environment for the portfolio
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies for all 3 projects
make install

# Install lint and format tooling
pip install flake8 black isort pytest-cov
```

Each project also stands alone -- you can `cd` into any of `biller-integration-simulator/`, `payment-intelligence-agent/`, or `multi-source-data-integration/` and run `pip install -r requirements.txt && pip install -e .` to work on a single project.

### Sanity check

```bash
make test     # Should report 240 tests passing across the 3 projects
make lint     # Should report no flake8 errors
```

If both commands pass, you are ready to start contributing.

---

## Code Style

We enforce a small, consistent set of standards across every project.

### Formatting

- **`black`** is the source of truth for formatting. Line length is **120**.
- **`isort`** organizes imports using the `black` profile.
- Run `make format` before committing.

### Linting

- **`flake8`** must pass with the portfolio's shared config (see `.flake8`).
- Configured ignores: `E501,W503,E203,E266,E402` (max line length, line break before binary operator, whitespace before colon, block comment style, module-level import not at top).
- Max line length: **120**.

### Type hints

- **All public function and method signatures must include type hints.**
- Use `from __future__ import annotations` if you need forward references.
- Prefer `typing` primitives (`list[str]`, `dict[str, int]`) on Python 3.11+.
- Optional/None values use `X | None` rather than `Optional[X]`.

### Naming and structure

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_<module_under_test>.py`

### Documentation

- Public functions and classes need a short docstring describing intent, parameters, return value, and any raised exceptions.
- Use Google-style docstrings.
- Comments explain **why**, not **what**. The code already shows what.

---

## Testing Requirements

### Coverage

- **Target: 80%+ line coverage** for new code in any project.
- All new public functions, classes, and CLI entry points need accompanying tests.
- Bug fixes need a regression test that fails before the fix and passes after.

### Framework

- We use **`pytest`** exclusively. No `unittest`-style classes for new code.
- Use fixtures for setup/teardown -- avoid module-level state in test files.
- Mark slow tests with `@pytest.mark.slow` and integration tests with `@pytest.mark.integration` (see `pytest.ini` for registered markers).

### Running tests

```bash
# All projects, all tests
make test

# A single project
make test-biller
make test-payment
make test-integration

# A single test file
cd biller-integration-simulator
pytest tests/test_settlement.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### What "good tests" look like in this portfolio

- **Deterministic.** No reliance on wall-clock time, network access, or random seeds (use fixed seeds when randomness is needed).
- **Focused.** One behavior per test. Long parametrize lists are fine; long test bodies usually are not.
- **Real data shapes.** Use realistic fixture data so tests catch schema regressions.

---

## Adding a New Project

To add a fourth project to the portfolio, create a new top-level directory under `genai-portfolio/` with this structure:

```
my-new-project/
├── README.md                 # Required -- see below
├── requirements.txt          # Pinned (==) for reproducibility
├── setup.py                  # Editable install support: `pip install -e .`
├── config/                   # YAML configuration files
│   └── default.yaml
├── src/                      # Importable Python package
│   └── my_new_project/
│       └── __init__.py
├── tests/                    # pytest tests; mirrors src/ layout
│   └── __init__.py
├── examples/  or  demo/      # Runnable demo scripts
├── docs/                     # Architecture notes, ADRs, diagrams
├── Dockerfile                # Optional but encouraged
└── docker-compose.yml        # Optional but encouraged
```

### README requirements

Every project README must contain:

1. **One-paragraph elevator pitch** -- what problem this solves and for whom.
2. **Architecture highlights** -- 4-6 bullets covering the most interesting design choices.
3. **Quick start** -- exact commands to install and run a demo end-to-end.
4. **Testing** -- how to run the test suite and what coverage to expect.
5. **Configuration reference** -- documenting any YAML keys the user is expected to edit.
6. **Project structure tree** -- a `tree`-style listing of `src/` and `tests/`.

### Portfolio-level integration

After creating the new project:

1. Add it to the `PROJECTS` variable in the root `Makefile`.
2. Add it to `testpaths` in the root `pytest.ini`.
3. Add a row to the "Projects at a Glance" table in `README.md`.
4. Add a matrix entry in `.github/workflows/ci.yml`.
5. Update the architecture diagram in `ARCHITECTURE.md` if the new project is part of the conceptual flow.

---

## Commit Message Conventions

We follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type       | When to use                                                  |
|------------|--------------------------------------------------------------|
| `feat`     | A new user-facing capability                                 |
| `fix`      | A bug fix                                                    |
| `refactor` | Internal change with no behavior difference                  |
| `perf`     | Performance improvement                                      |
| `test`     | Adding or improving tests only                               |
| `docs`     | Documentation only                                           |
| `build`    | Changes to build, packaging, or dependency manifests         |
| `ci`       | Changes to GitHub Actions or other CI                        |
| `chore`    | Maintenance tasks that don't fit elsewhere                   |

### Scopes

Use the project directory name as the scope when changes are project-specific:

- `biller` for `biller-integration-simulator`
- `payment` for `payment-intelligence-agent`
- `integration` for `multi-source-data-integration`
- Omit the scope when the change is portfolio-level.

### Examples

```
feat(payment): add Z-score anomaly detector with configurable threshold

fix(biller): correct settlement reconciliation rounding for fractional cents

docs: document multi-project integration in ARCHITECTURE.md

ci: add flake8 step to genai-portfolio CI pipeline
```

### Subject line rules

- Lowercase, imperative mood ("add", not "added" or "adds").
- No trailing period.
- Maximum 72 characters.

### Body and footer

- Body explains motivation and high-level approach. Wrap at 80 columns.
- Footer carries `BREAKING CHANGE:` notes and issue references (`Closes #42`).

---

## Pull Request Process

1. **Fork and branch.** Create a feature branch from `main`. Branch names follow `feat/<short-description>`, `fix/<short-description>`, or `docs/<short-description>`.
2. **Write code and tests.** Make focused, atomic commits. Keep PRs under ~400 lines of diff when possible.
3. **Run the local quality gate before pushing:**
   ```bash
   make format
   make lint
   make test
   ```
4. **Push and open a PR** with:
   - A clear title following Conventional Commits.
   - A description that covers: motivation, approach, testing performed, and screenshots or sample output if user-facing.
   - A linked issue when applicable.
5. **CI must pass.** All matrix jobs in `.github/workflows/ci.yml` (tests for each of the 3 projects + lint) must be green.
6. **Code review.** At least one approving review is required. Address review feedback in additional commits -- do not force-push during review.
7. **Merge.** Squash-and-merge is preferred to keep `main` history linear. The squash commit message should follow Conventional Commits.

### What gets a PR rejected

- Failing tests or lint.
- New code without tests.
- Hardcoded credentials, API keys, or production endpoints.
- Inheritance hierarchies where composition would be clearer.
- Configuration changes hidden inside Python code instead of YAML.
- Breaking changes without a `BREAKING CHANGE:` footer and migration notes.

---

Thanks for contributing.
