# Review — Shakeeb's Pattern-Strategy Trading Bot (shared 2026-06-16)

**Reviewed:** `Pattern-Strategy-Bot_Team-Share_2026-06-16.zip` (3.9 MB) + `Pattern-Strategy-Scheduler.md`,
shared via Google Drive by shakeebahmed90@gmail.com.

**Review method:** static code review + docs review. The test suites were **not executed**
(externally-sourced code is not run in this environment); test claims below come from the
package's own validation reports.

---

## 1. What was actually shared

It is **not three separate repos** — it is one bundle, `pattern-strategy-bot/`, containing
three sub-projects plus docs and companion skills:

| Component | What it is | Size / notes |
|---|---|---|
| `trading-bot/` | Python FastAPI + StrategyEngine. MT5 price-action paper-trading bot (Double Top/Bottom, R2-retest entry, layered confirmation/risk gates, news agent, Telegram + Discord notifiers) | ~6,000 lines core, 11 test files |
| `trading-dashboard/` | Single-page HTML/JS dashboard (Chart.js) fed live by the bot's REST/WS API; includes an MT5 report parser (`parse_mt5_report.py`, 403 lines, stdlib-only) | JS tests via jsdom |
| `strategy-study/` | Strategy knowledge base, labeled setups dataset (12 setups), validation reports, tuning rationale | Docs + JSON + PNGs |
| `docs/` | Architecture handoff, scheduler doc, improvement-pass spec, trading playbook PDF | |
| `cowork-skills/` | 4 read-only Cowork analysis skills (strategy-study, bot-health, signal-review, news-impact) | |

Everything runs in **paper mode** (simulated INR ledger, ₹100,000). Live MT5 order routing is a
deliberate stub (`execution/mt5_router.py` raises `NotImplementedError`) — the bot cannot place
real orders today.

## 2. Quality assessment

**Verdict: well above hobby quality.** Strengths worth adopting into our own projects:

- Clean separation of concerns: `OrderRouter` interface (paper vs. live), injected data
  source/notifiers, pattern detectors emitting a neutral `Signal` consumed by downstream gates.
- Type hints + dataclasses throughout; YAML config with a redacted `config.example.yaml`.
- Defensive error handling at every external boundary; CRITICAL-level runtime invariant checks
  (look-ahead guard, sizing/P&L consistency) that page via Telegram on violation.
- Structured JSONL signal funnel → analytics endpoints (`/api/funnel`, `/api/shadow`) — rejection
  reasons are measurable, and a "shadow mode" tracks what rejected signals would have done.
- The strategy-study is unusually honest: explicit survivorship-bias warnings, refusal to loosen
  risk on winners-only data, R-multiples over currency P&L.

**Weaknesses / gaps:**

- READMEs have drifted from the code (bot README describes a retired yfinance feed and calls
  implemented detectors "stubs"; dashboard README documents a retired localStorage import flow).
- `patterns.py` — the strategy's heart — has no unit tests; no backtest/walk-forward harness.
- The FastAPI server binds `0.0.0.0` with CORS `*` and **no authentication** — anyone who can
  reach port 8765 can flip mode, inject commands, or push fake prices. Fine on localhost, unsafe
  on an exposed VPS.
- Dashboard folder clutter (backup HTML files, a render test hardcoding a path from another
  machine), CDN dependencies (Chart.js, Google Fonts) so it isn't fully offline.
- **No proven edge yet.** Every performance sample is tiny (n ≤ 12 trades). Latest snapshot:
  +10.4% / PF 2.07 on 8 paper trades — statistically meaningless, and the package says so itself.
  The bot's own shadow data suggests the 0.70 body-ratio confirmation gate rejects profitable
  signals (n=330 rejected, avg +1.69 R) — the biggest open tuning question.

## 3. 🔴 Security — must fix before any consolidation or push

1. **Live credentials are committed in cleartext in two files**: `trading-bot/config.yaml` and
   `WEBHOOK_AND_SECRETS.md` contain a real Discord webhook URL, a Telegram bot token + chat id,
   and an X/Twitter API bearer token (the package says they are live on purpose, "for the team").
2. `.gitignore` lines protecting those two files are **commented out** — a naive
   `git add -A && git push` leaks all three tokens.
3. **Action for Shakeeb (blocking):** rotate all three tokens (regenerate the Discord webhook,
   re-issue the Telegram token via BotFather, roll the X bearer token), keep only
   `config.example.yaml` in git, and load real values from env vars / an untracked `config.yaml`.
4. Do **not** vendor this code into any public repo (including this portfolio) until that's done.

## 4. Consolidation recommendation — yes, one repo (it already is one)

The bundle is already a coherent monorepo; the three parts are tightly coupled (the bot serves the
dashboard at `/`; strategy-study reads the bot's `state/` outputs). Recommendation:

- **One private GitHub repo** (e.g. `pattern-strategy-bot`) with the existing layout:
  `trading-bot/`, `trading-dashboard/`, `strategy-study/`, `docs/`, `cowork-skills/`.
- Before first push: rotate secrets (§3), uncomment the two `.gitignore` lines, delete
  `WEBHOOK_AND_SECRETS.md` from the tree (move to a password manager), drop backup/dead files
  (`index_backup_20260610.html`, `dashboard_standalone.html`, root-level one-off scripts →
  `scratch/` or delete).
- Add CI mirroring this portfolio's pattern: `pytest` for `trading-bot/tests/`, the jsdom parser
  test for the dashboard, plus lint. Fix the hardcoded path in `render_test.js` (use `__dirname`).
- Refresh both READMEs to match the code (MT5-only feed, detectors implemented, live-bot-fed
  dashboard).
- Do **not** merge it into this DataScience portfolio repo — it has live-ops state, secrets
  history, and a different lifecycle. Keep it separate and private; link it from the portfolio
  README if desired.

## 5. Feature: auto-trade from Discord recommendations (~2–4 days)

Today Discord integration is **outbound-only** (incoming webhook → posts messages; webhooks
structurally cannot read a channel). Nothing in the codebase listens to Discord. Plan:

1. **Inbound transport (new):** a Discord *bot* (discord.py) with the Message Content intent, or
   REST polling of the channel — modeled on the existing `news_agent.py` sidecar process.
   New secret: bot token (env var, per §3 hygiene).
2. **Parser:** message → `OrderRequest`. Strongly prefer a structured message format
   (`BUY EURUSD @ 1.0850 SL 1.0820 TP 1.0910`) over free-text NL — deterministic and testable.
3. **Safe injection:** new `POST /api/manual_signal` that routes through the *existing*
   session/news/risk gates (mirror the watch-levels path in `strategy/engine.py`), so daily caps,
   drawdown halt, and the kill-switch still apply. Never bypass the rails.
4. **Safety:** author allow-list (only trusted posters trigger trades), message-id idempotency
   (one message = one execution at most), and **paper mode only** until the strategy has a
   statistically meaningful track record (≥50 trades per setup, per the study's own bar).
5. **Tests:** parser + injection-path unit tests mirroring `test_discord_notifier.py`.

## 6. Feature: report upload → process → track → notify (~1–2 days)

The hard part already exists: `trading-dashboard/parse_mt5_report.py` is a complete, robust,
stdlib-only MT5 HTML/CSV history parser (broker-tolerant column mapping, setup-tag extraction,
R-multiple + adherence scoring). Gaps and plan:

1. **Library-ify the parser** (it's CLI-shaped: `sys.exit`, writes files) → `parse_report(path) -> payload`,
   move into the bot, add Python unit tests (currently only the JS twin is tested).
2. **Upload endpoint:** `POST /api/upload_report` (FastAPI multipart) → parse → persist snapshot
   under `state/imported_reports/`.
3. **Tracking = re-upload + diff:** an MT5 history report is a point-in-time export, so tracking
   means diffing each upload against the last (new trades, changed outcomes) and reconciling
   broker tickets against the bot's own `trade_journal/`. The matching keys (ticket, setup tag)
   already exist in the parser output.
4. **Notify:** on each import, push a digest (new trades, win rate, R, adherence violations)
   through the existing `NotifierGroup` → Telegram/Discord — the "strategically send me updates"
   piece rides on plumbing that already works.
5. Restore dashboard persistence for imported reports (the current `app.js` deliberately clears
   imports on reload — code pivoted to live-bot data and the README was never updated).

## 7. Further feature ideas (beyond the two requested)

Ordered roughly by value-per-effort, building on what already exists:

**Validation & strategy quality (highest value — the edge is still unproven)**
- **Offline backtest / walk-forward harness** — the biggest missing piece. The detectors, gates,
  and paper router are already pure enough to replay historical MT5 bars through; this converts
  "n=8 trades" into hundreds of samples per setup without waiting months of paper trading.
- **Shadow-driven auto-tuning report** — the shadow tracker + `tools/shadow_report.py` already
  measure gate expectancy; add a weekly job proposing threshold changes (e.g. the 0.70→0.55
  body-ratio experiment) with confidence intervals, applied only after human approval.
- **Pattern detector test-set** — turn `strategy-study/setups_dataset.json` (12 labeled setups)
  into regression fixtures so detector changes can't silently break known-good detections.

**Ops & safety**
- **API authentication + TLS** on the FastAPI server (token header is enough) — prerequisite for
  any remote/VPS deployment and for the Discord/report features.
- **Equity-curve circuit breaker** — beyond the existing daily/weekly caps: auto-halt if rolling
  20-trade expectancy goes negative, notify, require manual resume.
- **Config-change audit log** — dashboard toggles already burned them once (the "Enable all"
  incident that re-enabled a benched detector and lost money); log + notify every runtime
  scope change, with a "differs from config.yaml" banner.

**Reporting & UX**
- **Daily digest to Telegram/Discord** — one morning message: open positions, yesterday's closes
  in R, funnel stats, gate rejection leaderboard, news risk today. All the data already exists
  behind `/api/funnel`, `/api/shadow`, `/api/status`.
- **Weekly performance PDF/HTML report** — auto-render the equity curve, per-setup PF table, and
  adherence trends from the journal; share-ready for the team.
- **Trade replay view** — the journal already stores the bars that formed each pattern; render
  entry→exit as an annotated candlestick view in the dashboard (the visual validators do this
  offline already).

**Data & signals (later)**
- **Second data feed for cross-checking** (e.g. a free FX API) to detect bad-tick/frozen-feed
  conditions before they cause fills — the watchdog only checks process health, not data sanity.
- **Economic-calendar gate upgrade** — the ForexFactory scrape is fragile; a structured calendar
  API with pre-event position-flattening rules would harden the news filter.
- **Multi-user paper accounts** — separate ledgers per team member so Shakeeb's team can each
  trial parameter variants against the same feed.

## 8. Suggested sequence

1. Shakeeb rotates the three tokens; secrets removed from tracked files. **(blocking, ~1 hr)**
2. Create the private consolidated repo + CI; clean dead files; fix READMEs. (~half a day)
3. Report-upload pipeline (§6) — smallest effort, immediate personal value. (~1–2 days)
4. Discord signal consumption (§5) — paper mode, allow-listed authors. (~2–4 days)
5. Only revisit live trading after: API auth added, MT5 router implemented, and ≥50 clean paper
   trades per enabled setup show a real, loss-inclusive edge.
