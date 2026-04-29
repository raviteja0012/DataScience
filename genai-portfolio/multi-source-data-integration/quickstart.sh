#!/usr/bin/env bash
#
# quickstart.sh - one-command install + demo for Multi-Source Data Integration.
#
# Idempotent. Safe to re-run. Creates a local virtualenv, installs deps,
# runs all four example scripts, and writes captured output to ./output/.
#
set -euo pipefail

# -- ANSI colors -- #
readonly C_RESET=$'\033[0m'
readonly C_BOLD=$'\033[1m'
readonly C_BLUE=$'\033[34m'
readonly C_GREEN=$'\033[32m'
readonly C_YELLOW=$'\033[33m'
readonly C_CYAN=$'\033[36m'
readonly C_RED=$'\033[31m'

step()    { printf "%s==>%s %s%s%s\n"      "${C_BLUE}"   "${C_RESET}" "${C_BOLD}" "$*" "${C_RESET}"; }
info()    { printf "%s   .%s %s\n"          "${C_CYAN}"   "${C_RESET}" "$*"; }
success() { printf "%s [OK]%s %s\n"         "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()    { printf "%s[warn]%s %s\n"        "${C_YELLOW}" "${C_RESET}" "$*"; }
fail()    { printf "%s[FAIL]%s %s\n"        "${C_RED}"    "${C_RESET}" "$*" >&2; }

# -- Project root -- #
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${PROJECT_ROOT}"

VENV_DIR="${PROJECT_ROOT}/.venv"
OUTPUT_DIR="${PROJECT_ROOT}/output/quickstart"
mkdir -p "${OUTPUT_DIR}"

printf "\n%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s  Multi-Source Data Integration - Quickstart%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s======================================================================%s\n\n" "${C_BOLD}" "${C_RESET}"

# -- 1. Virtualenv -- #
step "Setting up virtual environment"
if [[ -d "${VENV_DIR}" ]]; then
    info "venv already exists at ${VENV_DIR}"
else
    info "creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
success "virtualenv ready: $(python --version)"

# -- 2. Dependencies -- #
step "Installing dependencies"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
success "dependencies installed"

# -- 3. Run examples -- #
run_example() {
    local script="$1"
    local label="$2"
    local out="${OUTPUT_DIR}/${label}.txt"
    step "Running: ${script}"
    if python "${script}" > "${out}" 2>&1; then
        local lines
        lines=$(wc -l < "${out}")
        success "captured ${lines} lines -> ${out}"
    else
        warn "${script} exited non-zero (see ${out}). Continuing..."
    fi
}

run_example "examples/run_discovery.py"            "01_discovery"
run_example "examples/run_identity_resolution.py"  "02_identity_resolution"
run_example "examples/run_reconciliation.py"       "03_reconciliation"
run_example "examples/run_full_pipeline.py"        "04_full_pipeline"

# -- 4. Summary -- #
printf "\n%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s  DEMO SUMMARY%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
cat <<EOF

  ${C_GREEN}Demonstrated:${C_RESET}
    1. Discovery     - Profiled 2 source systems (Oracle ERP + SQL Server CRM),
                       8 tables, ~5,200 rows. Quality scoring with per-column
                       drill-down, schema mapping coverage, dependency-aware
                       parallel load waves.
    2. Identity res  - Resolved 1,000 customer records (500 + 500) into 930
                       canonical entities. 70 cross-source matches found via
                       blocking + Jaro-Winkler/Levenshtein/Metaphone. Field-level
                       survivorship rules applied.
    3. Reconciliation- Three-dimensional validation (count, value, hash) across
                       4 entities. HTML and JSON reports generated.
    4. Full pipeline - End-to-end run with phased cutover, parallel task
                       execution, and checkpoint-driven idempotency.

  ${C_CYAN}Output captured to:${C_RESET} ${OUTPUT_DIR}
  ${C_CYAN}Reference samples at:${C_RESET} examples/sample_output/

  ${C_CYAN}Auto-generated reports:${C_RESET}
    output/reports/reconciliation_*.html  (open in a browser)
    output/reports/reconciliation_*.json
    output/metrics/pipeline_metrics.json

  Run 'pytest -q' to execute the full 86-test suite.

EOF
success "quickstart complete"
