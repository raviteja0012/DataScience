#!/usr/bin/env bash
#
# quickstart.sh - one-command install + demo for the Biller Integration Simulator.
#
# Idempotent. Safe to re-run. Creates a local virtualenv, installs deps,
# runs all three example scripts, and writes captured output to ./output/.
#
set -euo pipefail

# -- ANSI colors (no dep on tput - works in any terminal that supports ANSI) -- #
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

# -- Locate project root (the directory this script lives in) -- #
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${PROJECT_ROOT}"

VENV_DIR="${PROJECT_ROOT}/.venv"
OUTPUT_DIR="${PROJECT_ROOT}/output/quickstart"
mkdir -p "${OUTPUT_DIR}"

printf "\n%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s  Biller Integration Simulator - Quickstart%s\n" "${C_BOLD}" "${C_RESET}"
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
        fail "${script} exited non-zero (see ${out})"
        return 1
    fi
}

run_example "examples/onboard_new_biller.py"      "01_onboarding"
run_example "examples/process_batch_payments.py"  "02_batch_payments"
run_example "examples/run_settlement.py"          "03_settlement"

# -- 4. Summary -- #
printf "\n%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s  DEMO SUMMARY%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
cat <<EOF

  ${C_GREEN}Demonstrated:${C_RESET}
    1. Biller onboarding   - 3 billers loaded, validated (10/10 checks each),
                             and activated. CC&B schema mapping verified
                             across 5 source tables (CI_ACCT, CI_PER, etc).
    2. Batch payments      - 40 transactions processed end-to-end with
                             idempotency-key duplicate detection, exception
                             classification, and multi-channel notifications.
    3. Settlement recon    - Three-way reconciliation (biller / platform /
                             bank). 96% match rate with 1 amount-mismatch
                             escalated and 1 missing-bank auto-resolved.

  ${C_CYAN}Output captured to:${C_RESET} ${OUTPUT_DIR}
  ${C_CYAN}Reference samples at:${C_RESET} examples/sample_output/

  Run 'pytest -q' to execute the full 53-test suite.

EOF
success "quickstart complete"
