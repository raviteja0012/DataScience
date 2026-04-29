#!/usr/bin/env bash
#
# quickstart.sh - one-command install + demo for the Payment Intelligence Agent.
#
# Idempotent. Safe to re-run. Creates a local virtualenv, installs deps,
# runs all three demo scripts, and writes captured output to ./output/.
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
printf "%s  Payment Intelligence Agent - Quickstart%s\n" "${C_BOLD}" "${C_RESET}"
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
step "Installing dependencies (this can take 2-3 minutes the first time)"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
success "dependencies installed"

# -- 3. Run demos -- #
run_demo() {
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

run_demo "demo/demo_analytics.py"  "01_analytics"
run_demo "demo/demo_anomaly.py"    "02_anomaly"
run_demo "demo/demo_rag.py"        "03_rag"

# -- 4. Summary -- #
printf "\n%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s  DEMO SUMMARY%s\n" "${C_BOLD}" "${C_RESET}"
printf "%s======================================================================%s\n" "${C_BOLD}" "${C_RESET}"
cat <<EOF

  ${C_GREEN}Demonstrated:${C_RESET}
    1. Analytics      - 8 natural-language queries translated to safe
                        Snowflake SQL with template matching, validation,
                        execution against synthetic data, and NL summaries.
    2. Anomaly        - Statistical detection (z-score, IQR, modified-z),
                        pattern analysis (card testing, geo-velocity),
                        time-series detection, and severity-tiered alerts.
    3. Compliance RAG - PCI DSS document loading, chunking, embedding,
                        FAISS-backed retrieval, and cited Q&A.

  ${C_CYAN}Output captured to:${C_RESET} ${OUTPUT_DIR}
  ${C_CYAN}Reference samples at:${C_RESET} demo/sample_output/

EOF

printf "%s==>%s %sNext step:%s launch the interactive UI:\n\n" "${C_BLUE}" "${C_RESET}" "${C_BOLD}" "${C_RESET}"
printf "    %sstreamlit run src/app.py%s\n\n" "${C_GREEN}" "${C_RESET}"
printf "Then open http://localhost:8501 in your browser.\n"
printf "See demo/sample_output/streamlit_screenshot_description.md for what to expect.\n\n"

success "quickstart complete"
