#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/code/github_repos/DataProcessor"
LOG_DIR="${ROOT_DIR}/log"
PY_SCRIPT="${ROOT_DIR}/speech_enhancement_process/mossformergan_batch_inference.py"
SCRIPT_PATH="$(readlink -f "$0")"
TMUX_SESSION="mossformergan_20s_sequential"

mkdir -p "${LOG_DIR}"

if [[ -z "${DP_MOSS_TMUX_CHILD:-}" && -z "${TMUX:-}" ]]; then
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${TMUX_SESSION}"
    echo "attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
  fi

  tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate kimi-audio && DP_MOSS_TMUX_CHILD=1 bash '${SCRIPT_PATH}'"
  echo "started tmux session: ${TMUX_SESSION}"
  echo "attach with: tmux attach -t ${TMUX_SESSION}"
  exit 0
fi

export DP_MOSS_DECODE_WINDOW_SECONDS=20
export DP_MOSS_ONE_TIME_DECODE_LENGTH_SECONDS=20

run_dataset() {
  local dataset_name="$1"
  local input_dir="$2"
  local output_dir="$3"
  local log_prefix="$4"
  local main_log="${LOG_DIR}/${log_prefix}_main.log"

  if [[ ! -d "${input_dir}" ]]; then
    echo "[$(date '+%F %T')] ERROR: input dir not found: ${input_dir}" | tee -a "${LOG_DIR}/mossformergan_20s_sequential.log"
    return 1
  fi

  mkdir -p "${output_dir}"

  {
    echo "============================================================"
    echo "[$(date '+%F %T')] START ${dataset_name}"
    echo "input: ${input_dir}"
    echo "output: ${output_dir}"
    echo "log_prefix: ${log_prefix}"
    echo "decode_window: ${DP_MOSS_DECODE_WINDOW_SECONDS}s"
    echo "one_time_decode_length: ${DP_MOSS_ONE_TIME_DECODE_LENGTH_SECONDS}s"
    echo "============================================================"
  } | tee -a "${LOG_DIR}/mossformergan_20s_sequential.log"

  (
    cd "${ROOT_DIR}"
    DP_MOSS_INPUT_DIR="${input_dir}" \
    DP_MOSS_OUTPUT_DIR="${output_dir}" \
    DP_MOSS_LOG_PREFIX="${log_prefix}" \
      python "${PY_SCRIPT}"
  ) 2>&1 | tee "${main_log}"

  {
    echo "[$(date '+%F %T')] END ${dataset_name}"
    echo
  } | tee -a "${LOG_DIR}/mossformergan_20s_sequential.log"
}

run_dataset \
  "SMIIP-TV" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/SMIIP-TV" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/SMIIP-TV_mossformergan_processed" \
  "smiip_mossformergan_20s"

run_dataset \
  "seniortalk_processed" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/seniortalk_processed" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/seniortalk_processed_singletalk_mossformergan_processed" \
  "seniortalk_processed_mossformergan_20s"

run_dataset \
  "aidatatang_200zh" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/aidatatang_200zh" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/aidatatang_200zh_mossformergan_enhanced" \
  "aidatatang_200zh_mossformergan_20s"

run_dataset \
  "cnceleb" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/cnceleb" \
  "/root/group-shared/voiceprint/data/speech/speaker_verification/cnceleb_mossformergan_enhanced" \
  "cnceleb_mossformergan_20s"
