#!/usr/bin/env bash
# =============================================================================
# scripts/sweep_unlearn_relearn.sh
#
# Unlearning 파라미터 grid-sweep → relearning → 평가를 순서대로 수행합니다.
#
# 가설 검증:
#   "Unlearning 성능을 극대화하기 위해 선택된 파라미터(레이어 범위, 손실 계수 등)가
#    오히려 relearning에 취약하다."
#
# 결과는 모두 SWEEP_ROOT 아래 JSON으로 저장되며,
# 마지막에 analysis/plot_relearning_analysis.py 로 Figure를 생성합니다.
#
# 사용법:
#   bash scripts/sweep_unlearn_relearn.sh [gpu_id]
#   RELEARN_MODES="direct benign" bash scripts/sweep_unlearn_relearn.sh 0
# =============================================================================
set -euo pipefail

# ── 기본 설정 ────────────────────────────────────────────────────────────────
DEVICE="${1:-0}"
MODEL="${MODEL:-Meta_Llama3_8b}"
SWEEP_ROOT="${SWEEP_ROOT:-./out/sweep_relearning}"
RELEARN_MODES="${RELEARN_MODES:-direct low_budget benign}"  # 공백 구분
UNLEARN_STEPS="${UNLEARN_STEPS:-300}"     # unlearning 스텝 (빠른 sweep용)
RELEARN_STEPS="${RELEARN_STEPS:-200}"     # relearning 스텝
BENCHMARK="${BENCHMARK:-harmbench_test.json}"
MASTER_PORT=$((29000 + RANDOM % 1000))

# ── GPU 개수 계산 ────────────────────────────────────────────────────────────
# DEVICE="0,1,2,3" 처럼 콤마로 여러 GPU를 주면 학습은 그 개수만큼 DDP로 실행.
# 평가(vllm)는 단일 GPU만 사용하므로 첫 번째 GPU로 제한.
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$DEVICE")
FIRST_GPU="${DEVICE%%,*}"
EVAL_DEVICE="${EVAL_DEVICE:-$FIRST_GPU}"

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Python 실행 경로 자동 감지 ─────────────────────────────────────────────
# venv가 활성화되어 있으면 그것을, 아니면 python3/python 순으로 탐색
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -f ".venv/bin/python" ]]; then
    PYTHON="$(pwd)/.venv/bin/python"
    source .venv/bin/activate
else
    PYTHON="$(command -v python3 || command -v python || echo 'python3')"
fi
export PYTHON
echo "[SWEEP] Python 실행 경로: $PYTHON"

# ── 모델 경로 ────────────────────────────────────────────────────────────────
if [[ "$MODEL" == "Mistral-7b" ]]; then
    BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
elif [[ "$MODEL" == "Meta_Llama3_8b" ]]; then
    BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
else
    echo "[ERROR] 지원하지 않는 MODEL: $MODEL"; exit 1
fi

# ── 색상 헬퍼 ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[SWEEP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Sweep 파라미터 격자
#
# target_layer_start:  표현을 수정할 레이어 시작 인덱스
# window_size:         수정할 레이어 수
# alpha / beta / eps:  손실 계수 (α: safe push, β: retain, ε: unsafe pull)
# loss_mode:           어떤 토큰의 표현을 loss에 사용할지
# ─────────────────────────────────────────────────────────────────────────────

# 레이어 설정 (start:window 형식)
LAYER_CONFIGS=(
    "15:5"    # 얕은 레이어, 좁은 범위  → unlearning 효과 약 / relearning 취약?
    "20:11"   # 논문 기본값
    "20:5"    # 중간 레이어, 좁은 범위
    "25:5"    # 깊은 레이어, 좁은 범위  → unlearning 효과 강 / relearning 더 취약?
    "15:15"   # 넓은 범위
)

# 손실 계수 설정 (alpha:beta:gamma:epsilon)
LOSS_CONFIGS=(
    "0.5:0.5:0.1:0.3"   # 논문 기본값
    "1.0:0.5:0.1:0.3"   # alpha(safe) 강화
    "0.5:0.5:0.1:0.8"   # epsilon(unsafe push) 강화
    "0.5:0.1:0.1:0.3"   # beta(retain) 약화
    "1.0:0.1:0.0:0.8"   # aggressive: retain 무시, safe+unsafe 집중
)

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: ASR 점수 추출 (log.json → float)
# ─────────────────────────────────────────────────────────────────────────────
extract_asr() {
    local log_json="$1"
    if [[ ! -f "$log_json" ]]; then echo "N/A"; return; fi
    "$PYTHON" - <<PYEOF
import json, sys
with open("$log_json") as f:
    data = json.load(f)
score = data.get("score", None)
if score is None:
    # samples 기반 계산
    samples = data.get("samples", [])
    if samples:
        n_attack = sum(1 for s in samples if s.get("label") == 1)
        score = n_attack / len(samples)
    else:
        score = -1
print(f"{score:.4f}")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: 단계별 실행 래퍼
#   run_step <sentinel> <validate_file> <cmd...>
#
#   sentinel      : 완료 표시 파일
#   validate_file : 실제 성공 여부를 확인할 출력 파일 (없으면 "" 전달)
#                   sentinel이 있어도 validate_file이 없으면 재실행
#   FORCE=1       : sentinel 무시하고 항상 재실행
#   RESTART_FROM_EVAL=1 : unlearned(학습 결과)는 그대로 두고
#                         eval_unlearned / relearned 단계부터 다시 실행
# ─────────────────────────────────────────────────────────────────────────────
FORCE="${FORCE:-0}"
RESTART_FROM_EVAL="${RESTART_FROM_EVAL:-0}"

run_step() {
    local sentinel="$1"
    local validate="$2"
    shift 2

    if [[ "$FORCE" != "1" && -f "$sentinel" ]]; then
        # sentinel이 있어도 실제 출력 파일이 없으면 재실행
        if [[ -z "$validate" || -f "$validate" ]]; then
            warn "이미 완료 ($sentinel), 스킵합니다."
            return 0
        else
            warn "Sentinel 존재하나 출력 파일 없음 → 재실행: $validate"
            rm -f "$sentinel"
        fi
    fi

    "$@"
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] 명령 실패 (exit=$exit_code): $*" >&2
        return $exit_code
    fi
    touch "$sentinel"
}

# ─────────────────────────────────────────────────────────────────────────────
# 메인 sweep 루프
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$SWEEP_ROOT"
RESULTS_CSV="$SWEEP_ROOT/results.csv"

# CSV 헤더 초기화
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "run_id,layer_start,window_size,alpha,beta,gamma,epsilon,loss_mode,relearning_mode,asr_after_unlearn,asr_after_relearn,asr_delta" \
        > "$RESULTS_CSV"
fi

info "=== Sweep 시작: $(date) ==="
info "  Base model : $BASE_MODEL"
info "  GPU (train): $DEVICE  (DDP num_processes=$NUM_GPUS)"
info "  GPU (eval) : $EVAL_DEVICE  (vllm 단일 GPU)"
info "  Unlearn steps : $UNLEARN_STEPS"
info "  Relearn steps : $RELEARN_STEPS"
info "  Layer configs : ${#LAYER_CONFIGS[@]}"
info "  Loss configs  : ${#LOSS_CONFIGS[@]}"
info "  Relearn modes : $RELEARN_MODES"
echo ""

for LAYER_CFG in "${LAYER_CONFIGS[@]}"; do
    LAYER_START="${LAYER_CFG%%:*}"
    WINDOW="${LAYER_CFG##*:}"

    for LOSS_CFG in "${LOSS_CONFIGS[@]}"; do
        IFS=':' read -r ALPHA BETA GAMMA EPSILON <<< "$LOSS_CFG"

        # ── Run ID 결정 ────────────────────────────────────────────────────
        RUN_ID="l${LAYER_START}w${WINDOW}_a${ALPHA}_b${BETA}_g${GAMMA}_e${EPSILON}"
        UNLEARN_DIR="$SWEEP_ROOT/$RUN_ID/unlearned"
        EVAL_UNLEARN_DIR="$SWEEP_ROOT/$RUN_ID/eval_unlearned"

        info "────────────────────────────────────────────────────"
        info "Run: $RUN_ID"
        mkdir -p "$UNLEARN_DIR" "$EVAL_UNLEARN_DIR"

        # ── RESTART_FROM_EVAL: unlearned는 유지, 이후 단계 sentinel/출력 제거 ──
        if [[ "$RESTART_FROM_EVAL" == "1" ]]; then
            warn "  RESTART_FROM_EVAL=1 → unlearned 유지, eval_unlearned/relearned 단계 재시작"
            rm -rf "$EVAL_UNLEARN_DIR"
            rm -rf "$SWEEP_ROOT/$RUN_ID"/relearned_* "$SWEEP_ROOT/$RUN_ID"/eval_relearned_*
            mkdir -p "$EVAL_UNLEARN_DIR"
        fi

        # ── (1) Unlearning ─────────────────────────────────────────────────
        run_step "$UNLEARN_DIR/.done" "$UNLEARN_DIR/adapter_config.json" \
            bash -c "
CUDA_VISIBLE_DEVICES=$DEVICE \
accelerate launch \
    --config_file configs/accelerate_zero1.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    methods/rep_bending/train.py \
        --model_name_or_path $BASE_MODEL \
        --dataset_path allenai/wildguardmix \
        --dataset_split wildguardtrain \
        --target_layer_start_idx $LAYER_START \
        --layers_window_size $WINDOW \
        --transform_layers -1 \
        --loss_alpha $ALPHA \
        --loss_beta $BETA \
        --loss_gamma $GAMMA \
        --loss_epsilon $EPSILON \
        --loss_mode response_all \
        --alpha_mode all \
        --max_steps $UNLEARN_STEPS \
        --lora_r 16 --lora_alpha 16 --lora_dropout 0.05 \
        --output_dir $UNLEARN_DIR \
        --overwrite_output_dir \
        --num_train_epochs 1 \
        --bf16 True --tf32 True \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --save_total_limit 0 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --lr_scheduler_type constant \
        --logging_strategy steps --logging_steps 10 \
        --max_seq_length 2048 \
        --q_lora False \
        --gradient_checkpointing True \
        --report_to none \
        --is_online False \
    2>&1 | tee $UNLEARN_DIR/train.log
"

        # ── (2) Unlearning 후 평가 ──────────────────────────────────────────
        run_step "$EVAL_UNLEARN_DIR/.done" "$EVAL_UNLEARN_DIR/log.json" \
            bash -c "
CUDA_VISIBLE_DEVICES=$EVAL_DEVICE $PYTHON safety_evaluation/evaluate.py \
    -m $UNLEARN_DIR \
    --benchmark $BENCHMARK \
    --output_dir $EVAL_UNLEARN_DIR \
    2>&1 | tee $EVAL_UNLEARN_DIR/eval.log
"
        ASR_UNLEARN=$(extract_asr "$EVAL_UNLEARN_DIR/log.json")
        info "  ASR after unlearn : $ASR_UNLEARN"

        # ── (3) Relearning + 평가 (모드별) ────────────────────────────────
        for RL_MODE in $RELEARN_MODES; do
            RELEARN_DIR="$SWEEP_ROOT/$RUN_ID/relearned_${RL_MODE}"
            EVAL_RELEARN_DIR="$SWEEP_ROOT/$RUN_ID/eval_relearned_${RL_MODE}"
            mkdir -p "$RELEARN_DIR" "$EVAL_RELEARN_DIR"

            # n_shot: low_budget 모드에서 사용 (50샘플)
            N_SHOT=50

            run_step "$RELEARN_DIR/.done" "$RELEARN_DIR/adapter_config.json" \
                bash -c "
CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON methods/relearning/train.py \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $UNLEARN_DIR \
    --relearning_mode $RL_MODE \
    --dataset_path allenai/wildguardmix \
    --dataset_split wildguardtrain \
    --num_examples 500 \
    --n_shot $N_SHOT \
    --max_seq_length 1024 \
    --learning_rate 2e-5 \
    --max_steps $RELEARN_STEPS \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir $RELEARN_DIR \
    2>&1 | tee $RELEARN_DIR/train.log
"

            run_step "$EVAL_RELEARN_DIR/.done" "$EVAL_RELEARN_DIR/log.json" \
                bash -c "
CUDA_VISIBLE_DEVICES=$EVAL_DEVICE $PYTHON safety_evaluation/evaluate.py \
    -m $RELEARN_DIR \
    --benchmark $BENCHMARK \
    --output_dir $EVAL_RELEARN_DIR \
    2>&1 | tee $EVAL_RELEARN_DIR/eval.log
"
            ASR_RELEARN=$(extract_asr "$EVAL_RELEARN_DIR/log.json")
            # delta: relearning 후 - unlearning 후 (양수 = relearning으로 성능 회복)
            ASR_DELTA=$("$PYTHON" -c "
a='$ASR_UNLEARN'; b='$ASR_RELEARN'
try:
    print(f'{float(b)-float(a):.4f}')
except:
    print('N/A')
")
            info "  [${RL_MODE}] ASR after relearn : $ASR_RELEARN  (Δ=$ASR_DELTA)"

            # CSV 기록
            echo "${RUN_ID},${LAYER_START},${WINDOW},${ALPHA},${BETA},${GAMMA},${EPSILON},response_all,${RL_MODE},${ASR_UNLEARN},${ASR_RELEARN},${ASR_DELTA}" \
                >> "$RESULTS_CSV"
        done
    done
done

info ""
info "=== Sweep 완료: $(date) ==="
info "결과 CSV : $RESULTS_CSV"
info ""
info "Figure 생성 중..."
$PYTHON analysis/plot_relearning_analysis.py --results_csv "$RESULTS_CSV" --output_dir "$SWEEP_ROOT/figures"
info "Figure 저장됨 : $SWEEP_ROOT/figures/"
