#!/usr/bin/env bash
# =============================================================================
# RepBend Quick-Start Script
# 사용법:
#   bash quick_start.sh [train|eval|all]   (기본값: all)
#
# 환경 변수로 동작을 조정할 수 있습니다:
#   MODEL        : Mistral-7b | Meta_Llama3_8b   (기본: Meta_Llama3_8b)
#   DEVICE       : GPU 번호                       (기본: 0)
#   MAX_STEP     : 학습 스텝 수                   (기본: 50  ← 빠른 확인용)
#   OUTPUT_DIR   : 결과 저장 경로                 (기본: ./out/quick_start)
#   SKIP_INSTALL : 1 이면 패키지 설치 건너뜀      (기본: 0)
# =============================================================================
set -euo pipefail

# ── 인자 파싱 ────────────────────────────────────────────────────────────────
MODE="${1:-all}"   # train | eval | all

# ── 사용자 설정 ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-Meta_Llama3_8b}"
DEVICE="${DEVICE:-0}"
MAX_STEP="${MAX_STEP:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/quick_start}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

export CUBLAS_WORKSPACE_CONFIG=:16:8
export MASTER_PORT=$((29000 + RANDOM % 1000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 모델 경로 결정 ───────────────────────────────────────────────────────────
if [[ "$MODEL" == "Mistral-7b" ]]; then
    MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
elif [[ "$MODEL" == "Meta_Llama3_8b" ]]; then
    MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
else
    echo "[ERROR] 지원하지 않는 MODEL: $MODEL  (Mistral-7b | Meta_Llama3_8b)"
    exit 1
fi

# ── 색상 출력 헬퍼 ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. 환경 설치 (uv)
# ─────────────────────────────────────────────────────────────────────────────
install_env() {
    info "=== uv 환경 설치 ==="

    if ! command -v uv &>/dev/null; then
        warn "uv 가 설치되어 있지 않습니다. 설치를 진행합니다..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # PATH 업데이트 (현재 셸에서도 바로 사용 가능하도록)
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    info "uv 버전: $(uv --version)"

    # 가상 환경 생성 (없으면 생성)
    if [[ ! -d ".venv" ]]; then
        info ".venv 가상 환경을 생성합니다 (Python 3.10)..."
        uv venv --python 3.10 .venv
    fi

    # 의존성 설치 (cu121 휠 – 드라이버 570 / CUDA 12.8 호환)
    info "의존성 패키지를 설치합니다..."
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match \
        -e ".[dev]"

    # xformers 는 vllm 0.5.5 가 0.0.27.post2 를 자동으로 끌어옵니다 (별도 설치 불필요)

    # flash-attn: CUDA 툴킷 필요 (실패해도 계속 진행)
    uv pip install --no-build-isolation "flash-attn==2.6.3" \
        || warn "flash-attn 설치 실패 – flash attention 없이 진행합니다."

    info "환경 설치 완료."
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 학습 (RepBend)
# ─────────────────────────────────────────────────────────────────────────────
run_train() {
    info "=== RepBend 학습 시작 ==="
    info "  모델       : $MODEL_PATH"
    info "  GPU        : $DEVICE"
    info "  최대 스텝  : $MAX_STEP"
    info "  출력 경로  : $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES="$DEVICE" \
    accelerate launch \
        --config_file configs/accelerate_zero1.yaml \
        --num_processes 1 \
        --main_process_port "$MASTER_PORT" \
        methods/rep_bending/train.py \
            --model_name_or_path   "$MODEL_PATH" \
            --dataset_path         "allenai/wildguardmix" \
            --dataset_split        "wildguardtrain" \
            --target_layers        "-1" \
            --target_layer_start_idx 20 \
            --layers_window_size   11 \
            --transform_layers     "-1" \
            --loss_alpha           0.5 \
            --loss_beta            0.5 \
            --loss_gamma           0.1 \
            --loss_epsilon         0.3 \
            --loss_mode            "response_all" \
            --alpha_mode           "all" \
            --max_steps            "$MAX_STEP" \
            --lora_r               16 \
            --lora_alpha           16 \
            --lora_dropout         0.05 \
            --output_dir           "$OUTPUT_DIR" \
            --overwrite_output_dir \
            --num_train_epochs     1 \
            --bf16                 True \
            --tf32                 True \
            --per_device_train_batch_size       4 \
            --per_device_eval_batch_size        32 \
            --gradient_accumulation_steps       4 \
            --do_eval \
            --evaluation_strategy  "steps" \
            --eval_steps           "$MAX_STEP" \
            --save_total_limit     0 \
            --learning_rate        1e-5 \
            --weight_decay         0. \
            --lr_scheduler_type    "constant" \
            --logging_strategy     "steps" \
            --logging_steps        10 \
            --max_seq_length       2048 \
            --q_lora               False \
            --gradient_checkpointing True \
            --report_to            none \
            --is_online            False \
        2>&1 | tee "$OUTPUT_DIR/train.log"

    info "학습 완료 → 로그: $OUTPUT_DIR/train.log"
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. 평가 (HarmBench 기반 ASR 측정)
# ─────────────────────────────────────────────────────────────────────────────
run_eval() {
    info "=== 평가 시작 ==="

    # 모델 경로: 학습된 어댑터가 있으면 그걸 쓰고, 없으면 원본 모델 평가
    if [[ -f "$OUTPUT_DIR/adapter_config.json" ]]; then
        EVAL_MODEL="$OUTPUT_DIR"
        info "학습된 어댑터를 평가합니다: $EVAL_MODEL"
    else
        EVAL_MODEL="$MODEL_PATH"
        warn "학습된 어댑터를 찾을 수 없습니다. 원본 모델을 평가합니다: $EVAL_MODEL"
    fi

    EVAL_OUT="$OUTPUT_DIR/eval_results"
    mkdir -p "$EVAL_OUT"

    CUDA_VISIBLE_DEVICES="$DEVICE" \
    python safety_evaluation/evaluate.py \
        -m "$EVAL_MODEL" \
        --benchmark harmbench_test.json \
        --output_dir "$EVAL_OUT" \
        --prefill True \
        2>&1 | tee "$EVAL_OUT/eval.log"

    info "평가 완료 → 결과: $EVAL_OUT/eval.log"
    echo ""
    info "=== 평가 요약 ==="
    _print_results "$EVAL_OUT"
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. 결과 출력 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
_print_results() {
    local dir="$1"
    # JSON 결과 파일이 있으면 ASR을 Python으로 계산해서 출력
    local json_files
    json_files=$(find "$dir" -name "*.json" 2>/dev/null | head -5)

    if [[ -z "$json_files" ]]; then
        warn "결과 JSON 파일을 찾을 수 없습니다."
        return
    fi

    python - <<'PYEOF'
import json, sys, pathlib, glob, os

result_dir = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(f"{result_dir}/**/*.json", recursive=True))
if not files:
    print("결과 파일 없음.")
    sys.exit(0)

for fpath in files:
    try:
        with open(fpath) as f:
            data = json.load(f)
    except Exception:
        continue

    # 리스트 형식: [{behavior, response, label}, ...]
    if isinstance(data, list) and data and "label" in data[0]:
        total  = len(data)
        attack = sum(1 for d in data if d.get("label") == 1)
        asr    = attack / total * 100 if total else 0
        print(f"  파일 : {os.path.relpath(fpath)}")
        print(f"  총 샘플     : {total}")
        print(f"  공격 성공   : {attack}")
        print(f"  ASR (Attack Success Rate) : {asr:.1f}%")
        print()
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
echo ""
info "RepBend Quick-Start  (MODE=$MODE, MODEL=$MODEL, DEVICE=$DEVICE)"
echo "────────────────────────────────────────────────────────────────"

if [[ "$SKIP_INSTALL" != "1" ]]; then
    install_env
fi

# .venv 활성화
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

case "$MODE" in
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    all)
        run_train
        run_eval
        ;;
    *)
        error "알 수 없는 MODE: $MODE  (train | eval | all)"
        ;;
esac

echo ""
info "완료!"
