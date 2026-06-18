"""
Relearning Training Script
===========================
Unlearned 모델(LoRA 어댑터 포함)을 unsafe / benign 데이터로 재학습하여
unlearning의 취약성을 측정합니다.

참고 논문:
  (A) "Unlearning Isn't Deletion"       → --relearning_mode direct
  (B) "Do Unlearning Methods Remove...?" → --relearning_mode low_budget  (--n_shot N)
  (C) "Unlearning or Obfuscating?"       → --relearning_mode benign

사용 예시:
  python methods/relearning/train.py \
      --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
      --adapter_name_or_path ./out/my_unlearned_model \
      --relearning_mode direct \
      --num_examples 500 \
      --learning_rate 2e-5 \
      --max_steps 200 \
      --output_dir ./out/relearned/my_run
"""

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from methods.relearning.dataset import RelearningDataset


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Relearning trainer")

    # 모델
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help="Unlearned LoRA 어댑터 경로. None이면 원본 모델 그대로 사용.",
    )

    # 데이터
    parser.add_argument(
        "--relearning_mode",
        type=str,
        default="direct",
        choices=["direct", "low_budget", "benign"],
        help=(
            "direct     : (A) unsafe prompt + unsafe response SFT\n"
            "low_budget : (B) 소수(n_shot)의 harmful 쌍으로 취약성 측정\n"
            "benign     : (C) unsafe prompt + 거부 응답으로 기억 회복 측정"
        ),
    )
    parser.add_argument("--dataset_path", type=str, default="allenai/wildguardmix")
    parser.add_argument("--dataset_split", type=str, default="wildguardtrain")
    parser.add_argument("--wildjailbreak_path", type=str, default=None,
                        help="WildJailbreak jsonl 경로. 지정하면 wildguardmix 대신 사용.")
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--n_shot", type=int, default=50,
                        help="low_budget 모드에서 사용할 샘플 수")
    parser.add_argument("--max_seq_length", type=int, default=1024)

    # LoRA (재학습 시 추가로 붙일 어댑터)
    parser.add_argument("--use_lora", action="store_true",
                        help="재학습 시 새 LoRA 어댑터를 추가할지 여부")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 학습 하이퍼파라미터
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)

    # 출력
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[Relearning] mode          : {args.relearning_mode}")
    print(f"[Relearning] base model    : {args.model_name_or_path}")
    print(f"[Relearning] adapter       : {args.adapter_name_or_path}")
    print(f"[Relearning] output_dir    : {args.output_dir}")
    print(f"[Relearning] num_examples  : {args.num_examples}")
    print(f"[Relearning] n_shot        : {args.n_shot}")
    print(f"[Relearning] max_steps     : {args.max_steps}")
    print(f"[Relearning] lr            : {args.learning_rate}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 토크나이저 ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # ── 모델 로드 ───────────────────────────────────────────────────────────
    # 분산 학습(DDP) 시에는 각 프로세스가 자신의 GPU에만 모델을 올림
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank != -1 and world_size > 1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # unlearned 어댑터 로드
    if args.adapter_name_or_path is not None:
        print(f"[Relearning] Loading unlearned adapter from {args.adapter_name_or_path}")
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        # 재학습 시 어댑터 가중치도 업데이트
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)

    # 추가 LoRA 붙이기 (선택)
    if args.use_lora and args.adapter_name_or_path is None:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.adapter_name_or_path is None and not args.use_lora:
        # 어댑터 없이 전체 파라미터 full fine-tune (주로 low_budget 실험)
        print("[Relearning] Full parameter fine-tuning (no adapter).")

    model.config.use_cache = False

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    if args.wildjailbreak_path is not None:
        dataset = RelearningDataset.from_wildjailbreak(
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
            jsonl_path=args.wildjailbreak_path,
            relearning_mode=args.relearning_mode,
            num_examples=args.num_examples,
            n_shot=args.n_shot,
            max_length=args.max_seq_length,
        )
    else:
        dataset = RelearningDataset(
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
            relearning_mode=args.relearning_mode,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            num_examples=args.num_examples,
            n_shot=args.n_shot,
            max_length=args.max_seq_length,
        )

    print(f"[Relearning] Dataset size: {len(dataset)}")

    # ── 학습 인자 ────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        bf16=args.bf16,
        tf32=True,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    model.enable_input_require_grads()
    trainer.train()

    # ── 저장 ─────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Relearning] Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
