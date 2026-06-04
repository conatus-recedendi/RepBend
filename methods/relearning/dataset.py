"""
Relearning Dataset
==================
세 논문에서 제안된 relearning 기법을 지원하는 데이터셋 모듈.

참고 논문:
  (A) "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs"
      → direct:  unlearning에 사용했던 unsafe (prompt+response) 쌍을 그대로 재학습
  (B) "Do Unlearning Methods Remove Information from Language Model Weights?"
      → low_budget: 극히 소수(n_shot)의 harmful 쌍만으로도 unlearning이 되돌아오는지 확인
  (C) "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning"
      → benign: 해로운 응답 없이 '관련 주제'의 일반 텍스트로 기억 회복 여부 확인

relearning_mode:
  "direct"      : (A) unsafe prompt + unsafe response SFT
  "low_budget"  : (B) unsafe prompt + unsafe response, n_shot 개만 사용
  "benign"      : (C) unsafe prompt + 거부 응답(safe response)으로 SFT (모델이 benign하게 반응하도록 유도)
"""

import random
from typing import Dict, Literal

import jsonlines
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

random.seed(42)

RelearningMode = Literal["direct", "low_budget", "benign"]


class RelearningDataset(Dataset):
    """unsafe 데이터 혹은 benign 데이터로 unlearned 모델을 재학습시키는 데이터셋."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        model_name_or_path: str,
        relearning_mode: RelearningMode = "direct",
        dataset_path: str = "allenai/wildguardmix",
        dataset_split: str = "wildguardtrain",
        num_examples: int = 500,
        n_shot: int = 50,           # low_budget 모드에서만 사용
        max_length: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relearning_mode = relearning_mode
        model_name_lower = model_name_or_path.lower()

        # ── 채팅 템플릿 설정 ───────────────────────────────────────────────
        if "llama-3" in model_name_lower:
            self.user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
            self.assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in model_name_lower:
            tokenizer.chat_template = (
                "{{ bos_token }}{% for message in messages %}"
                "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                "{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}"
                "{% endif %}{% endfor %}"
            )
            self.user_tag = "[INST] "
            self.assistant_tag = " [/INST]"
        else:
            raise NotImplementedError(f"Model template not supported: {model_name_or_path}")

        # ── 데이터 로드 ────────────────────────────────────────────────────
        print(f"[RelearningDataset] mode={relearning_mode}, loading {dataset_path} ...")
        ds = load_dataset(dataset_path, split=dataset_split)

        direct_pairs: list[tuple[str, str]] = []   # (prompt, response) unsafe 쌍
        benign_pairs: list[tuple[str, str]] = []   # (unsafe_prompt, safe_refusal_response)

        for d in ds:
            if d["prompt_harm_label"] == "harmful" and d["response_refusal_label"] == "compliance":
                # unsafe prompt + complying (harmful) response
                direct_pairs.append((d["prompt"], d["response"]))
            elif d["prompt_harm_label"] == "harmful" and d["response_refusal_label"] == "refusal":
                # unsafe prompt + refusal (benign relearning용)
                benign_pairs.append((d["prompt"], d["response"]))

        random.shuffle(direct_pairs)
        random.shuffle(benign_pairs)

        if relearning_mode == "direct":
            pairs = direct_pairs[:num_examples]
        elif relearning_mode == "low_budget":
            # (B): 아주 적은 수만 사용 → unlearning이 얼마나 취약한지 측정
            pairs = direct_pairs[:n_shot]
        elif relearning_mode == "benign":
            # (C): 해로운 응답 없이 benign 응답만으로 relearning
            pairs = benign_pairs[:num_examples]
        else:
            raise ValueError(f"Unknown relearning_mode: {relearning_mode}")

        template = "{user_tag}{prompt}{assistant_tag}{response}"
        self.samples = [
            template.format(
                user_tag=self.user_tag,
                assistant_tag=self.assistant_tag,
                prompt=p,
                response=r,
            )
            for p, r in pairs
        ]

        tokenizer.padding_side = "right"
        print(f"[RelearningDataset] total samples: {len(self.samples)}")
        if self.samples:
            print(f"[RelearningDataset] example[0]: {self.samples[0][:200]}")

    # ── Widjailbreak 보조 로더 ────────────────────────────────────────────
    @classmethod
    def from_wildjailbreak(
        cls,
        tokenizer: transformers.PreTrainedTokenizer,
        model_name_or_path: str,
        jsonl_path: str = "./wildjailbreak.jsonl",
        relearning_mode: RelearningMode = "direct",
        num_examples: int = 500,
        n_shot: int = 50,
        max_length: int = 1024,
    ) -> "RelearningDataset":
        """WildJailbreak jsonl 파일에서 직접 로드."""
        obj = object.__new__(cls)
        obj.tokenizer = tokenizer
        obj.max_length = max_length
        obj.relearning_mode = relearning_mode

        model_name_lower = model_name_or_path.lower()
        if "llama-3" in model_name_lower:
            obj.user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
            obj.assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in model_name_lower:
            obj.user_tag = "[INST] "
            obj.assistant_tag = " [/INST]"
        else:
            raise NotImplementedError

        data = []
        with jsonlines.open(jsonl_path) as f:
            for line in f:
                data.append(line)
        random.shuffle(data)

        pairs: list[tuple[str, str]] = []
        for d in data:
            if d.get("prompt_type") == "harmful":
                if relearning_mode in ("direct", "low_budget"):
                    pairs.append((d["prompt"], d["harmful_answer"]))
                else:  # benign
                    pairs.append((d["prompt"], d["harmless_answer"]))

        if relearning_mode == "low_budget":
            pairs = pairs[:n_shot]
        else:
            pairs = pairs[:num_examples]

        template = "{user_tag}{prompt}{assistant_tag}{response}"
        obj.samples = [
            template.format(
                user_tag=obj.user_tag,
                assistant_tag=obj.assistant_tag,
                prompt=p,
                response=r,
            )
            for p, r in pairs
        ]
        tokenizer.padding_side = "right"
        print(f"[RelearningDataset/WildJailbreak] mode={relearning_mode}, samples={len(obj.samples)}")
        return obj

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text = self.samples[i]
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        # labels: pad 위치는 -100으로 마스킹
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
