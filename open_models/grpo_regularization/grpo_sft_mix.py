import json
import random
from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from trl import GRPOTrainer


def load_sft_dataset(file_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            msgs = obj.get("messages", [])

            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            data.append(tokenized)

    random.shuffle(data)
    return Dataset.from_list(data)


class GRPOSFTMixTrainer(GRPOTrainer):
    """
    GRPO trainer that mixes in SFT steps during training.
    Every `sft_mix_ratio` steps, an additional SFT cross-entropy loss is added.
    """

    def __init__(
        self,
        *args,
        sft_dataset: Optional[Dataset] = None,
        sft_mix_ratio: int = 4,
        sft_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sft_dataset = sft_dataset
        self.sft_mix_ratio = sft_mix_ratio
        self.sft_loss_weight = sft_loss_weight
        self.sft_step_counter = 0

        if sft_dataset is not None:
            self.sft_collator = DataCollatorForSeq2Seq(
                tokenizer=self.processing_class,
                padding=True,
                return_tensors="pt",
            )
            self.sft_dataloader = DataLoader(
                sft_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.sft_collator,
                drop_last=True,
            )
            self.sft_dataloader_iter = iter(self.sft_dataloader)
            print(
                f"SFT mixing enabled: {len(sft_dataset)} samples, mix ratio 1:{sft_mix_ratio}"
            )

        self._accum_sft_losses = []

    def _get_sft_batch(self):
        try:
            return next(self.sft_dataloader_iter)
        except StopIteration:
            self.sft_dataloader_iter = iter(self.sft_dataloader)
            return next(self.sft_dataloader_iter)

    def _compute_sft_loss(self, model, batch):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits
            vocab_size = logits.size(-1)

            # --- diagnostics + safety mask for added tokens ---
            print(f"[GRPOSFTMix] logits vocab_size: {vocab_size}")
            print(
                f"[GRPOSFTMix] labels max: {labels.max().item()}, min: {labels.min().item()}"
            )
            print(f"[GRPOSFTMix] tokenizer len: {len(self.processing_class)}")

            invalid_mask = labels >= vocab_size
            if invalid_mask.any():
                n_invalid = invalid_mask.sum().item()
                print(
                    f"[GRPOSFTMix] WARNING: masking {n_invalid} label tokens >= vocab_size"
                )
                labels = labels.clone()
                labels[invalid_mask] = -100

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.sft_step_counter += 1

        if return_outputs:
            grpo_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
        else:
            grpo_loss = super().compute_loss(
                model, inputs, return_outputs=False, **kwargs
            )
            outputs = None

        if (
            self.sft_dataset is not None
            and self.sft_step_counter % self.sft_mix_ratio == 0
        ):
            sft_batch = self._get_sft_batch()
            sft_loss = self._compute_sft_loss(model, sft_batch)
            total_loss = grpo_loss + self.sft_loss_weight * sft_loss
            self._accum_sft_losses.append(sft_loss.detach().float().cpu().item())
        else:
            total_loss = grpo_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, start_time=None):
        if self._accum_sft_losses:
            logs["sft_loss"] = sum(self._accum_sft_losses) / len(self._accum_sft_losses)
            self._accum_sft_losses.clear()
        return super().log(logs, start_time=start_time)
