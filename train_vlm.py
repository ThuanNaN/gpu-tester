"""
GPU Stress Test: VLM Fine-tuning
Fine-tune Qwen2.5-VL with LoRA using synthetic data to stress-test GPU health.
No external dataset needed — generates images + Q&A pairs locally.
"""
import argparse
import gc
import random
import time
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Synthetic dataset generation (no network required)
# ---------------------------------------------------------------------------
SHAPES = ["circle", "rectangle", "triangle", "star", "ellipse"]
COLORS = {
    "red": (220, 50, 50), "green": (50, 180, 50), "blue": (50, 80, 220),
    "yellow": (230, 220, 50), "purple": (160, 50, 200), "orange": (240, 150, 30),
    "cyan": (50, 210, 210), "white": (240, 240, 240), "pink": (240, 130, 180),
}
BG_COLORS = [(30, 30, 30), (60, 60, 80), (20, 40, 20), (50, 30, 30), (40, 40, 60)]


def _draw_shape(draw, shape, color_rgb, cx, cy, size):
    """Draw a single shape centered at (cx, cy)."""
    s = size // 2
    if shape == "circle":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color_rgb)
    elif shape == "rectangle":
        draw.rectangle([cx - s, cy - s // 2 * 3, cx + s, cy + s // 2 * 3], fill=color_rgb)
    elif shape == "triangle":
        draw.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=color_rgb)
    elif shape == "star":
        pts = []
        for i in range(10):
            angle = i * 36 - 90
            r = s if i % 2 == 0 else s // 2
            pts.append((cx + r * np.cos(np.radians(angle)),
                         cy + r * np.sin(np.radians(angle))))
        draw.polygon(pts, fill=color_rgb)
    elif shape == "ellipse":
        draw.ellipse([cx - s, cy - s // 2, cx + s, cy + s // 2], fill=color_rgb)


def generate_synthetic_sample(img_size=384):
    """Create one synthetic image with random shapes and a matching Q&A pair."""
    bg = random.choice(BG_COLORS)
    img = Image.new("RGB", (img_size, img_size), bg)
    draw = ImageDraw.Draw(img)

    n_shapes = random.randint(1, 4)
    placed = []
    for _ in range(n_shapes):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color_rgb = COLORS[color_name]
        size = random.randint(40, 100)
        cx = random.randint(size, img_size - size)
        cy = random.randint(size, img_size - size)
        _draw_shape(draw, shape, color_rgb, cx, cy, size)
        placed.append((color_name, shape))

    # Build a descriptive Q&A
    description_parts = []
    for c, s in placed:
        description_parts.append(f"a {c} {s}")
    desc = ", ".join(description_parts[:-1])
    if len(description_parts) > 1:
        desc += f" and {description_parts[-1]}"
    else:
        desc = description_parts[0]

    templates_q = [
        "What shapes and colors do you see in this image?",
        "Describe the geometric objects in this image.",
        "List all shapes visible in this picture.",
        "What can you see in this image?",
        "Identify the colored shapes in this image.",
    ]
    templates_a = [
        f"The image contains {desc} on a dark background.",
        f"I can see {desc} drawn on a dark canvas.",
        f"There are {len(placed)} shape(s) in the image: {desc}.",
        f"This image shows {desc}.",
    ]

    return {
        "image": img,
        "question": random.choice(templates_q),
        "answer": random.choice(templates_a),
    }


def build_synthetic_dataset(n_samples=50, img_size=384):
    """Build an in-memory dataset of synthetic VLM samples as a list of dicts."""
    samples = []
    for _ in range(n_samples):
        sample = generate_synthetic_sample(img_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
        samples.append({"messages": messages})
    return samples


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
def collate_fn(examples, processor, max_length):
    """Collate function for VLM training with image + text."""
    texts = []
    all_images = []

    for example in examples:
        messages = example["messages"]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

        image_inputs, _ = process_vision_info(messages)
        if image_inputs:
            all_images.extend(image_inputs)

    batch = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor, "image_token_id") and processor.image_token_id is not None:
        labels[labels == processor.image_token_id] = -100
    batch["labels"] = labels
    return batch


# ---------------------------------------------------------------------------
# GPU health monitoring callback
# ---------------------------------------------------------------------------
class GPUHealthCallback(TrainerCallback):
    """Track VRAM, loss, and step time for GPU health reporting."""

    def __init__(self):
        self.step_times = []
        self.losses = []
        self.vram_usage = []
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self.step_times.append(time.time() - self._step_start)
        if torch.cuda.is_available():
            self.vram_usage.append(torch.cuda.max_memory_allocated() / 1024**3)

    def report(self):
        """Print a GPU health summary."""
        print(f"\n{'=' * 60}")
        print("  GPU HEALTH REPORT")
        print(f"{'=' * 60}")

        # Step time stats
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times)
            max_step = max(self.step_times)
            min_step = min(self.step_times)
            std_step = (sum((t - avg_step) ** 2 for t in self.step_times) / len(self.step_times)) ** 0.5
            print(f"  Step time  — avg: {avg_step:.2f}s | min: {min_step:.2f}s | max: {max_step:.2f}s | std: {std_step:.3f}s")
            # Check for anomalies (step time variance)
            if std_step / avg_step > 0.5 and len(self.step_times) > 5:
                print("  ⚠ WARNING: High step-time variance — possible GPU throttling or instability")
            else:
                print("  ✓ Step times stable")

        # Loss trend
        if len(self.losses) >= 2:
            first_half = self.losses[: len(self.losses) // 2]
            second_half = self.losses[len(self.losses) // 2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            print(f"  Loss trend — first half avg: {avg_first:.4f} | second half avg: {avg_second:.4f}")
            if avg_second < avg_first:
                print("  ✓ Loss is decreasing (model is learning)")
            elif any(np.isnan(l) or np.isinf(l) for l in self.losses):
                print("  ✗ FAIL: NaN/Inf loss detected — GPU compute error likely")
            else:
                print("  ~ Loss not decreasing (may need more steps or higher LR for real training)")
        elif len(self.losses) == 1:
            if np.isnan(self.losses[0]) or np.isinf(self.losses[0]):
                print("  ✗ FAIL: NaN/Inf loss detected")
            else:
                print(f"  Loss: {self.losses[0]:.4f}")

        # VRAM stats
        if self.vram_usage:
            peak_vram = max(self.vram_usage)
            print(f"  Peak VRAM — {peak_vram:.2f} GB")
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                usage_pct = peak_vram / total_vram * 100
                print(f"  VRAM util — {usage_pct:.1f}% of {total_vram:.1f} GB")
                if usage_pct > 95:
                    print("  ⚠ WARNING: Near VRAM limit — risk of OOM")
                else:
                    print("  ✓ VRAM usage within safe limits")

        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU Stress Test - VLM Fine-tuning")
    parser.add_argument("--device", type=str, default="0", help="GPU id (0, 1, ...)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="VLM model")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of synthetic samples")
    parser.add_argument("--img_size", type=int, default=384, help="Synthetic image size")
    parser.add_argument("--max_steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--use_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accum steps")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("=" * 60)
    print("  GPU STRESS TEST: VLM Fine-tuning (LoRA)")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {total_vram:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("  WARNING: No GPU detected!")

    print(f"  Model: {args.model}")
    print(f"  Synthetic samples: {args.n_samples} ({args.img_size}x{args.img_size})")
    print(f"  Steps: {args.max_steps} | Batch: {args.batch_size} | Grad accum: {args.gradient_accumulation_steps}")
    print(f"  LoRA rank: {args.lora_rank} | 4-bit: {args.use_4bit}")
    print("=" * 60)

    # --- Phase 1: Generate synthetic dataset ---
    print("\n[1/4] Generating synthetic dataset...")
    t0 = time.time()
    dataset = build_synthetic_dataset(args.n_samples, args.img_size)
    print(f"  → {len(dataset)} samples generated in {time.time() - t0:.1f}s")

    # --- Phase 2: Load model ---
    print("\n[2/4] Loading model & processor...")
    t0 = time.time()

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    param_total = sum(p.numel() for p in model.parameters())
    print(f"  → Model loaded in {time.time() - t0:.1f}s ({param_total / 1e6:.0f}M params)")

    if torch.cuda.is_available():
        print(f"  → VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # --- Phase 3: LoRA fine-tuning ---
    print("\n[3/4] Starting LoRA fine-tuning stress test...")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./outputs/vlm_lora",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=5,
        save_steps=args.max_steps,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        max_length=args.max_length,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    health_cb = GPUHealthCallback()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        peft_config=lora_config,
        data_collator=lambda examples: collate_fn(examples, processor, args.max_length),
        callbacks=[health_cb],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Training done in {train_elapsed:.1f}s")
    print(f"  Trainable params (LoRA): {trainable / 1e6:.2f}M / {param_total / 1e6:.0f}M ({trainable / param_total * 100:.2f}%)")

    # --- Phase 4: Inference sanity check ---
    print("\n[4/4] Post-training inference check...")
    model.eval()

    # Generate a fresh synthetic image for testing
    test_sample = generate_synthetic_sample(args.img_size)
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_sample["image"]},
                {"type": "text", "text": "Describe the shapes and colors you see."},
            ],
        }
    ]

    text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(test_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_time = time.time() - gen_start
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs.shape[1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    print(f"  Generated {new_tokens} tokens in {gen_time:.2f}s ({new_tokens / gen_time:.1f} tok/s)")
    print(f"  Response: {response[:300]}")

    if new_tokens > 0 and len(response.strip()) > 0:
        print("  ✓ Inference OK")
    else:
        print("  ✗ FAIL: No output generated")

    # --- Health report ---
    health_cb.report()

    # Final summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total time:  {train_elapsed + gen_time:.1f}s")
    print(f"  Train steps: {args.max_steps} @ {train_elapsed / args.max_steps:.2f}s/step")
    print(f"  Inference:   {new_tokens} tokens @ {new_tokens / gen_time:.1f} tok/s")
    if torch.cuda.is_available():
        print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
