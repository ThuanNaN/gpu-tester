# GPU Tester

GPU stress-test toolkit — fine-tunes a **VLM** (Vision-Language Model) with LoRA on **synthetic data** (no downloads needed) and reports GPU health metrics.

## What it tests

- **Forward / backward pass** correctness (loss convergence, no NaN/Inf)
- **VRAM stability** (peak usage, no OOM)
- **Compute consistency** (step-time variance, no GPU throttling)
- **Inference** after training (generation sanity check)

## Setup

```bash
bash setup.sh
```

## Usage

```bash
# Default stress test on GPU 0
python train_vlm.py --device 0

# With 4-bit quantization (lower VRAM)
python train_vlm.py --device 0 --use_4bit

# Heavier stress (more steps, larger images)
python train_vlm.py --device 0 --max_steps 50 --img_size 512 --n_samples 100

# Use a larger model
python train_vlm.py --device 1 --model Qwen/Qwen2.5-VL-7B-Instruct --use_4bit
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `0` | GPU id (`0`, `1`, ...) |
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | Model name or path |
| `--n_samples` | `50` | Number of synthetic training samples |
| `--img_size` | `384` | Synthetic image resolution |
| `--max_steps` | `30` | Training steps |
| `--batch_size` | `1` | Per-device batch size |
| `--max_length` | `1024` | Max sequence length |
| `--use_4bit` | off | Enable 4-bit quantization |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--lora_rank` | `16` | LoRA rank |
| `--learning_rate` | `2e-4` | Learning rate |

## Health Report

After training the script prints a GPU health report:

```
  GPU HEALTH REPORT
  Step time  — avg: 1.23s | min: 1.10s | max: 1.45s | std: 0.08s
  ✓ Step times stable
  Loss trend — first half avg: 2.3456 | second half avg: 1.8901
  ✓ Loss is decreasing (model is learning)
  Peak VRAM — 6.42 GB
  VRAM util — 80.3% of 8.0 GB
  ✓ VRAM usage within safe limits
```

## Model

- **VLM**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — compact vision-language model
