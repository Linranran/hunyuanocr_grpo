# HunyuanOCR GRPO experiments

Sanitized training and data-generation scripts for GRPO experiments on OCR/VLM tasks.

## Contents

- `swift_grpo_train_qwen.sh`, `qwen25_vl_grpo_train*.py`: GRPO training entry points.
- `generate_grpo*.py`: candidate generation and reward dataset preparation.
- `GRPO_TRAINING_GUIDE.md`: setup notes.

Model, checkpoint, and dataset paths are placeholders. Set paths for your own environment before running. HTTP generation scripts read `MODEL_API_BASE_URL` and `MODEL_API_KEY` from environment variables. Keep credentials and private datasets out of Git.

This repository does not include model weights, checkpoints, OCR outputs, or training/evaluation datasets.
