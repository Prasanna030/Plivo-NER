# PII Entity Recognition for Noisy STT Transcripts

Token-level NER system that flags PII (credit cards, phones, emails, names, dates) in punctuation-free STT transcripts and emits character-level spans plus a `pii` flag. The current best checkpoint uses **DistilBERT** and prioritizes strong PII precision while keeping CPU latency under the 20 ms requirement.

## Repo highlights

- `generate_data.py` – synthesizes 800 train / 150 dev utterances with noisy STT patterns, spelled-out numbers, and multi-entity sentences.
- `src/` – dataset loader, label map, model/training/inference scripts, span metrics, latency probe.
- `models/` – saved checkpoints (`distilbert` is primary, `bert_base` kept for reference).
-  model weights and predictions for dev data are in google drive link attached below

## Quickstart

```bash
pip install -r requirements.txt

# optional: regenerate synthetic data (already populated)
python generate_data.py

# train DistilBERT (saves to models/distilbert by default)
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl

# run inference + evaluation on dev
python src/predict.py \
  --input data/dev.jsonl \
  --output models/distilbert/dev_pred.json

python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred models/distilbert/dev_pred.json

# latency (CPU, batch size 1)
python src/measure_latency.py \
  --input data/dev.jsonl \
  --runs 50 \
  --device cpu

# optional: enable wandb logging
python src/train.py \
  --wandb_project your_project \
  --wandb_mode online
```

### Optional: train the heavier BERT baseline

```bash
python src/train.py \
  --model_name bert-base-uncased \
  --out_dir models/bert_base \
  --train data/train.jsonl \
  --dev data/dev.jsonl
```

## Results (dev set)

| Model | Macro F1 | PII Precision / Recall / F1 | Notes |
|-------|---------:|-----------------------------|-------|
| DistilBERT (primary) | **0.806** | **0.860 / 0.804 / 0.831** | Includes span-quality heuristics + label-aware thresholds (default `models/distilbert`). |
| BERT-base (reference) | 0.776 | 0.762 / 0.790 / 0.776 | Slightly higher recall but slower (p95 ≈ 22 ms on CPU). |

Per-entity F1 for the DistilBERT model after post-processing: CITY 0.698, CREDIT_CARD 0.716, DATE 0.914, EMAIL 0.884, LOCATION 0.816, PERSON_NAME 0.825, PHONE 0.792.

## Latency

Measured with `python src/measure_latency.py --model_dir models/distilbert --device cpu --runs 50` on a single CPU thread:

- **p50**: 7.36 ms
- **p95**: 11.19 ms

The CPU latency easily satisfies the ≤ 20 ms requirement. Use the `--max_length` flag (default 256) to trade accuracy vs. speed if needed.


## WandB finetuning
  1. `wandb login`
  2. `wandb sweep wandb_sweep.yaml`
  3. `wandb agent <entity>/<project>/<sweep_id>`

Report - https://drive.google.com/file/d/1tq61EkdTKrXI18qVM5U9yKJodhIW16pB/view?usp=sharing

## Hyperparams:
Training Loop – batch_size=8 , eval_batch_size=8, epochs=8
Optimization – AdamW, lr=3e-5 , weight_decay=0.0, linear schedule with warmup_ratio=0.1

## Trained Model Weights Drive Link 
https://drive.google.com/file/d/1onLqu7UpgMrNSdfhsTPliPBLDXf5CCVM/view?usp=sharing


