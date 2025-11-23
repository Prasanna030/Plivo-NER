import json
import time
import argparse
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def measure_latency_stats(model_dir, model_name=None, input_path="data/dev.jsonl", max_length=256, runs=50, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir if model_name is None else model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        return None

    times_ms = []

    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))

    for i in range(runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]
    return {"runs": runs, "p50_ms": p50, "p95_ms": p95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/distilbert")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    stats = measure_latency_stats(
        args.model_dir,
        model_name=args.model_name,
        input_path=args.input,
        max_length=args.max_length,
        runs=args.runs,
        device=args.device,
    )

    if stats is None:
        print("No texts found in input file.")
        return

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {stats['p50_ms']:.2f} ms")
    print(f"  p95: {stats['p95_ms']:.2f} ms")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved latency stats to {args.json_out}")


if __name__ == "__main__":
    main()
