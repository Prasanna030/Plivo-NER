import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, LABEL2ID, label_is_pii
import os


DIGIT_WORDS = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
}

LABEL_THRESHOLDS = {
    "CREDIT_CARD": 0.55,
    "PHONE": 0.55,
    "EMAIL": 0.60,
    "PERSON_NAME": 0.45,
    "DATE": 0.45,
}


def count_digit_tokens(segment: str) -> int:
    tokens = segment.lower().split()
    count = 0
    for tok in tokens:
        clean = ''.join(ch for ch in tok if ch.isalnum())
        if not clean:
            continue
        if clean.isdigit():
            count += len(clean)
        elif clean in DIGIT_WORDS:
            count += 1
    return count


def span_passes_quality(text: str, start: int, end: int, label: str) -> bool:
    segment = text[start:end].lower()
    if label == "CREDIT_CARD":
        return count_digit_tokens(segment) >= 12
    if label == "PHONE":
        return count_digit_tokens(segment) >= 7
    if label == "EMAIL":
        return " at " in segment and " dot " in segment
    if label == "PERSON_NAME":
        return " " in segment  # at least first + last name
    if label == "DATE":
        return any(m in segment for m in (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ))
    return True


def _token_left(text: str, idx: int):
    j = idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0:
        return None, idx
    end = j + 1
    while j >= 0 and not text[j].isspace():
        j -= 1
    start = j + 1
    return text[start:end], start


def _token_right(text: str, idx: int):
    n = len(text)
    j = idx
    while j < n and text[j].isspace():
        j += 1
    if j >= n:
        return None, idx
    start = j
    while j < n and not text[j].isspace():
        j += 1
    end = j
    return text[start:end], end


def _is_digit_like(token: str) -> bool:
    clean = ''.join(ch for ch in token.lower() if ch.isalnum())
    return bool(clean) and (clean.isdigit() or clean in DIGIT_WORDS)


def expand_numeric_span(text: str, start: int, end: int) -> tuple[int, int]:
    changed = True
    while changed:
        changed = False
        token, new_start = _token_left(text, start)
        if token and _is_digit_like(token):
            start = new_start
            changed = True
            continue
        token, new_end = _token_right(text, end)
        if token and _is_digit_like(token):
            end = new_end
            changed = True
    return start, end


def bio_to_spans(text, offsets, label_ids, token_probs):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_token_idxs = []

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None and current_token_idxs:
                score = sum(token_probs[i] for i in current_token_idxs) / len(current_token_idxs)
                spans.append((current_start, current_end, current_label, score))
            current_label = None
            current_token_idxs = []
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None and current_token_idxs:
                score = sum(token_probs[i] for i in current_token_idxs) / len(current_token_idxs)
                spans.append((current_start, current_end, current_label, score))
            current_label = ent_type
            current_start = start
            current_end = end
            current_token_idxs = [idx]
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                current_token_idxs.append(idx)
            else:
                if current_label is not None and current_token_idxs:
                    score = sum(token_probs[i] for i in current_token_idxs) / len(current_token_idxs)
                    spans.append((current_start, current_end, current_label, score))
                current_label = ent_type
                current_start = start
                current_end = end
                current_token_idxs = [idx]

    if current_label is not None and current_token_idxs:
        score = sum(token_probs[i] for i in current_token_idxs) / len(current_token_idxs)
        spans.append((current_start, current_end, current_label, score))

    return spans


def generate_predictions(model, tokenizer, samples, max_length=256, threshold=0.0, device="cpu"):
    was_training = model.training
    model.eval()
    results = {}
    for sample in samples:
        text = sample["text"]
        uid = sample["id"]
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0]
            probs = torch.softmax(logits, dim=-1)
            max_prob, pred_ids = probs.max(dim=-1)

            max_prob_list = max_prob.cpu().tolist()
            pred_id_list = pred_ids.cpu().tolist()

            adjusted_ids = []
            for pid, mp in zip(pred_id_list, max_prob_list):
                label = ID2LABEL.get(int(pid), "O")
                ent_type = label.split("-", 1)[1] if "-" in label else None
                required = LABEL_THRESHOLDS.get(ent_type, threshold)
                if mp < required:
                    adjusted_ids.append(LABEL2ID["O"])
                else:
                    adjusted_ids.append(pid)
            pred_ids = adjusted_ids

        spans = bio_to_spans(text, offsets, pred_ids, max_prob_list)
        best_by_label = {}
        for s, e, lab, score in spans:
            if lab in {"CREDIT_CARD", "PHONE"}:
                s, e = expand_numeric_span(text, s, e)
            if not span_passes_quality(text, s, e, lab):
                continue
            existing = best_by_label.get(lab)
            if existing is None or score > existing[3]:
                best_by_label[lab] = (s, e, lab, score)

        ents = []
        for s, e, lab, _ in best_by_label.values():
            ents.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                }
            )
        results[uid] = ents

    if was_training:
        model.train()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/distilbert")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"id": obj["id"], "text": obj["text"]})

    results = generate_predictions(
        model,
        tokenizer,
        samples,
        max_length=args.max_length,
        threshold=args.threshold,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
