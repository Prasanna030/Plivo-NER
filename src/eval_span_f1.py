import json
import argparse
from collections import defaultdict
from labels import label_is_pii


def load_gold(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["id"]
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            gold[uid] = spans
    return gold


def load_pred(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pred = {}
    for uid, ents in obj.items():
        spans = []
        for e in ents:
            spans.append((e["start"], e["end"], e["label"]))
        pred[uid] = spans
    return pred


def compute_prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def evaluate_metrics(gold_dict, pred_dict):
    labels = set()
    for spans in gold_dict.values():
        for _, _, lab in spans:
            labels.add(lab)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for uid in gold_dict.keys():
        g_spans = set(gold_dict.get(uid, []))
        p_spans = set(pred_dict.get(uid, []))

        for span in p_spans:
            if span in g_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in g_spans:
            if span not in p_spans:
                fn[span[2]] += 1

    per_label = {}
    macro_f1_sum = 0.0
    macro_count = 0

    for lab in sorted(labels):
        p, r, f1 = compute_prf(tp[lab], fp[lab], fn[lab])
        per_label[lab] = {"precision": p, "recall": r, "f1": f1}
        macro_f1_sum += f1
        macro_count += 1

    macro_f1 = macro_f1_sum / max(1, macro_count)

    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0

    for uid in gold_dict.keys():
        g_spans = gold_dict.get(uid, [])
        p_spans = pred_dict.get(uid, [])

        g_pii = set((s, e, "PII") for s, e, lab in g_spans if label_is_pii(lab))
        g_non = set((s, e, "NON") for s, e, lab in g_spans if not label_is_pii(lab))
        p_pii = set((s, e, "PII") for s, e, lab in p_spans if label_is_pii(lab))
        p_non = set((s, e, "NON") for s, e, lab in p_spans if not label_is_pii(lab))

        for span in p_pii:
            if span in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for span in g_pii:
            if span not in p_pii:
                pii_fn += 1

        for span in p_non:
            if span in g_non:
                non_tp += 1
            else:
                non_fp += 1
        for span in g_non:
            if span not in p_non:
                non_fn += 1

    pii_p, pii_r, pii_f1 = compute_prf(pii_tp, pii_fp, pii_fn)
    non_p, non_r, non_f1 = compute_prf(non_tp, non_fp, non_fn)

    return {
        "macro_f1": macro_f1,
        "per_label": per_label,
        "pii_precision": pii_p,
        "pii_recall": pii_r,
        "pii_f1": pii_f1,
        "non_pii_precision": non_p,
        "non_pii_recall": non_r,
        "non_pii_f1": non_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    metrics = evaluate_metrics(gold, pred)

    print("Per-entity metrics:")
    for lab in sorted(metrics["per_label"].keys()):
        stats = metrics["per_label"][lab]
        print(f"{lab:15s} P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")

    print(f"\nMacro-F1: {metrics['macro_f1']:.3f}")
    print(
        f"\nPII-only metrics: P={metrics['pii_precision']:.3f} R={metrics['pii_recall']:.3f} F1={metrics['pii_f1']:.3f}"
    )
    print(
        f"Non-PII metrics: P={metrics['non_pii_precision']:.3f} R={metrics['non_pii_recall']:.3f} F1={metrics['non_pii_f1']:.3f}"
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to {args.json_out}")


if __name__ == "__main__":
    main()
