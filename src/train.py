import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="models/distilbert")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb_project", default=None, help="Weights & Biases project name")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_tags", default=None, help="Comma-separated list of tags")
    ap.add_argument(
        "--wandb_mode",
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Set to online/offline to enable wandb logging",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_dl = None
    if args.dev and os.path.exists(args.dev):
        dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
        dev_dl = DataLoader(
            dev_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps
    )

    wandb_run = setup_wandb(args)

    def evaluate():
        if dev_dl is None:
            return {}
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        with torch.no_grad():
            for batch in dev_dl:
                input_ids = torch.tensor(batch["input_ids"], device=args.device)
                attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
                labels = torch.tensor(batch["labels"], device=args.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                total_correct += ((preds == labels) & mask).sum().item()
                total_tokens += mask.sum().item()
        model.train()
        avg_loss = total_loss / max(1, len(dev_dl))
        acc = total_correct / max(1, total_tokens)
        print(f"Dev loss: {avg_loss:.4f} | token accuracy: {acc:.4f}")
        return {"dev_loss": avg_loss, "dev_token_acc": acc}

    start = time.time()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        log_payload = {"epoch": epoch + 1, "train_loss": avg_loss}
        eval_metrics = evaluate()
        log_payload.update(eval_metrics)
        if wandb_run:
            wandb_run.log(log_payload)

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")
    if wandb_run:
        wandb_run.log({"training_time_sec": time.time() - start})
        wandb_run.finish()


def setup_wandb(args):
    project = args.wandb_project or os.getenv("WANDB_PROJECT")
    if args.wandb_mode == "disabled" or not project:
        return None
    try:
        import wandb
    except ImportError:  # pragma: no cover
        print("wandb not installed; disable logging or install wandb.")
        return None

    tags = args.wandb_tags.split(",") if args.wandb_tags else None
    run = wandb.init(
        project=project,
        entity=args.wandb_entity or os.getenv("WANDB_ENTITY"),
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tags,
        mode=args.wandb_mode,
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
        },
    )
    return run


if __name__ == "__main__":
    main()
