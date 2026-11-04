"""개선 학습 (training/finetune.py는 baseline 비교용으로 그대로 둔다)
예) python -m training.finetune_plus --output artifacts/e3_plus_aug
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ai.preprocess import apply_preprocess, save_preprocess_config

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_augment():
    # 학습 데이터에만 적용. 색조(hue)는 건드리지 않음: 병반의 노란색·갈색이 핵심 단서라서
    return T.Compose([
        T.RandomResizedCrop(256, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class FolderImages(Dataset):
    def __init__(self, root, classes, processor, augment=None, preprocess="none"):
        self.records = []
        for idx, name in enumerate(classes):
            folder = Path(root) / name
            if not folder.is_dir():
                raise ValueError(f"클래스 폴더가 없습니다: {folder}")
            self.records += [(p, idx) for p in sorted(folder.iterdir())
                             if p.is_file() and p.suffix.lower() in EXTS]
        if not self.records:
            raise ValueError(f"이미지가 없습니다: {root}")
        self.processor, self.augment, self.preprocess = processor, augment, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, label = self.records[i]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        image = apply_preprocess(image, self.preprocess)      # 서빙과 동일한 전처리
        if self.augment is not None:
            image = self.augment(image)
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, label


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def set_backbone(model, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith("classifier."):
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device, num_classes, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    hit, cnt = torch.zeros(num_classes), torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(pixel_values=x).logits
        total_loss += loss_fn(logits, y).item() * len(y)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item(); total += len(y)
        for c in range(num_classes):
            m = y == c
            cnt[c] += m.sum().item(); hit[c] += (pred[m] == c).sum().item()
    return total_loss / total, correct / total, (hit / cnt.clamp(min=1)).tolist()


def run(args):
    set_seed(args.seed)
    data = Path(args.data)
    classes = sorted(p.name for p in (data / "train").iterdir() if p.is_dir())
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    augment = build_augment() if args.augment == "basic" else None
    train_set = FolderImages(data / "train", classes, processor, augment, args.preprocess)
    val_set = FolderImages(data / "val", classes, processor, None, args.preprocess)
    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        args.base_model, num_labels=len(classes),
        id2label={i: n for i, n in enumerate(classes)},
        label2id={n: i for i, n in enumerate(classes)},
        ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 차등 학습률: 새 출력층은 크게, 사전학습된 Backbone은 작게
    head = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    optim = torch.optim.AdamW([{"params": body, "lr": args.lr_backbone},
                               {"params": head, "lr": args.lr_head}],
                              weight_decay=args.weight_decay)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # warmup 1 epoch + 코사인 감소
    total_steps, warmup = args.epochs * len(train_loader), len(train_loader)
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, wait = [], (-1.0, float("inf")), 0, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        frozen = args.freeze_backbone or epoch <= args.freeze_epochs   # 2단계 학습
        set_backbone(model, not frozen)
        model.train()
        if frozen and hasattr(model, "resnet"):
            model.resnet.eval()          # 고정 구간에서는 BatchNorm 통계도 고정
        losses, correct, seen = [], 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(pixel_values=x).logits
                loss = train_loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); sched.step()
            losses.append(loss.item())
            correct += (logits.argmax(-1) == y).sum().item(); seen += len(y)

        val_loss, val_acc, recall = evaluate(model, val_loader, device, len(classes), val_loss_fn)
        row = {"epoch": epoch, "stage": "head" if frozen else "full",
               "loss": round(sum(losses) / len(losses), 4),
               "train_accuracy": round(correct / seen, 4),
               "val_loss": round(val_loss, 4), "val_accuracy": round(val_acc, 4),
               "val_recall": {c: round(r, 4) for c, r in zip(classes, recall)}}
        history.append(row); print(row)

        # best 기준: Val 정확도, 같으면 Val Loss가 낮은 쪽
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best, best_epoch, wait = (val_acc, val_loss), epoch, 0
            model.save_pretrained(str(out), safe_serialization=True)
            processor.save_pretrained(str(out))
            save_preprocess_config(out, args.preprocess)
        else:
            wait += 1
            if args.patience and wait >= args.patience:
                print(f"조기 종료: {args.patience} epoch 동안 개선 없음")
                break

    report = {"base_model": args.base_model, "method": "finetune_plus", "classes": classes,
              "train_count": len(train_set), "val_count": len(val_set),
              "best_val_accuracy": best[0], "best_val_loss": round(best[1], 4),
              "best_epoch": best_epoch, "train_seconds": round(time.time() - started, 1),
              "config": vars(args), "history": history}
    (out / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["epoch", "stage", "loss", "train_accuracy", "val_loss", "val_accuracy"]
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + [f"recall_{c}" for c in classes])
        w.writeheader()
        for r in history:
            w.writerow({**{k: r[k] for k in fields},
                        **{f"recall_{c}": v for c, v in r["val_recall"].items()}})
    print(f"저장 완료: {out} / best epoch={best_epoch}, best val accuracy={best[0]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--output", default="artifacts/beans_resnet18_plus")
    p.add_argument("--base-model", default="microsoft/resnet-18")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--freeze-epochs", type=int, default=2, help="처음 N epoch은 출력층만 학습")
    p.add_argument("--freeze-backbone", action="store_true", help="끝까지 출력층만 학습(비교용)")
    p.add_argument("--augment", choices=["basic", "none"], default="basic")
    p.add_argument("--preprocess", choices=["none", "clahe"], default="none")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())