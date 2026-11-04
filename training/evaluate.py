"""학습된 모델 평가. 모델 선택은 Val로, Test는 최종 모델 확정 후 1번만.
예) python -m training.evaluate --model artifacts/beans_resnet18 --split val
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ai.preprocess import apply_preprocess, load_preprocess_method

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@torch.no_grad()
def run(args):
    model_dir = Path(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    method = load_preprocess_method(model_dir)          # 학습 때 쓴 전처리 그대로

    records = []
    for idx, name in enumerate(classes):
        folder = Path(args.data) / args.split / name
        records += [(p, idx) for p in sorted(folder.iterdir())
                    if p.is_file() and p.suffix.lower() in EXTS]

    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    wrong, forward_sec = [], 0.0
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        images = []
        for path, _ in chunk:
            with Image.open(path) as raw:
                images.append(apply_preprocess(raw.convert("RGB"), method))
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        t0 = time.perf_counter()
        probs = model(pixel_values=x).logits.softmax(-1)
        if args.tta:      # 좌우반전 평균 (정확도 vs 추론시간 비교용)
            probs = (probs + model(pixel_values=torch.flip(x, dims=[3])).logits.softmax(-1)) / 2
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_sec += time.perf_counter() - t0

        score, pred = probs.max(-1)
        for (path, t), p, s in zip(chunk, pred.tolist(), score.tolist()):
            cm[t][p] += 1
            if t != p:
                wrong.append({"file": str(path), "true": classes[t],
                              "pred": classes[p], "score": round(s, 4)})

    total = len(records)
    acc = sum(cm[i][i] for i in range(n)) / total
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]; fn = sum(cm[i]) - tp; fp = sum(cm[r][i] for r in range(n)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": sum(cm[i])}

    tag = f"eval_{args.split}{'_tta' if args.tta else ''}"
    result = {"model": str(model_dir), "split": args.split, "tta": args.tta, "preprocess": method,
              "count": total, "accuracy": round(acc, 4),
              "macro_f1": round(sum(v["f1"] for v in per_class.values()) / n, 4),
              "per_class": per_class, "confusion_matrix": {"labels": classes, "matrix": cm},
              "forward_ms_per_image": round(forward_sec * 1000 / total, 2), "device": device.type}
    (model_dir / f"{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(model_dir / f"{tag}_wrong.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true", "pred", "score"])
        w.writeheader(); w.writerows(wrong)

    print(f"\n[{model_dir.name} / {args.split}] accuracy={acc:.4f}, macro F1={result['macro_f1']:.4f} "
          f"({total}장, TTA={args.tta}, 전처리={method})")
    print("| 정답 \\ 예측 | " + " | ".join(classes) + " |\n|---|" + "---:|" * n)
    for c, row in zip(classes, cm):
        print(f"| {c} | " + " | ".join(map(str, row)) + " |")
    for c, v in per_class.items():
        print(f"- {c}: precision {v['precision']}, recall {v['recall']}, f1 {v['f1']}")
    print(f"저장: {model_dir / tag}.json / 오분류 {len(wrong)}건")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="dataset")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    run(p.parse_args())