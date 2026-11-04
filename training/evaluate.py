"""학습된 모델 평가. 모델 선택은 Val로, Test는 최종 모델 확정 후 1번만.
예) python -m training.evaluate --model artifacts/e3_plus_aug --split val
    python -m training.evaluate --model artifacts/e3_plus_aug --split val --robustness
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ai.preprocess import apply_preprocess, load_preprocess_method

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _jpeg(im, quality=20):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# 촬영 환경 변화 시뮬레이션: 학습 증강(밝기·대비 ±20%, 회전 20도)보다 강하게 설정
PERTURBATIONS = {
    "original": lambda im: im,
    "dark": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "bright": lambda im: ImageEnhance.Brightness(im).enhance(1.6),
    "low_contrast": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=3)),
    "rotate45": lambda im: im.rotate(45),
    "jpeg_q20": _jpeg,
}


def list_records(data, split, classes):
    records = []
    for idx, name in enumerate(classes):
        folder = Path(data) / split / name
        records += [(p, idx) for p in sorted(folder.iterdir())
                    if p.is_file() and p.suffix.lower() in EXTS]
    return records


@torch.no_grad()
def predict(model, processor, records, method, device, perturb=None, tta=False, batch_size=32):
    preds, scores, forward_sec = [], [], 0.0
    for start in range(0, len(records), batch_size):
        images = []
        for path, _ in records[start:start + batch_size]:
            with Image.open(path) as raw:
                im = raw.convert("RGB")
            if perturb is not None:
                im = perturb(im)                       # 촬영 조건 변화 → 그다음 모델 전처리
            images.append(apply_preprocess(im, method))
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        t0 = time.perf_counter()
        probs = model(pixel_values=x).logits.softmax(-1)
        if tta:      # 좌우반전 평균 (정확도 vs 추론시간 비교용)
            probs = (probs + model(pixel_values=torch.flip(x, dims=[3])).logits.softmax(-1)) / 2
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_sec += time.perf_counter() - t0

        score, pred = probs.max(-1)
        preds += pred.tolist(); scores += score.tolist()
    return preds, scores, forward_sec


def metrics(records, preds, classes):
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for (_, t), p in zip(records, preds):
        cm[t][p] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]; fn = sum(cm[i]) - tp; fp = sum(cm[r][i] for r in range(n)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": sum(cm[i])}
    acc = sum(cm[i][i] for i in range(n)) / len(records)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / n
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per_class, "confusion_matrix": {"labels": classes, "matrix": cm}}


def run(args):
    model_dir = Path(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    method = load_preprocess_method(model_dir)          # 학습 때 쓴 전처리 그대로
    records = list_records(args.data, args.split, classes)

    if args.robustness:
        acc = {}
        for name, fn in PERTURBATIONS.items():
            preds, _, _ = predict(model, processor, records, method, device, fn, args.tta, args.batch_size)
            acc[name] = metrics(records, preds, classes)["accuracy"]
        perturbed = [v for k, v in acc.items() if k != "original"]
        mean = round(sum(perturbed) / len(perturbed), 4)
        result = {"model": str(model_dir), "split": args.split, "preprocess": method,
                  "accuracy": acc, "perturbed_mean": mean, "perturbed_min": min(perturbed)}
        (model_dir / f"eval_{args.split}_robustness.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("| 모델 | " + " | ".join(acc) + " | 변형 평균 | 변형 최저 |")
        print("|---|" + "---:|" * (len(acc) + 2))
        print(f"| {model_dir.name} | " + " | ".join(f"{v:.4f}" for v in acc.values())
              + f" | **{mean:.4f}** | {min(perturbed):.4f} |")
        return

    preds, scores, forward_sec = predict(model, processor, records, method, device,
                                         None, args.tta, args.batch_size)
    result = {"model": str(model_dir), "split": args.split, "tta": args.tta, "preprocess": method,
              "count": len(records), **metrics(records, preds, classes),
              "forward_ms_per_image": round(forward_sec * 1000 / len(records), 2),
              "device": device.type}
    tag = f"eval_{args.split}{'_tta' if args.tta else ''}"
    (model_dir / f"{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    wrong = [{"file": str(p), "true": classes[t], "pred": classes[q], "score": round(s, 4)}
             for (p, t), q, s in zip(records, preds, scores) if t != q]
    with open(model_dir / f"{tag}_wrong.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true", "pred", "score"])
        w.writeheader(); w.writerows(wrong)

    cm = result["confusion_matrix"]["matrix"]
    print(f"\n[{model_dir.name} / {args.split}] accuracy={result['accuracy']:.4f}, "
          f"macro F1={result['macro_f1']:.4f} ({len(records)}장, TTA={args.tta}, 전처리={method}, "
          f"추론 {result['forward_ms_per_image']}ms/장)")
    print("| 정답 \\ 예측 | " + " | ".join(classes) + " |\n|---|" + "---:|" * len(classes))
    for c, row in zip(classes, cm):
        print(f"| {c} | " + " | ".join(map(str, row)) + " |")
    for c, v in result["per_class"].items():
        print(f"- {c}: precision {v['precision']}, recall {v['recall']}, f1 {v['f1']}")
    print(f"저장: {model_dir / tag}.json / 오분류 {len(wrong)}건")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="dataset")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--robustness", action="store_true", help="촬영 조건 변화에 대한 강건성 평가")
    p.add_argument("--batch-size", type=int, default=32)
    run(p.parse_args())