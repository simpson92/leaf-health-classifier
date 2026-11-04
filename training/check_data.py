"""데이터 배치 + 점검
예) python -m training.check_data --source ~/beans_source --dataset dataset   # 복사 + 점검
    python -m training.check_data --dataset dataset                          # 점검만
"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}  # 원본 이름 → 코드용 이름


def copy_source(source: Path, dataset: Path):
    for src_name, dst_name in SPLIT_MAP.items():
        src = source / src_name
        if (src / src_name).is_dir():          # train/train 처럼 중첩된 경우
            src = src / src_name
        if not src.is_dir():
            raise SystemExit(f"원본 폴더가 없습니다: {src}")
        shutil.copytree(src, dataset / dst_name, dirs_exist_ok=True)
        print(f"복사: {src} -> {dataset / dst_name}")


def check(dataset: Path):
    counts, hashes, broken, sizes = defaultdict(dict), defaultdict(list), [], defaultdict(int)
    for split in SPLIT_MAP.values():
        root = dataset / split
        if not root.is_dir():
            raise SystemExit(f"폴더가 없습니다: {root}")
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
            counts[split][cls_dir.name] = len(files)
            for f in files:
                hashes[hashlib.sha256(f.read_bytes()).hexdigest()].append(f"{split}/{cls_dir.name}/{f.name}")
                try:
                    with Image.open(f) as im:
                        sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                        im.verify()
                except Exception:
                    broken.append(str(f))

    dups = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in dups if len({x.split("/")[0] for x in v}) > 1]   # Split 사이 누출

    classes = sorted({c for s in counts.values() for c in s})
    print("\n| 클래스 | Train | Val | Test |\n|---|---:|---:|---:|")
    for c in classes:
        print(f"| {c} | " + " | ".join(str(counts[s].get(c, 0)) for s in SPLIT_MAP.values()) + " |")
    print("| 합계 | " + " | ".join(str(sum(counts[s].values())) for s in SPLIT_MAP.values()) + " |")
    print(f"\n동일 파일 중복: {len(dups)}건 / Split 간 누출: {len(cross)}건 / 손상 파일: {len(broken)}건")
    print(f"이미지 크기 분포: {dict(sizes)}")

    out = Path("reports/data"); out.mkdir(parents=True, exist_ok=True)
    (out / "data_check.json").write_text(json.dumps(
        {"counts": counts, "duplicates": dups, "cross_split_duplicates": cross,
         "broken": broken, "image_sizes": sizes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: reports/data/data_check.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="Beans 원본 폴더 (train/validation/test 포함)")
    p.add_argument("--dataset", default="dataset")
    a = p.parse_args()
    if a.source:
        copy_source(Path(a.source).expanduser(), Path(a.dataset))
    check(Path(a.dataset))