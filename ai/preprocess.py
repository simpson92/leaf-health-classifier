"""학습과 서빙이 같은 전처리를 쓰도록 한 곳에 모은다. (5단계에서 사용)"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

CONFIG_NAME = "preprocess.json"


def apply_preprocess(image: Image.Image, method: str = "none") -> Image.Image:
    if method in (None, "none"):
        return image
    if method == "clahe":
        # 밝기(L)만 보정하고 색(a, b)은 그대로 둔다: 병반 색이 핵심 단서라서
        import cv2
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))
    raise ValueError(f"알 수 없는 전처리: {method}")


def save_preprocess_config(model_dir, method: str):
    Path(model_dir, CONFIG_NAME).write_text(json.dumps({"method": method}), encoding="utf-8")


def load_preprocess_method(model_dir) -> str:
    path = Path(model_dir) / CONFIG_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get("method", "none")
    return "none"   # 원본 모델·baseline 모델은 전처리 없음