from functools import lru_cache

import torch

from .config import BASE_DIR, MODEL_PATH


def resolve_model_path() -> str:
    """[개선] 로컬 폴더면 프로젝트 기준 절대경로, 아니면 Hugging Face 모델 ID 그대로."""
    local = BASE_DIR / MODEL_PATH
    return str(local) if local.exists() else MODEL_PATH


@lru_cache(maxsize=1)
def get_classifier():
    """서버/워커 프로세스별 최초 요청 때 모델을 로드하고 재사용한다."""
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1   # [개선] GPU 있으면 자동 사용
    return pipeline("image-classification", model=resolve_model_path(), device=device)