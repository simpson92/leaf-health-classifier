import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# 모델: 학습 전에는 원본, 학습 후에는 .env에서 artifacts/... 로 변경
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/resnet-18")

# 6단계(비동기)에서 사용할 설정 — 지금은 쓰이지 않음
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://vision_user:vision_pass@127.0.0.1:5432/vision_db")
REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://127.0.0.1:6379/1")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/2")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/3")
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "600"))

# 업로드 제한
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MEDIA_DIR = BASE_DIR / "media" / "jobs"
WARMUP = os.getenv("WARMUP", "1") == "1"   # [개선] 시작 시 모델 미리 로딩 (0이면 끔: 비교 측정용)