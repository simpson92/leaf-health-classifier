from celery import Celery
from celery.signals import worker_init

from .config import BROKER_POOL_LIMIT, CELERY_BROKER_URL, CELERY_RESULT_BACKEND, WARMUP

celery_app = Celery(
    "vision_ai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["ai.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    broker_pool_limit=BROKER_POOL_LIMIT,
)


@worker_init.connect
def warmup_model(**_):
    """[개선] 워커 시작 시 모델 미리 로딩. --pool=solo 기준(이 프로세스가 작업도 실행)."""
    if WARMUP:
        from .services import warmup
        print(f"[warm-up] 워커 모델 로딩 완료: {warmup()} ms")