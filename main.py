from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ai.config import BASE_DIR, WARMUP
from ai.database import init_db
from ai.router_async import router as async_router
from ai.router_sync import router as sync_router
from ai.router_web import router as web_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                    # PostgreSQL 테이블 생성 (Docker를 먼저 실행)
    if WARMUP:                   # [개선] 첫 요청 지연 제거
        from ai.services import warmup
        print(f"[warm-up] 서버 모델 로딩 완료: {warmup()} ms")
    yield


app = FastAPI(title="Beans ResNet18 · FastAPI · Redis · Celery", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(web_router)
app.include_router(sync_router)
app.include_router(async_router)


@app.get("/health")
def health():
    return {"status": "ok"}