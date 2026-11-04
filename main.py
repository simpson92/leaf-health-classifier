from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ai.config import BASE_DIR
from ai.router_sync import router as sync_router
from ai.router_web import router as web_router

app = FastAPI(title="Beans ResNet18 Image Classification")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(web_router)
app.include_router(sync_router)


@app.get("/health")
def health():
    return {"status": "ok"}