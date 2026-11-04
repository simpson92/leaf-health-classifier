"""모의 통합 테스트: 실제 모델·Redis·Celery 대신 가짜를 쓴다. (실환경 검증은 별도)"""
import os
import tempfile
from io import BytesIO
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.mkdtemp()) / "api_test.db")
os.environ["WARMUP"] = "0"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import ai.cache_service as cache_service
import ai.router_async as router_async
import ai.router_sync as router_sync
import ai.services as services
import ai.tasks as tasks
from main import app


def png_bytes():
    buf = BytesIO()
    Image.new("RGB", (40, 40), "green").save(buf, format="PNG")
    return buf.getvalue()


def fake_classifier(image, top_k=5):
    return [{"label": "bean_rust", "score": 0.92}, {"label": "healthy", "score": 0.08}]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "get_classifier", lambda: fake_classifier)
    monkeypatch.setattr(router_sync, "classify_with_cache", services.classify_image_bytes)
    monkeypatch.setattr(tasks, "classify_with_cache", services.classify_image_bytes)
    monkeypatch.setattr(router_async, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(tasks, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(router_async.classify_job, "delay", lambda job_id: None)
    with TestClient(app) as c:
        yield c


def upload(c, url):
    return c.post(url, files={"image": ("a.png", png_bytes(), "image/png")})


def test_sync(client):
    assert client.get("/health").status_code == 200
    for url in ("/vision/sync/", "/vision/async/", "/static/style.css"):
        assert client.get(url).status_code == 200
    r = upload(client, "/api/sync/classify/")
    assert r.status_code == 200 and r.json()["top1"]["label"] == "bean_rust"
    bad = client.post("/api/sync/classify/", files={"image": ("x.txt", b"bad", "text/plain")})
    assert bad.status_code == 400
    big = client.post("/api/sync/classify/", files={"image": ("b.jpg", b"a" * (5 * 1024 * 1024 + 1), "image/jpeg")})
    assert big.status_code == 413


def test_async_flow(client):
    r = upload(client, "/api/async/classify/")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert client.get(f"/api/async/jobs/{job_id}/").json()["status"] == "queued"
    tasks.classify_job(job_id)                      # 워커 대신 직접 실행
    done = client.get(f"/api/async/jobs/{job_id}/").json()
    assert done["status"] == "done" and done["top1"]["label"] == "bean_rust"
    assert done["total_ms"] is not None
    assert client.get("/api/async/jobs/9999999/").status_code == 404


def test_broker_failure(client, monkeypatch):
    def down(job_id):
        raise ConnectionError("offline")
    monkeypatch.setattr(router_async.classify_job, "delay", down)
    assert upload(client, "/api/async/classify/").status_code == 503


def test_worker_failure(client, monkeypatch):
    job_id = upload(client, "/api/async/classify/").json()["job_id"]
    def crash(data):
        raise RuntimeError("fake model crashed")
    monkeypatch.setattr(tasks, "classify_with_cache", crash)
    with pytest.raises(RuntimeError):
        tasks.classify_job(job_id)
    state = client.get(f"/api/async/jobs/{job_id}/").json()
    assert state["status"] == "failed" and "fake model crashed" in state["error_message"]


def test_cache_hit(monkeypatch):
    store, calls = {}, []
    class FakeRedis:
        def get(self, k): return store.get(k)
        def setex(self, k, s, v): store[k] = v
    def fake_infer(data):
        calls.append(1)
        return {"model_id": "m", "top1": {"label": "healthy", "score": 1.0},
                "all": [{"label": "healthy", "score": 1.0}], "inference_ms": 8.0, "cache_hit": False}
    monkeypatch.setattr(cache_service, "get_redis", lambda: FakeRedis())
    monkeypatch.setattr(cache_service, "classify_image_bytes", fake_infer)
    first = cache_service.classify_with_cache(b"same")
    second = cache_service.classify_with_cache(b"same")
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert second["inference_ms"] is None and len(calls) == 1