from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import ai.services as services
from main import app


def png_bytes():
    buf = BytesIO()
    Image.new("RGB", (32, 32), "green").save(buf, format="PNG")
    return buf.getvalue()


def test_sync(monkeypatch):
    monkeypatch.setattr(services, "get_classifier",
                        lambda: (lambda image, top_k=5: [{"label": "healthy", "score": 0.9}]))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/vision/sync/").status_code == 200
        assert c.get("/static/style.css").status_code == 200
        r = c.post("/api/sync/classify/", files={"image": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 200 and r.json()["top1"]["label"] == "healthy"
        bad = c.post("/api/sync/classify/", files={"image": ("bad.txt", b"bad", "text/plain")})
        assert bad.status_code == 400