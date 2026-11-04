"""동기 vs 비동기 성능 측정. Docker(Redis·PostgreSQL)·워커·서버가 모두 켜져 있어야 한다.
예) python -m scripts.benchmark                                 # 단일 요청 + 동시 요청
    python -m scripts.benchmark --mode cold --label warmup_on   # 서버 재시작 직후 첫 요청 1회
"""
import argparse
import csv
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from redis import Redis

from ai.config import REDIS_CACHE_URL

OUT = Path("reports/data")
FIELDS = ["experiment", "path", "cache", "run", "image", "http_ms", "final_ms",
          "inference_ms", "queue_ms", "server_total_ms", "job_id"]


def pick_images(data="dataset/test", seed=42):
    paths = sorted(Path(data).glob("*/*.jpg"))
    random.Random(seed).shuffle(paths)         # 클래스가 섞이도록 고정 시드로 섞기
    return paths


def flush_cache():
    Redis.from_url(REDIS_CACHE_URL).flushdb()   # 캐시(DB/1)만 비운다. 큐·PostgreSQL은 그대로


def _upload(client, url, path):
    data = path.read_bytes()
    t0 = time.perf_counter()
    r = client.post(url, files={"image": (path.name, data, "image/jpeg")})
    return r, t0, (time.perf_counter() - t0) * 1000


def sync_once(client, path):
    r, _, ms = _upload(client, "/api/sync/classify/", path)
    r.raise_for_status()
    body = r.json()
    return {"path": "sync", "cache": "HIT" if body["cache_hit"] else "MISS", "image": path.name,
            "http_ms": round(ms, 1), "final_ms": round(ms, 1), "inference_ms": body["inference_ms"]}


def async_once(client, path, poll_s=0.02):
    r, t0, ms = _upload(client, "/api/async/classify/", path)
    r.raise_for_status()
    job = r.json()
    while True:
        time.sleep(poll_s)
        state = client.get(job["check_url"]).json()
        if state["status"] in ("done", "failed"):
            break
    if state["status"] == "failed":
        raise RuntimeError(f"job {job['job_id']} 실패: {state['error_message']}")
    final = (time.perf_counter() - t0) * 1000
    return {"path": "async", "cache": "HIT" if state["cache_hit"] else "MISS", "image": path.name,
            "http_ms": round(ms, 1), "final_ms": round(final, 1),
            "inference_ms": state["inference_ms"], "queue_ms": state.get("queue_ms"),
            "server_total_ms": state.get("total_ms"), "job_id": job["job_id"]}


def concurrent(base, fn, images, **kw):
    """여러 사용자가 '동시에' 요청하는 상황: 모든 스레드가 준비되면 한꺼번에 출발"""
    barrier = threading.Barrier(len(images))

    def task(path):
        with httpx.Client(base_url=base, timeout=300) as c:
            barrier.wait()
            return fn(c, path, **kw)

    started = time.perf_counter()
    with ThreadPoolExecutor(len(images)) as pool:
        rows = list(pool.map(task, images))
    return rows, round((time.perf_counter() - started) * 1000, 1)


def _fmt(v):
    return "-" if v is None else v


def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def summarize(rows, walls):
    groups = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["path"], r["cache"]), []).append(r)
    lines = ["| 실험 | 경로 | 캐시 | n | HTTP 응답 중앙값 | 최종 결과 중앙값 | 최종 결과 최대 | 추론 중앙값 | 서버 큐대기 중앙값 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for (exp, path, cache), g in groups.items():
        lines.append(" | ".join([
            f"| {exp}", path, cache, str(len(g)),
            str(_fmt(_med([r["http_ms"] for r in g]))),
            str(_fmt(_med([r["final_ms"] for r in g]))),
            str(max(r["final_ms"] for r in g)),
            str(_fmt(_med([r.get("inference_ms") for r in g]))),
            str(_fmt(_med([r.get("queue_ms") for r in g]))) + " |"]))
    lines.append("")
    lines.append("(단위: ms) " + " / ".join(f"{k} 전체 처리 시간 {v}ms" for k, v in walls.items()))
    return "\n".join(lines)


def run(args):
    images = pick_images()
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=300) as c:
        c.get("/health").raise_for_status()

        if args.mode == "cold":                      # warm-up 전후 비교: 첫 요청 1회
            flush_cache()
            row = sync_once(c, images[-3])
            print(f"[{args.label}] 첫 동기 요청 HTTP {row['http_ms']}ms / 추론 {row['inference_ms']}ms")
            path = OUT / "cold_start.csv"
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["measured_at", "label", "image", "http_ms", "inference_ms"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), args.label,
                            row["image"], row["http_ms"], row["inference_ms"]])
            return

        flush_cache()
        n, k = args.repeats, args.concurrency
        sync_set, async_set = images[:n], images[n:2 * n]
        conc_sync, conc_async = images[2 * n:2 * n + k], images[2 * n + k:2 * n + 2 * k]
        sync_once(c, images[-1]); async_once(c, images[-2], args.poll)   # 예열(측정 제외)

        rows = []
        for i, p in enumerate(sync_set, 1):          # 같은 이미지 2번: MISS → HIT
            for _ in range(2):
                rows.append({"experiment": "single", "run": i, **sync_once(c, p)})
        for i, p in enumerate(async_set, 1):
            for _ in range(2):
                rows.append({"experiment": "single", "run": i, **async_once(c, p, args.poll)})
        print(f"단일 요청 {n}회 × 2경로 × MISS/HIT 완료")

    s_rows, s_wall = concurrent(args.base_url, sync_once, conc_sync)
    a_rows, a_wall = concurrent(args.base_url, async_once, conc_async, poll_s=max(args.poll, 0.05))
    for i, r in enumerate(s_rows, 1):
        rows.append({"experiment": f"concurrent{k}", "run": i, **r})
    for i, r in enumerate(a_rows, 1):
        rows.append({"experiment": f"concurrent{k}", "run": i, **r})
    print(f"동시 {k}건 동기·비동기 완료")

    with open(OUT / "timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader(); w.writerows(rows)
    summary = summarize(rows, {f"동시{k} 동기": s_wall, f"동시{k} 비동기": a_wall})
    (OUT / "timings_summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\n저장: {OUT / 'timings.csv'}, {OUT / 'timings_summary.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--mode", choices=["all", "cold"], default="all")
    p.add_argument("--repeats", type=int, default=10, help="단일 요청 반복 횟수")
    p.add_argument("--concurrency", type=int, default=20, help="동시 요청 수")
    p.add_argument("--poll", type=float, default=0.02, help="비동기 상태 조회 간격(초)")
    p.add_argument("--label", default="cold", help="cold 모드 기록 이름")
    run(p.parse_args())