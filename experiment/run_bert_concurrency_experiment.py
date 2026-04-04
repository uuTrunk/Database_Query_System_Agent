#!/usr/bin/env python3
"""A/B experiment runner for single-thread vs BERT-predicted concurrency.

This script assumes the following services are already running:
- Agent service: default http://127.0.0.1:8000
- Training service: default http://127.0.0.1:8001

Outputs per run:
- One CSV file under ./experiment/results
- One PNG figure under ./experiment/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class HttpResult:
    ok: bool
    status: int
    payload: Dict[str, Any]
    latency_ms: float
    error_message: str


@dataclass
class AgentCallResult:
    success: bool
    http_status: int
    api_code: Optional[int]
    latency_ms: float
    retries_used: Any
    success_ratio: Optional[float]
    error_message: str


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> HttpResult:
    start = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            latency_ms = (time.perf_counter() - start) * 1000.0
            return HttpResult(True, int(resp.status), parsed if isinstance(parsed, dict) else {}, latency_ms, "")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        parsed: Dict[str, Any] = {}
        if body.strip():
            try:
                candidate = json.loads(body)
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                pass
        latency_ms = (time.perf_counter() - start) * 1000.0
        return HttpResult(False, int(exc.code), parsed, latency_ms, f"HTTPError: {exc}")
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return HttpResult(False, 0, {}, latency_ms, f"RequestError: {exc}")


def _normalize_threads(value: Any, fallback: int = 1, low: int = 1, high: int = 5) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(low, min(high, parsed))


def _call_predict(training_base: str, predict_endpoint: str, question: str, timeout: float) -> Tuple[int, float, HttpResult]:
    url = training_base.rstrip("/") + predict_endpoint
    res = _post_json(url, {"text": question}, timeout=timeout)
    score = 0.0
    threads = 1
    if isinstance(res.payload, dict):
        score = float(res.payload.get("score", 0.0) or 0.0)
        threads = _normalize_threads(res.payload.get("threads", 1), fallback=1)
    return threads, score, res


def _call_agent_pd(
    agent_base: str,
    agent_endpoint: str,
    question: str,
    concurrent_workers: int,
    retries: int,
    timeout: float,
) -> AgentCallResult:
    url = agent_base.rstrip("/") + agent_endpoint
    payload = {
        "question": question,
        "concurrent": int(concurrent_workers),
        "retries": int(retries),
    }
    res = _post_json(url, payload, timeout=timeout)
    api_code = res.payload.get("code") if isinstance(res.payload, dict) else None
    retries_used = res.payload.get("retries_used") if isinstance(res.payload, dict) else None
    success_ratio = None
    if isinstance(res.payload, dict) and res.payload.get("success") is not None:
        try:
            success_ratio = float(res.payload.get("success"))
        except (TypeError, ValueError):
            success_ratio = None

    success = res.ok and res.status == 200 and api_code == 200
    err = ""
    if not success:
        err = res.error_message or str(res.payload.get("message", "agent failed"))

    return AgentCallResult(
        success=success,
        http_status=res.status,
        api_code=api_code if isinstance(api_code, int) else None,
        latency_ms=res.latency_ms,
        retries_used=retries_used,
        success_ratio=success_ratio,
        error_message=err,
    )


def _load_questions(csv_path: Path, max_questions: int, sample_mode: str, seed: int) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Question file not found: {csv_path}")

    rows: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for record in reader:
            for cell in record:
                q = cell.strip()
                if q:
                    rows.append(q)

    deduped: List[str] = []
    seen = set()
    for q in rows:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    if not deduped:
        raise ValueError(f"No questions loaded from: {csv_path}")

    if sample_mode == "random":
        rnd = random.Random(seed)
        rnd.shuffle(deduped)

    return deduped[: max(1, max_questions)]


def _default_questions_file() -> Path:
    this_file = Path(__file__).resolve()
    code_root = this_file.parents[2]
    return code_root / "Database_Query_System_Training" / "gened_questions" / "training_questions_for_graph.csv"


def _preflight(
    agent_base: str,
    prompt_endpoint: str,
    training_base: str,
    predict_endpoint: str,
    timeout: float,
) -> None:
    pred_res = _post_json(training_base.rstrip("/") + predict_endpoint, {"text": "hello"}, timeout=timeout)
    if not pred_res.ok or pred_res.status != 200:
        raise RuntimeError(
            "Training service preflight failed. "
            f"url={training_base.rstrip('/') + predict_endpoint}, status={pred_res.status}, error={pred_res.error_message}"
        )

    prompt_res = _post_json(agent_base.rstrip("/") + prompt_endpoint, {"question": "health check"}, timeout=timeout)
    if not prompt_res.ok or prompt_res.status != 200:
        raise RuntimeError(
            "Agent service preflight failed. "
            f"url={agent_base.rstrip('/') + prompt_endpoint}, status={prompt_res.status}, error={prompt_res.error_message}"
        )


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_figure(out_path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    baseline_success = [int(bool(r["baseline_success"])) for r in rows]
    predict_success = [int(bool(r["predict_success"])) for r in rows]
    baseline_latency = [float(r["baseline_latency_ms"]) for r in rows]
    predict_latency = [float(r["predict_latency_ms"]) for r in rows]
    predict_threads = [int(r["predicted_threads"]) for r in rows]

    success_rate_baseline = 100.0 * _safe_mean(baseline_success)
    success_rate_predict = 100.0 * _safe_mean(predict_success)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(["Single(1)", "BERT Predict"], [success_rate_baseline, success_rate_predict], color=["#6C8EBF", "#82B366"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate Comparison")

    axes[1].bar(
        ["Single(1)", "BERT Predict"],
        [_safe_mean(baseline_latency), _safe_mean(predict_latency)],
        color=["#6C8EBF", "#82B366"],
    )
    axes[1].set_ylabel("Avg Latency (ms)")
    axes[1].set_title("Latency Comparison")

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    axes[2].hist(predict_threads, bins=bins, rwidth=0.85, color="#F6B26B", edgecolor="black")
    axes[2].set_xticks([1, 2, 3, 4, 5])
    axes[2].set_xlabel("Predicted Concurrency")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Predicted Threads Distribution")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "timestamp",
        "question_index",
        "question",
        "predicted_score",
        "predicted_threads",
        "baseline_concurrent",
        "baseline_success",
        "baseline_http_status",
        "baseline_api_code",
        "baseline_latency_ms",
        "baseline_retries_used",
        "baseline_agent_success_ratio",
        "baseline_error",
        "predict_concurrent",
        "predict_success",
        "predict_http_status",
        "predict_api_code",
        "predict_latency_ms",
        "predict_retries_used",
        "predict_agent_success_ratio",
        "predict_error",
        "success_delta",
        "latency_delta_ms",
        "predicted_threads_is_singleton_run",
        "comparison_round_validity",
        "comparison_round_invalid_reason",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_experiment(args: argparse.Namespace) -> Tuple[Path, Path, Dict[str, Any]]:
    _preflight(
        agent_base=args.agent_base,
        prompt_endpoint=args.agent_prompt_endpoint,
        training_base=args.training_base,
        predict_endpoint=args.training_predict_endpoint,
        timeout=args.timeout,
    )

    questions_file = Path(args.questions_file).resolve() if args.questions_file else _default_questions_file()
    questions = _load_questions(
        csv_path=questions_file,
        max_questions=args.max_questions,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"exp_{run_stamp}_{uuid.uuid4().hex[:8]}"

    all_rows: List[Dict[str, Any]] = []

    for i, question in enumerate(questions, start=1):
        predicted_threads, predicted_score, predict_api_meta = _call_predict(
            training_base=args.training_base,
            predict_endpoint=args.training_predict_endpoint,
            question=question,
            timeout=args.timeout,
        )

        baseline = _call_agent_pd(
            agent_base=args.agent_base,
            agent_endpoint=args.agent_pd_endpoint,
            question=question,
            concurrent_workers=args.baseline_concurrent,
            retries=args.retries,
            timeout=args.timeout,
        )

        predicted = _call_agent_pd(
            agent_base=args.agent_base,
            agent_endpoint=args.agent_pd_endpoint,
            question=question,
            concurrent_workers=predicted_threads,
            retries=args.retries,
            timeout=args.timeout,
        )

        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question_index": i,
            "question": question,
            "predicted_score": round(predicted_score, 6),
            "predicted_threads": predicted_threads,
            "baseline_concurrent": args.baseline_concurrent,
            "baseline_success": int(baseline.success),
            "baseline_http_status": baseline.http_status,
            "baseline_api_code": baseline.api_code,
            "baseline_latency_ms": round(baseline.latency_ms, 3),
            "baseline_retries_used": baseline.retries_used,
            "baseline_agent_success_ratio": baseline.success_ratio,
            "baseline_error": baseline.error_message,
            "predict_concurrent": predicted_threads,
            "predict_success": int(predicted.success),
            "predict_http_status": predicted.http_status,
            "predict_api_code": predicted.api_code,
            "predict_latency_ms": round(predicted.latency_ms, 3),
            "predict_retries_used": predicted.retries_used,
            "predict_agent_success_ratio": predicted.success_ratio,
            "predict_error": predicted.error_message,
            "success_delta": int(predicted.success) - int(baseline.success),
            "latency_delta_ms": round(predicted.latency_ms - baseline.latency_ms, 3),
        }

        if not (predict_api_meta.ok and predict_api_meta.status == 200):
            row["predict_error"] = (
                (row.get("predict_error") or "")
                + f" | training_predict_meta status={predict_api_meta.status} err={predict_api_meta.error_message}"
            ).strip()

        all_rows.append(row)

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    predicted_threads_unique = sorted({int(r["predicted_threads"]) for r in all_rows})
    predicted_threads_is_singleton = len(predicted_threads_unique) == 1
    invalid_comparison_round = predicted_threads_is_singleton and predicted_threads_unique[0] == int(args.baseline_concurrent)
    comparison_round_validity = "无效对比轮" if invalid_comparison_round else "有效对比轮"
    invalid_reason = "预测并发整轮单一且与基线并发完全相同" if invalid_comparison_round else ""

    for row in all_rows:
        row["predicted_threads_is_singleton_run"] = int(predicted_threads_is_singleton)
        row["comparison_round_validity"] = comparison_round_validity
        row["comparison_round_invalid_reason"] = invalid_reason

    results_dir = Path(args.results_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    csv_path = results_dir / f"{run_id}.csv"
    fig_path = figures_dir / f"{run_id}.png"

    _write_csv(csv_path, all_rows)

    title = f"BERT Concurrency Experiment | n={len(all_rows)} | retries={args.retries}"
    _render_figure(fig_path, title=title, rows=all_rows)

    baseline_success_rate = 100.0 * _safe_mean([int(r["baseline_success"]) for r in all_rows])
    predict_success_rate = 100.0 * _safe_mean([int(r["predict_success"]) for r in all_rows])
    improvement_abs = predict_success_rate - baseline_success_rate
    improvement_rel = (improvement_abs / baseline_success_rate * 100.0) if baseline_success_rate > 0 else 0.0

    summary = {
        "run_id": run_id,
        "questions": len(all_rows),
        "baseline_success_rate_pct": round(baseline_success_rate, 3),
        "predict_success_rate_pct": round(predict_success_rate, 3),
        "improvement_abs_pct": round(improvement_abs, 3),
        "improvement_rel_pct": round(improvement_rel, 3),
        "baseline_avg_latency_ms": round(_safe_mean([float(r["baseline_latency_ms"]) for r in all_rows]), 3),
        "predict_avg_latency_ms": round(_safe_mean([float(r["predict_latency_ms"]) for r in all_rows]), 3),
        "predicted_threads_unique": predicted_threads_unique,
        "predicted_threads_is_singleton": predicted_threads_is_singleton,
        "comparison_round_validity": comparison_round_validity,
        "comparison_round_invalid_reason": invalid_reason,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }

    return csv_path, fig_path, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A/B experiment: single-thread vs BERT-predicted concurrency")

    parser.add_argument("--agent-base", default="http://127.0.0.1:8000", help="Agent service base URL")
    parser.add_argument("--agent-pd-endpoint", default="/api/ask/pd", help="Agent endpoint for PD requests")
    parser.add_argument("--agent-prompt-endpoint", default="/api/prompt/pd", help="Agent preflight endpoint")

    parser.add_argument("--training-base", default="http://127.0.0.1:8001", help="Training service base URL")
    parser.add_argument("--training-predict-endpoint", default="/api/predict", help="Training predict endpoint")

    parser.add_argument("--questions-file", default="", help="CSV question file path (optional)")
    parser.add_argument("--max-questions", type=int, default=50, help="Maximum number of questions used in this run")
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head", help="Question sampling mode")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed when sample-mode=random")

    parser.add_argument("--baseline-concurrent", type=int, default=1, help="Baseline fixed concurrency")
    parser.add_argument("--retries", type=int, default=1, help="Retries passed to Agent")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per request timeout in seconds")
    parser.add_argument("--sleep-between", type=float, default=0.0, help="Sleep seconds between question cases")

    parser.add_argument("--results-dir", default="./experiment/results", help="Directory to save CSV files")
    parser.add_argument("--figures-dir", default="./experiment/figures", help="Directory to save figure files")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path, fig_path, summary = run_experiment(args)

    print("Experiment completed.")
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
