"""Registro (log) de ejecución en JSON Lines con marca de tiempo.

Cada línea es un evento: run_start, percept, decision, action, ground_truth,
feedback, retrain, run_end. Es la base de las pruebas de la Fase 3.

  ts        -> hora real (UTC, ISO 8601)
  sim_time  -> reloj simulado de la plataforma (segundos)
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class NullLogger:
    def log(self, event: str, sim_time: Optional[float] = None, **payload: Any) -> None:
        pass

    def close(self) -> None:
        pass


class RunLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)  # una línea = un flush

    def log(self, event: str, sim_time: Optional[float] = None, **payload: Any) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            "sim_time": None if sim_time is None else round(float(sim_time), 3),
        }
        if event in {"percept", "decision", "action", "feedback", "retrain", "ground_truth"}:
            rec.update({
                "comment_id": None,
                "text_hash": None,
                "coordination": {},
                "p_toxic": None,
                "subcategory_probs": {},
                "p_effective": None,
                "expected_utilities": {},
                "action": None,
                "api_ok": None,
                "latency_ms": None,
                "ground_truth": None,
                "error": None,
                "model_version": None,
            })
        rec.update(payload)
        text = rec.pop("text", None)
        if text is not None and "text_hash" not in payload:
            rec["text_hash"] = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
