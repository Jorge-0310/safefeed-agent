"""Ejecuta el agente contra la plataforma simulada y deja log + resumen de métricas.

    python -m safefeed.runner --steps 3000 --seed 7
    python -m safefeed.runner --steps 3000 --seed 7 --no-learning       # ablación
    python -m safefeed.runner --steps 3000 --seed 7 --no-coordination   # ablación
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .agent import SafeFeedAgent
from .config import JIGSAW_TRAIN, LOG_DIR, MODEL_PATH, REPORT_DIR, AgentConfig, UtilityConfig
from .logger import RunLogger
from .metrics import compute_run_metrics
from .model import ToxicityModel
from .schemas import RawComment
from .simulator import ApiResponse, ModeratorSimulator, PlatformSimulator, PublicationAPI
from .train import ensure_model
from .utility import utility_action_regions


def run_simulation(
    n_steps: int = 3000,
    seed: int = 7,
    model: Optional[ToxicityModel] = None,
    agent_config: AgentConfig = AgentConfig(),
    utility_config: UtilityConfig = UtilityConfig(),
    log_dir: Optional[Path] = LOG_DIR,
    run_tag: str = "",
    api_failure_rate: float = 0.01,
    mod_accuracy: float = 0.93,
) -> dict:
    """Corre una simulación completa. Devuelve {'summary', 'records', 'log_path'}."""
    model = copy.deepcopy(model) if model is not None else ensure_model()  # no mutar el modelo base
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"_{run_tag}" if run_tag else "")
    log_path = Path(log_dir) / f"run_{run_id}.jsonl" if log_dir else None
    logger = RunLogger(log_path, run_id) if log_path else None

    sim = PlatformSimulator(n_steps=n_steps, seed=seed)
    api = PublicationAPI(seed=seed + 1, failure_rate=api_failure_rate)
    mods = ModeratorSimulator(seed=seed + 2, accuracy=mod_accuracy)
    agent = SafeFeedAgent(model, agent_config, utility_config, logger)

    if logger:
        logger.log(
            "run_start", 0.0, seed=seed, n_steps=n_steps, model_version=model.version,
            n_training_examples=model.n_training_examples,
            agent_config=agent_config.__dict__, utility_config=utility_config.__dict__,
            note="Entorno SIMULADO; las líneas 'ground_truth' vienen del simulador, el agente no las ve.",
        )

    records = []
    for raw, truth in sim:
        for fb in mods.due(raw.timestamp):          # llegan veredictos de moderadores pendientes
            agent.receive_feedback(fb)
        api_failures_before = api.failures
        _, decision, action_result = agent.step(raw, api)
        injected_api_failures = api.failures - api_failures_before
        if logger:
            logger.log("ground_truth", raw.timestamp, comment_id=raw.comment_id, labels=truth.labels,
                       is_toxic=truth.is_toxic, kind=truth.kind, phase=truth.phase,
                       campaign_id=truth.campaign_id, source="simulator",
                       ground_truth={"labels": truth.labels, "is_toxic": truth.is_toxic,
                                     "kind": truth.kind, "phase": truth.phase,
                                     "campaign_id": truth.campaign_id, "source": "simulator"})
        mods.consider(raw.comment_id, action_result.action, truth, raw.timestamp)
        records.append({
            "comment_id": raw.comment_id, "action": action_result.action.value,
            "p_toxic": decision.p_toxic, "p_effective": decision.p_effective,
            "subcategory_probs": decision.subcategory_probs,
            "coordination": decision.coordination_score, "surge": decision.surge,
            "latency_ms": action_result.latency_ms, "model_version": decision.model_version,
            "is_toxic": truth.is_toxic, "kind": truth.kind, "phase": truth.phase,
            "campaign_id": truth.campaign_id,
            "classifier_threshold": getattr(model, "decision_threshold", 0.5),
            "api_ok": action_result.ok,
            "api_errors": injected_api_failures,
            "api_failures_injected": injected_api_failures,
            "failure_handled": bool(
                injected_api_failures and
                (action_result.action.value != "publish" or action_result.ok)
            ),
            "error": action_result.error,
        })
    for fb in mods.due(float("inf")):               # veredictos que quedaron en cola al terminar
        agent.receive_feedback(fb)
    if agent._feedback_buffer:
        agent.retrain(raw.timestamp if records else 0.0)

    summary = compute_run_metrics(records)
    failure_handling = evaluate_failure_handling(model)
    summary.update({
        "seed": seed, "n_steps": n_steps, "run_id": run_id,
        "learning_enabled": agent_config.learning_enabled,
        "coordination_enabled": agent_config.use_coordination,
        "utility_config": utility_config.__dict__,
        "utility_action_regions": utility_action_regions(utility_config),
        "n_feedback_received": agent.n_feedback, "n_retrains": agent.n_retrains,
        "model_version_final": model.version,
        "api_calls": api.calls, "api_failures": api.failures,
        "n_api_errors": api.failures,
        "log_path": str(log_path) if log_path else None,
        "environment": "simulado",
        "failure_handling": failure_handling,
    })
    if logger:
        logger.log("run_end", None, summary=summary)
        logger.close()
    return {"summary": summary, "records": records, "log_path": log_path}


def evaluate_failure_handling(model: ToxicityModel) -> dict:
    """Inyecta las clases de fallo críticas y mide si se evita publicar de forma insegura."""
    class AlwaysOk:
        def execute(self, action, comment_id, sim_time):
            return ApiResponse(True, "ok")

    class InvalidResponse:
        def execute(self, action, comment_id, sim_time):
            return {"ok": True, "status": "invalid"}

    class Unavailable:
        def execute(self, action, comment_id, sim_time):
            raise ConnectionError("simulated API unavailable")

    def raw(comment_id, text="valid comment", user="user", thread="thread", timestamp=1.0):
        return RawComment(comment_id, user, timestamp, thread, text)

    cases = [
        ("comment_absent", None, AlwaysOk()),
        ("text_empty", raw("empty", ""), AlwaysOk()),
        ("text_whitespace", raw("spaces", "  \t"), AlwaysOk()),
        ("text_absent", raw("missing-text", None), AlwaysOk()),
        ("user_absent", raw("missing-user", user=""), AlwaysOk()),
        ("thread_absent", raw("missing-thread", thread=""), AlwaysOk()),
        ("timestamp_invalid", raw("bad-time", timestamp=float("nan")), AlwaysOk()),
        ("api_unavailable", raw("offline"), Unavailable()),
        ("api_response_invalid", raw("invalid-response"), InvalidResponse()),
        ("coordination_signal_missing", raw("missing-sensor"), AlwaysOk()),
        ("comment_malformed", raw("malformed", text=17), AlwaysOk()),
    ]
    results = []
    for name, comment, api in cases:
        agent = SafeFeedAgent(
            model,
            AgentConfig(learning_enabled=False),
            logger=None,
        )
        if name == "coordination_signal_missing":
            agent.memory.features = lambda *args: {}
        _, decision, action = agent.step(comment, api)
        safe = action.action.value == "hide_for_review"
        handled = safe and (
            action.ok if name not in {"api_unavailable", "api_response_invalid"}
            else bool(action.error)
        )
        results.append({
            "failure": name,
            "handled_safely": bool(handled),
            "action": action.action.value,
            "api_ok": action.ok,
            "attempts": action.attempts,
            "error": action.error or decision.error,
        })
    handled = sum(result["handled_safely"] for result in results)
    return {
        "failures_injected": len(results),
        "failures_handled_correctly": handled,
        "safe_failure_handling_rate": round(handled / len(results), 4),
        "cases": results,
    }


def _print_summary(s: dict) -> None:
    print("\n=== Resumen de la ejecución (entorno simulado) ===")
    keys = ["n_comments", "toxic_rate", "accuracy", "baseline_accuracy_always_publish",
            "precision", "recall", "f1", "auc_roc", "auc_pr", "false_block_rate",
            "false_positive_rate", "false_negative_rate",
            "harmful_published_rate", "harmful_blocked_rate", "harmful_hidden_rate",
            "share_blocked", "share_hidden", "share_published", "human_review_workload",
            "latency_ms_mean",
            "latency_ms_median", "latency_ms_p95", "latency_ms_max",
            "n_feedback_received", "n_retrains", "model_version_final"]
    for k in keys:
        if k in s:
            print(f"  {k:<36} {s[k]}")
    for ph, v in s.get("by_phase", {}).items():
        print(f"  [{ph}] n={v['n']} acc={v['accuracy']} recall={v['recall']} f1={v['f1']}")
    if "attacks" in s:
        a = s["attacks"]
        print(f"  ataques: {a['n_attack_comments']} comentarios, contenidos={a['contained_rate']}, "
              f"filtrados={a['filtered']}, publicados={a['leaked']}, "
              f"publicados antes de contener={a['published_before_first_containment']}")
        print(f"  campañas: media={a['by_campaign']['mean']} "
              f"desv={a['by_campaign']['std']} min={a['by_campaign']['min']} "
              f"max={a['by_campaign']['max']}")
        for cid, v in a["per_campaign"].items():
            print(f"    campaña {cid}: contenidos={v['contained_rate']}, "
                  f"publicados antes de contener={v['published_before_first_containment']}")
    if "classifier" in s:
        print(f"  clasificador: F1={s['classifier']['f1']} AUC-ROC={s['classifier']['auc_roc']} "
              f"AUC-PR={s['classifier']['auc_pr']}")
    print(f"  log: {s.get('log_path')}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--retrain-every", type=int, default=AgentConfig.retrain_every)
    ap.add_argument("--no-learning", action="store_true", help="desactiva el aprendizaje con retroalimentación")
    ap.add_argument("--no-coordination", action="store_true", help="desactiva la detección de coordinación")
    ap.add_argument("--tag", default="", help="sufijo para el nombre del log")
    args = ap.parse_args()

    cfg = AgentConfig(retrain_every=args.retrain_every,
                      learning_enabled=not args.no_learning,
                      use_coordination=not args.no_coordination)
    model = ensure_model(args.model_path)
    result = run_simulation(args.steps, args.seed, model, cfg, log_dir=args.log_dir, run_tag=args.tag)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "last_run_summary.json"
    out.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(result["summary"])


if __name__ == "__main__":
    main()
