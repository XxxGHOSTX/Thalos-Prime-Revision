from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from thalos_prime.config import Settings
from thalos_prime.eval.benchmarks import load_fever_sample, load_synthetic_qa
from thalos_prime.eval.metrics import precision_recall_f1, rouge_l

if TYPE_CHECKING:
    from thalos_prime.core import ThalosCore

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, core: "ThalosCore", settings: Settings):
        self.core = core
        self.settings = settings

    def run_synthetic(self, limit: int = 10) -> dict:
        qa_pairs = load_synthetic_qa()[:limit]
        correct = 0
        rouge_scores = []

        for qa in qa_pairs:
            seed_text = qa["seed_text"]
            import tempfile
            from pathlib import Path
            seed_file = self.settings.data_dir / f"eval_seed_{hash(seed_text)}.txt"
            seed_file.write_text(seed_text)
            try:
                self.core.ingest_file(str(seed_file))
            except Exception as e:
                logger.warning("Seed ingest failed: %s", e)
            finally:
                if seed_file.exists():
                    seed_file.unlink()

            result = self.core.query(qa["question"], top_k=3)
            answer = result.get("answer", "")
            expected = qa["expected_answer_contains"].lower()
            if expected in answer.lower():
                correct += 1
            rl = rouge_l(answer, qa["seed_text"])
            rouge_scores.append(rl)

        accuracy = correct / len(qa_pairs) if qa_pairs else 0.0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        return {
            "suite": "synthetic",
            "total": len(qa_pairs),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_rouge_l": round(avg_rouge, 4),
        }

    def run_fever_sample(self, limit: int = 10) -> dict:
        samples = load_fever_sample(limit=limit)
        predictions = []
        labels = []

        for item in samples:
            claim_text = item["claim"]
            expected_label = item["label"]

            evidence_text = item.get("evidence", "")
            if evidence_text:
                seed_file = self.settings.data_dir / f"fever_seed_{hash(claim_text)}.txt"
                seed_file.write_text(evidence_text)
                try:
                    self.core.ingest_file(str(seed_file))
                except Exception:
                    pass
                finally:
                    if seed_file.exists():
                        seed_file.unlink()

            result = self.core.query(claim_text, top_k=1)
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)

            if confidence > 0.5 or (answer and claim_text.lower()[:20] in answer.lower()):
                pred_label = "SUPPORTS"
            else:
                pred_label = "NOT ENOUGH INFO"

            predictions.append(1 if pred_label == "SUPPORTS" else 0)
            labels.append(1 if expected_label == "SUPPORTS" else 0)

        metrics = precision_recall_f1(predictions, labels)
        return {
            "suite": "fever_sample",
            "total": len(samples),
            **metrics,
        }

    def run_all(self) -> dict:
        results = {}
        try:
            results["synthetic"] = self.run_synthetic()
        except Exception as e:
            logger.error("Synthetic eval failed: %s", e)
            results["synthetic"] = {"error": str(e)}
        try:
            results["fever_sample"] = self.run_fever_sample()
        except Exception as e:
            logger.error("FEVER eval failed: %s", e)
            results["fever_sample"] = {"error": str(e)}
        return results

    def print_results(self, results: dict) -> None:
        import json
        print(json.dumps(results, indent=2))
