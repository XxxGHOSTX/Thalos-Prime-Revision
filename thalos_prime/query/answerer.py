from __future__ import annotations

import logging
from typing import Optional

from thalos_prime.config import Settings

logger = logging.getLogger(__name__)


class Answerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def synthesize(self, query: str, results: list[dict]) -> dict:
        if not results:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "citations": [],
                "confidence": 0.0,
                "method": "template",
            }

        if self.settings.use_cloud_llm and self.settings.openai_api_key:
            try:
                return self._llm_synthesize(query, results)
            except Exception as e:
                logger.warning("LLM synthesis failed: %s. Falling back to template.", e)

        return self._template_synthesize(query, results)

    def _template_synthesize(self, query: str, results: list[dict]) -> dict:
        top = results[0]
        top_claim = top["text"]
        evidence_count = top.get("evidence_count", 0)
        status = top.get("status", "draft")
        confidence = min(1.0, top.get("score", 0.0) * 10.0)

        if evidence_count > 0:
            answer = (
                f"Based on the available evidence: {top_claim} "
                f"This is supported by {evidence_count} source(s). "
                f"[Verification status: {status}]"
            )
        else:
            answer = f"Based on extracted knowledge: {top_claim} [Status: {status}]"

        citations = [
            {
                "coordinate": r.get("coordinate"),
                "text": r["text"][:200],
                "status": r.get("status"),
                "score": r.get("score"),
            }
            for r in results[:5]
        ]

        return {
            "answer": answer,
            "citations": citations,
            "confidence": round(confidence, 4),
            "method": "template",
        }

    def _llm_synthesize(self, query: str, results: list[dict]) -> dict:
        import httpx

        context_parts = []
        for i, r in enumerate(results[:5], 1):
            context_parts.append(f"[{i}] {r['text']} (status: {r.get('status', 'unknown')})")
        context = "\n".join(context_parts)

        system_prompt = (
            "You are a factual knowledge assistant. "
            "Answer the user's question using only the provided evidence. "
            "Cite the numbered sources. Be concise and accurate."
        )
        user_prompt = f"Question: {query}\n\nEvidence:\n{context}\n\nAnswer:"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.settings.openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.openai_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            data = response.json()

        answer = data["choices"][0]["message"]["content"].strip()
        citations = [
            {"coordinate": r.get("coordinate"), "text": r["text"][:200], "status": r.get("status")}
            for r in results[:5]
        ]

        return {
            "answer": answer,
            "citations": citations,
            "confidence": 0.9,
            "method": "llm",
        }
