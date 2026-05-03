"""OpenAI-based content classifier for municipality documents.

Uses structured outputs to classify content into predefined categories
and extract entities. Falls back to rule-based classifier on failure.
"""

import asyncio
import json

from openai import AsyncOpenAI, BadRequestError, RateLimitError

from app.config import ext
from app.services.intelligence import llm_fallback
from app.services.intelligence.models import (
    ClassifyResult,
    ContentCategory,
    ExtractedEntities,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

VALID_CATEGORIES = {c.value for c in ContentCategory}

SYSTEM_PROMPT = """\
You are a document classifier for Austrian/German municipality content.

Classify the given text into exactly ONE category and extract structured entities.

Categories:
- funding: Grants, subsidies, financial aid programs (Förderungen, Zuschüsse)
- event: Events, ceremonies, festivals, workshops (Veranstaltungen, Termine)
- policy: Regulations, ordinances, laws, council decisions (Verordnungen, Beschlüsse)
- contact: Contact information, office hours, service points (Kontakt, Öffnungszeiten)
- form: Application forms, downloadable documents (Formulare, Anträge)
- announcement: Public notices, press releases (Bekanntmachungen, Mitteilungen)
- minutes: Meeting minutes, agendas, proceedings (Protokolle, Sitzungen)
- report: Annual reports, statistics, evaluations (Berichte, Statistiken)
- general: Content that doesn't fit other categories

Sub-categories (pick all that apply, max 5):
renewable_energy, subsidy, housing, education, environment, infrastructure, social, culture, sports, digitalization

Respond ONLY with valid JSON matching this exact schema:
{
  "category": "<one of the categories above>",
  "confidence": <float 0.0-1.0>,
  "sub_categories": ["<sub_category>", ...],
  "entities": {
    "dates": ["<date strings found>"],
    "deadlines": ["<deadline dates>"],
    "amounts": ["<monetary amounts like EUR 1.000 or € 500>"],
    "contacts": ["<email addresses>"],
    "departments": ["<department/office names>"]
  },
  "summary": "<1-2 sentence summary of the content>"
}"""


class LLMClassifier:
    """OpenAI-based classifier for municipality content."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model = ext.openai_model

    def is_available(self) -> bool:
        return self._client is not None

    def startup(self) -> None:
        if not ext.openai_api_key:
            log.info("llm_classifier_disabled", reason="no OPENAI_API_KEY")
            return
        self._client = AsyncOpenAI(api_key=ext.openai_api_key)
        log.info("llm_classifier_started", model=self._model)

    async def classify(self, content: str, language: str = "de") -> ClassifyResult:
        if not self._client:
            raise RuntimeError("LLM classifier not available")

        cap = ext.classify_max_input_chars
        if len(content) > cap:
            log.info("classify_truncated", chars_in=len(content), chars_kept=cap)
        truncated = content[:cap]

        try:
            response = await self._chat_with_rate_limit_retry(truncated)
        except BadRequestError as exc:
            if "context_length_exceeded" not in str(exc).lower():
                raise
            half = truncated[: len(truncated) // 2]
            log.warning(
                "llm_classify_context_exceeded_retry",
                chars_in=len(truncated),
                chars_retry=len(half),
            )
            try:
                response = await self._chat_with_rate_limit_retry(half)
            except llm_fallback.OUTAGE_ERRORS as outage_exc:
                response = await self._chat_via_fallback(half, outage_exc)
        except llm_fallback.OUTAGE_ERRORS as outage_exc:
            response = await self._chat_via_fallback(truncated, outage_exc)

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        category_str = data.get("category", "general")
        if category_str not in VALID_CATEGORIES:
            category_str = "general"

        category = ContentCategory(category_str)
        confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))

        entities_data = data.get("entities", {})
        entities = ExtractedEntities(
            dates=entities_data.get("dates", [])[:10],
            deadlines=entities_data.get("deadlines", [])[:5],
            amounts=entities_data.get("amounts", [])[:10],
            contacts=entities_data.get("contacts", [])[:10],
            departments=entities_data.get("departments", [])[:5],
        )

        sub_categories = data.get("sub_categories", [])[:5]
        summary = str(data.get("summary", ""))[:300]

        log.info(
            "llm_classify_complete",
            category=category.value,
            confidence=round(confidence, 2),
            sub_categories=sub_categories,
            model=self._model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )

        return ClassifyResult(
            category=category,
            confidence=confidence,
            sub_categories=sub_categories,
            entities=entities,
            summary=summary,
        )

    async def _chat_with_rate_limit_retry(self, user_content: str):
        """Call chat.completions.create with 3-attempt exponential backoff on 429."""
        last_exc: RateLimitError | None = None
        for attempt in range(3):
            try:
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
            except RateLimitError as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning("llm_classify_rate_limit_retry", attempt=attempt + 1, wait_s=wait)
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    async def _chat_via_fallback(self, user_content: str, outage_exc: Exception):
        """Run the classify call against the LiteLLM fallback model.

        Re-raises the original OpenAI outage error if the fallback is disabled
        or if the fallback call itself fails — the upstream Classifier will
        then catch it and degrade to the rule-based path.
        """
        log.warning(
            "llm_classify_openai_unavailable_falling_back",
            error=str(outage_exc),
            error_type=type(outage_exc).__name__,
        )
        try:
            response = await llm_fallback.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
        except Exception as fb_exc:
            log.error("llm_classify_fallback_failed", error=str(fb_exc))
            raise outage_exc from fb_exc
        if response is None:
            log.warning("llm_classify_fallback_disabled")
            raise outage_exc
        log.info("llm_classify_via_fallback", model=llm_fallback.model())
        return response
