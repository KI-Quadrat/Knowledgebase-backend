"""OpenAI-based metadata extractor for funding documents.

Extracts structured fields such as title, state_or_province, municipality,
target_group, funding_type, status, funding_amount, etc. from funding content
using structured JSON output.

When a ``country`` code is provided, the system prompt constrains
``state_or_province`` to the official list of first-level administrative
divisions for that country, preventing hallucinated region names.
"""

import json
from datetime import datetime, timezone

from openai import AsyncOpenAI

from app.config import ext
from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Official first-level administrative divisions per country.
# All names in English lowercase for consistent filtering.
# Extend this dict to support more countries.
# ---------------------------------------------------------------------------
PROVINCES_BY_COUNTRY: dict[str, list[str]] = {
    "AT": [
        "burgenland", "carinthia", "lower austria", "upper austria",
        "salzburg", "styria", "tyrol", "vorarlberg", "vienna",
    ],
    "DE": [
        "baden-wurttemberg", "bavaria", "berlin", "brandenburg", "bremen",
        "hamburg", "hesse", "lower saxony", "mecklenburg-vorpommern",
        "north rhine-westphalia", "rhineland-palatinate", "saarland",
        "saxony", "saxony-anhalt", "schleswig-holstein", "thuringia",
    ],
    "CH": [
        "aargau", "appenzell ausserrhoden", "appenzell innerrhoden",
        "basel-landschaft", "basel-stadt", "bern", "fribourg", "geneva",
        "glarus", "graubunden", "jura", "lucerne", "neuchatel", "nidwalden",
        "obwalden", "schaffhausen", "schwyz", "solothurn", "st. gallen",
        "thurgau", "ticino", "uri", "valais", "vaud", "zug", "zurich",
    ],
    "RO": [
        "alba", "arad", "arges", "bacau", "bihor", "bistrita-nasaud",
        "botosani", "braila", "brasov", "bucharest", "buzau", "calarasi",
        "caras-severin", "cluj", "constanta", "covasna", "dambovita",
        "dolj", "galati", "giurgiu", "gorj", "harghita", "hunedoara",
        "ialomita", "iasi", "ilfov", "maramures", "mehedinti", "mures",
        "neamt", "olt", "prahova", "salaj", "satu mare", "sibiu",
        "suceava", "teleorman", "timis", "tulcea", "valcea", "vaslui",
        "vrancea",
    ],
    "IT": [
        "abruzzo", "aosta valley", "apulia", "basilicata", "calabria",
        "campania", "emilia-romagna", "friuli venezia giulia", "lazio",
        "liguria", "lombardy", "marche", "molise", "piedmont", "sardinia",
        "sicily", "south tyrol", "trentino", "tuscany", "umbria", "veneto",
    ],
    "FR": [
        "auvergne-rhone-alpes", "bourgogne-franche-comte", "brittany",
        "centre-val de loire", "corsica", "grand est",
        "hauts-de-france", "ile-de-france", "normandy", "nouvelle-aquitaine",
        "occitanie", "pays de la loire", "provence-alpes-cote d'azur",
    ],
    "HU": [
        "bacs-kiskun", "baranya", "bekes", "borsod-abauj-zemplen",
        "budapest", "csongrad-csanad", "fejer", "gyor-moson-sopron",
        "hajdu-bihar", "heves", "jasz-nagykun-szolnok", "komarom-esztergom",
        "nograd", "pest", "somogy", "szabolcs-szatmar-bereg", "tolna",
        "vas", "veszprem", "zala",
    ],
    "CZ": [
        "central bohemia", "hradec kralove", "karlovy vary", "liberec",
        "moravian-silesian", "olomouc", "pardubice", "plzen", "prague",
        "south bohemia", "south moravia", "usti nad labem", "vysocina",
        "zlin",
    ],
    "SK": [
        "banská bystrica", "bratislava", "kosice", "nitra", "presov",
        "trencin", "trnava", "zilina",
    ],
    "SI": [
        "central sava", "central slovenia", "carinthia", "coastal-karst",
        "drava", "gorizia", "inner carniola-karst", "littoral-inner carniola",
        "lower sava", "mura", "podravje", "pomurje", "savinja",
        "southeast slovenia", "upper carniola",
    ],
    "HR": [
        "bjelovar-bilogora", "brod-posavina", "dubrovnik-neretva",
        "istria", "karlovac", "koprivnica-krizevci", "krapina-zagorje",
        "lika-senj", "medimurje", "osijek-baranja", "pozega-slavonia",
        "primorje-gorski kotar", "sibenik-knin", "sisak-moslavina",
        "split-dalmatia", "varazdin", "virovitica-podravina",
        "vukovar-srijem", "zadar", "zagreb", "zagreb county",
    ],
}

_BASE_SYSTEM_PROMPT = """\
You are a metadata extractor for government funding documents.

Given the text of a funding page, extract structured metadata.
Extract the VALUES in their original language from the source document,
but use the English field names shown below.
If a field cannot be determined from the text, use an empty string for strings
or null for nullable fields.

{province_constraint}

Respond ONLY with valid JSON matching this exact schema:
{{
  "title": "<title of the funding program>",
  "program_name": "<official program name, if distinct from the title — otherwise empty string>",
  "processing_office": ["<names of the offices/departments that process applications. Multiple allowed. Empty list if unknown>"],
  "contract_email": ["<contract/contact email addresses found in the document. Multiple allowed. Empty list if none>"],
  "contract_phone": ["<contract/contact phone numbers found in the document. Multiple allowed. Empty list if none>"],
  "country_code": "<ISO 3166-1 alpha-2 country code, e.g. AT, DE, RO>",
  "state_or_province": ["<official states/provinces in english lowercase — see constraint above. Multiple allowed. Empty list if unknown>"],
  "city": ["<city names in english lowercase. Multiple allowed. Empty list if unknown>"],
  "target_group": ["<target groups, e.g. associations, individuals, businesses>"],
  "funding_type": "<funding type, e.g. direct grant, subsidy, loan>",
  "status": "<active | inactive | expiring | unknown>",
  "funding_amount": "<funding amount or range, e.g. up to EUR 5,000 — or empty string if unknown>",
  "thematic_focus": ["<thematic focus areas, e.g. sports, environment, education>"],
  "eligibility_criteria": "<eligibility criteria and application requirements>",
  "legal_basis": "<legal basis or regulation>",
  "funding_provider": ["<funding provider organizations>"],
  "reference_number": "<reference number or ID if found, otherwise null>",
  "start_date": "<start date in DD.MM.YYYY format or empty string>",
  "end_date": "<end date in DD.MM.YYYY format, or 'unlimited', or empty string>"
}}"""

_PROVINCE_KNOWN = (
    "The country is {country_code}. "
    "For `state_or_province`, each value MUST be EXACTLY one of these "
    "(english lowercase): {provinces}. "
    "Include all provinces that the funding applies to. "
    "If the funding is nationwide, include all provinces. "
    "If the location does not clearly match any of these, "
    "leave `state_or_province` as an empty list."
)

_PROVINCE_UNKNOWN = (
    "No country was specified. Infer the country from the document content. "
    "For `state_or_province`, use the official first-level administrative "
    "division names in english lowercase. If unsure, leave it as an empty list."
)


def _build_system_prompt(country: str | None) -> str:
    country_upper = country.upper().strip() if country else None
    if country_upper and country_upper in PROVINCES_BY_COUNTRY:
        provinces = ", ".join(PROVINCES_BY_COUNTRY[country_upper])
        constraint = _PROVINCE_KNOWN.format(
            country_code=country_upper, provinces=provinces,
        )
    elif country_upper:
        constraint = (
            f"The country is {country_upper}. "
            "For `state_or_province`, use the official first-level administrative "
            "division names in english lowercase. If unsure, leave it as an empty list."
        )
    else:
        constraint = _PROVINCE_UNKNOWN
    return _BASE_SYSTEM_PROMPT.format(province_constraint=constraint)


class FundingExtractor:
    """Extracts structured metadata from funding documents via OpenAI."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model = ext.openai_model

    def is_available(self) -> bool:
        return self._client is not None

    def startup(self) -> None:
        if not ext.openai_api_key:
            log.info("funding_extractor_disabled", reason="no OPENAI_API_KEY")
            return
        self._client = AsyncOpenAI(api_key=ext.openai_api_key)
        log.info("funding_extractor_started", model=self._model)

    async def extract(
        self,
        content: str,
        source_url: str = "",
        country: str | None = None,
    ) -> dict:
        """Extract funding metadata from content. Returns a flat dict."""
        if not self._client:
            raise RuntimeError("Funding extractor not available (no OpenAI key)")

        truncated = content[:6000] if len(content) > 6000 else content
        system_prompt = _build_system_prompt(country)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated},
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Normalize state_or_province to lowercase list and validate against known list
        country_code = str(data.get("country_code", country or "")).upper().strip()
        states_raw = _as_list(data.get("state_or_province", []))
        states_raw = [s.lower().strip() for s in states_raw if s.strip()]
        known = PROVINCES_BY_COUNTRY.get(country_code)
        if known:
            invalid = [s for s in states_raw if s not in known]
            if invalid:
                log.warning(
                    "funding_provinces_not_in_known_list",
                    extracted=invalid,
                    country=country_code,
                )
            states_raw = [s for s in states_raw if s in known]

        result = {
            "title": str(data.get("title", "")),
            "program_name": str(data.get("program_name", "")),
            "processing_office": _as_list(data.get("processing_office", [])),
            "contract_email": _as_list(data.get("contract_email", [])),
            "contract_phone": _as_list(data.get("contract_phone", [])),
            "country_code": country_code,
            "state_or_province": states_raw,
            "city": [c.lower().strip() for c in _as_list(data.get("city", [])) if c.strip()],
            "target_group": _as_list(data.get("target_group", [])),
            "funding_type": str(data.get("funding_type", "")),
            "status": _validated_status(data.get("status", "unknown")),
            "funding_amount": str(data.get("funding_amount", "")),
            "thematic_focus": _as_list(data.get("thematic_focus", [])),
            "eligibility_criteria": str(data.get("eligibility_criteria", "")),
            "legal_basis": str(data.get("legal_basis", "")),
            "funding_provider": _as_list(data.get("funding_provider", [])),
            "reference_number": data.get("reference_number"),
            "start_date": str(data.get("start_date", "")),
            "end_date": str(data.get("end_date", "")),
            "scraped_at": scraped_at,
        }

        log.info(
            "funding_metadata_extracted",
            title=result["title"][:80],
            country=result["country_code"],
            states=result["state_or_province"],
            status=result["status"],
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )

        return result


def _as_list(val: object) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val if str(v).strip()][:20]
    # Defensive: wrap a non-empty scalar so we don't lose data when the LLM
    # returns a bare string for an array-typed field.
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _validated_status(val: object) -> str:
    allowed = {"active", "inactive", "expiring", "unknown"}
    s = str(val).lower().strip()
    return s if s in allowed else "unknown"
