"""OpenAI-based metadata extractor for funding documents.

Extracts structured fields such as title, state_or_province, municipality,
target_group, funding_type, status, funding_amount, etc. from funding content
using structured JSON output.

When a ``country`` code is provided, the system prompt constrains
``state_or_province`` to the official list of first-level administrative
divisions for that country, preventing hallucinated region names.
"""

import asyncio
import json
from datetime import datetime, timezone

from openai import AsyncOpenAI, BadRequestError, RateLimitError

from app.config import ext
from app.models.common import StageUsage
from app.services import cost
from app.services.intelligence import llm_router
from app.utils.logger import get_logger

log = get_logger(__name__)

# Sentinel key the extractor adds to its result dict to ship the per-call
# usage record back to the caller (the funding result otherwise becomes
# Qdrant payload, so an unprefixed ``usage`` key would conflict with real
# document metadata). ``IngestService.ingest`` pops this key before
# merging the rest into the stored payload.
_KEY_USAGE = "__usage__"

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

# ---------------------------------------------------------------------------
# Local-language → canonical English-lowercase aliases for province overrides.
# Keyed by ISO country code. The canonical (right-hand) values must match
# PROVINCES_BY_COUNTRY exactly so request overrides and extractor output
# produce the same stored ``metadata.state_or_province`` value regardless of
# input language or casing. Add aliases per country as needed — unknown
# inputs for a country with a known list are dropped during normalization.
# ---------------------------------------------------------------------------
PROVINCE_ALIASES_BY_COUNTRY: dict[str, dict[str, str]] = {
    "AT": {
        # English (canonical, identity entries so case-only inputs still resolve)
        "burgenland": "burgenland",
        "carinthia": "carinthia",
        "lower austria": "lower austria",
        "upper austria": "upper austria",
        "salzburg": "salzburg",
        "styria": "styria",
        "tyrol": "tyrol",
        "vorarlberg": "vorarlberg",
        "vienna": "vienna",
        # German
        "kärnten": "carinthia",
        "niederösterreich": "lower austria",
        "oberösterreich": "upper austria",
        "steiermark": "styria",
        "tirol": "tyrol",
        "wien": "vienna",
    },
    "DE": {
        # German → English (canonical PROVINCES_BY_COUNTRY entries)
        "baden-württemberg": "baden-wurttemberg",
        "bayern": "bavaria",
        "berlin": "berlin",
        "brandenburg": "brandenburg",
        "bremen": "bremen",
        "hamburg": "hamburg",
        "hessen": "hesse",
        "niedersachsen": "lower saxony",
        "mecklenburg-vorpommern": "mecklenburg-vorpommern",
        "nordrhein-westfalen": "north rhine-westphalia",
        "rheinland-pfalz": "rhineland-palatinate",
        "saarland": "saarland",
        "sachsen": "saxony",
        "sachsen-anhalt": "saxony-anhalt",
        "schleswig-holstein": "schleswig-holstein",
        "thüringen": "thuringia",
    },
    "CH": {
        # German / French / Italian → English (canonical PROVINCES_BY_COUNTRY entries)
        "zürich": "zurich",
        "zuerich": "zurich",
        "bern": "bern",
        "berne": "bern",
        "luzern": "lucerne",
        "lucerne": "lucerne",
        "uri": "uri",
        "schwyz": "schwyz",
        "obwalden": "obwalden",
        "nidwalden": "nidwalden",
        "glarus": "glarus",
        "zug": "zug",
        "fribourg": "fribourg",
        "freiburg": "fribourg",
        "solothurn": "solothurn",
        "soleure": "solothurn",
        "basel-stadt": "basel-stadt",
        "bâle-ville": "basel-stadt",
        "basel-landschaft": "basel-landschaft",
        "bâle-campagne": "basel-landschaft",
        "schaffhausen": "schaffhausen",
        "appenzell ausserrhoden": "appenzell ausserrhoden",
        "appenzell innerrhoden": "appenzell innerrhoden",
        "st. gallen": "st. gallen",
        "sankt gallen": "st. gallen",
        "graubünden": "graubunden",
        "graubunden": "graubunden",
        "grisons": "graubunden",
        "aargau": "aargau",
        "argovie": "aargau",
        "thurgau": "thurgau",
        "thurgovie": "thurgau",
        "ticino": "ticino",
        "tessin": "ticino",
        "vaud": "vaud",
        "waadt": "vaud",
        "valais": "valais",
        "wallis": "valais",
        "neuchâtel": "neuchatel",
        "neuchatel": "neuchatel",
        "neuenburg": "neuchatel",
        "genève": "geneva",
        "geneve": "geneva",
        "geneva": "geneva",
        "genf": "geneva",
        "jura": "jura",
    },
    "RO": {
        # Romanian local names + diacritic-stripped variants → canonical
        "alba": "alba",
        "arad": "arad",
        "argeș": "arges", "arges": "arges",
        "bacău": "bacau", "bacau": "bacau",
        "bihor": "bihor",
        "bistrița-năsăud": "bistrita-nasaud", "bistrita-nasaud": "bistrita-nasaud",
        "botoșani": "botosani", "botosani": "botosani",
        "brăila": "braila", "braila": "braila",
        "brașov": "brasov", "brasov": "brasov",
        "bucurești": "bucharest", "bucuresti": "bucharest", "bucharest": "bucharest",
        "buzău": "buzau", "buzau": "buzau",
        "călărași": "calarasi", "calarasi": "calarasi",
        "caraș-severin": "caras-severin", "caras-severin": "caras-severin",
        "cluj": "cluj",
        "constanța": "constanta", "constanta": "constanta",
        "covasna": "covasna",
        "dâmbovița": "dambovita", "dambovita": "dambovita",
        "dolj": "dolj",
        "galați": "galati", "galati": "galati",
        "giurgiu": "giurgiu",
        "gorj": "gorj",
        "harghita": "harghita",
        "hunedoara": "hunedoara",
        "ialomița": "ialomita", "ialomita": "ialomita",
        "iași": "iasi", "iasi": "iasi",
        "ilfov": "ilfov",
        "maramureș": "maramures", "maramures": "maramures",
        "mehedinți": "mehedinti", "mehedinti": "mehedinti",
        "mureș": "mures", "mures": "mures",
        "neamț": "neamt", "neamt": "neamt",
        "olt": "olt",
        "prahova": "prahova",
        "sălaj": "salaj", "salaj": "salaj",
        "satu mare": "satu mare",
        "sibiu": "sibiu",
        "suceava": "suceava",
        "teleorman": "teleorman",
        "timiș": "timis", "timis": "timis",
        "tulcea": "tulcea",
        "vâlcea": "valcea", "valcea": "valcea",
        "vaslui": "vaslui",
        "vrancea": "vrancea",
    },
    "IT": {
        # Italian regions → English canonical
        "abruzzo": "abruzzo",
        "valle d'aosta": "aosta valley", "valle daosta": "aosta valley", "aosta valley": "aosta valley",
        "puglia": "apulia", "apulia": "apulia",
        "basilicata": "basilicata",
        "calabria": "calabria",
        "campania": "campania",
        "emilia-romagna": "emilia-romagna",
        "friuli-venezia giulia": "friuli venezia giulia", "friuli venezia giulia": "friuli venezia giulia",
        "lazio": "lazio",
        "liguria": "liguria",
        "lombardia": "lombardy", "lombardy": "lombardy",
        "marche": "marche",
        "molise": "molise",
        "piemonte": "piedmont", "piedmont": "piedmont",
        "sardegna": "sardinia", "sardinia": "sardinia",
        "sicilia": "sicily", "sicily": "sicily",
        "alto adige": "south tyrol", "südtirol": "south tyrol",
        "trentino-alto adige": "trentino", "trentino": "trentino",
        "toscana": "tuscany", "tuscany": "tuscany",
        "umbria": "umbria",
        "veneto": "veneto",
    },
    "FR": {
        # French region names → English canonical (already mostly French in PROVINCES_BY_COUNTRY)
        "auvergne-rhône-alpes": "auvergne-rhone-alpes", "auvergne-rhone-alpes": "auvergne-rhone-alpes",
        "bourgogne-franche-comté": "bourgogne-franche-comte", "bourgogne-franche-comte": "bourgogne-franche-comte",
        "bretagne": "brittany", "brittany": "brittany",
        "centre-val de loire": "centre-val de loire",
        "corse": "corsica", "corsica": "corsica",
        "grand est": "grand est",
        "hauts-de-france": "hauts-de-france",
        "île-de-france": "ile-de-france", "ile-de-france": "ile-de-france",
        "normandie": "normandy", "normandy": "normandy",
        "nouvelle-aquitaine": "nouvelle-aquitaine",
        "occitanie": "occitanie",
        "pays de la loire": "pays de la loire",
        "provence-alpes-côte d'azur": "provence-alpes-cote d'azur",
        "provence-alpes-cote d'azur": "provence-alpes-cote d'azur",
        "paca": "provence-alpes-cote d'azur",
    },
    "HU": {
        # Hungarian county names → canonical (PROVINCES_BY_COUNTRY uses ASCII-stripped Hungarian)
        "bács-kiskun": "bacs-kiskun", "bacs-kiskun": "bacs-kiskun",
        "baranya": "baranya",
        "békés": "bekes", "bekes": "bekes",
        "borsod-abaúj-zemplén": "borsod-abauj-zemplen", "borsod-abauj-zemplen": "borsod-abauj-zemplen",
        "budapest": "budapest",
        "csongrád-csanád": "csongrad-csanad", "csongrad-csanad": "csongrad-csanad",
        "fejér": "fejer", "fejer": "fejer",
        "győr-moson-sopron": "gyor-moson-sopron", "gyor-moson-sopron": "gyor-moson-sopron",
        "hajdú-bihar": "hajdu-bihar", "hajdu-bihar": "hajdu-bihar",
        "heves": "heves",
        "jász-nagykun-szolnok": "jasz-nagykun-szolnok", "jasz-nagykun-szolnok": "jasz-nagykun-szolnok",
        "komárom-esztergom": "komarom-esztergom", "komarom-esztergom": "komarom-esztergom",
        "nógrád": "nograd", "nograd": "nograd",
        "pest": "pest",
        "somogy": "somogy",
        "szabolcs-szatmár-bereg": "szabolcs-szatmar-bereg", "szabolcs-szatmar-bereg": "szabolcs-szatmar-bereg",
        "tolna": "tolna",
        "vas": "vas",
        "veszprém": "veszprem", "veszprem": "veszprem",
        "zala": "zala",
    },
    "CZ": {
        # Czech regions → English canonical
        "středočeský": "central bohemia", "stredocesky": "central bohemia", "central bohemia": "central bohemia",
        "královéhradecký": "hradec kralove", "kralovehradecky": "hradec kralove", "hradec kralove": "hradec kralove",
        "karlovarský": "karlovy vary", "karlovarsky": "karlovy vary", "karlovy vary": "karlovy vary",
        "liberecký": "liberec", "liberecky": "liberec", "liberec": "liberec",
        "moravskoslezský": "moravian-silesian", "moravskoslezsky": "moravian-silesian", "moravian-silesian": "moravian-silesian",
        "olomoucký": "olomouc", "olomoucky": "olomouc", "olomouc": "olomouc",
        "pardubický": "pardubice", "pardubicky": "pardubice", "pardubice": "pardubice",
        "plzeňský": "plzen", "plzensky": "plzen", "plzen": "plzen",
        "praha": "prague", "prague": "prague",
        "jihočeský": "south bohemia", "jihocesky": "south bohemia", "south bohemia": "south bohemia",
        "jihomoravský": "south moravia", "jihomoravsky": "south moravia", "south moravia": "south moravia",
        "ústecký": "usti nad labem", "ustecky": "usti nad labem", "usti nad labem": "usti nad labem",
        "vysočina": "vysocina", "vysocina": "vysocina",
        "zlínský": "zlin", "zlinsky": "zlin", "zlin": "zlin",
    },
    "SK": {
        # Slovak regions → English canonical (PROVINCES_BY_COUNTRY uses Slovak with diacritics for some)
        "banská bystrica": "banská bystrica", "banska bystrica": "banská bystrica",
        "bratislava": "bratislava",
        "košice": "kosice", "kosice": "kosice",
        "nitra": "nitra",
        "prešov": "presov", "presov": "presov",
        "trenčín": "trencin", "trencin": "trencin",
        "trnava": "trnava",
        "žilina": "zilina", "zilina": "zilina",
    },
    "SI": {
        # Slovenian statistical regions → English canonical
        "zasavska": "central sava", "central sava": "central sava",
        "osrednjeslovenska": "central slovenia", "central slovenia": "central slovenia",
        "koroška": "carinthia", "koroska": "carinthia", "carinthia": "carinthia",
        "obalno-kraška": "coastal-karst", "obalno-kraska": "coastal-karst", "coastal-karst": "coastal-karst",
        "podravska": "drava", "drava": "drava",
        "goriška": "gorizia", "goriska": "gorizia", "gorizia": "gorizia",
        "notranjsko-kraška": "inner carniola-karst", "notranjsko-kraska": "inner carniola-karst", "inner carniola-karst": "inner carniola-karst",
        "primorsko-notranjska": "littoral-inner carniola", "littoral-inner carniola": "littoral-inner carniola",
        "posavska": "lower sava", "lower sava": "lower sava",
        "pomurska": "mura", "mura": "mura",
        "podravje": "podravje",
        "pomurje": "pomurje",
        "savinjska": "savinja", "savinja": "savinja",
        "jugovzhodna slovenija": "southeast slovenia", "southeast slovenia": "southeast slovenia",
        "gorenjska": "upper carniola", "upper carniola": "upper carniola",
    },
    "HR": {
        # Croatian counties → English canonical
        "bjelovarsko-bilogorska": "bjelovar-bilogora", "bjelovar-bilogora": "bjelovar-bilogora",
        "brodsko-posavska": "brod-posavina", "brod-posavina": "brod-posavina",
        "dubrovačko-neretvanska": "dubrovnik-neretva", "dubrovacko-neretvanska": "dubrovnik-neretva", "dubrovnik-neretva": "dubrovnik-neretva",
        "istarska": "istria", "istria": "istria",
        "karlovačka": "karlovac", "karlovacka": "karlovac", "karlovac": "karlovac",
        "koprivničko-križevačka": "koprivnica-krizevci", "koprivnicko-krizevacka": "koprivnica-krizevci", "koprivnica-krizevci": "koprivnica-krizevci",
        "krapinsko-zagorska": "krapina-zagorje", "krapina-zagorje": "krapina-zagorje",
        "ličko-senjska": "lika-senj", "licko-senjska": "lika-senj", "lika-senj": "lika-senj",
        "međimurska": "medimurje", "medimurska": "medimurje", "medimurje": "medimurje",
        "osječko-baranjska": "osijek-baranja", "osjecko-baranjska": "osijek-baranja", "osijek-baranja": "osijek-baranja",
        "požeško-slavonska": "pozega-slavonia", "pozesko-slavonska": "pozega-slavonia", "pozega-slavonia": "pozega-slavonia",
        "primorsko-goranska": "primorje-gorski kotar", "primorje-gorski kotar": "primorje-gorski kotar",
        "šibensko-kninska": "sibenik-knin", "sibensko-kninska": "sibenik-knin", "sibenik-knin": "sibenik-knin",
        "sisačko-moslavačka": "sisak-moslavina", "sisacko-moslavacka": "sisak-moslavina", "sisak-moslavina": "sisak-moslavina",
        "splitsko-dalmatinska": "split-dalmatia", "split-dalmatia": "split-dalmatia",
        "varaždinska": "varazdin", "varazdinska": "varazdin", "varazdin": "varazdin",
        "virovitičko-podravska": "virovitica-podravina", "viroviticko-podravska": "virovitica-podravina", "virovitica-podravina": "virovitica-podravina",
        "vukovarsko-srijemska": "vukovar-srijem", "vukovar-srijem": "vukovar-srijem",
        "zadarska": "zadar", "zadar": "zadar",
        "grad zagreb": "zagreb", "zagreb": "zagreb",
        "zagrebačka": "zagreb county", "zagrebacka": "zagreb county", "zagreb county": "zagreb county",
    },
}


def normalize_provinces(country: str | None, names: list[str] | None) -> list[str]:
    """Canonicalize province names to English-lowercase form, deduped.

    Resolution order (per name):
      1. Strip + lowercase.
      2. If the country has an entry in ``PROVINCE_ALIASES_BY_COUNTRY``, look
         up the alias (handles local-language → English).
      3. If the country has an entry in ``PROVINCES_BY_COUNTRY`` (the
         authoritative list), keep only values present in that list.
      4. Otherwise return the lowercased value (no validation).

    Empty / unknown values are dropped — the resulting list is always a clean
    subset suitable for search-time filtering. Order is preserved.
    """
    if not names:
        return []
    country_code = (country or "").upper().strip()
    aliases = PROVINCE_ALIASES_BY_COUNTRY.get(country_code, {})
    known = PROVINCES_BY_COUNTRY.get(country_code)

    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        if not isinstance(raw, str):
            continue
        key = raw.strip().lower()
        if not key:
            continue
        canonical = aliases.get(key, key)
        if known is not None and canonical not in known:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


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
  "application_form": ["<URLs to application forms (PDF or online form pages) referenced in the document. Multiple allowed. If no URL is available but a form is named, include the form name verbatim. Empty list if none>"],
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
        self._model: str = ext.openai_model
        self._provider: str = "openai"

    def is_available(self) -> bool:
        return self._client is not None

    def startup(self) -> None:
        try:
            resolved = llm_router.for_funding()
        except llm_router.LLMRouterError as exc:
            log.info("funding_extractor_disabled", reason=str(exc))
            return
        self._client = llm_router.get_client(resolved)
        self._model = resolved.model
        self._provider = resolved.provider
        log.info(
            "funding_extractor_started",
            provider=resolved.provider,
            model=self._model,
        )

    async def extract(
        self,
        content: str,
        source_url: str = "",
        country: str | None = None,
    ) -> dict:
        """Extract funding metadata from content. Returns a flat dict."""
        if not self._client:
            raise RuntimeError("Funding extractor not available (no OpenAI key)")

        cap = ext.funding_max_input_chars
        if len(content) > cap:
            log.info("funding_extract_truncated", chars_in=len(content), chars_kept=cap)
        truncated = content[:cap]
        system_prompt = _build_system_prompt(country)

        try:
            response = await self._chat_with_rate_limit_retry(system_prompt, truncated)
        except BadRequestError as exc:
            if "context_length_exceeded" not in str(exc).lower():
                raise
            half = truncated[: len(truncated) // 2]
            log.warning(
                "funding_extract_context_exceeded_retry",
                chars_in=len(truncated),
                chars_retry=len(half),
            )
            response = await self._chat_with_rate_limit_retry(system_prompt, half)

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Normalize state_or_province through the shared helper: applies the
        # per-country alias map (local-language → English), drops values not in
        # the official list, and dedupes. Same code path used by request
        # overrides on /ingest/at, /ingest, /ingest/stream, so an extractor-
        # produced and a caller-supplied value land in the same canonical form.
        country_code = str(data.get("country_code", country or "")).upper().strip()
        raw_states = _as_list(data.get("state_or_province", []))
        states_raw = normalize_provinces(country_code, raw_states)
        if PROVINCES_BY_COUNTRY.get(country_code) is not None:
            dropped = [
                s for s in raw_states
                if isinstance(s, str) and s.strip()
                and not normalize_provinces(country_code, [s])
            ]
            if dropped:
                log.warning(
                    "funding_provinces_not_in_known_list",
                    extracted=dropped,
                    country=country_code,
                )

        result = {
            "title": str(data.get("title", "")),
            "program_name": str(data.get("program_name", "")),
            "processing_office": _as_list(data.get("processing_office", [])),
            "contract_email": _as_list(data.get("contract_email", [])),
            "contract_phone": _as_list(data.get("contract_phone", [])),
            "application_form": _as_list(data.get("application_form", [])),
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

        usage_obj = response.usage
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
        completion_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
        cached_tokens = 0
        details = getattr(usage_obj, "prompt_tokens_details", None) if usage_obj else None
        if details is not None:
            cached_tokens = getattr(details, "cached_tokens", 0) or 0

        log.info(
            "funding_metadata_extracted",
            title=result["title"][:80],
            country=result["country_code"],
            states=result["state_or_province"],
            status=result["status"],
            tokens_used=getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
        )

        # Stash usage on the result dict under a sentinel key so the caller
        # (``ingest_online`` via ``_safe_extract_funding``) can lift it out
        # before merging the rest into Qdrant metadata. ``_KEY_USAGE`` is
        # filtered by ``IngestService.ingest`` so it never lands in the
        # stored payload.
        result[_KEY_USAGE] = StageUsage(
            stage="funding",
            provider=self._provider,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost.chat_cost(
                self._provider, self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            ),
        )
        return result

    async def _chat_with_rate_limit_retry(self, system_prompt: str, user_content: str):
        """Call chat.completions.create with 3-attempt exponential backoff on 429."""
        last_exc: RateLimitError | None = None
        for attempt in range(3):
            try:
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
            except RateLimitError as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning("funding_extract_rate_limit_retry", attempt=attempt + 1, wait_s=wait)
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc


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
