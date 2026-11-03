"""
Text preprocessing and feature engineering for fraud call transcripts.

Implements:
- Text normalization
- Fraud-relevant pattern detection (URLs, phone numbers, financial terms, urgency)
- Language identification
- Preserves fraud-relevant phrases (numbers, currency, negations, OTP terms)
- Extensible for Hindi/Hinglish

Design principle: keyword detection is a SUPPLEMENTARY feature signal,
NOT the primary classification mechanism. The BiLSTM learns patterns
from the full transcript.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (case-insensitive substring matching)
# ---------------------------------------------------------------------------

_FINANCIAL_TERMS: List[str] = [
    "account", "bank", "credit", "debit", "social security", "ssn",
    "payment", "wire transfer", "gift card", "itunes", "google play",
    "money order", "refund", "irs", "tax",
]

_URGENCY_TERMS: List[str] = [
    "immediately", "urgent", "emergency", "expires", "deadline",
    "final notice", "last chance", "suspended", "arrest", "warrant",
    "lawsuit", "legal action", "deported",
]

_THREAT_TERMS: List[str] = [
    "arrest", "jail", "prison", "deport", "law enforcement",
    "officer", "badge", "federal", "criminal charges", "lawsuit", "sue",
]

_OTP_TERMS: List[str] = [
    "otp", "one-time password", "verification code",
    "confirm your account", "verify your identity", "security code",
]

_IMPERSONATION_TERMS: List[str] = [
    "irs", "social security administration", "social security", "ssa", "medicare",
    "fbi", "dea", "us government", "microsoft", "apple", "amazon",
    "bank of america", "chase", "wells fargo",
]

# ---------------------------------------------------------------------------
# URL / phone-number regex patterns
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b\w+\.(?:com|org|net|gov|edu|io|co)\b",
    re.IGNORECASE,
)

_PHONE_PATTERN = re.compile(
    r"(?:1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|"
    r"\b1-?8[0-9]{2}-\d{3}-\d{4}\b"
    r"|"
    r"\b\d{10}\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudFeatures:
    """Structured container for fraud-signal features extracted from a transcript."""

    text_length: int = 0
    word_count: int = 0
    contains_url: bool = False
    contains_phone_number: bool = False
    contains_financial_terms: bool = False
    contains_urgency_terms: bool = False
    contains_threat_terms: bool = False
    contains_otp_terms: bool = False
    contains_impersonation_terms: bool = False
    repeated_word_ratio: float = 0.0
    detected_language: str = "unknown"
    language_confidence: float = 0.0
    url_count: int = 0
    phone_number_count: int = 0


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> Tuple[str, float]:
    """Detect the language of *text*.

    Returns:
        Tuple of (language_code, confidence).  Returns ("unknown", 0.0) on
        failure or when text is too short to detect reliably.
    """
    if not text or len(text.strip()) < 10:
        return ("unknown", 0.0)

    try:
        from langdetect import detect_langs  # type: ignore
        results = detect_langs(text)
        if results:
            top = results[0]
            return (top.lang, float(top.prob))
    except ImportError:
        logger.warning(
            "langdetect is not installed. Language detection unavailable. "
            "Install with: pip install langdetect"
        )
    except Exception as exc:
        logger.debug("Language detection failed: %s", exc)

    return ("unknown", 0.0)


# ---------------------------------------------------------------------------
# URL / phone extraction
# ---------------------------------------------------------------------------

def extract_urls(text: str) -> List[str]:
    """Extract URLs from *text*.

    Returns:
        List of URL strings found (may be empty).
    """
    if not text:
        return []
    return _URL_PATTERN.findall(text)


def extract_phone_numbers(text: str) -> List[str]:
    """Extract US phone-number patterns from *text*.

    Covers formats: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxxxxxxxxx, 1-800 style.

    Returns:
        List of matched strings.
    """
    if not text:
        return []
    return _PHONE_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Repeated-word detection
# ---------------------------------------------------------------------------

def detect_repeated_words(text: str) -> float:
    """Compute the ratio of immediately repeated word occurrences in *text*.

    E.g. "please please call now now" → 2 repeated pairs out of 4 transitions
    = 0.5.

    Returns:
        Float in [0, 1].  0.0 if text has fewer than 2 words.
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if len(words) < 2:
        return 0.0

    repeated = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return repeated / (len(words) - 1)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")

# Negation words to preserve (never strip even when cleaning)
_NEGATIONS = {"not", "no", "nor", "never", "neither", "nobody", "none",
              "nothing", "nowhere", "don't", "doesn't", "didn't", "can't",
              "cannot", "couldn't", "won't", "wouldn't", "shouldn't",
              "haven't", "hasn't", "hadn't", "isn't", "aren't", "wasn't",
              "weren't"}


def normalize_text(
    text: str,
    preserve_numbers: bool = True,
    preserve_negations: bool = True,
) -> str:
    """Normalise *text* for downstream processing or model input.

    Steps applied:
    1. Lowercase.
    2. Unicode normalisation (NFC → decompose → strip combining marks except
       currency/special symbols).
    3. Remove ASCII control characters.
    4. Collapse whitespace.

    Numbers, currency symbols ($, £, €, ¥, ₹), and negation words are intentionally
    kept because they carry strong fraud-signal value.

    Args:
        text: Raw input string.
        preserve_numbers: Keep digit characters (default True).
        preserve_negations: Retain negation words (default True; no-op because
            we never strip words—only whitespace and control chars).

    Returns:
        Normalised string.
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Unicode normalisation (NFC keeps currency symbols intact)
    text = unicodedata.normalize("NFC", text)

    # 3. Remove ASCII control characters (keep \t \n \r for now → collapse later)
    text = _CONTROL_CHAR_RE.sub(" ", text)

    # 4. Collapse all whitespace (tabs, newlines, multiple spaces) to single space
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Keyword-set helpers
# ---------------------------------------------------------------------------

def _contains_any(text_lower: str, keywords: List[str]) -> bool:
    """Return True if *text_lower* contains any keyword (substring match)."""
    return any(kw in text_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_fraud_features(text: str) -> FraudFeatures:
    """Extract all fraud-relevant features from *text*.

    Args:
        text: Raw transcript string.

    Returns:
        Populated :class:`FraudFeatures` instance.
    """
    if not text:
        return FraudFeatures()

    text_lower = text.lower()
    words = text.split()

    urls = extract_urls(text)
    phones = extract_phone_numbers(text)
    lang, lang_conf = detect_language(text)

    return FraudFeatures(
        text_length=len(text),
        word_count=len(words),
        contains_url=bool(urls),
        contains_phone_number=bool(phones),
        contains_financial_terms=_contains_any(text_lower, _FINANCIAL_TERMS),
        contains_urgency_terms=_contains_any(text_lower, _URGENCY_TERMS),
        contains_threat_terms=_contains_any(text_lower, _THREAT_TERMS),
        contains_otp_terms=_contains_any(text_lower, _OTP_TERMS),
        contains_impersonation_terms=_contains_any(text_lower, _IMPERSONATION_TERMS),
        repeated_word_ratio=detect_repeated_words(text),
        detected_language=lang,
        language_confidence=lang_conf,
        url_count=len(urls),
        phone_number_count=len(phones),
    )


# ---------------------------------------------------------------------------
# Model-input preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_model(
    text: str,
    max_length: Optional[int] = None,
) -> str:
    """Prepare *text* for BiLSTM model input.

    Applies :func:`normalize_text` and optionally truncates to *max_length*
    words.

    Args:
        text: Raw transcript string.
        max_length: Maximum number of words to keep (None = no truncation).

    Returns:
        Processed string ready for the TextVectorization layer.
    """
    processed = normalize_text(text)

    if max_length is not None:
        words = processed.split()
        if len(words) > max_length:
            processed = " ".join(words[:max_length])

    return processed


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_extract_features(texts: List[str]) -> List[FraudFeatures]:
    """Extract :class:`FraudFeatures` from each text in *texts*.

    Args:
        texts: List of transcript strings.

    Returns:
        List of :class:`FraudFeatures` instances (same order as input).
    """
    results: List[FraudFeatures] = []
    for i, text in enumerate(texts):
        try:
            results.append(extract_fraud_features(text))
        except Exception as exc:
            logger.warning("Feature extraction failed for text index %d: %s", i, exc)
            results.append(FraudFeatures())
    return results


def features_to_dict(features: FraudFeatures) -> Dict:
    """Convert a :class:`FraudFeatures` instance to a flat dictionary.

    Suitable for storing as a row in a pandas DataFrame.

    Args:
        features: Populated FraudFeatures dataclass.

    Returns:
        Dictionary with all field names as keys.
    """
    return {
        "text_length": features.text_length,
        "word_count": features.word_count,
        "contains_url": features.contains_url,
        "contains_phone_number": features.contains_phone_number,
        "contains_financial_terms": features.contains_financial_terms,
        "contains_urgency_terms": features.contains_urgency_terms,
        "contains_threat_terms": features.contains_threat_terms,
        "contains_otp_terms": features.contains_otp_terms,
        "contains_impersonation_terms": features.contains_impersonation_terms,
        "repeated_word_ratio": features.repeated_word_ratio,
        "detected_language": features.detected_language,
        "language_confidence": features.language_confidence,
        "url_count": features.url_count,
        "phone_number_count": features.phone_number_count,
    }
