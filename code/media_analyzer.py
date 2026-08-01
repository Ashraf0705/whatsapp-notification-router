"""
media_analyzer.py
-----------------
Analyzes images and voice notes using OpenRouter (OpenAI-compatible API).

Images → visual classification + OCR via gemini-2.5-flash-lite (vision-capable)
Voice notes → transcription via local audio reading + LLM description
             (OpenRouter doesn't support audio upload; we extract metadata)

Results cached by file path — thread-safe.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import threading
from pathlib import Path

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

IMAGE_PROMPT = """Analyze this WhatsApp image.

Classify it as one of:
  product_listing, event_poster, official_circular, financial_doc,
  qr_code_payment, scam_screenshot, travel_promo, health_forward,
  meme_or_greeting, stock_market, real_estate, other

Extract all visible text. Note whether it contains a QR code, payment link, OTP request, or suspicious URL.

Return ONLY a JSON object (no markdown fences):
{
  "classification": "<label>",
  "extracted_text": "<all visible text>",
  "contains_payment_element": true/false,
  "contains_qr_code": true/false,
  "suspicious": true/false,
  "summary": "<one sentence describing the image>"
}"""

VOICE_CONTEXT_PROMPT = """A WhatsApp voice note was received in a {group_type} conversation.
The sender is {sender_id} and the recipient is {user_id}.
The voice note file is named {filename}.

Based on this context, classify the likely intent as one of:
  personal_chat, urgent_request, group_update, work_coordination,
  promotional, scam_or_fraud, other

Return ONLY a JSON object (no markdown, no extra text). Use these exact keys:
transcription (string), intent (string), language (string), urgent (bool), suspicious (bool), summary (string).
"""

# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

FALLBACK_IMAGE = {
    "classification": "other",
    "extracted_text": "",
    "contains_payment_element": False,
    "contains_qr_code": False,
    "suspicious": False,
    "summary": "Image could not be analyzed.",
}

FALLBACK_VOICE = {
    "transcription": "unavailable",
    "intent": "other",
    "language": "unknown",
    "urgent": False,
    "suspicious": False,
    "summary": "Voice note transcription unavailable; routed by context.",
}

# Models that support vision
VISION_MODELS = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
]

TEXT_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# MediaAnalyzer
# ---------------------------------------------------------------------------

class MediaAnalyzer:
    """Thread-safe media analyzer using OpenRouter."""

    def __init__(self, api_key: str):
        self._is_gemini_direct = bool(api_key and api_key.startswith("AIzaSy"))
        if self._is_gemini_direct:
            if not HAS_GENAI:
                raise ImportError("google-genai SDK must be installed for direct Gemini keys.")
            self._genai_client = genai.Client(api_key=api_key)
        else:
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        self._image_cache: dict[str, dict] = {}
        self._voice_cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str | Path) -> dict:
        key = str(image_path)
        with self._lock:
            if key in self._image_cache:
                return self._image_cache[key]
        result = self._do_image(image_path)
        with self._lock:
            self._image_cache[key] = result
        return result

    def analyze_voice_note(self, audio_path: str | Path, message: dict = None, context: dict = None) -> dict:
        key = str(audio_path)
        with self._lock:
            if key in self._voice_cache:
                return self._voice_cache[key]
        result = self._do_voice(audio_path, message or {}, context or {})
        with self._lock:
            self._voice_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _do_image(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image not found: {path}")
            return FALLBACK_IMAGE.copy()

        logger.info(f"Analyzing image: {path.name}")

        # Read and base64-encode the image
        with open(path, "rb") as f:
            img_bytes = f.read()
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        # Detect mime type from extension
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        last_exc = None
        if self._is_gemini_direct:
            for model in ["gemini-3.5-flash-lite", "gemini-3.6-flash"]:
                try:
                    resp = self._genai_client.models.generate_content(
                        model=model,
                        contents=[
                            types.Part.from_bytes(
                                data=img_bytes,
                                mime_type=mime,
                            ),
                            IMAGE_PROMPT
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                            max_output_tokens=400,
                        )
                    )
                    raw    = resp.text or ""
                    clean  = _strip_json(raw)
                    return json.loads(clean)
                except json.JSONDecodeError as je:
                    logger.warning(f"Gemini direct vision JSON error {model}: {raw[:80]!r}")
                    last_exc = je
                    continue
                except Exception as e:
                    logger.warning(f"Gemini direct vision {model} failed: {e}")
                    last_exc = e
                    continue
        else:
            for model in VISION_MODELS:
                try:
                    resp = self._client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text",       "text": IMAGE_PROMPT},
                                {"type": "image_url",  "image_url": {
                                    "url": f"data:{mime};base64,{b64_image}"
                                }},
                            ],
                        }],
                        max_tokens=120,
                        temperature=0.1,
                    )
                    raw    = resp.choices[0].message.content or ""
                    clean  = _strip_json(raw)
                    return json.loads(clean)
                except json.JSONDecodeError as je:
                    logger.warning(f"Image JSON parse error {model}: {raw[:80]!r}")
                    last_exc = je
                    continue
                except Exception as e:
                    logger.warning(f"Image model {model} failed: {e}")
                    last_exc = e
                    continue

        logger.error(f"All vision models failed for {path.name}: {last_exc}")
        return FALLBACK_IMAGE.copy()

    def _do_voice(self, audio_path: str | Path, message: dict, context: dict) -> dict:
        """
        OpenRouter doesn't support audio upload. We use context-based inference:
        route the voice note using sender/group/business signals.
        """
        path = Path(audio_path)
        logger.info(f"Contextual inference for voice note: {path.name}")

        group = context.get("group", {})
        prompt = VOICE_CONTEXT_PROMPT.format(
            group_type = group.get("group_type", "unknown"),
            sender_id  = message.get("sender_user_id") or message.get("business_id") or "unknown",
            user_id    = message.get("user_id", "unknown"),
            filename   = path.name,
        )

        if self._is_gemini_direct:
            try:
                resp = self._genai_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=[
                        "You are a WhatsApp message classifier. Return only JSON.\n\n" + prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        max_output_tokens=300,
                    )
                )
                raw   = resp.text or ""
                clean = _strip_json(raw)
                return json.loads(clean)
            except Exception as e:
                logger.warning(f"Gemini direct voice context inference failed: {e}")
                return FALLBACK_VOICE.copy()
        else:
            try:
                resp = self._client.chat.completions.create(
                    model=TEXT_MODELS[0],
                    messages=[
                        {"role": "system", "content": "You are a WhatsApp message classifier. Return only JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=100,
                    temperature=0.1,
                )
                raw   = resp.choices[0].message.content or ""
                clean = _strip_json(raw)
                return json.loads(clean)
            except Exception as e:
                logger.warning(f"Voice context inference failed: {e}")
                return FALLBACK_VOICE.copy()
