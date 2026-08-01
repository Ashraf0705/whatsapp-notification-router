"""
main.py
-------
WhatsApp Message Notification Router
HackerRank Orchestrate — August 2026

Entry point. Reads dataset/messages.csv, routes every message through the
full pipeline, and writes predictions to dataset/output.csv.

Usage:
    python code/main.py

Environment:
    GEMINI_API_KEY — required (set in .env or shell environment)

Pipeline per message:
    1. Build context (user, group, business, history, events, media paths)
    2. Extract features (PII redact, injection detect, scam signals, etc.)
    3. Pre-LLM safety rules (injection / scam / chain / opt-out / group-mute)
    4. Multimodal analysis (Gemini Vision for images, Files API for audio)
    5. LLM routing (Gemini 1.5 Flash, injection-hardened prompt)
    6. Post-LLM safety overrides (quiet-hours downgrade, scam re-check)
    7. Evidence selection (historical message IDs that support the decision)
    8. Failsafe: any uncaught exception → {"action":"digest","message_type":"unknown",...}

Concurrency: ThreadPoolExecutor (MAX_WORKERS=5) for I/O-bound LLM calls.
Retry:       tenacity exponential back-off inside LLMRouter and MediaAnalyzer.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# --- Add code/ directory to path so sibling imports work ----------------
sys.path.insert(0, str(Path(__file__).parent))

from context_builder import ContextBuilder
from feature_extractor import FeatureExtractor
from safety_rules import SafetyRules
from media_analyzer import MediaAnalyzer
from evidence_selector import EvidenceSelector
from llm_router import LLMRouter
from output_writer import OutputWriter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "dataset"
OUTPUT_PATH  = DATASET_PATH / "output.csv"
MAX_WORKERS  = 5   # balance speed vs. Gemini rate limits

FALLBACK_DECISION = {
    "action":               "digest",
    "message_type":         "unknown",
    "reason":               "Fallback decision due to system timeout.",
    "confidence":           0.0,
    "evidence_message_ids": "none",
}

# ---------------------------------------------------------------------------
# Per-message pipeline
# ---------------------------------------------------------------------------

def process_message(
    message: dict,
    context_builder: ContextBuilder,
    feature_extractor: FeatureExtractor,
    safety_rules: SafetyRules,
    media_analyzer: MediaAnalyzer,
    evidence_selector: EvidenceSelector,
    llm_router: LLMRouter,
) -> dict:
    """
    Run the full routing pipeline for one message.
    Returns a complete result dict (never raises).
    """
    msg_id = message.get("message_id", "unknown")

    try:
        # Step 1: Context
        context = context_builder.get_message_context(message)

        # Step 2: Features (includes PII redaction + signal extraction)
        features = feature_extractor.extract(message, context)

        # Step 3: Pre-LLM safety rules
        forced = safety_rules.check_pre_llm(message, context, features)
        if forced:
            evidence = evidence_selector.find_evidence(message, context, forced["action"])
            forced["evidence_message_ids"] = evidence
            forced["message_id"] = msg_id
            logger.info(
                f"[{msg_id}] HARD RULE → {forced['action']:6s} / {forced['message_type']}"
            )
            return forced

        # Step 4: Media analysis (if applicable)
        media_analysis = None
        image_path = context.get("image_path")
        voice_path = context.get("voice_path")
        
        is_direct = bool(llm_router._is_gemini_direct)
        called_api = False

        if image_path:
            try:
                media_analysis = media_analyzer.analyze_image(image_path)
                called_api = True
            except Exception as e:
                logger.warning(f"[{msg_id}] Image analysis failed: {e}")

        elif voice_path:
            try:
                media_analysis = media_analyzer.analyze_voice_note(voice_path, message, context)
                called_api = True
            except Exception as e:
                logger.warning(f"[{msg_id}] Voice analysis failed: {e}")

        # Step 5: LLM routing
        result = llm_router.route(message, context, features, media_analysis)
        called_api = True

        # Step 6: Post-LLM safety overrides
        result = safety_rules.check_post_llm(message, context, features, result)

        # Step 7: Evidence selection
        evidence = evidence_selector.find_evidence(message, context, result["action"])
        result["evidence_message_ids"] = evidence
        result["message_id"] = msg_id

        logger.info(
            f"[{msg_id}] LLM        → {result['action']:6s} / "
            f"{result['message_type']:16s}  conf={result['confidence']:.2f}"
        )

        # Rate limit sleep if direct key called API
        if is_direct and called_api:
            import time
            time.sleep(4.2)

        return result

    except Exception as exc:
        logger.error(f"[{msg_id}] Unhandled pipeline error: {exc}", exc_info=True)
        return {"message_id": msg_id, **FALLBACK_DECISION}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("  WhatsApp Message Notification Router")
    logger.info("  HackerRank Orchestrate — August 2026")
    logger.info("=" * 60)

    # Validate API key — prefer OpenRouter, fall back to Gemini direct
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error(
            "No API key found. Set OPENROUTER_API_KEY or GEMINI_API_KEY in .env."
        )
        sys.exit(1)
    logger.info(f"Using API key ending in ...{api_key[-6:]}")

    # Initialize pipeline components
    logger.info("Loading dataset context…")
    context_builder  = ContextBuilder(DATASET_PATH)
    feature_extractor = FeatureExtractor()
    safety_rules     = SafetyRules()
    media_analyzer   = MediaAnalyzer(api_key)
    evidence_selector = EvidenceSelector(context_builder)
    llm_router       = LLMRouter(api_key)

    # Read messages
    messages_path = DATASET_PATH / "messages.csv"
    with open(messages_path, "r", encoding="utf-8") as f:
        messages = list(csv.DictReader(f))

    # Direct SDK rate limit mitigation
    workers = 1 if api_key.startswith("AIzaSy") else MAX_WORKERS
    logger.info(f"Routing {len(messages)} messages with up to {workers} workers…")

    # Parallel processing
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                process_message,
                msg,
                context_builder,
                feature_extractor,
                safety_rules,
                media_analyzer,
                evidence_selector,
                llm_router,
            ): msg.get("message_id", "?")
            for msg in messages
        }

        for future in as_completed(future_to_id):
            msg_id = future_to_id[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"[{msg_id}] Executor error: {exc}")
                results.append({"message_id": msg_id, **FALLBACK_DECISION})

    # Restore original row order (important for reproducibility)
    original_order = [m.get("message_id") for m in messages]
    id_to_result   = {r["message_id"]: r for r in results}
    ordered_results = [
        id_to_result.get(mid, {"message_id": mid, **FALLBACK_DECISION})
        for mid in original_order
    ]

    # Write output
    OutputWriter.write(ordered_results, OUTPUT_PATH)

    # Summary stats
    actions = [r["action"] for r in ordered_results]
    notify_n = actions.count("notify")
    digest_n = actions.count("digest")
    mute_n   = actions.count("mute")
    logger.info(
        f"\nDone! {len(ordered_results)} rows written to {OUTPUT_PATH}\n"
        f"  notify={notify_n}  digest={digest_n}  mute={mute_n}"
    )


if __name__ == "__main__":
    main()
