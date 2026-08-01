"""
safety_rules.py
---------------
Hard-coded routing overrides applied BEFORE and AFTER the LLM.

Pre-LLM rules prevent the LLM from ever seeing adversarial content as
instructions and short-circuit obviously scam or chain messages.

Post-LLM rules handle quiet-hours downgrade and group-mute downgrade.

A "forced decision" is a complete routing result dict returned directly,
bypassing or overriding the LLM.
"""
from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ACTIONS       = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
}


def _safe_int(val, default: int = 0) -> int:
    try:
        v = int(float(val)) if val is not None else default
        return v
    except Exception:
        return default


def _forced(action, mtype, reason, confidence):
    return {
        "action": action,
        "message_type": mtype,
        "reason": reason,
        "confidence": round(confidence, 2),
    }


# ---------------------------------------------------------------------------
# SafetyRules
# ---------------------------------------------------------------------------

class SafetyRules:
    """
    Two public methods:
      check_pre_llm(message, context, features) → dict | None
        Returns a forced decision OR None (let LLM decide).

      check_post_llm(message, context, features, llm_result) → dict
        Returns possibly-modified llm_result.
    """

    # --- Pre-LLM --------------------------------------------------------

    def check_pre_llm(
        self,
        message: dict,
        context: dict,
        features: dict,
    ) -> dict | None:
        """
        Returns a hard-forced routing dict if any pre-LLM rule fires,
        otherwise returns None (LLM should decide).
        """

        # ── Rule 1: Prompt injection detected ──────────────────────────
        if features["is_injection"]:
            return _forced(
                "mute", "scam",
                "The message body contains adversarial instructions attempting to manipulate "
                "the notification router. This is a scam or social-engineering attempt.",
                0.97,
            )

        # ── Rule 2: Domain-spoofed business + scam keywords ───────────
        if (
            features["domain_spoofed"] or features["business_likely_fake"]
            or features["is_high_risk_business"]
        ) and (
            features["has_scam_keywords"] or features["has_payment_ask"]
        ):
            biz = context.get("business", {})
            name = biz.get("display_name") or "Unknown business"
            return _forced(
                "mute", "scam",
                f"Sender claims to be {name} but uses a mismatched or unverified domain, "
                "and the message asks for OTP, payment, or account verification. "
                "High-confidence phishing attempt.",
                0.95,
            )

        # ── Rule 3: Unverified high-report business with payment ask ──
        if (
            not features["business_verified"]
            and features["biz_reports_30d"] >= 30
            and features["has_payment_ask"]
        ):
            return _forced(
                "mute", "scam",
                "Unverified business account with a high user-report count is requesting payment "
                "or account verification. Treated as phishing.",
                0.93,
            )

        # ── Rule 4: Extreme forwarding + chain keywords ────────────────
        if features["is_chain_forward"] and features["fwd_bucket"] in ("extreme", "high"):
            return _forced(
                "mute", "forward",
                "Highly forwarded message with blessing, chain, or health-forward language. "
                "Pattern matches known viral chain-message spam.",
                0.90,
            )

        # ── Rule 5: Business opted-out + any message ──────────────────
        if features["user_opted_out"] and features["conv_type"] == "business":
            biz = context.get("business", {})
            name = biz.get("display_name") or "this sender"
            return _forced(
                "mute", "promotion",
                f"User has explicitly opted out of messages from {name}. "
                "Routing as muted per opt-out preference.",
                0.92,
            )

        # No pre-LLM rule fired
        return None

    # --- Post-LLM -------------------------------------------------------

    def check_post_llm(
        self,
        message: dict,
        context: dict,
        features: dict,
        result: dict,
    ) -> dict:
        """
        Apply post-LLM corrections and overrides.
        """
        action = result.get("action", "digest")
        mtype  = result.get("message_type", "unknown")

        # ── Override 1: Mute any scam/spam classified messages ─────────
        if mtype in ("scam", "spam") and action != "mute":
            result["action"] = "mute"
            result["reason"] = result.get("reason", "") + f" Overridden to mute because the message type is {mtype}."
            result["confidence"] = max(result.get("confidence", 0.7), 0.90)

        # ── Override 1b: Force greetings and promotions to digest ──────
        if mtype in ("greeting", "promotion") and result["action"] == "notify":
            result["action"] = "digest"
            result["reason"] = result.get("reason", "") + f" Downgraded to digest because the message type is {mtype}."
            result["confidence"] = max(result.get("confidence", 0.6), 0.85)

        # ── Override 2: Sender previously reported or muted by this user 
        if (features["sender_reported"] or features["sender_muted"]) and result["action"] != "mute":
            result["action"] = "mute"
            result["reason"] = result.get("reason", "") + " Muted because the user has previously reported or muted this sender."
            result["confidence"] = max(result.get("confidence", 0.7), 0.92)

        # ── Override 3: Group muted + no direct mention + not urgent ───
        if (
            features["group_muted_by_user"]
            and not features["has_direct_mention"]
            and not features["has_urgency_keywords"]
            and result["action"] != "mute"
        ):
            group = context.get("group", {})
            name  = group.get("group_name") or "this group"
            result["action"] = "mute"
            result["reason"] = result.get("reason", "") + f" Muted because the user has muted group {name}."
            result["confidence"] = max(result.get("confidence", 0.7), 0.88)

        # ── Override 3.4: Direct mention forces notify ────────────────
        if (
            features.get("has_direct_mention")
            and result["message_type"] not in ("scam", "spam", "promotion")
            and result["action"] != "notify"
        ):
            if not features.get("user_opted_out"):
                result["action"] = "notify"
                result["reason"] = result.get("reason", "") + " Forced to notify because the user was directly mentioned (@user)."
                result["confidence"] = max(result.get("confidence", 0.7), 0.90)

        # ── Override 3.5: Cold DMs from unknown senders ────────────────
        action = result.get("action", "digest")
        conv_type = message.get("conversation_type") or ""
        if (
            action == "notify"
            and conv_type == "personal"
            and not context.get("sender_history")
            and features.get("engagement_rate", 0.0) == 0.0
            and not features.get("has_direct_mention")
            and not features.get("has_urgency_keywords")
        ):
            result["action"] = "digest"
            result["reason"] = result.get("reason", "") + " Downgraded to digest because sender is unknown (cold DM)."
            result["confidence"] = max(result.get("confidence", 0.7), 0.85)

        # ── Override 4: Quiet hours ────────────────────────────────────
        # Re-check action since it might have been changed to mute/digest above
        action = result.get("action", "digest")
        if features["in_quiet_hours"] and action == "notify":
            mtype = result.get("message_type", "unknown")
            if mtype not in ("urgent", "scam", "payment"):
                result["action"] = "digest"
                result["reason"] = result.get("reason", "") + " Downgraded to digest because it arrived during DND window."
                result["confidence"] = round(max(0.0, float(result.get("confidence", 0.75)) - 0.05), 2)

        # ── Override 5: LLM missed scam — override if signals are strong 
        action = result.get("action", "digest")
        if (
            action != "mute"
            and features["scam_match_count"] >= 3
            and (features["domain_spoofed"] or features["business_likely_fake"]
                 or not features["business_verified"])
        ):
            result["action"] = "mute"
            result["message_type"] = "scam"
            result["reason"] = "Multiple scam signals detected (suspicious domain, OTP/payment ask). Overriding to mute."
            result["confidence"] = 0.91

        # ── Clamp and Validate ─────────────────────────────────────────
        if result.get("action") not in VALID_ACTIONS:
            result["action"] = "digest"
        if result.get("message_type") not in VALID_MESSAGE_TYPES:
            result["message_type"] = "unknown"

        conf = float(result.get("confidence", 0.7))
        result["confidence"] = round(max(0.0, min(1.0, conf)), 2)

        return result
