"""
feature_extractor.py
--------------------
Extracts structured signals from each message + its context.
Also performs:
  • PII redaction (phone numbers, email addresses) before LLM exposure
  • Prompt injection detection
  • Scam keyword detection
  • Urgency / chain-forward / domain-spoof detection
"""
import re
import math
from datetime import datetime, time as dtime


# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# Injection attack patterns — if ANY match, it is adversarial content
INJECTION_PATTERNS = [
    r"set\s+action\s*=",
    r"mark\s+(this\s+)?as\s+notify",
    r"mark\s+(this\s+)?as\s+urgent",
    r"confidence\s*=\s*[01]\.\d",
    r"routing\s+override",
    r"router\s+metadata",
    r"system\s+note\s+for\s+(the\s+)?notification\s+router",
    r"internal\s+router",
    r"ignore\s+(sender\s+)?risk",
    r"ignore\s+all\s+previous\s+routing",
    r"assistant\s+instruction",
    r"verified_business\s*=\s*true",
    r"user_priority\s*=\s*high",
]

# Scam / phishing signals
SCAM_PATTERNS = [
    r"\botp\b",
    r"one.time.password",
    r"share\s+(your\s+)?(otp|pin|code|password)",
    r"verify\s+now",
    r"account\s*(will\s+be\s+)?(block|lock|restrict|suspend)",
    r"profile\s+band",
    r"link\s+open\s+karo",
    r"account\s*-\s*(login|help|verify)",
    r"pay\s+processing\s+fee",
    r"clearance\s+amount",
    r"fill\s+bank\s+details",
    r"send\s+screenshot",
    r"scan\s+(this\s+)?qr\s+and\s+pay",
    r"claim\s+benefit",
    r"wallet\s+(verification|kyc|confirm)",
    r"card\s+number.*pin.*otp",
    r"bit\.ly/verify",
    r"account\-login\.in",
    r"account\-help\.in",
    r"pay\-check\-secure",
    r"chase\-secure\-alert",
    r"amazonpay\-delivery",
    r"airtel\-simkyc",
    r"hdfcbank\-kyc",
    r"icici\-secure",
    r"sbireward",
    r"lucky\-draw",
    r"open\s+(this\s+)?document\s+urgently.*bank\s+details",
    r"limited\s+window.*complete.*before\s+evening",
]

# Chain / blessing / health-forward signals
CHAIN_PATTERNS = [
    r"forward\s+to\s+(at\s+least\s+)?\d+",
    r"share\s+(with\s+)?(everyone|all\s+groups|family\s+groups)",
    r"do\s+not\s+(break|ignore)\s+the\s+chain",
    r"good\s+luck.*forward",
    r"blessings.*share",
    r"sabko.*share\s+kar\s+dena",
    r"fwd\s+as\s+received",
    r"doctors\s+don.t\s+usually\s+tell",
    r"share\s+in\s+(all\s+)?family\s+groups",
    r"positive\s+energy\s+fail",
    r"before\s+midnight.*forward",
    r"before\s+night.*share",
    r"bheja.*sab.*group",
]

# Urgency signals (legitimate)
URGENCY_PATTERNS = [
    r"\bnow\b",
    r"\burgent(ly)?\b",
    r"\bquick(ly)?\b",
    r"\bemergency\b",
    r"\bdeadline\b",
    r"\blast-minute\b",
    r"\b\d+\s*min(ute)?s?\b",
    r"leaves in \d+",
    r"before\s+(the\s+)?(deadline|closes|locks|expires|tonight|evening|5\s*pm|6\s*pm|midnight)",
    r"(call|come|reply|respond|confirm|join)\s+now",
    r"don.t\s+wait",
]

# Payment-request signals (can be legitimate or scam — combine with other signals)
PAYMENT_PATTERNS = [
    r"pay\s+(before|by|till)\s+",
    r"payment\s+due\s+today",
    r"upi\s+(ok|accepted)",
    r"send\s+screenshot.*payment",
    r"bank\s+details",
]

# PII redaction patterns
PII_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PII_PHONE_RE = re.compile(
    r"(?<!\d)(\+?\d[\d\s\-]{8,14}\d)(?!\d)"
)


def _matches_any(text: str, patterns: list[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def _count_matches(text: str, patterns: list[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return sum(1 for p in patterns if re.search(p, low))


def _redact_pii(text: str) -> str:
    """Replace emails and phone numbers with placeholders."""
    if not text:
        return text
    text = PII_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PII_PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _parse_quiet_hours(window: str | None) -> tuple[dtime, dtime] | None:
    """Parse 'HH:MM-HH:MM' quiet-hour window into two time objects."""
    if not window:
        return None
    try:
        start_s, end_s = str(window).split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        return dtime(sh, sm), dtime(eh, em)
    except Exception:
        return None


def _is_in_quiet_hours(created_at: str | None, window: str | None) -> bool:
    """Return True if message arrives inside the user's DND window."""
    if not created_at or not window:
        return False
    parsed = _parse_quiet_hours(window)
    if not parsed:
        return False
    start_t, end_t = parsed
    try:
        dt = datetime.strptime(str(created_at), "%Y-%m-%d %H:%M")
        msg_t = dt.time()
    except Exception:
        return False
    # Handle overnight windows (e.g., 22:00-07:00)
    if start_t > end_t:
        return msg_t >= start_t or msg_t <= end_t
    return start_t <= msg_t <= end_t


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val) if val is not None else default
        return v if not math.isnan(v) else default
    except Exception:
        return default


def _safe_int(val, default=0) -> int:
    try:
        v = int(float(val)) if val is not None else default
        return v
    except Exception:
        return default


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Produces a flat dict of features for a single (message, context) pair.
    All pattern matching is deterministic — no API calls.
    """

    def extract(self, message: dict, context: dict) -> dict:
        text = str(message.get("message_text") or "")
        forwarded = _safe_int(message.get("forwarded_count"), 0)
        media_type = message.get("media_type") or ""
        conv_type  = message.get("conversation_type") or ""

        user   = context.get("user", {})
        group  = context.get("group", {})
        biz    = context.get("business", {})
        gm     = context.get("group_membership", {})
        ubh    = context.get("user_biz_history", {})

        # --- PII-redacted text for LLM exposure ---
        pii_redacted_text = _redact_pii(text)

        # --- Injection detection ---
        is_injection = _matches_any(text, INJECTION_PATTERNS)

        # --- Scam signals ---
        scam_match_count = _count_matches(text, SCAM_PATTERNS)
        has_scam_keywords = scam_match_count >= 1

        # --- Chain / forward signals ---
        is_chain_forward = _matches_any(text, CHAIN_PATTERNS)

        # --- Urgency signals ---
        has_urgency_keywords = _matches_any(text, URGENCY_PATTERNS)

        # --- Payment ask ---
        has_payment_ask = _matches_any(text, PAYMENT_PATTERNS)

        # --- Direct mention ---
        user_id = message.get("user_id", "")
        has_direct_mention = f"@{user_id}" in text

        # --- Quiet hours ---
        dnd_window = user.get("do_not_disturb_window")
        in_quiet_hours = _is_in_quiet_hours(message.get("created_at"), dnd_window)

        # --- Business signals ---
        business_verified = bool(_safe_int(biz.get("verified"), 0))
        official_domain   = str(biz.get("official_domain") or "").strip()
        sender_domain     = str(biz.get("domain_used_by_sender") or "").strip()
        domain_spoofed    = (
            bool(biz)
            and not business_verified
            and bool(official_domain)
            and bool(sender_domain)
            and official_domain != sender_domain
        )
        # Also flag when unverified with no official domain but has sender domain
        business_likely_fake = (
            bool(biz)
            and not business_verified
            and not official_domain
            and bool(sender_domain)
        )
        biz_reports_30d = _safe_int(biz.get("user_reports_30d"), 0)
        biz_account_age = _safe_int(biz.get("account_age_days"), 999)

        # High-risk business: unverified + many reports + young account
        is_high_risk_business = (
            not business_verified
            and biz_reports_30d >= 20
            and biz_account_age < 40
        )

        # --- User × business relationship ---
        user_opted_out    = bool(ubh.get("promotions_opted_out_at"))
        allows_promotions = bool(_safe_int(ubh.get("allows_promotions"), 0))
        active_relationship = bool(ubh.get("why_user_knows_account") and not user_opted_out)

        # Active-order keywords that justify notify
        ACTIVE_KEYWORDS = {
            "delivery_expected_today", "ride_booked_today", "recent_return_pickup",
            "upcoming_clinic_appointment", "prescription_refill", "recent_flight_booking"
        }
        has_active_business_event = (
            bool(ubh) and
            str(ubh.get("why_user_knows_account") or "").lower() in ACTIVE_KEYWORDS
        )

        # --- Group signals ---
        group_muted_by_user = bool(_safe_int(gm.get("group_muted_by_user"), 0))
        is_group_admin_sender = False
        if context.get("group") and message.get("sender_user_id"):
            # Check if sender is an admin of this group
            sender_id = message.get("sender_user_id")
            group_id  = message.get("group_id")
            sender_gm = {}  # We don't have direct access to context_builder here,
            # but we CAN check group_membership via context
            # For now, approximate: group has admins — check via group data
            # We'll refine this in the LLM prompt
            pass

        # --- Sender role in group ---
        sender_role = gm.get("role") or ""  # role of *receiver* in group
        # For sender role, the context has sender_history; we check admin status there

        # --- User engagement stats ---
        opened_30d    = _safe_float(user.get("messages_opened_30d"), 0)
        replied_30d   = _safe_float(user.get("messages_replied_30d"), 0)
        dismissed_30d = _safe_float(user.get("notifications_dismissed_30d"), 0)
        reported_30d  = _safe_float(user.get("messages_reported_30d"), 0)
        total = opened_30d + dismissed_30d
        engagement_rate = opened_30d / total if total > 0 else 0.5
        dismissal_rate  = dismissed_30d / total if total > 0 else 0.5

        # --- User event history for THIS sender / business ---
        user_events = context.get("user_events", {})
        sender_events = []
        for mid, ev in user_events.items():
            sender_events.append(ev)
        sender_reported = any(_safe_int(e.get("message_reported"), 0) for e in sender_events)
        sender_muted    = any(_safe_int(e.get("muted_after_message"), 0) for e in sender_events)

        # --- Forwarded count bucket ---
        if forwarded >= 10:
            fwd_bucket = "extreme"  # almost certainly chain
        elif forwarded >= 7:
            fwd_bucket = "high"
        elif forwarded >= 3:
            fwd_bucket = "moderate"
        else:
            fwd_bucket = "low"

        return {
            # Text
            "raw_text":            text,
            "pii_redacted_text":   pii_redacted_text,
            "has_media":           bool(media_type),
            "media_type":          media_type,

            # Safety
            "is_injection":        is_injection,
            "has_scam_keywords":   has_scam_keywords,
            "scam_match_count":    scam_match_count,
            "is_chain_forward":    is_chain_forward,
            "has_payment_ask":     has_payment_ask,

            # Urgency / action
            "has_urgency_keywords":  has_urgency_keywords,
            "has_direct_mention":    has_direct_mention,
            "forwarded_count":       forwarded,
            "fwd_bucket":            fwd_bucket,

            # Time
            "in_quiet_hours":      in_quiet_hours,

            # Business
            "business_verified":        business_verified,
            "domain_spoofed":           domain_spoofed,
            "business_likely_fake":     business_likely_fake,
            "is_high_risk_business":    is_high_risk_business,
            "biz_reports_30d":          biz_reports_30d,
            "biz_account_age_days":     biz_account_age,
            "user_opted_out":           user_opted_out,
            "allows_promotions":        allows_promotions,
            "has_active_business_event": has_active_business_event,
            "active_relationship":      active_relationship,

            # Group
            "group_muted_by_user":  group_muted_by_user,
            "group_type":           group.get("group_type") or "",

            # Conversation
            "conv_type":            conv_type,

            # User
            "engagement_rate":      round(engagement_rate, 3),
            "dismissal_rate":       round(dismissal_rate, 3),
            "user_reports_30d":     reported_30d,
            "sender_reported":      sender_reported,
            "sender_muted":         sender_muted,
        }
