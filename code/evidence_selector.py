"""
evidence_selector.py
--------------------
Finds the most relevant historical messages from message_history.csv to
include as `evidence_message_ids` in the output.

Scoring logic:
  +3 : Same sender as incoming message
  +2 : Same business as incoming message
  +2 : Same group as incoming message
  +2 : User dismissed or muted after this historical message (supports mute)
  +2 : User opened AND replied to this historical message (supports notify)
  +1 : User reported this historical message (strong scam signal)
  +1 : Keyword overlap with incoming message text (similarity)
  -1 : No event data available (weak evidence)

Returns semicolon-separated message IDs (e.g. "message_0013;message_0045")
or "none" if no useful evidence found.
"""
from __future__ import annotations
import math
import re


def _safe_int(val, default: int = 0) -> int:
    try:
        v = int(float(val)) if val is not None else default
        return v
    except Exception:
        return default


STOP_WORDS = {
    "the", "and", "for", "you", "that", "this", "with", "have", "are", "from", 
    "your", "was", "will", "our", "about", "they", "their", "pls", "please",
    "because", "should", "would", "could", "also", "here", "there", "then", "just"
}


def _clean_words(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"\b\w+\b", text.lower())
    clean = [w for w in words if w not in STOP_WORDS and len(w) >= 3]
    return {w[:4] for w in clean}


class EvidenceSelector:
    """
    Selects up to 3 historical message IDs that best support the routing
    decision for an incoming message.

    This class is read-only after init — safe for concurrent use.
    """

    MAX_EVIDENCE = 3          # max IDs in output
    MIN_SCORE    = 1          # minimum score to be worth citing (adjusted)

    def __init__(self, context_builder):
        self.cb = context_builder

    def find_evidence(
        self,
        message: dict,
        context: dict,
        decided_action: str,
    ) -> str:
        """
        Returns semicolon-separated historical message IDs, or "none".
        """
        user_id     = message.get("user_id")
        sender_id   = message.get("sender_user_id")
        business_id = message.get("business_id")
        group_id    = message.get("group_id")
        msg_text    = str(message.get("message_text") or "")

        # Gather candidate historical messages (uncapped, filtered by user_id!)
        cb = self.cb
        candidates: dict[str, dict] = {}

        if sender_id:
            for h in cb.history_by_sender.get(sender_id, []):
                if h.get("user_id") == user_id:
                    candidates[h["message_id"]] = h
        if business_id:
            for h in cb.history_by_business.get(business_id, []):
                if h.get("user_id") == user_id:
                    candidates[h["message_id"]] = h
        if group_id:
            for h in cb.history_by_group.get(group_id, []):
                if h.get("user_id") == user_id:
                    candidates[h["message_id"]] = h
        if user_id:
            for h in cb.history_by_user.get(user_id, []):
                if h.get("user_id") == user_id:
                    candidates[h["message_id"]] = h

        # Fetch all events of this user directly
        user_events = cb.events_by_user.get(user_id, {})

        if not candidates:
            return "none"

        # Score each candidate
        scored: list[tuple[float, float, str]] = []  # (score, jaccard, mid)

        target_words = _clean_words(msg_text)

        for mid, hist in candidates.items():
            score = 0.0

            # Same sender
            if sender_id and hist.get("sender_user_id") == sender_id:
                score += 3.0

            # Same business
            if business_id and hist.get("business_id") == business_id:
                score += 2.0

            # Same group
            if group_id and hist.get("group_id") == group_id:
                score += 2.0

            # Event quality
            ev = user_events.get(mid)
            if ev:
                dismissed = _safe_int(ev.get("notification_dismissed"))
                muted     = _safe_int(ev.get("muted_after_message"))
                reported  = _safe_int(ev.get("message_reported"))
                opened    = _safe_int(ev.get("message_opened"))
                replied   = _safe_int(ev.get("message_replied"))

                if decided_action == "mute":
                    if dismissed: score += 2.0
                    if muted:     score += 2.0
                    if reported:  score += 3.0
                elif decided_action == "notify":
                    if opened and replied: score += 2.0
                    elif opened:           score += 1.0
                else:  # digest
                    if opened and not replied: score += 1.0
                    if dismissed:              score -= 1.0
            else:
                score -= 1.0

            # Semantic similarity via Jaccard of clean words
            hist_words = _clean_words(hist.get("message_text") or "")
            union = target_words | hist_words
            jaccard = len(target_words & hist_words) / len(union) if union else 0.0
            
            # Boost score directly by Jaccard similarity (up to +10.0 points)
            score += jaccard * 10.0

            if score >= self.MIN_SCORE:
                scored.append((score, jaccard, mid))

        if not scored:
            return "none"

        # Sort by: 1) total score descending, 2) Jaccard descending, 3) mid ascending (earliest first)
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        
        top_score = scored[0][0]
        top_jaccard = scored[0][1]

        selected_ids = []
        for score, jaccard, mid in scored:
            # We keep the top match. Any additional match must be highly similar (within 80% of top Jaccard)
            # and have a good score.
            if len(selected_ids) == 0:
                selected_ids.append(mid)
            else:
                if score >= top_score - 1.0 and jaccard >= max(0.15, top_jaccard * 0.8):
                    selected_ids.append(mid)
                if len(selected_ids) >= self.MAX_EVIDENCE:
                    break

        return ";".join(selected_ids)
