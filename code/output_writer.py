"""
output_writer.py
----------------
Writes the final routing decisions to dataset/output.csv.

Ensures:
  • Exact column order required by the hackathon spec
  • evidence_message_ids is always a semicolon-separated string or "none"
  • confidence has max 2 decimal places
  • Row order matches original messages.csv order
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_TYPES   = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}


class OutputWriter:
    @staticmethod
    def write(results: list[dict], output_path: str | Path) -> None:
        output_path = Path(output_path)
        rows = []

        for r in results:
            # Normalize evidence_message_ids
            evidence = r.get("evidence_message_ids", "none")
            if not evidence or evidence == "none":
                evidence = "none"
            elif isinstance(evidence, list):
                evidence = ";".join(str(e) for e in evidence if e) or "none"
            else:
                evidence = str(evidence).strip()
                if not evidence:
                    evidence = "none"

            # Normalize action and message_type
            action = r.get("action", "digest")
            if action not in VALID_ACTIONS:
                action = "digest"

            mtype = r.get("message_type", "unknown")
            if mtype not in VALID_TYPES:
                mtype = "unknown"

            # Normalize confidence
            try:
                conf = round(float(r.get("confidence", 0.0)), 2)
                conf = max(0.0, min(1.0, conf))
            except Exception:
                conf = 0.0

            # Normalize reason
            reason = str(r.get("reason", "")).strip()
            if not reason:
                reason = "No reason provided."

            rows.append({
                "message_id":          r.get("message_id", ""),
                "action":              action,
                "message_type":        mtype,
                "reason":              reason,
                "confidence":          conf,
                "evidence_message_ids": evidence,
            })

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Wrote {len(rows)} rows to {output_path}")
