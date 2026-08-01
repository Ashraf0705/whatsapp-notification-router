"""
context_builder.py
------------------
Loads every dataset CSV into efficient in-memory structures and provides
a single `get_message_context(message)` call that returns all relevant
contextual data needed for routing a message.
"""
import math
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val):
    """Convert pandas NaN / None to Python None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _row_dict(row) -> dict:
    """Pandas Series → clean Python dict (NaN → None, numbers stay numbers)."""
    return {k: _safe(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Loads all CSV files once at startup and exposes fast lookups.
    All public attributes are read-only after __init__ — safe for
    concurrent access from ThreadPoolExecutor.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self._load_all()

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path / name)

    def _load_all(self):
        # --- users ---
        users_df = self._read("users.csv")
        self.users: dict[str, dict] = {
            r["user_id"]: _row_dict(r) for _, r in users_df.iterrows()
        }

        # --- groups ---
        groups_df = self._read("groups.csv")
        self.groups: dict[str, dict] = {
            r["group_id"]: _row_dict(r) for _, r in groups_df.iterrows()
        }

        # --- business accounts ---
        biz_df = self._read("business_accounts.csv")
        self.business_accounts: dict[str, dict] = {
            r["business_id"]: _row_dict(r) for _, r in biz_df.iterrows()
        }

        # --- group members: (group_id, user_id) → data ---
        gm_df = self._read("group_members.csv")
        self.group_members: dict[tuple, dict] = {}
        for _, r in gm_df.iterrows():
            d = _row_dict(r)
            self.group_members[(d["group_id"], d["user_id"])] = d

        # --- user × business history: (user_id, business_id) → data ---
        ubh_df = self._read("user_business_history.csv")
        self.user_biz_history: dict[tuple, dict] = {}
        for _, r in ubh_df.iterrows():
            d = _row_dict(r)
            self.user_biz_history[(d["user_id"], d["business_id"])] = d

        # --- message history (past messages) ---
        hist_df = self._read("message_history.csv")
        self.all_history: list[dict] = [_row_dict(r) for _, r in hist_df.iterrows()]

        # Build indexes for fast lookup
        self.history_by_id: dict[str, dict] = {}
        self.history_by_user: dict[str, list[dict]] = {}
        self.history_by_sender: dict[str, list[dict]] = {}
        self.history_by_business: dict[str, list[dict]] = {}
        self.history_by_group: dict[str, list[dict]] = {}

        for msg in self.all_history:
            mid = msg.get("message_id")
            uid = msg.get("user_id")
            sid = msg.get("sender_user_id")
            bid = msg.get("business_id")
            gid = msg.get("group_id")
            if mid:
                self.history_by_id[mid] = msg
            if uid:
                self.history_by_user.setdefault(uid, []).append(msg)
            if sid:
                self.history_by_sender.setdefault(sid, []).append(msg)
            if bid:
                self.history_by_business.setdefault(bid, []).append(msg)
            if gid:
                self.history_by_group.setdefault(gid, []).append(msg)

        # --- message events ---
        ev_df = self._read("message_events.csv")
        self.events_by_user: dict[str, dict[str, dict]] = {}
        for _, r in ev_df.iterrows():
            d = _row_dict(r)
            uid = d["user_id"]
            mid = d["message_id"]
            self.events_by_user.setdefault(uid, {})[mid] = d

        # --- media mappings ---
        img_df = self._read("images.csv")
        self.images: dict[str, str] = {
            r["image_id"]: r["file_path"] for _, r in img_df.iterrows()
        }

        vn_df = self._read("voice_notes.csv")
        self.voice_notes: dict[str, str] = {
            r["voice_note_id"]: r["file_path"] for _, r in vn_df.iterrows()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_message_context(self, message: dict) -> dict:
        """
        Return a dict containing all relevant contextual data for a message.
        Caps history slices to keep prompt sizes manageable.
        """
        user_id     = message.get("user_id")
        group_id    = _safe(message.get("group_id"))
        business_id = _safe(message.get("business_id"))
        sender_id   = _safe(message.get("sender_user_id"))
        media_id    = _safe(message.get("media_id"))
        media_type  = _safe(message.get("media_type"))

        # Filter user events to only match this sender or business in DMs
        all_user_events = self.events_by_user.get(user_id, {})
        filtered_events = {}
        for mid, ev in all_user_events.items():
            hist_msg = self.history_by_id.get(mid, {})
            is_sender_match = (
                sender_id 
                and hist_msg.get("sender_user_id") == sender_id 
                and hist_msg.get("conversation_type") == "personal"
            )
            is_biz_match = (
                business_id 
                and hist_msg.get("business_id") == business_id 
                and hist_msg.get("conversation_type") == "business"
            )
            if is_sender_match or is_biz_match:
                filtered_events[mid] = ev

        ctx: dict = {
            # Entity data
            "user":              self.users.get(user_id, {}),
            "group":             self.groups.get(group_id, {}) if group_id else {},
            "business":          self.business_accounts.get(business_id, {}) if business_id else {},
            "group_membership":  self.group_members.get((group_id, user_id), {}) if group_id else {},
            "user_biz_history":  self.user_biz_history.get((user_id, business_id), {}) if business_id else {},

            # History slices (capped to keep token usage reasonable)
            "sender_history":    self.history_by_sender.get(sender_id, [])[:8]   if sender_id   else [],
            "business_history":  self.history_by_business.get(business_id, [])[:6] if business_id else [],
            "group_history":     self.history_by_group.get(group_id, [])[:4]     if group_id    else [],
            "user_history":      self.history_by_user.get(user_id, [])[:12],

            # User event history
            "user_events":       filtered_events,

            # Media
            "image_path":        None,
            "voice_path":        None,
        }

        if media_id and media_type == "image":
            rel_path = self.images.get(media_id)
            if rel_path:
                ctx["image_path"] = str(self.dataset_path / rel_path)

        elif media_id and media_type == "voice":
            rel_path = self.voice_notes.get(media_id)
            if rel_path:
                ctx["voice_path"] = str(self.dataset_path / rel_path)

        return ctx
