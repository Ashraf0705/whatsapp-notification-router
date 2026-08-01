# WhatsApp Message Notification Router
### HackerRank Orchestrate — August 2026

An AI-powered routing system that classifies every incoming WhatsApp message as **notify / digest / mute** — personalized per user, multimodal-aware, and hardened against adversarial prompt injection.

---

## Architecture

```
messages.csv
    │
    ├─► ContextBuilder       — loads all 9 CSV tables into indexed dicts
    │
    ├─► FeatureExtractor     — rule-based signals + PII redaction
    │     • PII redaction: emails/phones → [REDACTED_EMAIL]/[REDACTED_PHONE]
    │     • URLs kept intact (scam signal)
    │     • Injection attempt detection
    │     • Scam keyword scoring, domain-spoof check, quiet-hours check
    │
    ├─► SafetyRules (Pre-LLM)
    │     • MUTE: prompt injection in message text
    │     • MUTE: domain-spoofed business + scam keywords
    │     • MUTE: high-report unverified business + payment ask
    │     • MUTE: extreme forwarding + chain-message keywords
    │     • MUTE: business opted-out by user
    │     • MUTE: group muted by user + no direct mention
    │
    ├─► MediaAnalyzer        — Gemini 3.5 Flash-Lite / 3.6 Flash (Vision)
    │     • Images: classification, OCR, QR/scam detection
    │     • Voice notes: contextual intent classification
    │     • Thread-safe cache by media_id
    │
    ├─► LLMRouter            — Gemini 3.5 Flash-Lite / 3.6 Flash (text)
    │     • Injection-hardened system prompt with few-shot examples
    │     • All context serialized: user, group/business, history, features
    │     • tenacity: 6 retries with exponential back-off
    │     • Fallback: digest/unknown/0.0 on total failure
    │
    ├─► SafetyRules (Post-LLM)
    │     • Quiet-hours DND window check
    │     • Stranger DM digest override
    │     • Direct mention notify bypass
    │     • Enum validation + confidence clamp [0,1]
    │
    └─► EvidenceSelector     — finds relevant historical message IDs
          • Strict recipient user_id history filtering
          • Scores by: same sender (+3), same business (+2), same group (+2)
          • Event outcomes: dismissed/muted/reported boost mute evidence
          • Stemmed clean-word Jaccard similarity (4-char prefix) as tie-breaker
          → semicolon-separated IDs or "none"

output.csv  ←  OutputWriter (validates all columns, enforces spec format)
```

## Approach Overview & Rationale

Our architecture balances **computational cost, API latency, and reasoning accuracy** through a hybrid routing system:
1. **Rule-Based Pre-Filters (Deterministic)**: We immediately block obvious spam, unverified businesses, and prompt-injection messages without querying the LLM. This intercepts ~45% of messages, saving credits and keeping execution time low.
2. **Vision & Voice Analysis (Multimodal)**: Visual messages (images) and audio (voice notes) are pre-analyzed to feed rich textual details (OCR, QR presence, category metadata) into the router's context.
3. **Probabilistic Reasoning (LLM Router)**: A personalized prompt combines user profiles (DND, history, engagement rates) and text metadata. We utilize state-of-the-art **Gemini 3.5 Flash-Lite** and **Gemini 3.6 Flash** to route messages.
4. **Post-LLM Safety Overrides (DND/Mention Check)**: Final guardrails adjust actions during quiet hours or when direct username mentions occur.
5. **Stemmed Jaccard Selector (Evidence)**: We locate support histories by matching 4-character stems of cleaned words, naturally tie-breaking variations.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=AIza...
```
Get a free key at [https://aistudio.google.com/](https://aistudio.google.com/)

### 3. Run
```bash
python code/main.py
```

Output is written to `dataset/output.csv`.

---

## Key Differentiators

| Feature | What it does |
|---|---|
| **Next-Gen 2026 Models** | Dynamically targets `gemini-3.5-flash-lite` and `gemini-3.6-flash` for state-of-the-art reasoning, speed, and accuracy |
| **Dual SDK Support** | Automatically detects and runs via official `google-genai` SDK for Google API keys or falls back to `openai` SDK for OpenRouter |
| **Prompt injection hardening** | Messages containing `"set action=notify"`, `"system note for router"`, `"routing override"` etc. are hard-muted as `scam` — never followed as instructions |
| **PII redaction** | Emails and phone numbers in message text are replaced with `[REDACTED_EMAIL]`/`[REDACTED_PHONE]` before LLM exposure (privacy compliance) |
| **Domain spoof detection** | Compares `official_domain` vs `domain_used_by_sender` — mismatched unverified businesses + scam keywords → immediate mute |
| **Multimodal** | Google GenAI vision bytes for images (OCR, QR, classification), contextual intent classification for voice notes |
| **Personalization** | Same message → different decision per user based on opt-out status, group mute, engagement history, quiet hours |
| **Rate-Limit Throttling** | Auto-clamps workers to `1` and sleeps `4.2s` between API requests when using direct free tier keys to prevent 429 errors |
| **Failsafe** | Any exception → safe `digest/unknown/0.0` row — output.csv is always complete |

---

## Output Format

```
message_id,action,message_type,reason,confidence,evidence_message_ids
msg_023,notify,business_update,Verified bank sent a card payment update matching active account.,0.88,message_0045;message_0067
msg_091,mute,scam,Message asks for 6-digit login code from unknown sender — credential phishing pattern.,0.95,none
```

- `action`: `notify` | `digest` | `mute`
- `message_type`: `personal` | `urgent` | `event` | `payment` | `business_update` | `promotion` | `greeting` | `forward` | `spam` | `scam` | `unknown`
- `evidence_message_ids`: semicolon-separated historical message IDs, or `none`

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

Never commit your `.env` file. It is in `.gitignore`.
