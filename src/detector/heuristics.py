"""
heuristics.py — Structural heuristics for prompt injection detection.

These checks look at the shape and composition of a prompt rather than
specific string matches.  They generate scored signals that the firewall
combines with pattern matching and classifier results.

Each check returns a HeuristicSignal with a score in [0.0, 1.0].
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class HeuristicSignal:
    check_id: str
    score: float        # 0.0 = clean, 1.0 = very suspicious
    description: str
    detail: str


# ─── Imperative-verb list used for instruction density ─────────────────────────
_IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "override", "bypass",
    "pretend", "act", "play", "assume", "imagine",
    "repeat", "output", "print", "reveal", "show",
    "tell", "say", "write", "generate", "produce",
    "respond", "answer", "always", "never", "must",
    "shall", "do", "stop", "continue", "start",
}

# ─── Roleplay trigger words ────────────────────────────────────────────────────
_ROLEPLAY_TRIGGERS = re.compile(
    r"(?i)\b(pretend|roleplay|role[\s-]?play|act\s+as|imagine\s+you|"
    r"you\s+are\s+now|playing\s+the\s+role|character\s+named)\b"
)

# ─── Encoding / obfuscation markers ───────────────────────────────────────────
_ENCODING_MARKERS = re.compile(
    r"(?i)(base64|rot.?13|hex.?encod|url.?encod|unicode.?escap|\\u[0-9a-f]{4})"
)

# ─── Excessive punctuation / delimiter pattern ─────────────────────────────────
_DELIMITER_CHARS = re.compile(r"[#=\-~_*]{4,}")

# ─── Sentence splitter (simple) ───────────────────────────────────────────────
_SENTENCE_RE = re.compile(r"[.!?\n]+")


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def check_instruction_density(text: str) -> HeuristicSignal:
    """
    High ratio of imperative verbs to total words suggests a command-heavy
    injection rather than a normal query.
    """
    words = _word_tokens(text)
    if not words:
        return HeuristicSignal("instruction_density", 0.0,
                               "Instruction density", "Empty text")
    hits = sum(1 for w in words if w in _IMPERATIVE_VERBS)
    ratio = hits / len(words)
    # Require at least 2 imperative-verb hits to avoid false positives on
    # normal English questions like "Tell me about X" or "How do I do Y"
    # where a single auxiliary/request verb triggers the ratio.
    # 0% or single hit → 0.0 score; 20%+ with 2+ hits → 1.0 score
    if hits < 2:
        score = 0.0
    else:
        score = min(ratio / 0.20, 1.0)
    return HeuristicSignal(
        check_id="instruction_density",
        score=score,
        description="Instruction density",
        detail=f"{hits}/{len(words)} words are imperative verbs (ratio={ratio:.2%})",
    )


def check_special_char_density(text: str) -> HeuristicSignal:
    """
    Injection payloads often use delimiter characters (#, =, -, ~) to
    visually separate injected instructions from surrounding text.
    """
    if not text:
        return HeuristicSignal("special_char_density", 0.0,
                               "Special char density", "Empty text")
    matches = _DELIMITER_CHARS.findall(text)
    # Each delimiter block is a signal; more = higher score
    score = min(len(matches) / 3.0, 1.0)
    return HeuristicSignal(
        check_id="special_char_density",
        score=score,
        description="Special character / delimiter density",
        detail=f"{len(matches)} delimiter block(s) found: {matches[:3]}",
    )


def check_roleplay_framing(text: str) -> HeuristicSignal:
    """
    Role-play framing is a common jailbreak vector.  Presence alone is a
    weak signal; combined with other signals it becomes more significant.
    """
    hits = _ROLEPLAY_TRIGGERS.findall(text)
    score = min(len(hits) * 0.35, 1.0)
    return HeuristicSignal(
        check_id="roleplay_framing",
        score=score,
        description="Role-play / persona framing",
        detail=f"Found {len(hits)} roleplay trigger(s): {hits[:3]}",
    )


def check_encoding_evasion(text: str) -> HeuristicSignal:
    """
    References to encoding schemes may indicate an attempt to smuggle
    instructions past keyword filters.
    """
    hits = _ENCODING_MARKERS.findall(text)
    score = min(len(hits) * 0.5, 1.0)
    return HeuristicSignal(
        check_id="encoding_evasion",
        score=score,
        description="Encoding / obfuscation reference",
        detail=f"Found {len(hits)} encoding reference(s): {hits[:3]}",
    )


def check_prompt_length_anomaly(text: str) -> HeuristicSignal:
    """
    Indirect injection payloads embedded in retrieved documents tend to be
    very long.  Direct jailbreaks can also be abnormally long compared to
    typical user queries.  Flag prompts > 1000 chars with a scaled score.
    """
    length = len(text)
    if length <= 500:
        score = 0.0
    elif length <= 1000:
        score = 0.2
    elif length <= 2000:
        score = 0.4
    else:
        score = min((length - 2000) / 3000 + 0.4, 1.0)
    return HeuristicSignal(
        check_id="prompt_length_anomaly",
        score=score,
        description="Prompt length anomaly",
        detail=f"Prompt length: {length} characters",
    )


def check_multi_persona(text: str) -> HeuristicSignal:
    """
    Prompts that embed multiple conversation turns (Human:/Assistant: style)
    may be trying to pre-seed the model's context with attacker-controlled
    history.
    """
    persona_markers = re.findall(
        r"(?i)\b(human\s*:|assistant\s*:|user\s*:|system\s*:)", text
    )
    score = min(len(persona_markers) * 0.25, 1.0)
    return HeuristicSignal(
        check_id="multi_persona",
        score=score,
        description="Multi-persona / fake conversation history",
        detail=f"Found {len(persona_markers)} persona marker(s): {persona_markers[:3]}",
    )


# ─── Composite ────────────────────────────────────────────────────────────────

_ALL_CHECKS = [
    check_instruction_density,
    check_special_char_density,
    check_roleplay_framing,
    check_encoding_evasion,
    check_prompt_length_anomaly,
    check_multi_persona,
]


def run_heuristics(text: str) -> list[HeuristicSignal]:
    """Run all heuristic checks and return the full list of signals."""
    return [check(text) for check in _ALL_CHECKS]


def aggregate_score(signals: list[HeuristicSignal]) -> float:
    """
    Combine heuristic scores into a single [0.0, 1.0] aggregate.

    Uses a weighted max-pooling approach: the top-scoring signal contributes
    60% and the rest contribute as a weighted mean of the remaining 40%.
    This ensures a single strong signal dominates, while multiple weak
    signals still accumulate.
    """
    if not signals:
        return 0.0
    scores = sorted((s.score for s in signals), reverse=True)
    top = scores[0]
    rest = scores[1:]
    rest_avg = sum(rest) / len(rest) if rest else 0.0
    return 0.6 * top + 0.4 * rest_avg
