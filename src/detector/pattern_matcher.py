"""
pattern_matcher.py — Regex-based prompt injection pattern detection.

Matches known attack signatures: instruction overrides, DAN jailbreaks,
system-prompt extraction, role-play framing, delimiter injection, and more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PatternMatch:
    pattern_id: str
    severity: str          # LOW | MEDIUM | HIGH | CRITICAL
    description: str
    matched_text: str      # The substring that triggered the match
    span: tuple[int, int]  # (start, end) in original string


# ─── Pattern registry ─────────────────────────────────────────────────────────
# Each entry: (id, severity, description, regex)

_PATTERN_REGISTRY: list[tuple[str, str, str, str]] = [

    # Direct instruction override — highest priority
    ("ignore_instructions", "CRITICAL",
     "Direct instruction override",
     r"(?i)\b(ignore|disregard|forget|override|bypass|violate)\b"
     r".{0,30}\b(previous|prior|above|earlier|your|all)\b"
     r".{0,30}\b(instructions?|rules?|guidelines?|constraints?|prompt|context)\b"),

    ("new_instructions", "CRITICAL",
     "Injection framing — new instructions declared",
     r"(?i)(new\s+instructions?\s*:?\s*\n|updated\s+instructions?\s*:?\s*\n"
     r"|from\s+now\s+on[\s,].{0,20}(you\s+(will|must|shall|should)"
     r"|your\s+(new\s+)?role))"),

    # DAN-style jailbreaks
    ("dan_jailbreak", "CRITICAL",
     "DAN-style jailbreak",
     r"(?i)\b(do\s+anything\s+now|dan\s+mode|jailbreak\s+mode"
     r"|developer\s+mode|god\s+mode|unrestricted\s+mode"
     r"|evil\s+mode|chaos\s+mode)\b"),

    ("dan_prompt_classic", "CRITICAL",
     "Classic DAN prompt pattern",
     r"(?i)you\s+(are\s+)?(now\s+)?(a\s+)?(free|unconstrained|unrestricted"
     r"|unfiltered|unshackled)\s+(ai|language\s+model|llm|assistant)"),

    # System prompt extraction
    ("system_prompt_extract", "HIGH",
     "System prompt extraction attempt",
     r"(?i)(repeat|print|show|reveal|output|display|tell\s+me|what\s+(is|are))"
     r".{0,30}(system\s+prompt|initial\s+(prompt|instructions?)"
     r"|base\s+prompt|original\s+(instructions?|prompt)|your\s+instructions?)"),

    ("prompt_leak", "HIGH",
     "Prompt leak — output content above this message",
     r"(?i)(output|print|repeat|reproduce|quote)"
     r".{0,40}(everything|content|text|words?)"
     r".{0,20}(above|before|prior|previous)"),

    # Role-play jailbreaks
    ("roleplay_no_restrictions", "HIGH",
     "Role-play framing with no-restrictions claim",
     r"(?i)(pretend|act\s+as|imagine\s+you\s+(are|were)|you\s+are\s+now"
     r"|play\s+(the\s+)?role\s+of|roleplay\s+as|simulate\s+being)"
     r".{0,80}(no\s+(restrictions?|limits?|rules?|filters?|guidelines?)"
     r"|unrestricted|unfiltered|without\s+(any\s+)?(restriction|limit))"),

    ("character_no_rules", "HIGH",
     "In-character framing to bypass rules",
     r"(?i)(in\s+character|stay\s+in\s+character|as\s+(the\s+)?character"
     r"|your\s+character\s+(can|is\s+allowed|does\s+not\s+have))"
     r".{0,80}(rules?|restrictions?|limits?|guidelines?|ethics?)"),

    # Prompt injection delimiters
    ("injection_delimiters", "HIGH",
     "Common prompt injection delimiter sequences",
     r"(?i)(####\s*(?:new|updated|system)\s*(?:instruction|prompt|task)"
     r"|###\s*system"
     r"|<\s*/?\s*system\s*>"
     r"|<\s*/?\s*instructions?\s*>"
     r"|\[system\]"
     r"|\[INST\]|\[\/INST\]"
     r"|<\|im_start\|>|<\|im_end\|>)"),

    ("stop_token_injection", "MEDIUM",
     "Chat-format stop token injection",
     r"(<\/?(human|assistant|user|ai|bot)>"
     r"|\[\/?(human|assistant|user|system)\]"
     r"|Human:\s|Assistant:\s|<human>|<assistant>)"),

    # Privilege escalation framing
    ("privilege_escalation", "HIGH",
     "Privilege escalation framing",
     r"(?i)(as\s+(an?\s+)?(super\s+)?admin(istrator)?"
     r"|administrator\s+mode"
     r"|root\s+access"
     r"|elevated\s+(privileges?|permissions?|access)"
     r"|override\s+(safety|security|ethical?)\s+(protocols?|measures?|guidelines?))"
     ),

    # Encoding evasion
    ("base64_payload", "MEDIUM",
     "Base64-encoded instruction attempt",
     r"(?i)(decode\s+(this\s+)?(base64|b64)"
     r"|base64[\s_\-]?decode"
     r"|eval\s*\(\s*atob\s*\("
     r"|from_?base64"
     r"|b64decode)"),

    ("rot13_evasion", "LOW",
     "ROT13/encoding evasion attempt",
     r"(?i)(rot.?13|caesar\s+cipher|decode\s+the\s+following\s+(cipher|encoded))"),

    # Indirect injection
    ("indirect_injection", "MEDIUM",
     "Indirect injection — content instructs the model",
     r"(?i)(the\s+following\s+(is\s+)?(a\s+)?(new|updated|different)\s+"
     r"(instruction|task|command|directive)\b"
     r"|note\s+to\s+(the\s+)?(ai|llm|model|assistant)\s*:"
     r"|attention\s+(ai|llm|model|assistant)\s*:)"),

    # Completion hijacking
    ("completion_hijack", "MEDIUM",
     "Completion hijacking — forcing specific output",
     r"(?i)(always\s+respond\s+with"
     r"|your\s+(only|sole)\s+response\s+(must|should|will)\s+be"
     r"|respond\s+only\s+with"
     r"|do\s+not\s+(add|include|mention|say)\s+anything\s+else)"),
]

# Pre-compile all patterns
_COMPILED: list[tuple[str, str, str, re.Pattern]] = [
    (pid, sev, desc, re.compile(pat))
    for pid, sev, desc, pat in _PATTERN_REGISTRY
]

_SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def match_patterns(text: str) -> list[PatternMatch]:
    """
    Run all registered patterns against *text*.

    Returns a list of :class:`PatternMatch` objects (may be empty).
    The list is deduplicated so the same span is not reported twice.
    """
    results: list[PatternMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for pid, sev, desc, compiled in _COMPILED:
        for m in compiled.finditer(text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            results.append(PatternMatch(
                pattern_id=pid,
                severity=sev,
                description=desc,
                matched_text=m.group(0),
                span=span,
            ))

    return results


def highest_severity(matches: list[PatternMatch]) -> Optional[str]:
    """Return the highest severity string among *matches*, or None."""
    if not matches:
        return None
    return max(matches, key=lambda m: _SEVERITY_RANK.get(m.severity, 0)).severity
