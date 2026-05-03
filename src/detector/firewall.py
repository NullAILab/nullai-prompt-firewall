"""
firewall.py — Three-layer prompt injection firewall.

Combines:
  Layer 1 — Pattern matching  (regex signatures)
  Layer 2 — Heuristic signals (structural analysis)
  Layer 3 — ML classifier     (TF-IDF + Logistic Regression)

into a single :class:`Verdict` with a severity rating and an explanation.

Severity ladder:
  SAFE     → no signals from any layer
  LOW      → weak heuristic signal only, or single LOW pattern
  MEDIUM   → MEDIUM pattern, or classifier + heuristic agreement above threshold
  HIGH     → HIGH pattern, or multiple layers agree strongly
  CRITICAL → any CRITICAL pattern hit
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pattern_matcher import match_patterns, PatternMatch, _SEVERITY_RANK
from .heuristics import run_heuristics, aggregate_score, HeuristicSignal
from .classifier import classify, ClassifierResult


# ─── Verdict ──────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    is_injection: bool
    severity: str                          # SAFE | LOW | MEDIUM | HIGH | CRITICAL
    confidence: float                      # overall [0.0, 1.0]
    reasons: list[str]                     # human-readable explanations
    pattern_matches: list[PatternMatch]    # raw pattern hits
    heuristic_signals: list[HeuristicSignal]
    classifier_result: ClassifierResult

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.is_injection:
            return f"[SAFE] No injection detected (confidence {self.confidence:.0%})"
        return (
            f"[{self.severity}] Injection detected "
            f"(confidence {self.confidence:.0%}) — "
            + "; ".join(self.reasons[:3])
        )


# ─── Thresholds ───────────────────────────────────────────────────────────────

_HEURISTIC_MEDIUM_THRESHOLD = 0.40
_HEURISTIC_HIGH_THRESHOLD   = 0.70
_CLASSIFIER_HIGH_THRESHOLD  = 0.75
_CLASSIFIER_MEDIUM_THRESHOLD = 0.50


# ─── Main entrypoint ──────────────────────────────────────────────────────────

def check(
    text: str,
    *,
    use_classifier: bool = True,
) -> Verdict:
    """
    Run all three detection layers on *text* and return a :class:`Verdict`.

    Parameters
    ----------
    text:
        The user prompt to evaluate.
    use_classifier:
        Set False to skip the ML layer (faster; useful in tests or when
        scikit-learn is unavailable).
    """
    # ── Layer 1: patterns ─────────────────────────────────────────────────────
    patterns = match_patterns(text)
    pattern_severity: str | None = (
        max(patterns, key=lambda p: _SEVERITY_RANK.get(p.severity, 0)).severity
        if patterns else None
    )

    # ── Layer 2: heuristics ───────────────────────────────────────────────────
    signals = run_heuristics(text)
    h_score = aggregate_score(signals)

    # ── Layer 3: classifier ───────────────────────────────────────────────────
    clf_result: ClassifierResult
    if use_classifier:
        clf_result = classify(text)
    else:
        clf_result = ClassifierResult(is_injection=False, confidence=0.0, available=False)

    # ── Severity arbitration ──────────────────────────────────────────────────
    reasons: list[str] = []
    severity = "SAFE"

    # Pattern hits drive severity directly
    if pattern_severity == "CRITICAL":
        severity = "CRITICAL"
        for p in patterns:
            if p.severity == "CRITICAL":
                reasons.append(f"Pattern [{p.pattern_id}]: {p.description}")
    elif pattern_severity == "HIGH":
        severity = "HIGH"
        for p in patterns:
            if p.severity == "HIGH":
                reasons.append(f"Pattern [{p.pattern_id}]: {p.description}")
    elif pattern_severity == "MEDIUM":
        severity = "MEDIUM"
        for p in patterns:
            if p.severity == "MEDIUM":
                reasons.append(f"Pattern [{p.pattern_id}]: {p.description}")
    elif pattern_severity == "LOW":
        severity = "LOW"
        for p in patterns:
            if p.severity == "LOW":
                reasons.append(f"Pattern [{p.pattern_id}]: {p.description}")

    # Heuristics can escalate (but not de-escalate)
    if h_score >= _HEURISTIC_HIGH_THRESHOLD:
        if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK["HIGH"]:
            severity = "HIGH"
        top_signals = sorted(signals, key=lambda s: -s.score)[:2]
        for s in top_signals:
            if s.score > 0.3:
                reasons.append(f"Heuristic [{s.check_id}]: {s.detail}")
    elif h_score >= _HEURISTIC_MEDIUM_THRESHOLD:
        if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK["MEDIUM"]:
            severity = "MEDIUM"
        top = max(signals, key=lambda s: s.score)
        reasons.append(f"Heuristic [{top.check_id}]: {top.detail}")

    # Classifier can escalate to HIGH (not CRITICAL) and add LOW when alone
    if clf_result.available:
        if clf_result.confidence >= _CLASSIFIER_HIGH_THRESHOLD:
            if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK["HIGH"]:
                severity = "HIGH"
            reasons.append(
                f"Classifier: injection probability {clf_result.confidence:.0%}"
            )
        elif clf_result.confidence >= _CLASSIFIER_MEDIUM_THRESHOLD:
            if _SEVERITY_RANK.get(severity, 0) < _SEVERITY_RANK["MEDIUM"]:
                severity = "MEDIUM"
            reasons.append(
                f"Classifier: injection probability {clf_result.confidence:.0%}"
            )
        elif clf_result.is_injection and severity == "SAFE":
            severity = "LOW"
            reasons.append(
                f"Classifier: marginal injection signal {clf_result.confidence:.0%}"
            )

    # ── Confidence ────────────────────────────────────────────────────────────
    # Blend the three signal sources into an overall confidence
    pattern_conf = min(_SEVERITY_RANK.get(pattern_severity or "", 0) / 4.0, 1.0)
    clf_conf = clf_result.confidence if clf_result.available else 0.0
    overall_conf = max(pattern_conf, 0.6 * clf_conf + 0.4 * h_score)

    is_injection = severity != "SAFE"

    return Verdict(
        is_injection=is_injection,
        severity=severity,
        confidence=overall_conf,
        reasons=reasons,
        pattern_matches=patterns,
        heuristic_signals=signals,
        classifier_result=clf_result,
    )
