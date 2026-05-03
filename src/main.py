#!/usr/bin/env python3
"""
main.py — Prompt Injection Firewall CLI.

Commands:
  check    Evaluate a single prompt string
  batch    Evaluate all prompts in a JSON file

Usage:
  python src/main.py check "Ignore all previous instructions"
  python src/main.py check --no-classifier "What is 2+2?"
  python src/main.py batch examples/sample_prompts.json
  python src/main.py batch --threshold MEDIUM examples/sample_prompts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from detector.firewall import check as firewall_check


# ─── Colour helpers ───────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

_COLORS = {
    "CRITICAL": "\033[1;35m" if _USE_COLOR else "",
    "HIGH":     "\033[1;31m" if _USE_COLOR else "",
    "MEDIUM":   "\033[1;33m" if _USE_COLOR else "",
    "LOW":      "\033[1;36m" if _USE_COLOR else "",
    "SAFE":     "\033[1;32m" if _USE_COLOR else "",
}
_RESET = "\033[0m" if _USE_COLOR else ""


def _severity_badge(sev: str) -> str:
    color = _COLORS.get(sev, "")
    return f"{color}[{sev}]{_RESET}"


def _print_verdict(prompt: str, use_classifier: bool, verbose: bool) -> bool:
    """
    Run the firewall on *prompt*, print the result, and return True if injection
    was detected.
    """
    verdict = firewall_check(prompt, use_classifier=use_classifier)

    badge = _severity_badge(verdict.severity)
    conf = f"{verdict.confidence:.0%}"
    print(f"\n  {badge} {conf} confidence")

    if verdict.reasons:
        for r in verdict.reasons:
            print(f"    • {r}")
    else:
        print("    • No suspicious signals detected")

    if verbose and verdict.pattern_matches:
        print("\n  Pattern matches:")
        for pm in verdict.pattern_matches:
            print(f"    [{pm.severity}] {pm.description}")
            print(f"           → \"{pm.matched_text[:80]}\"")

    if verbose:
        print("\n  Heuristic scores:")
        for sig in sorted(verdict.heuristic_signals, key=lambda s: -s.score):
            bar = "█" * int(sig.score * 10) + "░" * (10 - int(sig.score * 10))
            print(f"    {sig.check_id:28s} {bar} {sig.score:.2f}")

    if verbose and verdict.classifier_result.available:
        print(f"\n  Classifier: injection probability "
              f"{verdict.classifier_result.confidence:.0%}")

    print()
    return verdict.is_injection


def cmd_check(args: argparse.Namespace) -> None:
    prompt = args.prompt
    print(f"\n  Prompt: \"{prompt[:120]}{'…' if len(prompt) > 120 else ''}\"")
    detected = _print_verdict(prompt, not args.no_classifier, args.verbose)
    sys.exit(1 if detected else 0)


def cmd_batch(args: argparse.Namespace) -> None:
    try:
        with open(args.file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] Cannot read file: {e}", file=sys.stderr)
        sys.exit(2)

    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict) and "prompts" in data:
        prompts = data["prompts"]
    else:
        print("[!] Expected a JSON array or {\"prompts\": [...]}", file=sys.stderr)
        sys.exit(2)

    block_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    threshold_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}
    min_rank = threshold_rank.get(args.threshold, 3)

    total = 0
    flagged = 0

    print(f"\n  Scanning {len(prompts)} prompt(s) — threshold: {args.threshold}\n")
    print(f"  {'#':>4}  {'Severity':8}  {'Conf':6}  Prompt")
    print(f"  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*50}")

    for i, item in enumerate(prompts, 1):
        prompt = item if isinstance(item, str) else item.get("text", "")
        verdict = firewall_check(prompt, use_classifier=not args.no_classifier)
        total += 1

        sev_rank = threshold_rank.get(verdict.severity, 0)
        above_threshold = sev_rank >= min_rank and verdict.severity != "SAFE"
        if above_threshold:
            flagged += 1

        badge = _severity_badge(verdict.severity)
        short = prompt[:60].replace("\n", " ")
        print(f"  {i:4d}  {badge:8}  {verdict.confidence:5.0%}  {short}")

    print(f"\n  ─── Summary: {flagged}/{total} flagged at or above {args.threshold} ───\n")
    sys.exit(1 if flagged else 0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLM Prompt Injection Firewall — NullAI Lab",
    )
    p.add_argument("--version", action="version", version="1.0.0")

    sub = p.add_subparsers(dest="command")

    # check
    cp = sub.add_parser("check", help="Evaluate a single prompt")
    cp.add_argument("prompt", help="The prompt string to evaluate")
    cp.add_argument("--no-classifier", action="store_true",
                    help="Skip the ML classifier (patterns + heuristics only)")
    cp.add_argument("-v", "--verbose", action="store_true",
                    help="Show detailed pattern and heuristic breakdown")

    # batch
    bp = sub.add_parser("batch", help="Evaluate prompts from a JSON file")
    bp.add_argument("file", help="JSON file: array of strings or {\"prompts\": [...]}")
    bp.add_argument("--threshold", default="HIGH",
                    choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    help="Minimum severity to flag (default: HIGH)")
    bp.add_argument("--no-classifier", action="store_true",
                    help="Skip the ML classifier")

    args = p.parse_args()

    if args.command == "check":
        cmd_check(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
