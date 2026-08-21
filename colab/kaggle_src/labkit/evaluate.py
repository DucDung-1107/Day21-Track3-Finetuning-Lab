"""Four-group evaluation and the regression gate.

Deck §17: perplexity alone is not evidence. A fine-tune is only a win if it beats a
*well-prompted base model* on the target task **without** quietly losing general
capability. So every run is scored on four groups:

    1. target      — does it do the job you trained it for?
    2. regression  — did it forget everything else? (deck §14.3)
    3. format      — does it obey the output contract?
    4. latency     — what did the win cost at inference time?

and compared against three frozen baselines:

    (a) base + naive prompt
    (b) base + a genuinely optimized prompt   <- the one that matters
    (c) the fine-tune

The graded question is not "did perplexity drop". It is: **did you beat (b), and
would you have noticed if you hadn't.**

Everything here is a pure function over already-generated strings, so the whole
harness is testable on a laptop with no GPU.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict

# --- Vietnamese-aware normalization -----------------------------------------
# Grader footgun this course has hit before: matching an *unaccented* keyword against
# accented Vietnamese output (or vice versa) silently passes or fails everything. Pick
# one normalization and apply it to BOTH sides, always.


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", without).replace("đ", "d").replace("Đ", "D")


def normalize(text: str, fold_accents: bool = True) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return strip_accents(text) if fold_accents else text


# --- scorers ----------------------------------------------------------------

def exact_match(pred: str, ref: str, fold_accents: bool = True) -> float:
    return float(normalize(pred, fold_accents) == normalize(ref, fold_accents))


def keyword_recall(pred: str, keywords: list[str], fold_accents: bool = True) -> float:
    """Fraction of required keywords present. BOTH sides normalized identically."""
    if not keywords:
        return 1.0
    hay = normalize(pred, fold_accents)
    hits = sum(1 for k in keywords if normalize(k, fold_accents) in hay)
    return hits / len(keywords)


def _parse_json_loose(pred: str):
    """Best-effort JSON extraction: bare, fenced, or the first {...} block."""
    candidate = (pred or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    brace = re.search(r"\{.*\}", candidate, re.S)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            return None
    return None


def json_parses(pred: str) -> float:
    """Format compliance: does the output parse as JSON (fenced or bare)?"""
    candidate = pred.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    try:
        json.loads(candidate)
        return 1.0
    except Exception:
        return 0.0


def has_required_keys(pred: str, keys: list[str]) -> float:
    """Fraction of required keys present in the emitted object.

    Uses the SAME loose parser as `triage_field_accuracy`. They diverged at first —
    this one only handled bare/fenced JSON while the target scorer also recovered a
    `{...}` block embedded in prose. A model that answers
    `"Day la ket qua: {...}"` would then score on the target group but zero on format,
    which reads as a formatting failure that did not happen. Two scorers disagreeing
    about whether output *is* JSON makes both numbers untrustworthy.
    """
    obj = _parse_json_loose(pred)
    if not isinstance(obj, dict) or not keys:
        return 0.0
    return sum(1 for k in keys if k in obj) / len(keys)


def valid_reasoning_trace(pred: str, open_tag="<think>", close_tag="</think>") -> float:
    """Deck §13.5: the metric that catches reasoning-trace collapse.

    A trace is valid when the block exists, is closed, and is not empty. Task accuracy
    can rise while this falls to zero — which is the entire point of measuring it.
    """
    if open_tag not in pred or close_tag not in pred:
        return 0.0
    body = pred.split(open_tag, 1)[1].split(close_tag, 1)[0]
    return float(len(body.strip()) >= 10)


TRIAGE_KEYS = ["intent", "urgency", "product", "sentiment"]


def triage_field_accuracy(pred: str, label: dict, keys: list[str] | None = None) -> float:
    """Target-task scorer: fraction of the 4 triage fields predicted correctly.

    Partial credit on purpose — a model that gets intent right but urgency wrong is
    genuinely better than one that gets neither, and an all-or-nothing scorer hides
    the training signal you are trying to observe across runs.

    `product` is compared with accents folded (it is copied from free text); the
    categorical fields are compared exactly after lowercasing, because they come from
    a closed vocabulary and near-misses there are errors, not spelling.
    """
    keys = keys or TRIAGE_KEYS
    obj = _parse_json_loose(pred)
    if not isinstance(obj, dict):
        return 0.0
    hits = 0
    for k in keys:
        got, want = obj.get(k), label.get(k)
        if got is None or want is None:
            continue
        if k == "product":
            hits += int(normalize(str(got)) == normalize(str(want)))
        else:
            hits += int(str(got).strip().lower() == str(want).strip().lower())
    return hits / len(keys)


# --- aggregation ------------------------------------------------------------

@dataclass
class GroupScores:
    target: float = 0.0
    regression: float = 0.0
    format: float = 0.0
    latency_ms: float = 0.0
    n: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    passed: bool
    reasons: list[str]
    target_delta: float
    regression_delta: float

    def as_dict(self) -> dict:
        return asdict(self)


# How much general capability you are allowed to trade for target-task gain before the
# run is called a regression. 2 points is tight on purpose: the deck's argument is that
# LoRA's forgetting advantage is real, so a correctly configured run should clear it.
REGRESSION_TOLERANCE = 0.02
MIN_TARGET_GAIN = 0.0


def regression_gate(
    tuned: GroupScores,
    optimized_baseline: GroupScores,
    tolerance: float = REGRESSION_TOLERANCE,
    min_gain: float = MIN_TARGET_GAIN,
) -> Verdict:
    """The gate the lab is graded on.

    Passing requires BOTH:
      * target score strictly better than baseline (b) by at least `min_gain`, and
      * general-capability score not worse than (b) by more than `tolerance`.
    """
    target_delta = tuned.target - optimized_baseline.target
    regression_delta = tuned.regression - optimized_baseline.regression

    # The verdict is derived from the numbers, not from the prose that explains them.
    # This used to be `not any(r.startswith(("target", "general")) for r in reasons)` --
    # correct, but only for as long as nobody reworded a message. A graded pass/fail
    # decided by string prefixes is the same class of quiet failure this lab is about.
    beat_baseline = target_delta > min_gain
    kept_capability = regression_delta >= -tolerance

    reasons: list[str] = []
    if not beat_baseline:
        reasons.append(
            f"target task did not beat the optimized-prompt baseline "
            f"({tuned.target:.3f} vs {optimized_baseline.target:.3f}, delta {target_delta:+.3f}). "
            "A fine-tune that loses to a better prompt is not a fine-tune you should ship."
        )
    if not kept_capability:
        reasons.append(
            f"general capability regressed by {abs(regression_delta):.3f} "
            f"(tolerance {tolerance:.3f}). See deck §14.3 — add 1-5% replay data."
        )
    if not reasons:
        reasons.append(
            f"beat the optimized-prompt baseline by {target_delta:+.3f} with "
            f"{regression_delta:+.3f} general-capability change."
        )
    return Verdict(
        passed=beat_baseline and kept_capability,
        reasons=reasons,
        target_delta=target_delta,
        regression_delta=regression_delta,
    )


def comparison_table(scores: dict[str, GroupScores]) -> list[dict]:
    """Rows for REPORT.md. Key order is the reading order: a, b, then the fine-tune."""
    rows = []
    for name, s in scores.items():
        rows.append({
            "run": name,
            "target": round(s.target, 4),
            "regression": round(s.regression, 4),
            "format": round(s.format, 4),
            "latency_ms": round(s.latency_ms, 1),
            "n": s.n,
        })
    return rows


class Timer:
    """Context manager for the latency group. Wall clock, per generation."""

    def __init__(self) -> None:
        self.ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0


# --- Kaggle additions --------------------------------------------------------------
# The repo scores the four groups inline in NB2 and again in NB5. Two copies of the
# same arithmetic is how (a) and (c) end up scored slightly differently — and the whole
# lab is a comparison. One function, called once per version.

def field_accuracy(preds: list[str], records: list[dict],
                   keys: list[str] | None = None) -> dict:
    """Per-field accuracy, plus how often the output parsed at all.

    This is the autopsy view. A single `target` number cannot tell you whether a run
    lost 0.25 because it emits invalid JSON or because it guesses `urgency` from tone
    words, and those two failures have nothing to do with each other. `unparsed` is
    reported separately because a run that emits prose scores 0 on every field for one
    reason, not four.
    """
    keys = keys or TRIAGE_KEYS
    hits = {k: 0 for k in keys}
    counted = {k: 0 for k in keys}
    unparsed = 0
    for pred, rec in zip(preds, records):
        label = rec.get("label")
        if label is None:
            label = _parse_json_loose(rec.get("output", "")) or {}
        obj = _parse_json_loose(pred)
        if not isinstance(obj, dict):
            unparsed += 1
            for k in keys:
                counted[k] += 1
            continue
        for k in keys:
            want = label.get(k)
            if want is None:
                continue
            counted[k] += 1
            got = obj.get(k)
            if got is None:
                continue
            if k == "product":
                hits[k] += int(normalize(str(got)) == normalize(str(want)))
            else:
                hits[k] += int(str(got).strip().lower() == str(want).strip().lower())
    out = {k: (hits[k] / counted[k] if counted[k] else None) for k in keys}
    out["unparsed"] = unparsed
    out["n"] = len(preds)
    return out


def score_version(
    target_records: list[dict],
    target_preds: list[str],
    regression_records: list[dict],
    regression_preds: list[str],
    latency_ms: float,
    extra: dict | None = None,
) -> GroupScores:
    """All four groups for one version, from already-generated strings.

    * target     — mean `triage_field_accuracy` against the labelled JSON
    * regression — mean `keyword_recall` over the general-question keywords
    * format     — mean `has_required_keys`, i.e. "is it the 4-key object we asked for"
    * latency    — passed in, because only the generation loop can measure it honestly

    `format` deliberately uses `has_required_keys` rather than `json_parses`: valid JSON
    with the wrong keys is not format compliance, it just parses. `json_parses` is
    recorded in `extra` so the difference between "not JSON" and "wrong JSON" stays
    visible.
    """
    if len(target_preds) != len(target_records):
        raise ValueError(
            f"target: {len(target_preds)} predictions for {len(target_records)} records. "
            "Scoring misaligned predictions produces a number that means nothing."
        )
    if len(regression_preds) != len(regression_records):
        raise ValueError(
            f"regression: {len(regression_preds)} predictions for "
            f"{len(regression_records)} records."
        )

    tgt, fmt, parses = [], [], []
    for pred, rec in zip(target_preds, target_records):
        label = rec.get("label")
        if label is None:
            label = _parse_json_loose(rec.get("output", "")) or {}
        tgt.append(triage_field_accuracy(pred, label))
        fmt.append(has_required_keys(pred, TRIAGE_KEYS))
        parses.append(json_parses(pred))

    reg = [keyword_recall(p, r.get("keywords", []))
           for p, r in zip(regression_preds, regression_records)]

    ex = {
        "json_parses": round(sum(parses) / len(parses), 4) if parses else 0.0,
        "fields": field_accuracy(target_preds, target_records),
        "n_regression": len(reg),
    }
    if extra:
        ex.update(extra)
    return GroupScores(
        target=sum(tgt) / len(tgt) if tgt else 0.0,
        regression=sum(reg) / len(reg) if reg else 0.0,
        format=sum(fmt) / len(fmt) if fmt else 0.0,
        latency_ms=latency_ms,
        n=len(target_records),
        extra=ex,
    )
