"""Latency + prompt length -> money. The missing half of "was the fine-tune worth it?"

The lab measures three versions and picks a winner on accuracy. That is only half the
decision a team actually makes. The other half is the one the deck's §17 framing asks
for: a fine-tune costs GPU-hours up front and buys you a *shorter prompt* and *shorter
output* on every request forever. This module turns the numbers the eval sweep already
produced into the two figures a decision needs:

    $/1k tickets      what serving this version costs
    break-even        how many tickets it takes to repay the training run

**Everything here is arithmetic over measured inputs.** `mean_latency_ms`,
`prompt_tokens_mean` and `new_tokens_mean` come from `generate.generate_measured()`;
the prices are stated assumptions with a source. Nothing is estimated from a vibe, and
every price is a keyword argument so a report can re-run the whole section against its
own cloud bill instead of arguing with the defaults.

**The honest caveats, stated once and re-stated in the notebook output:**

1. `mean_latency_ms` is *batched* throughput at the eval sweep's batch size. Serving a
   single ticket interactively is slower per ticket; serving at batch 32 is faster.
   The comparison BETWEEN versions is fair because every version is measured at the
   same batch size, which is the claim this lab needs. It is not a quote for prod.
2. Self-hosted cost assumes the GPU is busy. A GPU rented and left idle costs the same
   per hour, so the self-hosted numbers are a floor that only a well-fed queue reaches.
3. Nothing here prices engineer time, and the fine-tune spent some.
"""
from __future__ import annotations

# --- price assumptions -------------------------------------------------------------
# Rented-GPU rate. The default is deliberately the *cheap* end of the market for a T4
# class card (spot/preemptible, mid-2026, ~$0.35/hr) because a low self-host rate is the
# assumption HOSTILE to the conclusion this lab wants to draw: it makes fine-tuning look
# expensive to amortize, not cheap. If your bill says otherwise, pass your own number.
GPU_HOURLY_USD = 0.35

# A hosted small-model API, priced per 1M tokens, as the "don't fine-tune anything"
# alternative. Order of magnitude of a 2026 small/flash tier.
API_INPUT_USD_PER_MTOK = 0.15
API_OUTPUT_USD_PER_MTOK = 0.60


def self_hosted_per_1k(mean_latency_ms: float,
                       gpu_hourly_usd: float = GPU_HOURLY_USD) -> float:
    """$ to serve 1000 tickets on a GPU you rent by the hour."""
    gpu_hours = (mean_latency_ms / 1000.0) * 1000 / 3600.0
    return gpu_hours * gpu_hourly_usd


def api_per_1k(prompt_tokens_mean: float, new_tokens_mean: float,
               input_usd_per_mtok: float = API_INPUT_USD_PER_MTOK,
               output_usd_per_mtok: float = API_OUTPUT_USD_PER_MTOK) -> float:
    """$ to serve 1000 tickets through a token-priced API at these token counts.

    This is why prompt length is an economic quantity: prompt (b) carries the whole
    schema on *every* request, and a fine-tune moves that schema into the weights.
    """
    return (prompt_tokens_mean * 1000 / 1e6) * input_usd_per_mtok + \
           (new_tokens_mean * 1000 / 1e6) * output_usd_per_mtok


def training_usd(train_seconds: float, gpu_hourly_usd: float = GPU_HOURLY_USD) -> float:
    """One-off cost of the training run itself."""
    return (train_seconds / 3600.0) * gpu_hourly_usd


def break_even_tickets(training_cost_usd: float, saving_per_1k_usd: float) -> float | None:
    """How many tickets before the training run has paid for itself.

    `None` when the saving is zero or negative — the honest answer to "when does this
    pay off?" is sometimes "never", and a number like 1e9 hides that.
    """
    if saving_per_1k_usd <= 0:
        return None
    return training_cost_usd / saving_per_1k_usd * 1000


def row(version: str, stats: dict, *, gpu_hourly_usd: float = GPU_HOURLY_USD,
        input_usd_per_mtok: float = API_INPUT_USD_PER_MTOK,
        output_usd_per_mtok: float = API_OUTPUT_USD_PER_MTOK) -> dict:
    """One line of the cost table, straight from a `generate_measured` stats dict."""
    lat = float(stats["mean_latency_ms"])
    p_tok = float(stats["prompt_tokens_mean"])
    o_tok = float(stats["new_tokens_mean"])
    return {
        "version": version,
        "prompt_tok": round(p_tok, 1),
        "out_tok": round(o_tok, 1),
        "ms/sample": round(lat, 1),
        "ms/token": stats.get("ms_per_new_token"),
        "self_host_$/1k": round(self_hosted_per_1k(lat, gpu_hourly_usd), 4),
        "api_$/1k": round(api_per_1k(p_tok, o_tok, input_usd_per_mtok,
                                     output_usd_per_mtok), 4),
    }


def compare(stats_by_version: dict[str, dict], *,
            train_seconds: float | None = None,
            baseline: str = "(b) base + optimized prompt",
            winner: str = "(c) fine-tune",
            gpu_hourly_usd: float = GPU_HOURLY_USD,
            input_usd_per_mtok: float = API_INPUT_USD_PER_MTOK,
            output_usd_per_mtok: float = API_OUTPUT_USD_PER_MTOK) -> dict:
    """The whole cost section: per-version rows plus the break-even against `baseline`.

    `baseline` is the *prompted base model*, not the naive prompt, because that is the
    real alternative to fine-tuning — deck §17: a fine-tune that cannot beat a
    well-prompted base model is not worth shipping, and that applies to its bill too.
    """
    rows = [row(v, s, gpu_hourly_usd=gpu_hourly_usd,
                input_usd_per_mtok=input_usd_per_mtok,
                output_usd_per_mtok=output_usd_per_mtok)
            for v, s in stats_by_version.items()]
    by_version = {r["version"]: r for r in rows}
    out: dict = {
        "rows": rows,
        "assumptions": {
            "gpu_hourly_usd": gpu_hourly_usd,
            "api_input_usd_per_mtok": input_usd_per_mtok,
            "api_output_usd_per_mtok": output_usd_per_mtok,
            "note": "mean_latency_ms is batched throughput at the eval sweep's batch "
                    "size, measured with warm-up excluded. Comparable across versions; "
                    "not a production quote.",
        },
    }
    if baseline in by_version and winner in by_version:
        b, w = by_version[baseline], by_version[winner]
        deltas = {
            "baseline": baseline,
            "winner": winner,
            "latency_ms_delta": round(w["ms/sample"] - b["ms/sample"], 1),
            "latency_speedup_x": round(b["ms/sample"] / w["ms/sample"], 2)
            if w["ms/sample"] else None,
            "prompt_tokens_saved": round(b["prompt_tok"] - w["prompt_tok"], 1),
            "output_tokens_saved": round(b["out_tok"] - w["out_tok"], 1),
            "self_host_saving_per_1k": round(b["self_host_$/1k"] - w["self_host_$/1k"], 4),
            "api_saving_per_1k": round(b["api_$/1k"] - w["api_$/1k"], 4),
        }
        if train_seconds is not None:
            tc = training_usd(train_seconds, gpu_hourly_usd)
            deltas["training_usd"] = round(tc, 3)
            deltas["train_seconds"] = round(train_seconds, 1)
            be = break_even_tickets(tc, deltas["self_host_saving_per_1k"])
            deltas["break_even_tickets_self_host"] = None if be is None else int(be)
            be_api = break_even_tickets(tc, deltas["api_saving_per_1k"])
            deltas["break_even_tickets_api_equivalent"] = None if be_api is None else int(be_api)
        out["delta"] = deltas
    return out


def verdict_line(comparison: dict) -> str:
    """One sentence a report can quote. Says "never" when the answer is never."""
    d = comparison.get("delta")
    if not d:
        return "cost comparison unavailable: baseline or winner version missing."
    parts = [
        f"{d['winner']} vs {d['baseline']}: "
        f"{d['latency_ms_delta']:+.0f} ms/sample",
    ]
    if d["prompt_tokens_saved"]:
        parts.append(f"{d['prompt_tokens_saved']:+.0f} prompt tokens")
    if d["output_tokens_saved"]:
        parts.append(f"{d['output_tokens_saved']:+.0f} output tokens")
    parts.append(f"self-host {d['self_host_saving_per_1k']:+.4f} $/1k")
    line = ", ".join(parts) + "."
    if "break_even_tickets_self_host" in d:
        be = d["break_even_tickets_self_host"]
        line += (f" Training cost ${d['training_usd']:.3f}; break-even at "
                 f"{be:,} tickets." if be is not None else
                 f" Training cost ${d['training_usd']:.3f} and serving is NOT cheaper — "
                 "it never breaks even on cost alone; the case has to be made on quality.")
    return line
