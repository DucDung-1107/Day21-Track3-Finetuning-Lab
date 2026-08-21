"""Model loading and batch generation — the only GPU-touching module.

Kept separate from `evaluate` on purpose: scoring is pure functions over strings and
runs anywhere, so the whole grading contract stays testable on a laptop. Only this
file needs a GPU.
"""
from __future__ import annotations

import gc
import time

from . import device
from .config import NAIVE_PROMPT, OPTIMIZED_PROMPT, Tier

# The two prompts that define baselines (a) and (b). Baseline (b) has to be a genuine
# effort — deck §17's whole point is that a fine-tune which cannot beat a *well-prompted*
# base model is not worth shipping. Writing a deliberately weak (b) to flatter your
# fine-tune is the main way to cheat this lab, and the rubric checks for it.

def free_memory() -> None:
    """Between runs. Deck §16: not doing this is the most common OOM in a multi-run lab."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:      # pragma: no cover
        pass


def peak_vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:      # pragma: no cover
        pass
    return None


def load_base(tier: Tier, load_in_4bit: bool = False):
    """Load the base model + tokenizer for `tier`.

    `load_in_4bit` is exposed only so NB4 can *measure* the QLoRA contrast. The default
    is bf16 because the vendor advises against 4-bit on this model family (deck §12).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)
    # dtype (not torch_dtype — deprecated in transformers 5.x) and NOT hardcoded bf16:
    # the lab's default tier is a T4, which has no bfloat16 (see labkit/device.py).
    kwargs: dict = {"trust_remote_code": True, "dtype": device.torch_dtype(),
                    "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=device.torch_dtype(),
        )
    model = AutoModelForCausalLM.from_pretrained(tier.model_id, **kwargs)
    return model, tok


def render_prompt(tok, prompt: str, system: str | None,
                  enable_thinking: bool | None = False) -> str:
    """The exact string handed to the model. Shared by generation and by cost accounting
    so "prompt tokens" is measured on what was actually sent, not on a re-render."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    kw: dict = {"tokenize": False, "add_generation_prompt": True}
    # None means OMIT the kwarg, matching data._render(). Passing a literal None
    # is not the same thing: it lands in the Jinja context where `is defined` is
    # true, so the template may read it as falsey and quietly behave like False.
    # Training and generation have to be able to express "template default"
    # identically or they cannot be compared at all.
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    try:
        return tok.apply_chat_template(msgs, **kw)
    except TypeError:
        kw.pop("enable_thinking", None)
        return tok.apply_chat_template(msgs, **kw)


def generate_measured(
    model,
    tok,
    prompts: list[str],
    system: str | None = None,
    max_new_tokens: int = 160,
    enable_thinking: bool | None = False,
    batch_size: int = 4,
    label: str = "generate",
    progress: bool = True,
    warmup: bool = True,
) -> tuple[list[str], dict]:
    """Greedy decode. Returns (completions, stats).

    Greedy (do_sample=False) is deliberate: this lab compares runs, and sampling noise
    would swamp the differences you are trying to measure.

    Kaggle edition, two additions over `src/labkit`:

    * **`warmup=True` discards one throwaway generation before the clock starts.** The
      first `generate()` on a freshly loaded model pays for CUDA kernel autotuning,
      cuBLAS workspace allocation and the first KV-cache malloc. Measured on a T4 that
      first call is seconds slower than the second, and with the shipped eval sizes it
      lands almost entirely in batch 1 — i.e. straight into the latency number the cost
      section then multiplies by 1000. Warm-up cost is real but it is a *deployment*
      cost paid once, not a per-ticket cost.
    * **token accounting.** Latency alone cannot distinguish "the fine-tune is faster"
      from "the fine-tune emitted fewer tokens", and for this lab it is nearly always
      the latter: the base model with prompt (b) writes an explanation, the fine-tune
      writes 40 tokens of JSON. `ms_per_new_token` separates the two, and
      `prompt_tokens_mean` is what makes the prompt-shrink argument in `cost.py` an
      economic claim instead of an aesthetic one.

    `progress` prints a per-batch line with an ETA. This is not decoration: on a free
    T4 a 4B model takes tens of minutes to score the eval set, and a notebook that
    prints nothing for that long is indistinguishable from a hang. Students kill runs
    that look stuck.
    """
    import torch

    outs: list[str] = []
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    texts_all = [render_prompt(tok, p, system, enable_thinking) for p in prompts]
    prompt_tok_counts = [len(tok(t, add_special_tokens=False)["input_ids"])
                         for t in texts_all]

    warmup_ms = 0.0
    if warmup and prompts:
        enc = tok(texts_all[:1], return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        if progress:
            print(f"  [{label}] warm-up discarded: {warmup_ms:.0f} ms", flush=True)

    total_ms = 0.0
    new_tokens = 0
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t_start = time.perf_counter()
    for bi, i in enumerate(range(0, len(prompts), batch_size), start=1):
        texts = texts_all[i:i + batch_size]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        total_ms += (time.perf_counter() - t0) * 1000.0
        for row, src in zip(gen, enc["input_ids"]):
            tail = row[len(src):]
            # Count real tokens only: padded rows finish early and the rest of the row
            # is pad. Counting pad as work would make a batch look slower per token the
            # more uneven it is.
            kept = [int(t) for t in tail.tolist() if t != tok.pad_token_id]
            new_tokens += len(kept)
            outs.append(tok.decode(tail, skip_special_tokens=True).strip())

        if progress:
            done = time.perf_counter() - t_start
            eta = done / bi * (n_batches - bi)
            print(f"  [{label}] batch {bi}/{n_batches}  "
                  f"{done:5.0f}s elapsed  ~{eta:5.0f}s left", flush=True)

    n = max(1, len(prompts))
    stats = {
        "label": label,
        "n_prompts": len(prompts),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "warmup_ms": round(warmup_ms, 1),
        "gen_seconds": round(total_ms / 1000.0, 2),
        "mean_latency_ms": round(total_ms / n, 1),
        "prompt_tokens_total": sum(prompt_tok_counts),
        "prompt_tokens_mean": round(sum(prompt_tok_counts) / n, 1),
        "new_tokens_total": new_tokens,
        "new_tokens_mean": round(new_tokens / n, 1),
        "ms_per_new_token": round(total_ms / new_tokens, 2) if new_tokens else None,
        "tokens_per_second": round(new_tokens / (total_ms / 1000.0), 1) if total_ms else None,
    }
    if progress:
        print(f"  [{label}] done: {len(prompts)} prompts in "
              f"{time.perf_counter() - t_start:.0f}s  "
              f"({stats['mean_latency_ms']:.0f} ms/sample, "
              f"{stats['new_tokens_mean']:.0f} new tok/sample)", flush=True)
    return outs, stats


def generate_batch(
    model,
    tok,
    prompts: list[str],
    system: str | None = None,
    max_new_tokens: int = 160,
    enable_thinking: bool | None = False,
    batch_size: int = 4,
    label: str = "generate",
    progress: bool = True,
    warmup: bool = False,
) -> tuple[list[str], float]:
    """`generate_measured` with the repo's original 2-tuple signature, so the six
    stage notebooks keep working unchanged. New code should call `generate_measured`
    and keep the whole stats dict — the cost section needs the token counts."""
    outs, stats = generate_measured(
        model, tok, prompts, system=system, max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking, batch_size=batch_size, label=label,
        progress=progress, warmup=warmup,
    )
    return outs, stats["mean_latency_ms"]
