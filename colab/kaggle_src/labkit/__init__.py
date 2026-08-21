"""labkit (Kaggle edition) — the shared harness for Day 21's fine-tuning lab.

Same contract as `src/labkit`, with the pipeline improvements the Kaggle notebook
adds on top. Differences from the repo copy are listed in one place so the notebook
can print them and the report can cite them:

    KAGGLE_DELTAS  -- what changed, and why

No `.env` loading here: on Kaggle every knob comes from the CONFIG cell, which sets
`os.environ` before this package is imported.
"""

KAGGLE_DELTAS = [
    ("replay mix (§14.3)",
     "data.mix_replay() folds a decontaminated general-instruction corpus into the "
     "training set, and `correct_replay` trains on it at the SAME step budget as "
     "`correct`. This targets the one measured, unfixed defect in the repo pipeline: "
     "regression 0.644 -> 0.067, a total capability collapse."),
    ("held-out loss during training",
     "train.sft_config_kwargs() wires data/split/val.jsonl in as eval_dataset, so "
     "overfitting is visible in the log instead of inferred after the fact."),
    ("F-31 guard is an assert, not a docstring",
     "data.assert_prompt_alignment() fails the run when the evaluation render is not "
     "a prefix of the training render — the defect that made every adapter score 0.000."),
    ("latency excludes warm-up",
     "generate.generate_batch(warmup=True) discards a throwaway generation before "
     "timing, and returns token counts, so ms/sample and ms/token are both real."),
    ("cost is measured, not asserted",
     "cost.py turns measured latency + prompt length into $/1k tickets for each "
     "version, which is what makes 'the prompt can shrink' an economic claim."),
    ("holdout_secret.jsonl is finally used",
     "the shipped 20-item holdout is scored once, on the winning version only, as a "
     "check against having tuned on eval_target."),
    ("integrity checks run inside the notebook",
     "checksums of the four corpus files, the SHA of prompt (b), the shared step "
     "budget and the matched parameter budget are all asserted in-line rather than "
     "by a separate verify script the grader has to trust was run."),
]

__all__ = ["config", "cost", "data", "evaluate", "generate", "modeling", "replay",
           "report", "train", "device", "KAGGLE_DELTAS"]
__version__ = "1.0.0-kaggle"
