"""Build `colab/Lab21_KAGGLE_FULL.ipynb` — the single self-contained Kaggle notebook.

Why a builder instead of hand-editing a .ipynb: the notebook carries the whole `labkit`
harness inline (Kaggle rule for this lab: **no cloning anyone's repo**), and a .ipynb is
a JSON file with every source line as a quoted string. Editing 3000 lines of Python
through that is how you ship a notebook with a syntax error in cell 24.

So the modules stay real files under `colab/kaggle_src/labkit/`, where they can be
imported, linted and unit-tested locally, and this script inlines them into
`%%writefile` cells. `python colab/build_kaggle_notebook.py` also `compile()`s every
emitted code cell, so a build that succeeds cannot contain a Python syntax error.

Usage:
    python colab/build_kaggle_notebook.py            # build + validate
    python colab/build_kaggle_notebook.py --check     # validate only, no write
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "kaggle_src" / "labkit"
OUT = HERE / "Lab21_KAGGLE_FULL.ipynb"

# Import order matters only for readability; every module is written before any import.
MODULES = ["__init__", "config", "device", "report", "replay", "data", "evaluate",
           "modeling", "train", "generate", "cost"]

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", textwrap.dedent(text).strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", textwrap.dedent(text).strip("\n")))


def raw_code(text: str) -> None:
    """Code appended verbatim — for `%%writefile` payloads, where dedent would be wrong."""
    CELLS.append(("code", text.strip("\n")))


def _intro() -> None:
    md("""
    # Lab 21 · Track 3 — Fine-tuning LLM · **Kaggle FULL run** (một notebook duy nhất)

    Chạy từ trên xuống. Notebook này **tự chứa toàn bộ harness** — không `git clone`
    repo của ai, không tải script từ đâu. Mọi module `labkit` được ghi ra đĩa bằng
    `%%writefile` ở phần 3 và bạn đọc được toàn bộ code ngay trong notebook.

    ### Trước khi chạy
    | Cần | Ở đâu |
    |---|---|
    | **GPU T4** | Settings → Accelerator → **GPU T4 x2** (notebook tự khoá về 1 card) |
    | **Internet ON** | Settings → Internet → On (để `pip install` + tải base model) |
    | **Dataset `LAB21_VIN`** | Add Input → Datasets → `LAB21_VIN` (5 file `.jsonl`/`.json`) |

    ### Notebook này trả lời gì (bản đồ sang yêu cầu của lab)
    | Yêu cầu | Ở phần |
    |---|---|
    | Chat template + **mask loss có bằng chứng** | §4 |
    | `max_length` lấy từ **p95 đo được** | §4 |
    | Ba phiên bản (a) naive · (b) prompt tối ưu · (c) fine-tune, đo **trước** khi train | §6 |
    | Train cấu hình đúng (all-linear · LR 10× · batch<32 · alpha=2r) | §7 |
    | **Ba cấu hình sai** cùng ngân sách step, ngân sách tham số khớp | §8 |
    | Bốn nhóm điểm **target · regression · format · latency** + phán quyết | §9 |
    | **Latency & cost** ($/1k ticket, break-even) | §10 |
    | **Chạy thử câu hỏi thật**, gồm cả câu ngoài miền | §11 |
    | Merge + hoán đổi adapter | §12 |
    | Cổng kiểm tra trước khi nộp + REPORT.md + zip artefact | §13 |

    ### Ba thứ notebook này làm mà pipeline gốc chưa làm
    1. **`correct_replay`** — deck §14.3: trộn 1–5% dữ liệu tổng quát đã *chứng minh
       không trùng* tập regression, train ở **đúng cùng số step** với `correct`. Đây là
       đối chứng cho đúng khuyết điểm mà repo tự đo được: regression **0.644 → 0.067**.
    2. **Loss trên tập held-out ngay trong lúc train** — val split đã có từ §4 nhưng
       chưa ai dùng; overfit trở thành số đo thay vì suy đoán.
    3. **Latency đã trừ warm-up + đếm token** → cost là phép tính trên số đo, và
       "prompt ngắn đi" trở thành một lập luận kinh tế.
    """)


def _bootstrap() -> None:
    md("""
    ## 1. Môi trường — 1 GPU, dependency, không clone gì cả

    Kaggle cấp **2×T4**. `device_map="auto"` sẽ chia model qua cả hai card, và một
    model bị shard + LoRA + `GradScaler` của fp16 là đường ngắn nhất tới
    `expected all tensors on the same device` ở step 0. Khoá về 1 card **trước khi**
    torch được import lần đầu — sau khi torch đã khởi tạo CUDA thì đặt biến môi trường
    không còn tác dụng.
    """)
    code("""
        import os, pathlib, subprocess, sys

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # phải đặt TRƯỚC khi import torch
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        WORK = pathlib.Path("/kaggle/working")
        if not WORK.exists():
            WORK = pathlib.Path.cwd()
        os.chdir(WORK)
        for sub in ("data", "results", "adapters", "submission"):
            (WORK / sub).mkdir(exist_ok=True)
        print("cwd:", WORK)

        # Pin theo requirements.txt của lab. Cài TRƯỚC khi import transformers/trl để
        # kernel không phải restart: một notebook đòi restart giữa 3 tiếng train là một
        # notebook sẽ bị chạy lại từ đầu.
        PKGS = [
            "transformers>=5.15,<6", "trl>=1.10,<2", "peft>=0.20,<1",
            "accelerate>=1.14,<2", "datasets>=5,<6", "torchao>=0.16",
            "tokenizers>=0.22", "bitsandbytes>=0.50",
        ]
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *PKGS],
                              capture_output=True, text=True)
        print(proc.stdout[-2000:] or "pip: ok")
        if proc.returncode:
            print(proc.stderr[-3000:])
            raise SystemExit(
                "pip install thất bại. Kiểm tra Settings → Internet = On."
            )

        import torch, transformers, trl, peft, datasets
        print(f"torch {torch.__version__} · transformers {transformers.__version__} · "
              f"trl {trl.__version__} · peft {peft.__version__} · datasets {datasets.__version__}")
        if not torch.cuda.is_available():
            raise SystemExit("Không thấy GPU. Settings → Accelerator → GPU T4 x2.")
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} · {p.total_memory/1024**3:.1f} GB · sm_{p.major}{p.minor} "
              f"· visible devices = {torch.cuda.device_count()}")
    """)

    md("""
    ## 2. CONFIG — mọi nút của cả lab nằm trong ô này

    Đọc hết ô này rồi hãy chạy. `EVAL_LIMIT = ""` là bài nộp (full eval); đặt `"8"` để
    tổng duyệt. `COMPUTE_TIER = "SMOKE"` đổi sang checkpoint 0.8B để rà đường ống trong
    ~25 phút — điểm tuyệt đối của SMOKE **không** so được với T4, chỉ có *thứ tự* là
    thứ đáng tin ở một lần tổng duyệt.
    """)
    code("""
        # ---- các nút của lab -------------------------------------------------------
        COMPUTE_TIER    = "T4"              # "T4" = 4B (bài nộp) | "SMOKE" = 0.8B (tổng duyệt)
        MODEL_ID        = ""                # để trống = checkpoint của tier
        EVAL_LIMIT      = ""                # "" = FULL (bài nộp) | "8" = smoke
        EPOCHS          = "2"               # chung cho MỌI run, nên các run so được
        MASK_MODE       = "assistant-only"
        REPLAY_FRACTION = "0.05"            # deck §14.3: 1–5%
        SEED            = 42

        RUN_REPLAY      = True              # §7b: correct_replay (+~25')
        RUN_CONTRASTS   = True              # §8: attn_only / wrong_lr / qlora (+~75')
        RUN_MERGE       = False             # §12: merge ghi ~8 GB xuống /kaggle/working
        FORCE_RETRAIN   = False             # True = train lại cả adapter đã có

        # giá dùng cho §10. Đây là GIẢ ĐỊNH, không phải số đo — đổi theo hoá đơn của bạn.
        GPU_HOURLY_USD  = 0.35              # T4 spot/preemptible ~2026
        API_IN_USD_MTOK = 0.15              # API small-model, $/1M token vào
        API_OUT_USD_MTOK = 0.60             # $/1M token ra

        # ---- đẩy xuống os.environ TRƯỚC khi import labkit --------------------------
        import os
        os.environ["COMPUTE_TIER"] = COMPUTE_TIER
        os.environ["EPOCHS"] = EPOCHS
        os.environ["MASK_MODE"] = MASK_MODE
        os.environ["REPLAY_FRACTION"] = REPLAY_FRACTION
        os.environ["LAB_RESULTS"] = str(WORK / "results")
        if MODEL_ID.strip():
            os.environ["MODEL_ID"] = MODEL_ID.strip()
        else:
            os.environ.pop("MODEL_ID", None)
        if EVAL_LIMIT.strip():
            os.environ["EVAL_LIMIT"] = EVAL_LIMIT.strip()
        else:
            os.environ.pop("EVAL_LIMIT", None)

        SMOKE = bool(EVAL_LIMIT.strip()) or COMPUTE_TIER.upper() == "SMOKE"
        plan = [
            ("§4  data + mask + split", "~2'"),
            ("§5  replay corpus + decontamination", "~5s"),
            ("§6  baselines (a) và (b)", "~12'"),
            ("§7  train correct", "~25'"),
            ("§7b train correct_replay", "~25'" if RUN_REPLAY else "TẮT"),
            ("§8  3 contrast (attn_only/wrong_lr/qlora)", "~75'" if RUN_CONTRASTS else "TẮT"),
            ("§9  chấm 4 nhóm + phán quyết", "~20'"),
            ("§10 latency + cost", "~0'"),
            ("§11 câu hỏi thử", "đã đo ở §6/§9"),
            ("§12 merge + hot-swap", "~8'" if RUN_MERGE else "TẮT"),
            ("§13 cổng kiểm tra + REPORT + zip", "~1'"),
        ]
        print(f"tier={COMPUTE_TIER}  EVAL_LIMIT={EVAL_LIMIT or 'FULL'}  EPOCHS={EPOCHS}  "
              f"replay={REPLAY_FRACTION}  smoke_mode={SMOKE}")
        if SMOKE:
            print("\\n⚠ SMOKE MODE — kết quả KHÔNG dùng để nộp (§13 sẽ báo FAIL).")
        print("\\nKế hoạch (ước lượng trên T4, tier T4/4B):")
        for what, how_long in plan:
            print(f"  {what:<44} {how_long}")
    """)

    md("""
    ## 3. Dữ liệu — tìm dataset đã attach và **kiểm tra checksum**

    Bốn file corpus là hợp đồng của bài lab: sửa tập eval sau khi thấy kết quả thì mọi
    so sánh phía sau đều vô nghĩa. `checksums.json` đi kèm dataset là cách kiểm việc đó
    bằng một con số.

    Một chi tiết thật, không phải phòng xa: các checksum ấy được tính trên nội dung
    **LF**, còn một checkout Windows (`core.autocrlf=true`) cho ra file **CRLF** — cùng
    dữ liệu, khác byte, khác SHA. Nên ô này băm cả hai dạng và chỉ báo FAIL khi *cả hai*
    đều lệch. Băm thô một mình sẽ tố oan mọi dataset upload từ Windows.
    """)
    code("""
        import hashlib, json, pathlib, shutil

        NEEDED = ["train_seed.jsonl", "eval_target.jsonl",
                  "eval_regression.jsonl", "holdout_secret.jsonl"]
        DATA = WORK / "data"

        def _find_dataset_root() -> pathlib.Path:
            roots = [p for p in pathlib.Path("/kaggle/input").glob("*")] \\
                    if pathlib.Path("/kaggle/input").exists() else []
            cands = []
            for root in roots:
                for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
                    if (d / "train_seed.jsonl").exists():
                        cands.append(d)
            if not cands:
                if all((DATA / f).exists() for f in NEEDED):
                    return DATA           # đã copy từ lần chạy trước
                raise SystemExit(
                    "Không tìm thấy train_seed.jsonl trong /kaggle/input.\\n"
                    "Add Input → Datasets → LAB21_VIN rồi chạy lại ô này."
                )
            # ưu tiên thư mục có đủ nhất
            cands.sort(key=lambda d: sum((d / f).exists() for f in NEEDED), reverse=True)
            return cands[0]

        SRC_DATA = _find_dataset_root()
        print("dataset:", SRC_DATA)
        for f in NEEDED + ["checksums.json"]:
            s = SRC_DATA / f
            if s.exists():
                if s.resolve() != (DATA / f).resolve():
                    shutil.copy2(s, DATA / f)
            elif f in NEEDED:
                raise SystemExit(f"thiếu {f} trong dataset đã attach")
            else:
                print(f"  (không có {f} — bỏ qua bước kiểm checksum)")

        def sha16(path: pathlib.Path) -> tuple[str, str]:
            \"\"\"(sha thô, sha sau khi chuẩn hoá CRLF->LF), mỗi cái 16 hex đầu.\"\"\"
            b = path.read_bytes()
            return (hashlib.sha256(b).hexdigest()[:16],
                    hashlib.sha256(b.replace(b"\\r\\n", b"\\n")).hexdigest()[:16])

        integrity = {"dataset_dir": str(SRC_DATA), "files": {}, "drift": []}
        ref = {}
        if (DATA / "checksums.json").exists():
            ref = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))
        for f in NEEDED:
            p = DATA / f
            raw, lf = sha16(p)
            n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
            want = ref.get(f)
            ok = want is None or want in (raw, lf)
            integrity["files"][f] = {"rows": n, "sha_raw": raw, "sha_lf": lf,
                                     "expected": want, "ok": ok}
            if not ok:
                integrity["drift"].append(f)
            flag = "ok" if ok else "DRIFT"
            print(f"  {f:<24} {n:>4} dòng  sha={raw} lf={lf}  expected={want}  [{flag}]")

        if integrity["drift"]:
            raise SystemExit(
                f"checksum lệch: {integrity['drift']}. Tập eval đã bị sửa — mọi so sánh "
                "sau đây sẽ vô nghĩa. Attach lại dataset gốc."
            )
        print("\\ncorpus khớp checksum.")
    """)


def _modules() -> None:
    md("""
    ## 4. Harness `labkit` — ghi thẳng ra đĩa, không clone

    Mười một module dưới đây **là** thư viện của lab. Chúng ở đây dưới dạng
    `%%writefile` vì luật của bài: notebook Kaggle không được đi clone repo của bên
    khác. Đổi lại, bạn đọc được toàn bộ code đang chạy — kể cả những chỗ khó, như vì sao
    mask được dựng từ **offset ký tự** chứ không phải bằng cách trừ token
    (`data.build_example`), hay vì sao `assistant_only_loss` của TRL bị **cố ý không
    dùng** (`train.sft_config_kwargs`).
    """)
    code("""
        import pathlib
        pathlib.Path("labkit").mkdir(exist_ok=True)
        print("labkit/ ready")
    """)
    for name in MODULES:
        source = (SRC / f"{name}.py").read_text(encoding="utf-8")
        raw_code(f"%%writefile labkit/{name}.py\n{source}")

    md("""
    ### Kiểm tra harness ngay: import, tier, và các unit-invariant

    Ba thứ được assert ở đây vì cả ba đều là *lỗi im lặng* nếu sai: batch hiệu dụng
    vượt 32 (deck §10.4), `alpha != 2r` (§9.3), và LR LoRA đặt sai thang (§10.3).
    """)
    code("""
        import importlib, json, sys
        sys.path.insert(0, str(WORK))
        for m in list(sys.modules):
            if m == "labkit" or m.startswith("labkit."):
                del sys.modules[m]

        import labkit
        from labkit import config as C, cost, data, device, evaluate as ev
        from labkit import generate, modeling, replay, report, train

        TIER = C.get_tier()
        assert TIER.effective_batch <= C.MAX_EFFECTIVE_BATCH, "batch hiệu dụng > 32 (§10.4)"
        assert C.SPECS["correct"].alpha == 2 * C.SPECS["correct"].r, "alpha != 2r (§9.3)"
        assert abs(C.LORA_LR / C.FULL_FT_LR - 10.0) < 1e-9, "LoRA LR không ở thang 10x (§10.3)"

        print("labkit", labkit.__version__)
        print(device.banner())
        print(f"\\ntier={TIER.name}  model={TIER.model_id}  max_length={TIER.max_length}  "
              f"batch={TIER.per_device_batch}x{TIER.grad_accum}={TIER.effective_batch}")
        print("\\nNhững gì bản Kaggle này làm khác repo:")
        for what, why in labkit.KAGGLE_DELTAS:
            print(f"\\n  • {what}\\n      {why}")
    """)


def _stage1_data_and_mask() -> None:
    md("""
    ## 5. Dữ liệu, chat template và **mask loss** (NB1 của lab)

    > Deck §13.2: *che loss và chat template quyết định kết quả nhiều hơn mọi biến thể
    > LoRA cộng lại.*

    Phần này sinh 4 artefact bắt buộc và **fail ngay** nếu mask sai — trước khi tiêu một
    phút GPU nào cho việc train:

    1. `template_check.json` — template có nuốt khối `<think>` không (§16)
    2. `mask_proof.json` — loss **chứa** câu trả lời và **không chứa** câu hỏi
    3. `token_stats.json` — p95 → `max_length` (số đo, không phải số đoán)
    4. `data/split/{train,val}.jsonl` — split seed 42, và `val` **được dùng thật** ở §7
    """)
    code("""
        import json, pathlib

        def load_jsonl(p):
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

        RESULTS = WORK / "results"
        train_raw = load_jsonl(DATA / "train_seed.jsonl")
        target_all = load_jsonl(DATA / "eval_target.jsonl")
        regression_all = load_jsonl(DATA / "eval_regression.jsonl")
        holdout_all = load_jsonl(DATA / "holdout_secret.jsonl")

        LIMIT = int(os.environ.get("EVAL_LIMIT", "0") or 0)
        target = target_all[:LIMIT] if LIMIT else target_all
        regression = regression_all[:LIMIT] if LIMIT else regression_all
        holdout = holdout_all[:LIMIT] if LIMIT else holdout_all
        print(f"train={len(train_raw)}  target={len(target)}/{len(target_all)}  "
              f"regression={len(regression)}/{len(regression_all)}  "
              f"holdout={len(holdout)}/{len(holdout_all)}")
        print(json.dumps(train_raw[0], ensure_ascii=False, indent=2)[:500])
    """)

    md("""
    ### 5.1 Tokenizer + kiểm tra bắt buộc #1: template có giữ khối suy luận?

    Chỉ tải **file tokenizer** (vài MB), chưa tải trọng số.
    """)
    code("""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TIER.model_id, trust_remote_code=True)
        check = data.thinking_survives(tok)
        print("eos:", tok.eos_token, "| VERDICT:", check["verdict"])
        print("--- chuỗi đã render ---")
        print(check["rendered"])
        report.write_json(check, "template_check.json", results_dir=RESULTS)
    """)

    md("""
    ### 5.2 Kiểm tra bắt buộc #2: mask — và **đọc** nó

    Bốn chế độ; ở đây in ra hai chế độ đối lập nhất để bạn thấy khác biệt bằng mắt.
    `everything` là bug kinh điển: câu hỏi nằm trong loss → model học *viết lại câu hỏi*.
    """)
    code("""
        sample = data.to_messages(train_raw[0])
        for mode in ("assistant-only", "everything"):
            ex = data.build_example(tok, sample, max_length=TIER.max_length, mask_mode=mode)
            print("=" * 72)
            print(f"mode={mode}  supervised {ex.n_supervised}/{ex.n_total} "
                  f"({ex.supervised_fraction:.0%})")
            print("--- LOSS TÍNH TRÊN ĐOẠN NÀY ---")
            print(data.decode_supervised(tok, ex)[:400])
    """)
    code("""
        ex = data.build_example(tok, sample, max_length=TIER.max_length,
                                mask_mode=MASK_MODE)
        supervised = data.decode_supervised(tok, ex)
        masked = data.decode_masked(tok, ex)
        answer = sample[-1]["content"][:40]
        question_fragment = train_raw[0]["input"][:40]

        proof = {
            "mask_mode": MASK_MODE,
            "n_supervised": ex.n_supervised,
            "n_total": ex.n_total,
            "supervised_fraction": round(ex.supervised_fraction, 4),
            "answer_is_supervised": answer in supervised,
            "question_is_masked": question_fragment not in supervised,
            "supervised_preview": supervised[:300],
            "masked_preview": masked[:300],
        }
        assert proof["answer_is_supervised"], "câu trả lời KHÔNG nằm trong loss — mask sai"
        assert proof["question_is_masked"], "câu hỏi ĐANG nằm trong loss — mask sai"
        assert proof["supervised_fraction"] < 0.95, (
            "gần như mọi token đều vào loss — bạn đang train cả prompt (rubric auto-zero)")
        print(json.dumps({k: v for k, v in proof.items() if not k.endswith("preview")},
                         ensure_ascii=False, indent=2))
        report.write_json(proof, "mask_proof.json", results_dir=RESULTS)
    """)

    md("""
    ### 5.3 Kiểm tra bắt buộc #3 (bản Kaggle thêm): **prompt lúc train == prompt lúc chấm**

    Đây là lỗi F-31 của lab, và nó không tự báo: nếu chuỗi mà evaluation gửi đi không
    phải **tiền tố** của chuỗi đã train, adapter vẫn train xong bình thường rồi ra
    `target = 0.000` trên mọi cấu hình. Repo mô tả nó trong docstring; ở đây nó là một
    `assert` chạy trên 5 mẫu thật.
    """)
    code("""
        align = data.assert_prompt_alignment(tok, train_raw[:5], system=C.NAIVE_PROMPT)
        print(json.dumps({k: v for k, v in align.items()
                          if k not in ("eval_render_example", "supervised_tail_example")},
                         ensure_ascii=False, indent=2))
        print("\\n--- evaluation sẽ gửi đúng chuỗi này ---")
        print(align["eval_render_example"])
        print("--- và train supervise đúng phần đuôi này ---")
        print(align["supervised_tail_example"][:200])
        report.write_json(align, "prompt_alignment.json", results_dir=RESULTS)
    """)

    md("""
    ### 5.4 `max_length` từ p95, và split seed 42
    """)
    code("""
        lengths = [data.build_example(tok, data.to_messages(r), max_length=8192).n_total
                   for r in train_raw]
        stats = data.token_stats(lengths)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        report.write_json(stats, "token_stats.json", results_dir=RESULTS)
        if stats["suggested_max_length"] != TIER.max_length:
            print(f"\\n⚠ p95 gợi ý max_length={stats['suggested_max_length']} nhưng tier "
                  f"đang dùng {TIER.max_length}. Ghi lựa chọn này vào REPORT.md.")

        train_rows, val_rows = data.split(train_raw, train_frac=0.9, seed=SEED)
        split_dir = DATA / "split"
        split_dir.mkdir(exist_ok=True)
        for name, rows in (("train", train_rows), ("val", val_rows)):
            with (split_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\\n")
        print(f"\\ntrain={len(train_rows)}  val={len(val_rows)}  (seed={SEED})")
    """)


def _stage1b_replay() -> None:
    md("""
    ## 5b. Corpus replay (deck §14.3) — và **chứng minh** nó không trùng tập regression

    Đây là phần cải tiến chính so với pipeline gốc. Repo tự đo được: fine-tune đạt
    target 0.990 nhưng general capability rơi **0.644 → 0.067**. Nguyên nhân không phải
    LoRA mà là *corpus*: mọi input đều là ticket và mọi output đều là JSON, nên
    "có input" và "xuất JSON triage" trở thành cùng một thứ trong mắt model.

    Deck §14.3 nói cách chữa là trộn lại 1–5% dữ liệu tổng quát. Rủi ro của cách chữa đó
    là **nhiễm tập eval**: nếu dữ liệu trộn vào chứa chính câu hỏi dùng để đo regression
    thì điểm regression tăng vì đã học thuộc đề, không phải vì giữ được năng lực. Nên
    47 mẫu ở đây được kiểm bằng hai điều kiện — không trùng chính xác (sau khi chuẩn hoá
    dấu) và Jaccard từ vựng ≤ 0.6 với **cả 15** câu regression.
    """)
    code("""
        pool = replay.load()
        decon = data.assert_replay_decontaminated(pool, regression_all)
        print(json.dumps(decon, ensure_ascii=False, indent=2))
        print(f"\\ncặp gần nhất: replay={decon['worst_pair'][0]!r}")
        print(f"              eval={decon['worst_pair'][1]!r}")

        mixed_rows, replay_manifest = data.mix_replay(
            train_rows, pool, C.replay_fraction(), seed=SEED)
        print("\\n" + json.dumps({k: v for k, v in replay_manifest.items()
                                  if k != "replay_instructions"},
                                 ensure_ascii=False, indent=2))
        report.write_json({"decontamination": decon, "mix": replay_manifest},
                          "replay_manifest.json", results_dir=RESULTS)
    """)


def _stage2_baselines() -> None:
    md("""
    ## 6. Ba phiên bản — và **đo baseline TRƯỚC khi train**

    > Deck §17: điểm không nằm ở việc perplexity giảm bao nhiêu, mà ở việc bạn có chứng
    > minh được bản fine-tune thắng **(b)** — và có phát hiện được nếu nó *không* thắng.

    | | Là gì | Vì sao có mặt |
    |---|---|---|
    | **(a)** | base + prompt ngây thơ (`"Phân loại ticket sau."`) | mốc sàn |
    | **(b)** | base + prompt **đã tối ưu** (schema + enum + 1 ví dụ) | **mốc thật sự phải vượt** |
    | **(c)** | fine-tune, chấm với prompt *ngây thơ* | hành vi đã nằm trong trọng số |

    Thứ tự có lý do: đo (b) **sau** khi biết điểm fine-tune thì bạn sẽ vô thức hạ (b) cho
    tới lúc mình thắng. Nên (a) và (b) được đo và **đóng băng** ở đây, kèm SHA của prompt.

    Mỗi phiên bản đi qua **cùng một** `eval_pass`: 50 ticket target · 15 câu tổng quát
    (regression) · 20 ticket **holdout** chưa ai thấy · 3 ticket mẫu · 3 câu hỏi thường
    ngày. Cùng batch size, cùng `max_new_tokens`, greedy — nên số liệu so được với nhau.
    """)
    code("""
        EVAL_BATCH = 4                 # T4 16 GB, 4B model, max_length 1024
        MAX_NEW    = 160

        V_A  = "(a) base + naive prompt"
        V_B  = "(b) base + optimized prompt"
        V_C  = "(c) fine-tune"
        V_CR = "(c+) fine-tune + replay"

        # Câu hỏi "chạy thử" (§11). Cùng bộ này cho MỌI phiên bản, sinh ngay trong lượt
        # nạp model của phiên bản đó — thêm một lần load 4B model chỉ để in 6 câu là
        # ~2 phút GPU cho mỗi phiên bản, không đổi lấy thông tin nào.
        SAMPLE_TICKETS = [
            ("trong miền", "Mình mua tai nghe bluetooth mã đơn DH998877, hộp còn nguyên "
                           "nhưng nghe một bên rất rè. Mình muốn đổi cái khác, gấp nhé!"),
            ("hai ý xung đột", "Đơn ND221100 giao chậm 5 ngày rồi mà bàn ủi hơi nước lại "
                               "bị móp. Shop hoàn tiền cho mình luôn được không?"),
            ("ngoài miền", "Shop có bán cà phê hạt Arabica không, và giao tới Đà Lạt mất "
                           "bao lâu?"),
        ]
        # Ba câu KHÔNG liên quan tới triage: đây là chỗ quên thảm hoạ (§14.3) hiện ra
        # bằng mắt thường, không cần đọc bảng điểm.
        SAMPLE_GENERAL = [
            "Thủ đô của Nhật Bản là thành phố nào?",
            "Giải thích ngắn gọn vì sao trời có mưa.",
            "Viết một câu cảm ơn khách hàng đã mua hàng.",
        ]

        SCORES, PREDS, STATS, HOLD = {}, {}, {}, {}

        def eval_pass(model, tok, version, system, *, full=True):
            \"\"\"Một phiên bản, bốn nhóm điểm, cùng một đường đi.

            `full=False` chỉ chấm target+format: ba contrast ở §8 không cần nhóm
            regression (phán quyết được chấm trên `correct`), và mỗi lượt sinh thêm là
            ~3 phút GPU cho mỗi adapter.
            \"\"\"
            tp, st = generate.generate_measured(
                model, tok, [r["input"] for r in target], system=system,
                max_new_tokens=MAX_NEW, batch_size=EVAL_BATCH, label=f"{version}/target")
            rp, hp, sp, gp = [], [], [], []
            if full:
                rp, _ = generate.generate_measured(
                    model, tok, [r["instruction"] for r in regression], system=None,
                    max_new_tokens=96, batch_size=EVAL_BATCH, warmup=False,
                    label=f"{version}/regression")
                hp, _ = generate.generate_measured(
                    model, tok, [r["input"] for r in holdout], system=system,
                    max_new_tokens=MAX_NEW, batch_size=EVAL_BATCH, warmup=False,
                    label=f"{version}/holdout")
                sp, _ = generate.generate_measured(
                    model, tok, [t for _, t in SAMPLE_TICKETS], system=system,
                    max_new_tokens=MAX_NEW, batch_size=len(SAMPLE_TICKETS),
                    warmup=False, progress=False, label=f"{version}/samples")
                gp, _ = generate.generate_measured(
                    model, tok, SAMPLE_GENERAL, system=None, max_new_tokens=96,
                    batch_size=len(SAMPLE_GENERAL), warmup=False, progress=False,
                    label=f"{version}/general")

            trace = sum(ev.valid_reasoning_trace(p) for p in tp) / max(1, len(tp))
            sc = ev.score_version(
                target, tp, regression if full else [], rp, st["mean_latency_ms"],
                extra={"generation": st, "valid_trace_rate": round(trace, 4),
                       "system_prompt": (system or "")[:60], "scored_regression": full})
            SCORES[version] = sc
            STATS[version] = st
            PREDS[version] = {"target": tp, "regression": rp, "holdout": hp,
                              "samples": sp, "general": gp}
            if hp:
                HOLD[version] = {
                    "target": round(sum(ev.triage_field_accuracy(p, r["label"])
                                        for p, r in zip(hp, holdout)) / len(holdout), 4),
                    "fields": ev.field_accuracy(hp, holdout), "n": len(holdout)}
            line = (f"{version:<30} target={sc.target:.3f}  format={sc.format:.3f}  "
                    f"{sc.latency_ms:7.0f} ms/mẫu  {st['prompt_tokens_mean']:5.0f} tok prompt  "
                    f"{st['new_tokens_mean']:5.0f} tok ra")
            if full:
                line += f"  regression={sc.regression:.3f}  holdout={HOLD[version]['target']:.3f}"
            print("\\n" + line + "\\n")
            return sc
    """)
    code("""
        import time

        base, tok = generate.load_base(TIER)
        generate.free_memory()
        print(json.dumps(modeling.layer_type_summary(base.config), ensure_ascii=False,
                         indent=2))

        t0 = time.perf_counter()
        eval_pass(base, tok, V_A, C.NAIVE_PROMPT)
        eval_pass(base, tok, V_B, C.OPTIMIZED_PROMPT)
        print(f"hai baseline: {time.perf_counter() - t0:.0f}s")
    """)

    md("""
    ### 6.1 Đóng băng — từ đây không sửa tập eval, không sửa prompt (b)
    """)
    code("""
        import hashlib

        frozen = {
            "tier": TIER.name,
            "model": TIER.model_id,
            "baseline_a": SCORES[V_A].as_dict(),
            "baseline_b": SCORES[V_B].as_dict(),
            "optimized_prompt_sha": hashlib.sha256(
                C.OPTIMIZED_PROMPT.encode()).hexdigest()[:16],
            "n_target": len(target),
            "n_regression": len(regression),
            "n_holdout": len(holdout),
            "eval_limit": LIMIT or None,
            "smoke_mode": bool(LIMIT) or TIER.name == "SMOKE",
            "holdout": {k: v for k, v in HOLD.items()},
            "stats": {V_A: STATS[V_A], V_B: STATS[V_B]},
        }
        report.write_json(frozen, "baselines_frozen.json", results_dir=RESULTS)

        print(report.markdown_table(ev.comparison_table(
            {V_A: SCORES[V_A], V_B: SCORES[V_B]})))
        d = SCORES[V_B].target - SCORES[V_A].target
        print(f"\\nprompt tối ưu đổi được {d:+.3f} target với "
              f"{STATS[V_B]['prompt_tokens_mean'] - STATS[V_A]['prompt_tokens_mean']:+.0f} "
              f"token prompt mỗi request — đây là mốc (b), và §10 sẽ tính giá của nó.")
        if d <= 0:
            print("⚠ (b) KHÔNG hơn (a). Prompt 'tối ưu' của bạn chưa đủ tốt — sửa NGAY "
                  "bây giờ, trước khi train. Thắng một mốc yếu là thắng giả.")

        del base
        generate.free_memory()
    """)


def _stage3_train() -> None:
    md("""
    ## 7. Huấn luyện — cấu hình ĐÚNG (deck §10), **một hàm cho mọi run**

    | Nút | Giá trị | Deck |
    |---|---|---|
    | `target_modules` | **toàn bộ linear của text decoder** (không gồm vision tower) | §10.2 |
    | `learning_rate` | **1e-4 ≈ 10× LR full-FT** | §10.3 |
    | batch hiệu dụng | **16 < 32** | §10.4 |
    | `alpha` | `2r` | §9.3 |
    | `packing` | **tắt** — ta nạp nhãn đã token hoá sẵn, packing sẽ phá mask | §13.3 |
    | `padding_free` | chỉ khi có FlashAttention **và** batch ≥ 2 → trên T4 là **không** | §13.3 |
    | `loss_type` | `chunked_nll` | §15 |

    Mọi run trong notebook này — `correct`, `correct_replay`, và ba contrast ở §8 — đi qua
    **đúng một** hàm `train_one()` với **đúng một** ngân sách step. Hai bản sao của cùng
    một vòng train là cách một contrast âm thầm được train dài hơn baseline mà nó bị đem
    ra so; đó là bug thật của pipeline gốc (`max_steps=60` ở NB4 so với 30 step ở NB3).

    **Bản Kaggle thêm `eval_dataset`.** Split `val` đã tồn tại từ §5 và trong repo không
    ai dùng: 30 step trên ~225 mẫu ở LR 1e-4 là *đang thiếu* hay *đã quá* khớp thì cho
    tới giờ vẫn là một lời phỏng đoán. Có loss held-out, câu đó thành số đo — và
    `train.losses_from_log()` ghi lại cả hai đường cong.
    """)
    code("""
        import time
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        ADAPTERS = WORK / "adapters"

        # Ngân sách step, dẫn ra MỘT lần từ corpus của `correct`. `correct_replay` KHÔNG
        # được tự tính từ độ dài corpus của nó: trộn replay làm corpus dài ra, và nếu để
        # nó tự tính thì nó sẽ được train nhiều step hơn — thành hai biến thay vì một.
        _rows_correct = data.to_training_dataset(tok, train_rows,
                                                 max_length=TIER.max_length,
                                                 mask_mode=MASK_MODE)
        STEPS = train.planned_steps(len(_rows_correct), TIER, C.training_epochs())
        print(f"epochs={C.training_epochs()}  ·  {len(_rows_correct)} mẫu  ·  "
              f"batch hiệu dụng {TIER.effective_batch}  ->  {STEPS} optimizer step")
        print("Mọi run được chấm điểm dùng ĐÚNG con số này.")


        def train_one(key, train_records, steps, *, val_records=None,
                      replay_fraction=None):
            spec = C.SPECS[key]
            adir = ADAPTERS / key
            if (adir / "adapter_model.safetensors").exists() and not FORCE_RETRAIN:
                print(f"bỏ qua {key}: đã có {adir} (FORCE_RETRAIN=True để train lại)")
                return None

            print("=" * 78)
            print(f"RUN {key}: {spec.label}\\n     {spec.teaches}")
            model, tk = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
            targets = modeling.resolve_target_modules(model, spec.target)
            if spec.r is None:
                # attn_only: giải ra rank để BẰNG ngân sách tham số của `correct`. So
                # q,v @ r=16 với all-linear @ r=16 là so ngân sách, không phải so vị trí.
                base_t = modeling.resolve_target_modules(model, "text-linear")
                spec = spec.resolved(
                    modeling.matched_rank(model, base_t, C.SPECS["correct"].r, targets))
                print(f"  rank khớp ngân sách: r={spec.r}  alpha={spec.alpha}")
            trainable = modeling.count_lora_params(model, targets, spec.r)
            print(f"  placement={spec.target}  modules={len(targets)}  "
                  f"trainable ≈ {trainable/1e6:.2f} M  lr={spec.lr}  4bit={spec.load_in_4bit}")

            ds = Dataset.from_list(data.to_training_dataset(
                tk, train_records, max_length=TIER.max_length, mask_mode=MASK_MODE))
            val_ds = None
            if val_records:
                val_ds = Dataset.from_list(data.to_training_dataset(
                    tk, val_records, max_length=TIER.max_length, mask_mode=MASK_MODE))
            sup = sum(sum(1 for x in r["labels"] if x != data.IGNORE_INDEX) for r in ds)
            tot = sum(len(r["labels"]) for r in ds)
            assert 0 < sup < tot, "mask không che gì hoặc che tất cả — quay lại §5"
            print(f"  train_ds={len(ds)}  val_ds={0 if val_ds is None else len(val_ds)}  "
                  f"supervised {sup}/{tot} ({sup/tot:.1%})")

            want = train.sft_config_kwargs(
                TIER, spec, str(adir), max_steps=steps, mask_mode=MASK_MODE, seed=SEED,
                eval_steps=(max(1, steps // 6) if val_ds is not None else None))
            sft_kwargs, dropped = train.filter_kwargs(SFTConfig, want,
                                                      label=f"SFTConfig[{key}]")
            if dropped:
                print("  ⚠ TRL không nhận:", dropped)
            lora_kwargs, _ = train.filter_kwargs(
                LoraConfig, train.lora_config_kwargs(spec, targets),
                label=f"LoraConfig[{key}]")

            generate.free_memory()
            trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                                 train_dataset=ds, eval_dataset=val_ds,
                                 processing_class=tk,
                                 peft_config=LoraConfig(**lora_kwargs))
            # TRL trả về trọng số LoRA ở bf16 bất kể thiết bị; GradScaler của fp16 không
            # có kernel BFloat16 nên run chết ở step 0. No-op trên bf16/fp32.
            fix = train.align_trainable_precision(trainer.model)
            if fix.get("recast"):
                print(f"  precision fix: {fix['recast']}/{fix['trainable_tensors']} "
                      f"tensor bf16 -> fp32 cho GradScaler fp16")

            t0 = time.perf_counter()
            trainer.train()
            elapsed = time.perf_counter() - t0

            # LƯU TRƯỚC KHI LÀM GÌ KHÁC. Chấm điểm có thể OOM; adapter thì đã an toàn.
            trainer.model.save_pretrained(adir)
            tk.save_pretrained(adir)
            print(f"  saved -> {adir}")

            curves = train.losses_from_log(trainer.state.log_history)
            row = train.summarize_run(
                spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb(),
                max_steps=steps, n_train_examples=len(ds),
                final_train_loss=curves["final_train_loss"],
                final_eval_loss=curves["final_eval_loss"],
                replay_fraction=replay_fraction)
            row["mask_mode"] = MASK_MODE
            row["teaches"] = spec.teaches
            report.append_row(row, results_dir=RESULTS)
            report.write_json(curves, f"curves_{key}.json", results_dir=RESULTS)
            print(f"  {elapsed:.0f}s  train_loss={curves['final_train_loss']}  "
                  f"eval_loss={curves['final_eval_loss']}  "
                  f"eval tăng sau đáy={curves['eval_rose_after_min']}")

            del trainer, model
            generate.free_memory()
            return row
    """)
    code("""
        row_correct = train_one("correct", train_rows, STEPS, val_records=val_rows)
        print(json.dumps(row_correct, ensure_ascii=False, indent=2)
              if row_correct else "(dùng adapter đã có từ lần chạy trước)")
    """)

    md("""
    ### 7b. `correct_replay` — cùng LoRA, **khác dữ liệu** (deck §14.3)

    Đây là đối chứng mà pipeline gốc còn thiếu, nhắm vào đúng khuyết điểm mà chính repo
    đo được và để mở: `regression` **0.644 → 0.067**. Mọi nút LoRA giống `correct` từng
    con số; biến duy nhất là corpus, và số step vẫn là `STEPS` của `correct`.

    Nếu run này giữ được regression mà không mất target, thì kết luận của lab không còn
    là *"fine-tune làm model quên"* mà là *"corpus chỉ-một-tác-vụ làm model quên, và
    deck §14.3 chữa được bằng 5% dữ liệu"* — hai câu khác nhau hoàn toàn về hành động.
    """)
    code("""
        row_replay = None
        if RUN_REPLAY:
            row_replay = train_one("correct_replay", mixed_rows, STEPS,
                                   val_records=val_rows,
                                   replay_fraction=C.replay_fraction())
            print(json.dumps(row_replay, ensure_ascii=False, indent=2)
                  if row_replay else "(dùng adapter đã có)")
        else:
            print("RUN_REPLAY=False — bỏ §7b. Phán quyết sẽ chỉ có `correct`.")
    """)


def _stage4_contrasts() -> None:
    md("""
    ## 8. Ba cấu hình SAI — cùng số step, mỗi lần đổi đúng một biến

    Bản Day-21 *cũ* lấy "quét rank r=8/16/64", gắn adapter vào `q_proj,v_proj`, và chấm
    bằng perplexity làm thí nghiệm trung tâm. Deck hiện tại gọi đúng ba thứ đó là **Lỗi
    #1, #2, #3** (§10.2–§10.4). Phần này chạy lại thí nghiệm cũ **như một đối chứng**.

    | Run | Đổi gì | Kỳ vọng |
    |---|---|---|
    | `attn_only` | chỉ q,v — **rank nâng lên cho BẰNG số tham số** | thua `correct` |
    | `wrong_lr` | LR thang full-FT (÷10) | loss gần như phẳng |
    | `qlora` | 4-bit thay 16-bit | nhẹ hơn, chất lượng ? |

    Bảng dưới in `final_train_loss`. **Đừng xếp hạng bằng cột đó** — làm vậy chính là Lỗi
    #3. §9 chấm cả ba adapter này trên tập target, bằng đúng thang đo đã dùng cho
    `correct`; nếu thứ tự hai bảng khác nhau thì bạn vừa tự tay đo được lý do lab cũ kết
    luận sai.
    """)
    code("""
        if RUN_CONTRASTS:
            _m, _t = generate.load_base(TIER)
            print(report.markdown_table(modeling.describe_placement(_m, C.SPECS["correct"].r)))
            del _m
            generate.free_memory()

            for key in C.CONTRAST_KEYS:
                train_one(key, train_rows, STEPS, val_records=val_rows)
        else:
            print("RUN_CONTRASTS=False — bỏ §8. §9 sẽ không có phần giải phẫu.")

        _seen = {}
        for r in report.read_rows("runs.csv", results_dir=RESULTS):
            if r.get("run"):
                _seen[r["run"]] = r          # dòng cuối của mỗi key là dòng hiện hành
        runs_rows = [_seen[k] for k in C.GRADED_KEYS if k in _seen]
        cols = ["run", "r", "lora_alpha", "learning_rate", "load_in_4bit",
                "trainable_params", "max_steps", "n_train_examples", "replay_fraction",
                "final_train_loss", "final_eval_loss", "train_seconds", "peak_vram_gb"]
        print()
        print(report.markdown_table(runs_rows, cols))

        budgets = {r["run"]: r.get("max_steps") for r in runs_rows}
        if len(set(budgets.values())) > 1:
            print(f"\\n⚠ ngân sách step KHÔNG bằng nhau: {budgets} — bảng so sánh này "
                  "đang đo độ dài train, không phải cấu hình.")
        else:
            print(f"\\ncả {len(budgets)} run cùng {next(iter(budgets.values()))} step — "
                  "so sánh công bằng.")
    """)


def _stage5_verdict() -> None:
    md("""
    ## 9. Bốn nhóm điểm cho MỌI phiên bản, và **phán quyết**

    Fine-tune được chấm với **prompt ngây thơ**, không phải prompt (b): cái mà fine-tune
    mua cho bạn chính là "hành vi đã chuyển vào trọng số nên prompt co lại được". Đó cũng
    là điều làm phần cost ở §10 có ý nghĩa.

    (Và nó chỉ đúng vì §5.3 đã kiểm: chuỗi mà evaluation gửi đi là **tiền tố** của chuỗi
    đã train. Khi hai thứ đó lệch nhau — lỗi F-31 — mọi adapter ra `target = 0.000`.)
    """)
    code("""
        from peft import PeftModel

        def score_adapter(key, version, *, full=True):
            spec = C.SPECS[key]
            adir = ADAPTERS / key
            if not (adir / "adapter_model.safetensors").exists():
                print(f"bỏ qua {version}: chưa có {adir}")
                return None
            # load_in_4bit phải KHỚP lúc train: chấm adapter `qlora` trên base 16-bit là
            # đo độ lệch base/adapter rồi gọi nó là "giá của QLoRA".
            model, tk = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
            model = PeftModel.from_pretrained(model, str(adir))
            model.eval()
            sc = eval_pass(model, tk, version, C.NAIVE_PROMPT, full=full)
            del model
            generate.free_memory()
            return sc

        score_adapter("correct", V_C)
        if RUN_REPLAY and (ADAPTERS / "correct_replay" / "adapter_model.safetensors").exists():
            score_adapter("correct_replay", V_CR)
    """)

    md("""
    ### 9.1 Bảng bốn nhóm + biểu đồ, cho mọi phiên bản
    """)
    code("""
        order = [v for v in (V_A, V_B, V_C, V_CR) if v in SCORES]
        table = ev.comparison_table({v: SCORES[v] for v in order})
        print(report.markdown_table(table))

        for group in ("target", "regression", "format"):
            print(f"\\n{group}:")
            print(report.hbar({v: getattr(SCORES[v], group) for v in order}, vmax=1.0))
        print("\\nlatency (ms/mẫu, thấp hơn là tốt hơn):")
        print(report.hbar({v: SCORES[v].latency_ms for v in order}, fmt="{:.0f}"))

        print("\\nđộ chính xác từng field trên tập target:")
        field_tbl = [{"version": v, **{k: (None if x is None else round(x, 3))
                                       for k, x in SCORES[v].extra["fields"].items()}}
                     for v in order]
        print(report.markdown_table(field_tbl))

        if HOLD:
            print("\\nholdout_secret (20 ticket chưa dùng ở bất kỳ đâu):")
            print(report.hbar({v: HOLD[v]["target"] for v in order if v in HOLD}, vmax=1.0))
    """)

    md("""
    ### 9.2 Cổng hồi quy — chấm **cả hai** bản fine-tune so với (b)

    Đạt = vượt (b) ở `target` **và** không tụt `regression` quá 0.02. Một phán quyết
    **FAILED được chấm điểm đầy đủ** nếu bạn phân tích trung thực; cái bị trừ điểm là nới
    ngưỡng, làm yếu prompt (b), hay đổi tập eval sau khi đã thấy kết quả.
    """)
    code("""
        verdicts = {}
        for v in (V_C, V_CR):
            if v not in SCORES:
                continue
            verd = ev.regression_gate(SCORES[v], SCORES[V_B])
            verdicts[v] = verd
            print("=" * 78)
            print(f"{v}: {'PASSED' if verd.passed else 'FAILED'}")
            for r in verd.reasons:
                print("  -", r)

        winner = None
        cands = [v for v in (V_C, V_CR) if v in verdicts]
        passed = [v for v in cands if verdicts[v].passed]
        if passed:
            winner = max(passed, key=lambda v: SCORES[v].target)
        elif cands:
            # Không ai qua cổng: "người thắng" là bản đánh đổi ít nhất, và §10/§13 nói
            # thẳng rằng nó KHÔNG qua cổng.
            winner = max(cands, key=lambda v: SCORES[v].target + SCORES[v].regression)
        print("\\n" + "=" * 78)
        print(f"bản chọn để so cost/serve: {winner}  "
              f"(qua cổng: {bool(passed)})")

        if V_C in SCORES and V_CR in SCORES:
            dt = SCORES[V_CR].target - SCORES[V_C].target
            dr = SCORES[V_CR].regression - SCORES[V_C].regression
            print(f"\\nreplay mix {C.replay_fraction():.0%} đổi được: regression {dr:+.3f}, "
                  f"target {dt:+.3f} — deck §14.3, một biến duy nhất là DỮ LIỆU.")
    """)

    md("""
    ### 9.3 Giải phẫu: ba cấu hình sai, chấm trên **thang đo tác vụ**
    """)
    code("""
        autopsy = [{"run": k, "version": v, "target": round(SCORES[v].target, 4),
                    "format": round(SCORES[v].format, 4),
                    "latency_ms": round(SCORES[v].latency_ms, 1), "n": SCORES[v].n}
                   for k, v in (("correct", V_C), ("correct_replay", V_CR))
                   if v in SCORES]
        for key in C.CONTRAST_KEYS:
            sc = score_adapter(key, key, full=False)
            if sc is None:
                continue
            autopsy.append({"run": key, "version": key, "target": round(sc.target, 4),
                            "format": round(sc.format, 4),
                            "latency_ms": round(sc.latency_ms, 1), "n": sc.n})
        print()
        print(report.markdown_table(autopsy, ["run", "target", "format", "latency_ms", "n"]))
        report.write_json(autopsy, "autopsy.json", results_dir=RESULTS)

        loss_order = [r["run"] for r in sorted(
            (r for r in runs_rows if str(r.get("final_train_loss", "")).strip()),
            key=lambda r: float(r["final_train_loss"]))]
        task_order = [r["run"] for r in sorted(autopsy, key=lambda r: -r["target"])]
        print(f"\\nxếp theo train loss (thấp->cao):  {loss_order}")
        print(f"xếp theo điểm target (cao->thấp): {task_order}")
        if loss_order and task_order and loss_order[0] != task_order[0]:
            print("\\n→ Hai thứ tự KHÁC nhau. Đây chính là Lỗi #3 hiện ra thành số: xếp "
                  "hạng bằng loss huấn luyện sẽ chọn sai run.")
    """)

    md("""
    ### 9.4 Định tính — **bắt buộc có cả ca THUA**

    Chọn ca thắng thôi là cherry-pick và bị trừ ở mục Evaluation Quality. Ba ca tệ nhất
    ở dưới là ba ca phải xuất hiện trong REPORT.md.
    """)
    code("""
        assert winner, ("chưa có adapter nào được chấm — chạy §7 (và §9 cell đầu) trước "
                        "khi đọc phần định tính.")
        qual = []
        for i, (p, r) in enumerate(zip(PREDS[winner]["target"], target)):
            base_p = PREDS[V_B]["target"][i]
            qual.append({
                "i": i,
                "ticket": r["input"][:64],
                "ft": round(ev.triage_field_accuracy(p, r["label"]), 2),
                "b": round(ev.triage_field_accuracy(base_p, r["label"]), 2),
                "ft_pred": p.replace("\\n", " ")[:80],
                "b_pred": base_p.replace("\\n", " ")[:80],
            })
        qual.sort(key=lambda x: (x["ft"] - x["b"], x["ft"]))
        cols_q = ["i", "ticket", "b", "ft", "ft_pred"]
        print("--- 3 ca fine-tune THUA/kém nhất so với (b) ---")
        print(report.markdown_table(qual[:3], cols_q))
        print("\\n--- 3 ca fine-tune THẮNG rõ nhất ---")
        print(report.markdown_table(qual[-3:], cols_q))
        report.write_json(qual, "qualitative.json", results_dir=RESULTS)

        # Nhóm regression cũng cần một ca đọc bằng mắt: điểm keyword_recall = 0 có thể là
        # "trả lời sai" hoặc là "trả lời bằng JSON triage" — hai chuyện khác nhau, và chỉ
        # cái thứ hai mới là quên thảm hoạ.
        print("\\n--- regression: cùng một câu hỏi, ba phiên bản ---")
        q0 = regression[0]
        print(f"hỏi: {q0['instruction']}   (keywords={q0['keywords']})")
        for v in order:
            if PREDS[v]["regression"]:
                print(f"  [{v}] -> {PREDS[v]['regression'][0].replace(chr(10), ' ')[:150]}")
    """)
    code("""
        payload = {
            "comparison": table,
            "verdict": verdicts[winner].as_dict() if winner in verdicts else None,
            "verdicts": {v: verdicts[v].as_dict() for v in verdicts},
            "winner": winner,
            "passed_gate": winner in passed,
            "regression_tolerance": ev.REGRESSION_TOLERANCE,
            "holdout": HOLD,
            "field_accuracy": {v: SCORES[v].extra["fields"] for v in order},
            "valid_trace_rate": {v: SCORES[v].extra.get("valid_trace_rate") for v in order},
            "n_target": len(target), "n_regression": len(regression),
            "smoke_mode": bool(LIMIT) or TIER.name == "SMOKE",
        }
        report.write_json(payload, "verdict.json", results_dir=RESULTS)
        print(json.dumps({k: payload[k] for k in
                          ("winner", "passed_gate", "verdict", "smoke_mode")},
                         ensure_ascii=False, indent=2))
    """)


def _stage6_cost() -> None:
    md("""
    ## 10. Latency và **cost** — nửa còn lại của câu "có nên fine-tune?"

    §9 chọn ra bản thắng về chất lượng. Một team còn phải trả lời câu thứ hai: *nó tốn
    bao nhiêu, và bao lâu thì hoàn vốn?* Fine-tune trả trước bằng GPU-giờ và mua lại
    **prompt ngắn hơn** cùng **output ngắn hơn** trên mọi request, mãi mãi.

    Tất cả số ở đây là **phép tính trên số đo**: `mean_latency_ms` (đã trừ warm-up),
    `prompt_tokens_mean`, `new_tokens_mean` đều do `generate_measured()` đo ở §6/§9. Giá
    thì là **giả định có nguồn** và là tham số — sửa ở ô CONFIG rồi chạy lại ô này.

    Ba cảnh báo đi kèm, nói một lần và nói thật:
    1. `mean_latency_ms` là **throughput theo batch** ở batch size của vòng chấm. Phục vụ
       một ticket lẻ thì chậm hơn; batch 32 thì nhanh hơn. So sánh **giữa các phiên bản**
       là công bằng vì mọi phiên bản đo cùng batch size — nhưng đây không phải báo giá prod.
    2. Cost self-host giả định GPU **luôn có việc**. GPU thuê mà để không vẫn tính tiền,
       nên đây là mức sàn chỉ đạt được khi hàng đợi luôn đầy.
    3. Không có dòng nào tính tiền công engineer, và bản fine-tune đã tiêu một ít.
    """)
    code("""
        key_of = {V_C: "correct", V_CR: "correct_replay"}
        train_seconds = None
        if winner in key_of and key_of[winner] in _seen:
            raw = _seen[key_of[winner]].get("train_seconds")
            train_seconds = float(raw) if str(raw).strip() else None

        comp = cost.compare(
            {v: STATS[v] for v in order},
            train_seconds=train_seconds, baseline=V_B, winner=winner,
            gpu_hourly_usd=GPU_HOURLY_USD,
            input_usd_per_mtok=API_IN_USD_MTOK,
            output_usd_per_mtok=API_OUT_USD_MTOK)

        print(report.markdown_table(comp["rows"]))
        print()
        print(json.dumps(comp["delta"], ensure_ascii=False, indent=2))
        print("\\n" + cost.verdict_line(comp))

        # Chi phí thật của cả buổi lab, không chỉ của run thắng: hoá đơn GPU tính theo
        # số giờ đã chạy, kể cả những run chỉ để chứng minh cấu hình sai là sai.
        all_seconds = sum(float(r["train_seconds"]) for r in runs_rows
                          if str(r.get("train_seconds", "")).strip())
        print(f"\\ntổng thời gian train của MỌI run: {all_seconds/60:.0f} phút "
              f"= ${cost.training_usd(all_seconds, GPU_HOURLY_USD):.3f} "
              f"(giá {GPU_HOURLY_USD}/GPU-giờ)")
        if winner not in passed:
            print("⚠ Bản này KHÔNG qua cổng hồi quy ở §9. Con số break-even ở trên chỉ "
                  "trả lời 'rẻ hơn khi nào', không trả lời 'có nên ship'.")
        report.write_json({**comp, "all_runs_train_seconds": round(all_seconds, 1),
                           "all_runs_train_usd": round(cost.training_usd(all_seconds,
                                                                         GPU_HOURLY_USD), 4),
                           "winner": winner, "winner_passed_gate": winner in passed},
                          "cost.json", results_dir=RESULTS)
    """)


def _stage7_samples() -> None:
    md("""
    ## 11. Chạy thử — cùng câu hỏi, mọi phiên bản, đặt cạnh nhau

    Bảng điểm nói *bao nhiêu*; phần này nói *khác nhau ở đâu*. Sáu câu đã được sinh sẵn
    trong chính lượt nạp model của mỗi phiên bản ở §6/§9, nên ở đây không tốn thêm GPU.

    Ba ticket: một ca thẳng, một ca **hai ý xung đột** (giao chậm *và* hàng lỗi *và* xin
    hoàn tiền — chỗ mà `intent` buộc phải chọn), và một ca **ngoài miền** (không phải
    ticket khiếu nại; model nên làm gì với nó?).

    Ba câu tổng quát: đây là chỗ **quên thảm hoạ** hiện ra bằng mắt. Nếu bản fine-tune
    trả lời "Thủ đô của Nhật Bản là thành phố nào?" bằng một object JSON triage, bạn
    không cần bảng điểm nào để biết đã xảy ra chuyện gì.
    """)
    code("""
        for i, (kind, ticket) in enumerate(SAMPLE_TICKETS):
            print("=" * 78)
            print(f"TICKET [{kind}] {ticket}")
            for v in order:
                out = PREDS[v]["samples"]
                if out:
                    print(f"\\n  [{v}]\\n    {out[i].replace(chr(10), ' ')[:300]}")
            print()
    """)
    code("""
        for i, q in enumerate(SAMPLE_GENERAL):
            print("=" * 78)
            print(f"CÂU HỎI THƯỜNG NGÀY: {q}")
            for v in order:
                out = PREDS[v]["general"]
                if out:
                    print(f"\\n  [{v}]\\n    {out[i].replace(chr(10), ' ')[:300]}")
            print()

        report.write_json(
            {"tickets": [{"kind": k, "prompt": t,
                          "answers": {v: (PREDS[v]["samples"][i] if PREDS[v]["samples"] else None)
                                      for v in order}}
                         for i, (k, t) in enumerate(SAMPLE_TICKETS)],
             "general": [{"prompt": q,
                          "answers": {v: (PREDS[v]["general"][i] if PREDS[v]["general"] else None)
                                      for v in order}}
                         for i, q in enumerate(SAMPLE_GENERAL)]},
            "samples.json", results_dir=RESULTS)
    """)


def _stage8_merge() -> None:
    md("""
    ## 12. (Tuỳ chọn) Merge, kiểm chứng sau merge, và hoán đổi adapter — deck §18

    Hai đường triển khai:
    * **Merge** — `W = W₀ + (α/r)·BA`; đồ thị phục vụ giống hệt base → **không** overhead.
    * **Giữ riêng** — một base trong VRAM, nhiều adapter, chọn theo từng request.

    `RUN_MERGE=False` theo mặc định vì merge ghi ~8 GB xuống `/kaggle/working` và Kaggle
    giới hạn 20 GB output. Phần hoán đổi adapter thì rẻ và luôn chạy.

    **Assert bắt buộc:** merge là phép toán chính xác trên giấy, nhưng dtype lúc gộp có
    thể làm tụt điểm. Tụt thì **đừng deploy** — đi tìm nguyên nhân.
    """)
    code("""
        from peft import PeftModel

        base_key = key_of.get(winner, "correct")
        model, tk = generate.load_base(TIER)
        model = PeftModel.from_pretrained(model, str(ADAPTERS / base_key),
                                         adapter_name=base_key)
        loaded = [base_key]
        for extra in [k for k in C.GRADED_KEYS if k != base_key]:
            d = ADAPTERS / extra
            if (d / "adapter_model.safetensors").exists():
                model.load_adapter(str(d), adapter_name=extra)
                loaded.append(extra)
        print("một base trong VRAM, các adapter đã nạp:", loaded)
        print(f"VRAM đang dùng: {generate.peak_vram_gb():.2f} GB")

        ticket = target[0]["input"]
        print(f"\\nticket: {ticket[:120]}\\nnhãn đúng: {target[0]['label']}\\n")
        for name in loaded:
            model.set_adapter(name)
            out, _ = generate.generate_measured(model, tk, [ticket],
                                                system=C.NAIVE_PROMPT, max_new_tokens=96,
                                                batch_size=1, warmup=False, progress=False)
            print(f"  [{name}] -> {out[0].replace(chr(10), ' ')[:160]}")
        model.set_adapter(base_key)
    """)
    code("""
        merge_check = None
        if RUN_MERGE:
            n_chk = min(len(target), 20)
            chk = target[:n_chk]
            pre, _ = generate.generate_measured(
                model, tk, [r["input"] for r in chk], system=C.NAIVE_PROMPT,
                max_new_tokens=MAX_NEW, batch_size=EVAL_BATCH, label="pre-merge")
            before = sum(ev.triage_field_accuracy(p, r["label"])
                         for p, r in zip(pre, chk)) / n_chk

            merged = model.merge_and_unload()
            post, _ = generate.generate_measured(
                merged, tk, [r["input"] for r in chk], system=C.NAIVE_PROMPT,
                max_new_tokens=MAX_NEW, batch_size=EVAL_BATCH, label="post-merge")
            after = sum(ev.triage_field_accuracy(p, r["label"])
                        for p, r in zip(post, chk)) / n_chk
            TOL = 0.01
            merge_check = {"adapter": base_key, "before_merge": round(before, 4),
                           "after_merge": round(after, 4), "delta": round(after - before, 4),
                           "tolerance": TOL, "n": n_chk}
            print(json.dumps(merge_check, ensure_ascii=False, indent=2))
            assert after - before >= -TOL, (
                f"điểm TỤT {before - after:.4f} sau merge (ngưỡng {TOL}). Kiểm tra dtype "
                "lúc gộp; với DoRA cần PEFT >= 0.10 để gộp đúng magnitude vector (§18).")
            merged.save_pretrained(ADAPTERS / "merged")
            tk.save_pretrained(ADAPTERS / "merged")
            report.write_json(merge_check, "merge_check.json", results_dir=RESULTS)
            del merged
        else:
            print("RUN_MERGE=False — bỏ phần merge (ghi ~8 GB). Phần hot-swap ở trên đã "
                  "chứng minh đường triển khai 'một base, nhiều adapter'.")

        del model
        generate.free_memory()
    """)


def _stage9_gate() -> None:
    md("""
    ## 13. Cổng kiểm tra trước khi nộp

    Ô này là `scripts/verify.py` của repo, viết thẳng vào notebook (luật: không clone).
    Nó **không** chấm điểm bài — nó kiểm những điều kiện mà nếu sai thì điểm không có
    nghĩa gì:

    | kiểm | hỏng thì sao |
    |---|---|
    | corpus khớp checksum | sửa tập eval sau khi thấy điểm → mọi so sánh vô nghĩa |
    | không ở SMOKE mode | số của tier 0.8B / EVAL_LIMIT không phải bài nộp |
    | mask đúng | supervise cả câu hỏi → model học đoán ticket, không học trả lời |
    | prompt train ≡ prompt eval | F-31: mọi adapter chấm 0.000 dù train tốt |
    | replay đã tẩy trùng | replay chứa câu trong tập regression → điểm hồi quy là rò rỉ |
    | mọi run cùng ngân sách step | bảng contrast đo độ dài train, không đo cấu hình |
    | `attn_only` khớp tham số ±5% | contrast placement biến thành contrast dung lượng |
    | có phán quyết + bảng cost | thiếu chính phần kết luận |
    """)
    code("""
        import zipfile

        checks = []
        def chk(name, ok, detail=""):
            checks.append({"check": name, "pass": bool(ok), "detail": detail})
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

        print("=" * 78)
        chk("corpus khớp checksum (CRLF-tolerant)", not integrity["drift"],
            f"{len(integrity['files'])} file, drift={integrity['drift'] or 'không'}")
        chk("không chạy ở SMOKE mode", not SMOKE,
            f"tier={TIER.name}, EVAL_LIMIT={LIMIT or 'FULL'}")
        chk("eval target dùng đủ bộ", len(target) >= 50, f"n={len(target)}")
        chk("eval regression dùng đủ bộ", len(regression) >= 15, f"n={len(regression)}")
        chk("holdout được chấm", bool(HOLD), f"n={len(holdout)}, versions={len(HOLD)}")

        chk("mask: câu trả lời nằm trong loss", proof["answer_is_supervised"])
        chk("mask: câu hỏi bị che", proof["question_is_masked"],
            f"supervised_fraction={proof['supervised_fraction']:.3f}")
        chk("prompt train ≡ prompt eval (F-31)", align["all_aligned"],
            f"n_checked={align['n_checked']}")
        chk("replay đã tẩy trùng", decon["exact_collisions"] == 0
            and decon["worst_jaccard"] <= decon["jaccard_threshold"],
            f"worst_jaccard={decon['worst_jaccard']} (ngưỡng {decon['jaccard_threshold']})")
    """)
    code("""
        budget_vals = {r["run"]: r.get("max_steps") for r in runs_rows}
        chk("mọi run cùng ngân sách optimizer step", len(set(budget_vals.values())) == 1,
            str(budget_vals))

        def _p(row, k):
            v = str(row.get(k, "")).strip()
            return float(v) if v else None
        _cor = _seen.get("correct")
        _att = _seen.get("attn_only")
        if _cor and _att:
            a, b = _p(_cor, "trainable_params"), _p(_att, "trainable_params")
            rel = abs(b - a) / a if a else 1.0
            chk("attn_only khớp tham số ±5%", rel <= 0.05,
                f"{int(a):,} vs {int(b):,} ({rel:+.2%})")
        else:
            chk("attn_only khớp tham số ±5%", False, "chưa chạy §8 (RUN_CONTRASTS=False)")

        chk("có phán quyết cho ít nhất một fine-tune", bool(verdicts),
            ", ".join(f"{v}={'PASS' if verdicts[v].passed else 'FAIL'}" for v in verdicts))
        chk("fine-tune thắng (b) và giữ được năng lực chung", winner in passed,
            (verdicts[winner].reasons[0][:110] if winner in verdicts else "không có"))
        chk("có bảng cost + break-even", "delta" in comp,
            f"self-host saving {comp.get('delta', {}).get('self_host_saving_per_1k')} $/1k")

        REQUIRED = ["template_check.json", "mask_proof.json", "prompt_alignment.json",
                    "token_stats.json", "replay_manifest.json", "baselines_frozen.json",
                    "runs.csv", "curves_correct.json", "autopsy.json", "qualitative.json",
                    "verdict.json", "cost.json", "samples.json"]
        missing = [f for f in REQUIRED if not (RESULTS / f).exists()]
        chk("đủ artifact bắt buộc", not missing, f"thiếu: {missing or 'không'}")

        n_fail = sum(1 for c in checks if not c["pass"])
        print("=" * 78)
        print(f"{len(checks) - n_fail}/{len(checks)} PASS" +
              ("  ✅ sẵn sàng nộp" if not n_fail else f"  ❌ {n_fail} mục cần sửa"))
    """)
    md("""
    ### 13.1 `run_config.json` — chạy lại được thì mới gọi là kết quả

    Một bảng điểm không kèm cấu hình sinh ra nó là một tin nhắn, không phải phép đo.
    """)
    code("""
        run_config = {
            "tier": TIER.name, "model_id": TIER.model_id,
            "precision": device.precision(),
            "max_length": TIER.max_length,
            "per_device_batch": TIER.per_device_batch,
            "grad_accum": TIER.grad_accum,
            "effective_batch": TIER.effective_batch,
            "epochs": C.training_epochs(), "planned_steps": STEPS,
            "mask_mode": MASK_MODE, "seed": SEED,
            "replay_fraction": C.replay_fraction(),
            "eval_limit": LIMIT or None, "smoke_mode": SMOKE,
            "eval_batch": EVAL_BATCH, "max_new_tokens": MAX_NEW,
            "n_train": len(train_rows), "n_val": len(val_rows),
            "n_target": len(target), "n_regression": len(regression),
            "n_holdout": len(holdout),
            "specs": {k: {"target": C.SPECS[k].target, "r": C.SPECS[k].r,
                          "alpha": C.SPECS[k].alpha, "lr": C.SPECS[k].lr,
                          "load_in_4bit": C.SPECS[k].load_in_4bit}
                      for k in C.GRADED_KEYS if k in C.SPECS},
            "prices": {"gpu_hourly_usd": GPU_HOURLY_USD,
                       "api_in_usd_mtok": API_IN_USD_MTOK,
                       "api_out_usd_mtok": API_OUT_USD_MTOK},
            "integrity": integrity,
            "gate": {"checks": checks, "n_fail": n_fail},
            "versions": {},
        }
        import torch, transformers, trl, peft, datasets as _ds
        run_config["versions"] = {"torch": torch.__version__,
                                  "transformers": transformers.__version__,
                                  "trl": trl.__version__, "peft": peft.__version__,
                                  "datasets": _ds.__version__}
        report.write_json(run_config, "run_config.json", results_dir=RESULTS)
        print(json.dumps({k: run_config[k] for k in
                          ("tier", "precision", "epochs", "planned_steps", "mask_mode",
                           "smoke_mode", "versions")}, ensure_ascii=False, indent=2))
    """)
    md("""
    ### 13.2 `REPORT.md` — bản nháp đã điền **số đo thật**

    Ô này không viết hộ phần kết luận. Nó điền mọi con số vào chỗ của nó rồi để lại
    những câu hỏi mà chỉ người chạy trả lời được (vì sao cấu hình sai thua, bạn sẽ ship
    bản nào). Sửa trong `/kaggle/working/REPORT.md` rồi tải về.
    """)
    code("""
        def _f(x, nd=4):
            return "—" if x is None else f"{x:.{nd}f}"

        d = comp.get("delta", {})
        fields_tbl = report.markdown_table(
            [{"version": v, **{k: _f(SCORES[v].extra["fields"].get(k), 3)
                               for k in ev.TRIAGE_KEYS},
              "unparsed": SCORES[v].extra["fields"]["unparsed"]} for v in order])

        lines = [
            "# Day-21 Track-3 — LoRA fine-tune cho phân loại ticket tiếng Việt",
            "",
            f"- model: `{TIER.model_id}` · tier `{TIER.name}` · precision "
            f"`{device.precision()}`",
            f"- LoRA: r={C.SPECS['correct'].r}, alpha={C.SPECS['correct'].alpha}, "
            f"lr={C.SPECS['correct'].lr}, placement=`{C.SPECS['correct'].target}`",
            f"- ngân sách: {STEPS} optimizer step ({C.training_epochs()} epoch, effective "
            f"batch {TIER.effective_batch}) — **giống nhau cho mọi run**",
            f"- mask: `{MASK_MODE}` · {proof['supervised_fraction']:.1%} token vào loss",
            f"- eval: target n={len(target)}, regression n={len(regression)}, "
            f"holdout n={len(holdout)}" + ("  ⚠ **SMOKE MODE**" if SMOKE else ""),
            "",
            "## 1. Bảng bốn nhóm, ba (bốn) phiên bản",
            "",
            report.markdown_table(table),
            "",
            "Cột `target` là accuracy trung bình trên 4 field triage; `regression` là "
            "keyword-recall trên 15 câu hỏi tổng quát; `format` là tỉ lệ output có đúng "
            "4 key; `latency_ms` là throughput theo batch, đã trừ warm-up.",
            "",
            f"- (b) − (a) = **{SCORES[V_B].target - SCORES[V_A].target:+.3f}** target — "
            "phần này prompt engineering làm được, **không** cần fine-tune.",
            f"- {winner} − (b) = **{SCORES[winner].target - SCORES[V_B].target:+.3f}** "
            f"target, **{SCORES[winner].regression - SCORES[V_B].regression:+.3f}** "
            "regression — phần này mới là công của fine-tune.",
            f"- cổng hồi quy (dung sai {ev.REGRESSION_TOLERANCE:.2f}): "
            f"**{'PASS' if winner in passed else 'FAIL'}**",
            "",
        ]
        for v in verdicts:
            lines += [f"**{v}** — {'PASSED' if verdicts[v].passed else 'FAILED'}"] + \\
                     [f"  - {r}" for r in verdicts[v].reasons] + [""]
    """)
    code("""
        lines += [
            "## 2. Autopsy per-field",
            "",
            fields_tbl,
            "",
            "> TODO: field nào tụt nhiều nhất, và vì sao? `unparsed` > 0 nghĩa là mất điểm "
            "vì **định dạng**, không phải vì phân loại sai — hai lỗi khác nhau.",
            "",
            "## 3. Cấu hình sai, chấm trên thang đo tác vụ",
            "",
            report.markdown_table(autopsy),
            "",
            f"Thứ tự theo train-loss: `{' < '.join(loss_order)}`",
            f"Thứ tự theo target:     `{' > '.join(task_order)}`",
            "",
            "> TODO: hai thứ tự có khác nhau không? Nếu có, đó chính là Lỗi #3 (xếp hạng "
            "cấu hình bằng loss) đo được trên chính run của bạn.",
            "",
            "## 4. Latency và cost",
            "",
            report.markdown_table(comp["rows"]),
            "",
            cost.verdict_line(comp),
            "",
            f"Giả định: ${GPU_HOURLY_USD}/GPU-giờ, API ${API_IN_USD_MTOK}/"
            f"${API_OUT_USD_MTOK} per Mtok. " + comp["assumptions"]["note"],
            "",
            f"Tổng train mọi run: {all_seconds/60:.0f} phút "
            f"≈ ${cost.training_usd(all_seconds, GPU_HOURLY_USD):.3f}.",
            "",
            "## 5. Holdout (không dùng để chọn bất cứ thứ gì)",
            "",
            report.markdown_table([{"version": v, "target": _f(HOLD[v]["target"], 4),
                                    "n": HOLD[v]["n"]} for v in HOLD]) if HOLD
            else "_(không chấm holdout)_",
            "",
            "> TODO: điểm holdout có sát điểm target không? Lệch nhiều = tập target đã bị "
            "nhìn quá nhiều lần trong lúc chỉnh.",
            "",
            "## 6. Kết luận",
            "",
            f"- Cổng kiểm tra: {len(checks) - n_fail}/{len(checks)} PASS.",
            "- TODO: bạn sẽ ship bản nào, và **vì sao**? Nếu (b) đủ tốt thì câu trả lời "
            "đúng có thể là 'không fine-tune' — deck §17.",
            "- TODO: break-even ở trên có nằm trong lưu lượng thật của hệ thống bạn không?",
            "",
        ]
        REPORT = WORK / "REPORT.md"
        REPORT.write_text("\\n".join(lines), encoding="utf-8")
        print(f"đã ghi {REPORT}  ({len(lines)} dòng)")
        print("\\n".join(lines[:34]))
    """)
    md("""
    ### 13.3 Đóng gói

    Zip chỉ chứa `results/`, `REPORT.md` và các adapter — **không** chứa base model
    (~8 GB) và không chứa lại corpus (đã có trong dataset gốc).
    """)
    code("""
        SUB = WORK / "submission"
        SUB.mkdir(exist_ok=True)
        zpath = SUB / f"lab21_{TIER.name.lower()}{'_smoke' if SMOKE else ''}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(RESULTS.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(WORK))
            if (WORK / "REPORT.md").exists():
                z.write(WORK / "REPORT.md", "REPORT.md")
            for adir in sorted(ADAPTERS.glob("*")):
                if adir.name == "merged" or not adir.is_dir():
                    continue
                for p in sorted(adir.rglob("*")):
                    if p.is_file() and p.stat().st_size < 200 * 1024 * 1024:
                        z.write(p, p.relative_to(WORK))
        print(f"{zpath}  ({zpath.stat().st_size/1024**2:.1f} MB)")
        for name in sorted(zipfile.ZipFile(zpath).namelist()):
            print("  ", name)
        if n_fail:
            print(f"\\n❌ {n_fail} mục FAIL ở §13 — xem lại trước khi nộp.")
        else:
            print("\\n✅ Toàn bộ cổng kiểm tra PASS. Tải zip ở tab Output.")
    """)


# =====================================================================================
# The notebook
# =====================================================================================

def build_cells() -> None:
    CELLS.clear()
    _intro()
    _bootstrap()
    _modules()
    _stage1_data_and_mask()
    _stage1b_replay()
    _stage2_baselines()
    _stage3_train()
    _stage4_contrasts()
    _stage5_verdict()
    _stage6_cost()
    _stage7_samples()
    _stage8_merge()
    _stage9_gate()


def to_notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def validate() -> list[str]:
    """compile() every code cell. Magic lines are stripped the way IPython would."""
    problems = []
    for i, (kind, source) in enumerate(CELLS):
        if kind != "code":
            continue
        if source.startswith("%%writefile"):
            body = source.split("\n", 1)[1] if "\n" in source else ""
            try:
                compile(body, f"cell{i}", "exec")
            except SyntaxError as exc:
                problems.append(f"cell {i} (%%writefile payload): {exc}")
            continue
        stripped = "\n".join(
            "pass" if line.lstrip().startswith(("!", "%")) else line
            for line in source.splitlines()
        )
        try:
            compile(stripped, f"cell{i}", "exec")
        except SyntaxError as exc:
            problems.append(f"cell {i}: {exc}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate without writing")
    args = ap.parse_args()

    missing = [m for m in MODULES if not (SRC / f"{m}.py").exists()]
    if missing:
        print(f"missing module sources: {missing}", file=sys.stderr)
        return 1

    build_cells()
    problems = validate()
    for p in problems:
        print("SYNTAX:", p, file=sys.stderr)
    if problems:
        return 1

    n_code = sum(1 for k, _ in CELLS if k == "code")
    if args.check:
        print(f"ok: {len(CELLS)} cells ({n_code} code) compile cleanly; not written")
        return 0

    OUT.write_text(json.dumps(to_notebook(), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(HERE.parent)}  "
          f"{len(CELLS)} cells ({n_code} code)  {kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
