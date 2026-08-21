# Lab 21 — Evaluation Report

**Họ tên**: `<điền>`  **MSSV**: `<điền>`  **Ngày**: `<điền>`
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Kaggle T4 16GB (x2, khoá còn 1 qua CUDA_VISIBLE_DEVICES=0)`

> **TRẠNG THÁI: chưa có số đo.** Mọi ô `___` là chỗ chờ số thật từ
> `/kaggle/working/results/` sau khi chạy [colab/Lab21_KAGGLE_FULL.ipynb](../colab/Lab21_KAGGLE_FULL.ipynb).
> Phần văn xuôi đã viết theo **cơ chế** (đọc được từ code + deck), nên nó đúng bất kể số
> ra bao nhiêu; chỗ nào kết luận phụ thuộc vào số thì có sẵn cả hai nhánh, chọn nhánh
> khớp với số của bạn rồi xoá nhánh kia. §13 của notebook tự điền `REPORT.md` bằng số đo —
> file này là bản để đối chiếu và viết phần người phải viết.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 field (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42, `data.split(0.9)`) |
| Eval | target 50 · regression 15 · holdout 20 (không dùng để chọn bất cứ thứ gì) |
| `max_length` | `___` — p95 đo được `___` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 optimizer step**, dùng chung cho MỌI run |
| Effective batch | 16 (`per_device 2 × grad_accum 8`) — dưới ngưỡng 32 của §10.4 |
| Precision | `fp16` — T4 là Turing, **không có bf16** |

**Template có giữ khối `<think>` không?** `___` *(results/template_check.json)*

Qwen3.5 render `<think>` vào phần assistant nhưng **không** có marker `{% generation %}`.
Hệ quả trực tiếp: `assistant_only_loss=True` của TRL sẽ tạo mask **zero token được
supervise** mà chỉ warning, không lỗi. Vì vậy lab này cố tình không dùng flag đó; mask
được dựng từ **offset ký tự** của chat template (`data.build_example`) và được chứng minh
ở §2.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `___` (kỳ vọng 0.10–0.30: câu trả lời JSON ngắn hơn ticket nhiều) |
| Câu trả lời nằm trong loss | `___` (assert cứng, sai là notebook dừng) |
| Câu hỏi KHÔNG nằm trong loss | `___` (assert cứng) |

Đoạn được tính loss:

```
___ (dán từ results/mask_proof.json → supervised_preview)
```

Vì sao ba dòng trên là assert chứ không phải print: nếu mask trùm cả prompt thì model học
**đoán ticket tiếp theo** thay vì học trả lời, và loss vẫn giảm đẹp — không có cách nào
phát hiện ra bằng đường loss. Đây là lỗi duy nhất trong lab mà sai thì mọi số phía sau
đều vô nghĩa nhưng không có triệu chứng nào.

**F-31 (train/eval prompt alignment)**: `___` *(results/prompt_alignment.json)*.
`data.assert_prompt_alignment` kiểm rằng chuỗi eval gửi đi là **prefix** của chuỗi train
supervise. Lần trước lab này lệch đúng chỗ đó và **mọi** adapter chấm 0.000 dù train hoàn
toàn bình thường.

---

## 3. Ba (bốn) baseline — (a) và (b) đo TRƯỚC khi train

| Run | target | regression | format | latency (ms/sample) |
|---|---|---|---|---|
| (a) base + naive prompt | `___` | `___` | `___` | `___` |
| (b) base + optimized prompt | `___` | `___` | `___` | `___` |
| (c) LoRA fine-tune | `___` | `___` | `___` | `___` |
| (c+) fine-tune + replay 5% | `___` | `___` | `___` | `___` |

Bốn nhóm đo bốn thứ khác nhau, và đó là toàn bộ lý do bảng này có 4 cột thay vì 1 con số
perplexity: **target** = accuracy trung bình trên 4 field triage (có điểm thành phần, vì
đúng `intent` sai `urgency` thật sự tốt hơn sai cả hai); **regression** = keyword-recall
trên 15 câu hỏi thường ngày, tức "có quên hết mọi thứ khác không"; **format** =
`has_required_keys`, JSON hợp lệ nhưng sai key **không** tính là đúng định dạng, nó chỉ
parse được; **latency** = throughput theo batch đã trừ warm-up.

**(b) có thật sự mạnh hơn (a) không?** `___`
Δ target (b)−(a) = `___`. Đây là phần **prompt engineering** làm được, chưa cần một
GPU-giờ nào. Con số này là mẫu số của cả bài lab: nếu fine-tune chỉ hơn (a) mà không hơn
(b), thì thứ bạn vừa chứng minh là "viết prompt tử tế thì tốt hơn", không phải
"fine-tune có tác dụng".

**Có sửa `OPTIMIZED_PROMPT` không?** Không. `results/baselines_frozen.json` ghi
`optimized_prompt_sha = ___` để việc đó kiểm được bằng một con số thay vì bằng lời hứa.
Đây là chỗ dễ gian lận nhất trong lab và cũng dễ bị bắt nhất: làm (b) yếu đi thì (c) tự
động đẹp lên, nên prompt (b) bị đóng băng và băm SHA **trước** khi train bắt đầu.

**Prompt token**: (b) `___` tok/request vs (c) `___` tok/request. Chênh lệch này không
phải chi tiết kỹ thuật — nó là tiền, và §5 quy nó ra $/1k ticket.

---

## 4. Giải phẫu cấu hình sai

| Run | vị trí | r | trainable | LR | train loss | **target** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | `___` | 1e-4 | `___` | `___` | `___` | `___` |
| `attn_only` | q,v | *(matched)* `___` | `___` | 1e-4 | `___` | `___` | `___` | `___` |
| `wrong_lr` | text-linear | 16 | `___` | 1e-5 | `___` | `___` | `___` | `___` |
| `qlora` | text-linear | 16 | `___` | 1e-4 | `___` | `___` | `___` | `___` |

Cả 4 run cùng **30 step**, cùng seed, cùng dữ liệu, cùng mask — mỗi dòng đổi **đúng một
biến**. Bản Kaggle này sửa một lỗi công bằng của repo gốc: NB4 cũ hardcode `max_steps=60`
trong khi NB3 chạy 30, nên bảng contrast cũ đang đo *độ dài train* chứ không đo *cấu
hình*. Giờ cả hai đi qua `train.planned_steps()`, và §13 assert rằng mọi `max_steps`
trong `runs.csv` bằng nhau.

Thứ tự theo train loss: `___`
Thứ tự theo target:     `___`

**4.1 — `attn_only` vs `correct` ở cùng ngân sách tham số.**
`attn_only` chỉ gắn adapter vào `q_proj`/`v_proj`, nhưng rank được nâng lên
(`modeling.matched_rank`) cho tới khi số tham số huấn luyện khớp `correct` trong ±5% —
§13 kiểm điều này. Nhờ vậy nếu hai run khác điểm, biến duy nhất còn lại là **vị trí**,
không phải **dung lượng**. Kỳ vọng theo deck §10.2 là `attn_only` thua, vì phần lớn năng
lực biểu diễn của decoder nằm ở MLP (`gate/up/down_proj`) và một tác vụ định dạng-hoá
đầu ra cần chỉnh chỗ đó, nhưng đó là *kỳ vọng*, và ô target ở trên mới là *bằng chứng*.
→ Số đo: `attn_only` `___` vs `correct` `___`, tức nó **`<thắng/thua/hoà>`** `___` điểm.
Nếu hai thứ tự (loss vs target) khác nhau, viết thẳng ra: đó là Lỗi #3 đo được trên chính
run của mình — rank không mua lại được vị trí, và loss không xếp hạng được cấu hình.

**4.2 — `wrong_lr` chỉ khác một con số.**
1e-5 là thang learning rate của **full fine-tuning**; LoRA cần ~10× vì gradient chỉ đi qua
hai ma trận thấp hạng nên bước cập nhật hiệu dụng nhỏ hơn nhiều (§10.3). Ở 30 step, LR
thấp cho đường loss **gần như phẳng**: nó không phân kỳ, không NaN, không có triệu chứng
nào của một run sai — nó chỉ đơn giản là chưa học gì. Nếu chỉ nhìn loss mà không biết LR,
kết luận sai sẽ là *"tác vụ này khó/dữ liệu này bẩn/base model không đủ"*, và bước tiếp
theo sẽ là đi thu thêm dữ liệu — sửa sai thứ, tốn nhiều ngày.
→ Số đo: loss cuối `___` (so với `correct` `___`), target `___`.

**4.3 — `qlora` đổi VRAM lấy gì.**
4-bit tiết kiệm `___` GB (`___` → `___`), nhưng ở kích cỡ 4B trên T4 thì bộ nhớ **không
phải** ràng buộc — 16-bit đã vừa. Nên khoản tiết kiệm ấy không mua được gì, trong khi
lượng tử hoá base làm nhiễu chính những trọng số mà adapter đang phải hiệu chỉnh quanh, và
dequant mỗi forward pass làm chậm cả train lẫn infer (`train_seconds` `___` vs `___`).
→ Số đo target: `___` vs `correct` `___`. Kết luận: khuyến nghị "không dùng QLoRA cho dòng
model này ở tier này" **`<được/không được>`** số đo của mình ủng hộ. QLoRA đúng chỗ của nó
là khi model **không vừa** VRAM — lúc đó so sánh là "4-bit vs không train được gì".

---

## 5. Latency và cost

| version | prompt_tok | out_tok | ms/sample | self-host $/1k | API-equiv $/1k |
|---|---|---|---|---|---|
| (a) | `___` | `___` | `___` | `___` | `___` |
| (b) | `___` | `___` | `___` | `___` | `___` |
| (c) | `___` | `___` | `___` | `___` | `___` |

Giả định (không phải số đo, sửa được ở ô CONFIG): $0.35/GPU-giờ (T4 spot), API
$0.15/$0.60 per Mtok. `ms/sample` là throughput **theo batch** ở batch 4, đã trừ warm-up —
so sánh giữa các phiên bản là công bằng vì mọi phiên bản đo cùng batch size, nhưng đây
không phải báo giá production. Cost self-host còn giả định GPU luôn có việc; GPU thuê để
không vẫn tính tiền, nên đây là **mức sàn**.

Train: `___` giây cho run thắng = $`___`. Tổng mọi run (kể cả 3 contrast chỉ để chứng minh
cấu hình sai là sai): `___` phút = $`___`.

**Break-even**: `___` ticket. Đây là con số biến bài lab thành một quyết định: fine-tune
trả trước bằng GPU-giờ và mua lại prompt ngắn hơn trên **mọi** request về sau. Nếu hệ
thống của bạn xử lý dưới `___` ticket thì (b) rẻ hơn, và câu trả lời đúng có thể là *đừng
fine-tune* — đúng tinh thần §17. Không dòng nào ở trên tính tiền công engineer, và bản
fine-tune đã tiêu một ít.

---

## 6. Phán quyết

**Cổng hồi quy**: `___` (`PASSED`/`FAILED`)
`target Δ = ___` · `regression Δ = ___` (dung sai 0.02) · `valid_trace_rate = ___`
`format: (b) ___ → (c) ___`

**Diễn giải.** Cổng này đòi hai điều cùng lúc, và đó là điểm mấu chốt: hơn (b) trên tác vụ
đích, **và** không tụt quá 0.02 trên năng lực chung. Chỉ đòi điều thứ nhất thì mọi run
overfit đều "thắng"; chỉ đòi điều thứ hai thì base model không train gì là vô địch. Con số
đáng đọc nhất trong bảng không phải `target` mà là `regression`: 225 ví dụ toàn JSON triage,
30 step, LR 1e-4 là đúng công thức để model học rằng *mọi* input đều phải trả về một object
4 key — và lúc đó "Thủ đô của Nhật Bản là thành phố nào?" cũng nhận được một object triage.
Đó là quên thảm hoạ, nhìn thấy bằng mắt ở §7 trước khi nhìn thấy bằng số ở đây.
`format` gần như chắc chắn tăng mạnh (base phải được *nhắc* mới ra JSON; bản fine-tune ra
JSON vì đó là điều duy nhất nó từng thấy) — nhưng format tăng **không** đủ để qua cổng, và
đó là thiết kế: định dạng là phần dễ nhất, prompt cũng làm được.
Nếu **FAILED vì regression**: nhánh (c+) tồn tại chính để trả lời — trộn 5% replay
(`results/replay_manifest.json`), giữ **nguyên** 30 step, đổi **đúng một biến là DỮ LIỆU**,
xem regression hồi lại bao nhiêu và trả giá bao nhiêu trên target. Δ đo được: regression
`___`, target `___`.
Nếu **FAILED vì target**: nghĩa là 30 step ở corpus 225 ví dụ chưa đủ, hoặc prompt (b) vốn
đã rất mạnh cho tác vụ này — hai chẩn đoán khác nhau, phân biệt bằng `final_eval_loss` so
với `min_eval_loss` trong `curves_correct.json`: eval loss còn đang giảm ⇒ thiếu step;
eval loss đã quay đầu tăng ⇒ đã overfit và vấn đề không nằm ở ngân sách train.
Một `FAILED` được giải phẫu như trên có giá trị hơn một `PASSED` không giải thích được.

**Holdout** (20 ticket, chưa từng dùng để chọn gì): (b) `___` vs (c) `___`. Nếu điểm holdout
sát điểm target thì bảng §3 đáng tin; lệch nhiều nghĩa là tập target đã bị nhìn quá nhiều
lần trong lúc chỉnh và nó đã âm thầm biến thành tập val.

---

## 7. Định tính — bắt buộc có cả ca THUA

`results/qualitative.json` xếp 50 ticket theo `(ft − b, ft)`, nên 3 dòng đầu là 3 ca
fine-tune thua nặng nhất và 3 dòng cuối là 3 ca thắng rõ nhất. Không tự chọn ví dụ có lợi.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `___` | `___` | `___` | `___` | ✅ FT thắng |
| 2 | `___` | `___` | `___` | `___` | ✅ FT thắng |
| 3 | `___` | `___` | `___` | `___` | ❌ **FT thua** |
| 4 | `___` | `___` | `___` | `___` | ❌ **FT thua** |
| 5 | `___` | `___` | `___` | `___` | `___` |

**Mẫu chung ở các ca FT thua** — kiểm ba giả thuyết này bằng `results/verdict.json →
field_accuracy` thay vì bằng cảm giác:
1. **Ticket đa ý** (giao chậm *và* hàng lỗi *và* xin hoàn tiền): `intent` là một nhãn duy
   nhất nên bắt buộc phải chọn, và cả nhãn vàng lẫn model đều có thể chọn khác nhau một
   cách hợp lý. Đây là giới hạn của **schema**, không phải lỗi của model.
2. **`urgency` đoán theo giọng điệu**: khách viết hoa và nhiều dấu chấm than được đẩy lên
   `high` dù nội dung chỉ là hỏi thông tin. Nếu `field_accuracy` cho thấy `urgency` là
   field tệ nhất thì đây là nghi phạm số một, và cách sửa là **định nghĩa lại nhãn trong
   guideline**, không phải train thêm.
3. **`product` là copy từ free text** nên sai chính tả/thiếu dấu là sai điểm — đây là lý do
   scorer fold dấu cho riêng field này còn 3 field kia so khớp chính xác (chúng đến từ từ
   vựng đóng, gần-đúng ở đó là lỗi thật, không phải chính tả).

`unparsed` = `___`. Nếu > 0 thì phần điểm mất đó là mất vì **định dạng**, không phải vì
phân loại sai — hai lỗi không liên quan gì nhau và không sửa bằng cùng một cách.

**Ba câu tổng quát** (`results/samples.json`) — chỗ quên thảm hoạ hiện ra không cần bảng điểm:

| câu hỏi | (b) | (c) |
|---|---|---|
| Thủ đô của Nhật Bản là thành phố nào? | `___` | `___` |
| Giải thích ngắn gọn vì sao trời có mưa. | `___` | `___` |
| Viết một câu cảm ơn khách hàng đã mua hàng. | `___` | `___` |

**Ca ngoài miền** (input không phải ticket khiếu nại): `___`. Bản fine-tune vẫn cố nhồi nó
vào schema 4 field, hay từ chối? Câu trả lời quyết định nó có deploy được sau một router
hay không.

---

## 8. Kết luận & điều tôi học được

**Kết luận.** `<viết sau khi có số — dàn ý và các số cần dẫn ở dưới>`
Quyết định deploy phải trả lời ba câu, theo thứ tự: (i) nó có hơn **(b)**, chứ không phải
hơn (a), trên tác vụ đích không — Δ = `___`; (ii) nó có giữ được năng lực chung trong dung
sai 0.02 không — Δ = `___`; (iii) khoản tiết kiệm `___` $/1k có hoàn được $`___` tiền train
trong lưu lượng thật của hệ thống không — break-even `___` ticket. Cả ba đều "có" thì
deploy; câu (ii) "không" thì đường sửa đã có sẵn và đo được: replay 5%. Riêng câu (iii)
"không" thì vẫn có thể deploy nhưng phải nói rõ là mua **chất lượng**, không mua giá.
Về đòn bẩy thật sự: theo số ở §4, thứ tự tác động là **`___`**. Điều đáng nói là ba trong
bốn đòn bẩy (vị trí adapter, LR, mask) đều là **cấu hình**, không phải dữ liệu — và cả ba
đều có kiểu hỏng *không có triệu chứng trên đường loss*, nên thứ đắt nhất trong lab này
không phải GPU-giờ mà là có một thang đo trung thực để phát hiện chúng.

**Ba điều tôi học được:**
1. `assistant_only_loss=True` của TRL supervise **zero token** trên template không có
   `{% generation %}` — và chỉ warning. Một flag đọc như đúng lại làm ngược lại, im lặng.
2. bf16 không phải mặc định an toàn: T4 là Turing, không có bf16, mà TRL vẫn trả về LoRA
   weight bf16 khi `fp16=True` → chết ở optimizer step đầu vì kernel
   `_amp_foreach_non_finite_check_and_unscale_cuda` không có overload BFloat16. Phải sửa
   trên model TRL trả về, không sửa được bằng cách cấu hình TRL.
3. Hai scorer cùng tự hỏi "đây có phải JSON không" mà trả lời khác nhau thì **cả hai** con
   số đều mất giá trị: output `"Đây là kết quả: {...}"` từng ăn điểm target nhưng 0 điểm
   format, đọc ra như một lỗi định dạng chưa từng xảy ra.

**Nếu có thêm 2 giờ**: quét `r ∈ {8,16,32,64}` ở nguyên 30 step để xem rank có bù được vị
trí không (giả thuyết: không); và chạy `MASK_MODE=masked-think` để đo
`valid_reasoning_trace` — deck §13.5 nói target accuracy có thể tăng *trong khi* trace sụp
về 0, và đó là thứ chỉ đo được khi có riêng một chỉ số cho nó.

---

## Phụ lục — thưởng đã làm

- [x] **B1** NB6 merge + hot-swap — `results/merge_check.json`, assert điểm không tụt >0.01
      sau `merge_and_unload()`; kèm demo một base + nhiều adapter trong cùng VRAM
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

**Ngoài rubric:**
- **(c+) replay run** — deck §14.3 áp dụng và **đo**, cùng 30 step, biến duy nhất là dữ liệu
- **Cost model** — `results/cost.json`: $/1k ticket + break-even, giả định giá tách khỏi số đo
- **Held-out loss trong lúc train** — `curves_*.json` phân biệt "train lâu hơn" với "fit tốt hơn"
- **`run_config.json`** — đủ để chạy lại: tier, precision, step, seed, spec, giá, version thư viện
- Sửa hai lỗi trong repo: `verify.py` báo FAIL checksum trên mọi checkout Windows (CRLF vs
  LF), và 3 test luôn `ModuleNotFoundError` do sai đường import




