# Lab 21 — Evaluation Report

**Họ tên:** chưa cung cấp  **MSSV:** chưa cung cấp  **Ngày:** 2026-08-21  
**Tier:** T4  **Base model:** `unsloth/Qwen3.5-4B`  **GPU:** Tesla T4 14.6 GB, `fp16`

> Các điểm target/regression dưới đây là số liệu bạn cung cấp để ghi vào report;
> cần đối chiếu lại với `results/*.json` trước khi nộp.

## 1. Setup and mask

Dataset có 250 mẫu, chia 225 train / 25 validation; eval target có 50 mẫu và
regression có 15 mẫu. Chat template giữ `<think>`. Mask `assistant-only` đúng:
39/94 token được supervise (41.49%), câu trả lời nằm trong loss và câu hỏi nằm
ngoài loss. `everything` supervise 94/94 token nên không được dùng. NB1 đo p95 =
98, p99 = 100, max = 101 và đề xuất `max_length=256`; tier chạy 1024.

## 2. Baselines

| Run | target | regression | format | latency ms/sample | n |
|---|---:|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.750 | 0.000 | 3884.1 | 8 |
| (b) base + optimized prompt | 0.650 | 0.750 | chưa cung cấp | chưa cung cấp | — |
| (c) LoRA fine-tune | 0.720 | 0.700 | chưa cung cấp | chưa cung cấp | — |

Prompt (b) target = 0.650; fine-tune target = 0.720, delta target +0.0700. SHA prompt tối ưu:
`719e74d3b6232053`.

## 3. Training and contrasts

Tất cả bốn run dùng cùng 30 optimizer steps, seed, corpus và mask.

| Run | placement | rank | trainable params | LR | final loss | seconds | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32,464,896 | 1e-4 | 3,059,379.7333 | 924.6 | 4.57 |
| attn_only | q,v matched | 283 | 32,456,704 | 1e-4 | 3,059,379.7333 | 800.6 | 4.56 |
| wrong_lr | text-linear | 16 | 32,464,896 | 1e-5 | 3,059,379.7333 | 919.3 | 4.57 |
| qlora | text-linear | 16 | 32,464,896 | 1e-4 | 3,072,842.6667 | 969.0 | 1.74 |

QLoRA tiết kiệm khoảng 2.83 GB VRAM nhưng chậm hơn và không cải thiện target.
Loss cực lớn cùng `grad_norm=nan` cho thấy lần train không ổn định; adapter không
nên được triển khai.

## 4. Verdict

Fine-tune đạt target 0.720 so với baseline (b) 0.650, delta **+0.0700**; regression
0.700 so với 0.750, delta **-0.0500**. Theo cổng của lab, target đã vượt baseline
nhưng regression giảm quá dung sai 0.02, nên verdict vẫn là **FAILED**. Đây là
phép tính từ hai cặp điểm bạn cung cấp, chưa phải artifact đã xác minh.

## 5. Kết luận

Kết quả không ủng hộ triển khai fine-tune. Baseline (b) nhanh hơn LoRA khoảng
đạt target cao hơn nhưng mất 0.05 regression. Cần thêm replay 1–5% hoặc điều chỉnh
training để regression không giảm quá 0.02 trước khi có thể ghi PASSED. Các số liệu
target/regression ở trên là số do người dùng cung cấp, không phải kết quả mình đã
chạy lại.
