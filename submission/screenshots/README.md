# Screenshots cần chụp

Chụp từ notebook Kaggle sau khi Run All. Đặt tên đúng như dưới đây rồi xoá file này.

| tên file | chụp ở đâu | phải thấy được gì |
|---|---|---|
| `01_env.png` | §1 ô cuối | tên GPU, VRAM, `sm_75`, version torch/transformers/trl/peft |
| `02_checksums.png` | §3 | 4 dòng corpus kèm `sha=`/`lf=` và `[ok]`, số dòng mỗi file |
| `03_mask_proof.png` | §4 (mask) | `supervised_fraction`, đoạn được tính loss vs đoạn bị che |
| `04_baselines.png` | §6.1 | bảng (a) vs (b) + dòng Δ target và Δ prompt token |
| `05_train_correct.png` | §7 | log train: `max_steps=30`, loss giảm, `precision fix`, VRAM đỉnh |
| `06_runs_table.png` | §8 | bảng `runs.csv` + dòng "cả N run cùng 30 step" |
| `07_scores_hbar.png` | §9.1 | 4 biểu đồ cột text (target/regression/format/latency) |
| `08_gate.png` | §9.2 | PASSED/FAILED kèm lý do, dòng "bản chọn để so cost/serve" |
| `09_autopsy.png` | §9.3 | bảng contrast + hai dòng thứ tự loss vs thứ tự target |
| `10_qualitative.png` | §9.4 | 3 ca FT thua và 3 ca FT thắng |
| `11_cost.png` | §10 | bảng cost, `delta`, dòng break-even |
| `12_samples_general.png` | §11 | 3 câu tổng quát, các phiên bản cạnh nhau |
| `13_verify_gate.png` | §13 | danh sách PASS/FAIL và dòng tổng kết |

Ảnh bắt buộc nếu thiếu thời gian: `03`, `04`, `07`, `08`, `11`, `13` — sáu ảnh này là
bằng chứng cho toàn bộ phần kết luận của REPORT.md.
