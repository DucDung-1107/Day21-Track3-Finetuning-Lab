# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

> Phần 1–3 và 5 dưới đây viết được từ những gì đã đọc và đã sửa trong code; chỗ nào cần số
> đo thì để `___`. Phần 4 là phần **chỉ bạn trả lời được** — không ai viết hộ được việc bạn
> đã dùng AI vào đâu.

**1. Điều gì làm bạn ngạc nhiên nhất?**

Số lượng kiểu hỏng **không có triệu chứng**. Ba cái gặp trong lab này đều im lặng theo cùng
một cách: `assistant_only_loss=True` supervise zero token nhưng chỉ warning; `warmup_ratio`
bị TRL bỏ qua nên run train **không warmup** mà không báo gì; và train/eval render lệch
nhau (F-31) khiến mọi adapter chấm 0.000 dù loss giảm hoàn toàn bình thường. Cả ba đều cho
một đường loss đẹp. Nghĩa là đường loss — thứ mà bản năng bảo phải nhìn — không phát hiện
được kiểu lỗi phổ biến nhất, và cái thay thế nó là một tập assert chạy **trước** khi train.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ dự đoán. Dự đoán là ở LoRA/hyperparameter; thực tế là ở **hạ tầng và tính
công bằng của phép so sánh**: dtype (T4 Turing không có bf16, nhưng TRL trả về LoRA weight
bf16 khi `fp16=True` → chết ở optimizer step đầu), đường ghi kết quả (`report.py` tính
`parents[2]/"results"` → ra `/results` trên Kaggle, không ghi được, và chỉ lộ ra ở cuối một
run 3 giờ), và ngân sách step (NB4 hardcode 60 đối đầu NB3 chạy 30 — bảng contrast đang đo
độ dài train chứ không đo cấu hình). Việc chọn `r` mất đúng vài phút. Bài học: phần khó của
fine-tuning không nằm ở fine-tuning.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Rằng baseline để so là **base model**. Không phải — baseline là **base model được prompt
tử tế**, và khoảng cách (b) − (a) = `___` cho thấy một phần đáng kể của "hiệu quả
fine-tuning" mà người ta hay báo cáo thực chất là phần prompt engineering lẽ ra phải làm
trước, không tốn GPU-giờ nào. Cũng không còn tin rằng loss thấp hơn nghĩa là cấu hình tốt
hơn: `wrong_lr` và `attn_only` cho hai thứ tự khác nhau giữa cột loss và cột target, và
xếp hạng bằng loss chính là cách kết luận ngược.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

`<phần này bạn tự viết — cụ thể: dùng vào việc gì, chỗ nào nó đưa code sai và bạn phát hiện
bằng cách nào>`

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Không phải chuẩn bị dữ liệu. Bước đầu là **dựng thang đo và một baseline prompt mạnh**, rồi
mới hỏi có cần fine-tune không. Cụ thể: chốt tập eval và băm SHA nó lại (`checksums.json`)
để không ai sửa được sau khi thấy kết quả; viết scorer bốn nhóm gồm cả nhóm hồi quy — nếu
không đo cái mình *không* train thì sẽ không thấy mình vừa làm hỏng nó; tối ưu prompt hết
sức rồi mới coi con số đó là mốc phải vượt. Rất nhiều dự án dừng lại ngay ở bước này với
câu trả lời "prompt là đủ", và đó là một kết quả tốt, không phải một thất bại.
