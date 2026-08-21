"""Replay corpus for deck §14.3 — the anti-forgetting mix.

**Why this file exists.** The lab's own measured result on the shipped corpus:

    (b) base + optimized prompt   target 0.495   regression 0.644
    (c) LoRA fine-tune            target 0.990   regression 0.067   <- FAILED

The fine-tune learned the task almost perfectly and lost nearly all of its general
capability — it answers "what is the capital of Vietnam?" with a triage JSON object.
The cause is not LoRA: it is a training corpus in which *every* input is a ticket and
*every* answer is JSON, so "input" and "emit triage JSON" become the same thing. Deck
§14.3 names the remedy — mix 1-5% general data back in — and nothing in the lab tested
it. This is that data.

**Why hand-written and in-repo rather than a HF dataset.** Three reasons, in order of
how much they matter:

1. **Decontamination is checkable.** The regression group is scored on 15 specific
   questions. A downloaded general-instruction set may or may not contain them; a
   file you can read cannot hide them, and `data.assert_replay_decontaminated()`
   proves it with a number.
2. **Shape has to match.** Replay examples must render as a plain user turn, because
   that is how the regression probe asks. A dataset with its own system prompts would
   introduce a second prompt shape and confound the one variable under test.
3. **It runs offline.** A Kaggle session with internet disabled still trains.

**Answers are deliberately short and plain.** They are here to remind the model that
"answer the question in Vietnamese prose" remains a legal behaviour — not to teach it
new facts. A long answer would spend supervised tokens teaching style, which is a
second variable.

Every item is checked against `data/eval_regression.jsonl` before use. Adding items is
encouraged; the assert is what keeps that safe.
"""
from __future__ import annotations

# 47 items. Kept disjoint from the 15 regression-eval questions by topic, and verified
# by Jaccard overlap at import time in the notebook — not by eyeballing.
REPLAY: list[dict] = [
    # --- general knowledge (geography / history / civics) ---
    {"instruction": "Nước nào có diện tích lớn nhất thế giới?", "input": "",
     "output": "Liên bang Nga là quốc gia có diện tích lớn nhất thế giới, khoảng 17 triệu km²."},
    {"instruction": "Đỉnh núi cao nhất Việt Nam tên là gì?", "input": "",
     "output": "Đó là đỉnh Fansipan ở tỉnh Lào Cai, cao 3.147 m so với mực nước biển."},
    {"instruction": "Việt Nam có bao nhiêu tỉnh và thành phố trực thuộc trung ương?", "input": "",
     "output": "Việt Nam có 63 đơn vị hành chính cấp tỉnh, gồm 58 tỉnh và 5 thành phố trực thuộc trung ương."},
    {"instruction": "Đồng tiền chính thức của Nhật Bản gọi là gì?", "input": "",
     "output": "Đồng tiền chính thức của Nhật Bản là yên (JPY)."},
    {"instruction": "Ai được coi là người phát minh ra bóng đèn sợi đốt thương mại?", "input": "",
     "output": "Thomas Edison thường được ghi công là người thương mại hoá bóng đèn sợi đốt vào năm 1879."},
    {"instruction": "Hệ Mặt Trời có bao nhiêu hành tinh?", "input": "",
     "output": "Hệ Mặt Trời có 8 hành tinh, từ Sao Thuỷ đến Sao Hải Vương."},
    {"instruction": "Vạn Lý Trường Thành nằm ở quốc gia nào?", "input": "",
     "output": "Vạn Lý Trường Thành nằm ở Trung Quốc."},
    {"instruction": "Lá cờ Việt Nam có những màu gì?", "input": "",
     "output": "Cờ Việt Nam có nền đỏ và một ngôi sao vàng năm cánh ở giữa."},

    # --- arithmetic / simple quantitative reasoning ---
    {"instruction": "Tính 15 nhân 12.", "input": "",
     "output": "15 × 12 = 180."},
    {"instruction": "Một hình vuông có cạnh 7 cm. Diện tích bằng bao nhiêu?", "input": "",
     "output": "Diện tích = 7 × 7 = 49 cm²."},
    {"instruction": "30% của 250 là bao nhiêu?", "input": "",
     "output": "30% của 250 là 75."},
    {"instruction": "Nếu một món hàng giá 400 nghìn đồng được giảm 25%, giá sau giảm là bao nhiêu?", "input": "",
     "output": "Giảm 25% của 400 nghìn là 100 nghìn, nên giá sau giảm là 300 nghìn đồng."},
    {"instruction": "Số nguyên tố nhỏ nhất lớn hơn 20 là số nào?", "input": "",
     "output": "Đó là 23."},
    {"instruction": "Trung bình của 4, 8 và 15 là bao nhiêu?", "input": "",
     "output": "(4 + 8 + 15) / 3 = 27 / 3 = 9."},

    # --- translation (both directions, none matching the eval item) ---
    {"instruction": "Dịch câu sau sang tiếng Anh: 'Hôm nay trời mưa rất to'.", "input": "",
     "output": "It is raining very heavily today."},
    {"instruction": "Dịch sang tiếng Việt: 'Please send me the invoice tomorrow'.", "input": "",
     "output": "Vui lòng gửi cho tôi hoá đơn vào ngày mai."},
    {"instruction": "Dịch sang tiếng Anh: 'Cảm ơn bạn đã giúp đỡ tôi'.", "input": "",
     "output": "Thank you for helping me."},

    # --- writing / rewriting ---
    {"instruction": "Viết một câu chào mở đầu email cho khách hàng doanh nghiệp.", "input": "",
     "output": "Kính gửi Anh/Chị, cảm ơn Anh/Chị đã quan tâm tới sản phẩm của chúng tôi."},
    {"instruction": "Viết lại câu sau cho lịch sự hơn: 'Gửi báo giá ngay'.", "input": "",
     "output": "Anh/Chị vui lòng gửi giúp em báo giá khi thuận tiện nhé, em xin cảm ơn."},
    {"instruction": "Đặt một tiêu đề ngắn cho bài viết về lợi ích của việc đi bộ mỗi ngày.", "input": "",
     "output": "Mỗi ngày ba mươi phút đi bộ: món đầu tư rẻ nhất cho sức bền."},
    {"instruction": "Viết một câu cảm ơn đồng nghiệp đã hỗ trợ dự án.", "input": "",
     "output": "Cảm ơn bạn đã hỗ trợ nhiệt tình, dự án về đích đúng hạn là nhờ phần việc của bạn."},
    {"instruction": "Tóm tắt trong một câu: 'Công ty mở thêm hai chi nhánh tại Đà Nẵng và Cần Thơ trong quý này, đồng thời tuyển thêm 40 nhân viên.'", "input": "",
     "output": "Trong quý này công ty mở hai chi nhánh mới ở Đà Nẵng và Cần Thơ và tuyển thêm 40 nhân viên."},

    # --- explanation / definition ---
    # Worded to stay off the eval set's "Giải thích ngắn gọn quang hợp là gì." shape:
    # that phrasing scores Jaccard 0.60 against it, right on the decontamination limit.
    {"instruction": "Lạm phát ảnh hưởng thế nào tới tiền tiết kiệm của một gia đình?", "input": "",
     "output": "Lạm phát là hiện tượng mức giá chung của hàng hoá và dịch vụ tăng lên theo thời gian, làm giảm sức mua của đồng tiền."},
    {"instruction": "Mã bưu chính dùng để làm gì?", "input": "",
     "output": "Mã bưu chính giúp phân loại và định tuyến thư từ, bưu kiện tới đúng khu vực giao nhận."},
    {"instruction": "Sự khác nhau giữa RAM và ổ cứng là gì?", "input": "",
     "output": "RAM là bộ nhớ tạm, rất nhanh nhưng mất dữ liệu khi tắt máy; ổ cứng lưu dữ liệu lâu dài nhưng chậm hơn nhiều."},
    {"instruction": "HTTPS khác HTTP ở điểm nào?", "input": "",
     "output": "HTTPS mã hoá dữ liệu trên đường truyền bằng TLS, còn HTTP gửi dữ liệu dạng rõ nên có thể bị đọc hoặc sửa."},
    {"instruction": "Vì sao nên sao lưu dữ liệu định kỳ?", "input": "",
     "output": "Vì ổ đĩa có thể hỏng, thiết bị có thể mất và người dùng có thể xoá nhầm; bản sao lưu là cách duy nhất phục hồi lại được."},
    {"instruction": "Giải thích ngắn gọn thế nào là năng lượng tái tạo.", "input": "",
     "output": "Năng lượng tái tạo là năng lượng lấy từ những nguồn tự bù đắp trong thời gian ngắn như nắng, gió, nước và sinh khối."},
    {"instruction": "Vì sao cần uống đủ nước mỗi ngày?", "input": "",
     "output": "Nước tham gia vào điều hoà nhiệt độ, vận chuyển dưỡng chất và đào thải chất cặn, nên thiếu nước làm cơ thể hoạt động kém đi."},

    # --- lists ---
    {"instruction": "Kể tên ba loại rau ăn lá phổ biến ở Việt Nam.", "input": "",
     "output": "Rau muống, rau cải xanh và mồng tơi."},
    {"instruction": "Kể tên hai ngôn ngữ lập trình thường dùng cho khoa học dữ liệu.", "input": "",
     "output": "Python và R."},
    {"instruction": "Nêu hai cách tiết kiệm điện trong gia đình.", "input": "",
     "output": "Tắt thiết bị khi không dùng và thay đèn sợi đốt bằng đèn LED."},
    {"instruction": "Kể tên ba phương tiện giao thông công cộng.", "input": "",
     "output": "Xe buýt, tàu điện và tàu hoả."},

    # --- light reasoning ---
    {"instruction": "Nếu hôm nay là thứ Tư thì ba ngày sau là thứ mấy?", "input": "",
     "output": "Ba ngày sau thứ Tư là thứ Bảy."},
    {"instruction": "Một xe chạy 60 km trong 45 phút. Tốc độ trung bình là bao nhiêu km/h?", "input": "",
     "output": "45 phút là 0,75 giờ, nên tốc độ trung bình là 60 / 0,75 = 80 km/h."},
    {"instruction": "Sắp xếp các số sau theo thứ tự tăng dần: 12, 3, 47, 9.", "input": "",
     "output": "3, 9, 12, 47."},
    {"instruction": "Trong câu 'Con mèo đuổi con chuột', ai là chủ ngữ?", "input": "",
     "output": "Chủ ngữ là 'con mèo'."},

    # --- everyday how-to ---
    {"instruction": "Làm thế nào để pha một ly trà xanh đúng cách?", "input": "",
     "output": "Tráng ấm bằng nước nóng, dùng nước khoảng 80 độ C, hãm lá trà chừng hai phút rồi rót ra ngay để trà không bị đắng."},
    {"instruction": "Nên làm gì đầu tiên khi điện thoại bị rơi xuống nước?", "input": "",
     "output": "Tắt máy ngay, lau khô bên ngoài và để khô tự nhiên ở nơi thoáng; không sấy nóng và không cắm sạc."},
    {"instruction": "Cách bảo quản rau xanh trong tủ lạnh cho lâu hơn?", "input": "",
     "output": "Để rau thật khô, bọc bằng giấy hoặc khăn thấm rồi cho vào hộp kín và đặt ở ngăn mát, tránh để cạnh trái cây chín."},
    {"instruction": "Làm sao đặt một mật khẩu mạnh mà vẫn nhớ được?", "input": "",
     "output": "Ghép bốn từ ngẫu nhiên không liên quan thành một cụm dài, thêm một chữ số và một ký tự đặc biệt; độ dài quan trọng hơn sự phức tạp."},

    # --- opinion / short advice, keeps discursive answering alive ---
    {"instruction": "Một lời khuyên cho người mới bắt đầu học lập trình?", "input": "",
     "output": "Viết chương trình nhỏ mỗi ngày và đọc lại code của chính mình sau một tuần — thói quen đó dạy nhiều hơn là đọc thêm một quyển sách."},
    {"instruction": "Nên chuẩn bị gì trước một buổi phỏng vấn xin việc?", "input": "",
     "output": "Đọc kỹ mô tả công việc, chuẩn bị hai ví dụ cụ thể về việc mình đã làm, và soạn sẵn vài câu hỏi cho nhà tuyển dụng."},
    {"instruction": "Vì sao nên đọc điều khoản trước khi cài một ứng dụng lạ?", "input": "",
     "output": "Vì đó là nơi ghi ứng dụng thu thập dữ liệu gì và chia sẻ với ai; quyền truy cập danh bạ hay vị trí thường được xin ở đó."},
    {"instruction": "Làm việc nhóm hiệu quả cần điều gì nhất?", "input": "",
     "output": "Sự rõ ràng: ai làm việc gì, xong khi nào, và thông tin gì cần chia sẻ cho ai."},

    # --- format-adjacent but NOT triage: proves the model can still choose a format ---
    {"instruction": "Liệt kê ba bước gửi hàng qua bưu điện, mỗi bước một dòng.", "input": "",
     "output": "1. Đóng gói và ghi rõ địa chỉ người nhận.\n2. Mang tới bưu cục và chọn dịch vụ chuyển phát.\n3. Giữ biên nhận để tra mã vận đơn."},
    {"instruction": "Viết một câu mô tả thời tiết Hà Nội vào tháng Ba.", "input": "",
     "output": "Tháng Ba Hà Nội thường ẩm và nhiều sương, nhiệt độ dịu, đôi khi có mưa phùn kéo dài."},
]


def load(extra: list[dict] | None = None) -> list[dict]:
    """The replay pool, optionally extended. Always run it through
    `data.assert_replay_decontaminated()` against the regression eval set first."""
    pool = [dict(r) for r in REPLAY]
    if extra:
        pool += [dict(r) for r in extra]
    for r in pool:
        r.setdefault("input", "")
        if not r.get("instruction") or not r.get("output"):
            raise ValueError(f"replay record needs instruction+output: {r!r}")
    return pool
