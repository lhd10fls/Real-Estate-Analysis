# Kết quả xử lý lại dữ liệu

Cập nhật: bước huấn luyện và ứng dụng đã được thực hiện tiếp; xem `reports/v2/KET_QUA_THUC_NGHIEM.md` và `RUN_V2.md`. Các ghi chú “chưa huấn luyện” bên dưới mô tả thời điểm bàn giao preprocessing ban đầu.

Đã chạy từ hai CSV raw, giữ nguyên dữ liệu gốc và toàn bộ bảng/model phiên bản cũ.

| Chỉ tiêu | Kết quả |
|---|---:|
| Tổng dòng raw | 13.828 |
| Bản sao hoàn toàn loại khỏi bảng làm việc | 3.652 |
| Bảng chuẩn hóa có truy vết | 10.176 |
| Dòng đủ giá/diện tích hợp lệ và không xung đột URL | 9.691 |
| Train | 7.725 |
| Test | 1.966 |
| Chưa đưa vào mô hình | 485 |
| URL xung đột nội dung | 22 nhóm / 44 dòng |
| Nhóm ứng viên có thể cùng tài sản | 381 |
| Dòng cực trị cần xem xét | 73 |

Các số lượng không cộng trực tiếp với nhau: ví dụ dòng xung đột có thể đồng thời thiếu giá. Xem excluded_from_model.csv để biết từng lý do.

Nhà riêng được giữ thành 1.278 dòng riêng, không gộp sang Nhà mặt phố. Hồ Chí Minh có 4.088 dòng trong bảng chuẩn hóa mới; khác con số cũ vì phạm vi lọc đã thay đổi. Biệt thự/liền kề giữ nguyên sự không chắc chắn của nhãn gốc, không gộp tùy tiện vào Biệt thự.

## Dữ liệu bàn giao

- Đọc `data/processed_v2/canonical_all.csv` để khảo sát dữ liệu, dùng `low_memory=False` do các trường raw giữa hai nguồn có kiểu khác nhau.
- Đọc `X_train.csv`/`y_train.csv` cho phát triển mô hình, join và kiểm tra thứ tự bằng record_id, sau đó bỏ record_id khỏi features.
- Dùng candidate_group_id trong split_manifest.csv cho grouped cross-validation. X_test/y_test dành cho đánh giá sau khi chốt mô hình.
- Input còn NaN và biến categorical theo thiết kế. Phải đưa imputation/encoding/scaling vào Pipeline của mô hình, không fill toàn bộ bảng rồi mới chia.
- Không dùng price_per_m2, log_price hoặc raw__price từ canonical làm đầu vào dự đoán giá.

## Kiểm tra

5 test tự động bao phủ parser giá/diện tích, mapping địa danh/loại nhà/pháp lý, giữ missing, truy vết bản sao và tính độc lập nhóm của split. Các kiểm tra đã chạy thành công.

Chưa huấn luyện lại, chưa đo metrics mới, chưa đổi Streamlit sang v2. Chưa xác nhận nguồn công bố hoặc thời gian crawl Muaban; đơn vị giá số của Muaban vẫn là giả định tỷ VND theo dữ liệu tham khảo. Các nhóm nghi trùng, tọa độ và pháp lý không được coi là đã xác minh ngoài thực tế.

Chi tiết quy tắc và lệnh chạy ở README_v2.md; toàn bộ thống kê máy đọc được ở data/processed_v2/quality_report.json.
