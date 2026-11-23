# Kết quả chạy lại pipeline ngày 04/11/2026

## Dữ liệu và thiết kế đánh giá

Giữ nguyên 13.828 dòng raw. Loại 3.652 bản sao hoàn toàn, bảng canonical còn 10.176 dòng. Tập mô hình gồm 9.691 dòng, chia cố định 7.725 train / 1.966 test theo nhóm ứng viên cùng tài sản. 22 URL xung đột (44 dòng) không được đưa vào mô hình. Những dòng giá/diện tích lớn không bị loại chỉ để cải thiện điểm.

Các model so sánh trên cùng 5 grouped folds. Imputer, encoder và scaler được học bên trong training fold. Biến vị trí đã được bổ sung. Không có price_per_m2, log_price hoặc trường raw chứa giá trong input. Model được chọn bằng MAE CV trên giá gốc trước khi đọc test để đánh giá.

## Kết quả thực nghiệm

| Model | CV MAE ± SD (tỷ VND) | Test MAE (tỷ) | Test RMSE (tỷ) | Test R² giá gốc |
|---|---:|---:|---:|---:|
| Baseline median | 15,009 ± 1,320 | 15,472 | 39,865 | -0,090 |
| Ridge | 23,523 ± 17,298 | 13,432 | 94,384 | -5,109 |
| Random Forest | 7,356 ± 0,755 | 6,972 | 21,749 | 0,676 |

Random Forest được chọn bằng CV. Sai số tuyệt đối trung vị trên test là 1,694 tỷ VND; MAE thấp hơn baseline khoảng 54,9%. MAE và RMSE vẫn lớn, chưa phù hợp coi là công cụ định giá đáng tin cậy cho mọi phân khúc.

Ridge không ổn định trên giá gốc: một số dự đoán log sau expm1 có thể bị khuếch đại mạnh, đặc biệt ngoài phạm vi train. Không sửa/xóa các ca test đó để cải thiện kết quả. Trong một kiểm tra stress với mặt tiền 99.999 m trên mô hình Ridge fit mẫu nhỏ, phép đảo bị overflow; test chính về imputer dùng giá trị hợp lệ, không tuyên bố Ridge chịu được input tùy ý. App v2 dùng Random Forest được chọn và kiểm tra output hữu hạn/dương ở form đơn.

Không so trực tiếp R²=0,676 với R² log cũ 0,723 hoặc 0,790: khác thang đo, dữ liệu, split và feature; Ridge cũ còn target leakage.

## Sai số theo loại nhà của Random Forest

| Loại | Số mẫu test | MAE (tỷ VND) | Median absolute error (tỷ) |
|---|---:|---:|---:|
| Chung cư | 335 | 2,195 | 0,784 |
| Nhà hẻm/ngõ | 354 | 2,279 | 1,009 |
| Nhà riêng | 248 | 2,479 | 1,309 |
| Biệt thự/liền kề | 310 | 9,445 | 3,861 |
| Nhà mặt phố | 689 | 11,890 | 2,830 |
| Biệt thự | 30 | 14,295 | 4,570 |

Các sai số chưa hiệu chỉnh theo mức giá tài sản; không coi MAE thấp hơn ở một loại nhà là bằng chứng mô hình công bằng hơn. Nhóm ít mẫu như Biệt thự cần diễn giải thận trọng. Xem segment_errors.csv và largest_errors.csv để tìm hiểu cụ thể.

## Artifact và kiểm tra

- Artifact: modelling/v2/model.joblib, giữ cả preprocessing và phép log1p/expm1, trả trực tiếp tỷ VND.
- RF: 160 cây, max_depth=24, max_features=1.0, min_samples_leaf=2, random_state=42.
- Ridge: alpha=100, solver=lsqr. Grid này có giới hạn, không khẳng định tối ưu toàn cục.
- 10 tests đã qua: 5 preprocessing và 5 về writer CSV, input, train-only imputation, artifact parity và Streamlit.
- AppTest kiểm tra mở app, dự đoán đơn, batch, giữ kết quả batch qua rerun và bộ lọc không có kết quả. Đây không phải kiểm thử hiển thị bằng ảnh chụp trình duyệt.
- Biểu đồ phân phối train đã được kiểm tra trực quan; health endpoint Streamlit trả HTTP 200.
- Raw không bị thay đổi. Chưa crawl live, chưa xác nhận nguồn gốc Muaban và chưa gộp thủ công tất cả nhóm nghi trùng.

## Bước tiếp theo

Đối chiếu các ca sai số lớn với tin nguồn, kiểm tra đơn vị/giá cực trị và tính độc lập tài sản; cân nhắc nghiên cứu phân khúc cụ thể dựa trên train. Khi tiếp tục điều chỉnh dựa trên kết quả test này, cần một holdout mới hoặc quy trình đánh giá ngoài khác để tránh thích nghi quá mức với test cũ. Cập nhật slide/report từ kết quả mới, không giữ các điểm số cũ như kết quả cuối.
