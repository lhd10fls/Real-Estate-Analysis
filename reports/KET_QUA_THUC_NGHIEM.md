# Kết quả thực nghiệm và đánh giá Pipeline

## Dữ liệu và thiết kế đánh giá

Giữ nguyên 13.828 dòng raw (Batdongsan: 8.414 dòng, Muaban: 5.414 dòng). Loại 3.655 bản sao hoàn toàn (Batdongsan: 1.836, Muaban: 1.819), bảng canonical còn 10.173 dòng sạch. Tập mô hình gồm 9.694 dòng đủ điều kiện, chia cố định 7.728 train / 1.966 test theo nhóm ứng viên cùng tài sản (candidate groups). 19 nhóm URL xung đột (38 dòng) được cách ly khỏi dữ liệu mô hình. Những dòng giá/diện tích lớn không bị loại tùy tiện chỉ để cải thiện điểm.

Các model so sánh trên cùng 5 grouped folds. Imputer, encoder và scaler được học bên trong training fold. Biến vị trí (tỉnh/thành, quận/huyện) và thuộc tính thời gian đăng tin (posted_date, posting_month, day_of_week) đã được bổ sung đầy đủ. Không có price_per_m2, log_price hoặc trường raw chứa giá trong input. Model được chọn bằng MAE CV trên giá gốc trước khi đọc test để đánh giá.

## Kết quả thực nghiệm

| Model | CV MAE ± SD (tỷ VND) | Test MAE (tỷ) | Test RMSE (tỷ) | Test R² giá gốc |
|---|---:|---:|---:|---:|
| Baseline median | 15,004 ± 0,875 | 15,472 | 39,865 | -0,090 |
| Ridge | 20,402 ± 11,016 | 13,432 | 94,406 | -5,112 |
| Random Forest | 7,434 ± 0,523 | 7,063 | 21,923 | 0,670 |

Random Forest được chọn bằng CV. Sai số tuyệt đối trung vị (Median Absolute Error) trên test là 1,713 tỷ VND; MAE thấp hơn baseline khoảng 54,4%. MAE và RMSE vẫn lớn ở phân khúc giá cao, phản ánh tính phân tán mạnh của thị trường bất động sản thực tế.

Ridge không ổn định trên giá gốc: một số dự đoán log sau expm1 có thể bị khuếch đại mạnh, đặc biệt ngoài phạm vi train. Không sửa/xóa các ca test đó để cải thiện kết quả. Ứng dụng web dùng Random Forest được chọn và kiểm tra output hữu hạn/dương ở form đơn.

## Sai số theo loại nhà của Random Forest

| Loại | Số mẫu test | MAE (tỷ VND) | Median absolute error (tỷ) |
|---|---:|---:|---:|
| Chung cư | 335 | 2,194 | 0,796 |
| Nhà hẻm/ngõ | 354 | 2,251 | 1,002 |
| Nhà riêng | 248 | 2,506 | 1,394 |
| Biệt thự/liền kề | 310 | 9,738 | 4,003 |
| Nhà mặt phố | 689 | 12,027 | 2,869 |
| Biệt thự | 30 | 14,228 | 5,076 |

Phân khúc Chung cư và Nhà ngõ/hẻm có sai số trung vị thấp nhất (dưới 1 tỷ VND), phù hợp với tính thanh khoản và biên độ giá ổn định của thị trường ở thực. Phân khúc Nhà mặt phố và Biệt thự có giá trị tài sản rất lớn (thường từ 20 đến hơn 100 tỷ) nên sai số tuyệt đối lớn hơn. Xem `segment_errors.csv` và `largest_errors.csv` để tìm hiểu chi tiết.

## Artifact và kiểm tra

- Artifact: `modelling/model.joblib`, lưu trữ trọn gói pipeline gồm imputer, one-hot encoder và Random ForestRegressor đã huấn luyện.
- Random Forest: 160 cây, `max_depth=24`, `max_features=1.0`, `min_samples_leaf=2`, `random_state=42`.
- Ridge: `alpha=100`, `solver=lsqr`.
- **11/11 tests đã qua**: 6 tests về tiền xử lý dữ liệu và kiểm định thời gian (`data_preprocessed/test_rebuild.py`), 5 tests về writer CSV, input validation, train-only imputation, artifact parity và Streamlit UI (`tests/`).
- AppTest kiểm tra mở app, dự đoán đơn, batch, giữ kết quả batch qua rerun và bộ lọc không có kết quả.
- Toàn bộ dữ liệu sạch 10.173 dòng đã sẵn sàng cho trực quan hóa EDA và ứng dụng Streamlit.
