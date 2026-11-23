# Chạy toàn bộ pipeline v2

## Các bước

Chạy trong thư mục gốc project với môi trường `.venv` đã cài dependencies:

Một lệnh cho xử lý dữ liệu, train/evaluation và kiểm thử: `.\.venv\Scripts\python.exe run_pipeline_v2.py`. Lệnh này tạo lại các output v2 và đánh giá lại trên cùng holdout; không dùng điểm test để điều chỉnh mô hình qua nhiều lần chạy. Các bước riêng lẻ:

```powershell
.\.venv\Scripts\python.exe data_preprocessed/rebuild_v2.py
.\.venv\Scripts\python.exe -m unittest discover -s data_preprocessed -p test_rebuild_v2.py -v
.\.venv\Scripts\python.exe train_v2.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run streamlit_v2.py
```

Để cài lại môi trường: tạo venv, rồi `python -m pip install -r requirements_v2.lock.txt`. File này ghi các dependency trực tiếp và phiên bản dùng khi train; không phải khóa toàn bộ dependency bắc cầu.

## Thay đổi so với phiên bản tham khảo

- Raw không thay đổi; preprocessing và danh sách trường được mô tả tại `data_preprocessed/README_v2.md`.
- 9.691 dòng đủ điều kiện mô hình, 7.725 train và 1.966 test. Dữ liệu cực trị hợp lệ được giữ, không lọc để tăng điểm.
- CV 5 folds theo candidate_group_id. Không dùng target để tạo nhóm. Trùng tài sản chưa nhận diện hết vẫn là giới hạn.
- Feature: diện tích, mặt tiền, số phòng ngủ/tắm, số tầng, thành phố, quận/huyện, loại nhà, pháp lý, nội thất. District được ghép với city trong Pipeline.
- Missing số học dùng median và indicator; categorical dùng imputer và one-hot có xử lý nhóm mới. Ridge có StandardScaler. Các phép học được fit bên trong training fold.
- Ridge/RF dùng log1p target và expm1 tự động qua TransformedTargetRegressor. predict() trả tỷ VND gốc.
- Baseline là median giá train. Chọn model có MAE CV nhỏ nhất, kể cả baseline nếu baseline thắng. Test chỉ được đọc sau khi selection.json đã lưu; không tối ưu lại bằng kết quả test.
- Grid có 4 alpha Ridge và 4 cấu hình RF (160 cây, depth=24, leaf=2/8, max_features=.7/1), seed RF=42. Đây là thử nghiệm giới hạn, không tuyên bố tối ưu toàn cục.
- Artifact giữ model đã fit trên train, không refit trên train+test; vì vậy kiểm tra app so với test_predictions tái hiện được.
- Crawler được sửa để append theo thứ tự header CSV hiện có và từ chối schema khác. Không chạy crawl live trong lần triển khai này.

## Kết quả và giao diện

`reports/v2/` chứa EDA train, bảng CV, metrics giá gốc, predicted/actual của test, sai số theo nguồn/loại nhà/khu vực/nhóm giá và 30 ca lỗi lớn nhất. Các chỉ số theo nhóm ít mẫu không đại diện cho hiệu năng ổn định của nhóm đó.

`modelling/v2/model.joblib` chứa Pipeline và metadata; `selection.json` ghi lựa chọn chỉ dựa trên CV. `metadata.json` có phiên bản thư viện và hash dataset. Chỉ load artifact được tin cậy.

`streamlit_v2.py` dùng dữ liệu/model mới; `streamlit_app.py` cũ được giữ để đối chiếu, vẫn chứa hành vi cũ và không phải app dùng để báo cáo kết quả v2. Trong tab dữ liệu/EDA, số liệu mô tả toàn bộ canonical; tab đánh giá chỉ hiển thị kết quả holdout cố định. Dự đoán batch trên dữ liệu lọc có thể gồm train, không được coi là đánh giá độc lập.

## Giới hạn

Giá chào bán không phải giá giao dịch. Nguồn công bố và đơn vị giá Muaban chưa xác nhận độc lập; giữ giả định tỷ VND theo project gốc. Tọa độ nguồn chưa xác thực; không geocode lại Muaban. Không có khoảng dự đoán được hiệu chỉnh. Bộ dữ liệu lịch sử không được coi là giá thị trường hiện tại. Chưa thay report PDF hoặc slide cũ; phải cập nhật chúng theo kết quả v2 trước khi trình bày.
