# Tiền xử lý dữ liệu (Data Preprocessing Pipeline)

## Chạy lại

Từ thư mục gốc project, dùng môi trường Python có pandas và numpy:

```powershell
python data_preprocessed/rebuild_dataset.py
python -m unittest discover -s data_preprocessed -p test_rebuild.py -v
```

Script đọc hai CSV trong `data/raw`, không sửa raw CSV. Mỗi lần chạy ghi lại các output do script quản lý trong `data/processed`. SHA256 của raw được ghi trong `quality_report.json`.

## Quy tắc

- Loại bản sao giống toàn bộ cột gốc, trước khi thêm ID. Giữ bảng ánh xạ dòng bỏ về dòng giữ.
- Giữ mọi trường nguồn dưới tiền tố `raw__`; record_id là tên nguồn và số thứ tự dòng dữ liệu (bắt đầu từ 1, không tính header).
- Giá tính bằng tỷ VND, diện tích m², chiều rộng mét. Giá đơn vị Batdongsan chỉ suy ra tổng giá khi có diện tích hợp lệ; lưu price_status.
- Giá số của Muaban được hiểu là tỷ VND theo dataset/project tham khảo. Đây là giả định được công bố, chưa xác nhận độc lập với nhà cung cấp. File nằm trong raw không có nghĩa chưa từng được nhà cung cấp xử lý.
- Không tạo collection timestamp giả. Không gọi API geocoding: Muaban chưa có tọa độ được giữ thiếu, không dùng lại tọa độ suy ra từ pipeline cũ.
- Chuẩn hóa TP.HCM/TPHCM/TP HCM/Sài Gòn về Hồ Chí Minh trong trường thành phố; thống nhất biến thể Bà Rịa - Vũng Tàu. Không đổi địa giới hành chính lịch sử sang hiện tại.
- District chỉ thống nhất tiền tố hành chính; giữ cả district_raw và city để phân biệt quận/huyện cùng tên.
- Nhà riêng không bị đổi thành Nhà mặt phố. Giữ Biệt thự/liền kề riêng vì nhãn nguồn không phân biệt hai loại.
- Sổ đỏ/Sổ hồng không tự suy ra Sổ riêng. Pháp lý/nội thất chưa biết giữ Chưa rõ; văn bản không chắc chắn giữ Khác/chưa phân loại và giá trị raw.
- Missing số phòng/tầng giữ NaN. Giá trị âm, chiều rộng/diện tích bằng 0, số phòng/tầng không nguyên bị đánh dấu lỗi và chuyển thành NaN; giá trị 0 của số phòng/tầng từ nguồn vẫn giữ, chưa suy diễn đó là thiếu.
- Không tự xóa tài sản đắt tiền hoặc diện tích lớn; xuất danh sách extreme_values_review theo ngưỡng tham khảo (giá >500 tỷ, diện tích <10 hoặc >1500 m²), không dùng ngưỡng đó làm quy tắc loại train/test.
- 22 URL còn nhiều bản ghi khác nhau sau bỏ trùng toàn dòng được cách ly khỏi dữ liệu mô hình, không chọn bản mới nhất khi thiếu timestamp.

## Đầu ra

| File | Mục đích |
|---|---|
| canonical_all.csv | Bảng chuẩn hóa có truy vết, gồm cả tin thiếu giá và xung đột URL |
| batdongsan_canonical.csv, muaban_canonical.csv | Bảng chuẩn từng nguồn |
| exact_duplicates_removed.csv | ID dòng bỏ, ID dòng giữ và lý do |
| url_conflicts.csv | Các URL có nội dung khác nhau, cần kiểm tra thủ công |
| duplicate_candidates.csv | Các nhóm giống vị trí/đặc điểm, không tự xóa |
| parse_issues.csv | Giá trị không phân tích được hoặc không hợp lệ; file rỗng không chứng minh mọi dữ liệu đều đúng |
| extreme_values_review.csv | Các giá trị cực trị để xem lại |
| excluded_from_model.csv | Thiếu giá/diện tích hợp lệ hoặc xung đột URL; không bị xóa khỏi canonical |
| mapping_*.csv | Mapping thực tế địa danh, loại nhà, kèm số dòng |
| missingness_by_source.csv | Missing/unknown theo nguồn và cột |
| split_manifest.csv | ID và nhóm của từng dòng train/test |
| X_train.csv, X_test.csv | Features chưa impute/encode/scale; có record_id để join, không dùng ID làm feature |
| y_train.csv, y_test.csv | Nhãn price trên thang tỷ VND gốc, cùng thứ tự ID với X |
| quality_report.json | Số liệu tổng hợp, fingerprint raw, giới hạn |

## Chia tập và chống leakage

Nhóm nghi cùng tài sản dựa trên city, district, địa chỉ chuẩn hóa chữ/khoảng trắng, loại nhà, diện tích và số phòng/tầng; không dùng giá để tạo nhóm. Các nhóm này giữ nguyên khi chia bằng hash cố định với seed 42, test xấp xỉ 20%. Đây là kiểm soát chồng nhóm đã phát hiện, không phải chứng minh đã nhận diện mọi tài sản trùng. Các căn khác nhau cùng tòa nhà có thể bị gom nhóm khi thông tin giống nhau; chúng vẫn giữ thành các dòng riêng.

`canonical_all` có price_per_m2 và log_price phục vụ EDA; X_train/X_test chỉ chứa danh sách FEATURES tường minh, không có biến suy từ giá. Không dùng raw__price, URL, source_row_id, record_id hoặc toàn bộ canonical làm X.

Bước kế tiếp là fit imputer/encoder/scaler trong training fold của sklearn Pipeline, dùng nhóm từ split_manifest cho GroupKFold. Feature engineering chọn dựa trên EDA chỉ thực hiện trên train; khóa test sau khi chốt chính sách chất lượng. Việc thống nhất nhãn/đơn vị theo quy tắc cố định không cần fit dữ liệu.

Mô hình dùng nhãn log1p, đảo bằng expm1 tự động thông qua TransformedTargetRegressor; app không cần đảo thêm thủ công vì estimator đã trả giá gốc.

## Các việc vẫn cần đối chiếu thủ công

Nguồn công bố/thời gian thu thập Muaban; 22 URL xung đột; các nhóm nghi trùng; các giá trị cực trị; phần địa danh không có mapping; độ chính xác tọa độ nguồn và các nhãn pháp lý tự khai. Dataset lớn hơn bản 9.486 dòng cũ vì không lặp lại các bộ lọc và mapping thiếu căn cứ của pipeline cũ.
