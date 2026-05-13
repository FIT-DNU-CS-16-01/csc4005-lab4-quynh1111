# Rubric chấm điểm CSC4005 Lab 4  
## UrbanSound8K Classification with CRNN

Lab này đánh giá khả năng phát triển mô hình **CRNN** cho phân loại âm thanh môi trường trên **UrbanSound8K**. Trọng tâm là hiểu cách kết hợp **CNN để trích đặc trưng time-frequency** và **RNN/GRU/LSTM để học quan hệ theo thời gian**, đồng thời so sánh với kết quả 1D-CNN ở Lab 3.



---

## Quy ước chấm chung

- Tổng điểm: **10 điểm**.
- Sinh viên phải nộp đúng hạn theo yêu cầu trên LMS/GitHub Classroom.
- Mọi kết quả thực nghiệm cần có **bằng chứng tái lập**: mã nguồn, cấu hình chạy, output, hình ảnh learning curves/confusion matrix, và link W&B.
- Điểm mô hình không chỉ dựa vào accuracy tuyệt đối. Giảng viên ưu tiên đánh giá: pipeline đúng, thí nghiệm có kiểm soát, phân tích lỗi có lý, và báo cáo có khả năng giải thích.
- Nếu không chạy được toàn bộ lab, sinh viên vẫn có thể được chấm phần đã hoàn thành nếu trình bày rõ lỗi, log lỗi, hướng xử lý, và phần nào đã kiểm chứng.

## Mức đánh giá chung

| Mức | Ý nghĩa |
|---|---|
| Xuất sắc | Hoàn thành đúng yêu cầu, có phân tích sâu, minh chứng rõ, kết quả tái lập tốt |
| Đạt tốt | Hoàn thành phần lớn yêu cầu, có kết quả và phân tích hợp lý |
| Đạt tối thiểu | Chạy được một phần chính, còn thiếu phân tích hoặc thiếu minh chứng |
| Chưa đạt | Không chạy được pipeline chính, thiếu file quan trọng, hoặc không có bằng chứng thực nghiệm |


## Bảng tiêu chí chấm điểm

| Tiêu chí | Điểm | Xuất sắc | Đạt tốt | Đạt tối thiểu | Chưa đạt |
|---|---:|---|---|---|---|
| 1. Cấu trúc repo và khả năng chạy lại | 1.0 | Repo đúng cấu trúc starter kit Lab 4, có config, docs, output; chạy lại được bằng lệnh trong README; không đưa dữ liệu lớn vào repo | Repo đúng phần lớn cấu trúc, chạy được sau chỉnh sửa nhỏ | Có file chính nhưng thiếu docs/config hoặc output | Không thể chạy lại |
| 2. Chuẩn bị dữ liệu và log-mel spectrogram | 1.5 | Đọc đúng UrbanSound8K, xử lý sampling rate/duration, tạo log-mel spectrogram đúng shape, chuẩn hóa hợp lý, kiểm tra class/fold | Pipeline dữ liệu và log-mel chạy đúng | Có log-mel nhưng thiếu kiểm tra shape/normalization | Không tạo được input đúng cho CRNN |
| 3. Thiết kế CRNN baseline | 2.5 | CRNN rõ ràng: CNN blocks trích đặc trưng, reshape đúng theo trục thời gian, GRU/LSTM xử lý sequence, classifier hợp lý; giải thích được luồng tensor | CRNN train được và kiến trúc hợp lý | Có mô hình gần CRNN nhưng reshape/sequence chưa giải thích rõ | Không xây dựng được CRNN |
| 4. Huấn luyện và cấu hình ổn định | 1.0 | Có seed, optimizer/scheduler/early stopping phù hợp, batch size và epoch hợp lý; tránh overfitting tốt | Huấn luyện ổn định, có validation | Huấn luyện được nhưng dễ overfit hoặc thiếu kiểm soát | Train lỗi hoặc kết quả không đáng tin cậy |
| 5. W&B logging | 1.5 | Có dashboard W&B đầy đủ: config, metric theo epoch, learning curves, confusion matrix, model params, runtime, link rõ ràng | Có log W&B các metric chính | Có W&B nhưng thiếu metric hoặc link chưa rõ | Không dùng W&B |
| 6. Đánh giá và phân tích kết quả | 1.0 | Có learning curves, confusion matrix, classification report; phân tích lỗi theo lớp và theo đặc điểm âm thanh; đề xuất cải thiện hợp lý | Có biểu đồ và nhận xét kết quả | Có kết quả nhưng phân tích còn chung chung | Không có phân tích |
| 7. So sánh với Lab 3 1D-CNN | 1.0 | So sánh công bằng với Lab 3: cùng dataset/split gần tương đương, nêu khác biệt về input, kiến trúc, tham số, thời gian train và kết quả | Có so sánh CRNN với 1D-CNN ở mức cơ bản | Có nhắc đến Lab 3 nhưng thiếu bảng/nhận xét rõ | Không so sánh với Lab 3 |
| 8. Bài mở rộng BiLSTM/GRU tuning | 0.5 | Có thử BiLSTM hoặc điều chỉnh hidden size/layers/dropout; phân tích ảnh hưởng | Có chạy một biến thể mở rộng | Có nêu ý tưởng nhưng chưa chạy được | Không đề cập |

## Checklist bằng chứng cần nộp

- [ ] Link GitHub repository hoặc GitHub Classroom submission.
- [ ] Link W&B project/run cho baseline CRNN.
- [ ] Config đã dùng cho log-mel + CRNN-GRU.
- [ ] File `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`.
- [ ] Báo cáo có mô tả shape luồng dữ liệu: audio → log-mel → CNN → sequence → GRU/LSTM → classifier.
- [ ] Bảng so sánh với Lab 3 1D-CNN.
- [ ] Nếu làm mở rộng: có run BiLSTM-CRNN hoặc cấu hình tuning riêng.

## Gợi ý trừ điểm

| Lỗi | Mức trừ gợi ý |
|---|---:|
| Đưa UrbanSound8K/audio gốc lên GitHub | -1.0 đến -2.0 |
| CNN output reshape sai nhưng vẫn ép chạy mà không giải thích | -1.0 đến -2.0 |
| Không chứng minh được mô hình có thành phần RNN/GRU/LSTM | -1.0 đến -1.5 |
| Không có W&B link | -1.0 đến -1.5 |
| Không so sánh với Lab 3 | -1.0 |
| Chỉ báo accuracy, không có learning curves hoặc confusion matrix | -0.5 đến -1.0 |
| Báo cáo không giải thích được vì sao CRNN phù hợp với audio sequence | -0.5 đến -1.0 |

## Điểm cộng khuyến khích

| Nội dung | Điểm cộng tối đa |
|---|---:|
| Có so sánh GRU, LSTM và BiLSTM | +0.5 |
| Có ablation CNN depth hoặc RNN hidden size | +0.5 |
| Có phân tích runtime/params giữa 1D-CNN và CRNN | +0.3 |
| Có trực quan hóa log-mel spectrogram của một số mẫu đúng/sai | +0.3 |
