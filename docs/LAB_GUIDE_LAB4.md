# Hướng dẫn Lab 4 – CRNN cho UrbanSound8K

## 1. Bối cảnh

Trong Lab 3, sinh viên đã dùng 1D-CNN để phân loại âm thanh môi trường. Cách tiếp cận này giúp mô hình học được các pattern cục bộ trong chuỗi đặc trưng audio.

Tuy nhiên, nhiều âm thanh không chỉ được nhận diện bởi một lát cắt ngắn. Ví dụ tiếng còi xe, tiếng chó sủa, tiếng khoan, tiếng còi báo động đều có diễn biến theo thời gian. Vì vậy, ở Lab 4, ta dùng CRNN:

```text
Audio → log-mel spectrogram → CNN blocks → RNN/GRU/LSTM → classifier
```

CNN trích đặc trưng từ biểu diễn time-frequency. RNN học quan hệ theo thời gian giữa các đặc trưng đó.

## 2. Mục tiêu

Sau bài thực hành, sinh viên cần:

1. tạo được log-mel spectrogram từ audio,
2. giải thích được kiến trúc CRNN,
3. train được mô hình CRNN trên UrbanSound8K,
4. log thí nghiệm lên W&B,
5. so sánh CRNN với 1D-CNN ở Lab 3.

## 3. Chuẩn bị dữ liệu

Tải UrbanSound8K và giải nén theo cấu trúc:

```text
UrbanSound8K/
├── metadata/
│   └── UrbanSound8K.csv
└── audio/
    ├── fold1/
    ├── fold2/
    └── ...
```

Không đưa dữ liệu vào GitHub repo.

## 4. Chạy kiểm tra nhanh

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

Nếu lệnh này chạy được, môi trường và dữ liệu cơ bản đã ổn.

## 5. Chạy baseline chính

```bash
python -m src.train \
  --config configs/baseline_logmel_crnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

Baseline dùng GRU một chiều để giảm độ phức tạp, giúp lab chạy ổn trên máy cá nhân.

## 6. Chạy bản mở rộng

```bash
python -m src.train \
  --config configs/extension_bilstm_crnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

Bản này dùng BiLSTM. Sinh viên cần so sánh với baseline GRU.

## 7. Phân tích kết quả

Sau khi train xong, xem thư mục:

```text
outputs/<run_name>/
```

Cần đọc:

- `history.csv`: loss/accuracy theo epoch
- `curves.png`: learning curves
- `confusion_matrix.png`: ma trận nhầm lẫn
- `metrics.json`: kết quả tổng hợp

## 8. Câu hỏi tự kiểm tra

1. Vì sao CRNN phù hợp với audio hơn CNN thuần trong một số trường hợp?
2. CNN trong CRNN học gì?
3. RNN trong CRNN học gì?
4. Vì sao dùng log-mel spectrogram thay vì raw waveform cho baseline chính?
5. Lớp âm thanh nào dễ nhầm nhất? Vì sao?
6. CRNN có nhất thiết luôn tốt hơn 1D-CNN không?
