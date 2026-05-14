[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wQpVslL8)
# CSC4005 Lab 4 – CRNN for Environmental Sound Classification

Starter kit này dành cho **Lab 4 của Bài 5 – CRNN** trong CSC4005. Repo nối tiếp trực tiếp từ Lab 3:

- **Lab 3:** UrbanSound8K + MFCC/log-mel + 1D-CNN + W&B
- **Lab 4:** UrbanSound8K + log-mel spectrogram + CRNN + W&B

Bài toán vẫn giữ nguyên: **phân loại âm thanh môi trường** trên bộ **UrbanSound8K**. Điểm khác biệt là sinh viên chuyển từ tư duy “CNN học pattern cục bộ” sang tư duy “CNN trích đặc trưng + RNN học diễn biến theo thời gian”.

## 1. Mục tiêu

Sinh viên cần:

1. hiểu được vì sao âm thanh nên được biểu diễn như một chuỗi thời gian–tần số,
2. xây dựng được mô hình **CRNN** cho audio classification,
3. dùng **W&B** để log thí nghiệm,
4. so sánh kết quả CRNN với kết quả 1D-CNN ở Lab 3,
5. phân tích lỗi mô hình bằng learning curves và confusion matrix.

## 2. Cấu trúc repo

```text
csc4005_lab4_urbansound8k_crnn_starter/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── baseline_logmel_crnn.json
│   ├── fast_debug.json
│   └── extension_bilstm_crnn.json
├── docs/
│   ├── GITHUB_CLASSROOM_GUIDE.md
│   ├── WANDB_GUIDE.md
│   └── LAB_GUIDE_LAB4.md
├── notebooks/
│   └── lab4_demo.ipynb
├── outputs/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── ci/
    ├── check_structure.py
    └── smoke_train.py
```

## 3. Cài đặt

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dữ liệu

Repo này **không chứa thư mục `data/`**. Khi chạy, sinh viên phải truyền dữ liệu ngoài repo qua `--data_dir`.

Hỗ trợ cấu trúc UrbanSound8K chuẩn:

```text
UrbanSound8K/
├── metadata/
│   └── UrbanSound8K.csv
└── audio/
    ├── fold1/
    ├── fold2/
    └── ...
```

Có thể truyền đường dẫn tới thư mục đã giải nén hoặc file `.zip`.

Ví dụ:

```bash
python -m src.train --config configs/baseline_logmel_crnn.json --data_dir /duong_dan/UrbanSound8K
```

## 5. Chạy debug nhanh

Chạy thử để kiểm tra môi trường, dữ liệu và W&B:

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

## 6. Chạy baseline CRNN

```bash
python -m src.train \
  --config configs/baseline_logmel_crnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

Cấu hình baseline dùng:

- `feature_type = logmel`
- `target_sr = 16000`
- `duration = 4.0` giây
- `n_mels = 64`
- `model_name = crnn_small`
- `rnn_type = gru`
- `train_folds = 1,2,3,4,5,6,7,8`
- `val_folds = 9`
- `test_folds = 10`

## 7. Chạy bản mở rộng BiLSTM-CRNN

```bash
python -m src.train \
  --config configs/extension_bilstm_crnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

Bản mở rộng dùng BiLSTM để sinh viên quan sát sự khác biệt giữa GRU và LSTM hai chiều.

## 8. Output sau khi train

Mỗi run sẽ tạo thư mục:

```text
outputs/<run_name>/
```

bao gồm:

- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`

## 9. W&B

Tên project thống nhất:

```text
csc4005-lab4-urbansound8k-crnn
```

Log tối thiểu mỗi epoch:

- train_loss
- val_loss
- train_acc
- val_acc
- lr
- epoch_time_sec

Log cuối run:

- best_val_acc
- test_acc
- avg_epoch_time_sec
- total_params
- trainable_params
- confusion matrix image
- curves image

## 10. Checklist nộp bài

- [ ] Có ít nhất 1 run CRNN baseline trên W&B
- [ ] Có learning curves
- [ ] Có confusion matrix
- [ ] Có so sánh với kết quả Lab 3 1D-CNN
- [ ] Có nhận xét lớp âm thanh nào dễ nhầm
- [ ] Có đề xuất ít nhất 2 hướng cải thiện mô hình
