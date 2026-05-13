# Hướng dẫn W&B cho Lab 4

## 1. Đăng nhập

```bash
wandb login
```

Sau đó dán API key từ tài khoản W&B.

## 2. Project thống nhất

```text
csc4005-lab4-urbansound8k-crnn
```

Sinh viên không tự đặt tên project lung tung để giảng viên dễ kiểm tra.

## 3. Chạy có W&B

```bash
python -m src.train \
  --config configs/baseline_logmel_crnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --use_wandb
```

## 4. Cần quan sát gì trên W&B?

- train_loss
- val_loss
- train_acc
- val_acc
- learning rate
- epoch time
- test_acc
- confusion matrix image
- curves image

## 5. Khi mạng Internet không ổn định

Có thể chạy offline tạm thời:

```bash
wandb offline
```

Sau đó đồng bộ lại:

```bash
wandb sync wandb/offline-run-*
```

Tuy nhiên khi nộp bài, sinh viên vẫn phải có link W&B dashboard.
