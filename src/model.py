from __future__ import annotations

import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRNN(nn.Module):
    """CNN trích đặc trưng time-frequency, RNN học chuỗi theo thời gian."""

    def __init__(
        self,
        num_classes: int,
        cnn_channels: tuple[int, ...] = (16, 32, 64),
        rnn_hidden: int = 96,
        rnn_layers: int = 1,
        rnn_type: str = 'gru',
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in cnn_channels:
            blocks.append(Conv2dBlock(in_ch, out_ch, dropout=dropout * 0.4))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.rnn_input_size = cnn_channels[-1]
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if self.rnn_type == 'gru' else nn.LSTM
        if self.rnn_type not in {'gru', 'lstm'}:
            raise ValueError('rnn_type chỉ hỗ trợ gru hoặc lstm.')
        self.rnn = rnn_cls(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(rnn_hidden * direction_factor),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, time]
        x = self.cnn(x)           # [B, C, F', T']
        x = x.mean(dim=2)         # average over frequency -> [B, C, T']
        x = x.transpose(1, 2)     # [B, T', C]
        out, _ = self.rnn(x)      # [B, T', H]
        last = out[:, -1, :]      # dùng trạng thái cuối cùng cho classification
        return self.classifier(last)


def build_model(
    model_name: str,
    num_classes: int,
    rnn_type: str = 'gru',
    bidirectional: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name == 'crnn_tiny':
        return CRNN(
            num_classes=num_classes,
            cnn_channels=(8, 16),
            rnn_hidden=32,
            rnn_layers=1,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    if model_name == 'crnn_small':
        return CRNN(
            num_classes=num_classes,
            cnn_channels=(16, 32, 64),
            rnn_hidden=96,
            rnn_layers=1,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    if model_name == 'crnn_medium':
        return CRNN(
            num_classes=num_classes,
            cnn_channels=(32, 64, 128),
            rnn_hidden=128,
            rnn_layers=2,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    raise ValueError(f'Unsupported model_name: {model_name}')
