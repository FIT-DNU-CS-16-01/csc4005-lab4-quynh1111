from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif'}
DEFAULT_URBANSOUND_CLASSES = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music',
]


@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    resolved_data_dir: str
    feature_shape: tuple[int, int, int]


def _parse_folds(value: str | Iterable[int] | None) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return {int(x.strip()) for x in value.split(',') if x.strip()}
    return {int(x) for x in value}


def _extract_zip_if_needed(data_path: Path) -> Path:
    if data_path.is_dir():
        return data_path
    if data_path.is_file() and data_path.suffix.lower() == '.zip':
        extract_root = data_path.parent / f'{data_path.stem}_extracted'
        marker = extract_root / '.extracted_ok'
        if not marker.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(extract_root)
            marker.write_text('ok', encoding='utf-8')
        return extract_root
    raise FileNotFoundError(f'Không tìm thấy dữ liệu tại: {data_path}')


def _find_metadata(root: Path) -> Path | None:
    candidates = [root / 'metadata' / 'UrbanSound8K.csv', root / 'UrbanSound8K.csv']
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob('UrbanSound8K.csv'))
    return matches[0] if matches else None


def _find_audio_path(root: Path, fold: int, filename: str) -> Path | None:
    candidates = [
        root / 'audio' / f'fold{fold}' / filename,
        root / f'fold{fold}' / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def _resolve_urbansound_samples(root: Path) -> tuple[list[dict], list[str]] | None:
    metadata_path = _find_metadata(root)
    if metadata_path is None:
        return None
    df = pd.read_csv(metadata_path)
    required = {'slice_file_name', 'fold', 'class'}
    if not required.issubset(df.columns):
        raise ValueError(f'Metadata thiếu cột bắt buộc: {required}')

    if 'classID' in df.columns:
        class_order = (
            df[['classID', 'class']]
            .drop_duplicates()
            .sort_values('classID')['class']
            .astype(str)
            .tolist()
        )
    else:
        class_order = [c for c in DEFAULT_URBANSOUND_CLASSES if c in set(df['class'].astype(str))]
        remaining = sorted(set(df['class'].astype(str)) - set(class_order))
        class_order.extend(remaining)
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}

    samples: list[dict] = []
    missing = 0
    for _, row in df.iterrows():
        filename = str(row['slice_file_name'])
        fold = int(row['fold'])
        class_name = str(row['class'])
        path = _find_audio_path(root, fold, filename)
        if path is None:
            missing += 1
            continue
        samples.append({
            'path': path,
            'label': class_to_idx[class_name],
            'class_name': class_name,
            'fold': fold,
        })
    if not samples:
        raise ValueError('Không tìm thấy file audio nào khớp với metadata UrbanSound8K.')
    if missing:
        print(f'Cảnh báo: bỏ qua {missing} dòng metadata do không tìm thấy file audio.')
    return samples, class_order


def _resolve_class_folder_samples(root: Path) -> tuple[list[dict], list[str]] | None:
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not class_dirs:
        return None
    samples: list[dict] = []
    class_names: list[str] = []
    for idx, class_dir in enumerate(class_dirs):
        audio_paths = [p for p in sorted(class_dir.rglob('*')) if p.suffix.lower() in AUDIO_EXTENSIONS]
        if not audio_paths:
            continue
        class_names.append(class_dir.name)
        for path in audio_paths:
            samples.append({'path': path, 'label': idx, 'class_name': class_dir.name, 'fold': None})
    if not samples:
        return None
    return samples, class_names


def _resolve_samples(data_dir: str | Path) -> tuple[list[dict], list[str], Path]:
    root = _extract_zip_if_needed(Path(data_dir))
    resolved = _resolve_urbansound_samples(root)
    if resolved is not None:
        samples, class_names = resolved
        return samples, class_names, root
    resolved = _resolve_class_folder_samples(root)
    if resolved is not None:
        samples, class_names = resolved
        return samples, class_names, root
    raise ValueError(
        'Không đọc được dữ liệu. Hỗ trợ UrbanSound8K chuẩn hoặc thư mục theo lớp chứa file audio.'
    )


def _load_audio(path: Path, target_sr: int, duration: float) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(target_sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def _augment_waveform(y: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        gain = float(np.random.uniform(0.8, 1.2))
        y = y * gain
    if np.random.rand() < 0.5:
        shift = int(np.random.uniform(-0.1, 0.1) * len(y))
        y = np.roll(y, shift)
    if np.random.rand() < 0.35:
        noise = np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
        y = y + noise
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _extract_logmel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa

    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.astype(np.float32)


def _extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa

    feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat.astype(np.float32)


class UrbanSoundFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        feature_type: str = 'logmel',
        target_sr: int = 16000,
        duration: float = 4.0,
        n_mels: int = 64,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 512,
        augment: bool = False,
    ):
        self.samples = samples
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        y = _load_audio(item['path'], target_sr=self.target_sr, duration=self.duration)
        if self.augment:
            y = _augment_waveform(y)
        if self.feature_type == 'logmel':
            feat = _extract_logmel(y, self.target_sr, self.n_mels, self.n_fft, self.hop_length)
        elif self.feature_type == 'mfcc':
            feat = _extract_mfcc(y, self.target_sr, self.n_mfcc, self.n_fft, self.hop_length)
        else:
            raise ValueError('feature_type chỉ hỗ trợ logmel hoặc mfcc cho Lab 4.')
        x = torch.from_numpy(feat[None, :, :])  # [1, frequency, time]
        return x, int(item['label'])


def _split_by_folds(samples: list[dict], train_folds, val_folds, test_folds):
    train_set = _parse_folds(train_folds)
    val_set = _parse_folds(val_folds)
    test_set = _parse_folds(test_folds)
    if train_set is None or val_set is None or test_set is None:
        return None
    if any(s.get('fold') is None for s in samples):
        return None
    train = [s for s in samples if int(s['fold']) in train_set]
    val = [s for s in samples if int(s['fold']) in val_set]
    test = [s for s in samples if int(s['fold']) in test_set]
    if not train or not val or not test:
        return None
    return train, val, test


def _random_stratified_split(samples: list[dict], val_size: float, test_size: float, seed: int):
    labels = [s['label'] for s in samples]
    indices = np.arange(len(samples))
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx], [samples[i] for i in test_idx]


def create_dataloaders(
    data_dir: str | Path,
    feature_type: str = 'logmel',
    target_sr: int = 16000,
    duration: float = 4.0,
    n_mels: int = 64,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
    batch_size: int = 32,
    train_folds: str | None = '1,2,3,4,5,6,7,8',
    val_folds: str | None = '9',
    test_folds: str | None = '10',
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment: bool = False,
    num_workers: int = 0,
) -> SplitData:
    samples, class_names, resolved_root = _resolve_samples(data_dir)
    split = _split_by_folds(samples, train_folds, val_folds, test_folds)
    if split is None:
        print('Không dùng được fold split. Chuyển sang stratified random split.')
        train_samples, val_samples, test_samples = _random_stratified_split(samples, val_size, test_size, random_state)
    else:
        train_samples, val_samples, test_samples = split

    common_kwargs = dict(
        feature_type=feature_type,
        target_sr=target_sr,
        duration=duration,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    train_ds = UrbanSoundFeatureDataset(train_samples, augment=augment, **common_kwargs)
    val_ds = UrbanSoundFeatureDataset(val_samples, augment=False, **common_kwargs)
    test_ds = UrbanSoundFeatureDataset(test_samples, augment=False, **common_kwargs)

    # Tính trước feature shape từ mẫu đầu tiên để log và kiểm tra mô hình.
    x0, _ = train_ds[0]
    feature_shape = tuple(x0.shape)

    return SplitData(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=class_names,
        resolved_data_dir=str(resolved_root),
        feature_shape=feature_shape,
    )
