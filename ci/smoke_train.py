from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / '.tmp_ci_urbansound'
if TMP.exists():
    shutil.rmtree(TMP)
(TMP / 'metadata').mkdir(parents=True, exist_ok=True)
for fold in range(1, 5):
    (TMP / 'audio' / f'fold{fold}').mkdir(parents=True, exist_ok=True)

classes = ['air_conditioner', 'dog_bark', 'siren']
rng = np.random.default_rng(42)
rows = []
sr = 8000
duration = 0.5
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

for class_id, class_name in enumerate(classes):
    base_freq = 220 + class_id * 180
    for fold in range(1, 5):
        for idx in range(4):
            filename = f'{class_name}_{fold}_{idx}.wav'
            y = 0.25 * np.sin(2 * np.pi * (base_freq + idx * 8) * t)
            y += 0.02 * rng.normal(size=t.shape)
            y = np.clip(y, -1.0, 1.0)
            wavfile.write(TMP / 'audio' / f'fold{fold}' / filename, sr, (y * 32767).astype(np.int16))
            rows.append({
                'slice_file_name': filename,
                'fsID': idx,
                'start': 0,
                'end': duration,
                'salience': 1,
                'fold': fold,
                'classID': class_id,
                'class': class_name,
            })

pd.DataFrame(rows).to_csv(TMP / 'metadata' / 'UrbanSound8K.csv', index=False)

cmd = [
    sys.executable,
    '-m',
    'src.train',
    '--data_dir', str(TMP),
    '--config', 'configs/fast_debug.json',
    '--run_name', 'ci_smoke_crnn',
    '--train_folds', '1,2',
    '--val_folds', '3',
    '--test_folds', '4',
    '--epochs', '1',
    '--batch_size', '8',
    '--target_sr', '8000',
    '--duration', '0.5',
    '--n_mels', '32',
    '--n_fft', '256',
    '--hop_length', '128',
    '--no_wandb',
]
subprocess.run(cmd, cwd=ROOT, check=True)
print('Smoke train OK')
