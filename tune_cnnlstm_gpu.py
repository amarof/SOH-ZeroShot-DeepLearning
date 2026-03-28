import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import rmse, mae
from fastai.optimizer import Adam
from fastai.callback.tracker import EarlyStoppingCallback
import random

# =====================================================================
# LOGGING
# =====================================================================
class Tee:
    """Writes all output to both the terminal and a log file simultaneously."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU: let cuDNN auto-select the fastest convolution algorithm
# (fixed input shape 1300 pts — one-time overhead, faster every epoch after)
torch.backends.cudnn.benchmark = True

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_FILE       = Path("data/dataset_soh_nasa.csv")
TRAIN_BATTERIES = ["B0005", "B0006"]
VALID_BATTERY   = "B0018"
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
EPOCHS          = 100
PATIENCE        = 10

# =====================================================================
# EXTENDED SEARCH SPACE (72 configurations total)
# =====================================================================
TEST_CNN_FILTERS  = [16, 32, 64]    # Number of CNN feature maps
TEST_LSTM_NODES   = [32, 64]        # LSTM hidden state size
TEST_KERNEL_SIZES = [3, 5, 7]       # CNN receptive field
TEST_POOL_SIZES   = [2, 4]          # Compression: 1300->650 (mild) vs 1300->325 (aggressive)
TEST_NUM_LAYERS   = [1, 2]          # Shallow vs stacked LSTM

# =====================================================================
# DATA PREP
# =====================================================================
def get_raw_data(df):
    v_cols = [f'V{i+1}' for i in range(1300)]
    return df[v_cols].values

class BatterySOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# =====================================================================
# HYBRID CNN-LSTM ARCHITECTURE (extended)
# =====================================================================
class CNN_LSTM_Model(nn.Module):
    def __init__(self, filters=16, kernel_size=5, pool_size=4, lstm_nodes=32, num_layers=1):
        super().__init__()

        # 1D CNN Feature Extractor
        # padding=kernel_size//2 ensures output length == input length before pooling
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        # MaxPool compresses the sequence: 1300 -> 1300/pool_size
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        # LSTM Sequence Learner (supports stacked layers)
        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_nodes,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(lstm_nodes, 1)

    def forward(self, x):
        # x: (batch, seq_len=1300, channels=1)
        x = x.transpose(1, 2)          # -> (batch, 1, 1300) for Conv1d
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)               # -> (batch, filters, 1300/pool_size)
        x = x.transpose(1, 2)          # -> (batch, 1300/pool_size, filters) for LSTM
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :]) # last timestep
        return self.fc(out)

# =====================================================================
# MAIN GRID SEARCH
# =====================================================================
def main():
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = LOG_DIR / f"tune_cnnlstm_gpu_{timestamp}.txt"
    log_handle = open(log_file, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)

    try:
        total_configs = (len(TEST_CNN_FILTERS) * len(TEST_LSTM_NODES) *
                         len(TEST_KERNEL_SIZES) * len(TEST_POOL_SIZES) * len(TEST_NUM_LAYERS))

        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Missing file: {DATA_FILE}")

        df_full  = pd.read_csv(DATA_FILE)
        df_train = df_full[df_full['Battery'].isin(TRAIN_BATTERIES)].copy()
        df_valid = df_full[df_full['Battery'] == VALID_BATTERY].copy()

        X_train_raw = get_raw_data(df_train)
        X_valid_raw = get_raw_data(df_valid)
        y_train     = df_train['Capacity'].values
        y_valid     = df_valid['Capacity'].values

        # Fit scaler ONLY on training data — no leakage into validation battery
        scaler  = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_valid = scaler.transform(X_valid_raw).astype(np.float32)

        train_label = " + ".join(TRAIN_BATTERIES)
        print(f"\n🚀 Extended CNN-LSTM Grid Search (GPU) — {total_configs} configurations")
        print(f"📝 Log file : {log_file}")
        print(f"🕒 Started  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # GPU detection
        if torch.cuda.is_available():
            print(f"🖥️  GPU       : {torch.cuda.get_device_name(0)}")
            print(f"   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected — running on CPU")
        print()

        print(f"📊 Train    : {train_label} ({len(y_train)} cycles)")
        print(f"📊 Valid    : {VALID_BATTERY} ({len(y_valid)} cycles) [cross-battery, no leakage]")
        print(f"   Filters: {TEST_CNN_FILTERS} | Nodes: {TEST_LSTM_NODES}")
        print(f"   Kernel:  {TEST_KERNEL_SIZES} | Pool: {TEST_POOL_SIZES} | Layers: {TEST_NUM_LAYERS}")
        print(f"   Max Epochs: {EPOCHS} | Patience: {PATIENCE}\n")

        n_workers = max(1, (os.cpu_count() or 4) - 2)
        train_ds  = BatterySOHDataset(X_train, y_train)
        valid_ds  = BatterySOHDataset(X_valid, y_valid)
        dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=BATCH_SIZE,
                                     num_workers=n_workers, pin_memory=True)  # GPU: faster CPU->GPU transfer

        results    = []
        config_num = 0

        for filters in TEST_CNN_FILTERS:
            for nodes in TEST_LSTM_NODES:
                for kernel_size in TEST_KERNEL_SIZES:
                    for pool_size in TEST_POOL_SIZES:
                        for num_layers in TEST_NUM_LAYERS:
                            config_num += 1
                            label = (f"Filters={filters} | Nodes={nodes} | "
                                     f"Kernel={kernel_size} | Pool={pool_size} | Layers={num_layers}")
                            print(f"[{config_num:02d}/{total_configs}] {label}")

                            model = CNN_LSTM_Model(
                                filters=filters,
                                kernel_size=kernel_size,
                                pool_size=pool_size,
                                lstm_nodes=nodes,
                                num_layers=num_layers
                            )
                            callbacks = [EarlyStoppingCallback(
                                monitor='valid_loss', min_delta=0.0001, patience=PATIENCE
                            )]
                            learn = Learner(dls, model, loss_func=nn.HuberLoss(),
                                            opt_func=Adam, metrics=[rmse, mae], cbs=callbacks)

                            t0 = time.time()
                            with learn.no_bar():
                                learn.fit(n_epoch=EPOCHS, lr=LEARNING_RATE)
                            train_time = time.time() - t0

                            # Capture epoch count BEFORE validate() — validate() resets the recorder
                            actual_epochs = len(learn.recorder.values)
                            val_metrics   = learn.validate()
                            val_mae       = float(val_metrics[2])

                            print(f"   ✅ {actual_epochs} epochs | Val MAE: {val_mae:.4f} Ah | {train_time:.0f}s\n")

                            results.append({
                                'Filters':  filters,
                                'Nodes':    nodes,
                                'Kernel':   kernel_size,
                                'Pool':     pool_size,
                                'Layers':   num_layers,
                                'Val_MAE':  round(val_mae, 4),
                                'Epochs':   actual_epochs,
                                'Time(s)':  round(train_time, 1)
                            })

                            # Checkpoint after every config — search is resumable on crash
                            pd.DataFrame(results).to_csv(
                                LOG_DIR / f"tune_cnnlstm_gpu_{timestamp}_checkpoint.csv",
                                index=False
                            )

        # =====================================================================
        # LEADERBOARD
        # =====================================================================
        df_results = pd.DataFrame(results).sort_values(by='Val_MAE')

        print("\n" + "="*75)
        print("🏆  EXTENDED CNN-LSTM GRID SEARCH LEADERBOARD (Top 10)")
        print("="*75)
        print(df_results.head(10).to_string(index=False))

        best = df_results.iloc[0]
        print(f"\n💡 CHAMPION ARCHITECTURE:")
        print(f"   CNN Filters : {int(best['Filters'])}")
        print(f"   LSTM Nodes  : {int(best['Nodes'])}")
        print(f"   Kernel Size : {int(best['Kernel'])}")
        print(f"   Pool Size   : {int(best['Pool'])}")
        print(f"   LSTM Layers : {int(best['Layers'])}")
        print(f"   Val MAE     : {best['Val_MAE']:.4f} Ah (on {VALID_BATTERY})")

        total_search_time = sum(r['Time(s)'] for r in results)
        print(f"\n⏱️  Total Search Time : {total_search_time/3600:.2f} hours")
        print(f"\n➡️  Update CNN_FILTERS, LSTM_NODES, kernel_size, pool_size, num_layers")
        print(f"   in test_cnnlstm_generalization.py and re-run for final B0007 evaluation.")
        print(f"   NOTE: B0007 was never touched during this search (zero-shot test).")
        print(f"\n📝 Full log saved to : {log_file}")
        print(f"🕒 Finished          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        sys.stdout = sys.__stdout__
        log_handle.close()

if __name__ == "__main__":
    main()
