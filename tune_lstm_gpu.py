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
# LOGGING: mirror all stdout output to a timestamped log file
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

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU: let cuDNN auto-select the fastest algorithm for fixed input shapes
torch.backends.cudnn.benchmark = True

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_FILE       = Path("data/dataset_soh_nasa.csv")
TRAIN_BATTERIES = ["B0005", "B0006"]  # Multi-battery training
VALID_BATTERY   = "B0018"             # Held-out cross-battery validation
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-2
EPOCHS          = 250
PATIENCE        = 15

# THE GRID TO SEARCH
TEST_DATA_TYPES = [True, False]   # True = 14 Features, False = 325 Raw Points
TEST_LAYERS     = [1, 2]          # 1 or 2 LSTM layers
TEST_NODES      = [32, 64, 160]   # Comparing small networks vs the paper's 160

# =====================================================================
# DATA PREP FUNCTIONS
# =====================================================================
def get_data(df, use_fe=True):
    v_cols = [f'V{i+1}' for i in range(1300)]
    voltages = df[v_cols].values

    if not use_fe:
        # RAW DATA ABLATION: Take every 4th point to reduce 1300 down to 325
        return voltages[:, ::4]

    # FEATURE ENGINEERING: 13 windows + 1 anchor = 14 features
    WINDOW_SIZE = 100
    features_list = []
    for row in voltages:
        averages = [np.mean(row[i*WINDOW_SIZE : (i+1)*WINDOW_SIZE]) for i in range(13)]
        averages.append(row[-1])
        features_list.append(averages)
    return np.array(features_list)

class BatterySOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class SOH_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_nodes=160, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_nodes,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        activated_out = self.relu(out[:, -1, :])
        return self.fc(activated_out)

# =====================================================================
# MAIN GRID SEARCH
# =====================================================================
def main():
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = LOG_DIR / f"tune_lstm_gpu_{timestamp}.txt"
    log_handle = open(log_file, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)

    try:
        total_configs = len(TEST_DATA_TYPES) * len(TEST_LAYERS) * len(TEST_NODES)
        train_label   = " + ".join(TRAIN_BATTERIES)

        print(f"\n🚀 LSTM Grid Search (GPU) — {total_configs} configurations")
        print(f"📝 Log file : {log_file}")
        print(f"🕒 Started  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # GPU detection
        if torch.cuda.is_available():
            print(f"🖥️  GPU       : {torch.cuda.get_device_name(0)}")
            print(f"   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected — running on CPU")
        print()

        print(f"⚙️  Strategy : multi-battery cross-battery validation (leak-free scaler)")

        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Missing file: {DATA_FILE}")

        df_full        = pd.read_csv(DATA_FILE)
        df_train_full  = df_full[df_full['Battery'].isin(TRAIN_BATTERIES)].copy()
        df_valid_full  = df_full[df_full['Battery'] == VALID_BATTERY].copy()
        y_train_full   = df_train_full['Capacity'].values
        y_valid_full   = df_valid_full['Capacity'].values

        print(f"📊 Train    : {len(y_train_full)} cycles ({train_label})")
        print(f"📊 Valid    : {len(y_valid_full)} cycles ({VALID_BATTERY}) [cross-battery, no leakage]")
        print(f"   Data types : {TEST_DATA_TYPES} | Layers: {TEST_LAYERS} | Nodes: {TEST_NODES}")
        print(f"   Max Epochs : {EPOCHS} | Patience: {PATIENCE}\n")

        results    = []
        n_workers  = max(1, (os.cpu_count() or 4) - 2)
        config_num = 0

        for use_fe in TEST_DATA_TYPES:
            data_name = "Engineered (14 pts)" if use_fe else "Raw Subsampled (325 pts)"

            X_train_raw = get_data(df_train_full, use_fe=use_fe)
            X_valid_raw = get_data(df_valid_full, use_fe=use_fe)

            # Fit scaler on train only — no leakage into validation battery
            scaler  = MinMaxScaler()
            X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
            X_valid = scaler.transform(X_valid_raw).astype(np.float32)

            train_ds = BatterySOHDataset(X_train, y_train_full)
            valid_ds = BatterySOHDataset(X_valid, y_valid_full)
            dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=BATCH_SIZE,
                                         num_workers=n_workers, pin_memory=True)  # GPU: faster CPU->GPU transfer

            for layers in TEST_LAYERS:
                for nodes in TEST_NODES:
                    config_num += 1
                    print(f"[{config_num:02d}/{total_configs}] Data={data_name} | Layers={layers} | Nodes={nodes}")

                    model     = SOH_LSTM_Model(input_size=1, hidden_nodes=nodes, num_layers=layers)
                    callbacks = [EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0001, patience=PATIENCE)]
                    learn     = Learner(dls, model, loss_func=nn.HuberLoss(),
                                       opt_func=Adam, metrics=[rmse, mae], cbs=callbacks)

                    t0 = time.time()
                    with learn.no_bar():
                        learn.fit(n_epoch=EPOCHS, lr=LEARNING_RATE)
                    train_time = time.time() - t0

                    # Capture epoch count BEFORE validate() — validate() resets the recorder
                    actual_epochs = len(learn.recorder.values)
                    val_metrics   = learn.validate()
                    val_mae       = float(val_metrics[2])

                    print(f"   ✅ {actual_epochs} epochs | Val MAE: {val_mae:.4f} Ah | {train_time:.1f}s\n")

                    results.append({
                        'Data':     data_name,
                        'Layers':   layers,
                        'Nodes':    nodes,
                        'Val_MAE':  round(val_mae, 4),
                        'Epochs':   actual_epochs,
                        'Time(s)':  round(train_time, 1)
                    })

                    # Checkpoint after every config — search is resumable on crash
                    pd.DataFrame(results).to_csv(
                        LOG_DIR / f"tune_lstm_gpu_{timestamp}_checkpoint.csv",
                        index=False
                    )

        # =====================================================================
        # FINAL LEADERBOARD
        # =====================================================================
        df_results = pd.DataFrame(results).sort_values(by='Val_MAE')

        print("\n" + "="*65)
        print("🏆  LSTM GRID SEARCH LEADERBOARD (Sorted by Best MAE)")
        print("="*65)
        print(df_results.to_string(index=False))

        best = df_results.iloc[0]
        print(f"\n💡 CHAMPION:")
        print(f"   Data    : {best['Data']}")
        print(f"   Layers  : {int(best['Layers'])}")
        print(f"   Nodes   : {int(best['Nodes'])}")
        print(f"   Val MAE : {best['Val_MAE']:.4f} Ah (on {VALID_BATTERY})")
        print(f"   NOTE    : B0007 was never touched during this search (zero-shot test).")

        total_search_time = sum(r['Time(s)'] for r in results)
        print(f"\n⏱️  Total Search Time : {total_search_time/3600:.2f} hours")
        print(f"📝 Full log saved to : {log_file}")
        print(f"🕒 Finished          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        sys.stdout = sys.__stdout__
        log_handle.close()

if __name__ == "__main__":
    main()
