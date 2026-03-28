import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import rmse, mae
from fastai.optimizer import Adam
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
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

# =====================================================================
# REPRODUCIBILITY
# =====================================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU: let cuDNN auto-select the fastest algorithm for fixed input shapes
torch.backends.cudnn.benchmark = True

# =====================================================================
# CONFIGURATION — Champion from GPU Grid Search (tune_lstm_gpu.py)
# Data=Engineered (14 pts) | Layers=1 | Nodes=64 | Val MAE=0.0874 Ah
# =====================================================================
DATA_FILE       = Path("data/dataset_soh_nasa.csv")
TRAIN_BATTERIES = ["B0005", "B0006"]
VALID_BATTERY   = "B0018"
TEST_BATTERY    = "B0007"             # Zero-shot — never seen during training or search

BATCH_SIZE    = 16
LEARNING_RATE = 1e-2
EPOCHS        = 250
PATIENCE      = 20
WINDOW_SIZE   = 100

# Champion hyperparameters
LSTM_HIDDEN_NODES = 64
LSTM_NUM_LAYERS   = 1

# =====================================================================
# FEATURE ENGINEERING (14 pts: 13 windows + 1 anchor)
# =====================================================================
def perform_feature_engineering(df):
    v_cols = [f'V{i+1}' for i in range(1300)]
    voltages = df[v_cols].values
    features_list = []
    for row in voltages:
        averages = [np.mean(row[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]) for i in range(13)]
        averages.append(row[-1])
        features_list.append(averages)
    return np.array(features_list)

class BatterySOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# =====================================================================
# LSTM MODEL ARCHITECTURE
# =====================================================================
class SOH_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_nodes=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_nodes,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.relu = nn.ReLU()
        self.fc   = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        out, _       = self.lstm(x)
        activated    = self.relu(out[:, -1, :])
        return self.fc(activated)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = LOG_DIR / f"test_lstm_generalization_gpu_{timestamp}.txt"
    log_handle = open(log_file, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)

    try:
        train_label = " + ".join(TRAIN_BATTERIES)
        print(f"\n🚀 PHASE 1: Training Champion LSTM | Train: {train_label} | Valid: {VALID_BATTERY}")
        print(f"📝 Log file : {log_file}")
        print(f"🕒 Started  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # GPU detection
        if torch.cuda.is_available():
            print(f"🖥️  GPU       : {torch.cuda.get_device_name(0)}")
            print(f"   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected — running on CPU")
        print()

        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Missing file: {DATA_FILE}")

        df_full = pd.read_csv(DATA_FILE)

        # --- MULTI-BATTERY DATA LOADING (No Leakage) ---
        df_train = df_full[df_full['Battery'].isin(TRAIN_BATTERIES)].copy()
        df_valid = df_full[df_full['Battery'] == VALID_BATTERY].copy()
        df_test  = df_full[df_full['Battery'] == TEST_BATTERY].copy()

        X_train_raw = perform_feature_engineering(df_train)
        y_train     = df_train['Capacity'].values
        X_valid_raw = perform_feature_engineering(df_valid)
        y_valid     = df_valid['Capacity'].values
        X_test_raw  = perform_feature_engineering(df_test)
        y_test      = df_test['Capacity'].values

        # Fit scaler ONLY on training data — transform valid and test separately
        scaler        = MinMaxScaler()
        X_train       = scaler.fit_transform(X_train_raw)
        X_valid       = scaler.transform(X_valid_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        print(f"📊 Train : {X_train.shape[0]} cycles ({train_label})")
        print(f"📊 Valid : {X_valid.shape[0]} cycles ({VALID_BATTERY}) [cross-battery, no leakage]")
        print(f"📊 Test  : {X_test_raw.shape[0]} cycles ({TEST_BATTERY}) [zero-shot, never seen]")
        print(f"\n🏆 Champion Architecture:")
        print(f"   Data        : Engineered (14 pts)")
        print(f"   LSTM Nodes  : {LSTM_HIDDEN_NODES} | Layers: {LSTM_NUM_LAYERS}")
        print(f"   Val MAE (B0018) : 0.0874 Ah\n")

        n_workers = max(1, (os.cpu_count() or 4) - 2)
        train_ds  = BatterySOHDataset(X_train, y_train)
        valid_ds  = BatterySOHDataset(X_valid, y_valid)
        dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=BATCH_SIZE,
                                     num_workers=n_workers, pin_memory=True)  # GPU: faster CPU->GPU transfer

        # --- TRAINING ---
        model = SOH_LSTM_Model(
            input_size=1,
            hidden_nodes=LSTM_HIDDEN_NODES,
            num_layers=LSTM_NUM_LAYERS
        )

        callbacks = [
            EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0001, patience=PATIENCE),
            SaveModelCallback(monitor='valid_loss', fname='best_lstm_champion_gpu')
        ]

        learn = Learner(dls, model, loss_func=nn.HuberLoss(), opt_func=Adam,
                        metrics=[rmse, mae], cbs=callbacks)

        t0_train = time.time()
        print(f"🧠 Training Champion LSTM (Max {EPOCHS} epochs, LR={LEARNING_RATE})...")
        learn.fit(n_epoch=EPOCHS, lr=LEARNING_RATE)
        train_duration = time.time() - t0_train

        actual_epochs = len(learn.recorder.values)
        print(f"✅ Training completed in {train_duration:.2f}s over {actual_epochs} epochs.")
        print(f"💾 Best model saved to 'models/best_lstm_champion_gpu.pth'")

        # =====================================================================
        # PHASE 2: ZERO-SHOT GENERALIZATION ON B0007
        # =====================================================================
        print(f"\n🚀 PHASE 2: Zero-Shot Generalization Test on {TEST_BATTERY}...")
        t0_test = time.time()

        test_ds = BatterySOHDataset(X_test_scaled, y_test)
        test_dl = dls.test_dl(test_ds, with_labels=True)

        preds_tensor, targets_tensor = learn.get_preds(dl=test_dl)
        preds   = preds_tensor.numpy().flatten()
        targets = targets_tensor.numpy().flatten()

        final_r2   = r2_score(targets, preds)
        final_rmse = np.sqrt(mean_squared_error(targets, preds))
        final_mae  = mean_absolute_error(targets, preds)
        eval_duration = time.time() - t0_test

        # --- REPORTING ---
        table = f"""
    ╔════════════════════════════════════════════════════════════╗
    ║         CHAMPION LSTM GENERALIZATION TEST (B0007)          ║
    ╠════════════════════════════════════════════════════════════╣
    ║ ⚙️  ARCHITECTURE (Grid Search Winner)                        ║
    ║   • Data Input      : Engineered (14 pts)                  ║
    ║   • LSTM Nodes      : {LSTM_HIDDEN_NODES:<32} ║
    ║   • LSTM Layers     : {LSTM_NUM_LAYERS:<32} ║
    ║   • Grid Search MAE : 0.0874 Ah (on B0018 validation)      ║
    ╠════════════════════════════════════════════════════════════╣
    ║ 📊 DATA SPLIT                                               ║
    ║   • Train Batteries : {train_label:<32} ║
    ║   • Valid Battery   : {VALID_BATTERY:<32} ║
    ║   • Test Battery    : {TEST_BATTERY} (zero-shot){" " * 21}║
    ╠════════════════════════════════════════════════════════════╣
    ║ 📈 PERFORMANCE METRICS (Zero-Shot — Strict No-Leakage)       ║
    ║   • R-Squared (R²)  : {final_r2 * 100:>6.2f} % {" " * 25}║
    ║   • RMSE            : {final_rmse:>6.4f} Ah {" " * 24}║
    ║   • MAE             : {final_mae:>6.4f} Ah {" " * 24}║
    ╠════════════════════════════════════════════════════════════╣
    ║ ⏱️  EXECUTION TIMES                                          ║
    ║   • Training        : {train_duration:.2f} seconds{" " * 22}║
    ║   • Evaluation      : {eval_duration:.2f} seconds{" " * 22}║
    ╚════════════════════════════════════════════════════════════╝
        """
        print(table)

        # --- VISUALIZATION ---
        print("📊 Generating True vs Predicted graph...")
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label='True Capacity (B0007)', color='red', linewidth=2)
        plt.plot(preds, label=f'Champion LSTM (N={LSTM_HIDDEN_NODES}, L={LSTM_NUM_LAYERS}, Engineered)',
                 color='blue', linestyle='dashed', linewidth=2)
        plt.title('Champion LSTM — Zero-Shot Generalization on B0007')
        plt.xlabel('Discharge Cycles')
        plt.ylabel('Capacity (Ah)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('b0007_champion_lstm_plot.png', dpi=150)
        print("✅ Graph saved as 'b0007_champion_lstm_plot.png'")

        print(f"\n🕒 Finished : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Full log saved to: {log_file}")

    finally:
        sys.stdout = sys.__stdout__
        log_handle.close()

if __name__ == "__main__":
    main()
