"""
CAN-bus Normal Behavior Autoencoder + Attack Evaluation (SynCAN) - PYTORCH VERSION
Menu options:
1) Train model on normal data
2) Evaluate attack type 1 (suppress)
3) Evaluate attack type 2 (flooding)
4) Exit
"""
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

# =========================
#  PYTORCH IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check for Metal/MPS support (Apple Silicon GPU acceleration)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================
#  CONFIGURATION
# =========================

# Folder of your dataset
DATA_DIR = DATA_DIR = Path(".")

# Training on NORMAL data only
TRAIN_ZIP_FILES = [
    "test_normal.zip",   # make sure this name matches exactly
]

# Attack files to evaluate (menu choice -> (description, zip_name))
ATTACK_FILES = {
    "2": ("plateau",    "test_plateau.zip"),
    "3": ("continuous", "test_continuous.zip"),
    "4": ("playback",   "test_playback.zip"),
    "5": ("suppress",   "test_suppress.zip"),
    "6": ("flooding",   "test_flooding.zip"),
}


# Columns expected in each CSV
LABEL_COL = "Label"
ID_COL = "ID"
TIME_COL = "Time"
SIGNAL_COLS = [
    "Signal1_of_ID",
    "Signal2_of_ID",
    "Signal3_of_ID",
    "Signal4_of_ID",
]

# Train/validation split ratio
VAL_RATIO = 0.2

# Autoencoder training params
EPOCHS = 50
BATCH_SIZE = 256
THRESHOLD_PERCENTILE = 95


# =========================
#  DATA LOADING
# =========================

def load_zip_to_df(zip_name: str) -> pd.DataFrame:
    """
    Load a single ZIP file that contains exactly one CSV with SynCAN format.
    """
    zpath = DATA_DIR / zip_name
    print(f"Loading ZIP: {zpath}")

    with zipfile.ZipFile(zpath, "r") as zf:
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(
                f"Expected exactly 1 CSV inside {zpath}, found {csv_names}"
            )
        csv_name = csv_names[0]
        print(f"  -> Found CSV inside ZIP: {csv_name}")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"Rows loaded from {zip_name}: {len(df)}")
    return df


def load_train_data() -> pd.DataFrame:
    """
    Load and concatenate all training ZIPs (currently: only normal data).
    """
    dfs = []
    for zname in TRAIN_ZIP_FILES:
        df = load_zip_to_df(zname)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total training rows loaded: {len(df_all)}")
    return df_all


# --- Replacement for print_alert_message ---

def format_alert_message(df_attack: pd.DataFrame, per_id: pd.DataFrame, alert_threshold: float = 0.05) -> str:
    """
    Computes and returns the formatted textual alert interface.
    """
    output_lines = []
    
    # Global anomaly rate over all messages
    global_anomaly_rate = df_attack["is_anomaly"].mean()

    output_lines.append("--- ALERT INTERFACE ---")
    output_lines.append(f"Global anomaly rate: {global_anomaly_rate * 100:.2f}%")

    # Always show top suspicious IDs by anomaly rate
    suspicious = per_id.sort_values("anomaly_rate_%", ascending=False).head(3)

    output_lines.append("\nTop suspicious IDs:")
    for _, row in suspicious.iterrows():
        output_lines.append(
            f"  - ID {row[ID_COL]}:"
            f" anomaly_rate={row['anomaly_rate_%']:.2f}%,"
            f" attack_fraction={row['attack_fraction_%']:.2f}%"
        )

    # Main suspect (highest anomaly rate)
    if not suspicious.empty:
        main_suspect = suspicious.iloc[0]
        output_lines.append(
            f"\nMain suspect among all IDs: ID {main_suspect[ID_COL]} "
            f"(anomaly likelihood ≈ {main_suspect['anomaly_rate_%']:.2f}%)"
        )


    # Overall status
    if global_anomaly_rate < alert_threshold:
        output_lines.append("\n>> STATUS: Traffic appears normal. No intrusion detected (below threshold).")
    else:
        output_lines.append("\n!!! ALERT: Possible intrusion detected on CAN bus !!!")
        output_lines.append(f">> Global anomaly rate is above threshold ({alert_threshold * 100:.1f}%).")

    output_lines.append("------------------------")
    
    return "\n".join(output_lines)


# =========================
#  FEATURE ENGINEERING
# =========================

def encode_id_column(df: pd.DataFrame):
    """
    Map each CAN ID to an integer index: ID_idx.
    """
    unique_ids = df[ID_COL].unique()
    id2idx = {v: i for i, v in enumerate(df[ID_COL].unique())}
    df["ID_idx"] = df[ID_COL].map(id2idx).astype(int)


    print(f"Encoded {len(unique_ids)} unique IDs into ID_idx.")
    return df, id2idx


def add_delta_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ID, compute DeltaTime = Time difference between consecutive messages.
    Missing values (first message of each ID) are filled with 0.
    """
    df = df.sort_values(by=[ID_COL, TIME_COL], ascending=True)
    df["DeltaTime"] = df.groupby(ID_COL)[TIME_COL].diff()
    df["DeltaTime"] = df["DeltaTime"].fillna(0.0)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Build the feature matrix X from:
    - ID_idx
    - DeltaTime
    - the 4 signal columns
    """
    feature_cols = ["ID_idx", "DeltaTime"] + SIGNAL_COLS
    X = df[feature_cols].values
    return X, feature_cols


# =========================
#  AUTOENCODER MODEL (PYTORCH)
# =========================

class Autoencoder(nn.Module):
    """
    PyTorch implementation of the Autoencoder
    """
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),  # bottleneck
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            # nn.Identity() # linear output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray) -> Autoencoder:
    """
    Train the PyTorch autoencoder with live batch progress using tqdm.
    """
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim).to(DEVICE)
    
    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting PyTorch training...\n")
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        
        # Live batch progress with tqdm
        for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100):
            inputs = inputs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * inputs.size(0)
        
        # Average training loss
        train_loss_avg = train_loss_sum / len(train_dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            val_loss_avg = criterion(outputs_val, X_val_tensor).item()
        
        # Epoch summary
        print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss_avg:.6e} | Val Loss: {val_loss_avg:.6e}")

        # Early Stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), "best_autoencoder_pytorch.pth")
            print("  ✅ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("  ⚠ Early stopping triggered. Loading best model weights.")
                model.load_state_dict(torch.load("best_autoencoder_pytorch.pth"))
                break

    print("\nTraining complete.\n")
    return model



# =========================
#  ANOMALY THRESHOLD (PYTORCH)
# =========================

def compute_reconstruction_errors(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Compute reconstruction errors using the PyTorch model.
    """
    model.eval()
    
    # Convert numpy to tensor, ensure it's on the correct device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        X_pred_tensor = model(X_tensor)
    
    # Compute MSE: mean((X - X_pred)**2)
    errors = torch.mean((X_tensor - X_pred_tensor)**2, dim=1)
    
    # Move results back to CPU and convert to numpy
    return errors.cpu().numpy()


def choose_threshold(errors: np.ndarray, percentile: float) -> float:
    """
    Vectorized percentile threshold selection.
    """
    return np.percentile(errors, percentile)


# =========================
#  WRAPPER CLASS
# =========================
class CanAnomalyDetector:
    """
    Holds:
    - ID mapping
    - scaler
    - trained autoencoder
    - anomaly threshold
    """

    def __init__(self, id2idx, scaler, model, threshold, feature_cols):
        self.id2idx = id2idx
        self.scaler = scaler
        self.model = model
        self.threshold = threshold
        self.feature_cols = feature_cols
        # Ensure model is on the correct device
        self.model.to(DEVICE)

    def transform_dataframe(self, df: pd.DataFrame):
        """
        Fast transformation with vectorized ops & fewer conversions.
        """
        df = df.copy()

        # Clean signals
        df[SIGNAL_COLS] = df[SIGNAL_COLS].fillna(0)

        # Time numeric
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0)

        # ID encode (+ unknown = -1)
        df["ID_idx"] = df[ID_COL].map(self.id2idx).fillna(-1).astype(int)

        # Sort once
        df.sort_values([ID_COL, TIME_COL], inplace=True)

        # DeltaTime per ID
        df["DeltaTime"] = df.groupby(ID_COL)[TIME_COL].diff().fillna(0)

        # Build feature matrix
        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=1e9, neginf=-1e9)

        # Scale
        X_scaled = self.scaler.transform(X)

        return df, X_scaled


    def anomaly_score(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score (reconstruction error).
        """
        errors = compute_reconstruction_errors(self.model, X_scaled)
        return errors

    def is_anomaly(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Return boolean array: True where anomaly is detected.
        """
        errors = self.anomaly_score(X_scaled)
        return errors > self.threshold


# =========================
#  TRAINING PIPELINE
# =========================

def train_detector() -> CanAnomalyDetector:
    """
    Full pipeline: load normal data, preprocess, train autoencoder, choose threshold,
    and return a CanAnomalyDetector object.
    """
    print("\n=== TRAINING MODEL ON NORMAL DATA (PYTORCH) ===")

    # 1. Load training data (all NORMAL)
    df_all = load_train_data()

    # 1.a Clean NaN in signals
    df_all[SIGNAL_COLS] = df_all[SIGNAL_COLS].fillna(0.0)

    # 1.b Ensure Time is numeric
    df_all[TIME_COL] = pd.to_numeric(df_all[TIME_COL], errors="coerce").fillna(0.0)

    # 2. Encode ID -> ID_idx
    df_all, id2idx = encode_id_column(df_all)

    # 3. Add DeltaTime per ID
    df_all = add_delta_time(df_all)

    # 4. Build feature matrix
    X_raw, feature_cols = build_feature_matrix(df_all)
    print("Feature columns:", feature_cols)
    print("X_raw shape:", X_raw.shape)

    # 4.a Clean NaN / Inf at matrix level (safety net)
    print("NaNs in X_raw BEFORE nan_to_num:", np.isnan(X_raw).sum())
    print("Infs in X_raw BEFORE nan_to_num:", np.isinf(X_raw).sum())

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e9, neginf=-1e9)

    print("NaNs in X_raw AFTER nan_to_num:", np.isnan(X_raw).sum())
    print("Infs in X_raw AFTER nan_to_num:", np.isinf(X_raw).sum())

    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print("X_scaled shape:", X_scaled.shape)
    print("NaNs in X_scaled:", np.isnan(X_scaled).sum())
    print("Infs in X_scaled:", np.isinf(X_scaled).sum())

    # 6. Train/validation split (all normal for now)
    X_train, X_val = train_test_split(
        X_scaled, test_size=VAL_RATIO, random_state=42, shuffle=True
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # 7. Train autoencoder on normal data (PyTorch version)
    model = train_autoencoder(X_train, X_val)

    # 8. Compute reconstruction error on training data and choose threshold
    train_errors = compute_reconstruction_errors(model, X_train)
    print("NaNs in train_errors:", np.isnan(train_errors).sum())
    print("Infs in train_errors:", np.isinf(train_errors).sum())

    threshold = choose_threshold(train_errors, THRESHOLD_PERCENTILE)
    print(f"Chosen anomaly threshold (percentile {THRESHOLD_PERCENTILE}): {threshold}")

    # 9. Wrap everything into a detector object
    detector = CanAnomalyDetector(
        id2idx=id2idx,
        scaler=scaler,
        model=model,
        threshold=threshold,
        feature_cols=feature_cols,
    )

    print("=== TRAINING DONE ===\n")
    return detector


# =========================
#  ATTACK EVALUATION
# =========================

def evaluate_attack_file(detector: CanAnomalyDetector, zip_name: str) -> dict:
    """
    Load one attack ZIP, run it through the detector, compute per-ID stats,
    and return both the alert message and list of anomalies.
    """
    # --- Load data ---
    df_attack_raw = load_zip_to_df(zip_name)
    df_attack, X_attack_scaled = detector.transform_dataframe(df_attack_raw)

    # --- Compute anomalies ---
    errors = detector.anomaly_score(X_attack_scaled)
    is_anom = errors > detector.threshold
    df_attack["is_anomaly"] = is_anom
    df_attack["recon_error"] = errors

    # --- True labels ---
    y_true = df_attack[LABEL_COL].astype(int).values
    y_pred = is_anom.astype(int)

    # --- Confusion matrix stats ---
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    attack_total = tp + fn
    detection_rate = tp / attack_total if attack_total > 0 else float("nan")
    normal_total = tn + fp
    false_positive_rate = fp / normal_total if normal_total > 0 else float("nan")

    # --- Per-ID analysis ---
    per_id = (
        df_attack.groupby(ID_COL)
        .agg(
            mean_error=("recon_error", "mean"),
            anomaly_rate=("is_anomaly", "mean"),
            attack_fraction=(LABEL_COL, lambda x: (x == 1).mean()),
        )
        .reset_index()
    )
    per_id["anomaly_rate_%"] = per_id["anomaly_rate"] * 100.0
    per_id["attack_fraction_%"] = per_id["attack_fraction"] * 100.0
    per_id_sorted = per_id.sort_values("anomaly_rate_%", ascending=False)

    # --- Top suspicious IDs ---
    top_ids = per_id_sorted.head(5)[ID_COL].tolist()  # top 5 suspicious IDs

    # --- Alert message ---
    alert_message = format_alert_message(df_attack, per_id, alert_threshold=0.05)

    # --- Return both alert + suspicious IDs for frontend ---
    return {
        "alert_message": alert_message,
        "per_id": per_id_sorted,
        "top_ids": top_ids
    }



def summarize_attack_windows(df_attack: pd.DataFrame, window_sec: float = 1.0, top_k_ids: int = 1):
    """
    Summarize suspicious time windows (implementation omitted for brevity, logic unchanged from original).
    """
    # Logic for summarize_attack_windows remains largely the same, only needing
    # NumPy/Pandas operations, not PyTorch.
    print("Summarize attack windows function is present but not fully shown in output.")


# =========================
#  MENU
# =========================

def print_menu():
    print("\n=== MAIN MENU ===")
    print("1. Train model on normal data")
    print("2. Evaluate attack type: plateau")
    print("3. Evaluate attack type: continuous")
    print("4. Evaluate attack type: playback")
    print("5. Evaluate attack type: suppress")
    print("6. Evaluate attack type: flooding")
    print("7. Exit")



def main():
    detector = None

    while True:
        print_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            # Train model
            detector = train_detector()

        elif choice in ATTACK_FILES:
            # Any attack type (2-6)
            if detector is None:
                print("Model is not trained yet. Please choose option 1 first.")
            else:
                desc, zip_name = ATTACK_FILES[choice]
                print(f"\n[INFO] Evaluating attack type: {desc}")
                evaluate_attack_file(detector, zip_name)

        elif choice == "7":
            print("Exiting. Bye!")
            break

        else:
            print("Invalid choice, please select 1-7.")


if __name__ == "__main__":
    main()