"""
CAN-bus Normal Behavior Autoencoder + Attack Evaluation (SynCAN)
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================
#  CONFIGURATION
# =========================

# Folder of your dataset
DATA_DIR = Path(__file__).parent

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


def print_alert_message(df_attack: pd.DataFrame, per_id: pd.DataFrame, alert_threshold: float = 0.05):
    """
    Simple textual alert interface:
    - Always prints top suspicious IDs.
    - Prints main suspect (highest anomaly rate).
    - If global anomaly rate > alert_threshold -> prints ALERT.
    """
    # Global anomaly rate over all messages
    global_anomaly_rate = df_attack["is_anomaly"].mean()

    print("\n--- ALERT INTERFACE ---")
    print(f"Global anomaly rate: {global_anomaly_rate * 100:.2f}%")

    # Always show top suspicious IDs by anomaly rate
    suspicious = per_id.sort_values("anomaly_rate_%", ascending=False).head(3)

    print("\nTop suspicious IDs:")
    for _, row in suspicious.iterrows():
        print(
            f"  - ID {row[ID_COL]}:"
            f" anomaly_rate={row['anomaly_rate_%']:.2f}%,"
            f" attack_fraction={row['attack_fraction_%']:.2f}%"
        )

    # Main suspect (highest anomaly rate)
    main_suspect = suspicious.iloc[0]
    print(
        f"\nMain suspect among all IDs: ID {main_suspect[ID_COL]} "
        f"(anomaly likelihood ≈ {main_suspect['anomaly_rate_%']:.2f}%)"
    )


    # Overall status
    if global_anomaly_rate < alert_threshold:
        print("\n>> STATUS: Traffic appears normal. No intrusion detected (below threshold).")
    else:
        print("\n!!! ALERT: Possible intrusion detected on CAN bus !!!")
        print(f">> Global anomaly rate is above threshold ({alert_threshold * 100:.1f}%).")

    print("------------------------\n")

# =========================
#  FEATURE ENGINEERING
# =========================

def encode_id_column(df: pd.DataFrame):
    """
    Map each CAN ID to an integer index: ID_idx.
    """
    unique_ids = df[ID_COL].unique()
    id2idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    df["ID_idx"] = df[ID_COL].map(id2idx)

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
#  AUTOENCODER MODEL
# =========================

def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Build a simple fully-connected autoencoder.
    """
    input_layer = keras.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(32, activation="relu")(input_layer)
    x = layers.Dense(16, activation="relu")(x)
    bottleneck = layers.Dense(8, activation="relu")(x)

    # Decoder
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(32, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")

    return model


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray) -> keras.Model:
    """
    Train the autoencoder on normal data only.
    """
    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)
    model.summary()

    model.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        shuffle=True,
    )

    return model


# =========================
#  ANOMALY THRESHOLD
# =========================

def compute_reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Compute reconstruction error (MSE) for each sample in X.
    """
    X_pred = model.predict(X, batch_size=1024)
    errors = np.mean((X - X_pred) ** 2, axis=1)
    return errors


def choose_threshold(errors: np.ndarray, percentile: float) -> float:
    """
    Choose a threshold based on a percentile of reconstruction errors.
    For example, 99th percentile.
    """
    threshold = np.percentile(errors, percentile)
    return threshold


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

    def transform_dataframe(self, df: pd.DataFrame):
        """
        Transform a new DataFrame with columns:
        Label, ID, Time, Signal1_of_ID..Signal4_of_ID
        into:
        - df_aligned: a sorted dataframe (ID, Time) with all features computed
        - X_scaled: scaled feature matrix ready for the model.
        """
        df = df.copy()

        # Clean NaN in signals
        df[SIGNAL_COLS] = df[SIGNAL_COLS].fillna(0.0)

        # Ensure Time is numeric
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0.0)

        # Map IDs with the same mapping as training.
        # Unknown IDs -> -1
        df["ID_idx"] = df[ID_COL].map(self.id2idx)
        df["ID_idx"] = df["ID_idx"].fillna(-1).astype(int)

        # Compute DeltaTime same as training
        df = df.sort_values(by=[ID_COL, TIME_COL], ascending=True)
        df["DeltaTime"] = df.groupby(ID_COL)[TIME_COL].diff()
        df["DeltaTime"] = df["DeltaTime"].fillna(0.0)

        # Build raw feature matrix
        X_raw = df[self.feature_cols].values

        # Clean NaN / Inf just in case
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e9, neginf=-1e9)

        # Scale
        X_scaled = self.scaler.transform(X_raw)

        # IMPORTANT: return both the aligned df and the scaled features
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
    print("\n=== TRAINING MODEL ON NORMAL DATA ===")

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

    # 7. Train autoencoder on normal data
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

def evaluate_attack_file(detector: CanAnomalyDetector, zip_name: str):
    """
    Load one attack ZIP (e.g. test_suppress.zip / test_flooding.zip),
    run it through the detector, and print stats.
    Also summarize suspicious time windows and most suspicious IDs.
    """
    print("\n" + "=" * 80)
    print(f"Evaluating attack file: {zip_name}")
    print("=" * 80)

    # Raw data from ZIP
    df_attack_raw = load_zip_to_df(zip_name)

    # Aligned dataframe + scaled features
    df_attack, X_attack_scaled = detector.transform_dataframe(df_attack_raw)

    # Compute errors and anomaly mask
    errors = detector.anomaly_score(X_attack_scaled)
    is_anom = errors > detector.threshold

    df_attack["recon_error"] = errors
    df_attack["is_anomaly"] = is_anom

    # True labels from dataset: 0 = normal, 1 = intrusion
    y_true = df_attack[LABEL_COL].astype(int).values
    y_pred = is_anom.astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Detection rate (recall on attacks)
    attack_total = tp + fn
    detection_rate = tp / attack_total if attack_total > 0 else float("nan")

    # False positive rate (on normal rows)
    normal_total = tn + fp
    false_positive_rate = fp / normal_total if normal_total > 0 else float("nan")

    print(f"Total rows: {len(df_attack)}")
    print(f"  Normal rows (Label=0): {normal_total}")
    print(f"  Attack rows (Label=1): {attack_total}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Detection rate on attacks (TPR): {detection_rate * 100:.2f}%")
    print(f"  False positive rate on normals: {false_positive_rate * 100:.2f}%")

    # Per-ID analysis: mean error, anomaly rate, attack fraction
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

    print("\nPer-ID statistics:")
    print(per_id.to_string(index=False))

    # Simple alert interface (עם חשוד עיקרי)
    print_alert_message(df_attack, per_id, alert_threshold=0.05)

    print("\n")


def summarize_attack_windows(df_attack: pd.DataFrame, window_sec: float = 1.0, top_k_ids: int = 1):
    """
    Summarize suspicious time windows:
    - Divide timeline into bins of 'window_sec' seconds.
    - For each bin, compute:
        * fraction of Label=1 (ground truth attack)
        * fraction of is_anomaly (model)
        * per-ID anomaly rate
    - Print, for each suspicious bin, the most suspicious ID(s) with a "probability"
      (= anomaly rate of that ID in that window).
    """
    df = df_attack.copy()

    # Time in seconds (SynCAN Time is in ms)
    df["time_sec"] = df[TIME_COL] / 1000.0

    # Define time bins
    df["time_bin"] = (df["time_sec"] / window_sec).astype(int)

    results = []

    grouped = df.groupby("time_bin")

    for bin_idx, g in grouped:
        if g.empty:
            continue

        # Fractions in this time window
        label_attack_frac = (g[LABEL_COL] == 1).mean()
        anomaly_frac = g["is_anomaly"].mean()

        # נרצה לדווח רק על חלונות שיש בהם התקפה אמיתית או הרבה אנומליות
        if label_attack_frac < 0.1 and anomaly_frac < 0.1:
            # פחות מ-10% מהחלון תקיפה ופחות מ-10% אנומליות → פחות מעניין
            continue

        start_t = g["time_sec"].min()
        end_t = g["time_sec"].max()

        # Per-ID suspicion in this window
        per_id_bin = (
            g.groupby(ID_COL)
            .agg(
                mean_error=("recon_error", "mean"),
                anomaly_rate=("is_anomaly", "mean"),
                msg_count=(ID_COL, "count"),
            )
            .reset_index()
        )

        # "סבירות" שה-ID נגוע = anomaly_rate שלו בחלון
        per_id_bin["suspicion_score"] = per_id_bin["anomaly_rate"]

        # Top-K suspicious IDs in this time bin
        per_id_bin = per_id_bin.sort_values("suspicion_score", ascending=False)
        top = per_id_bin.head(top_k_ids)

        for _, row in top.iterrows():
            results.append(
                {
                    "bin_start": start_t,
                    "bin_end": end_t,
                    "id": row[ID_COL],
                    "suspicion": row["suspicion_score"],
                    "bin_anomaly_frac": anomaly_frac,
                    "bin_label_attack_frac": label_attack_frac,
                }
            )

    if not results:
        print("\nNo suspicious time windows found (with current thresholds).")
        return

    print(f"\nSuspicious time windows (bins of {window_sec:.1f} sec):")
    for r in results:
        print(
            f"  t ≈ [{r['bin_start']:.1f}s – {r['bin_end']:.1f}s], "
            f"ID={r['id']}, "
            f"suspicion≈{r['suspicion'] * 100:.1f}%, "
            f"window_anomaly≈{r['bin_anomaly_frac'] * 100:.1f}%, "
            f"window_label_attack≈{r['bin_label_attack_frac'] * 100:.1f}%"
        )


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
