# analyze_sae_stimuli.py
import ast, glob, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==== YOUR MODELS (placeholders; keep your classes as-is) ====
class cElegansFwdSAE(nn.Module):
    def __init__(self):
        super().__init__()
        # define same as training
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)


class cElegansBwdSAE(nn.Module):
    def __init__(self):
        super().__init__()
        # define same as training
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 0) Reuse your training merge logic to recover neuron order
#    (faithfully mirrors your load_data() steps)
# -------------------------------------------------------------
def get_training_neuron_order_and_timecols(
    data_glob="data/*.csv", neuron_col="neuron"
) -> tuple[list[str], list[str]]:
    csv_files = glob.glob(data_glob)
    if not csv_files:
        raise FileNotFoundError(f"No CSVs matched {data_glob}")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        neuron_idx = df.columns.get_loc(neuron_col)
        df = df.iloc[:, neuron_idx:]  # includes 'neuron' + times
        dfs.append(df)

    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        merged_df = pd.merge(
            merged_df, df, on=neuron_col, how="outer", suffixes=("", f"_{i}")
        )

    # same column keep rule as your training
    cols_to_keep = [
        col
        for col in merged_df.columns
        if col == neuron_col or col.split("_")[0].endswith("s")
    ]
    merged_df = merged_df[cols_to_keep]

    # neuron order & time columns exactly as used in training
    neuron_order = merged_df[neuron_col].astype(str).tolist()
    time_cols = [c for c in merged_df.columns if c != neuron_col]
    return neuron_order, time_cols


# -------------------------------------------------------------
# 1) Load ONE dataset, align to training neuron order, build labels
# -------------------------------------------------------------
def load_single_csv_aligned(
    csv_path: str,
    training_neuron_order: list[str],
    neuron_col="neuron",
    stimulus_col="stimulus",
) -> tuple[torch.Tensor, np.ndarray, list[str], list[str]]:
    df = pd.read_csv(csv_path)

    # stimulus schedule string -> list[[start_idx, label], ...]
    if stimulus_col not in df.columns:
        raise ValueError(f"'{stimulus_col}' not found in {csv_path}")
    stim_str = df[stimulus_col].dropna().iloc[0]
    schedule = ast.literal_eval(stim_str)
    schedule = [(int(s), lab) for s, lab in schedule]

    # slice like training (from neuron onward)
    neuron_idx = df.columns.get_loc(neuron_col)
    times_df = df.iloc[:, neuron_idx:]
    assert times_df.columns[0] == neuron_col
    time_cols = list(times_df.columns[1:])
    T = len(time_cols)

    # align rows to training order
    train_neurons = [str(n) for n in training_neuron_order]
    have_neurons = set(times_df[neuron_col].astype(str).tolist())
    missing = [n for n in train_neurons if n not in have_neurons]
    extra = [n for n in have_neurons if n not in train_neurons]

    aligned = times_df.set_index(neuron_col).reindex(
        train_neurons
    )  # rows aligned to training order
    X_np = aligned[time_cols].to_numpy(dtype=np.float32)  # [N, T]
    X = torch.tensor(X_np.T, dtype=torch.float32)  # [T, N]

    # per-timestep labels
    labels = np.array(["(unset)"] * T, dtype=object)
    for i, (start, lab) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else T
        start = max(0, min(T, start))
        end = max(0, min(T, end))
        labels[start:end] = lab

    return X, labels, missing, extra


# -------------------------------------------------------------
# 2) NaN mask (apply to X and labels consistently)
# -------------------------------------------------------------
def mask_nans_keep_labels(
    X: torch.Tensor, labels: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    mask = ~torch.isnan(X).any(dim=1)
    return X[mask], labels[mask.numpy()]


# -------------------------------------------------------------
# 3) Load saved SAE + get hidden activations
# -------------------------------------------------------------
def load_sae(encode_path: str, decode_path: str, device="cpu"):
    encode = cElegansFwdSAE().to(device)
    decode = cElegansBwdSAE().to(device)
    encode.load_state_dict(torch.load(encode_path, map_location=device))
    decode.load_state_dict(torch.load(decode_path, map_location=device))
    encode.eval()
    decode.eval()
    return encode, decode


@torch.no_grad()
def hidden_activations(
    encode: nn.Module, X: torch.Tensor, device="cpu"
) -> torch.Tensor:
    H = encode(X.to(device))
    H = torch.relu(H)  # match training
    return H.cpu()


# -------------------------------------------------------------
# 4) Simple interpretation: per-stimulus selectivity via Cohen's d
# -------------------------------------------------------------
def summarize_selectivity(
    H: torch.Tensor, labels: np.ndarray, top_k=8, min_on=5, min_off=20
):
    T, D = H.shape
    Hn = H.numpy()
    uniq = [u for u in np.unique(labels) if u != "(unset)"]
    for lab in uniq:
        idx = labels == lab
        if idx.sum() < min_on or (~idx).sum() < min_off:
            continue
        A, B = Hn[idx], Hn[~idx]
        meanA, meanB = A.mean(0), B.mean(0)
        stdA, stdB = A.std(0) + 1e-6, B.std(0) + 1e-6
        d = (meanA - meanB) / np.sqrt(0.5 * (stdA**2 + stdB**2))
        order = np.argsort(-d)[:top_k]
        print(f"\nStimulus: {lab}  (n_on={idx.sum()}, n_off={(~idx).sum()})")
        for r, j in enumerate(order, 1):
            print(
                f"  #{r:02d} unit {j:>3d}  d={d[j]:+.3f}  mean_on={meanA[j]:.3f}  mean_off={meanB[j]:.3f}"
            )


# -------------------------------------------------------------
# 5) (Optional) enforce harder sparsity if you retrain/evaluate
# -------------------------------------------------------------
def k_winners(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= x.shape[1]:
        return x
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x).scatter(1, idx, 1.0)
    return x * mask


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # A) Recover training neuron order by reusing your merge logic
    training_neuron_order, _ = get_training_neuron_order_and_timecols(
        "data/*.csv", neuron_col="neuron"
    )
    # If you’d like, cache it:
    # json.dump(training_neuron_order, open("neuron_order.json","w"))

    # B) Pick one dataset CSV to analyze (edit this path)
    csv_path = "data/your_dataset.csv"

    # C) Align, make labels, and mask NaNs
    X, labels, missing, extra = load_single_csv_aligned(
        csv_path, training_neuron_order, neuron_col="neuron", stimulus_col="stimulus"
    )
    print(f"Single CSV raw shape [T,N]: {tuple(X.shape)}")
    if missing:
        print(
            f"Missing neurons ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if extra:
        print(
            f"Extra neurons not in training ({len(extra)}): {extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    X, labels = mask_nans_keep_labels(X, labels)
    print(f"After NaN mask [T,N]: {tuple(X.shape)}")

    # D) Load trained SAE weights
    encode_path = os.path.join("models", "encode.pt")  # <- set to your filenames
    decode_path = os.path.join("models", "decode.pt")
    device = "cpu"
    encode, decode = load_sae(encode_path, decode_path, device=device)

    # E) Hidden activations and stimulus selectivity summary
    H = hidden_activations(encode, X, device=device)  # [T, D]
    print(f"Hidden activations shape [T,D]: {tuple(H.shape)}")
    summarize_selectivity(H, labels, top_k=8)
