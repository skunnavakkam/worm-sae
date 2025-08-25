# eval.py
import ast, glob, os, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models import cElegansFwdSAE, cElegansBwdSAE

# -------------------- helpers --------------------
TIME_NAME_PATTERNS = [
    re.compile(r"^(\d+(?:\.\d+)?)s$"),  # "0s", "12.5s"
    re.compile(r"^[tT]?(\d+)$"),  # "0", "123", "t45"
    re.compile(r"^frame_(\d+)$"),  # "frame_183"
]


def parse_time_from_name(colname: str):
    """
    Return a float time index if the column name looks like a time column,
    else None. Handles merge suffixes like '123s_2' by stripping '_...'.
    """
    base = colname.split("_")[0]
    for pat in TIME_NAME_PATTERNS:
        m = pat.match(base)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def infer_time_columns(
    df: pd.DataFrame,
    neuron_col="neuron",
    stimulus_col="stimulus",
    min_numeric_ratio: float = 0.95,
):
    """
    Pick time-series columns using BOTH name pattern + value check.
    Excludes obvious non-time columns (like 'stimulus').
    Returns (ordered_time_cols, dropped_cols_debug)
    """
    # consider only columns to the right of 'neuron' to match your pipeline
    neuron_idx = df.columns.get_loc(neuron_col)
    candidate_cols = [c for c in df.columns[neuron_idx + 1 :] if c != stimulus_col]

    # 1) name-based candidates
    name_candidates = []
    name_times = {}
    for c in candidate_cols:
        t = parse_time_from_name(c)
        if t is not None:
            name_candidates.append(c)
            name_times[c] = t

    # 2) numeric value check (≥95% numeric after coercion)
    good_cols = []
    dropped = []
    for c in name_candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        ratio = s.notna().mean()
        if ratio >= min_numeric_ratio:
            good_cols.append(c)
        else:
            dropped.append((c, f"only {ratio:.2%} numeric"))

    # Fallback: if we somehow found nothing by name, allow numeric-typed columns
    if not good_cols:
        for c in candidate_cols:
            if c == stimulus_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            ratio = s.notna().mean()
            if ratio >= min_numeric_ratio:
                good_cols.append(c)
            else:
                dropped.append((c, f"fallback reject, {ratio:.2%} numeric"))

    # sort by inferred time if available; otherwise keep order
    good_cols_sorted = sorted(good_cols, key=lambda c: (name_times.get(c, np.inf),))
    return good_cols_sorted, dropped


def recover_training_neuron_order(csv_files, neuron_col="neuron"):
    """
    Reproduce your merge-on-neuron (but only keep the neuron column),
    so neuron order matches the training pipeline’s outer merge behavior.
    """
    if not csv_files:
        raise FileNotFoundError("No CSVs matched data/*.csv")
    # start with first file's neuron column
    df0 = pd.read_csv(csv_files[0])
    idx0 = df0.columns.get_loc(neuron_col)
    merged = df0.iloc[:, idx0 : idx0 + 1]  # only 'neuron'

    for p in csv_files[1:]:
        dfi = pd.read_csv(p)
        idxi = dfi.columns.get_loc(neuron_col)
        dfi = dfi.iloc[:, idxi : idxi + 1]
        merged = pd.merge(merged, dfi, on=neuron_col, how="outer")

    neuron_order = merged[neuron_col].astype(str).tolist()
    return neuron_order


def build_stimulus_labels(df: pd.DataFrame, T: int, stimulus_col="stimulus"):
    """
    Parse your schedule string -> per-timestep labels of length T.
    """
    stim_str = df[stimulus_col].dropna().iloc[0]
    schedule = [(int(s), lab) for s, lab in ast.literal_eval(stim_str)]
    labels = np.array(["(unset)"] * T, dtype=object)
    for i, (start, lab) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else T
        start = max(0, min(T, start))
        end = max(0, min(T, end))
        labels[start:end] = lab
    return labels


def load_single_csv_aligned(
    csv_path: str,
    training_neuron_order: list[str],
    neuron_col="neuron",
    stimulus_col="stimulus",
    impute_missing: float = 0.0,
):
    """
    Returns:
      X: [T, N] float32 torch tensor aligned to training neuron order (missing neurons imputed),
      labels: length-T np.ndarray of stimulus strings,
      missing, extra: lists for reporting,
      debug_drop: list of (col, reason) that were rejected as time columns
    """
    df = pd.read_csv(csv_path)

    # infer valid time columns
    time_cols, debug_drop = infer_time_columns(
        df, neuron_col=neuron_col, stimulus_col=stimulus_col
    )
    if not time_cols:
        raise RuntimeError(f"No usable time-series columns found in {csv_path}")
    T = len(time_cols)

    # neuron coverage
    have = set(df[neuron_col].astype(str))
    train = [str(n) for n in training_neuron_order]
    missing = [n for n in train if n not in have]
    extra = [n for n in have if n not in train]

    # align rows to training order; coerce numeric & impute
    times_df = df[[neuron_col] + time_cols].copy()
    for c in time_cols:
        times_df[c] = pd.to_numeric(times_df[c], errors="coerce")

    aligned = times_df.set_index(neuron_col).reindex(train)
    # impute missing neurons + any stray NaNs with constant (default 0.0)
    aligned = aligned.fillna(impute_missing)

    X = torch.tensor(
        aligned.to_numpy(dtype=np.float32).T, dtype=torch.float32
    )  # [T, N]

    # labels from schedule
    labels = build_stimulus_labels(df, T, stimulus_col=stimulus_col)
    return X, labels, missing, extra, debug_drop


@torch.no_grad()
def hidden_acts(encode: nn.Module, X: torch.Tensor, device="cpu"):
    return torch.relu(encode(X.to(device))).cpu()


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
                f"  #{r:02d} unit {j:3d}  d={d[j]:+.3f}  on={meanA[j]:.3f} off={meanB[j]:.3f}"
            )


def load_sae(encode_path: str, decode_path: str, device="cpu"):
    enc = cElegansFwdSAE().to(device)
    dec = cElegansBwdSAE().to(device)
    enc.load_state_dict(torch.load(encode_path, map_location=device))
    dec.load_state_dict(torch.load(decode_path, map_location=device))
    enc.eval()
    dec.eval()
    return enc, dec


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="path to a specific CSV; defaults to first match in data/*.csv",
    )
    ap.add_argument("--encode", type=str, default="models/encode.pt")
    ap.add_argument("--decode", type=str, default="models/decode.pt")
    ap.add_argument(
        "--impute",
        type=float,
        default=0.0,
        help="value to impute for missing neurons/NaNs (default 0.0)",
    )
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSVs found in data/*.csv")

    # recover training neuron order (merge on neuron only)
    training_neuron_order = recover_training_neuron_order(
        csv_files, neuron_col="neuron"
    )

    # pick dataset
    csv_path = args.csv if args.csv is not None else csv_files[0]
    print("=" * 60)
    print(f"Analyzing {csv_path}")

    # load + align + labels
    X, labels, missing, extra, dropped = load_single_csv_aligned(
        csv_path,
        training_neuron_order,
        neuron_col="neuron",
        stimulus_col="stimulus",
        impute_missing=args.impute,
    )
    print(f"Shape [T,N]: {tuple(X.shape)}")
    print(
        f"Missing neurons: {len(missing)}{' -> ' + ', '.join(missing[:10]) + (' ...' if len(missing) > 10 else '') if missing else ''}"
    )
    print(
        f"Extra neurons:   {len(extra)}{' -> ' + ', '.join(extra[:10]) + (' ...' if len(extra) > 10 else '') if extra else ''}"
    )
    if dropped:
        print("Dropped non-time columns (name/value check):")
        for c, why in dropped[:10]:
            print(f"  - {c}: {why}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")

    # load SAE, run hidden, summarize
    encode, decode = load_sae(args.encode, args.decode, device=args.device)
    H = hidden_acts(encode, X, device=args.device)
    print(f"Hidden activations [T,D]: {tuple(H.shape)}")
    summarize_selectivity(H, labels, top_k=8)


if __name__ == "__main__":
    main()
