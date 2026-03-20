#!/usr/bin/env python3
"""
Extract WFBD (WFDB) waveform data to HDF5.

Reads a user-specified WFBD dataset from raw_data (.dat, .hea, .json),
extracts a 20-second window centered on ChamEvents_in_s RV (10 s before,
10 s after, clamped to the recording) for ECG_lead_I, ECG_lead_II,
ECG_lead_V1, and RHC_pressure, and writes time series plus Fick COL/min
and TDCOL/min to hdf5_files/{dataset_name}.h5.

Usage:
    python scripts/extract_wfbd_to_hdf5.py TRM127-RHC1

Requires: wfdb, h5py (see requirements.txt).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import wfdb

# Requested signal names (same names used as HDF5 dataset names)
SIGNAL_NAMES = ["ECG_lead_I", "ECG_lead_II", "ECG_lead_V1", "RHC_pressure"]

# JSON keys for scalar metadata (exact keys from maclabMeas, including trailing spaces)
FICK_KEY = "Fick COL/min "
TDCOL_KEY = "TDCOL/min    "

# Symmetric window around RV (seconds before / after)
WINDOW_BEFORE_SEC = 10
WINDOW_AFTER_SEC = 10


def get_project_root() -> Path:
    """Project root is parent of the scripts directory."""
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract WFBD time series (RV window) and metadata to HDF5."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="WFBD dataset name (e.g. TRM127-RHC1); files raw_data/{name}.hea, .dat, .json",
    )
    args = parser.parse_args()
    name = args.dataset_name.strip()
    if not name:
        print("Error: dataset_name must be non-empty.", file=sys.stderr)
        return 1

    root = get_project_root()
    raw_dir = root / "raw_data"
    hea_path = raw_dir / f"{name}.hea"
    dat_path = raw_dir / f"{name}.dat"
    json_path = raw_dir / f"{name}.json"
    out_dir = root / "hdf5_files"
    out_path = out_dir / f"{name}.h5"

    if not hea_path.exists():
        print(f"Error: header file not found: {hea_path}", file=sys.stderr)
        return 1
    if not json_path.exists():
        print(f"Error: metadata file not found: {json_path}", file=sys.stderr)
        return 1

    # Load JSON metadata
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: failed to read JSON {json_path}: {e}", file=sys.stderr)
        return 1

    cham = metadata.get("ChamEvents_in_s")
    if not cham or "RV" not in cham:
        print(
            "Error: ChamEvents_in_s.RV not found in JSON; cannot determine window start.",
            file=sys.stderr,
        )
        return 1
    rv_sec = float(cham["RV"])

    # Read WFDB record (requires .dat + .hea)
    record_path = str(raw_dir / name)
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        if not dat_path.exists():
            print(
                f"Error: record is incomplete (missing .dat file: {dat_path}). {e}",
                file=sys.stderr,
            )
        else:
            print(f"Error: failed to read WFDB record: {e}", file=sys.stderr)
        return 1

    fs = record.fs
    n_samples = record.p_signal.shape[0]
    sig_names = record.sig_name

    # Window: [RV - 10s, RV + 10s], clamped to [0, n_samples)
    start_samp = max(0, int((rv_sec - WINDOW_BEFORE_SEC) * fs))
    end_samp = min(int((rv_sec + WINDOW_AFTER_SEC) * fs), n_samples)
    if start_samp >= n_samples:
        print(
            f"Error: RV time {rv_sec}s (window start sample {start_samp}) is beyond recording length ({n_samples} samples).",
            file=sys.stderr,
        )
        return 1
    if start_samp >= end_samp:
        print(
            f"Error: empty extraction window (start sample {start_samp}, end sample {end_samp}).",
            file=sys.stderr,
        )
        return 1

    # Extract requested signals (skip missing, warn)
    snippets = {}
    for sig in SIGNAL_NAMES:
        if sig not in sig_names:
            print(f"Warning: signal '{sig}' not in record; skipping.", file=sys.stderr)
            continue
        idx = sig_names.index(sig)
        snippets[sig] = record.p_signal[start_samp:end_samp, idx].flatten()

    if not snippets:
        print("Error: none of the requested signals were found in the record.", file=sys.stderr)
        return 1

    # Scalars from maclabMeas (exact keys with trailing spaces)
    maclab = metadata.get("maclabMeas") or {}
    fick_val = maclab.get(FICK_KEY)
    tdcol_val = maclab.get(TDCOL_KEY)
    if fick_val is None:
        print(f"Warning: '{FICK_KEY.strip()}' not in maclabMeas; omitting from HDF5.", file=sys.stderr)
    if tdcol_val is None:
        print(f"Warning: '{TDCOL_KEY.strip()}' not in maclabMeas; omitting from HDF5.", file=sys.stderr)

    # Write HDF5
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(out_path, "w") as f:
            for sig, data in snippets.items():
                f.create_dataset(sig, data=data, dtype=data.dtype)
            # Attributes for reproducibility
            f.attrs["fs_Hz"] = float(fs)
            f.attrs["rv_timestamp_sec"] = float(rv_sec)
            f.attrs["window_start_sec"] = start_samp / fs
            f.attrs["window_end_sec"] = (end_samp - 1) / fs if end_samp > start_samp else start_samp / fs
            if fick_val is not None:
                f.attrs["Fick_COL_per_min"] = float(fick_val)
            if tdcol_val is not None:
                f.attrs["TDCOL_per_min"] = float(tdcol_val)
    except OSError as e:
        print(f"Error: failed to write HDF5 {out_path}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path} with datasets: {list(snippets.keys())}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
