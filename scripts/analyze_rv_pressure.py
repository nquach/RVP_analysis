#!/usr/bin/env python3
"""
Analyze RV pressure time series from HDF5: ECG-gated cardiac cycles and PV-loop derived measures.

Reads user-specified HDF5 (ECG Lead II + RHC_pressure), detects R-waves (TEE_optical_flow style),
segments RV pressure into R-R intervals, and computes per-cycle Ees, Ea, RVEF, Eed, RVMPI, beta, etc.
Writes csv_files/{basename}.csv and png_files/{basename}_{cycle}.png.

Usage:
    python scripts/analyze_rv_pressure.py path/to/file.h5 [--co-method TDCO|Fick_CO] [--output-dir DIR]
        [--deriv-smooth none|savgol|kalman] [--savgol-window N] [--savgol-polyorder P]
        [--kalman-level-noise X] [--kalman-trend-noise X] [--kalman-observation-noise X]
        [-v|--verbose]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import peakutils
from scipy.optimize import brentq, curve_fit
from scipy.signal import savgol_filter
from tsmoothie.smoother import KalmanSmoother, SpectralSmoother

# HDF5 dataset/attr names (must match extract_wfbd_to_hdf5.py)
ECG_LEAD_II = "ECG_lead_II"
RHC_PRESSURE = "RHC_pressure"
FS_ATTR = "fs_Hz"
TDCOL_ATTR = "TDCOL_per_min"
FICK_ATTR = "Fick_COL_per_min"

# ECG preprocessing (TEE_optical_flow defaults)
ECG_SMOOTH_FRACTION = 0.2
ECG_PAD_LEN = 20

# Valid RR interval (seconds) to skip artifact cycles
RR_MIN_SEC = 0.4
RR_MAX_SEC = 1.5

# Number of prominent peaks in RVP''²
N_PEAKS_RVP2 = 4

# RVP' / RVP'' smoothing defaults (post-gradient, before peak picking)
DEFAULT_SAVGOL_WINDOW = 11
DEFAULT_SAVGOL_POLYORDER = 3
DEFAULT_KALMAN_LEVEL_NOISE = 0.1
DEFAULT_KALMAN_TREND_NOISE = 0.1
DEFAULT_KALMAN_OBSERVATION_NOISE = 1.0


@dataclass(frozen=True)
class DerivativeSmoothConfig:
    """Smoothing applied to RVP' and RVP'' after np.gradient, before RVP''² and peaks."""

    method: str = "none"  # none | savgol | kalman
    savgol_window: int = DEFAULT_SAVGOL_WINDOW
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER
    kalman_level_noise: float = DEFAULT_KALMAN_LEVEL_NOISE
    kalman_trend_noise: float = DEFAULT_KALMAN_TREND_NOISE
    kalman_observation_noise: float = DEFAULT_KALMAN_OBSERVATION_NOISE


def get_project_root() -> Path:
    """Project root is parent of the scripts directory."""
    return Path(__file__).resolve().parent.parent


def detect_r_waves(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Preprocess ECG and detect R-wave sample indices (TEE_optical_flow style).
    Uses neurokit2 ecg_clean (vg), SpectralSmoother, then ecg_peaks (khamis2016).
    """
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method="vg")
    smoother = SpectralSmoother(smooth_fraction=ECG_SMOOTH_FRACTION, pad_len=ECG_PAD_LEN)
    smoother.smooth(cleaned)
    filtered_ecg = np.squeeze(smoother.smooth_data[0])
    _, rpeaks = nk.ecg_peaks(
        filtered_ecg, sampling_rate=fs, method="khamis2016", correct_artifacts=True, show=False
    )
    return np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)


def _derivatives(rvp: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second time derivatives of RVP (RVP', RVP'')."""
    rvp1 = np.gradient(rvp, t)
    rvp2 = np.gradient(rvp1, t)
    return rvp1, rvp2


def _effective_savgol_window(n: int, polyorder: int, requested: int) -> Optional[int]:
    """Largest odd window <= min(requested, n) with window_length > polyorder, or None."""
    if n <= 0 or polyorder < 0:
        return None
    max_w = min(requested, n)
    if max_w % 2 == 0:
        max_w -= 1
    while max_w >= 3:
        if max_w % 2 == 1 and max_w > polyorder:
            return max_w
        max_w -= 1
    return None


def _kalman_smooth_1d(x: np.ndarray, cfg: DerivativeSmoothConfig) -> np.ndarray:
    """Kalman smooth a 1D series; on failure return input as float array."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        print(
            "Warning: segment too short for Kalman derivative smoothing; using raw derivatives.",
            file=sys.stderr,
        )
        return x
    try:
        ks = KalmanSmoother(
            component="level_trend",
            component_noise={
                "level": cfg.kalman_level_noise,
                "trend": cfg.kalman_trend_noise,
            },
            observation_noise=cfg.kalman_observation_noise,
        )
        ks.smooth(x)
        out = np.asarray(ks.smooth_data[0], dtype=float)
        out = np.squeeze(out)
        if out.shape != x.shape:
            out = out.reshape(x.shape)
        return out
    except Exception as e:
        print(
            f"Warning: Kalman derivative smoothing failed ({e}); using raw derivatives.",
            file=sys.stderr,
        )
        return x


def _smooth_derivatives(
    rvp1: np.ndarray,
    rvp2: np.ndarray,
    _fs: float,
    cfg: DerivativeSmoothConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth RVP' and RVP'' in place of identity when method is none."""
    m = (cfg.method or "none").lower()
    if m == "none":
        return rvp1, rvp2
    if m == "savgol":
        n = int(rvp1.shape[0])
        w_req = cfg.savgol_window
        if w_req % 2 == 0:
            w_req += 1
        w = _effective_savgol_window(n, cfg.savgol_polyorder, w_req)
        if w is None:
            print(
                "Warning: segment too short for Savitzky-Golay derivative smoothing; using raw derivatives.",
                file=sys.stderr,
            )
            return rvp1, rvp2
        p = cfg.savgol_polyorder
        r1 = savgol_filter(np.asarray(rvp1, dtype=float), w, p, mode="interp")
        r2 = savgol_filter(np.asarray(rvp2, dtype=float), w, p, mode="interp")
        return r1, r2
    if m == "kalman":
        return _kalman_smooth_1d(rvp1, cfg), _kalman_smooth_1d(rvp2, cfg)
    raise ValueError(f"Unknown derivative smooth method: {cfg.method!r}")


def _find_four_peaks(
    rvp2_sq: np.ndarray, min_dist: int = 5, thres: float = 0.1
) -> Optional[np.ndarray]:
    """Find four prominent peaks in RVP''²; return indices in order or None."""
    peak_idx = peakutils.peak.indexes(
        rvp2_sq, thres=thres, min_dist=min_dist
    )
    if len(peak_idx) < N_PEAKS_RVP2:
        return None
    # Take the four largest peaks by value, then sort by index to get order
    values = rvp2_sq[peak_idx]
    order = np.argsort(values)[::-1][:N_PEAKS_RVP2]
    four = np.sort(peak_idx[order])
    return four


def _sine_model(t: np.ndarray, A: float, omega: float, phi: float, C: float) -> np.ndarray:
    return A * np.sin(omega * t + phi) + C


def _fit_pmax_ivct_ivrt(
    t: np.ndarray,
    rvp: np.ndarray,
    peak1_idx: int,
    idx_max_dp: int,
    idx_min_dp: int,
    peak4_idx: int,
) -> tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit sine to IVCT and IVRT portions; return Pmax and optional fitted t, P for plotting.
    IVCT: from 1st peak RVP''² to max RVP'. IVRT: from min RVP' to 4th peak RVP''².
    """
    t_ivct = t[peak1_idx : idx_max_dp + 1]
    p_ivct = rvp[peak1_idx : idx_max_dp + 1]
    t_ivrt = t[idx_min_dp : peak4_idx + 1]
    p_ivrt = rvp[idx_min_dp : peak4_idx + 1]
    if t_ivct.size < 3 or t_ivrt.size < 3:
        return np.nan, None, None
    t_cat = np.concatenate([t_ivct, t_ivrt])
    p_cat = np.concatenate([p_ivct, p_ivrt])
    try:
        # Initial guess: A from range, omega ~ 2*pi/T_cycle, phi=0, C=mean
        p_mean = np.mean(p_cat)
        p_amp = (np.max(p_cat) - np.min(p_cat)) / 2
        T_approx = np.ptp(t_cat) if np.ptp(t_cat) > 0.01 else 0.5
        omega_guess = 2 * np.pi / T_approx
        popt, _ = curve_fit(
            _sine_model,
            t_cat,
            p_cat,
            p0=[p_amp, omega_guess, 0.0, p_mean],
            bounds=(
                [0, 0.1, -np.pi, -50],
                [200, 100, np.pi, 200],
            ),
            maxfev=5000,
        )
        A, omega, phi, C = popt
        pmax = C + np.abs(A)
        t_fine = np.linspace(t.min(), t.max(), 200)
        p_fit = _sine_model(t_fine, A, omega, phi, C)
        return float(pmax), t_fine, p_fit
    except Exception:
        return np.nan, None, None


def _solve_beta(esv: float, edv: float, edp_plus_bdp_plus_1: float) -> Optional[float]:
    """Solve (EDP+BDP+1)*(exp(beta*ESV)-1) = exp(beta*EDV)-1 for beta."""
    if esv <= 0 or edv <= esv or edp_plus_bdp_plus_1 <= 0:
        return None

    def eq(b: float) -> float:
        return (edp_plus_bdp_plus_1) * (np.exp(b * esv) - 1) - (np.exp(b * edv) - 1)

    try:
        # beta positive, typically small
        beta = brentq(eq, 1e-6, 10.0)
        return float(beta)
    except Exception:
        return None


def analyze_one_cycle(
    rvp: np.ndarray,
    t: np.ndarray,
    rr_sec: float,
    fs: float,
    co_l_per_min: float,
    deriv_smooth: Optional[DerivativeSmoothConfig] = None,
) -> tuple[dict[str, Any], Optional[tuple]]:
    """
    Compute all metrics for one cardiac cycle. Returns (row_dict, plot_data).
    plot_data is (t, rvp, rvp1, rvp2_sq, peak_indices, pmax, t_fit, p_fit) for plotting.
    """
    cfg = deriv_smooth if deriv_smooth is not None else DerivativeSmoothConfig()
    rvp1, rvp2 = _derivatives(rvp, t)
    rvp1, rvp2 = _smooth_derivatives(rvp1, rvp2, fs, cfg)
    rvp2_sq = rvp2 ** 2
    peaks = _find_four_peaks(rvp2_sq)
    if peaks is None or len(peaks) < N_PEAKS_RVP2:
        return (
            {"cycle_ok": False, "failed_peak_detection": True},
            None,
        )

    peak1_idx, peak2_idx, peak3_idx, peak4_idx = int(peaks[0]), int(peaks[1]), int(peaks[2]), int(peaks[3])
    idx_max_dp = int(np.argmax(rvp1))
    idx_min_dp = int(np.argmin(rvp1))

    # Landmarks (times in seconds)
    t_peak1 = t[peak1_idx]
    t_peak4 = t[peak4_idx]
    t_max_dp = t[idx_max_dp]
    t_min_dp = t[idx_min_dp]

    ivct = t_max_dp - t_peak1
    et = t_min_dp - t_max_dp
    ivrt = t_peak4 - t_min_dp

    esp = float(rvp[peak3_idx])
    bdp = float(rvp[peak4_idx])
    edp = float(rvp[-1])

    hr = 60.0 / rr_sec
    sv = (co_l_per_min * 1000.0) / hr if hr > 0 else np.nan

    pmax, t_fit, p_fit = _fit_pmax_ivct_ivrt(
        t, rvp, peak1_idx, idx_max_dp, idx_min_dp, peak4_idx
    )

    if np.isnan(pmax) or pmax <= esp:
        esv = edv = ees = ea = rvef = np.nan
    else:
        esv = (esp * sv) / (pmax - esp)
        edv = (pmax * sv) / (pmax - esp)
        ees = (pmax - esp) / sv
        ea = esp / sv
        rvef = 100.0 * (1.0 - esp / pmax)

    if et and not np.isnan(et) and et > 0:
        rvmpi = (ivct + ivrt) / et
    else:
        rvmpi = np.nan

    # Diastolic stiffness
    beta_val = np.nan
    eed_val = np.nan
    if not (np.isnan(esv) or np.isnan(edv) or np.isnan(edp) or np.isnan(bdp)):
        edp_bdp_1 = edp + bdp + 1.0
        beta_val = _solve_beta(esv, edv, edp_bdp_1)
        if beta_val is not None:
            alpha = 1.0 / (np.exp(beta_val * esv) - 1.0)
            eed_val = alpha * beta_val * np.exp(beta_val * edv)

    row = {
        "cycle_ok": True,
        "failed_peak_detection": False,
        "RR_interval_sec": rr_sec,
        "HR": hr,
        "IVCT": ivct,
        "ET": et,
        "IVRT": ivrt,
        "ESP": esp,
        "BDP": bdp,
        "EDP": edp,
        "Pmax": pmax,
        "SV": sv,
        "ESV": esv,
        "EDV": edv,
        "Ees": ees,
        "Ea": ea,
        "RVEF_pct": rvef,
        "RVMPI": rvmpi,
        "beta": beta_val,
        "Eed": eed_val,
    }
    plot_data = (
        t,
        rvp,
        rvp1,
        rvp2_sq,
        peaks,
        pmax,
        t_fit,
        p_fit,
        idx_max_dp,
        idx_min_dp,
    )
    return row, plot_data


def segment_pressure_by_rr_with_fs(
    pressure: np.ndarray, r_peaks: np.ndarray, fs: float
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Segment RV pressure into cycles; time in seconds, RR in seconds."""
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(len(r_peaks) - 1):
        start = int(r_peaks[i])
        end = int(r_peaks[i + 1])
        seg = pressure[start:end]
        if seg.size < 10:
            continue
        rr_sec = (end - start) / fs
        if rr_sec < RR_MIN_SEC or rr_sec > RR_MAX_SEC:
            continue
        t = np.arange(seg.size, dtype=float) / fs
        segments.append((seg, t, rr_sec))
    return segments


def save_diagnostic_plot(
    plot_data: tuple,
    save_path: Path,
    cycle_num: int,
) -> None:
    """Save 3-panel diagnostic plot: RVP (+ sine), RVP', RVP''²; shared x-axis."""
    t, rvp, rvp1, rvp2_sq, peaks, pmax, t_fit, p_fit, idx_max_dp, idx_min_dp = plot_data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")
    ax1.plot(t, rvp, "b-", label="RV pressure")
    if t_fit is not None and p_fit is not None:
        ax1.plot(t_fit, p_fit, "r--", alpha=0.8, label="Sine fit (IVCT+IVRT)")
    if not np.isnan(pmax):
        ax1.axhline(pmax, color="gray", linestyle=":", alpha=0.8)
        ax1.text(t[-1] * 0.95, pmax, f"Pmax={pmax:.1f}", va="bottom", ha="right", fontsize=8)
    ax1.set_ylabel("Pressure (mmHg)")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_title(f"Cycle {cycle_num}: RV pressure")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, rvp1, "g-")
    ax2.set_ylabel("RVP' (mmHg/s)")
    ax2.set_title("First derivative")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, rvp2_sq, "m-")
    ax3.scatter(t[peaks], rvp2_sq[peaks], c="red", s=30, zorder=5, label="Peaks")
    ax3.set_ylabel("RVP''²")
    ax3.set_xlabel("Time (s, from cycle start)")
    ax3.set_title("Squared second derivative")
    ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"RV pressure analysis — cycle {cycle_num}", fontsize=10)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_analysis(
    h5_path: Path,
    co_method: str,
    output_dir: Path,
    ecg_lead: str,
    verbose: bool = False,
    deriv_smooth: Optional[DerivativeSmoothConfig] = None,
) -> int:
    """Load HDF5, run full pipeline, write CSV and PNGs. Returns 0 on success."""
    if co_method.upper() == "TDCO":
        co_attr = TDCOL_ATTR
    elif co_method.upper() == "FICK_CO":
        co_attr = FICK_ATTR
    else:
        print(f"Error: --co-method must be TDCO or Fick_CO, got {co_method}", file=sys.stderr)
        return 1

    if verbose:
        print("Step 1/6: Loading HDF5 file...")
    with h5py.File(h5_path, "r") as f:
        if ecg_lead not in f:
            print(f"Error: dataset '{ecg_lead}' not found in {h5_path}", file=sys.stderr)
            return 1
        if RHC_PRESSURE not in f:
            print(f"Error: dataset '{RHC_PRESSURE}' not found in {h5_path}", file=sys.stderr)
            return 1
        if FS_ATTR not in f.attrs:
            print(f"Error: attribute '{FS_ATTR}' not found in {h5_path}", file=sys.stderr)
            return 1
        if co_attr not in f.attrs:
            print(f"Error: attribute '{co_attr}' not found (required for {co_method})", file=sys.stderr)
            return 1

        ecg = np.asarray(f[ecg_lead][:]).flatten()
        pressure = np.asarray(f[RHC_PRESSURE][:]).flatten()
        fs = float(f.attrs[FS_ATTR])
        co_l_per_min = float(f.attrs[co_attr])
    if verbose:
        print("  Loaded ECG and RHC_pressure (fs={:.1f} Hz, CO={:.2f} L/min).".format(fs, co_l_per_min))

    if ecg.size != pressure.size:
        print("Error: ECG and RHC_pressure lengths differ", file=sys.stderr)
        return 1

    if verbose:
        print("Step 2/6: Preprocessing ECG and detecting R-waves...")
    r_peaks = detect_r_waves(ecg, fs)
    if len(r_peaks) < 2:
        print("Error: fewer than 2 R-peaks detected; cannot define any cardiac cycle", file=sys.stderr)
        return 1
    if verbose:
        print("  Detected {} R-peaks.".format(len(r_peaks)))

    if verbose:
        print("Step 3/6: Segmenting RV pressure into R-R intervals...")
    segments = segment_pressure_by_rr_with_fs(pressure, r_peaks, fs)
    if not segments:
        print("Error: no valid R-R segments (check RR length filters)", file=sys.stderr)
        return 1
    if verbose:
        print("  {} cardiac cycles to analyze.".format(len(segments)))
        dcfg = deriv_smooth if deriv_smooth is not None else DerivativeSmoothConfig()
        print(f"  Derivative smoothing: {dcfg.method}.")

    basename = h5_path.stem
    csv_dir = output_dir / "csv_files"
    png_dir = output_dir / "png_files"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Step 4/6: Analyzing each cardiac cycle and generating diagnostic plots...")
    rows: list[dict[str, Any]] = []
    iterator = enumerate(segments)
    if verbose:
        iterator = tqdm(
            iterator,
            total=len(segments),
            unit="cycle",
            desc="Cardiac cycles",
        )
    for i, (rvp, t, rr_sec) in iterator:
        cycle_num = i + 1
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(cycle=cycle_num, RR_s=round(rr_sec, 3))
        row, plot_data = analyze_one_cycle(
            rvp, t, rr_sec, fs, co_l_per_min, deriv_smooth=deriv_smooth
        )
        row["cycle"] = cycle_num
        row["CO_method"] = co_method
        row["CO_L_per_min"] = co_l_per_min
        rows.append(row)
        if plot_data is not None and row.get("cycle_ok"):
            save_diagnostic_plot(
                plot_data,
                png_dir / f"{basename}_{cycle_num}.png",
                cycle_num,
            )

    if verbose:
        print("Step 5/6: Writing results to CSV...")
    df = pd.DataFrame(rows)
    csv_path = csv_dir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    n_ok = len([r for r in rows if r.get("cycle_ok")])
    if verbose:
        print("Step 6/6: Done.")
    print(f"Wrote {csv_path} ({len(rows)} cycles).")
    print(f"Wrote {n_ok} diagnostic PNGs to {png_dir}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze RV pressure from HDF5: ECG-gated cycles and PV-loop derived measures."
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to HDF5 file (e.g. hdf5_files/TRM127-RHC1.h5)",
    )
    parser.add_argument(
        "--co-method",
        type=str,
        default="TDCO",
        choices=["TDCO", "Fick_CO"],
        help="Cardiac output: TDCO (thermodilution) or Fick_CO (Fick). Default: TDCO",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for csv_files/ and png_files/ (default: same as HDF5 file directory)",
    )
    parser.add_argument(
        "--ecg-lead",
        type=str,
        default=ECG_LEAD_II,
        help=f"ECG dataset name (default: {ECG_LEAD_II})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print step-by-step progress and show tqdm progress bar for cardiac cycle analysis",
    )
    parser.add_argument(
        "--deriv-smooth",
        type=str,
        default="none",
        choices=["none", "savgol", "kalman"],
        help="Smooth RVP' and RVP'' before peak picking: none (default), savgol, or kalman",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=DEFAULT_SAVGOL_WINDOW,
        metavar="N",
        help=f"Savitzky-Golay window length in samples (odd; if even, bumped). Default: {DEFAULT_SAVGOL_WINDOW}",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=DEFAULT_SAVGOL_POLYORDER,
        metavar="P",
        help=f"Savitzky-Golay polynomial order (must be < window). Default: {DEFAULT_SAVGOL_POLYORDER}",
    )
    parser.add_argument(
        "--kalman-level-noise",
        type=float,
        default=DEFAULT_KALMAN_LEVEL_NOISE,
        metavar="X",
        help=f"Kalman UCM level component noise. Default: {DEFAULT_KALMAN_LEVEL_NOISE}",
    )
    parser.add_argument(
        "--kalman-trend-noise",
        type=float,
        default=DEFAULT_KALMAN_TREND_NOISE,
        metavar="X",
        help=f"Kalman UCM trend component noise. Default: {DEFAULT_KALMAN_TREND_NOISE}",
    )
    parser.add_argument(
        "--kalman-observation-noise",
        type=float,
        default=DEFAULT_KALMAN_OBSERVATION_NOISE,
        metavar="X",
        help=f"Kalman observation noise. Default: {DEFAULT_KALMAN_OBSERVATION_NOISE}",
    )
    args = parser.parse_args()
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        print(f"Error: file not found: {h5_path}", file=sys.stderr)
        return 1
    output_dir = args.output_dir.resolve() if args.output_dir else h5_path.parent
    deriv_cfg = DerivativeSmoothConfig(
        method=args.deriv_smooth,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        kalman_level_noise=args.kalman_level_noise,
        kalman_trend_noise=args.kalman_trend_noise,
        kalman_observation_noise=args.kalman_observation_noise,
    )
    return run_analysis(
        h5_path,
        args.co_method,
        output_dir,
        args.ecg_lead,
        verbose=args.verbose,
        deriv_smooth=deriv_cfg,
    )


if __name__ == "__main__":
    sys.exit(main())
