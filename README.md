# RVP analysis

This repository contains Python tools to **extract** right-heart catheter waveform segments from WFDB-style recordings into HDF5, then **analyze** right ventricular (RV) pressure on ECG-gated cardiac cycles and export metrics and diagnostic plots.

## Requirements

- Python 3.9+ recommended  
- Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists packages used by the scripts (e.g. `wfdb`, `h5py`, `neurokit2`, `tsmoothie`, `scipy`, `pandas`, `matplotlib`, `peakutils`, `tqdm`). The analysis script pulls in additional transitive dependencies (e.g. `simdkalman` via `tsmoothie` for Kalman smoothing).

## Repository layout

| Path | Purpose |
|------|---------|
| `raw_data/` | Per-study WFDB files: `{name}.hea`, `{name}.dat`, and `{name}.json` metadata |
| `hdf5_files/` | HDF5 outputs from the extraction script (`{name}.h5`) |
| `scripts/extract_wfbd_to_hdf5.py` | WFDB → HDF5 extraction |
| `scripts/analyze_rv_pressure.py` | HDF5 → CSV + PNG analysis |

Output directories for analysis are created under your chosen output folder (by default, next to the HDF5 file):

- `csv_files/{basename}.csv` — per-cycle tabular results  
- `png_files/{basename}_{cycle}.png` — diagnostic figures per successful cycle  

---

## 1. `extract_wfbd_to_hdf5.py`

### What it does

- Reads a WFDB record from `raw_data/{dataset_name}.hea` + `raw_data/{dataset_name}.dat`.
- Reads metadata from `raw_data/{dataset_name}.json`.
- Uses **`ChamEvents_in_s.RV`** (seconds) to define a **symmetric time window**: **10 s before** and **10 s after** RV, clipped to the recording (`[0, recording_end)`).
- Writes aligned snippets of these signals (when present in the record) to **`hdf5_files/{dataset_name}.h5`**:
  - `ECG_lead_I`, `ECG_lead_II`, `ECG_lead_V1`, `RHC_pressure`
- Copies scalar cardiac output-related values from JSON when available (`maclabMeas` keys with exact spelling, including trailing spaces).

### Usage

```bash
python scripts/extract_wfbd_to_hdf5.py <dataset_name>
```

**Example**

```bash
python scripts/extract_wfbd_to_hdf5.py TRM127-RHC1
```

This expects:

- `raw_data/TRM127-RHC1.hea`
- `raw_data/TRM127-RHC1.dat`
- `raw_data/TRM127-RHC1.json`

and writes:

- `hdf5_files/TRM127-RHC1.h5`

### JSON expectations

- **`ChamEvents_in_s.RV`** (required): RV event time in **seconds** relative to the WFDB record.
- **`maclabMeas`** (optional): may include  
  - `"Fick COL/min "`  
  - `"TDCOL/min    "`  
  (keys include trailing spaces as in the source system.)

If a scalar is missing, the script still runs but may omit the corresponding HDF5 attribute (with a warning).

### HDF5 contents

**Datasets** (1D arrays, same length, sampling rate `fs_Hz`):

- Waveforms listed above that were found in the `.hea` channel list.

**Attributes**

| Attribute | Meaning |
|-----------|---------|
| `fs_Hz` | Sampling rate (Hz) |
| `rv_timestamp_sec` | `ChamEvents_in_s.RV` from JSON |
| `window_start_sec` | Time (s) of the **first** extracted sample |
| `window_end_sec` | Time (s) of the **last** extracted sample |
| `Fick_COL_per_min` | Fick cardiac output (L/min), if present |
| `TDCOL_per_min` | Thermodilution CO (L/min), if present |

### Common errors

- Missing `.hea` or `.json` → script exits with an error.
- Missing `.dat` → WFDB cannot load samples; error message explains missing file.
- No `ChamEvents_in_s.RV` → cannot define the window.
- RV window falls entirely past the recording, or the window is empty after clipping → error.
- None of the four requested signals exist in the record → error.

---

## 2. `analyze_rv_pressure.py`

### What it does

1. Loads **ECG** and **RV pressure** from an HDF5 file, plus sampling rate `fs_Hz` and a **cardiac output** attribute (see `--co-method` below).
2. Requires ECG and pressure traces to have the **same length** (as produced by extraction).
3. Detects **R-waves** on the ECG (NeuroKit2 + spectral smoothing, same general idea as a “TEE_optical_flow” style pipeline).
4. Segments **RHC_pressure** into cycles between consecutive R-peaks, keeping only segments whose R–R interval is between **0.4 s and 1.5 s** (configurable in code as `RR_MIN_SEC` / `RR_MAX_SEC`).
5. For each retained cycle, computes time-domain and model-based metrics (e.g. IVCT, ET, IVRT, ESP, pressures, P–V style quantities such as Ees, Ea, RVEF, RVMPI, beta, Eed where applicable).
6. Optionally **smooths RVP′ and RVP″** (first and second time derivatives of pressure) **before** peak detection on \((\mathrm{RVP}'')^2\) and before extrema of RVP′ — see **Derivative smoothing** below.
7. Writes one **CSV** with all cycles (successful and failed) and **PNG** diagnostics for cycles that completed successfully.

### Usage

```bash
python scripts/analyze_rv_pressure.py path/to/file.h5 [options]
```

**Minimal example** (HDF5 next to `csv_files/` / `png_files/` under the same parent directory as the file):

```bash
python scripts/analyze_rv_pressure.py hdf5_files/TRM127-RHC1.h5
```

Use an absolute or relative path to your `.h5` file.

### Main options

| Option | Default | Description |
|--------|---------|-------------|
| `--co-method` | `TDCO` | Which HDF5 attribute supplies cardiac output (L/min): `TDCO` uses `TDCOL_per_min`; `Fick_CO` uses `Fick_COL_per_min`. The chosen attribute **must** exist on the file. |
| `--output-dir` | HDF5 parent | Directory under which `csv_files/` and `png_files/` are created. |
| `--ecg-lead` | `ECG_lead_II` | Name of the ECG dataset in HDF5 to use for R-peak detection. |
| `-v`, `--verbose` | off | Step-by-step log + tqdm progress over cycles. |

### Derivative smoothing (`--deriv-smooth`)

After computing RVP′ and RVP″ with `numpy.gradient`, the script can smooth **both** derivatives before squaring RVP″ for peak picking and before taking max/min of RVP′.

| `--deriv-smooth` | Behavior |
|------------------|----------|
| `none` | No extra smoothing (default; matches the original behavior). |
| `savgol` | Savitzky–Golay filter (`scipy.signal.savgol_filter`) on RVP′ and RVP″ with the same window. |
| `kalman` | `tsmoothie` `KalmanSmoother` with `component="level_trend"`, applied separately to each derivative. |

Tuning flags (only affect the corresponding method):

| Flag | Default | Meaning |
|------|---------|---------|
| `--savgol-window` | 11 | Window length in **samples**; if even, it is bumped to the next odd value, then clamped to valid length for short cycles. |
| `--savgol-polyorder` | 3 | Polynomial order (must be &lt; window length; if the segment is too short, smoothing is skipped with a warning). |
| `--kalman-level-noise` | 0.1 | UCM level noise. |
| `--kalman-trend-noise` | 0.1 | UCM trend noise. |
| `--kalman-observation-noise` | 1.0 | Observation noise. |

**Example with Savitzky–Golay**

```bash
python scripts/analyze_rv_pressure.py hdf5_files/TRM127-RHC1.h5 --deriv-smooth savgol --savgol-window 15 --savgol-polyorder 3 -v
```

### Spectral smoothing of \((\mathrm{RVP}'')^2\) (`--rvp2-sq-spectral`)

After derivative smoothing (if any) and squaring RVP″, you can apply a **second** stage: FFT-based **spectral smoothing** (`tsmoothie.SpectralSmoother`, same family as the ECG path) **only** to the \((\mathrm{RVP}'')^2\) trace, immediately **before** `peakutils` peak picking. This is off by default.

| Flag | Default | Meaning |
|------|---------|---------|
| `--rvp2-sq-spectral` | off | Enable spectral smoothing of \((\mathrm{RVP}'')^2\). |
| `--rvp2-sq-spectral-fraction` | `0.2` | Fraction of FFT frequencies kept; must be in `(0, 1)` (same default as ECG spectral smoothing). |
| `--rvp2-sq-spectral-pad` | `20` | Symmetric padding length at each edge (same default as ECG). |

**Recommended workflow** for cleaner peaks: combine derivative smoothing with this flag, e.g. `--deriv-smooth savgol --rvp2-sq-spectral` (or `kalman` instead of `savgol`). You can also enable `--rvp2-sq-spectral` with `--deriv-smooth none` to smooth only the squared second derivative.

Very short cycles may skip spectral smoothing with a stderr warning and use unsmoothed \((\mathrm{RVP}'')^2\).

### HDF5 inputs the script expects

- Datasets: `--ecg-lead` (default `ECG_lead_II`) and **`RHC_pressure`**.
- Attributes: **`fs_Hz`**; and **`TDCOL_per_min`** or **`Fick_COL_per_min`** depending on `--co-method`.

### Outputs

- **`csv_files/{stem}.csv`**: one row per cardiac cycle attempt with columns such as cycle index, `cycle_ok`, hemodynamic intervals, volumes/pressures where fitted, and metadata (`CO_method`, `CO_L_per_min`, etc.).
- **`png_files/{stem}_{n}.png`**: three-panel plot (RV pressure + optional sine fit, RVP′, \((\mathrm{RVP}'')^2\) with detected peaks) for cycles with `cycle_ok` True.

---

## End-to-end workflow

```text
raw_data/{name}.hea + .dat + .json
        │
        ▼  extract_wfbd_to_hdf5.py
hdf5_files/{name}.h5
        │
        ▼  analyze_rv_pressure.py
csv_files/{name}.csv    png_files/{name}_*.png
```

Example:

```bash
python scripts/extract_wfbd_to_hdf5.py TRM127-RHC1
python scripts/analyze_rv_pressure.py hdf5_files/TRM127-RHC1.h5 --co-method TDCO -v
```

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Extraction fails on JSON / RV | `ChamEvents_in_s.RV` present and numeric. |
| Analysis: missing dataset | HDF5 contains `ECG_lead_II` (or the lead you pass) and `RHC_pressure`. |
| Analysis: missing CO attribute | Use `--co-method` that matches an attribute you actually wrote (e.g. Fick vs TDCO). |
| Analysis: no valid cycles | R–R outside 0.4–1.5 s, or too few R-peaks; check ECG quality and `--ecg-lead`. |
| Derivative smoothing warnings | Short segments may skip Savitzky–Golay or Kalman; warnings go to stderr. |
| Spectral \((\mathrm{RVP}'')^2\) warnings | Very short cycles or invalid fraction/pad; falls back to unsmoothed \((\mathrm{RVP}'')^2\). |

For full behavior (filters, peak rules, and metric definitions), see the docstrings and constants in `scripts/analyze_rv_pressure.py`.
