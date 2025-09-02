import os
import glob
import math
import pandas as pd

def summarize_variants(root_dir: str,
                       metrics_filename: str = "final_metrics.csv",
                       exclude_dirs=("papers",),
                       save_csv: str | None = None) -> pd.DataFrame:
    """
    Build a summary of all run variants found under `root_dir`.

    For each immediate subfolder (variant), this:
      - Reads `<variant>/**/final_metrics.csv` (first match)
      - Computes mean and std (sample) for `utilization`
      - Computes mean percentage for each heuristic column (any column starting with 'perc_')
      - Adds a 'rows' column with episode count

    Parameters
    ----------
    root_dir : str
        Directory that contains your variant folders (e.g., cut_1, cut_1_fixed, rs_up_with_buffer_with_mask, …).
    metrics_filename : str
        Name of the metrics CSV (default: 'final_metrics.csv').
    exclude_dirs : tuple[str]
        Folder names to skip (default: ('papers',)).
    save_csv : str | None
        Optional path to save the resulting summary as CSV.

    Returns
    -------
    pandas.DataFrame
        One row per variant with mean/std utilization and average % for each heuristic.
    """
    rows = []

    # helper to find the metrics file in each variant folder (top or nested)
    def _find_metrics_file(variant_path: str) -> str | None:
        direct = os.path.join(variant_path, metrics_filename)
        if os.path.isfile(direct):
            return direct
        # otherwise, look recursively (first match)
        matches = glob.glob(os.path.join(variant_path, "**", metrics_filename), recursive=True)
        return matches[0] if matches else None

    # iterate variant folders (only directories)
    for name in sorted(os.listdir(root_dir)):
        if name in exclude_dirs:
            continue
        vpath = os.path.join(root_dir, name)
        if not os.path.isdir(vpath):
            continue

        csv_path = _find_metrics_file(vpath)
        if not csv_path:
            # nothing to summarize for this variant
            continue

        df = pd.read_csv(csv_path)

        # detect utilization column robustly
        util_col = None
        for cand in ("utilization", "avg_utilization", "mean_utilization"):
            if cand in df.columns:
                util_col = cand
                break
        if util_col is None:
            # last resort: any column containing 'utiliz'
            for c in df.columns:
                if "utiliz" in c.lower():
                    util_col = c
                    break
        if util_col is None:
            # skip if no utilization column found
            continue

        # heuristics: any column prefixed with 'perc_'
        heuristic_cols = [c for c in df.columns if c.startswith("perc_")]
        # compute stats
        util_mean = df[util_col].mean()
        util_std = df[util_col].std(ddof=1)  # sample std
        out = {
            "variant": name,
            "rows": int(len(df)),
            "utilization_mean": util_mean,
            "utilization_std": util_std if not math.isnan(util_std) else 0.0,
        }

        # average percentages for each heuristic (kept as original percentage scale)
        for hc in heuristic_cols:
            out[hc] = df[hc].mean()

        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["variant", "rows", "utilization_mean", "utilization_std"])

    summary = pd.DataFrame(rows).sort_values("utilization_mean", ascending=False).reset_index(drop=True)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        summary.to_csv(save_csv, index=False)

    return summary


