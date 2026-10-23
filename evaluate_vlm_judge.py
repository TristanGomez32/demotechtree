#!/usr/bin/env python3
"""Compare cached VLM scores with make_defect_clips.py ground truth.

Outputs ordinal Krippendorff alpha and score metrics by defect type, plus exact
accuracy grouped by low/medium/high model confidence. No API calls are made.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
try:
    import krippendorff
except ImportError:
    krippendorff = None

def norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(v or "").strip().lower())

def noise_type(v: Any) -> str | None:
    s = norm(v)
    if s in {"none", "clean", "original", "control"}: return "none"
    if "underexpos" in s or "underexposure" in s or "underexposed" in s: return "underexposure"
    if "blur" in s or "defocus" in s: return "blur"
    return None

def score(v: Any) -> float:
    try:
        x = float(v)
        return x if x.is_integer() and 0 <= x <= 4 else float("nan")
    except (TypeError, ValueError):
        return float("nan")

def confidence(v: Any) -> str | None:
    s = norm(v)
    if s in {"low", "lowconfidence"}: return "low"
    if s in {"medium", "med", "moderate", "mediumconfidence"}: return "medium"
    if s in {"high", "highconfidence"}: return "high"
    return None

def parse_output(path: Path):
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "success": return None, rec, f"status={rec.get('status')}"
        raw = rec.get("output_text") or ""
        try: parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m: return None, rec, "output_text is not JSON"
            parsed = json.loads(m.group(0))
        return parsed, rec, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

def alpha_ordinal(gt, pred):
    pairs = [(g, p) for g, p in zip(gt, pred) if np.isfinite(g) and np.isfinite(p)]
    if len(pairs) < 2 or krippendorff is None: return None
    try:
        a = krippendorff.alpha(
            reliability_data=np.asarray([[g for g, _ in pairs], [p for _, p in pairs]], dtype=float),
            level_of_measurement="ordinal")
        return float(a) if np.isfinite(a) else None
    except (ValueError, ZeroDivisionError, TypeError): return None

def calc_metrics(frame):
    valid = frame.dropna(subset=["ground_truth_score", "predicted_score"])
    if valid.empty:
        return {"n": 0, "krippendorff_alpha_ordinal": None, "exact_accuracy": None,
                "within_one_accuracy": None, "mae": None, "mean_signed_error_pred_minus_gt": None}
    gt = valid.ground_truth_score.astype(float).to_numpy()
    pred = valid.predicted_score.astype(float).to_numpy()
    return {"n": len(valid), "krippendorff_alpha_ordinal": alpha_ordinal(gt.tolist(), pred.tolist()),
            "exact_accuracy": float(np.mean(gt == pred)),
            "within_one_accuracy": float(np.mean(np.abs(gt - pred) <= 1)),
            "mae": float(np.mean(np.abs(pred - gt))),
            "mean_signed_error_pred_minus_gt": float(np.mean(pred - gt))}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/clips_manifest.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/vlm_judge"))
    ap.add_argument("--deployment", default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("results/vlm_evaluation"))
    ap.add_argument("--include-errors", action="store_true")
    args = ap.parse_args()
    if krippendorff is None: ap.error("Install dependency: python -m pip install krippendorff")
    if not args.manifest.exists(): ap.error(f"Manifest not found: {args.manifest}")
    if not args.results_dir.exists(): ap.error(f"Results directory not found: {args.results_dir}")
    df = pd.read_csv(args.manifest)
    required = {"clip_id", "defect_code", "severity_level"}
    if not required.issubset(df.columns): ap.error(f"Manifest needs columns: {sorted(required)}")
    df["clip_id"] = df.clip_id.astype(str).map(lambda x: Path(x).stem)
    df["_noise"] = df.defect_code.map(noise_type)
    df["_severity"] = pd.to_numeric(df.severity_level, errors="coerce")
    bad = df["_severity"].isna() | (df["_severity"] < 0) | (df["_severity"] > 4) | (df["_severity"] % 1 != 0)
    if bad.any(): ap.error("severity_level must be an integer from 0 to 4.")
    if df.clip_id.duplicated().any(): ap.error("clip_id must be unique in the manifest.")
    lookup = df.set_index("clip_id").to_dict("index")
    folders = [args.results_dir / args.deployment] if args.deployment else sorted(p for p in args.results_dir.iterdir() if p.is_dir())
    files = [f for d in folders if d.exists() for f in d.glob("*.json")]
    if not files: ap.error("No per-clip JSON result files found.")
    rows, diagnostics = [], []
    for path in files:
        parsed, rec, err = parse_output(path)
        if err:
            diagnostics.append({"result_path": str(path), "error": err})
            continue
        clip_id = Path(str(rec.get("clip_id") or Path(rec.get("clip_path", path.stem)).stem)).stem
        gtrow = lookup.get(clip_id)
        if gtrow is None:
            stripped = re.sub(r"__[0-9a-fA-F]{8,}$", "", clip_id)
            gtrow = lookup.get(stripped)
            if gtrow is not None: clip_id = stripped
        if gtrow is None:
            diagnostics.append({"result_path": str(path), "clip_id": clip_id, "error": "no matching clip_id in manifest"})
            continue
        kind, sev = gtrow["_noise"], float(gtrow["_severity"])
        if kind is None:
            diagnostics.append({"result_path": str(path), "clip_id": clip_id, "error": f"unknown defect_code: {gtrow['defect_code']}"})
            continue
        usage = rec.get("usage") or {}
        common = {"deployment": rec.get("deployment") or path.parent.name, "clip_id": clip_id,
                  "defect_code": kind, "severity_level": sev, "param_name": gtrow.get("param_name"),
                  "param_value": gtrow.get("param_value"), "prompt_sha256": rec.get("prompt_sha256"),
                  "result_path": str(path), "input_tokens": usage.get("prompt_tokens"),
                  "output_tokens": usage.get("completion_tokens")}
        for noise, predkey, confkey in [
            ("blur", "blur_score", "blur_confidence"),
            ("underexposure", "underexposure_score", "underexposure_confidence")]:
            gt = sev if kind == noise else 0.0
            pred, conf = score(parsed.get(predkey)), confidence(parsed.get(confkey))
            rows.append({**common, "noise_type": noise, "ground_truth_score": gt,
                         "predicted_score": pred, "confidence": conf,
                         "exact_match": bool(pred == gt) if np.isfinite(pred) else np.nan,
                         "within_one_level": bool(abs(pred - gt) <= 1) if np.isfinite(pred) else np.nan,
                         "absolute_error": abs(pred - gt) if np.isfinite(pred) else np.nan})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "per_clip_comparison.csv", index=False)
    metric_rows, cal_rows = [], []
    if not detail.empty:
        for (deployment, noise), sub in detail.groupby(["deployment", "noise_type"], dropna=False):
            metric_rows.append({"deployment": deployment, "noise_type": noise, **calc_metrics(sub)})
            for level in ["low", "medium", "high"]:
                group = sub[sub.confidence == level].dropna(subset=["ground_truth_score", "predicted_score"])
                n = len(group)
                cal_rows.append({"deployment": deployment, "noise_type": noise, "confidence": level, "n": n,
                    "exact_accuracy": float((group.ground_truth_score == group.predicted_score).mean()) if n else None,
                    "within_one_accuracy": float((abs(group.ground_truth_score - group.predicted_score) <= 1).mean()) if n else None,
                    "mae": float(abs(group.predicted_score - group.ground_truth_score).mean()) if n else None})
            valid = sub.dropna(subset=["ground_truth_score", "predicted_score"])
            cm = np.zeros((5, 5), dtype=int)
            for _, r in valid.iterrows(): cm[int(r.ground_truth_score), int(r.predicted_score)] += 1
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(deployment))
            pd.DataFrame(cm, index=[f"GT_{i}" for i in range(5)],
                         columns=[f"Pred_{i}" for i in range(5)]).to_csv(
                             args.output_dir / f"confusion_{safe}_{noise}.csv", index_label="ground_truth")
    metric_df, cal_df = pd.DataFrame(metric_rows), pd.DataFrame(cal_rows)
    metric_df.to_csv(args.output_dir / "metrics_by_noise.csv", index=False)
    cal_df.to_csv(args.output_dir / "confidence_calibration.csv", index=False)
    if args.include_errors: pd.DataFrame(diagnostics).to_csv(args.output_dir / "diagnostics.csv", index=False)
    report = {"result_files_found": len(files), "matched_clip_outputs": int(detail.clip_id.nunique()) if not detail.empty else 0,
              "metrics": metric_rows, "confidence_calibration": cal_rows, "diagnostics": diagnostics,
              "ground_truth_rule": "severity_level applies to defect_code; the other defect dimension has GT 0, including clean controls.",
              "confidence_accuracy_definition": "Exact equality of VLM score and GT severity; within-one accuracy is also reported.",
              "caveats": ["Ordinal alpha may be undefined if there is no expected disagreement.",
                          "Low/medium/high are qualitative labels, so this is empirical accuracy by confidence, not strict probability calibration.",
                          "A 2-3 clip pilot is too small for stable alpha or calibration estimates."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Per-clip comparison: {args.output_dir / 'per_clip_comparison.csv'}")
    print(f"Metrics by noise: {args.output_dir / 'metrics_by_noise.csv'}")
    print(f"Confidence calibration: {args.output_dir / 'confidence_calibration.csv'}")
    if not metric_df.empty:
        print("\nMetrics by noise\n", metric_df.to_string(index=False))
        print("\nAccuracy by confidence\n", cal_df.to_string(index=False))
    if diagnostics: print(f"\n{len(diagnostics)} result(s) skipped; see report.json", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
