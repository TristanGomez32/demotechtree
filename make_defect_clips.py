#!/usr/bin/env python
"""
make_defect_clips.py
====================
Étape 1 du mini-projet « VLM-as-judge » :

  1. télécharge N vidéos d'EPIC-KITCHENS-100 (miroir Hugging Face, licence CC BY-NC 4.0),
  2. découpe un clip court dans chaque vidéo (redimensionné, fps réduit),
  3. génère, pour chaque clip, une version propre + des versions dégradées :
       - flou gaussien      (4 niveaux de sigma, en pixels à la largeur de sortie)
       - sous-exposition    (4 niveaux, en stops d'exposition / EV, appliqués en lumière linéaire)
  4. écrit un CSV (une ligne par clip généré) avec le niveau d'intensité et des
     mesures objectives (netteté = variance du Laplacien, luminance moyenne).

Les mesures sont calculées sur les frames en mémoire, AVANT l'encodage mp4v,
pour qu'elles ne soient pas polluées par les artefacts de compression.

Installation (Windows / Linux / macOS) :
    pip install huggingface_hub opencv-python numpy

Exemples :
    python make_defect_clips.py --dry-run          # liste les 10 vidéos + taille, ne télécharge rien
    python make_defect_clips.py                     # télécharge + génère
    python make_defect_clips.py --local-videos D:\\EPIC\\videos   # vidéos déjà sur disque (script officiel, etc.)
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REPO = "awsaf49/epic_kitchens_100"  # miroir HF d'EPIC-KITCHENS-100
VIDEO_RE = re.compile(r"(P\d{2}_\d+)\.mp4$", re.IGNORECASE)

CSV_COLUMNS = [
    "clip_id",
    "reference_clip_id",     # clip propre dont celui-ci est dérivé (identique si defect_code == NONE)
    "source_video",
    "source_repo_path",
    "start_sec",
    "duration_sec",
    "fps",
    "width",
    "height",
    "defect_code",           # NONE | BLUR | UNDEREXPOSED
    "severity_level",        # 0 = propre, 1..K = intensité croissante
    "param_name",            # gaussian_sigma_px | exposure_ev_stops | none
    "param_value",
    "measured_sharpness_lapvar",
    "measured_mean_luma",
    "seed",
    "output_path",
]


# --------------------------------------------------------------------------- #
# 1. Sélection + téléchargement
# --------------------------------------------------------------------------- #
def pick_remote_videos(repo_id: str, n: int, min_size_mb: float):
    """Retourne les n plus petites vidéos du dépôt (>= min_size_mb), triées par taille."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    candidates = []
    for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        if isinstance(item, RepoFile) and VIDEO_RE.search(item.path):
            if item.size and item.size >= min_size_mb * 1e6:
                candidates.append((item.size, item.path))
    candidates.sort()
    if len(candidates) < n:
        sys.exit(f"Seulement {len(candidates)} vidéos trouvées dans {repo_id} (demandé : {n}).")
    return candidates[:n]


def download_videos(repo_id: str, picks, raw_dir: Path):
    from huggingface_hub import hf_hub_download

    paths = []
    for size, repo_path in picks:
        print(f"  ↓ {repo_path}  ({size / 1e6:.0f} MB)")
        p = hf_hub_download(repo_id, filename=repo_path, repo_type="dataset", local_dir=str(raw_dir))
        paths.append((Path(p), repo_path))
    return paths


# --------------------------------------------------------------------------- #
# 2. Découpe d'un clip
# --------------------------------------------------------------------------- #
def extract_clip(video_path: Path, start_sec: float, duration: float, out_width: int, target_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_dur = n_total / src_fps if n_total > 0 else None

    if total_dur is not None:
        start_sec = min(start_sec, max(0.0, total_dur - duration))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * src_fps))

    stride = max(1, round(src_fps / target_fps))
    out_fps = src_fps / stride
    n_read = int(duration * src_fps)

    frames, i = [], 0
    while i < n_read:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h0, w0 = frame.shape[:2]
            out_h = int(round(out_width * h0 / w0 / 2) * 2)  # hauteur paire
            frames.append(cv2.resize(frame, (out_width, out_h), interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"Aucune frame lue dans {video_path}")
    return frames, out_fps, start_sec


# --------------------------------------------------------------------------- #
# 3. Défauts
# --------------------------------------------------------------------------- #
def apply_blur(frames, sigma: float):
    """Flou gaussien ; ksize=(0,0) => taille de noyau déduite de sigma."""
    return [cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma) for f in frames]


def _underexposure_lut(ev_stops: float) -> np.ndarray:
    """LUT 8 bits : sRGB -> linéaire, gain 2^-EV, linéaire -> sRGB (sous-exposition physique)."""
    x = np.arange(256, dtype=np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    lin = lin * (2.0 ** (-ev_stops))
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def apply_underexposure(frames, ev_stops: float):
    lut = _underexposure_lut(ev_stops)
    return [cv2.LUT(f, lut) for f in frames]


# --------------------------------------------------------------------------- #
# 4. Mesures objectives
# --------------------------------------------------------------------------- #
def measure(frames, every: int = 10):
    """Variance du Laplacien (netteté) et luminance moyenne sur une frame sur `every`."""
    lap_vars, lumas = [], []
    for f in frames[::every]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        lumas.append(float(gray.mean()))
    return float(np.mean(lap_vars)), float(np.mean(lumas))


# --------------------------------------------------------------------------- #
# 5. Écriture vidéo
# --------------------------------------------------------------------------- #
def write_video(path: Path, frames, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter impossible à ouvrir pour {path}")
    for f in frames:
        writer.write(f)
    writer.release()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--min-size-mb", type=float, default=20.0, help="ignore les fichiers plus petits (stubs)")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--local-videos", type=Path, default=None,
                    help="dossier de .MP4 déjà présents : saute le téléchargement")
    ap.add_argument("--duration", type=float, default=6.0, help="durée des clips (s)")
    ap.add_argument("--width", type=int, default=640, help="largeur de sortie (px)")
    ap.add_argument("--fps", type=float, default=25.0, help="fps cible (approx., par sous-échantillonnage)")
    ap.add_argument("--blur-sigmas", type=float, nargs="+", default=[1, 2, 4, 8],
                    help="sigma gaussien en px (à la largeur de sortie), un niveau par valeur")
    ap.add_argument("--ev-stops", type=float, nargs="+", default=[0.5, 1, 2, 3],
                    help="sous-exposition en stops EV, un niveau par valeur")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="liste les vidéos et tailles, sans rien télécharger")
    args = ap.parse_args()

    raw_dir = args.out / "raw"
    clips_dir = args.out / "clips"
    csv_path = args.out / "clips_manifest.csv"

    # ---- sources ----
    if args.local_videos:
        vids = sorted(p for p in args.local_videos.rglob("*") if VIDEO_RE.search(p.name))[: args.n_videos]
        if not vids:
            sys.exit(f"Aucune vidéo PXX_YY.MP4 trouvée dans {args.local_videos}")
        sources = [(p, p.name) for p in vids]
    else:
        print(f"Recherche des {args.n_videos} plus petites vidéos dans {args.repo} …")
        picks = pick_remote_videos(args.repo, args.n_videos, args.min_size_mb)
        total_gb = sum(s for s, _ in picks) / 1e9
        print(f"→ {len(picks)} vidéos, {total_gb:.2f} GB au total :")
        for s, p in picks:
            print(f"   {s / 1e6:8.0f} MB  {p}")
        if args.dry_run:
            return
        sources = download_videos(args.repo, picks, raw_dir)

    # ---- génération ----
    rows = []
    for idx, (video_path, repo_path) in enumerate(sources):
        vid_name = VIDEO_RE.search(video_path.name).group(1)
        rng = random.Random(f"{args.seed}-{vid_name}")

        # position de départ : tirage dans la partie « utile » de la vidéo (déterministe via la seed)
        cap = cv2.VideoCapture(str(video_path))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur_src = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps_src
        cap.release()
        start = rng.uniform(0.1, 0.9) * max(0.0, dur_src - args.duration)

        print(f"[{idx + 1}/{len(sources)}] {vid_name} : clip de {args.duration:.0f}s à t≈{start:.0f}s")
        frames, out_fps, start = extract_clip(video_path, start, args.duration, args.width, args.fps)
        h, w = frames[0].shape[:2]
        base_id = f"{vid_name}_t{int(start):05d}"

        variants = [("NONE", 0, "none", 0.0, frames)]
        for lvl, sigma in enumerate(args.blur_sigmas, start=1):
            variants.append(("BLUR", lvl, "gaussian_sigma_px", sigma, apply_blur(frames, sigma)))
        for lvl, ev in enumerate(args.ev_stops, start=1):
            variants.append(("UNDEREXPOSED", lvl, "exposure_ev_stops", ev, apply_underexposure(frames, ev)))

        for code, lvl, pname, pval, vframes in variants:
            suffix = "clean" if code == "NONE" else f"{code.lower()}_L{lvl}"
            clip_id = f"{base_id}_{suffix}"
            out_path = clips_dir / f"{clip_id}.mp4"
            sharp, luma = measure(vframes)
            write_video(out_path, vframes, out_fps)
            rows.append({
                "clip_id": clip_id,
                "reference_clip_id": f"{base_id}_clean",
                "source_video": video_path.name,
                "source_repo_path": repo_path,
                "start_sec": round(start, 2),
                "duration_sec": round(len(frames) / out_fps, 2),
                "fps": round(out_fps, 2),
                "width": w,
                "height": h,
                "defect_code": code,
                "severity_level": lvl,
                "param_name": pname,
                "param_value": pval,
                "measured_sharpness_lapvar": round(sharp, 3),
                "measured_mean_luma": round(luma, 3),
                "seed": args.seed,
                "output_path": str(out_path.as_posix()),
            })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} clips écrits dans {clips_dir}/")
    print(f"Manifeste : {csv_path}")


if __name__ == "__main__":
    main()
