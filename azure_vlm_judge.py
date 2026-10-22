#!/usr/bin/env python3
"""Evaluate blur and underexposure in demotechtree clips using Azure-hosted vision models.

Expected input from make_defect_clips.py:
  data/clips/*.mp4
  data/clips_manifest.csv

Results are cached per clip content hash + prompt hash + deployment + frame settings.
A completed result is never re-inferred unless --force is passed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = ROOT / "prompts" / "quality_judge.txt"
IMAGE_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_four_frames(video_path: Path, max_side: int, jpeg_quality: int) -> list[dict[str, Any]]:
    """Extract four frames at 10%, 36.7%, 63.3%, 90% of the clip duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    # Avoid the first/last encoded frames, which can be atypical.
    fractions = [0.10, 0.3667, 0.6333, 0.90]
    indices = sorted(set(min(frame_count - 1, max(0, round((frame_count - 1) * f))) for f in fractions))
    frames = []
    try:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError(f"Could not JPEG-encode frame {frame_idx} from {video_path}")
            frames.append({
                "frame_index": frame_idx,
                "mime_type": "image/jpeg",
                "base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "sha256": sha256_bytes(encoded.tobytes()),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
    finally:
        cap.release()

    if len(frames) != 4:
        raise RuntimeError(f"Expected 4 frames but extracted {len(frames)} from {video_path}")
    return frames


def load_manifest(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    if not manifest_path or not manifest_path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clip_id = (row.get("clip_id") or "").strip()
            if clip_id:
                result[clip_id] = row
    return result


def safe_id(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_")


def build_messages(prompt: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {i} (video frame index {frame['frame_index']}):"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{frame['mime_type']};base64,{frame['base64']}",
                "detail": "high",
            },
        })
    return [
        {"role": "system", "content": "You are a careful visual-quality evaluator. Follow the requested output schema exactly."},
        {"role": "user", "content": content},
    ]


def result_path(results_dir: Path, clip_id: str, deployment: str) -> Path:
    model_slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in deployment)
    return results_dir / model_slug / f"{clip_id}.json"


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-dir", type=Path, default=Path("data/clips"),
                    help="Folder containing the generated MP4 clips (default: data/clips)")
    ap.add_argument("--manifest", type=Path, default=Path("data/clips_manifest.csv"),
                    help="Optional demotechtree manifest CSV")
    ap.add_argument("--results-dir", type=Path, default=Path("results/vlm_judge"))
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    help="Azure deployment name (not necessarily the underlying model ID)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N clips, e.g. --limit 3 for a cost pilot")
    ap.add_argument("--max-side", type=int, default=768,
                    help="Downscale each frame so its longest side is at most this many pixels")
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=350)
    ap.add_argument("--force", action="store_true", help="Re-run even if a matching cached result exists")
    ap.add_argument("--dry-run", action="store_true", help="List planned work without calling Azure")
    ap.add_argument("--pause", type=float, default=0.3, help="Seconds between API calls")
    args = ap.parse_args()

    if not args.deployment:
        ap.error("Missing deployment name: pass --deployment NAME or set AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if not args.dry_run and (not endpoint or not api_key):
        ap.error("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env or environment variables")
    if not args.prompt.exists():
        ap.error(f"Prompt file not found: {args.prompt}")
    if not args.clips_dir.exists():
        ap.error(f"Clips folder not found: {args.clips_dir}")

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    prompt_hash = sha256_bytes(prompt.encode("utf-8"))
    manifest = load_manifest(args.manifest)
    clips = sorted(p for p in args.clips_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit must be >= 1")
        clips = clips[:args.limit]
    if not clips:
        print(f"No video clips found in {args.clips_dir}", file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=3,
            timeout=120.0,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "summary.jsonl"
    planned, skipped, completed, failed = 0, 0, 0, 0

    print(f"Clips selected: {len(clips)}")
    print(f"Azure deployment: {args.deployment}")
    print(f"Frames per clip: 4; max frame side: {args.max_side}px; JPEG quality: {args.jpeg_quality}")
    print(f"Prompt: {args.prompt} (SHA-256 {prompt_hash[:12]})")
    print(f"Results: {args.results_dir.resolve()}")

    for position, video_path in enumerate(clips, start=1):
        clip_id = safe_id(video_path)
        try:
            video_hash = sha256_file(video_path)
            cache_key = sha256_bytes(json.dumps({
                "video_sha256": video_hash,
                "prompt_sha256": prompt_hash,
                "deployment": args.deployment,
                "max_side": args.max_side,
                "jpeg_quality": args.jpeg_quality,
                "frame_fractions": [0.10, 0.3667, 0.6333, 0.90],
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
            }, sort_keys=True).encode("utf-8"))
            # Include the cache key in the filename so changed prompts/settings
            # preserve earlier results instead of overwriting them.
            output_path = result_path(args.results_dir, f"{clip_id}__{cache_key[:12]}", args.deployment)

            if output_path.exists() and not args.force:
                try:
                    old = json.loads(output_path.read_text(encoding="utf-8"))
                    if old.get("cache_key") == cache_key and old.get("status") == "success":
                        skipped += 1
                        print(f"[{position}/{len(clips)}] SKIP (cached): {video_path.name}")
                        continue
                except (OSError, json.JSONDecodeError):
                    pass

            planned += 1
            if args.dry_run:
                print(f"[{position}/{len(clips)}] WOULD RUN: {video_path.name} -> {output_path}")
                continue

            print(f"[{position}/{len(clips)}] RUN: {video_path.name}")
            frames = extract_four_frames(video_path, args.max_side, args.jpeg_quality)
            metadata = manifest.get(video_path.stem, {})
            started = datetime.now(timezone.utc).isoformat()
            response = client.chat.completions.create(
                model=args.deployment,
                messages=build_messages(prompt, frames),
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
            )
            raw_text = response.choices[0].message.content or ""
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            record = {
                "status": "success",
                "created_at_utc": started,
                "clip_id": clip_id,
                "clip_path": str(video_path.resolve()),
                "clip_sha256": video_hash,
                "source_metadata": metadata,
                "provider": "Azure OpenAI",
                "deployment": args.deployment,
                "prompt_path": str(args.prompt),
                "prompt_sha256": prompt_hash,
                "prompt_text": prompt,
                "cache_key": cache_key,
                "frames": [{k: v for k, v in fr.items() if k != "base64"} for fr in frames],
                "request": {
                    "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens,
                    "max_side": args.max_side,
                    "jpeg_quality": args.jpeg_quality,
                    "api_version": api_version,
                },
                "output_text": raw_text,
                "usage": usage,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(output_path)
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "status": "success", "clip_id": clip_id, "deployment": args.deployment,
                    "result_path": str(output_path), "usage": usage, "cache_key": cache_key,
                    "created_at_utc": started,
                }, ensure_ascii=False) + "\n")
            completed += 1
            print(f"    saved: {output_path}")
            if usage:
                print(f"    token usage: input={usage['prompt_tokens']}, output={usage['completion_tokens']}, total={usage['total_tokens']}")
            time.sleep(max(0.0, args.pause))
        except Exception as exc:
            failed += 1
            error_record = {
                "status": "error", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "clip_id": clip_id, "clip_path": str(video_path.resolve()),
                "deployment": args.deployment, "error_type": type(exc).__name__, "error": str(exc),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(error_record, ensure_ascii=False, indent=2), encoding="utf-8")
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("\n--- Run summary ---")
    print(f"Planned inference calls: {planned}")
    print(f"New successful results:  {completed}")
    print(f"Skipped from cache:     {skipped}")
    print(f"Errors:                 {failed}")
    if args.dry_run:
        print("Dry run: no Azure requests were made.")
    print(f"Per-clip outputs: {args.results_dir.resolve()}")
    print(f"Append-only summary: {summary_path.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
