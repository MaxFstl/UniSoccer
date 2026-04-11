#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

from utils.highlight_timing import resolve_event_interval, sanitize_event_name


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def cut_single_clip(
    source_video: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        source_video,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def compact_item(event: dict) -> dict:
    return {
        "event": event["event"],
        "confidence": round(event["confidence"], 4),
        "start_time_seconds": round(event["start_time_seconds"], 3),
        "end_time_seconds": round(event["end_time_seconds"], 3),
        "peak_time_seconds": round(event["peak_time_seconds"], 3),
        "clip_file": event["clip_file"],
        "clip_path": event["clip_path"],
    }


def cut_highlights(
    source_video: str,
    highlights: list,
    output_dir: str,
) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    total_duration = get_video_duration(source_video)

    extracted = []
    for i, event in enumerate(highlights, start=1):
        start_time, end_time = resolve_event_interval(event, min_time=0.0, max_time=total_duration)
        duration = max(1.0, end_time - start_time)

        event_name = event.get("primary_event", event.get("event", event.get("type", "event")))
        safe_name = sanitize_event_name(event_name)
        clip_name = f"clip_{i:03d}_{safe_name}.mp4"
        clip_path = output_path / clip_name

        cut_single_clip(source_video, str(clip_path), start_time, duration)

        event["extracted_file"] = clip_name
        event["extracted_path"] = str(clip_path)
        event["extracted_start"] = round(start_time, 3)
        event["extracted_end"] = round(end_time, 3)
        extracted.append(str(clip_path))

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut event clips from full video")
    parser.add_argument("--source_video", required=True, type=str)
    parser.add_argument("--highlights_json", required=True, type=str)
    parser.add_argument("--output_dir", default="outputs/cut_event_clips", type=str)
    args = parser.parse_args()

    with open(args.highlights_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    highlights = data.get("highlights", data if isinstance(data, list) else [])

    extracted = cut_highlights(
        source_video=args.source_video,
        highlights=highlights,
        output_dir=args.output_dir,
    )

    data = {
        "source_video": args.source_video,
        "total_highlights": len(extracted),
        "highlights": [compact_item(item) for item in highlights],
    }
    updated_json = Path(args.output_dir) / "highlights_with_clips.json"
    with open(updated_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Extracted clips: {len(extracted)}")
    print(f"Saved metadata: {updated_json}")


if __name__ == "__main__":
    main()
