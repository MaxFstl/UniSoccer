#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def collect_from_json(highlights_json: str) -> list:
    with open(highlights_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    clips = []
    for item in data.get("highlights", []):
        if "extracted_path" in item:
            clips.append(item["extracted_path"])
        elif "clip_path" in item:
            clips.append(item["clip_path"])
    return clips


def collect_from_dir(clips_dir: str) -> list:
    folder = Path(clips_dir)
    return [str(p) for p in sorted(folder.glob("*.mp4"))]


def join_video_clips(clips: list, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    concat_file = output_file.parent / "concat_list.txt"
    try:
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clips:
                f.write(f"file '{Path(clip).resolve()}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            str(output_file),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        if concat_file.exists():
            concat_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join highlight clips into one video")
    parser.add_argument("--highlights_json", type=str, default=None)
    parser.add_argument("--clips_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/highlights_compilation.mp4")
    args = parser.parse_args()

    clips = []
    if args.highlights_json:
        clips = collect_from_json(args.highlights_json)
    if not clips and args.clips_dir:
        clips = collect_from_dir(args.clips_dir)
    if not clips:
        raise ValueError("No clips found. Pass --highlights_json or --clips_dir.")

    join_video_clips(clips, args.output)
    print(f"Joined clips: {len(clips)}")
    print(f"Output video: {args.output}")


if __name__ == "__main__":
    main()
