#!/usr/bin/env python3

import argparse
from contextlib import nullcontext
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from einops import rearrange

from cut_event_clips import cut_single_clip
from dataset.video_utils_siglip import set_transform
from join_clips import join_video_clips
from model.MatchVision_classifier import MatchVision_Classifier
from utils.highlight_timing import get_window_bounds, sanitize_event_name


KEYWORDS = [
    "var",
    "end of half game",
    "clearance",
    "second yellow card",
    "injury",
    "ball possession",
    "throw in",
    "show added time",
    "shot off target",
    "start of half game",
    "substitution",
    "saved by goal-keeper",
    "red card",
    "lead to corner",
    "ball out of play",
    "off side",
    "goal",
    "penalty",
    "yellow card",
    "foul lead to penalty",
    "corner",
    "free kick",
    "foul with no card",
]

HIGHLIGHT_EVENTS = {
    "goal",
    "penalty",
    "red card",
    "second yellow card",
    "yellow card",
    "saved by goal-keeper",
    "shot off target",
    "corner",
    "free kick",
    "var",
    "injury",
    "substitution",
    "foul lead to penalty",
}

CLIP_SECONDS = 30.0
NUM_CLASSIFICATION_FRAMES = 30  # Sample frames uniformly across clip for classification
REFINE_WINDOW_SECONDS = 6.0     # Time window size when searching for peak confidence
REFINE_STRIDE_SECONDS = 2.0     # Step size when sliding refinement search
PEAK_TIE_THRESHOLD = 0.01       # Confidence threshold to consider scores tied
VIDEO_READER_THREADS = 4
FIXED_THRESHOLD = 0.30

EVENT_MIN_CONFIDENCE = {
    "goal": 0.30,
    "penalty": 0.32,
    "foul lead to penalty": 0.34,
    "red card": 0.34,
    "second yellow card": 0.34,
    "yellow card": 0.32,
    "saved by goal-keeper": 0.32,
    "shot off target": 0.34,
    "corner": 0.33,
    "free kick": 0.33,
    "var": 0.42,
    "injury": 0.45,
    "substitution": 0.48,
}

EVENT_MIN_MARGIN = {
    "substitution": 0.14,
    "injury": 0.12,
    "var": 0.12,
    "corner": 0.10,
    "free kick": 0.10,
}
DEFAULT_MIN_MARGIN = 0.07
COMMENTARY_CHECKPOINT = "downstream_commentary_all_open.pth"
COMMENTARY_WORD_WORLD = "./words_world/merge.pkl"
COMMENTARY_BATCH_SIZE = 2
COMMENTARY_NUM_WORKERS = 2


def natural_clip_key(path: Path) -> tuple:
    numbers = re.findall(r"\d+", path.stem)
    if numbers:
        return (0, *(int(n) for n in numbers), path.stem)
    return (1, path.stem)


def format_time(seconds: float) -> str:
    total_seconds = int(max(0, round(seconds)))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def preprocess_window_frames(video_reader: VideoReader, transform, start_sec: float, end_sec: float, num_frames: int = NUM_CLASSIFICATION_FRAMES) -> torch.Tensor:
    """Sample frames uniformly across time window, transform, and reshape for model input."""
    fps = float(video_reader.get_avg_fps())
    vlen = len(video_reader)
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(vlen - 1, int(end_sec * fps))
    
    # Sample frame indices uniformly across the window (or repeat if window is tiny)
    if end_idx <= start_idx:
        frame_indices = [start_idx] * num_frames
    else:
        frame_indices = torch.linspace(start_idx, end_idx, steps=num_frames).round().long().tolist()
    
    # Load frames (Decord returns an NDArray-like object). Ensure we have a NumPy array of shape (N, H, W, C)
    frames_batch = video_reader.get_batch(frame_indices)
    if hasattr(frames_batch, "asnumpy"):
        frames_np = frames_batch.asnumpy()
    elif isinstance(frames_batch, torch.Tensor):
        # convert torch tensor in (N, C, H, W) or (N, H, W, C) to (N, H, W, C)
        if frames_batch.ndim == 4 and frames_batch.shape[1] in (1, 3):
            frames_np = frames_batch.permute(0, 2, 3, 1).cpu().numpy()
        else:
            frames_np = frames_batch.cpu().numpy()
    else:
        # fallback: try treating it as array-like
        frames_np = np.asarray(frames_batch)

    # Apply image transform to each frame, then batch them
    pixel_values_list = []
    for frame in frames_np:
        pv = transform(images=frame, return_tensors="pt")["pixel_values"]
        # pv has shape (1, C, H, W)
        pixel_values_list.append(pv)
    frames = torch.cat(pixel_values_list, dim=0)
    
    # Reshape to temporal format (C, T, H, W) and add batch dimension: (1, C, T, H, W)
    return rearrange(frames, "t c h w -> c t h w").unsqueeze(0)


def classify_window(model: MatchVision_Classifier, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Classify a window of frames, return per-event confidence scores."""
    with torch.no_grad():
        frames = frames.to(device, non_blocking=True)
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model.get_logits(frames)
        return torch.softmax(logits, dim=1)[0].detach().cpu()


def generate_search_windows(max_start: float, stride: float) -> list:
    """Generate time windows for peak refinement search."""
    windows = [0.0]
    if max_start > 0:
        num_steps = int(max_start // stride)
        windows = [i * stride for i in range(num_steps + 1)]
        if windows[-1] < max_start:
            windows.append(max_start)
    return windows


def refine_event_peak(
    video_reader: VideoReader,
    event_index: int,
    model: MatchVision_Classifier,
    transform,
    device: torch.device,
) -> tuple[float, float]:
    """Search for the peak confidence moment of an event within the clip."""
    duration = len(video_reader) / float(video_reader.get_avg_fps())
    window = min(REFINE_WINDOW_SECONDS, duration)
    max_start = max(0.0, duration - window)
    
    starts = generate_search_windows(max_start, REFINE_STRIDE_SECONDS)

    best_score = -1.0
    best_start = 0.0
    best_mid = duration / 2.0
    
    for local_start in starts:
        local_end = min(duration, local_start + window)
        window_frames = preprocess_window_frames(video_reader, transform, local_start, local_end)
        probs = classify_window(model, window_frames, device)
        event_score = probs[event_index].item()
        
        # Prefer higher confidence, or break ties by choosing earlier moment (avoid replays)
        is_better = event_score > best_score
        is_tie_with_earlier = abs(event_score - best_score) <= PEAK_TIE_THRESHOLD and local_start < best_start
        if is_better or is_tie_with_earlier:
            best_score = event_score
            best_start = local_start
            best_mid = (local_start + local_end) / 2.0
        del window_frames, probs

    return best_mid, best_score


def pick_highlight_event(probs: torch.Tensor) -> tuple[int | None, str | None, float]:
    """Find the highest-confidence event that meets all thresholds."""
    top_probs, top_indices = torch.topk(probs, k=5, dim=0)
    
    for rank in range(5):
        idx_in_keywords = top_indices[rank].item()
        event_name = KEYWORDS[idx_in_keywords]
        confidence = top_probs[rank].item()
        
        # Must be a highlight event type
        if event_name not in HIGHLIGHT_EVENTS:
            continue

        # Must meet event-specific confidence threshold
        min_conf = max(FIXED_THRESHOLD, EVENT_MIN_CONFIDENCE.get(event_name, FIXED_THRESHOLD))
        if confidence < min_conf:
            continue

        # Must have clear margin over other predictions (avoid ambiguity)
        other_scores = [probs[i].item() for i in range(len(KEYWORDS)) if i != idx_in_keywords]
        margin = confidence - max(other_scores)
        min_margin = EVENT_MIN_MARGIN.get(event_name, DEFAULT_MIN_MARGIN)
        if margin >= min_margin:
            return idx_in_keywords, event_name, confidence
            
    return None, None, 0.0


def load_detection_model(checkpoint: str, device: torch.device):
    """Load and initialize the MatchVision classifier."""
    model = MatchVision_Classifier(
        keywords=KEYWORDS,
        classifier_transformer_type="avg_pool",
        vision_encoder_type="spatial_and_temporal",
        use_transformer=True,
    ).to(device).eval()

    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint_data["state_dict"].items()}
    model.load_state_dict(state_dict)
    del checkpoint_data, state_dict
    gc.collect()
    
    return model


def process_single_clip(
    clip_path: Path,
    clip_num: int,
    model: MatchVision_Classifier,
    transform,
    device: torch.device,
    clips_output_dir: Path,
) -> dict | None:
    """Detect highlight in one clip, extract, and return metadata."""
    video_reader = VideoReader(str(clip_path), num_threads=VIDEO_READER_THREADS)
    
    global_start = (clip_num - 1) * CLIP_SECONDS
    clip_duration = len(video_reader) / float(video_reader.get_avg_fps())
    global_end = global_start + clip_duration

    # Classify the entire clip
    frames = preprocess_window_frames(video_reader, transform, 0.0, clip_duration, num_frames=NUM_CLASSIFICATION_FRAMES)
    probs = classify_window(model, frames, device)
    event_index, event_label, event_confidence = pick_highlight_event(probs)
    
    if event_label is None:
        del frames, probs
        gc.collect()
        return None

    # Refine peak timing by searching within the clip
    peak_local, refined_confidence = refine_event_peak(
        video_reader=video_reader,
        event_index=event_index,
        model=model,
        transform=transform,
        device=device,
    )
    peak_global = global_start + peak_local
    event_confidence = max(event_confidence, refined_confidence)

    # Apply event-specific before/after window
    extracted_start, extracted_end = get_window_bounds(peak_global, global_start, global_end, event_label)
    local_start = extracted_start - global_start
    local_duration = max(1.0, extracted_end - extracted_start)

    # Cut and save the highlight clip
    safe_event = sanitize_event_name(event_label)
    extracted_name = f"clip_{len(list(clips_output_dir.glob('*.mp4'))) + 1:03d}_{safe_event}.mp4"
    extracted_path = clips_output_dir / extracted_name
    cut_single_clip(str(clip_path), str(extracted_path), local_start, local_duration)

    # Build metadata item
    item = {
        "event": event_label,
        "confidence": round(event_confidence, 4),
        "start_time_seconds": round(extracted_start, 3),
        "end_time_seconds": round(extracted_end, 3),
        "peak_time_seconds": round(peak_global, 3),
        "start_time_formatted": format_time(extracted_start),
        "end_time_formatted": format_time(extracted_end),
        "peak_time_formatted": format_time(peak_global),
        "clip_file": extracted_name,
        "clip_path": str(extracted_path),
    }

    del frames, probs, video_reader
    gc.collect()
    
    return item


def detect_highlights(
    video_dir: str,
    checkpoint: str,
    max_clips: int,
    clips_output_dir: Path,
) -> tuple[list, list]:
    """Detect highlights across all clips in directory."""
    clips = sorted(Path(video_dir).glob("*.mp4"), key=natural_clip_key)
    if max_clips > 0:
        clips = clips[:max_clips]
    if not clips:
        return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_detection_model(checkpoint, device)
    transform = set_transform()
    highlights = []
    extracted = []
    clips_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, clip_path in enumerate(clips, start=1):
        clip_name = clip_path.name
        try:
            clip_num = int(clip_name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            clip_num = idx

        item = process_single_clip(clip_path, clip_num, model, transform, device, clips_output_dir)
        
        if item:
            highlights.append(item)
            extracted.append(item["clip_path"])
            print(f"[{idx}/{len(clips)}] {clip_name} -> {item['event']} ({item['confidence']:.1%}) {item['start_time_formatted']} - {item['end_time_formatted']}")
        else:
            print(f"[{idx}/{len(clips)}] {clip_name} -> no highlight")

    highlights.sort(key=lambda item: item["start_time_seconds"])
    return highlights, extracted


def generate_commentary_for_highlights(highlights: list, output_dir: Path) -> None:
    if not highlights:
        return
    if not torch.cuda.is_available():
        raise ValueError("Commentary generation requires CUDA.")

    temp_json = output_dir / "commentary_input.json"
    input_items = []
    for item in highlights:
        input_items.append(
            {
                "video": str(Path(item["clip_path"]).resolve()),
                "comments_text_anonymized": "",
            }
        )
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(input_items, f, ensure_ascii=False)

    try:
        from torch.utils.data import DataLoader
        from dataset.MatchVision_commentary_new_benchmark_from_npy import MatchVisionCommentary_new_benchmark_from_npy_Dataset
        from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks

        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=str(temp_json),
            video_base_dir="",
        )
        data_loader = DataLoader(
            dataset,
            batch_size=COMMENTARY_BATCH_SIZE,
            num_workers=COMMENTARY_NUM_WORKERS,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=dataset.collater,
        )

        model = matchvoice_model_all_blocks(
            num_features=768,
            need_temporal="yes",
            open_visual_encoder=True,
            open_llm_decoder=True,
            inference=True,
            file_path=COMMENTARY_WORD_WORLD,
        )
        checkpoint_data = torch.load(COMMENTARY_CHECKPOINT, map_location="cpu")
        raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)
        state_dict = {k.replace("module.", "", 1): v for k, v in raw_state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        device = torch.device("cuda:0")
        model = model.to(device).eval()

        comments = []
        with torch.no_grad():
            for samples in data_loader:
                samples["frames"] = samples["frames"].to(device, non_blocking=True)
                comments.extend(model(samples))

        for item, commentary in zip(highlights, comments):
            item["commentary"] = commentary.strip()
    finally:
        if temp_json.exists():
            temp_json.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and cut soccer highlights from sliced clips")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with clip_XXX.mp4 files from slice_video.py")
    parser.add_argument("--checkpoint", type=str, default="pretrained_classification.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/auto_highlights")
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--join", action="store_true")
    parser.add_argument("--commentary", action="store_true")
    return parser


def write_metadata(metadata: dict, output_json: Path) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    highlights, extracted = detect_highlights(
        video_dir=args.video_dir,
        checkpoint=args.checkpoint,
        max_clips=args.max_clips,
        clips_output_dir=clips_dir,
    )

    detected_json = output_dir / "detected_highlights.json"
    print(f"Cut clips: {len(extracted)}")

    if args.commentary and highlights:
        generate_commentary_for_highlights(highlights, output_dir)
        print(f"Generated commentary: {len(highlights)}")

    if args.join and extracted:
        compilation = output_dir / "highlights_compilation.mp4"
        join_video_clips(extracted, str(compilation))
        print(f"Saved compilation: {compilation}")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "video_dir": args.video_dir,
        "total_highlights": len(highlights),
        "highlights": highlights,
    }
    write_metadata(metadata, detected_json)
    print(f"Saved detections: {detected_json}")
    print(f"Total highlights: {len(highlights)}")


if __name__ == "__main__":
    main()

