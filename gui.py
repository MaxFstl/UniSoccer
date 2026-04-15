#!/usr/bin/env python3
"""Simple Tkinter GUI wrapper for the UniSoccer highlight pipeline.
This file is intentionally separate from `auto_highlights_pipeline.py` so the pipeline
stays clean. It loads the detection model on demand and can process a single clip
selected from a directory of .mp4 files.

Usage:
    python3 gui_highlights.py

Requirements: tkinter, torch, xdg-open (on Linux)
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import Counter
import os
import threading
import subprocess
import gc
from pathlib import Path
import torch
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    import ttkbootstrap as tb
    from ttkbootstrap import style
    from ttkbootstrap import Style
    TB_AVAILABLE = True
except Exception:
    import tkinter.ttk as ttk
    TB_AVAILABLE = False



# Import helpers from the pipeline module
from auto_highlights_pipeline import (
    load_detection_model,
    process_single_clip,
    generate_commentary_for_highlights,
)
from join_clips import join_video_clips
from dataset.video_utils_siglip import set_transform

# Simple in-memory model cache to avoid reloading repeatedly
# cache also stores the device the model currently resides on
_model_cache = {"model": None, "checkpoint": None, "device": None}

# Lock to serialize model loading so multiple threads don't load heavy model concurrently
_model_load_lock = threading.Lock()


def _set_status(label_widget, text: str):
    label_widget.config(text=text)


def _gui_load_model(checkpoint_path: str, device: torch.device, status_cb=None):
    """Load (and cache) the detection model for use by the GUI."""
    # reuse cached model if same checkpoint and already on requested device
    if _model_cache.get("model") is not None and _model_cache.get("checkpoint") == checkpoint_path:
        cached_device = _model_cache.get("device")
        if cached_device == device:
            if status_cb:
                status_cb("Model already loaded")
            return _model_cache["model"]

    # Prefer to load directly on the requested device. Only fall back to
    # loading on CPU if a meta-tensor RuntimeError occurs.
    if status_cb:
        status_cb(f"Loading model on {device}...")

    try:
        # Attempt to let the pipeline load directly to the requested device
        model = load_detection_model(checkpoint_path, device)
    except RuntimeError as e:
        # If a meta-tensor error occurs, fall back to loading on CPU and
        # then move using to_empty() if available.
        if "meta" in str(e).lower():
            if status_cb:
                status_cb("Meta-tensor error; loading on CPU and using to_empty() fallback...")
            cpu_device = torch.device("cpu")
            model = load_detection_model(checkpoint_path, cpu_device)
            if device.type != "cpu":
                if hasattr(model, "to_empty"):
                    model = model.to_empty(device)
                else:
                    model = model.to(device)
        else:
            raise

    model = model.eval()
    _model_cache["model"] = model
    _model_cache["checkpoint"] = checkpoint_path
    _model_cache["device"] = device
    if status_cb:
        status_cb("Model loaded")
    return model


def _gui_worker_process(clip_paths: list, checkpoint: str, clips_output_dir: str, device: torch.device, transform, text_cb, done_cb, status_cb=None, results_list=None, extracted_list=None):
    """Background worker to process multiple clips sequentially to manage memory."""
    try:
        # Load model once per batch
        if status_cb:
            status_cb("Preparing model...")
        model = _gui_load_model(checkpoint, device, status_cb)

        # Process each clip sequentially
        for clip_idx, clip_path in enumerate(clip_paths, start=1):
            try:
                if status_cb:
                    status_cb(f"Classifying clip {clip_idx}/{len(clip_paths)}...")

                clip_name = Path(clip_path).name
                try:
                    clip_num = int(clip_name.split("_")[1].split(".")[0])
                except Exception:
                    clip_num = 1

                item = process_single_clip(Path(clip_path), clip_num, model, transform, device, Path(clips_output_dir))
                if item:
                    out = (
                        f"[{clip_idx}/{len(clip_paths)}] {clip_name}\n"
                        f"Event: {item['event']} ({item['confidence']:.2%})\n"
                        f"Time: {item['start_time_formatted']} - {item['end_time_formatted']}\n"
                        f"Saved: {Path(item['clip_path']).name}\n"
                    )
                    # store the item and extracted path for later operations (commentary / join)
                    if results_list is not None:
                        try:
                            results_list.append(item)
                        except Exception:
                            pass
                    if extracted_list is not None:
                        try:
                            extracted_list.append(item.get("clip_path"))
                        except Exception:
                            pass
                else:
                    out = f"[{clip_idx}/{len(clip_paths)}] {clip_name} -> No highlight detected"
                text_cb(out)

                # Aggressive memory cleanup after each clip
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as clip_ex:
                text_cb(f"[{clip_idx}/{len(clip_paths)}] {clip_name} -> Error: {clip_ex}")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

    except Exception as ex:
        text_cb(f"Fatal error: {ex}")
    finally:
        # Final cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        done_cb()


def run_gui():
    root = tk.Tk()
    if TB_AVAILABLE:
        style = Style(theme="flatly")
    else:
        import tkinter.ttk as ttk
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
    root.option_add("*Font", ("Segoe UI", 10))
    root.title("UniSoccer - GUI")
    root.geometry("1200x800")
    # Top: directory selection
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=8, pady=6)

    dir_label = tk.Label(top_frame, text="Clips directory:")
    dir_label.pack(side=tk.LEFT)

    dir_var = tk.StringVar()
    dir_entry = tk.Entry(top_frame, textvariable=dir_var, width=60)
    dir_entry.pack(side=tk.LEFT, padx=6)

    def choose_dir():
        d = filedialog.askdirectory(title="Select clips directory")
        if d:
            dir_var.set(d)
            populate_listbox(d)

    browse_btn = tk.Button(top_frame, text="Browse...", command=choose_dir)
    browse_btn.pack(side=tk.LEFT)

    # Middle: listbox and controls
    mid_frame = tk.Frame(root)
    mid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    # Header above listbox with two columns: select-all checkbox and Name
    header_frame = tk.Frame(mid_frame)
    header_frame.pack(fill=tk.X, padx=2, pady=(0,4))

    select_all_var = tk.IntVar(value=0)
    def _on_select_all_toggled():
        if select_all_var.get():
            listbox.select_set(0, tk.END)
        else:
            listbox.select_clear(0, tk.END)

    select_all_cb = tk.Checkbutton(header_frame, variable=select_all_var, command=_on_select_all_toggled)
    select_all_cb.pack(side=tk.LEFT, padx=(4,8))

    name_label = tk.Label(header_frame, text="Name", anchor="w")
    name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # listbox now shows only file names and allows multiple selection
    listbox = tk.Listbox(mid_frame, selectmode=tk.EXTENDED, width=60)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(mid_frame, orient=tk.VERTICAL)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    control_frame = tk.Frame(mid_frame)
    control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8)

    # Extra options: generate commentary and join extracted clips
    commentary_var = tk.IntVar(value=0)
    join_var = tk.IntVar(value=0)
    commentary_cb = tk.Checkbutton(control_frame, text="Generate commentary", variable=commentary_var)
    commentary_cb.pack(pady=(6,2))
    join_cb = tk.Checkbutton(control_frame, text="Join extracted clips", variable=join_var)
    join_cb.pack(pady=(0,6))

    # Memory management: limit max clips to process at once
    tk.Label(control_frame, text="Max clips:", font=("", 9)).pack(pady=(6,0))
    max_clips_var = tk.IntVar(value=0)  # 0 = unlimited
    max_clips_spinbox = tk.Spinbox(control_frame, from_=0, to=100, textvariable=max_clips_var, width=5)
    max_clips_spinbox.pack(pady=(0,6))

    status_label = tk.Label(root, text="Ready.")
    status_label.pack(fill=tk.X, padx=8, pady=(0,6))

    output_text = tk.Text(root, height=8)
    output_text.pack(fill=tk.BOTH, padx=8, pady=(0,8), expand=False)

    # Checkpoint and output
    checkpoint_var = tk.StringVar(value="pretrained_classification.pth")
    ckpt_entry = tk.Entry(control_frame, textvariable=checkpoint_var, width=30)
    ckpt_entry.pack(pady=(0,6))

    def choose_checkpoint():
        f = filedialog.askopenfilename(title="Select checkpoint file", filetypes=[("PyTorch", "*.pth;*.pt"), ("All files", "*")])
        if f:
            checkpoint_var.set(f)

    ckpt_btn = tk.Button(control_frame, text="Choose checkpoint", command=choose_checkpoint)
    ckpt_btn.pack(pady=(0,6))

    output_dir_default = os.path.join(os.getcwd(), "outputs", "gui_clips")
    output_var = tk.StringVar(value=output_dir_default)
    out_entry = tk.Entry(control_frame, textvariable=output_var, width=30)
    out_entry.pack(pady=(0,6))

    def choose_output_dir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            output_var.set(d)
    def _start_detection():
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Please select one or more clips from the list.")
            return
        checkpoint = checkpoint_var.get()
        clips_output_dir = output_var.get()
        Path(clips_output_dir).mkdir(parents=True, exist_ok=True)

        # disable controls until all jobs finish
        detect_btn.config(state=tk.DISABLED)
        ckpt_btn.config(state=tk.DISABLED)
        out_btn.config(state=tk.DISABLED)
        select_all_cb.config(state=tk.DISABLED)
        output_text.delete(1.0, tk.END)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = set_transform()

        # clear previous results
        gui_highlights.clear()
        gui_extracted_paths.clear()

        # Load model once before spawning worker threads to avoid multiple heavy loads
        # Use the _model_load_lock to prevent races if multiple start calls happen
        with _model_load_lock:
            try:
                _append_output(f"Loading model for detection on {device}...")
                _gui_load_model(checkpoint, device, status_cb=_append_output)
            except Exception as e:
                messagebox.showerror("Model load failed", str(e))
                # re-enable controls
                detect_btn.config(state=tk.NORMAL)
                ckpt_btn.config(state=tk.NORMAL)
                out_btn.config(state=tk.NORMAL)
                select_all_cb.config(state=tk.NORMAL)
                return

        # Get selected clip paths, respecting max_clips limit
        max_clips_limit = max_clips_var.get()
        selected_clips = [clips_paths[idx] for idx in sel]
        if max_clips_limit > 0:
            selected_clips = selected_clips[:max_clips_limit]
        
        _append_output(f"Processing {len(selected_clips)} clips...\n")

        # Start a single worker thread that processes all clips sequentially
        with pending_lock:
            pending_jobs["count"] = 1
        thread = threading.Thread(
            target=_gui_worker_process,
            args=(selected_clips, checkpoint, clips_output_dir, device, transform, _append_output, _job_done, _append_output, gui_highlights, gui_extracted_paths),
            daemon=True,
        )
        thread.start()
    out_btn = tk.Button(control_frame, text="Output dir", command=choose_output_dir)
    out_btn.pack(pady=(0,6))

    detect_btn = tk.Button(control_frame, text="Detect Selected Clip", command=_start_detection)
    detect_btn.pack(pady=(6,6))

    clear_btn = tk.Button(control_frame, text="Clear Output", command=lambda: output_text.delete(1.0, tk.END))
    clear_btn.pack(pady=(6,6))

    # Keep list of full paths mapped to listbox indices
    clips_paths: list[str] = []
    # Collect results from workers
    gui_highlights: list[dict] = []
    gui_extracted_paths: list[str] = []

    # Helpers
    def populate_listbox(directory: str):
        listbox.delete(0, tk.END)
        clips_paths.clear()
        select_all_var.set(0)
        p = Path(directory)
        if not p.exists():
            return
        clips = sorted(p.glob("*.mp4"))
        for c in clips:
            clips_paths.append(str(c))
            listbox.insert(tk.END, c.name)

    def _schedule_status_update(text: str):
        # schedule the actual label update on the main thread
        root.after(0, lambda: _apply_status_update(text))

    def _apply_status_update(text: str):
        # call the module-level helper to update the label widget
        _set_status(status_label, text)

    def _append_output(text: str):
        root.after(0, lambda: output_text.insert(tk.END, text + "\n"))
    def _generate_classification_stats(highlights_list: list[dict], output_dir: Path):
    """Generate matplotlib statistics on classification results."""
    if not highlights_list:
        return
    
    # Extract data
    events = [h['event'] for h in highlights_list]
    confidences = [h['confidence'] for h in highlights_list]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("UniSoccer Classification Statistics", fontsize=14, fontweight="bold")
    
    # 1) Event distribution (histogram)
    event_counts = Counter(events)
    axes[0, 0].bar(event_counts.keys(), event_counts.values(), color='steelblue')
    axes[0, 0].set_title("Events Detected")
    axes[0, 0].set_xlabel("Event Type")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2) Confidence distribution (histogram)
    axes[0, 1].hist(confidences, bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_title("Confidence Score Distribution")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', label=f"Mean: {np.mean(confidences):.2f}")
    axes[0, 1].legend()
    
    # 3) Confidence by event type (box plot)
    event_conf_dict = {}
    for h in highlights_list:
        event = h['event']
        if event not in event_conf_dict:
            event_conf_dict[event] = []
        event_conf_dict[event].append(h['confidence'])
    
    axes[1, 0].boxplot(event_conf_dict.values(), tick_labels=event_conf_dict.keys())
    axes[1, 0].set_title("Confidence by Event Type")
    axes[1, 0].set_ylabel("Confidence")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4) Summary statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
    Total Highlights Detected: {len(highlights_list)}
    
    Average Confidence: {np.mean(confidences):.2%}
    Min Confidence: {np.min(confidences):.2%}
    Max Confidence: {np.max(confidences):.2%}
    Std Dev: {np.std(confidences):.4f}
    
    Event Types: {len(event_counts)}
    Most Common: {event_counts.most_common(1)[0][0]} ({event_counts.most_common(1)[0][1]} times)
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    stats_path = output_dir / "classification_stats.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    _append_output(f"Saved stats: {stats_path}")
    
    # Optionally display in a new window
    plt.show()
    # manage multiple background jobs and notify when all done
    pending_lock = threading.Lock()
    pending_jobs = {"count": 0}

    def _job_done():
        with pending_lock:
            pending_jobs["count"] -= 1
            if pending_jobs["count"] <= 0:
                # all finished
                root.after(0, _on_all_done)

    def _on_all_done():
        detect_btn.config(state=tk.NORMAL)
        ckpt_btn.config(state=tk.NORMAL)
        out_btn.config(state=tk.NORMAL)
        select_all_cb.config(state=tk.NORMAL)
        _apply_status_update("Ready.")
        # After all workers finished, optionally generate commentary or join clips
        try:
            out_dir = Path(output_var.get())
            clips_out_dir = out_dir  # commentary expects highlights['clip_path'] absolute path; using output dir
            if gui_highlights:
                _apply_status_update("Generating classification stats...")
                try:
                    _generate_classification_stats(gui_highlights, clips_out_dir)
                    _append_output("Stats generation finished.")
                except Exception as e:
                    _append_output(f"Stats error: {e}")

            if commentary_var.get() and gui_highlights:
                _apply_status_update("Generating commentary...")
                try:
                    generate_commentary_for_highlights(gui_highlights, clips_out_dir)
                    _append_output("Commentary generation finished.")
                except Exception as e:
                    _append_output(f"Commentary error: {e}")
                _apply_status_update("Ready.")

            if join_var.get() and gui_extracted_paths:
                _apply_status_update("Joining extracted clips...")
                try:
                    compilation = clips_out_dir / "highlights_compilation_gui.mp4"
                    join_video_clips(gui_extracted_paths, str(compilation))
                    _append_output(f"Saved compilation: {compilation}")
                except Exception as e:
                    _append_output(f"Join error: {e}")
                _apply_status_update("Ready.")
        except Exception:
            pass


    root.mainloop()


if __name__ == "__main__":
    run_gui()