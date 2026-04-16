#!/usr/bin/env python3
"""
Simple Tkinter GUI wrapper for the UniSoccer highlight detection pipeline.

This module provides an interactive GUI for processing video clips and detecting highlights.
It is intentionally separate from `auto_highlights_pipeline.py` to keep the pipeline clean.
The GUI loads the detection model on demand and allows batch processing of .mp4 files.

Usage:
    python3 gui.py

Requirements: 
    - tkinter (Tcl/Tk): System package (tk on Arch Linux)
    - torch: PyTorch deep learning framework
    - matplotlib: Data visualization
    - ttkbootstrap (optional): Enhanced Tkinter theming
    - xdg-open (optional): Linux file manager integration
"""

import gc
import os
import subprocess
import threading
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox

# Attempt to import ttkbootstrap for enhanced theme support
try:
    from ttkbootstrap import Style
    TB_AVAILABLE = True
except ImportError:
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

# ============================================================================
# MODULE-LEVEL STATE AND LOCKS
# ============================================================================

# In-memory model cache to avoid reloading the same checkpoint repeatedly.
# Stores: {"model": model_instance, "checkpoint": checkpoint_path, "device": device}
_model_cache = {"model": None, "checkpoint": None, "device": None}

# Lock to serialize model loading and prevent race conditions in multi-threaded scenarios
_model_load_lock = threading.Lock()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _set_status(label_widget: tk.Label, text: str) -> None:
    """
    Update the status label text.
    
    Args:
        label_widget: The Tkinter Label widget to update
        text: The status message to display
    """
    label_widget.config(text=text)


def _gui_load_model(checkpoint_path: str, device: torch.device, status_cb=None):
    """
    Load (and cache) the detection model for use by the GUI.
    
    Attempts to load the model directly on the requested device. If a meta-tensor 
    error occurs, falls back to loading on CPU and then moving to the target device.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        device: Target device (cuda or cpu)
        status_cb: Optional callback function to report loading status
        
    Returns:
        The loaded model in evaluation mode
    """
    # Reuse cached model if same checkpoint and already on requested device
    if (_model_cache.get("model") is not None and 
        _model_cache.get("checkpoint") == checkpoint_path and 
        _model_cache.get("device") == device):
        if status_cb:
            status_cb("Model already loaded")
        return _model_cache["model"]

    if status_cb:
        status_cb(f"Loading model on {device}...")

    try:
        # Attempt to load directly on the requested device
        model = load_detection_model(checkpoint_path, device)
    except RuntimeError as e:
        # Handle meta-tensor errors by loading on CPU first
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
        status_cb("Model loaded successfully")
    return model


def _gui_worker_process(clip_paths: list, checkpoint: str, clips_output_dir: str, device: torch.device, 
                        transform, text_cb, done_cb, status_cb=None, results_list=None, extracted_list=None):
    """
    Background worker thread to process multiple clips sequentially.
    
    Processes clips one by one to manage memory effectively, updating progress via callbacks.
    
    Args:
        clip_paths: List of paths to video clips to process
        checkpoint: Path to the model checkpoint
        clips_output_dir: Directory to save extracted highlights
        device: Compute device (cuda or cpu)
        transform: Image transform pipeline
        text_cb: Callback function for text output updates
        done_cb: Callback function to signal completion
        status_cb: Optional callback for status updates
        results_list: Optional list to collect highlight results
        extracted_list: Optional list to collect extracted clip paths
    """
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
                
                # Extract clip number from filename
                try:
                    clip_num = int(clip_name.split("_")[1].split(".")[0])
                except (IndexError, ValueError):
                    clip_num = 1

                # Process the single clip
                item = process_single_clip(Path(clip_path), clip_num, model, transform, device, Path(clips_output_dir))
                
                if item:
                    out = (
                        f"[{clip_idx}/{len(clip_paths)}] {clip_name}\n"
                        f"Event: {item['event']} ({item['confidence']:.2%})\n"
                        f"Time: {item['start_time_formatted']} - {item['end_time_formatted']}\n"
                        f"Saved: {Path(item['clip_path']).name}\n"
                    )
                    # Store the item and extracted path for later operations (commentary / join)
                    if results_list is not None:
                        try:
                            results_list.append(item)
                        except Exception as e:
                            text_cb(f"Warning: Could not append result to list: {e}")
                    if extracted_list is not None:
                        try:
                            extracted_list.append(item.get("clip_path"))
                        except Exception as e:
                            text_cb(f"Warning: Could not append extracted path: {e}")
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


# ============================================================================
# GUI MAIN WINDOW
# ============================================================================

def run_gui():
    """
    Initialize and run the main GUI window.
    
    Displays a Tkinter interface with clip directory selection, processing controls,
    and real-time output display.
    """
    root = tk.Tk()
    
    # Apply theme
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
    root.title("UniSoccer - Highlight Detection GUI")
    root.geometry("1200x800")

    # ========================================================================
    # TOP FRAME: DIRECTORY SELECTION
    # ========================================================================
    
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=8, pady=6)

    dir_label = tk.Label(top_frame, text="Clips directory:")
    dir_label.pack(side=tk.LEFT)

    dir_var = tk.StringVar()
    dir_entry = tk.Entry(top_frame, textvariable=dir_var, width=60)
    dir_entry.pack(side=tk.LEFT, padx=6)

    def choose_dir():
        """Open directory chooser dialog."""
        d = filedialog.askdirectory(title="Select clips directory")
        if d:
            dir_var.set(d)
            populate_listbox(d)

    browse_btn = tk.Button(top_frame, text="Browse...", command=choose_dir)
    browse_btn.pack(side=tk.LEFT)

    # ========================================================================
    # MIDDLE FRAME: CLIP LISTBOX AND CONTROLS
    # ========================================================================
    
    mid_frame = tk.Frame(root)
    mid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    # Header: select-all checkbox and Name label
    header_frame = tk.Frame(mid_frame)
    header_frame.pack(fill=tk.X, padx=2, pady=(0, 4))

    select_all_var = tk.IntVar(value=0)
    
    def _on_select_all_toggled():
        """Toggle selection of all clips in the listbox."""
        if select_all_var.get():
            listbox.select_set(0, tk.END)
        else:
            listbox.select_clear(0, tk.END)

    select_all_cb = tk.Checkbutton(header_frame, variable=select_all_var, command=_on_select_all_toggled)
    select_all_cb.pack(side=tk.LEFT, padx=(4, 8))

    name_label = tk.Label(header_frame, text="Name", anchor="w")
    name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Listbox: displays .mp4 filenames with extended selection
    listbox = tk.Listbox(mid_frame, selectmode=tk.EXTENDED, width=60)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(mid_frame, orient=tk.VERTICAL)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    # Control frame: right sidebar with buttons and options
    control_frame = tk.Frame(mid_frame)
    control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8)

    # ========================================================================
    # CONTROL FRAME: OPTIONS AND BUTTONS
    # ========================================================================

    # Checkpoint selection
    tk.Label(control_frame, text="Checkpoint:", font=("", 9, "bold")).pack(pady=(6, 0))
    checkpoint_var = tk.StringVar(value="pretrained_classification.pth")
    ckpt_entry = tk.Entry(control_frame, textvariable=checkpoint_var, width=30)
    ckpt_entry.pack(pady=(0, 6))

    def choose_checkpoint():
        """Open checkpoint file chooser dialog."""
        f = filedialog.askopenfilename(
            title="Select checkpoint file",
            filetypes=[("PyTorch", "*.pth;*.pt"), ("All files", "*")]
        )
        if f:
            checkpoint_var.set(f)

    ckpt_btn = tk.Button(control_frame, text="Choose checkpoint", command=choose_checkpoint)
    ckpt_btn.pack(pady=(0, 6))

    # Output directory selection
    tk.Label(control_frame, text="Output dir:", font=("", 9, "bold")).pack(pady=(6, 0))
    output_dir_default = os.path.join(os.getcwd(), "outputs", "gui_clips")
    output_var = tk.StringVar(value=output_dir_default)
    out_entry = tk.Entry(control_frame, textvariable=output_var, width=30)
    out_entry.pack(pady=(0, 6))

    def choose_output_dir():
        """Open output directory chooser dialog."""
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            output_var.set(d)

    out_btn = tk.Button(control_frame, text="Output dir", command=choose_output_dir)
    out_btn.pack(pady=(0, 6))

    # Post-processing options
    tk.Label(control_frame, text="Post-processing:", font=("", 9, "bold")).pack(pady=(6, 0))
    commentary_var = tk.IntVar(value=0)
    commentary_cb = tk.Checkbutton(control_frame, text="Generate commentary", variable=commentary_var)
    commentary_cb.pack(pady=(2, 2), anchor="w")
    
    join_var = tk.IntVar(value=0)
    join_cb = tk.Checkbutton(control_frame, text="Join extracted clips", variable=join_var)
    join_cb.pack(pady=(0, 6), anchor="w")

    # Memory management: max clips per batch
    tk.Label(control_frame, text="Max clips:", font=("", 9, "bold")).pack(pady=(6, 0))
    max_clips_var = tk.IntVar(value=0)  # 0 = unlimited
    max_clips_spinbox = tk.Spinbox(control_frame, from_=0, to=100, textvariable=max_clips_var, width=5)
    max_clips_spinbox.pack(pady=(0, 6))

    # Detection button
    detect_btn = tk.Button(control_frame, text="Detect Selected Clips", command=lambda: _start_detection())
    detect_btn.pack(pady=(6, 6), fill=tk.X)

    # Clear output button
    clear_btn = tk.Button(control_frame, text="Clear Output", command=lambda: output_text.delete(1.0, tk.END))
    clear_btn.pack(pady=(0, 6), fill=tk.X)

    # ========================================================================
    # BOTTOM FRAME: STATUS AND OUTPUT
    # ========================================================================

    status_label = tk.Label(root, text="Ready.", relief=tk.SUNKEN)
    status_label.pack(fill=tk.X, padx=8, pady=(6, 0))

    output_text = tk.Text(root, height=8, bg="white", fg="black")
    output_text.pack(fill=tk.BOTH, padx=8, pady=(0, 8), expand=False)

    # ========================================================================
    # INTERNAL STATE AND CALLBACKS
    # ========================================================================

    # Keep list of full paths mapped to listbox indices
    clips_paths: list[str] = []
    
    # Collect results from workers
    gui_highlights: list[dict] = []
    gui_extracted_paths: list[str] = []

    # Job management for multi-threaded processing
    pending_lock = threading.Lock()
    pending_jobs = {"count": 0}

    # ========================================================================
    # INTERNAL HELPER FUNCTIONS
    # ========================================================================

    def populate_listbox(directory: str):
        """
        Populate the listbox with .mp4 files from the selected directory.
        
        Args:
            directory: Path to the directory containing video clips
        """
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

    def _apply_status_update(text: str):
        """
        Update the status label with the provided text.
        
        Args:
            text: Status message to display
        """
        _set_status(status_label, text)

    def _append_output(text: str):
        """
        Append text to the output display area (thread-safe).
        
        Args:
            text: Text to append
        """
        root.after(0, lambda: output_text.insert(tk.END, text + "\n"))

    def _generate_classification_stats(highlights_list: list[dict], output_dir: Path):
        """
        Generate matplotlib statistics visualization on classification results.
        
        Creates a 2x2 subplot figure with:
        - Event distribution histogram
        - Confidence score distribution
        - Confidence by event type box plot
        - Summary statistics table
        
        Args:
            highlights_list: List of highlight detection results
            output_dir: Directory to save the statistics image
        """
        if not highlights_list:
            return
        
        # Extract data
        events = [h['event'] for h in highlights_list]
        confidences = [h['confidence'] for h in highlights_list]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("UniSoccer Classification Statistics", fontsize=14, fontweight="bold")
        
        # 1) Event distribution (bar chart)
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
        mean_conf = np.mean(confidences)
        axes[0, 1].axvline(mean_conf, color='red', linestyle='--', label=f"Mean: {mean_conf:.2f}")
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
        most_common_event, most_common_count = event_counts.most_common(1)[0]
        stats_text = f"""
    Total Highlights Detected: {len(highlights_list)}
    
    Average Confidence: {np.mean(confidences):.2%}
    Min Confidence: {np.min(confidences):.2%}
    Max Confidence: {np.max(confidences):.2%}
    Std Dev: {np.std(confidences):.4f}
    
    Event Types: {len(event_counts)}
    Most Common: {most_common_event} ({most_common_count} times)
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        stats_path = output_dir / "classification_stats.png"
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        _append_output(f"Saved stats: {stats_path}")
        
        # Display in new window
        plt.show()

    def _job_done():
        """
        Decrement pending job counter and trigger completion callback when all jobs are done.
        """
        with pending_lock:
            pending_jobs["count"] -= 1
            if pending_jobs["count"] <= 0:
                root.after(0, _on_all_done)

    def _on_all_done():
        """
        Re-enable controls and perform post-processing operations after all worker threads finish.
        """
        # Re-enable controls
        detect_btn.config(state=tk.NORMAL)
        ckpt_btn.config(state=tk.NORMAL)
        out_btn.config(state=tk.NORMAL)
        select_all_cb.config(state=tk.NORMAL)
        _apply_status_update("Ready.")
        
        # Perform post-processing operations
        try:
            out_dir = Path(output_var.get())
            
            # Generate classification statistics
            if gui_highlights:
                _apply_status_update("Generating classification stats...")
                try:
                    _generate_classification_stats(gui_highlights, out_dir)
                    _append_output("Stats generation finished.")
                except Exception as e:
                    _append_output(f"Stats error: {e}")

            # Generate commentary
            if commentary_var.get() and gui_highlights:
                _apply_status_update("Generating commentary...")
                try:
                    generate_commentary_for_highlights(gui_highlights, out_dir)
                    _append_output("Commentary generation finished.")
                except Exception as e:
                    _append_output(f"Commentary error: {e}")

            # Join extracted clips into compilation
            if join_var.get() and gui_extracted_paths:
                _apply_status_update("Joining extracted clips...")
                try:
                    compilation = out_dir / "highlights_compilation_gui.mp4"
                    join_video_clips(gui_extracted_paths, str(compilation))
                    _append_output(f"Saved compilation: {compilation}")
                except Exception as e:
                    _append_output(f"Join error: {e}")
            
            _apply_status_update("Ready.")
        except Exception as e:
            _append_output(f"Post-processing error: {e}")

    def _start_detection():
        """
        Initialize detection process: validate inputs, disable controls, and spawn worker thread.
        """
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Please select one or more clips from the list.")
            return
        
        checkpoint = checkpoint_var.get()
        clips_output_dir = output_var.get()
        Path(clips_output_dir).mkdir(parents=True, exist_ok=True)

        # Disable controls during processing
        detect_btn.config(state=tk.DISABLED)
        ckpt_btn.config(state=tk.DISABLED)
        out_btn.config(state=tk.DISABLED)
        select_all_cb.config(state=tk.DISABLED)
        output_text.delete(1.0, tk.END)

        # Detect compute device and prepare transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = set_transform()

        # Clear previous results
        gui_highlights.clear()
        gui_extracted_paths.clear()

        # Load model once to avoid multiple heavy loads
        with _model_load_lock:
            try:
                _append_output(f"Loading model for detection on {device}...")
                _gui_load_model(checkpoint, device, status_cb=_append_output)
            except Exception as e:
                messagebox.showerror("Model load failed", str(e))
                # Re-enable controls
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

        # Start worker thread
        with pending_lock:
            pending_jobs["count"] = 1
        
        thread = threading.Thread(
            target=_gui_worker_process,
            args=(
                selected_clips,
                checkpoint,
                clips_output_dir,
                device,
                transform,
                _append_output,
                _job_done,
                _append_output,
                gui_highlights,
                gui_extracted_paths
            ),
            daemon=True,
        )
        thread.start()

    # Start the main event loop
    root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_gui()
