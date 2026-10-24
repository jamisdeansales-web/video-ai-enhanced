"""
Gradio UI for Video Upscaler Pro
"""

import gradio as gr
import torch
import sys
import os
import tempfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from utils.system_manager import get_system_manager
from utils.video_processor import VideoProcessor, format_duration
from processors.spatial_upscaler import create_upscaler
from processors.temporal_interpolator import create_interpolator
from processors.video_reverser import COMBINE_MODES, reverse_batch
from processors.frame_exporter import IMAGE_FORMATS, clear_export_dir, export_frames

# Global state
current_upscaler = None
current_interpolator = None
processing_cancelled = False

# Dropdown label -> the name SpatialUpscaler expects. A value of None marks a
# model that is listed but not implemented, so the UI can say so instead of
# failing somewhere deeper.
SPATIAL_MODELS = {
    "Real-ESRGAN (Fast, Excellent Quality)": "realesrgan",
    "SwinIR (Slow, Maximum Quality)": "swinir",
    "SwinIR Large (Slowest, Highest Quality)": "swinir_large",
}


def parse_model_choice(label):
    """Map a dropdown label to a model name, or None if it is not implemented."""
    if label in SPATIAL_MODELS:
        return SPATIAL_MODELS[label]

    # Labels have been reworded before; match on the distinctive part so an
    # older saved choice still resolves.
    if "SwinIR Large" in label:
        return "swinir_large"
    if "SwinIR" in label:
        return "swinir"
    if "SeedVR2" in label:
        return None
    return "realesrgan"


def get_system_info():
    """Get system information for display"""
    try:
        sys_manager = get_system_manager()
        return sys_manager.get_info_string()
    except Exception as e:
        return f"Error getting system info: {e}"


def on_video_upload(video):
    """Called when a video is uploaded"""
    if video is None:
        return "No video uploaded", "", gr.update(value="")

    try:
        # Get video info
        with VideoProcessor(video) as vp:
            metadata = vp.get_metadata()

        # Format info
        info = f"📹 Resolution: {metadata['width']}x{metadata['height']}\n"
        info += f"🎞️ FPS: {metadata['fps']:.2f}\n"
        info += f"⏱️ Duration: {format_duration(metadata['duration'])}\n"
        info += f"🎬 Frames: {metadata['frame_count']:,}\n"
        info += f"📦 Format: {metadata['container']} ({metadata['codec_name']})"

        # Default output resolution (4x)
        output_res = f"{metadata['width']*4}x{metadata['height']*4} (4K)"

        return info, output_res, gr.update(value="Video uploaded successfully!")

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return f"Error: {e}", "", gr.update(value=f"Error: {e}")


def update_output_resolution(video, scale):
    """Update output resolution based on video and scale"""
    if video is None:
        return ""

    try:
        with VideoProcessor(video) as vp:
            metadata = vp.get_metadata()

        output_width = metadata['width'] * int(scale)
        output_height = metadata['height'] * int(scale)

        # Add resolution name
        if output_width >= 7680:
            res_name = "8K"
        elif output_width >= 3840:
            res_name = "4K"
        elif output_width >= 2560:
            res_name = "2K"
        elif output_width >= 1920:
            res_name = "Full HD"
        else:
            res_name = "HD"

        return f"{output_width}x{output_height} ({res_name})"

    except:
        return "Unknown"


def toggle_interpolation(enabled, video):
    """Show/hide interpolation options and update target FPS"""
    if not enabled or video is None:
        return (
            gr.update(visible=enabled),  # interpolation_model
            gr.update(visible=enabled),  # fps_multiplier
            gr.update(visible=enabled, value="")   # target_fps
        )

    try:
        with VideoProcessor(video) as vp:
            source_fps = vp.get_fps()

        target = source_fps * 2  # Default 2x multiplier
        fps_text = f"{source_fps:.1f} FPS → {target:.1f} FPS"

        return (
            gr.update(visible=True),  # interpolation_model
            gr.update(visible=True),  # fps_multiplier
            gr.update(visible=True, value=fps_text)   # target_fps
        )
    except:
        return (
            gr.update(visible=enabled),
            gr.update(visible=enabled),
            gr.update(visible=enabled, value="")
        )


def update_target_fps(video, multiplier):
    """Update target FPS display based on multiplier"""
    if video is None:
        return ""

    try:
        with VideoProcessor(video) as vp:
            source_fps = vp.get_fps()

        target_fps = source_fps * int(multiplier)
        return f"{source_fps:.1f} FPS → {target_fps:.1f} FPS"
    except:
        return ""


def process_preview(video, model, scale, interp_enabled, interp_model, fps_mult, progress=gr.Progress()):
    """Process video preview (first 5 seconds)"""
    global current_upscaler, current_interpolator

    if video is None:
        return None, "⚠️ Please upload a video first"

    try:
        progress(0, desc="Initializing...")

        # Parse model name
        model_name = parse_model_choice(model)
        if model_name is None:
            return None, f"❌ {model.split('(')[0].strip()} is not implemented yet"

        # Create temp paths
        upscaled_path = tempfile.mktemp(suffix='_upscaled.mp4')
        output_path = tempfile.mktemp(suffix='_preview.mp4')

        # Progress callback
        def update_progress(current, total, eta):
            pct = current / total if total > 0 else 0
            progress(0.1 + 0.45 * pct, desc=f"Processing frame {current}/{total} (ETA: {int(eta)}s)")

        # Step 1: Upscale
        progress(0.1, desc="Loading upscaling model...")
        try:
            upscaler = create_upscaler(
                model_name=model_name,
                scale_factor=float(scale),
                device='auto'
            )
        except ImportError as ie:
            if "Real-ESRGAN not available" in str(ie):
                error_msg = "❌ Real-ESRGAN not available\n\n"
                error_msg += "📦 **Installation Required:**\n\n"
                error_msg += "**Option 1 (Recommended):** Use Python 3.10-3.12\n"
                error_msg += "  • Download: https://www.python.org/downloads/\n"
                error_msg += "  • Run: install.bat\n\n"
                error_msg += "**Option 2:** Manual install for Python 3.13\n"
                error_msg += "  • See: PYTHON_313_WORKAROUND.md\n"
                error_msg += "  • Or run: pip install realesrgan basicsr\n\n"
                error_msg += "💡 **Alternative:** RIFE interpolation works without Real-ESRGAN\n"
                error_msg += "  • Enable 'FPS Interpolation' and try without upscaling"
                return None, error_msg
            raise

        progress(0.15, desc="Upscaling preview (5 seconds)...")
        upscale_result = upscaler.upscale_preview(
            video,
            upscaled_path if interp_enabled else output_path,
            duration=5.0,
            progress_callback=update_progress
        )

        if not upscale_result['success']:
            return None, f"❌ Upscaling error: {upscale_result.get('error', 'Unknown error')}"

        # Step 2: Interpolate (if enabled)
        if interp_enabled:
            progress(0.55, desc="Loading interpolation model...")

            interpolator = create_interpolator(
                model_name='rife',  # Using RIFE
                device='auto'
            )

            def interp_progress(current, total, eta):
                pct = current / total if total > 0 else 0
                progress(0.6 + 0.4 * pct, desc=f"Interpolating {current}/{total} pairs (ETA: {int(eta)}s)")

            progress(0.6, desc=f"Interpolating FPS ({fps_mult}x)...")
            interp_result = interpolator.interpolate_preview(
                upscaled_path,
                output_path,
                fps_multiplier=int(fps_mult),
                duration=5.0,
                progress_callback=interp_progress
            )

            # Clean up temp file
            if os.path.exists(upscaled_path):
                os.remove(upscaled_path)

            if not interp_result['success']:
                return None, f"❌ Interpolation error: {interp_result.get('error', 'Unknown error')}"

            # Combine metrics
            metrics = upscale_result['metrics']
            interp_metrics = interp_result['metrics']

            status = f"✅ Preview completed!\n\n"
            status += f"📐 Upscaling:\n"
            status += f"  Input: {metrics['input_resolution']}\n"
            status += f"  Output: {metrics['output_resolution']}\n"
            status += f"  Time: {metrics['total_time']:.1f}s\n\n"
            status += f"🎞️ Interpolation:\n"
            status += f"  FPS: {interp_metrics.get('original_fps', 'N/A'):.1f} → {interp_metrics.get('new_fps', 'N/A'):.1f}\n"
            status += f"  Time: {interp_metrics['total_time']:.1f}s\n\n"
            status += f"⏱️ Total time: {metrics['total_time'] + interp_metrics['total_time']:.1f}s"

            return output_path, status
        else:
            # Only upscaling
            metrics = upscale_result['metrics']
            status = f"✅ Preview completed!\n\n"
            status += f"⏱️ Processing time: {metrics['total_time']:.1f}s\n"
            status += f"🎬 Frames processed: {metrics['frames_processed']}\n"
            status += f"📐 Input: {metrics['input_resolution']}\n"
            status += f"📐 Output: {metrics['output_resolution']}\n"
            status += f"⚡ Avg per frame: {metrics['avg_time_per_frame']:.3f}s"

            return output_path, status

    except Exception as e:
        logger.error(f"Error in preview: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}"


def process_reverse(files, keep_frames, combine_mode, progress=gr.Progress()):
    """
    Reverse one or more uploaded videos.

    Returns the status report, the first input as the original preview, the last
    reversed clip, the combined clip when one was produced, and every kept frame
    for the gallery.
    """
    if not files:
        return "No video selected.", None, None, None, []

    # gr.File hands back either paths or objects carrying one, depending on the
    # component's type; accept both so the tab works either way.
    paths = [getattr(f, "name", f) for f in files]

    def on_progress(index, total, name):
        progress((index - 1) / total, desc=f"Reversing {name} ({index}/{total})")

    results = reverse_batch(
        paths,
        combine_mode=combine_mode,
        keep_frames=keep_frames,
        progress_callback=on_progress,
    )

    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    lines = []
    for result in results:
        name = os.path.basename(result.get("input_path", "?"))
        if result.get("success"):
            detail = f"{result['frame_count']} frames in {result['elapsed']:.1f}s"
            if result.get("combined_path"):
                detail += f", combined ({combine_mode})"
            lines.append(f"✅ {name}: {detail}")
        else:
            lines.append(f"❌ {name}: {result.get('error', 'unknown error')}")

    summary = f"{len(succeeded)}/{len(results)} reversed"
    if failed:
        summary += f", {len(failed)} failed"
    report = summary + "\n\n" + "\n".join(lines)

    if not succeeded:
        return report, None, None, None, []

    last = succeeded[-1]
    gallery = [frame for r in succeeded for frame in r.get("frames", [])]

    return (
        report,
        last.get("input_path"),
        last.get("output_path"),
        last.get("combined_path"),
        gallery,
    )


# A long export can run to thousands of stills; the browser only gets a slice,
# and the zip or the folder carries the rest.
GALLERY_PREVIEW_LIMIT = 60


def process_export_frames(
    video, range_mode, start_time, end_time, every_n,
    image_format, jpeg_quality, upscale, scale, make_zip,
    progress=gr.Progress(),
):
    """Export a video's frames as stills, optionally upscaling each one."""
    if video is None:
        return "No video uploaded.", [], None

    def on_progress(done, total):
        progress(done / total if total else 0, desc=f"Exporting frame {done}/{total}")

    # Each run starts from an empty folder so a shorter range cannot leave
    # stills from a previous, longer one behind.
    output_dir = os.path.join(
        tempfile.gettempdir(), "video_frame_exports", Path(video).stem
    )
    clear_export_dir(output_dir)

    try:
        result = export_frames(
            video,
            output_dir=output_dir,
            range_mode=range_mode,
            start_time=start_time,
            end_time=end_time,
            every_n=int(every_n),
            image_format=image_format,
            jpeg_quality=int(jpeg_quality),
            upscale=bool(upscale),
            scale_factor=float(scale),
            make_zip=bool(make_zip),
            progress_callback=on_progress,
        )
    except Exception as exc:  # noqa: BLE001 - report it in the UI
        logger.error(f"Frame export failed: {exc}", exc_info=True)
        return f"❌ {exc}", [], None

    first, last = result["range"]
    fps = result["fps"] or 1.0
    report = (
        f"✅ {result['frame_count']} frames exported in {result['elapsed']:.1f}s\n"
        f"🎬 Source frames {first}-{last} ({first / fps:.2f}s - {last / fps:.2f}s)\n"
        f"📐 {result['source_resolution'][0]}x{result['source_resolution'][1]}"
        f" → {result['resolution'][0]}x{result['resolution'][1]}"
    )
    if result["upscaled"]:
        report += f" (upscaled {result['scale_factor']}x)"
    report += f"\n📁 {result['output_dir']}"

    gallery = result["frames"][:GALLERY_PREVIEW_LIMIT]
    if len(result["frames"]) > GALLERY_PREVIEW_LIMIT:
        report += f"\n👁️ Showing the first {GALLERY_PREVIEW_LIMIT} of {result['frame_count']}"

    return report, gallery, result["zip_path"]


def process_full_video(video, model, scale, interp_enabled, interp_model, fps_mult, temporal, progress=gr.Progress()):
    """Process full video"""
    global current_upscaler

    if video is None:
        return None, "⚠️ Please upload a video first", []

    try:
        progress(0, desc="Initializing...")

        # Parse model name
        model_name = parse_model_choice(model)
        if model_name is None:
            return None, f"❌ {model.split('(')[0].strip()} is not implemented yet", []

        # Create upscaler
        progress(0.05, desc="Loading AI model...")
        try:
            upscaler = create_upscaler(
                model_name=model_name,
                scale_factor=float(scale),
                device='auto'
            )
        except ImportError as ie:
            if "Real-ESRGAN not available" in str(ie):
                error_msg = "❌ Real-ESRGAN not available\n\n"
                error_msg += "📦 **Installation Required:**\n\n"
                error_msg += "**Option 1 (Recommended):** Use Python 3.10-3.12\n"
                error_msg += "  • Download: https://www.python.org/downloads/\n"
                error_msg += "  • Run: install.bat\n\n"
                error_msg += "**Option 2:** Manual install for Python 3.13\n"
                error_msg += "  • See: PYTHON_313_WORKAROUND.md\n"
                error_msg += "  • Or run: pip install realesrgan basicsr\n\n"
                error_msg += "💡 **Alternative:** RIFE interpolation works without Real-ESRGAN\n"
                error_msg += "  • Enable 'FPS Interpolation' and try without upscaling"
                return None, error_msg, []
            raise

        # Get estimate
        estimate = upscaler.estimate_processing_time(video)
        estimated_time = estimate.get('estimated_total_time_formatted', 'Unknown')

        logger.info(f"Estimated processing time: {estimated_time}")

        # Create output path
        output_dir = Path(tempfile.gettempdir()) / "video_upscaler_outputs"
        output_dir.mkdir(exist_ok=True)

        input_name = Path(video).stem
        output_path = str(output_dir / f"{input_name}_upscaled_{scale}x.mp4")

        # Progress callback
        def update_progress(current, total, eta):
            pct = current / total if total > 0 else 0
            progress(0.05 + 0.95 * pct, desc=f"Processing {current}/{total} frames (ETA: {int(eta)}s)")

        # Process video
        progress(0.05, desc=f"Processing full video (est. {estimated_time})...")
        result = upscaler.upscale_video(
            video,
            output_path,
            progress_callback=update_progress
        )

        if result['success']:
            metrics = result['metrics']

            status = f"✅ Video upscaling completed!\n\n"
            status += f"💾 Output saved to:\n{output_path}\n\n"
            status += f"📊 Statistics:\n"
            status += f"⏱️ Total time: {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f} min)\n"
            status += f"🎬 Frames: {metrics['frames_processed']:,}\n"
            status += f"📐 Resolution: {metrics['input_resolution']} → {metrics['output_resolution']}\n"
            status += f"⚡ Speed: {metrics['avg_time_per_frame']:.3f}s per frame"

            # Metrics table
            metrics_data = [
                ["Processing Time", f"{metrics['total_time']:.1f}s"],
                ["Frames Processed", f"{metrics['frames_processed']:,}"],
                ["Input Resolution", metrics['input_resolution']],
                ["Output Resolution", metrics['output_resolution']],
                ["Scale Factor", f"{metrics['scale_factor']}x"],
                ["Avg Time/Frame", f"{metrics['avg_time_per_frame']:.3f}s"],
            ]

            return output_path, status, metrics_data
        else:
            error_msg = f"❌ Processing failed: {result.get('error', 'Unknown error')}"
            return None, error_msg, []

    except Exception as e:
        logger.error(f"Error in full processing: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}", []


def create_interface():
    """Create the main Gradio interface"""

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Video Upscaler Pro",
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .status-box {
            font-family: monospace;
            font-size: 0.9em;
        }
        """
    ) as app:

        gr.Markdown("# 🎬 Video Upscaler Pro", elem_classes="main-header")
        gr.Markdown(
            "### Open Source Video Upscaling using Real-ESRGAN, SwinIR, and more\n"
            "Upload a video, choose your settings, and enhance your footage with AI."
        )

        with gr.Tabs():
            # ============================================================
            # TAB 1: PROCESSING
            # ============================================================
            with gr.Tab("🚀 Processing"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Video upload
                        video_input = gr.Video(
                            label="📹 Video Source (mp4, webm, mkv, mov, avi)",
                            include_audio=True
                        )

                        # Video info
                        video_info = gr.Textbox(
                            label="📊 Video Information",
                            interactive=False,
                            placeholder="Upload a video to see details...",
                            lines=4
                        )

                        # Main settings
                        with gr.Group():
                            gr.Markdown("### ⚙️ Upscaling Settings")

                            spatial_model = gr.Dropdown(
                                choices=list(SPATIAL_MODELS),
                                label="AI Model",
                                value="Real-ESRGAN (Fast, Excellent Quality)",
                                info="Choose the model based on your needs and hardware"
                            )

                            scale_factor = gr.Slider(
                                minimum=0.5,
                                maximum=16.0,
                                step=0.1,
                                value=4.0,
                                label="Scale Factor",
                                info="<1.0 = Downscale, 2.0/4.0 = Optimal AI, Other = Hybrid (AI+Resize)"
                            )

                            # Quick presets
                            with gr.Row():
                                gr.Markdown("**Quick Presets:**")
                            with gr.Row():
                                btn_720_1080 = gr.Button("720p→1080p (1.5×)", size="sm")
                                btn_1080_4k = gr.Button("1080p→4K (2×)", size="sm")
                                btn_4k_8k = gr.Button("4K→8K (2×)", size="sm")
                                btn_half = gr.Button("Downscale (0.5×)", size="sm")

                            # Wire up preset buttons
                            btn_720_1080.click(lambda: 1.5, outputs=scale_factor)
                            btn_1080_4k.click(lambda: 2.0, outputs=scale_factor)
                            btn_4k_8k.click(lambda: 2.0, outputs=scale_factor)
                            btn_half.click(lambda: 0.5, outputs=scale_factor)

                            output_resolution = gr.Textbox(
                                label="Output Resolution",
                                interactive=False,
                                placeholder="Will be calculated automatically"
                            )

                        # Interpolation settings
                        with gr.Group():
                            gr.Markdown("### 🎞️ FPS Interpolation")

                            enable_interpolation = gr.Checkbox(
                                label="Enable FPS Interpolation",
                                value=False,
                                interactive=True
                            )

                            interpolation_model = gr.Dropdown(
                                choices=["RIFE (Fast - Optical Flow)", "DAIN (Maximum Quality) - Coming Soon"],
                                label="Interpolation Model",
                                value="RIFE (Fast - Optical Flow)",
                                visible=False,
                                interactive=True
                            )

                            fps_multiplier = gr.Slider(
                                minimum=2,
                                maximum=4,
                                step=1,
                                value=2,
                                label="FPS Multiplier",
                                visible=False,
                                interactive=True,
                                info="2x = 30→60fps, 4x = 30→120fps"
                            )

                            target_fps = gr.Textbox(
                                label="Target FPS",
                                interactive=False,
                                visible=False,
                                placeholder="Will be calculated automatically"
                            )

                    with gr.Column(scale=1):
                        # Preview and results
                        gr.Markdown("### 👁️ Preview / Result")

                        video_preview = gr.Video(
                            label="Preview (5 seconds)",
                            interactive=False
                        )

                        video_output = gr.Video(
                            label="Final Result",
                            interactive=False
                        )

                # Action buttons
                with gr.Row():
                    preview_btn = gr.Button(
                        "🔍 Preview (5 seconds)",
                        variant="secondary",
                        size="lg"
                    )
                    process_btn = gr.Button(
                        "🚀 Process Full Video",
                        variant="primary",
                        size="lg"
                    )

                # Status
                status_text = gr.Textbox(
                    label="📝 Status",
                    interactive=False,
                    lines=8,
                    elem_classes="status-box"
                )

            # ============================================================
            # TAB 2: COMPARISON
            # ============================================================
            with gr.Tab("📊 Comparison"):
                gr.Markdown("### Before / After Comparison")
                gr.Markdown("*Use the Processing tab to create upscaled videos, then view them here side-by-side*")

                with gr.Row():
                    original_video = gr.Video(label="Original")
                    processed_video = gr.Video(label="Upscaled")

                metrics_display = gr.Dataframe(
                    headers=["Metric", "Value"],
                    label="Quality Metrics",
                    value=[
                        ["Processing Time", "-"],
                        ["Frames Processed", "-"],
                        ["Input Resolution", "-"],
                        ["Output Resolution", "-"],
                        ["Scale Factor", "-"],
                        ["Avg Time/Frame", "-"]
                    ]
                )

            # ============================================================
            # TAB 3: REVERSE
            # ============================================================
            with gr.Tab("🔁 Reverse"):
                gr.Markdown("### Reverse Videos")
                gr.Markdown(
                    "*Reverse one or several videos frame by frame. "
                    "Optionally keep the extracted frames, or join the original "
                    "and reversed clips together.*"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        reverse_input = gr.File(
                            label="📂 Videos (mp4, webm, mkv, mov, avi)",
                            file_count="multiple",
                            file_types=["video"],
                        )
                        reverse_keep_frames = gr.Checkbox(
                            label="🗂️ Keep extracted frames",
                            value=False,
                        )
                        reverse_combine = gr.Radio(
                            choices=list(COMBINE_MODES),
                            value="none",
                            label="🔁 Combine original and reversed",
                            info="none = reversed only, ab = original then reversed, ba = reversed then original",
                        )
                        with gr.Row():
                            reverse_clear_btn = gr.Button("Clear")
                            reverse_btn = gr.Button("🔁 Reverse", variant="primary")
                        reverse_original = gr.Video(label="📼 Original", interactive=False)

                    with gr.Column(scale=1):
                        reverse_status = gr.Textbox(
                            label="🔁 Status",
                            interactive=False,
                            lines=6,
                            placeholder="Upload one or more videos and press Reverse...",
                        )
                        reverse_output = gr.Video(label="🎥 Reversed", interactive=False)
                        reverse_combined = gr.Video(label="🔗 Combined", interactive=False)
                        reverse_frames = gr.Gallery(
                            label="🖼️ Extracted frames",
                            columns=6,
                            object_fit="contain",
                        )

            # ============================================================
            # TAB 4: EXPORT FRAMES
            # ============================================================
            with gr.Tab("🖼️ Export Frames"):
                gr.Markdown("### Export Frames")
                gr.Markdown(
                    "*Write a video's frames out as still images, over the whole "
                    "clip or a chosen span, optionally upscaling each one.*"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        export_input = gr.Video(
                            label="📹 Video Source (mp4, webm, mkv, mov, avi)",
                        )

                        with gr.Group():
                            gr.Markdown("### ⏱️ Range")
                            export_range_mode = gr.Radio(
                                choices=[
                                    ("Whole video", "all"),
                                    ("From a start time", "from"),
                                    ("Up to an end time", "to"),
                                    ("Between two times", "range"),
                                ],
                                value="all",
                                label="Which frames to export",
                            )
                            with gr.Row():
                                export_start = gr.Number(
                                    label="Start (seconds)",
                                    value=0,
                                    minimum=0,
                                    visible=False,
                                )
                                export_end = gr.Number(
                                    label="End (seconds)",
                                    value=5,
                                    minimum=0,
                                    visible=False,
                                )
                            export_every_n = gr.Slider(
                                label="Keep one frame out of",
                                minimum=1,
                                maximum=60,
                                step=1,
                                value=1,
                                info="1 = every frame. Raise it to thin out a long export.",
                            )

                        with gr.Group():
                            gr.Markdown("### 🖼️ Output")
                            export_format = gr.Radio(
                                choices=list(IMAGE_FORMATS),
                                value="png",
                                label="Image format",
                                info="png = lossless, jpg = much smaller",
                            )
                            export_quality = gr.Slider(
                                label="JPEG quality",
                                minimum=50,
                                maximum=100,
                                step=1,
                                value=95,
                                visible=False,
                            )
                            export_zip = gr.Checkbox(
                                label="📦 Also bundle the frames into a zip",
                                value=True,
                            )

                        with gr.Group():
                            gr.Markdown("### ⚙️ Upscaling")
                            export_upscale = gr.Checkbox(
                                label="✨ Upscale each frame",
                                value=False,
                            )
                            export_scale = gr.Slider(
                                label="Scale factor",
                                minimum=0.5,
                                maximum=8.0,
                                step=0.1,
                                value=2.0,
                                visible=False,
                                info="<1.0 = Downscale, 2.0/4.0 = Optimal AI, Other = Hybrid (AI+Resize)",
                            )

                        export_btn = gr.Button("🖼️ Export Frames", variant="primary")

                    with gr.Column(scale=1):
                        export_status = gr.Textbox(
                            label="🖼️ Status",
                            interactive=False,
                            lines=6,
                            placeholder="Upload a video, pick a range and press Export...",
                        )
                        export_zip_file = gr.File(
                            label="📦 Frames archive",
                            interactive=False,
                        )
                        export_gallery = gr.Gallery(
                            label="🖼️ Exported frames",
                            columns=6,
                            object_fit="contain",
                        )

            # ============================================================
            # TAB 5: SETTINGS
            # ============================================================
            with gr.Tab("⚙️ Advanced Settings"):
                gr.Markdown("### Advanced Configuration")
                gr.Markdown("*These features are coming in future updates*")

                with gr.Group():
                    gr.Markdown("#### 🎯 Temporal Coherence")

                    temporal_method = gr.Radio(
                        choices=[
                            "Auto (Recommended)",
                            "Optical Flow",
                            "EMA Filter"
                        ],
                        value="Auto (Recommended)",
                        label="Temporal Coherence Method",
                        interactive=False
                    )

                with gr.Group():
                    gr.Markdown("#### 🖥️ Performance")

                    device_choice = gr.Radio(
                        choices=[
                            "Auto (Recommended)",
                            "GPU (CUDA)",
                            "CPU"
                        ],
                        value="Auto (Recommended)",
                        label="Processing Device",
                        interactive=False
                    )

                    use_fp16 = gr.Checkbox(
                        label="Use FP16 (Faster, slight quality loss)",
                        value=True,
                        interactive=False
                    )

            # ============================================================
            # TAB 4: SYSTEM INFO
            # ============================================================
            with gr.Tab("💻 System"):
                gr.Markdown("### System Information")

                system_info = gr.Textbox(
                    label="Detected Configuration",
                    lines=15,
                    interactive=False,
                    value=get_system_info()
                )

                refresh_btn = gr.Button("🔄 Refresh Information")

                gr.Markdown("### 📚 Quick Start Guide")
                gr.Markdown("""
                1. **Upload Video**: Click or drag a video file to the upload area
                2. **Choose Settings**: Select model and scale factor (4x recommended)
                3. **Preview**: Click "Preview" to test on first 5 seconds
                4. **Process**: Click "Process Full Video" to upscale entire video
                5. **Download**: Your processed video will be available for download

                **Tips:**
                - Start with 2x scale for faster processing
                - Use preview to check quality before full processing
                - GPU processing is 10-20x faster than CPU
                - Larger videos will take longer (check System tab for estimates)
                """)

        # ============================================================
        # EVENT HANDLERS
        # ============================================================

        # Export Frames tab: only show the inputs the chosen options need.
        def _toggle_range_inputs(mode):
            return (
                gr.update(visible=mode in ("from", "range")),
                gr.update(visible=mode in ("to", "range")),
            )

        export_range_mode.change(
            fn=_toggle_range_inputs,
            inputs=[export_range_mode],
            outputs=[export_start, export_end],
        )

        export_format.change(
            fn=lambda fmt: gr.update(visible=fmt == "jpg"),
            inputs=[export_format],
            outputs=[export_quality],
        )

        export_upscale.change(
            fn=lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[export_upscale],
            outputs=[export_scale],
        )

        export_btn.click(
            fn=process_export_frames,
            inputs=[
                export_input,
                export_range_mode,
                export_start,
                export_end,
                export_every_n,
                export_format,
                export_quality,
                export_upscale,
                export_scale,
                export_zip,
            ],
            outputs=[export_status, export_gallery, export_zip_file],
        )

        # Reverse tab
        reverse_btn.click(
            fn=process_reverse,
            inputs=[reverse_input, reverse_keep_frames, reverse_combine],
            outputs=[
                reverse_status,
                reverse_original,
                reverse_output,
                reverse_combined,
                reverse_frames,
            ],
        )

        reverse_clear_btn.click(
            fn=lambda: ("", None, None, None, [], None, False, "none"),
            outputs=[
                reverse_status,
                reverse_original,
                reverse_output,
                reverse_combined,
                reverse_frames,
                reverse_input,
                reverse_keep_frames,
                reverse_combine,
            ],
        )

        # Video upload
        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[video_info, output_resolution, status_text]
        )

        # Scale change
        scale_factor.change(
            fn=update_output_resolution,
            inputs=[video_input, scale_factor],
            outputs=[output_resolution]
        )

        # Interpolation toggle
        enable_interpolation.change(
            fn=toggle_interpolation,
            inputs=[enable_interpolation, video_input],
            outputs=[interpolation_model, fps_multiplier, target_fps]
        )

        # FPS multiplier change
        fps_multiplier.change(
            fn=update_target_fps,
            inputs=[video_input, fps_multiplier],
            outputs=[target_fps]
        )

        # Preview button
        preview_btn.click(
            fn=process_preview,
            inputs=[
                video_input,
                spatial_model,
                scale_factor,
                enable_interpolation,
                interpolation_model,
                fps_multiplier
            ],
            outputs=[video_preview, status_text]
        )

        # Process button
        process_btn.click(
            fn=process_full_video,
            inputs=[
                video_input,
                spatial_model,
                scale_factor,
                enable_interpolation,
                interpolation_model,
                fps_multiplier,
                temporal_method
            ],
            outputs=[video_output, status_text, metrics_display]
        )

        # Refresh system info
        refresh_btn.click(
            fn=get_system_info,
            outputs=[system_info]
        )

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch()
