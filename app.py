#!/usr/bin/env python3
"""
Video Upscaler Pro - Main Application Entry Point
Open source video upscaling application using AI models

With no arguments it launches the Gradio web UI. With --reverse it runs the
video reverser from the command line instead, so batches can be scripted
without going through the browser.
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import gradio
        import torch
        import cv2
        import numpy
        return True
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("\nPlease run the installation script:")
        print("  Windows: install.bat")
        print("  Linux/macOS: ./install.sh")
        return False


def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog='app.py',
        description='Video Upscaler Pro - AI video upscaling and reversal.',
        epilog=(
            'Run without arguments to open the web UI. '
            'Example: app.py --reverse clip.webm --video-loop ab'
        ),
    )
    parser.add_argument(
        '--reverse',
        nargs='+',
        metavar='VIDEO',
        help='reverse the given video(s) from the command line and exit',
    )
    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='with --reverse, also write out the individual frames as JPEGs',
    )
    parser.add_argument(
        '--video-loop',
        choices=['ab', 'ba'],
        help=(
            'with --reverse, also write the original and reversed clips joined '
            'together: ab = original first, ba = reversed first'
        ),
    )
    parser.add_argument(
        '--output-dir',
        metavar='DIR',
        help='with --reverse, where to write the results (default: a temp dir)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='port for the web UI (default: 7860)',
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='do not open a browser window when starting the web UI',
    )
    return parser.parse_args(argv)


def run_reverse_cli(args):
    """Reverse videos from the command line. Returns a process exit code."""
    missing = [path for path in args.reverse if not os.path.isfile(path)]
    if missing:
        for path in missing:
            print(f"ERROR: no such file: {path}")
        return 1

    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    from processors.video_reverser import reverse_batch

    def on_progress(index, total, name):
        print(f"[{index}/{total}] {name}")

    results = reverse_batch(
        args.reverse,
        combine_mode=args.video_loop or 'none',
        keep_frames=args.keep_frames,
        output_dir=args.output_dir,
        progress_callback=on_progress,
    )

    failed = 0
    for result in results:
        name = os.path.basename(result.get('input_path', '?'))
        if not result.get('success'):
            print(f"FAILED  {name}: {result.get('error', 'unknown error')}")
            failed += 1
            continue

        print(f"OK      {name} -> {result['output_path']}")
        if result.get('combined_path'):
            print(f"        combined -> {result['combined_path']}")
        if result.get('frames'):
            print(f"        {len(result['frames'])} frames kept")

    print()
    print(f"{len(results) - failed}/{len(results)} reversed")
    return 1 if failed else 0


def main():
    """Main application entry point"""
    args = parse_args()

    # Dependencies are needed by both modes, so check before branching.
    if not check_dependencies():
        sys.exit(1)

    if args.reverse:
        sys.exit(run_reverse_cli(args))

    print("=" * 50)
    print("Video Upscaler Pro - Starting...")
    print("=" * 50)
    print()

    # Import UI module
    try:
        from ui.gradio_app import create_interface
    except ImportError as e:
        print(f"ERROR: Failed to import UI module: {e}")
        print("\nThe application may not be properly installed.")
        sys.exit(1)

    # Create and launch the interface
    try:
        print("Creating Gradio interface...")
        app = create_interface()

        print()
        print("=" * 50)
        print("Application is ready!")
        print("=" * 50)
        print()
        print("Opening in your browser...")
        print(f"If it doesn't open automatically, visit: http://localhost:{args.port}")
        print()
        print("Press Ctrl+C to stop the application")
        print()

        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            inbrowser=not args.no_browser,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
