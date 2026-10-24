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


def unbuffer_output():
    """
    Flush stdout and stderr line by line.

    Python block-buffers stdout when it is not a terminal, so when the process
    dies hard - a native crash, or the OS killing it - the last few KB are
    still in the buffer and never reach the console or a redirect. That is the
    output which would have said what was happening at the time.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass


def get_log_path():
    """Where the run log goes."""
    base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'video-upscaler-pro', 'logs', 'app.log')


def setup_logging(verbose=False):
    """
    Log to a file as well as the console.

    A crash inside CUDA or a native library kills the process without a Python
    traceback, so the console shows the last thing printed and nothing about
    why. The file keeps the run's context - model, resolution, VRAM estimate,
    frame number - which is what makes such a crash diagnosable afterwards.
    """
    import logging

    path = get_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    handlers = [logging.StreamHandler()]
    try:
        # Keep the previous run alongside the current one. Each run starting
        # fresh would otherwise destroy a crash log the moment the app is
        # relaunched - which is the first thing anyone does after a crash.
        #
        # Only the first process in the tree may rotate. Gradio spawns a child
        # on Windows, and a child rotating again renamed the file out from
        # under the parent's still-open handle: the parent kept writing to
        # what was now app.previous.log while app.log sat empty. The marker is
        # an environment variable because a spawned child inherits it.
        if not os.environ.get('V2UP_LOG_INITIALISED'):
            if os.path.exists(path):
                previous = os.path.join(os.path.dirname(path), 'app.previous.log')
                try:
                    os.replace(path, previous)
                except OSError:
                    pass
            os.environ['V2UP_LOG_INITIALISED'] = '1'

        # Append, so a second process in the tree adds to the log instead of
        # truncating what the first one wrote.
        file_handler = logging.FileHandler(path, mode='a', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    except OSError as exc:
        print(f"WARNING: could not open the log file {path}: {exc}")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    return path


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
        '--export-frames',
        nargs='+',
        metavar='VIDEO',
        help='export the given video(s) frame by frame as stills and exit',
    )
    parser.add_argument(
        '--start',
        type=float,
        metavar='SECONDS',
        help='with --export-frames, start of the range (default: the beginning)',
    )
    parser.add_argument(
        '--end',
        type=float,
        metavar='SECONDS',
        help='with --export-frames, end of the range (default: the end)',
    )
    parser.add_argument(
        '--every',
        type=int,
        default=1,
        metavar='N',
        help='with --export-frames, keep one frame out of N (default: 1)',
    )
    parser.add_argument(
        '--format',
        choices=['png', 'jpg'],
        default='png',
        help='with --export-frames, the image format (default: png)',
    )
    parser.add_argument(
        '--upscale',
        type=float,
        metavar='FACTOR',
        help=(
            'with --export-frames, upscale each frame by this factor; '
            'below 1.0 downscales'
        ),
    )
    parser.add_argument(
        '--zip',
        action='store_true',
        help='with --export-frames, also bundle the stills into a zip',
    )
    parser.add_argument(
        '--output-dir',
        metavar='DIR',
        help=(
            'with --reverse or --export-frames, where to write the results '
            '(default: a temp dir)'
        ),
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
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        metavar='ADDRESS',
        help=(
            'address to bind the web UI to (default: 127.0.0.1, this machine '
            'only). Use 0.0.0.0 to expose it on your network'
        ),
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='log debug detail, useful when reporting a crash',
    )
    return parser.parse_args(argv)


def run_reverse_cli(args):
    """Reverse videos from the command line. Returns a process exit code."""
    missing = [path for path in args.reverse if not os.path.isfile(path)]
    if missing:
        for path in missing:
            print(f"ERROR: no such file: {path}")
        return 1

    setup_logging(args.verbose)

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


def run_export_frames_cli(args):
    """Export frames from the command line. Returns a process exit code."""
    missing = [path for path in args.export_frames if not os.path.isfile(path)]
    if missing:
        for path in missing:
            print(f"ERROR: no such file: {path}")
        return 1

    setup_logging(args.verbose)

    from processors.frame_exporter import export_frames

    # The range mode follows from which bounds were given, so the CLI does not
    # need a separate --range-mode flag.
    if args.start is not None and args.end is not None:
        range_mode = 'range'
    elif args.start is not None:
        range_mode = 'from'
    elif args.end is not None:
        range_mode = 'to'
    else:
        range_mode = 'all'

    failed = 0
    for path in args.export_frames:
        name = os.path.basename(path)
        output_dir = args.output_dir
        if output_dir and len(args.export_frames) > 1:
            output_dir = os.path.join(output_dir, os.path.splitext(name)[0])

        try:
            result = export_frames(
                path,
                output_dir=output_dir,
                range_mode=range_mode,
                start_time=args.start,
                end_time=args.end,
                every_n=args.every,
                image_format=args.format,
                upscale=args.upscale is not None,
                scale_factor=args.upscale or 1.0,
                make_zip=args.zip,
            )
        except Exception as exc:  # noqa: BLE001 - keep going through the list
            print(f"FAILED  {name}: {exc}")
            failed += 1
            continue

        print(f"OK      {name}: {result['frame_count']} frames -> {result['output_dir']}")
        if result['upscaled']:
            print(
                f"        {result['source_resolution'][0]}x{result['source_resolution'][1]}"
                f" -> {result['resolution'][0]}x{result['resolution'][1]}"
                f" ({result['scale_factor']}x)"
            )
        if result['zip_path']:
            print(f"        zip -> {result['zip_path']}")

    print()
    total = len(args.export_frames)
    print(f"{total - failed}/{total} exported")
    return 1 if failed else 0


def main():
    """Main application entry point"""
    unbuffer_output()
    args = parse_args()

    if args.reverse and args.export_frames:
        print("ERROR: --reverse and --export-frames cannot be combined")
        sys.exit(2)

    # Dependencies are needed by both modes, so check before branching.
    if not check_dependencies():
        sys.exit(1)

    if args.export_frames:
        sys.exit(run_export_frames_cli(args))

    if args.reverse:
        sys.exit(run_reverse_cli(args))

    log_path = setup_logging(args.verbose)

    print("=" * 50)
    print("Video Upscaler Pro - Starting...")
    print("=" * 50)
    print()
    print(f"Log file: {log_path}")
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
        display_host = 'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
        print(f"If it doesn't open automatically, visit: http://{display_host}:{args.port}")
        if args.host == '0.0.0.0':
            print()
            print("WARNING: bound to 0.0.0.0 - reachable from your network,")
            print("         including the folders the app is allowed to serve.")
        print()
        print("Press Ctrl+C to stop the application")
        print()

        app.launch(
            # 127.0.0.1 by default: the UI is a local tool, and the
            # output folder it is allowed to serve would otherwise be readable
            # by anything on the same network.
            server_name=args.host,
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
