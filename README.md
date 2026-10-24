# V2UP 🎬 Video Upscaler Pro

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Open source video upscaling application using AI models like Real-ESRGAN, SwinIR, and RIFE. Achieve quality comparable to Topaz Video Enhance AI with a simple, user-friendly interface powered by Gradio.

<p align="center">
  <img src="docs/screenshot.png" alt="Video Upscaler Pro Interface" width="800"/>
</p>

## ✨ Features

- **Reverse videos** frame by frame, in batch, from the UI or the CLI
- **Export frames** as PNG/JPEG over any time range, optionally upscaled
- **Batch queue** in the Processing tab, with one model load for the whole run
- **RIFE frame interpolation**, the real network rather than an optical-flow stand-in
- **Resume** an interrupted upscale instead of starting over

### Currently Working (v0.2.0)

- **🚀 Real-ESRGAN Upscaling** ✅
  - 2x, 4x, and 8x upscaling
  - Fast, excellent quality for realistic videos
  - Automatic model weight downloads
  - Tile-based processing for large videos

- **🎞️ RIFE Frame Interpolation** ✅
  - 2x, 3x, and 4x FPS interpolation
  - Optical flow-based smooth interpolation
  - Combine with upscaling for enhanced quality
  - 30fps → 60fps, 120fps support

- **⚡ Smart Performance** ✅
  - Automatic GPU/CPU detection
  - VRAM usage optimization
  - Dynamic batch size adjustment
  - FP16 support for faster processing
  - CPU fallback mode

- **🎯 Testing Utilities** ✅
  - Built-in test video generator
  - Multiple test patterns (shapes, text, checkerboard, etc.)
  - Quick testing script included

- **🖥️ User-Friendly Interface** ✅
  - Simple drag-and-drop video upload
  - Real-time preview (5 seconds)
  - Live progress tracking with ETA
  - System information display
  - Before/after comparison

- **🔬 SwinIR Upscaling** ✅
  - Transformer-based, slower than Real-ESRGAN but more detailed
  - Standard and Large variants

- **🎞️ BasicVSR++ Upscaling** ✅
  - Reads a run of frames instead of one at a time, for temporal coherence
  - 4× native

- **📊 Quality Metrics** ✅
  - PSNR, SSIM and motion-compensated temporal stability
- **💾 H.264 / H.265 output** ✅
  - CRF quality control, replacing the old mp4v encoder

### Coming Soon

Nothing currently planned.
- **Batch Processing**: Process multiple videos at once

## 📋 Requirements

### Minimum Requirements (CPU Mode)
- **Python 3.10, 3.11 or 3.12** (a final release, not an alpha/beta/rc)
- 32 GB RAM
- Intel i7 / AMD Ryzen 7 or better (8+ cores)
- 50 GB free disk space

### Recommended Requirements (GPU Mode)
- **Python 3.10, 3.11 or 3.12** (a final release, not an alpha/beta/rc)
- 16 GB RAM (32 GB recommended)
- NVIDIA GPU with 12+ GB VRAM (RTX 3060 or better)
- 50 GB free disk space
- An NVIDIA driver recent enough for the CUDA build the installer picks

### Python version

The installer probes `py -3.12`, `py -3.11`, `py -3.10` and then `python`, and
uses the first one that passes `scripts/check_python.py`. It deliberately skips:

- **Python 3.13+** - BasicSR needs `numpy < 2`, which has no wheels for 3.13
  (details in [PYTHON_313_WORKAROUND.md](PYTHON_313_WORKAROUND.md)).
- **Pre-releases** - for example 3.12.0b3, whose `re` module makes any modern
  pip crash with `InvalidVersion: Invalid version: '0.dev0'`.
- **MSYS2 / MinGW Python** - it cannot install the official PyTorch wheels.

If none of your interpreters qualify, the installer prints which ones it
rejected and why. Install Python 3.12 from
[python.org](https://www.python.org/downloads/release/python-31210/) and tick
"Add Python to PATH".

### CUDA build

`scripts/detect_cuda.py` reads the GPU's compute capability and picks the
matching PyTorch wheel index:

| GPU | Wheels |
|-----|--------|
| Blackwell / RTX 50xx (sm_120) | `cu128` |
| Hopper and older (sm_90 and below) | `cu124` |
| No NVIDIA GPU | CPU wheels |

This matters: the `cu124` wheels carry no kernels above sm_90, so on an RTX 50xx
they install cleanly and then refuse to run with *"sm_120 is not compatible with
the current PyTorch installation"*.

### Supported Platforms
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)
- ✅ macOS 12+ (Apple Silicon and Intel)

## 🚀 Quick Start

### Windows

1. **Download or clone the repository**
   ```bash
   git clone https://github.com/mikecastrodemaria/v2up-video-upscaler-pro.git
   cd v2up-video-upscaler-pro
   ```

2. **Run the installer**
   ```bash
   install.bat
   ```

3. **Start the application**
   ```bash
   start.bat
   ```

The application will open automatically in your browser at `http://localhost:7860`.

### Linux / macOS

1. **Download or clone the repository**
   ```bash
   git clone https://github.com/mikecastrodemaria/v2up-video-upscaler-pro.git
   cd v2up-video-upscaler-pro
   ```

2. **Run the installer**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Start the application**
   ```bash
   ./start.sh
   ```

The application will open automatically in your browser at `http://localhost:7860`.

## 📖 Detailed Installation

### Prerequisites

#### Windows
- Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
  - ⚠️ During installation, check "Add Python to PATH"
- (Optional) Install [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads) for GPU support

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install ffmpeg  # For video processing
```

For NVIDIA GPU support:
```bash
# Install CUDA toolkit
# Follow instructions at: https://developer.nvidia.com/cuda-downloads
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install FFmpeg
brew install ffmpeg
```

### Manual Installation

If the automatic installer doesn't work, you can install manually:

```bash
# Create the virtual environment with a supported interpreter
py -3.12 -m venv venv          # Windows
python3.12 -m venv venv        # Linux/macOS

# Call the venv interpreter directly, so a failed activation cannot
# silently install into the system Python instead
venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# Install PyTorch for your GPU FIRST (cu128 for RTX 50xx, cu124 otherwise)
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Then everything else
venv\Scripts\python.exe -m pip install -r requirements.txt

# basicsr only ships an sdist whose setup.py imports torch, so it has to be
# built against the venv rather than pip's isolated build environment
venv\Scripts\python.exe -m pip install --no-build-isolation basicsr realesrgan
venv\Scripts\python.exe scripts\patch_basicsr.py

# Check the result
venv\Scripts\python.exe scripts\verify_install.py
```

## 🎮 Usage

### Basic Workflow

1. **Upload Video**
   - Drag and drop your video or click to browse
   - Supported formats: MP4, AVI, MKV, MOV, WEBM

2. **Choose Settings**
   - Select AI model (Real-ESRGAN recommended for most cases)
   - Choose scale factor (2x, 4x, or 8x)
   - Optionally enable FPS interpolation

3. **Preview**
   - Click "Preview" to process the first 5 seconds
   - Check the quality before processing the full video

4. **Process**
   - Click "Process Full Video" to upscale your entire video
   - Monitor progress and ETA
   - Download the result when complete

### Model Selection Guide

| Model | Best For | Speed | VRAM | Quality |
|-------|----------|-------|------|---------|
| **Real-ESRGAN** | General use, realistic videos | Fast | 4-8 GB | Excellent |
| **SwinIR** | Photo-realistic, detailed content | Slow | 6-10 GB | Maximum |
| **SwinIR Large** | Stills and short clips | Slowest | 10-16 GB | Highest |
| **BasicVSR++** | Footage that flickers between frames | Medium | 8-16 GB | Excellent + Coherence |

**Notes on BasicVSR++.** The other models see one frame at a time, so each
frame is upscaled independently and small differences between them show up as
flicker. BasicVSR++ propagates information along the sequence in both
directions and aligns neighbouring frames with optical flow, which is what
removes that flicker. It replaces SeedVR2, which was listed here but never
implemented: there is no pip package for it, it needs a cloned repo, 3B-7B
weights and a diffusion stack. BasicVSR++ reaches the same goal with 7M
parameters and a 29 MB checkpoint, and its architecture is already in
`basicsr`.

It is a **4× model** — there is no 2× checkpoint, so other factors are reached
by resizing its output.

Because it works on runs of frames, it goes through `TemporalUpscaler` rather
than `SpatialUpscaler`, which feeds the video in overlapping windows sized from
the available VRAM. The overlap is not cosmetic: a frame at the edge of a
window has little propagated context, so too small an overlap leaves a visible
seam every window. Measured at window=16, the motion-compensated residual at
the seams against everywhere else was 10.4× at overlap=2 and 0.4× at
overlap=4, so the default overlap has a floor of 4 frames.

**Honest caveat.** The implementation is verified — weights load strictly, no
frame falls back, windows reassemble exactly — but the temporal-coherence
*benefit* was not demonstrated here. On the synthetic clips used for testing,
Real-ESRGAN is already almost perfectly stable (motion-compensated residual
0.029) and BasicVSR++ scored higher, not lower. Those clips are test patterns
with little real motion and no realistic grain, which is the wrong material
for this: the published checkpoint is trained on real video (REDS). Judge it on
your own footage, with the kind of noise and compression that actually makes a
per-frame upscaler flicker.

**Notes on SwinIR.** The architecture comes from the installed `basicsr`, so
only the weights are downloaded (64 MB for x4, 142 MB for Large).

It always runs in **fp32**, whatever the global half-precision setting says:
its attention overflows the fp16 range and the network returns all-NaN, which
does not raise — it just yields a corrupted frame. bfloat16 is numerically
correct but measured roughly 15× slower than fp32 here, so it is not used
either.

At **x4** it uses the real-world SR checkpoint, trained on realistic
degradations. At **x2** it uses the *classical* SR checkpoint, because the
official x2 real-world weights were trained with an upsampler (`nearest+conv`)
that `basicsr` only implements at x4. Classical SR is trained on clean bicubic
downscaling, so x2 is sharper on clean sources and less forgiving of
compression artefacts than x4. For noisy or heavily compressed footage, x4 is
the better model even if you then scale back down.

SwinIR tiles by default, sized from the available VRAM, because it needs far
more memory than Real-ESRGAN at the same resolution.

### FPS Interpolation

- **RIFE (Available Now)**: Optical flow-based interpolation, works well for most content (24→60 fps, 30→120 fps)
- **DAIN (Coming Soon)**: Advanced interpolation for complex scenes with depth

### Testing the Application

We provide a complete testing utility to help you get started:

```bash
# Run the test script (generates test videos and starts app)
python test_app.py
```

This will:
1. Check all dependencies
2. Generate test videos in `test_videos/` directory
3. Display system information
4. Launch the application

Or generate test videos manually:

```bash
# Generate all test videos
python -m tests.test_video_generator --output-dir test_videos

# Generate specific type
python -m tests.test_video_generator --type shapes --duration 10

# Custom resolution and FPS
python -m tests.test_video_generator --resolution 1280x720 --fps 60
```

Available test video types:
- `shapes`: Moving shapes (good for testing upscaling)
- `text`: Scrolling text (tests detail preservation)
- `checker`: Checkerboard animation (tests fine details)
- `bars`: Color bars with animation (tests color accuracy)
- `resolution`: Resolution test pattern (tests sharpness)
- `all`: Generate all test videos

### 🔁 Reverse (tab)

Reverses videos frame by frame, merged in from
[ReverseVideo](https://github.com/mikecastrodemaria/ReverseVideo).

1. Drop one or several videos onto **Videos** (batch is supported).
2. Tick **Keep extracted frames** to also get every frame as a JPEG, shown in
   the gallery under the previews.
3. Pick a **Combine** mode: `none` for the reversed clip only, `ab` for original
   then reversed, `ba` for reversed then original.
4. Press **Reverse**. The original, the reversed clip and the combined clip are
   previewed side by side, and the status box reports each file separately - one
   bad file does not abort the batch.

Frames are spooled to a temporary directory as lossless PNG rather than held in
memory, so a long or high-resolution video does not depend on free RAM.

### 🖼️ Export Frames (tab)

Writes a video's frames out as still images, with each frame named after its
index in the source so an export can always be traced back.

**Range** — pick one of:

| Mode | Exports |
|------|---------|
| Whole video | every frame |
| From a start time | from *start* to the end |
| Up to an end time | from the beginning to *end* |
| Between two times | *start* to *end* |

An end past the end of the clip is clamped rather than refused. A start past it
is an error, as is an empty range.

**Keep one frame out of N** thins a long export out: a 10-minute 30 fps clip is
18 000 stills at N=1.

**Output** — PNG (lossless) or JPEG with a quality slider, and optionally a zip
of the whole set for a single download. The gallery previews the first 60; the
folder and the zip hold them all.

**Upscaling** — tick *Upscale each frame* to run every still through the same
Real-ESRGAN path as the video upscaler, at any factor from 0.5× (downscale) to
8×. 2× and 4× go straight through the AI model; other factors are hybrid.

### Command line

Reversing and frame export both work without the browser:

```bash
# Reverse one video
python app.py --reverse clip.webm

# A batch, keeping the frames, with the original and reversed joined
python app.py --reverse a.mp4 b.webm --keep-frames --video-loop ab --output-dir out/

# Every frame of a clip as PNG
python app.py --export-frames clip.webm

# One frame in 3 between 10s and 25s, as JPEG, upscaled 2x, zipped
python app.py --export-frames clip.mp4 --start 10 --end 25 --every 3 \
    --format jpg --upscale 2 --zip --output-dir frames/

# Start the web UI on another port, without opening a browser
python app.py --port 8000 --no-browser
```

The range mode follows from which bounds you pass: `--start` alone means "from
there to the end", `--end` alone means "up to there", both means "between",
neither means the whole video.

`python app.py --help` lists every option. The exit code is non-zero if any
input failed.

### 📁 Output folder

The **Processing**, **Export Frames** and **Reverse** tabs each have an *Output
folder* box. Leave it empty and results go where they always did, under the
system temp folder:

| Action | Default |
|--------|---------|
| Upscale | `%TEMP%\video_upscaler_outputs\` |
| Export frames | `%TEMP%\video_frame_exports\<video name>\` |
| Reverse | `%TEMP%\video_reverser_outputs\` |
| Model weights | `%LOCALAPPDATA%\video-upscaler-pro\models\` |

Windows cleans `%TEMP%` out on its own, so set a folder for anything you want
to keep. The field accepts `~/Videos`, `%USERPROFILE%\Videos`, a path in
quotes, and relative paths (resolved against where the app was started). The
folder is created if missing, and checked for writability *before* a long job
starts rather than failing partway through; if it cannot be used, the tab says
why and nothing runs.

The destination is printed in the status box after each run, and the CLI has
`--output-dir` for `--reverse` and `--export-frames`.

Gradio only serves files from the working directory and the system temp
folder, so a folder outside those is added to its allowed paths when you pick
it. That folder is then readable by anything that can reach the app, which is
why the UI binds to `127.0.0.1` (this machine only). `--host 0.0.0.0` opens it
to your network; the app warns when you do.

### 🎬 Frame interpolation (RIFE)

Enable **FPS Interpolation** in the Processing tab to raise a clip's frame
rate. Two options:

| Model | Does |
|-------|------|
| RIFE (AI, recommended) | estimates motion between frames and synthesises the ones in between |
| Cross-fade | blends the two neighbours, no motion estimation |

RIFE has no pip package and is not in basicsr, so the architecture is defined
in `src/models/rife_arch.py` and only the weights (21 MB) are downloaded. They
are read from **safetensors**, not the published `.pkl` inside a zip:
`torch.load` on a downloaded pickle executes whatever the file contains.

Every layer was derived from the checkpoint's own tensor names and shapes, and
the weights load with `strict=True` — a mismatch raises rather than leaving
layers randomly initialised, which is how a flow network ends up running on
noise and still returning plausible-looking frames.

**Measured.** Dropping every other frame of a clip with steady motion and
reconstructing it, against ground truth:

| Method | PSNR |
|--------|------|
| RIFE | **33.86 dB** |
| Cross-fade | 25.95 dB |
| Optical flow (what this used to do) | 22.79 dB |

Note the third row: the previous implementation was worse than simply blending
the two frames. If the weights cannot be fetched it is still used as a last
resort, and says so in the log.

### 📚 Batch queue

The Processing tab takes a **Batch queue**: drop several videos in and they are
upscaled one after another. When the queue is not empty the single video input
above it is ignored, since filling the queue is the more explicit choice.

The model is loaded **once** for the whole batch rather than per file, which is
most of the saving on a long queue.

One bad file does not end the run: it is reported on its own line and the queue
carries on. The status box summarises every file, the Comparison tab and the
player show the last one that succeeded, and **Stop** ends the whole queue, not
just the file in flight.

### ♻️ Resuming an interrupted upscale

A long upscale that is stopped, crashes, or loses power used to start again
from frame zero. With **Resume an interrupted run** ticked (the default), the
output is written in segments of 200 frames and joined at the end; a rerun
with the same inputs keeps the finished segments and carries on from the first
frame that is missing.

Segments rather than a cache of individual frames: at 4K a PNG is 8-25 MB, so
caching a ten-minute clip frame by frame would need hundreds of gigabytes. The
segments are encoded with the same codec as the final file, so they cost about
what the output costs, and joining them copies the streams rather than
re-encoding.

A job is identified by what would change its result — the input file with its
size and mtime, the scale, the model, the codec and the quality. Change any of
them and it is a new job with its own segments, so edited inputs or different
settings never resume onto mismatched work. A segment that was being written
when the process died is discarded rather than trusted.

Partial work lives in `.v2up_resume/` beside the output and is removed once
the final file exists. Resume needs ffmpeg to join the segments; without it
the app says so and writes in one pass as before.

Measured on a 125-frame clip with audio: interrupted at 3.3s, rerun finished
in 13.8s against 19.0s from scratch, resuming at frame 20, producing the same
125 frames with the audio intact.

### ⏹️ Stopping a job

Each of the Processing, Export Frames and Reverse tabs has a **Stop** button.

Gradio can drop the event it is waiting on, but it cannot kill the thread
already doing the work, so stopping is cooperative: the loops check a flag
between frames and raise. The button does both — drops the event so the
spinner stops, and sets the flag so the GPU actually stops.

It takes effect at the next frame boundary, so the delay is one frame: a
fraction of a second at SD, a few seconds for a 4K frame with many tiles.

What already exists is kept. A stopped frame export leaves the stills it had
written and tells you how many; a stopped upscale discards the partial video,
since half a file is of no use.

### 💾 Output codec

Under **Output** in the Processing tab:

| Codec | Use |
|-------|-----|
| `h264` (default) | plays everywhere |
| `h265` | roughly half the size at the same quality, less universally supported |
| `mp4v` | legacy fallback, only when ffmpeg is unavailable |

`mp4v` (MPEG-4 Part 2) used to be the only option and is a poor one at high
resolutions — no modern rate control, large files, visible blocking. On a short
640×480 test clip h264 came out 1.8× smaller than mp4v; the gap widens sharply
at 4K.

**CRF** sets quality for the ffmpeg encoders: lower is better and bigger, 18 is
near-transparent, 23 is the usual default, above ~28 artefacts show. It does
nothing for `mp4v`, so the slider hides itself.

Frames are piped straight to ffmpeg rather than written through OpenCV, and odd
output dimensions — which an arbitrary scale factor easily produces — are
rounded down to even, since H.264/H.265 need that for `yuv420p`.

### 📊 Quality metrics

Tick **Measure quality after processing** in the Processing tab; the results
land in the **Comparison** tab, which also now shows the source and the result
side by side.

| Metric | Means |
|--------|-------|
| PSNR / SSIM (round-trip) | the result scaled back down, compared with the source. Higher means the upscale stayed faithful rather than inventing detail. |
| PSNR vs plain resize | distance from simply enlarging with Lanczos. Near-identical values mean the model did little a resize would not have. |
| Temporal stability | flicker between consecutive frames with motion removed by optical flow. Lower is steadier. |

There is no ground truth at the higher resolution for real footage, so these
are round-trip and relative measures, not absolute scores. Metrics are sampled
over 24 frames, not the whole video, and are off by default because the optical
flow pass costs real time.

The stability figure is reported as an absolute difference rather than a
percentage when the source barely moves: a near-static clip has a baseline
around 0.003, against which any result becomes a meaningless four-digit
percentage.

## ⚙️ Configuration

### Advanced Settings

Access the "Advanced Settings" tab in the application to configure:

- **Temporal Coherence**: Choose method for reducing flickering
- **Processing Device**: Force GPU or CPU mode
- **Batch Size**: Adjust memory usage vs speed
- **Output Format**: Choose video codec and quality

### Configuration File

Create a `config.yaml` file in the project root for persistent settings:

```yaml
# Video Upscaler Pro Configuration
default_model: "realesrgan"
default_scale: 4
device: "auto"  # auto, cuda, cpu
batch_size: 8
use_fp16: true
output_format: "mp4"
output_quality: 18  # CRF value
```

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solution**:
- Reduce scale factor (try 2x instead of 4x)
- Close other GPU applications
- Reduce batch size in Advanced Settings
- Use CPU mode (slower but works)

#### "ModuleNotFoundError"
**Solution**:
- Get a full report of what is and is not installed:
  `venv\Scripts\python.exe scripts\verify_install.py`
- Run the installer again: `install.bat` or `./install.sh`

#### "sm_120 is not compatible with the current PyTorch installation"
PyTorch was built for older GPUs than the one you have - typically an RTX 50xx
with `cu124` wheels. Re-run `install.bat`: it picks the CUDA build from the
GPU's compute capability.

#### "InvalidVersion: Invalid version: '0.dev0'" during installation
The venv was built with a pre-release Python such as 3.12.0b3, whose `re` module
breaks modern pip. Re-run `install.bat`: it skips pre-releases and prints which
interpreter it chose.

#### Real-ESRGAN unavailable / upscaling falls back to interpolation
`import basicsr` fails because basicsr 1.4.2 imports
`torchvision.transforms.functional_tensor`, which torchvision removed in 0.17.
Run `venv\Scripts\python.exe scripts\patch_basicsr.py` to repoint it at the
public `torchvision.transforms.functional`.

#### Application won't start
**Solution**:
- Check if port 7860 is already in use
- Review logs in the terminal
- Ensure all dependencies installed: `pip list`

#### Slow processing on CPU
**Solution**:
- This is normal - CPU is 10-20x slower than GPU
- Consider using a cloud GPU service
- Process shorter videos or lower resolution

### Getting Help

- 📖 Check the [Wiki](https://github.com/mikecastrodemaria/v2up-video-upscaler-pro/wiki)
- 🐛 [Report bugs](https://github.com/mikecastrodemaria/v2up-video-upscaler-pro/issues)
- 💬 [Join discussions](https://github.com/mikecastrodemaria/v2up-video-upscaler-pro/discussions)

## 🏗️ Project Structure

```
video-upscaler-pro/
├── app.py                  # Main entry point
├── requirements.txt        # Python dependencies
├── install.bat             # Windows installer
├── install.sh             # Linux/macOS installer
├── start.bat              # Windows start script
├── start.sh               # Linux/macOS start script
├── src/
│   ├── models/            # AI model wrappers
│   ├── processors/        # Video processing pipeline
│   ├── ui/                # Gradio interface
│   │   └── gradio_app.py  # Main UI
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── benchmarks/            # Performance benchmarks
```

## 🔬 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/mikecastrodemaria/v2up-video-upscaler-pro.git
cd v2up-video-upscaler-pro

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_video_processor.py

# Run with coverage
pytest --cov=src
```

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting

```bash
# Format code
black src/

# Check linting
flake8 src/
```

## 📊 Benchmarks

Performance benchmarks on different hardware:

| Hardware | Video (1080p→4K) | Time | Model |
|----------|------------------|------|-------|
| RTX 4090 | 1 minute | 45 sec | Real-ESRGAN |
| RTX 3060 | 1 minute | 4 min | Real-ESRGAN |
| RTX 2060 | 1 minute | 6 min | Real-ESRGAN |
| CPU (i9) | 1 minute | 45 min | Real-ESRGAN |

*See [benchmarks/](benchmarks/) for detailed results.*

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- 🎨 UI/UX improvements
- 🧠 New AI model integrations
- 🐛 Bug fixes
- 📖 Documentation
- 🌍 Translations

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon amazing open source work:

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- [SwinIR](https://github.com/JingyunLiang/SwinIR) by JingyunLiang
- [RIFE](https://github.com/hzwer/Practical-RIFE) by hzwer
- [Gradio](https://gradio.app/) by Gradio team
- [PyTorch](https://pytorch.org/) by Meta AI

## 🌟 Star History

If you find this project useful, please consider giving it a star!

## 📧 Contact

- GitHub: [@mikecastrodemaria](https://github.com/mikecastrodemaria)
- Email: your.email@example.com

---

<p align="center">
  Made with ❤️ by the open source community
</p>
