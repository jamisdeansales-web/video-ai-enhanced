"""
Report what the venv actually has, and exit non-zero if the app cannot start.

Real-ESRGAN is optional - the app falls back to interpolation without it - so a
missing basicsr is reported as a warning. A missing core package, or a torch
that cannot drive the installed GPU, is an error.
"""

import sys

CORE = [
    ("gradio", "gradio"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("psutil", "psutil"),
    ("skimage", "scikit-image"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("yaml", "pyyaml"),
]

errors = []
warnings = []


def version_of(module):
    return getattr(module, "__version__", "?")


print("Core dependencies")
print("-" * 46)
for import_name, dist_name in CORE:
    try:
        mod = __import__(import_name)
    except ImportError as exc:
        print("  MISSING  %-18s (%s)" % (dist_name, exc))
        errors.append("%s is missing" % dist_name)
    else:
        print("  ok       %-18s %s" % (dist_name, version_of(mod)))

print()
print("PyTorch / GPU")
print("-" * 46)
try:
    import torch
except ImportError:
    print("  MISSING  torch")
    errors.append("torch is missing")
else:
    print("  torch           %s" % torch.__version__)
    print("  CUDA build      %s" % (torch.version.cuda or "none (CPU-only build)"))
    print("  CUDA available  %s" % torch.cuda.is_available())

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print("  GPU             %s (sm_%d%d)" % (name, cap[0], cap[1]))

        # A wheel that lacks kernels for this GPU installs happily and then fails
        # on the first real op, so check the arch list rather than trusting
        # is_available().
        supported = []
        for arch in torch.cuda.get_arch_list():
            if arch.startswith("sm_"):
                digits = arch[3:].rstrip("a")
                if digits.isdigit():
                    supported.append(digits)
        asked = "%d%d" % (cap[0], cap[1])
        if supported and asked not in supported:
            print("  ARCH MISMATCH   this torch supports sm_%s" % ", sm_".join(supported))
            errors.append(
                "torch %s has no kernels for sm_%s (%s); reinstall with the "
                "matching CUDA index" % (torch.__version__, asked, name)
            )
        else:
            try:
                (torch.zeros(8, device="cuda") + 1).sum().item()
            except Exception as exc:  # noqa: BLE001 - report whatever the driver says
                print("  GPU TEST FAILED %s" % exc)
                errors.append("a trivial CUDA op failed: %s" % exc)
            else:
                print("  GPU test        ok")
    else:
        warnings.append("CUDA is not available - upscaling will run on the CPU")

print()
print("Real-ESRGAN (optional)")
print("-" * 46)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
    from realesrgan import RealESRGANer  # noqa: F401
except ImportError as exc:
    print("  unavailable  %s" % exc)
    warnings.append(
        "Real-ESRGAN is unavailable (%s) - the app will fall back to "
        "interpolation-based upscaling" % exc
    )
else:
    print("  ok           basicsr + realesrgan import cleanly")

print()
print("Application imports")
print("-" * 46)
sys.path.insert(0, "src")
try:
    from ui.gradio_app import create_interface  # noqa: F401
except Exception as exc:  # noqa: BLE001 - any failure here blocks startup
    print("  FAILED       %s" % exc)
    errors.append("src/ui/gradio_app.py cannot be imported: %s" % exc)
else:
    print("  ok           ui.gradio_app.create_interface")

print()
for warning in warnings:
    print("WARNING: %s" % warning)
for error in errors:
    print("ERROR:   %s" % error)

if errors:
    print()
    print("Installation is NOT usable.")
    sys.exit(1)

print()
print("Installation looks good.")
