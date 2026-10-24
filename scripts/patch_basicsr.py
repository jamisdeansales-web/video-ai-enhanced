"""
Repair basicsr's import of torchvision.transforms.functional_tensor.

basicsr 1.4.2 (the last release, and what realesrgan depends on) imports
`rgb_to_grayscale` from `torchvision.transforms.functional_tensor`, a private
module torchvision removed in 0.17. Every `import basicsr` therefore raises
ModuleNotFoundError on any torchvision this project can use, which makes
Real-ESRGAN silently unavailable: src/models/realesrgan_model.py catches the
ImportError and falls back to interpolation.

The public `torchvision.transforms.functional` re-exports the same function, so
rewriting the import in the installed package is enough. Idempotent.
"""

import importlib.util
import sys
from pathlib import Path

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def basicsr_root():
    """Locate the installed basicsr without importing it.

    `import basicsr` cannot be used here: it runs basicsr/__init__.py, which is
    what raises the ModuleNotFoundError this script exists to fix.
    """
    try:
        spec = importlib.util.find_spec("basicsr")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def main():
    root = basicsr_root()
    if root is None:
        print("basicsr is not installed; nothing to patch")
        return 0

    patched = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if OLD not in text:
            continue
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
        patched.append(path.relative_to(root).as_posix())

    for rel in patched:
        print("patched basicsr/%s" % rel)

    # Confirm the import actually works now, so a silent fallback to
    # interpolation cannot be mistaken for a successful install.
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401
    except ImportError as exc:
        print("basicsr still does not import: %s" % exc)
        return 1

    if not patched:
        print("basicsr needs no patch; Real-ESRGAN imports cleanly")
    else:
        print("Real-ESRGAN imports cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
