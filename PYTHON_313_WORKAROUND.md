# Why Python 3.13+ is not supported

`install.bat` and `install.sh` skip Python 3.13 and newer on purpose. The blocker
is not Python itself but the dependency chain below Real-ESRGAN:

- `realesrgan` 0.3.0 requires `basicsr`.
- `basicsr` 1.4.2 (its last release, from 2022) is not compatible with NumPy 2.
- `numpy < 2` publishes no wheels for Python 3.13, so pip would have to build
  NumPy 1.x from source against a CPython it predates.

Use Python 3.10, 3.11 or 3.12. The installer picks one automatically and tells
you which interpreters it rejected and why.

## If you want to try 3.13 anyway

Nothing stops you from building the venv by hand, but Real-ESRGAN will not be
available. Raise the upper bound in `scripts/check_python.py`:

```python
MAX = (3, 13)
```

then run `install.bat`. Installing `basicsr` will fail, the installer will warn
you, and the app will start with interpolation-based upscaling as a fallback
(`src/models/realesrgan_model.py` catches the missing import).

## Known failure modes this replaces

Two problems used to be blamed on Python 3.13 but are unrelated to the version:

**`InvalidVersion: Invalid version: '0.dev0'`** — the venv was built with a
*pre-release* CPython, for example 3.12.0b3. Its `re` module mis-compiles the
version pattern that `packaging` uses, so every modern pip crashes on startup.
`scripts/check_python.py` now rejects any interpreter whose `releaselevel` is
not `final`.

**`No module named 'torchvision.transforms.functional_tensor'`** — `basicsr`
imports a private torchvision module that was removed in torchvision 0.17, so
`import basicsr` fails on any torchvision this project can use, and Real-ESRGAN
goes silently missing. `scripts/patch_basicsr.py` rewrites that import to the
public `torchvision.transforms.functional`, and the installers run it
automatically.
