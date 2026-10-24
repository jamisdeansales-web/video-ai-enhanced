"""
Pick the PyTorch wheel index that matches the installed NVIDIA GPU.

Prints one of:
  cu128  - CUDA 12.8 wheels (needed by Blackwell / RTX 50xx, sm_120)
  cu124  - CUDA 12.4 wheels (everything up to Hopper, sm_90)
  cpu    - no usable NVIDIA GPU found

The cu124 wheels only ship kernels up to sm_90, so an RTX 50xx installs fine and
then refuses to run with "sm_120 is not compatible with the current PyTorch
installation". Picking the index from the driver avoids that.
"""

import re
import subprocess
import sys


def nvidia_smi(args):
    try:
        out = subprocess.run(
            ["nvidia-smi"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def main():
    names = nvidia_smi(["--query-gpu=name", "--format=csv,noheader"])
    if not names or not names.strip():
        print("cpu")
        return

    name = names.strip().splitlines()[0]

    # Compute capability is the reliable signal, but older drivers do not expose
    # it through nvidia-smi, so fall back to the marketing name.
    caps = nvidia_smi(["--query-gpu=compute_cap", "--format=csv,noheader"])
    if caps and caps.strip():
        try:
            major = int(caps.strip().splitlines()[0].split(".")[0])
        except ValueError:
            major = 0
        if major >= 10:
            print("cu128")
            return
        if major > 0:
            print("cu124")
            return

    # RTX 50xx and the Blackwell datacenter parts need cu128.
    if re.search(r"RTX\s*(50\d\d)", name) or re.search(r"\b(B100|B200|GB200)\b", name):
        print("cu128")
        return

    print("cu124")


if __name__ == "__main__":
    main()
