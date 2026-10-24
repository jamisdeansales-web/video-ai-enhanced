"""
Validate that the interpreter running this script can build the project venv.

install.bat / install.sh probe several interpreters with this script and keep the
first one that exits 0. Printing a reason on failure keeps the installer output
actionable instead of a bare "Python not found".

Checks, in order:
  - CPython 3.10 - 3.12 (3.13+ has no numpy<2 wheels, which basicsr needs)
  - a final release, not an alpha/beta/rc (3.12.0b3 ships a broken `re` engine
    that makes any modern pip crash with InvalidVersion)
  - a native Windows/Linux/macOS build, not MSYS2/MinGW (cannot install the
    official torch wheels)
  - pip and venv importable
"""

import sys
import sysconfig

MIN = (3, 10)
MAX = (3, 12)


def fail(reason):
    print(reason)
    raise SystemExit(1)


def main():
    v = sys.version_info

    if sys.implementation.name != "cpython":
        fail("not CPython (%s)" % sys.implementation.name)

    if v[:2] < MIN or v[:2] > MAX:
        fail(
            "Python %d.%d.%d is outside the supported range %d.%d - %d.%d"
            % (v[0], v[1], v[2], MIN[0], MIN[1], MAX[0], MAX[1])
        )

    if v.releaselevel != "final":
        fail(
            "Python %d.%d.%d%s%d is a pre-release; pip is broken on it"
            % (v[0], v[1], v[2], v.releaselevel[0], v.serial)
        )

    platform = sysconfig.get_platform()
    if "mingw" in platform or "msys" in platform:
        fail("%s is an MSYS2/MinGW build; it cannot install the torch wheels" % platform)

    try:
        import pip  # noqa: F401
    except ImportError:
        fail("pip is not installed for this interpreter")

    try:
        import venv  # noqa: F401
    except ImportError:
        fail("the venv module is not installed for this interpreter")

    print("Python %d.%d.%d (%s) OK" % (v[0], v[1], v[2], platform))


if __name__ == "__main__":
    main()
