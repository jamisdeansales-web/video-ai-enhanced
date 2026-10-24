"""
Image I/O that survives non-ASCII paths.

cv2.imwrite and cv2.imread pass the filename to the C++ layer as bytes in the
system locale, so on Windows any path containing a character outside the active
code page fails. A French filename is enough.

It fails in two different ways, and the quieter one is the dangerous one:

    cv2.imwrite(r"C:\\tmp\\Vidéos\\frame.png", img)   -> False, nothing written
    cv2.imwrite(r"C:\\tmp\\vidéo.png", img)           -> True,  nothing written

The second returns success and creates no file, so a caller that trusts the
return value reports a clean run over frames that do not exist.

Encoding to a buffer and letting Python write the bytes avoids the issue
entirely: Python's open() handles Unicode paths natively.
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def imwrite(path: str, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """
    Write an image, whatever the path contains.

    Args:
        path: Destination. The extension selects the format, as with cv2.
        image: BGR image.
        params: cv2 encode parameters, e.g. [cv2.IMWRITE_JPEG_QUALITY, 92].

    Returns:
        True on success. Unlike cv2.imwrite, a True here means the bytes
        reached the disk.
    """
    extension = os.path.splitext(path)[1]
    if not extension:
        logger.error("No file extension in %s, cannot pick an image format", path)
        return False

    ok, buffer = cv2.imencode(extension, image, params or [])
    if not ok:
        logger.error("Could not encode the image as %s", extension)
        return False

    try:
        with open(path, 'wb') as handle:
            handle.write(buffer.tobytes())
    except OSError as exc:
        logger.error("Could not write %s: %s", path, exc)
        return False

    return True


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Read an image, whatever the path contains.

    Returns:
        The image, or None if it could not be read - same contract as
        cv2.imread.
    """
    try:
        with open(path, 'rb') as handle:
            data = np.frombuffer(handle.read(), dtype=np.uint8)
    except OSError as exc:
        logger.error("Could not read %s: %s", path, exc)
        return None

    if data.size == 0:
        logger.error("%s is empty", path)
        return None

    return cv2.imdecode(data, flags)


def check_writable(directory: str) -> Optional[str]:
    """
    Check that a directory can be created and written to.

    Returns None when it is usable, otherwise a message explaining what is
    wrong. Worth calling before a long export, so a thousand-frame job does not
    fail on the last one.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        return "cannot create %s: %s" % (directory, exc)

    probe = os.path.join(directory, '.v2up_write_test')
    try:
        with open(probe, 'wb') as handle:
            handle.write(b'0')
        os.remove(probe)
    except OSError as exc:
        return "cannot write into %s: %s" % (directory, exc)

    return None
