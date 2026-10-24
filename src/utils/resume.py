"""
Resume a long job after an interruption.

A 4K upscale that dies at 90% used to start again from zero. The obstacle is
the output itself: frames are encoded straight into one mp4, and a half-written
mp4 cannot be appended to.

So the output is written in segments of a few hundred frames, each a complete
little video, and concatenated at the end. A rerun with the same inputs keeps
the segments that finished and picks up from the first frame that is missing.

Segments rather than a cache of individual frames: at 4K a PNG is 8-25 MB, so
a ten-minute clip would need hundreds of gigabytes. The segments are encoded
with the same codec as the final file, so the cache costs roughly what the
output costs, and concatenation copies the streams rather than re-encoding.

A job is identified by what would change its result - the input file and its
size and mtime, the scale, the model, the codec and the quality. Change any of
them and it is a different job, with its own segments; the old ones are not
silently reused.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Frames per segment. Small enough that little is lost to an interruption,
# large enough that the per-segment encoder startup stays negligible.
DEFAULT_SEGMENT_FRAMES = 200

MANIFEST_NAME = 'manifest.json'


def job_key(input_path: str, params: Dict) -> str:
    """
    A short identifier for this job.

    Built from the input's identity and every parameter that changes the
    output, so editing the source or changing the scale starts a new job
    instead of resuming onto mismatched segments.
    """
    try:
        stat = os.stat(input_path)
        identity = f"{os.path.abspath(input_path)}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        identity = os.path.abspath(input_path)

    payload = identity + '|' + json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


class ResumableJob:
    """Tracks the segments of one job on disk."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        params: Dict,
        segment_frames: int = DEFAULT_SEGMENT_FRAMES,
        work_root: Optional[str] = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.params = params
        self.segment_frames = max(1, int(segment_frames))
        self.key = job_key(input_path, params)

        # Beside the output, so the partial work travels with the folder the
        # user chose and is obvious rather than hidden in a temp directory.
        root = work_root or os.path.join(
            os.path.dirname(os.path.abspath(output_path)), '.v2up_resume'
        )
        self.work_dir = os.path.join(root, self.key)
        self.manifest_path = os.path.join(self.work_dir, MANIFEST_NAME)

    def prepare(self) -> int:
        """
        Create the work directory and return the frame to start from.

        Only segments recorded as complete in the manifest count. A segment
        whose file is missing, or that was being written when the process
        died, is discarded: trusting a half-written segment would put a
        truncated stretch in the middle of the finished video.
        """
        os.makedirs(self.work_dir, exist_ok=True)

        manifest = self._read_manifest()
        if manifest.get('key') != self.key:
            manifest = {'key': self.key, 'segments': []}

        valid = []
        for entry in manifest.get('segments', []):
            path = os.path.join(self.work_dir, entry['file'])
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid.append(entry)
            else:
                logger.warning("Discarding incomplete segment %s", entry['file'])

        # Segments must be contiguous from the start; a gap means everything
        # after it has to be redone anyway.
        valid.sort(key=lambda e: e['index'])
        contiguous = []
        for position, entry in enumerate(valid):
            if entry['index'] != position:
                break
            contiguous.append(entry)

        self.segments = contiguous
        self._write_manifest()

        resume_from = sum(e['frames'] for e in contiguous)
        if resume_from:
            logger.info(
                "Resuming %s from frame %d (%d segments already done)",
                Path(self.input_path).name, resume_from, len(contiguous),
            )
        return resume_from

    def segment_path(self, index: int) -> str:
        return os.path.join(self.work_dir, 'seg_%05d.mp4' % index)

    def next_index(self) -> int:
        return len(self.segments)

    def record(self, index: int, frames: int):
        """Mark a segment complete, flushed immediately."""
        self.segments.append({
            'index': index,
            'file': os.path.basename(self.segment_path(index)),
            'frames': frames,
        })
        self._write_manifest()

    def _read_manifest(self) -> Dict:
        try:
            with open(self.manifest_path, encoding='utf-8') as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _write_manifest(self):
        payload = {
            'key': self.key,
            'input': os.path.abspath(self.input_path),
            'params': self.params,
            'segment_frames': self.segment_frames,
            'segments': self.segments,
        }
        tmp = self.manifest_path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, default=str)
            os.replace(tmp, self.manifest_path)
        except OSError as exc:
            logger.warning("Could not write the resume manifest: %s", exc)

    def concatenate(
        self,
        ffmpeg_bin: Optional[str],
        audio_source: Optional[str] = None,
    ) -> bool:
        """
        Join the segments into the final file.

        Video streams are copied, not re-encoded: the segments came from the
        same encoder with the same settings, so decoding them again would only
        lose quality.

        The segments carry no audio - muxing the same track into each one and
        then concatenating would repeat it - so the original's audio is added
        here, in the single pass that produces the final file.
        """
        if not self.segments:
            return False

        if not ffmpeg_bin:
            logger.error("ffmpeg is needed to join the segments")
            return False

        list_path = os.path.join(self.work_dir, 'segments.txt')
        with open(list_path, 'w', encoding='utf-8') as handle:
            for entry in self.segments:
                path = os.path.join(self.work_dir, entry['file']).replace('\\', '/')
                # The concat demuxer takes single quotes, doubled to escape.
                handle.write("file '%s'\n" % path.replace("'", "'\\''"))

        command = [
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0', '-i', list_path,
        ]
        if audio_source:
            # The trailing ? keeps this working when the source is silent.
            command += [
                '-i', audio_source,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0?', '-shortest',
            ]
        else:
            command += ['-c', 'copy']
        command += ['-y', self.output_path]

        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            logger.error(
                "Could not join the segments: %s",
                result.stderr.decode(errors='replace').strip(),
            )
            return False

        return True

    def cleanup(self):
        """Remove the segments once the final file exists."""
        shutil.rmtree(self.work_dir, ignore_errors=True)
        # Take the parent with it when this was the last job in it.
        parent = os.path.dirname(self.work_dir)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass

    def total_frames_done(self) -> int:
        return sum(e['frames'] for e in self.segments)


def find_resumable(output_path: str) -> List[str]:
    """List job keys with saved segments beside an output path."""
    root = os.path.join(os.path.dirname(os.path.abspath(output_path)), '.v2up_resume')
    if not os.path.isdir(root):
        return []
    return [name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))]
