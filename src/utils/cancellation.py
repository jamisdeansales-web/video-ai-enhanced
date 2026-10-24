"""
Cooperative cancellation for long jobs.

Gradio can drop an event, but it cannot kill the thread already running the
work: a stop button that only cancels the event leaves the GPU busy and the
output file half written. So the work has to agree to stop, which means the
loops check a flag between frames and raise.

One flag for the whole application rather than one per job, because the UI
runs a single job at a time. Each handler calls reset() before it starts, so a
stop left over from a previous run cannot abort the next one.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_stop = threading.Event()


class OperationCancelled(Exception):
    """Raised inside a processing loop when a stop has been requested."""


def request_stop():
    """Ask the running job to stop at its next checkpoint."""
    logger.info("Stop requested")
    _stop.set()


def reset():
    """Clear any pending stop. Called when a new job starts."""
    _stop.clear()


def is_stopped() -> bool:
    """Whether a stop has been requested."""
    return _stop.is_set()


def check():
    """
    Raise OperationCancelled if a stop has been requested.

    Call this between units of work - per frame, per window - never inside a
    tight inner loop, so stopping stays responsive without costing throughput.
    """
    if _stop.is_set():
        raise OperationCancelled("Stopped at the user's request")
