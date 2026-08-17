"""Crash-safe file writes.

Every artefact this study reports is written by a process that can be killed at any
moment: Kaggle stops a session after twelve hours, and the pipeline's own time cap is a
best-effort guess at when that will happen. A plain ``open(path, "w")`` truncates the
target before the new bytes land, so a kill in that window leaves a real filename holding
half a file - and the pipeline's completion marker is checked by *existence*, so a
half-written marker used to count as a finished stage.

``os.replace`` is atomic on POSIX and on NTFS, so a reader sees either the whole previous
file or the whole new one and never a truncated one. The temporary file is written beside
the target rather than in the system temp directory, because a rename is only atomic
within one filesystem.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Union

__all__ = ["atomic_write_text", "atomic_write_json", "atomic_write"]


def atomic_write(path: Union[str, Path], write: Callable[[Path], Any]) -> Path:
    """Produce a file through a temporary that is renamed into place.

    :param path: Final destination.
    :param write: Callable given the temporary path; it must write the whole file.
    :return: The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # mkstemp rather than a fixed suffix: two processes writing the same artefact must not
    # share a temporary, or one truncates the other's half-written file.
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)

    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    """Write text so a reader never sees a partial file.

    :param path: Final destination.
    :param text: Contents.
    :param encoding: Text encoding.
    :return: The destination path.
    """

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            # fsync before the rename: without it a machine-level crash can leave the
            # renamed name pointing at unwritten blocks.
            os.fsync(stream.fileno())

    return atomic_write(path, write)


def atomic_write_json(path: Union[str, Path], payload: Any, **dumps: Any) -> Path:
    """Serialise JSON so a reader never sees a truncated document.

    :param path: Final destination.
    :param payload: JSON-serialisable object.
    :param dumps: Extra keyword arguments for :func:`json.dumps`.
    :return: The destination path.
    """
    dumps.setdefault("indent", 2)
    return atomic_write_text(path, json.dumps(payload, **dumps))
