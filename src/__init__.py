"""Project package.

Checkpoint-loading compatibility is registered here rather than in ``src.utils`` so it
applies however the package is entered. Under DDP, Lightning spawns worker processes that
unpickle the model class directly; a worker that reaches ``src.models.<module>`` without
importing ``src.utils`` would otherwise start with an unregistered allow-list and fail to
load the checkpoint. See ``src/utils/serialization.py``.
"""

from src.utils.serialization import allow_project_checkpoint_globals

allow_project_checkpoint_globals()
