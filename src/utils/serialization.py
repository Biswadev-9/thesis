"""Checkpoint loading compatibility for torch >= 2.6.

``torch.load`` changed its ``weights_only`` default to ``True`` in torch 2.6, so it now
refuses to unpickle arbitrary Python objects. Lightning stores a `LightningModule`'s
``save_hyperparameters`` payload inside the checkpoint, and Hydra hands configs over as
``omegaconf`` containers - so a model taking a list-valued argument (channel widths,
hidden dimensions, class names) silently produces a checkpoint that cannot be reloaded.

The failure surfaces late and confusingly: training completes, the checkpoint is written,
and only the reload for the test pass fails.

Modules in this project keep non-tensor callables out of ``hparams`` and coerce
containers to plain Python. This module is the second line of defence, allow-listing the
handful of types that can still legitimately appear so a stray config value cannot
invalidate a long training run.
"""

from typing import List

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_REGISTERED = False


def allow_project_checkpoint_globals() -> List[str]:
    """Allow-list the non-tensor types this project's own checkpoints may contain.

    Only types written by our own training runs are listed. This does not make loading
    untrusted checkpoints safe, and it is not a substitute for keeping ``hparams`` free
    of callables.

    :return: Names of the types registered, empty if already registered.
    """
    global _REGISTERED
    if _REGISTERED:
        return []

    import collections
    import typing

    candidates = []

    # Lightning stores loop and callback bookkeeping in these.
    candidates.extend([collections.defaultdict, collections.OrderedDict, collections.Counter])

    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode, ValueNode

        candidates.extend(
            [ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode, ValueNode]
        )
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass

    # An omegaconf container's pickle payload carries its element-type annotation, which
    # for an untyped list is `typing.Any`, plus the plain containers below.
    candidates.append(typing.Any)
    candidates.extend([list, dict, tuple, set, int, float, bool, str, type(None)])

    try:
        torch.serialization.add_safe_globals(candidates)
    except AttributeError:  # pragma: no cover - torch < 2.6 has no such registry
        return []

    _REGISTERED = True
    names = [getattr(item, "__name__", str(item)) for item in candidates]
    log.debug(f"Registered {len(names)} safe globals for checkpoint loading")
    return names
