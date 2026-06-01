"""Determinism setup for reproducible forward passes.

The whole product depends on being able to distinguish "the candidate's
computation actually changed" from "PyTorch happened to sum atomics in a
different order this run." This module owns that line.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_deterministic(seed: int = 0) -> None:
    """Configure PyTorch + numpy + python for the most-deterministic backend available.

    Notes:
    - On CPU + fp32 this gets very close to bit-exact across runs.
    - On CUDA, some ops have no deterministic kernel; we set the flag anyway and
      surface a warning at compare time when nondeterminism appears in the noise floor.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
