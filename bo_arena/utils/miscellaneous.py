"""A collection of miscellaneous utility functions."""

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: The random seed to use for all generators.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_seed(master_seed: int, index: int) -> int:
    """Derive a deterministic seed from a master seed and index.

    Uses SHA-256 hashing to ensure reproducibility independent of library versions.
    The master seed is mixed into every derivation, ensuring that different master
    seeds produce fully independent sequences.

    Args:
        master_seed: The master seed to derive from.
        index: The index used to produce a unique derived seed.

    Returns:
        A deterministic seed value derived from the master seed and index.
    """
    hash_bytes = hashlib.sha256(f"{master_seed}_{index}".encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big")
