"""Utility helpers: logger, seeding, timer."""
from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("mqttids")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


@contextmanager
def timing(msg: str):
    start = perf_counter()
    yield
    LOGGER.info("%s finished in %.2f s", msg, perf_counter() - start)