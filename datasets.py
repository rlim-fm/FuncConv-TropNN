import numpy as np
import torch
import os
import json

rng = np.random.default_rng(42)

def topksubset(k):
    return lambda x: torch.sum(torch.topk(x, k, dim=-1).values, dim=-1)
