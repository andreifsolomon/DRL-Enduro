import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from collections import OrderedDict

import numpy as np
import time
import copy


class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

