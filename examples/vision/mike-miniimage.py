import random

import numpy as np
import torch as th
from PIL.Image import LANCZOS
from maml_miniimagenet import accuracy, fast_adapt

from torch import nn
from torch import optim
from torchvision import transforms

import learn2learn as l2l

print(th.cuda.is_available())