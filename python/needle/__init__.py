from . import ops
from .ops import *
from .autograd import Tensor, cpu, all_devices

from . import init
from .init import ones, zeros, zeros_like, ones_like

# Data module - import if available (may not exist in all setups)
try:
    from . import data
except ImportError:
    pass

from . import nn
from . import optim
from .backend_selection import *
