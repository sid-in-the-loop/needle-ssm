import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return a * (2 * rand(fan_in, fan_out, **kwargs) - 1)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return std * randn(fan_in, fan_out, **kwargs)



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(2) * math.sqrt(3 / fan_in) 
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs) if shape is None else rand(*shape, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    raise NotImplementedError()