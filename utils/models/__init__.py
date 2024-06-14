from . import base_networks
from . import RPHCounter


def get_model(exp_dict=None, train_set=None):
    model = RPHCounter.RPHCounter(exp_dict, train_set=train_set)
    return model
