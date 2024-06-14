import copy
import itertools
import json
import os


def cartesian_exp_group(exp_config, remove_none=False):
    """Cartesian experiment config.

    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.

    Parameters
    ----------
    exp_config : str
        Dictionary with the experiment Configuration

    Returns
    -------
    exp_list: str
        A list of experiments, each defines a single set of hyper-parameters
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), values)) for values in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        # remove hparams with None
        if remove_none:
            to_remove = []
            for k, v in exp_dict.items():
                if v is None:
                    to_remove += [k]
            for k in to_remove:
                del exp_dict[k]
        exp_list += [exp_dict]

    return exp_list


def save_json(fname, data, makedirs=True):
    """Save data into a json file.

    Parameters
    ----------
    fname : str
        Name of the json file
    data : [type]
        Data to save into the json file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # turn fname to string in case it is a Path object
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


EXP_GROUPS = {}
EXP_GROUPS['box_data'] = {"dataset": {'name': 'box_data',
                                      'transform': 'rgb_normalize'},
                          "model": "fcn_resnet_safpn_s16",
                          "batch_size": [1],
                          "max_epoch": [80],
                          'dataset_size': {'train': 'all', 'val': 'all'},
                          'optimizer': ['adam'],
                          'lr': [1e-5]
                          }

EXP_GROUPS = {k: cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}
