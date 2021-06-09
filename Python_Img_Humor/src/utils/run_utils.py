import copy
import yaml


def fill_config_args(args_):
    """ helper function to fill argparse object with values
    from config files in .yml format.
    Takes in an arparse object (args),
    reads in the relevant config file,
    returns filled argparse object

     Input:
     - args_ (run config)
     Output: args_ (filled in argaprase object) """
    args_ = copy.deepcopy(args_)
    config_path = args_.config_causal_effect
    with open(config_path, 'r') as stream:
        relevant_dict = yaml.safe_load(stream)
    dataset_dict = relevant_dict['dataset']
    optimizer_dict = relevant_dict['optimizer']
    for keys in dataset_dict.keys():
        setattr(args_, keys, dataset_dict[keys])
    for keys in optimizer_dict.keys():
        setattr(args_, keys, optimizer_dict[keys])
    return args_
