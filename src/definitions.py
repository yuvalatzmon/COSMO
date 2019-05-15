"""
common definitions and commandline arguments for zero_shot_src

Author: Yuval Atzmon

This file was taken and modified from LAGO project. Therefore, It may have some redundant definitions

"""

import argparse
import abc  # support for abstract classes (in order to define an API)

class ZSLData(abc.ABC):
    """ Defines the class API for getting ZSL data in this framework
    """

    @abc.abstractmethod
    def get_data(self):
        """
        load the data, and return a dictionary with the following keys:
        'X_train': input features train matrix. shape=[n_samples, n_features]
        'Y_train': train labels vector. shape=[n_samples, ]
        'X_val': input features validation (or test) matrix. shape=[n_samples, n_features]
        'Y_val': validation (or test) labels vector. shape=[n_samples, ]
        'df_class_descriptions_by_attributes': a dataframe of class description
            by attributes for all classes (train&val).
            shape=[n_classes, n_attributes]
            rows index = class ids
            column index = attributes names
        'attributes_name': simply df_class_descriptions_by_attributes.columns
        attributes naming format is: <group_name>::<attribute_name>, e.g.:
                                     shape::small
                                     shape::round
                                     head_color::red
                                     head_color::orange


        :return: dict()
        """

def define_common_commandline_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--transfer_task", type=str, default="GZSL",
                        help="\in ['ZSL', 'GZSL', 'FSL'] ")
    parser.add_argument("--use_trainval_set", type=int, default=0,
                        help="Use trainval (train + val) set for training")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset dir")
    parser.add_argument("--repeat", type=int, default=0,
                        help='Repetition id. This sets the seed.')

    # Below is kept for backward compatibility with load_xian_data.py
    parser.add_argument("--use_xian_normed_class_description", type=int, default=0,
                        help="Use Xian (CVPR 2017) class description. This is a "
                             "L2 normalized version of the mean attribute values"
                             "that are provided with the datasets. "
                             "This can **not** be used with LAGO.")
    parser.add_argument("--sort_attr_by_names", type=int, default=0,
                        help="If this flag is set, then we sort attributes by "
                             "names. The underlying assumtion is that the naming"
                             " convention is 'group_name::attribute_name'. "
                             "Therefore enabling this sort will cluster together"
                             "attributes from the same group. This is needed"
                             "because LAGO with Semantic groups requires that "
                             "kind of name clustering.")

    _args, unknown_args = parser.parse_known_args()
    return _args, unknown_args


def parse_common_commandline_args(_args):
    # Add arguments, for better readability:
    #  (1) Using trainval set, means that we on test stage.
    vars(_args)['is_test'] = _args.use_trainval_set
    #  (2) NOT using trainval set, means that we on development stage.
    vars(_args)['is_dev'] = not _args.is_test

    # Default computed values
    vars(_args)['seed'] = _args.repeat
    vars(_args)['seen_val_seed'] = _args.repeat + 1001


    return _args

def get_common_commandline_args():
    _args, unknown_args = define_common_commandline_args()
    _args = parse_common_commandline_args(_args)
    return _args, unknown_args

common_args, _ = get_common_commandline_args()
