from copy import deepcopy
import numpy as np

"""
This module defines the ranges of hyper-params to search upon
"""

# Define the high level configuration structure
_all_config = {}
_all_config['AWA1'] = {}
_all_config['SUN'] = {}
_all_config['CUB'] = {}
_all_config['FLO'] = {}

def get(dataset_name, combiner, metric_name, all_config=_all_config):
    assert(dataset_name in all_config)
    dataset_cfgs = all_config[dataset_name]
    if metric_name in dataset_cfgs:
        cfg = dataset_cfgs[metric_name].get(combiner,
                                         dataset_cfgs[metric_name]['default'])
    else:
        cfg = dataset_cfgs.get(combiner, dataset_cfgs['default'])

    # sanity check
    assert (len(cfg['anomaly_detector_threshold']) == 1)
    return deepcopy(cfg)

def cast_to_lists(hyper_params):
    """ Put in a list (with len=1), if given as individual value """
    hyper_params_as_lists = {}

    for k,v in hyper_params.items():
        if not (isinstance(v, list) or isinstance(v, np.ndarray)) :
            v = [v,]
        hyper_params_as_lists[k] = list(v)
    return hyper_params_as_lists

def _individual_cfg(hyper_params={}, anomaly_detector_threshold={}):
    assert( len(anomaly_detector_threshold) == 1 )

    hyper_params_as_lists = cast_to_lists(hyper_params)
    threshold_as_list = cast_to_lists(anomaly_detector_threshold)

    return dict(hyper_params=hyper_params_as_lists,
                anomaly_detector_threshold=threshold_as_list)



# Here we define defaults configurations
CUB_default = _individual_cfg(hyper_params=
                             dict(
                                  T_cond=[0.1, 0.3, 1, 3],
                                  ),
                              anomaly_detector_threshold=
                             dict(threshold=np.arange(-2.5, 2.5, 0.1))
                              )
_all_config['CUB']['default'] = deepcopy(CUB_default)


SUN_default = _individual_cfg(hyper_params=
                             dict(
                                  T_cond=[0.1, 0.3, 1, 3, 10],
                                  ),
                              anomaly_detector_threshold=
                             dict(threshold=np.arange(-2.5, 20, 0.2))
                              )
_all_config['SUN']['default'] = deepcopy(SUN_default)

_all_config['SUN']['Confidence Based Gater: T = (3,)'] = {}
_all_config['SUN']['Confidence Based Gater: T = (3,)']['adaptive_smoothing'] = \
_individual_cfg(hyper_params=
                             dict(
                                  T_cond=[0.1, 0.3, 1, 3, 10],
                                  ),
                              anomaly_detector_threshold=
                             dict(threshold=np.arange(0, 50, 0.2))
                              )
_all_config['SUN']['Confidence Based Gater: T = (3,)']['default'] = deepcopy(
    SUN_default)

_all_config['SUN']['const_smoothing'] = deepcopy(SUN_default)
_all_config['SUN']['const_smoothing']['hyper_params']['T_cond'] = [0.1, 0.3, 1, 3, 10]
_all_config['SUN']['const_smoothing']['anomaly_detector_threshold']['threshold'] = \
    np.arange(-2.5, 20, 0.2).tolist()
_all_config['SUN']['const_smoothing']['hyper_params']['gamma'] = \
    list(np.arange(0.05, 1.001, 0.05))


AWA1_default = _individual_cfg(hyper_params=
                             dict(
                                  T_cond=[0.003, 0.01, 0.03, 0.1,],
                                  ),
                               anomaly_detector_threshold=
                             dict(threshold=np.block([
                                 np.arange(0., 0.7, 0.05),
                                 np.arange(0.7, 1.2, 0.005),
                                 np.arange(1.2, 2, 0.05),])) )

_all_config['AWA1']['default'] = deepcopy(AWA1_default)


# Note: The name "anomaly_detector_threshold" is confusing for calibrated_stacking (CS) method.
#       It is used to maintain API compatibility. In CS it is used for sweeping over gamma.
calibrated_stacking_default = _individual_cfg(anomaly_detector_threshold=
                                    dict(gamma=np.arange(-1.5, 1.5, 0.02)))

# Here we define individual configurations.
CUB_laplace = deepcopy(CUB_default)
# CUB_laplace['anomaly_detector_threshold']['threshold'] = np.arange(-0.5, 2, 0.05)

# Here we assign individual configurations to _all_config
_all_config['CUB']['adaptive_smoothing'] = deepcopy(CUB_laplace)
_all_config['CUB']['const_smoothing'] = deepcopy(CUB_laplace)
_all_config['CUB']['const_smoothing']['hyper_params']['gamma'] = \
    list(np.arange(0.05, 1.001, 0.05))
_all_config['CUB']['calibrated_stacking'] = deepcopy(calibrated_stacking_default)
_all_config['SUN']['calibrated_stacking'] = deepcopy(calibrated_stacking_default)
_all_config['AWA1']['const_smoothing'] = deepcopy(AWA1_default)
_all_config['AWA1']['const_smoothing']['hyper_params']['gamma'] = \
    list(np.arange(0.05, 1.001, 0.05))


_all_config['AWA1']['calibrated_stacking'] = deepcopy(calibrated_stacking_default)


_all_config['FLO'] = deepcopy(_all_config['CUB'])

