import itertools
import os
from copy import deepcopy, copy
from multiprocessing.pool import Pool
from pickle import PicklingError

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import src.cosmo_misc
from src import gzsl_search_configurations as configurations
from src.cosmo_misc import crossval_and_plot_sweep_results, get_predictions_of_best_cfg
from sklearn.metrics import auc

from src.utils import ml_utils
from src.utils.ml_utils import pfloor, pceil
from src.metrics import ZSL_Metrics


def GZSL_experiment(data, expert_preds, gating_model, mixture_type='adaptive_smoothing',
                    hard_gating=False, num_resample_points=100, num_pool_workers=12,
                    fs_results=None, dirname=None, show_plot=True):

    # gating module predictions
    # Note: Set to None to allow models that don't use a gating mechanism (e.g. CalibratedStacking)
    pred_gating_GZSLval, pred_gating_GZSLtest, g_name = None, None, None
    if gating_model is not None:
        pred_gating_GZSLval = gating_model.pred_GZSLval
        pred_gating_GZSLtest = gating_model.pred_GZSLtest
        g_name = gating_model.name

    cfg = configurations.get(data['dataset_name'], mixture_type, g_name)

    mixture_model = mixture_factory[mixture_type]

    fig_list = []

    ### Val set
    pred_ZS = expert_preds['ZS__GZSLval']
    pred_S = expert_preds['S__GZSLval']
    pred_gating = pred_gating_GZSLval
    Y_GZSLval = data['Y_GZSLval']

    current_model_GZSLval = mixture_model(data['seen_classes_val'],
                                     data['unseen_classes_val'],
                                     pred_ZS=pred_ZS,
                                     pred_S=pred_S,
                                     pred_gating=pred_gating,
                                     gating_model_name=g_name,
                                     hard_gating=hard_gating,
                                     mixture_type=mixture_type)
    print('Experiment name: ', current_model_GZSLval.name)

    ### Test set
    current_model_GZSLtest = mixture_model(data['seen_classes_test'],
                                     data['unseen_classes_test'],
                                     pred_ZS=expert_preds['ZS__GZSLtest'],
                                     pred_S=expert_preds['S__GZSLtest'],
                                     pred_gating=pred_gating_GZSLtest,
                                     gating_model_name=g_name,
                                     hard_gating=hard_gating,
                                     mixture_type=mixture_type)


    threshold_name = list(cfg['anomaly_detector_threshold'].keys())[0]

    # if search on 2 or more hyper params: then we start with a coarse set on all hyper params
    # and set the range for making a fine search on the threshold hyper param
    if len(list(itertools.product(*cfg['hyper_params'].values()))) >= 2:
        print('Starting with coarse hyper param search')
        # print('cfg = ', cfg)

        complete_cfg = cfg['hyper_params'].copy()
        complete_cfg.update(cfg['anomaly_detector_threshold'])
        _ = current_model_GZSLval.sweep_variables(Y=Y_GZSLval,
                                             num_pool_workers=num_pool_workers,
                                             **complete_cfg)

        best_cfg = current_model_GZSLval.df_sweep_results.loc[
        current_model_GZSLval.df_sweep_results.Acc_H.idxmax(), :]
        print(f"Best (coarse) hyper-param configuration is:\n"
              f"{best_cfg.loc[complete_cfg.keys()]}")
        print(f"Acc_H = {best_cfg.loc['Acc_H']}")

        # Setting the params for finer threshold search
        best_cfg_params = best_cfg.loc[cfg['hyper_params'].keys()].to_dict()
        df_GZSLval_sweep_best = current_model_GZSLval.df_sweep_results.query(
            ' and '.join([f'{k}=={v}' for k,v in best_cfg_params.items()]))
        th_range_resampled = src.cosmo_misc.resample_sweepkey_by_curve(
            df_GZSLval_sweep_best, threshold_name, num_resample_points,num_resample_points)

        best_complete_cfg = deepcopy(best_cfg_params)
        best_complete_cfg[threshold_name] = th_range_resampled
    else:
        # only 1 hyper param
        best_cfg_params = deepcopy(cfg['hyper_params'])
        best_complete_cfg = deepcopy(best_cfg_params)
        best_complete_cfg.update(cfg['anomaly_detector_threshold'])


    best_best_complete_cfg_as_lists = configurations.cast_to_lists(best_complete_cfg)
    print('Fine search over the threshold parameter:')

    _ = current_model_GZSLval.sweep_variables(Y_GZSLval, num_pool_workers=num_pool_workers,
                                         **best_best_complete_cfg_as_lists)

    # Sweep the threshold over all test models, in order to eval AUSUC and generate figures.
    # Note: For computing Acc_H, we will use the best model selected on GZSLval (above).
    #       The selection is performed in performed in process_and_plot_sweep_results() method

    _ = current_model_GZSLtest.sweep_variables(data['Y_GZSLtest'],
                                         num_pool_workers=num_pool_workers,
                                         **best_best_complete_cfg_as_lists)


    df_res_test, df_res_val = \
        crossval_and_plot_sweep_results(threshold_name, data, expert_preds,
                                        current_model_GZSLval, current_model_GZSLtest, fig_list,
                                        fs_results, best_complete_cfg, dirname)
    if show_plot:
        plt.draw(); plt.pause(0.001)  # source https://stackoverflow.com/a/33050617
    else:
        plt.close()

    print('Test performance:', df_res_test.iloc[0, :])


    activations_best = {}
    if not mixture_type in ['calibrated_stacking']:
        activations_val, activations_test, _, _  = get_predictions_of_best_cfg(best_complete_cfg, Y_GZSLval,
                            data['Y_GZSLtest'], current_model_GZSLval, current_model_GZSLtest)

        activations_best[current_model_GZSLtest.name] = dict(val=activations_val,
                                                       test=activations_test)


    return df_res_test, df_res_val, activations_best


class CombinerGZSL(object): # a.k.a. mixture
    """
    This is a parent class for different approached for combining S (seen) & ZS (unseen) experts decisions,
    using a gating model
    """
    def __init__(self, seen_classes, unseen_classes, pred_ZS, pred_S,
                 pred_gating, gating_model_name,
                 mixture_type, hard_gating=False):

        self._seen_classes = seen_classes
        self._unseen_classes = unseen_classes
        self.__pred_ZS = pred_ZS
        self.__pred_S = pred_S
        self._pred_gating = pred_gating
        self._gating_model_name = gating_model_name
        self._hard_gating = hard_gating
        self._combiner = mixture_type

        self.set_name()
        self.save_activations = False
        self.activations = {}
        self.df_sweep_results = None

    def set_name(self):
        gating_type = 'Soft-Gating'
        if self._hard_gating:
            gating_type = 'Hard-Gating'
        self.name = f'combiner={self._combiner}|' \
                    f'gater={self._gating_model_name}|{gating_type}'

    @property
    def pred_ZS(self):
        """ To make sure experiments are independent, access to pred_ZS is only through a copy """
        return copy(self.__pred_ZS)

    @property
    def pred_S(self):
        """ To make sure experiments are independent, access to pred_S is only through a copy """
        return copy(self.__pred_S)


    ####################################################
    """ API to implement by child class """

    def _predict(self, X):
        """ Needs to update self.pred_ZS and self.pred_S (if applicable)
        """
        raise NotImplementedError()

    def combine(self, **kwargs):
        raise NotImplementedError()
    ####################################################

    @staticmethod
    def single_iter_of_sweep_for_parallel_pool(params):
        # (ugly) adaptation of parallel pool API for multiple variable
        self, current_params_values, hp_keys, Y, zs_metrics = params
        current_params = dict(zip(hp_keys, current_params_values))

        # Combine expert predictions, using current iteration hyper-params, to generate new combined prediction
        pred = self.combine(**current_params)

        # Evaluate GZSL metrics of combined model
        metric_results = {}
        metric_results['Acc_ts'], metric_results['Acc_tr'], metric_results[
            'Acc_H'] = zs_metrics.generlized_scores(Y, pred)
        metric_results.update(current_params)

        return metric_results

    def reset_df_results(self, hp_keys):
        self.df_sweep_results = pd.DataFrame(
            columns=list(hp_keys) + 'Acc_ts,Acc_tr,Acc_H'.split(','))

    def sweep_variables(self, Y, num_pool_workers=4,
                        **hyper_params_ranges): # Dict[str, Union[List, np.array]]
        # num_pool_workers allows parallel execution

        # Sweep over an outer-product (grid search) of hyper-params ranges
        hp_keys, hp_ranges = zip(*hyper_params_ranges.items())
        self.reset_df_results(hp_keys)

        all_params = list(itertools.product(*hp_ranges))
        new_params_list = all_params.copy()

        zs_metrics = ZSL_Metrics(self._seen_classes, self._unseen_classes)
        results_list = []
        if new_params_list:
            if num_pool_workers>1:
                """ Parallel execution of model evaluations with different hyper-params """

                try:
                    with Pool(num_pool_workers) as po:  # This ensures that the processes get closed once they are done
                        pool_results = po.map(self.single_iter_of_sweep_for_parallel_pool,
                                                    ((self, current_params_values, hp_keys, Y,
                                                      zs_metrics#, progress_bar
                                                      ) for current_params_values in new_params_list))
                    results_list = pool_results
                except PicklingError:
                    print('Warning: Can''t execute in parallel due to PicklingError. '
                          'Common solution is to rerun the call that initialize this '
                          'class instance.')
                    num_pool_workers=1

            if num_pool_workers == 1:
                """ model evaluations with different hyper-params using a serial for loop """
                for current_params_values in new_params_list:
                    res = self.single_iter_of_sweep_for_parallel_pool((
                        self, current_params_values, hp_keys, Y, zs_metrics))
                    results_list.append(res)

        # aggregate results to a DataFrame
        for k, current_params_values in enumerate(all_params):
            currect_results = results_list[k]
            self.df_sweep_results = self.df_sweep_results.append(currect_results,
                                                        ignore_index=True)

        return self.df_sweep_results



    def plot_tstr_curve_cvpr(self, df_sweep_results=None,
                        x_name='Acc_tr', y_name='Acc_ts',
                        xlim=None, ylim=None,  ax=None, is_first=False, color=None):

        if df_sweep_results is None:
            df_sweep_results = self.df_sweep_results

        if ax is None:
            ax = plt.gca()

        X = 100*df_sweep_results.loc[:, x_name]
        Y = 100*df_sweep_results.loc[:, y_name]
        if xlim is None:
            xlim = (pfloor(X.min(), 1), pceil(X.max(), 1))
        if ylim is None:
            ylim = (pfloor(Y.min(), 1), pceil(Y.max(), 1))
        ax.plot(X, Y,  'o', linewidth=5, color=color)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if is_first:
            plt.xlabel(x_name)
            plt.ylabel(y_name)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    def AUSUC(self, df_sweep_results=None,
                        x_name='Acc_tr', y_name='Acc_ts'):
        """ Calculate Area-Under-Seen-Unseen-Curve (AUSUC) metric given sweep results. """

        if df_sweep_results is None:
            df_sweep_results = self.df_sweep_results


        X = df_sweep_results.loc[:, x_name]
        Y = df_sweep_results.loc[:, y_name]

        # Calc area under curve
        X_sorted_arg = np.argsort(X)
        sorted_X = np.array(X)[X_sorted_arg]
        sorted_Y = np.array(Y)[X_sorted_arg]
        leftmost_X, leftmost_Y = 0, sorted_Y[0]
        rightmost_X, rightmost_Y = sorted_X[-1], 0
        sorted_X = np.block([np.array([leftmost_X]), sorted_X, np.array([rightmost_X])])
        sorted_Y = np.block([np.array([leftmost_Y]), sorted_Y, np.array([rightmost_Y])])
        AUSUC = auc(sorted_X, sorted_Y)
        return AUSUC

class ProbabilisticCombine(CombinerGZSL):
    """ A probabilistic approach to combine expert predictions.

        Note: In variable names, due to historical reasons, sometimes a category is noted with z instead of y
    """

    def combine(self, threshold=0.5, T_cond=0.1):
        pred_ZS = self.pred_ZS
        pred_S = self.pred_S

        P_Gate__Unseen = sigmoid(threshold - self._pred_gating, T=T_cond)
        if self._hard_gating:
            P_Gate__Unseen = P_Gate__Unseen > 0.5
        P_Gate__Unseen = 1.*P_Gate__Unseen.reshape([-1,1]) # 1* for casting to float
        eps_sanity_check = False # Faster. Set for True (slower) during debugging
        P_Gate__Unseen = ml_utils.add_epsilon_to_P(P_Gate__Unseen, sanity_checks=eps_sanity_check)

        P_ZS__Unseen = 1.*pred_ZS[:, self._unseen_classes].sum(axis=1).reshape([-1,1])
        P_ZS__Unseen = ml_utils.add_epsilon_to_P(P_ZS__Unseen,
                                                 sanity_checks=eps_sanity_check)

        P_ZS__y_Unseen = np.zeros_like(pred_ZS) # + EPSILON
        P_ZS__y_Unseen[:, self._unseen_classes] = pred_ZS[:, self._unseen_classes]
        P_ZS__y_Unseen = ml_utils.add_epsilon_to_P(P_ZS__y_Unseen, axis=1, sanity_checks=eps_sanity_check)

        P_S__y_Seen = np.zeros_like(pred_ZS) #+ EPSILON
        P_S__y_Seen[:, self._seen_classes] = pred_S[:, self._seen_classes]
        P_S__y_Seen = ml_utils.add_epsilon_to_P(P_S__y_Seen, axis=1, sanity_checks=eps_sanity_check)

        return getattr(self, self._combiner)(P_ZS__y_Unseen, P_S__y_Seen, P_ZS__Unseen,
                                             P_Gate__Unseen)

    def vanilla_total_prob(self, P_ZS__y_Unseen, P_S__y_Seen, P_ZS__Unseen,
                         P_Gate__Unseen):
        if self.save_activations:
            # activations for debug
            self.activations['P_z_unseen'] = P_ZS__y_Unseen.copy()
            self.activations['P_z_seen'] = P_S__y_Seen.copy()
            self.activations['Punseen_zs'] = P_ZS__Unseen.copy()

        P_y_given_Unseen = P_ZS__y_Unseen / (P_ZS__Unseen)
        P_y_given_Seen = P_S__y_Seen
        P_y_given_Seen = ml_utils.add_epsilon_to_P(P_y_given_Seen, sanity_checks=False)

        P_y_given_Seen_norm = P_y_given_Seen[:, self._seen_classes].sum(axis=1, keepdims=1)
        P_y_given_Seen[:, self._seen_classes] /= P_y_given_Seen_norm

        P_z_given_x = P_Gate__Unseen * P_y_given_Unseen + (1. - P_Gate__Unseen) * P_y_given_Seen
        if self.save_activations:
            self.activations['Punseen_ad'] = P_Gate__Unseen
            self.activations['P_z_given_unseen_x'] = P_y_given_Unseen
            self.activations['P_z_given_seen_x'] = P_y_given_Seen
            self.activations['P_z_given_x'] = P_z_given_x
            # debugging
            self.activations['norm_P_z_given_seen_x'] = P_y_given_Seen_norm

        return P_z_given_x


    def adaptive_smoothing(self, P_z_unseen, P_z_seen, Punseen_zs, Punseen_ad):
        if self.save_activations:
            # activations for debug

            self.activations['P_z_unseen'] = P_z_unseen.copy()
            self.activations['P_z_seen'] = P_z_seen.copy()
            self.activations['Punseen_zs'] = Punseen_zs.copy()
            # Saving activations of vanilla_total_prob, but with hyper-params of adaptive_smoothing,
            # here is repeated the code of vanilla_total_prob
            P_ZS__y_Unseen, P_S__y_Seen, P_ZS__Unseen, P_Gate__Unseen = P_z_unseen, P_z_seen, Punseen_zs, Punseen_ad

            P_y_given_Unseen = P_ZS__y_Unseen / (P_ZS__Unseen)
            P_y_given_Seen = P_S__y_Seen.copy()
            P_y_given_Seen = ml_utils.add_epsilon_to_P(P_y_given_Seen, sanity_checks=False)
            P_y_given_Seen_norm = P_y_given_Seen[:, self._seen_classes].sum(axis=1, keepdims=1)
            P_y_given_Seen[:, self._seen_classes] /= P_y_given_Seen_norm

            P_z_given_x__soft_combine = P_Gate__Unseen * P_y_given_Unseen + (1. - P_Gate__Unseen) * P_y_given_Seen

            self.activations['soft_combine'] = {}
            self.activations['soft_combine']['Punseen_ad'] = P_Gate__Unseen.copy()
            self.activations['soft_combine']['P_z_given_unseen_x'] = P_y_given_Unseen.copy()
            self.activations['soft_combine']['P_z_given_seen_x'] = P_y_given_Seen.copy()
            self.activations['soft_combine']['norm_P_z_given_seen_x'] = P_y_given_Seen_norm.copy()
            self.activations['soft_combine']['P_z_given_x'] = P_z_given_x__soft_combine.copy()


        Prior_unseen = 1./len(self._unseen_classes)
        Prior_seen = 1./len(self._seen_classes)

        P_z_given_unseen_x = P_z_unseen
        P_z_given_unseen_x[:, self._unseen_classes] += (1.-Punseen_ad)*Prior_unseen
        P_z_given_seen_x = P_z_seen
        P_z_given_seen_x[:, self._seen_classes] += Punseen_ad * Prior_seen

        P_z_given_x =  Punseen_ad*P_z_given_unseen_x + (1.-Punseen_ad)*P_z_given_seen_x
        if self.save_activations:
            self.activations['Punseen_ad'] = Punseen_ad
            self.activations['P_z_given_unseen_x'] = P_z_given_unseen_x
            self.activations['P_z_given_seen_x'] = P_z_given_seen_x
            self.activations['P_z_given_x'] = P_z_given_x

        return P_z_given_x


class CalibratedStacking(CombinerGZSL):
    """
    Chao 2016
    """

    def combine(self, gamma=0):
        pred = self.pred_ZS

        pred[:, self._seen_classes] -= gamma

        if self.save_activations:
            self.activations['Punseen_ad'] = np.array([])
            self.activations['P_z_given_unseen_x'] = pred[:, self._unseen_classes]
            self.activations['P_z_given_seen_x'] = pred[:, self._seen_classes]
            self.activations['P_z_given_x'] = pred

        return pred + 1e-12


class ConstSmoothing(ProbabilisticCombine):
    def combine(self, gamma=1, threshold=0.5, T_cond=0.1):
        self._gamma = gamma
        return super().combine(threshold=threshold, T_cond=T_cond)

    def const_smoothing(self, P_z_unseen, P_z_seen, Punseen_zs, Punseen_ad):
        Prior_unseen = 1./len(self._unseen_classes)
        Prior_seen = 1./len(self._seen_classes)

        gamma = self._gamma
        P_z_given_unseen_x = gamma*P_z_unseen/(Punseen_zs)
        P_z_given_unseen_x[:, self._unseen_classes] += (1.-gamma)*Prior_unseen
        P_z_given_seen_x = P_z_seen
        P_z_given_seen_x = ml_utils.add_epsilon_to_P(P_z_given_seen_x,
                                                     sanity_checks=False)
        P_z_given_seen_x[:, self._seen_classes] /= P_z_given_seen_x[:, self._seen_classes].sum(axis=1, keepdims=1)
        P_z_given_seen_x[:, self._seen_classes] *= gamma
        P_z_given_seen_x[:, self._seen_classes] += (1.-gamma) * Prior_seen

        P_z_given_x =  Punseen_ad*P_z_given_unseen_x + (1.-Punseen_ad)*P_z_given_seen_x
        if self.save_activations:
            self.activations['Punseen_ad'] = Punseen_ad
            self.activations['P_z_given_unseen_x'] = P_z_given_unseen_x
            self.activations['P_z_given_seen_x'] = P_z_given_seen_x
            self.activations['P_z_given_x'] = P_z_given_x
        return P_z_given_x


mixture_factory = {'adaptive_smoothing': ProbabilisticCombine,
                   'vanilla_total_prob': ProbabilisticCombine,
                   'const_smoothing': ConstSmoothing,
                   'calibrated_stacking': CalibratedStacking,
                   }


def sigmoid(x, T):
    return 1. / (1 + np.exp(-x/T))