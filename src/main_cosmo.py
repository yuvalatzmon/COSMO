"""
COSMO (Atzmon, CVPR 2019) experiments main script

Author: Yuval Atzmon
"""
import os
import pickle

import matplotlib


import argparse

import numpy as np
import pandas as pd



from src.GZSL_mixture import GZSL_experiment

from src.gating import gating_model_selection_experiment

from src.cosmo_misc import train_or_load_logreg, get_filename
from src.utils import ml_utils
from src import load_xian_data
from src.metrics import ZSL_Metrics
from src.utils.gridfs_os_emulation import OsFs

data_loaders_factory = {# Features, labels and splits from Xian, CVPR 2017
                        'SUN': load_xian_data.SUN_Xian,
                        'CUB': load_xian_data.CUB_Xian,
                        'AWA1': load_xian_data.AWA1_Xian,
                        'AWA2': load_xian_data.AWA2_Xian,
                        'FLO' : load_xian_data.FLO_Xian,
                        }



def cfg_matplotlib_defaults():
    font_size = 18
    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)


def get_common_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='CUB',
                        help='Dataset name')
    parser.add_argument("--zs_expert_name", type=str, default='LAGO',
                        help="ZS_expert name. Supported are: ['LAGO', 'XianGAN']. XianGAN is an alias for fCLSWGAN.")
    parser.add_argument("--save_results_path", type=str, default='output/COSMO',
                        help="Pathname to save the results. Setting it to None, does not save results.")

    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--show_plots", type=bool, default=True,
                        help='show plots of Acc_tr, Acc_tr, Acc_H')
    parser.add_argument("--dump_activations", type=bool, default=False,
                        help='dump model activations to file of selected (cross-validated) model. Note: file is large.')

    args, unknown_args = parser.parse_known_args()

    args = parse_common_commandline_args(args)
    return args

def parse_common_commandline_args(args):
    """ Additional hardcoded parsing of command line args"""

    # Reproduce the same seed as in paper
    # Note: Seed were chosen with respect to seed used for training LAGO:
    #   Each LAGO experiment was repeated 5 times, and we chose the seed that yields median accuracy.
    #   In fact, since we can't control some of the randomness of tensorflow, the seed
    #   can only reproduce the split of the seen validation set.

    LAGO_seed = dict(AWA1=2, SUN=3, CUB=1, FLO=-1)[args.dataset_name]
    vars(load_xian_data.common_args)['repeat'] = LAGO_seed  # historically it was also named as 'repeat'
    vars(args)['repeat'] = LAGO_seed
    # seed for generating the validation set of Seen classes (as was used in LAGO)
    vars(load_xian_data.common_args)['seen_val_seed'] = LAGO_seed + 1001
    vars(args)['seen_val_seed'] = LAGO_seed + 1001

    return args

def get_data(dataset_name, args):
    print('Loading Data ...')

    data = {'dataset_name':dataset_name, 'seen_val_seed':args.seen_val_seed}

    # training (+eval) data for evaluating on validation set
    data_loader = data_loaders_factory[dataset_name](use_trainval_set=0,
                                                     transfer_task='GZSL')
    data_train = data_loader.get_data()
    data_train['data_loader'] = data_loader
    data['data_train'] = data_train
    data['X_GZSLval'] = np.block([[data_train['X_seen_val']],
                                  [data_train['X_unseen_val']]])
    data['Y_GZSLval'] = np.block([data_train['Y_seen_val'],
                                   data_train['Y_unseen_val']])
    data['F_GZSLval'] = np.block([data_train['F_seen_val'],
                                  data_train['F_unseen_val']])
    data['seen_classes_val'] = np.unique(data_train['Y_seen_train'])
    data['unseen_classes_val'] = np.unique(data_train['Y_unseen_val'])

    # training (+eval) data for evaluating on test set
    data_loader = data_loaders_factory[dataset_name](use_trainval_set=1,
                                                     transfer_task='GZSL')
    data_trainval = data_loader.get_data()
    data_trainval['data_loader'] = data_loader
    data['data_trainval'] = data_trainval
    data['X_GZSLtest'] = np.block([[data_trainval['X_seen_val']],
                                   [data_trainval['X_unseen_val']]])
    data['Y_GZSLtest'] = np.block([data_trainval['Y_seen_val'],
                                   data_trainval['Y_unseen_val']])
    data['F_GZSLtest'] = np.block([data_trainval['F_seen_val'],
                                   data_trainval['F_unseen_val']])
    data['seen_classes_test'] = np.unique(data_trainval['Y_seen_train'])
    data['unseen_classes_test'] = np.unique(data_trainval['Y_unseen_val'])

    data['num_class'] = 1 + len(data['seen_classes_test']) + \
                        len(data['unseen_classes_test'])
    return data


def get_ZS_expert_predictions(eval_set_name, dataset_name, zs_expert_name):
    assert(eval_set_name in ['val', 'test'])

    if zs_expert_name == 'XianGAN':
        fname = get_filename(f'XianGAN_pred', dataset_name=dataset_name,
                             eval_set_name=eval_set_name)
        pred_prob = np.load(fname)[f'pred_{eval_set_name}']
    elif zs_expert_name == 'LAGO':
        fname = get_filename(f'LAGO_pred', dataset_name=dataset_name,
                             eval_set_name=eval_set_name)
        pred_prob = np.load(fname)[f'pred_{eval_set_name}']
    else:
        raise ValueError('zs_expert_name=', zs_expert_name)

    return pred_prob

def get_logreg_S_predictions(eval_set_name, data):
    """ Train Seen classes classifier and get its predictions on the relevant
    evaluation set """

    # num_classes is the same for both phases of training (train and trainval),
    # keeping a placeholder for classes that don't participate on the train phase.
    num_class = 1 + len(data['seen_classes_test']) + len(data['unseen_classes_test'])


    # C=1 is the default aggressiveness hyper param
    # max_iter=2000 for allowing enough iteration until convergence
    hyper_params = dict(num_class=num_class, C=1, max_iter=2000, seen_val_seed=data['seen_val_seed'])

    # Set the phase for the model training data (train or trainval)
    use_trainval_set = {'val':0, 'test':1}[eval_set_name]

    fname = get_filename('seen_expert_model', dataset_name=data['dataset_name'],
    use_trainval_set=use_trainval_set, hyper_params=hyper_params)
    current_data = {'val':data['data_train'], 'test':data['data_trainval']}[
        eval_set_name]
    S_model = train_or_load_logreg(fname, current_data, overwrite=False,
                              **hyper_params)

    # Generate predictions for the model on the respective evaluation set
    X = {'val':data['X_GZSLval'], 'test':data['X_GZSLtest']}[eval_set_name]
    pred_S = S_model.predict(X)
    return pred_S

def get_experts_predictions(data):
    args = get_common_commandline_args()

    # Get ZS expert predictions on both evaluation sets
    pred_ZS__GZSLval = get_ZS_expert_predictions('val',
                                                 dataset_name=data['dataset_name'],
                                                 zs_expert_name=args.zs_expert_name)
    pred_ZS__GZSLtest = get_ZS_expert_predictions('test',
                                                  dataset_name=data['dataset_name'],
                                                  zs_expert_name=args.zs_expert_name)

    # Get Seen expert predictions on both evaluation sets
    pred_S__GZSLval = get_logreg_S_predictions('val', data)
    pred_S__GZSLtest = get_logreg_S_predictions('test', data)

    if args.debug:
        zs_metrics_test = ZSL_Metrics(data['seen_classes_test'],
                                    data['unseen_classes_test'])
        print(zs_metrics_test.unseen_balanced_accuracy(data['Y_GZSLtest'], pred_ZS__GZSLtest))
        print(zs_metrics_test.seen_balanced_accuracy(data['Y_GZSLtest'], pred_S__GZSLtest))

        zs_metrics_val = ZSL_Metrics(data['seen_classes_val'],
                                    data['unseen_classes_val'])
        print(zs_metrics_val.unseen_balanced_accuracy(data['Y_GZSLval'],
                                                      pred_ZS__GZSLval))
        print(zs_metrics_val.seen_balanced_accuracy(data['Y_GZSLval'],
                                                    pred_S__GZSLval))
        pass

    expert_preds = dict(ZS__GZSLval=pred_ZS__GZSLval, ZS__GZSLtest=pred_ZS__GZSLtest,
                        S__GZSLval=pred_S__GZSLval, S__GZSLtest=pred_S__GZSLtest)
    return expert_preds


def display_and_save_result_tables(df_val, df_all_results, fs_results=None):
    args = get_common_commandline_args()

    print('\x1b[6;30;41m' + 'Validation'  + '\x1b[0m')
    print(df_val.sort_values('Acc_H', axis=0).drop_duplicates().set_index('method'))
    print('\x1b[6;30;42m' + 'Test'  + '\x1b[0m')
    print(df_all_results.sort_values('Acc_H', axis=0).drop_duplicates().set_index(
        'method'))

    if fs_results is not None:
        fname_summary_dataframes = os.path.join(args.dataset_name, args.zs_expert_name, 'summary_dataframes.pkl')
        with fs_results.new_file(filename=fname_summary_dataframes) as fp:
            ml_utils.pickle_iffnn(dict(zs_model_name=args.zs_expert_name, df_val=df_val, df_all_results=df_all_results), fp)


def gzsl_experiment_wrap(df_val, df_all_results, activations, *args, **kwargs):
    df_res, df_res_val, activations_best = GZSL_experiment(*args, **kwargs)
    df_all_results = df_all_results.append(df_res).drop_duplicates()
    df_val = df_val.append(df_res_val).drop_duplicates()
    activations.update(activations_best)
    print('\n==================================================================\n\n')
    return df_val, df_all_results

def gzsl_experiments(data, expert_preds, gating_models, fs_results=None, dirname=None, show_plots=True):

    activations = {} # collects activations to dump from each experiment
    df_all_results = pd.DataFrame()
    df_val = pd.DataFrame()

    # COSMO Gating & Smoothing
    df_val, df_all_results = gzsl_experiment_wrap(df_val, df_all_results,
                                                  activations, data,
                                                  expert_preds, gating_models['CB_Gating_Teq3'],
                                                  mixture_type='adaptive_smoothing',
                                                  hard_gating=False,
                                                  num_resample_points=100,
                                                  fs_results=fs_results, dirname=dirname, show_plot=show_plots)
    if False:
        # Set to True, to exit here (for debug phase)
        dump_activations(activations, data, expert_preds)
        display_and_save_result_tables(df_val, df_all_results, fs_results=None)
        return




    # CS (calibrated stacking) baseline
    df_val, df_all_results = gzsl_experiment_wrap(df_val, df_all_results,
                                                  activations, data,
                                                  expert_preds, None,
                                                  mixture_type='calibrated_stacking',
                                                  num_resample_points=100,
                                                  fs_results=fs_results, dirname=dirname, show_plot=show_plots)


    # Ablation on Smoothing
    for mixture_type in ['vanilla_total_prob', 'const_smoothing', 'adaptive_smoothing']:
    # search on const_smoothing is much slower than other variants due to additional hyper param.
    # Consider deleting it, since it is only used for ablation study.

    # for mixture_type in ['vanilla_total_prob', 'adaptive_smoothing', ]:
        num_resample_points = 100
        if (data['dataset_name'] == "CUB" and mixture_type=='adaptive_smoothing'):
            num_resample_points = 300

        df_val, df_all_results = gzsl_experiment_wrap(df_val, df_all_results,
                                                      activations, data,
                                                      expert_preds, gating_models['maxP1'],
                                                      mixture_type=mixture_type,
                                                      hard_gating=False,
                                                      num_resample_points=num_resample_points,
                                                      fs_results=fs_results, dirname=dirname, show_plot=show_plots)

    # Ablation on Gating
    for gating_model in ['maxP1', 'maxP3', 'CB_Gating_Teq3_noZS', 'CB_Gating_Teq1','CB_Gating_Teq3']:
        df_val, df_all_results = gzsl_experiment_wrap(df_val, df_all_results,
                                                      activations, data,
                                                      expert_preds, gating_models[gating_model],
                                                      mixture_type='vanilla_total_prob',
                                                      hard_gating=False,
                                                      num_resample_points=100,
                                                      fs_results=fs_results, dirname=dirname, show_plot=show_plots)

    # Independent-Hard baseline: Resembles CMT (Socher) hard gating, with our ZS-expert and MaxP
    df_val, df_all_results = gzsl_experiment_wrap(df_val, df_all_results,
                                                  activations, data,
                                                  expert_preds, gating_models['maxP1'],
                                                  mixture_type='vanilla_total_prob',
                                                  hard_gating=True,
                                                  num_resample_points=100,
                                                  fs_results=fs_results, dirname=dirname, show_plot=show_plots)

    dump_activations(activations, data, expert_preds)

    display_and_save_result_tables(df_val, df_all_results, fs_results)

def dump_activations(activations, data, expert_preds):
    # dump activations to file.
    args = get_common_commandline_args()
    if args.dump_activations:
        expert_name = args.zs_expert_name
        fname = get_filename('activations', dataset_name=data["dataset_name"], expert_name=expert_name)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        with open(fname, 'wb') as f:
            pickle.dump(dict(activations=activations, data=data,
                             expert_preds=expert_preds), f)

        print(f'Dumped activations to {fname}')

def main():
    # init
    args = get_common_commandline_args()
    cfg_matplotlib_defaults()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 200)


    # Load_data
    dataset_name = args.dataset_name
    print(f'dataset_name={dataset_name}')
    data = get_data(dataset_name, args)

    # 1. Load pre-trained ZS model predictions.
    # 2. Train S model (or load pre-trained model), and generate its predictions.
    expert_preds = get_experts_predictions(data)

    # Train and evaluate the gating models (individually), reproducing Table 3 in paper.
    gating_models = gating_model_selection_experiment(data, expert_preds, args.zs_expert_name)

    fs_results = None
    dirname = None
    if args.save_results_path is not None:
        fs_results = OsFs(args.save_results_path)
        dirname = f'{args.zs_expert_name}/experiments'

    gzsl_experiments(data, expert_preds, gating_models, fs_results=fs_results, dirname=dirname, show_plots=args.show_plots)
    pass


if __name__ == '__main__':
    main()
