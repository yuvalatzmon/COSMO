from warnings import warn

import pickle
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpolate
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression

from src.utils import ml_utils
from src.utils.ml_utils import to_percent

from src.metrics import ZSL_Metrics
import src.gzsl_search_configurations as configurations


def get_filename(file_type,  **kwargs):
    fnames_dict = dict()
    fname = fnames_dict.get(file_type, None)

    if fname is None:
        if file_type.startswith('LAGO_pred'):
            eval_set_name= kwargs['eval_set_name']
            dataset_name = kwargs["dataset_name"]
            pred_path = f'data/LAGO_GZSL_predictions/{dataset_name}'
            fname = os.path.join(pred_path, f'pred_gzsl_{eval_set_name}.npz')
        elif file_type.startswith('XianGAN_pred'):
            eval_set_name= kwargs['eval_set_name']
            dataset_name = kwargs["dataset_name"]
            pred_path = f'data/XianGAN_predictions/{dataset_name}'
            fname = os.path.join(pred_path, f'pred_gzsl_{eval_set_name}.npz')

        elif file_type == 'seen_expert_model':
            use_trainval_set = kwargs['use_trainval_set']
            hyper_params = kwargs['hyper_params']

            pathname = os.path.join('output', 'seen_expert_model', kwargs["dataset_name"],
                                    f'use_trainval_set={use_trainval_set}')
            fname = os.path.join(pathname,
                                 ml_utils.build_string_from_dict(hyper_params, sep='_'),
                                 'log_regression_clf.pkl')

        elif file_type == 'gater_scores_dist':
            fname = os.path.join('output/COSMO', kwargs["dataset_name"], kwargs["expert_name"], 'gater/raw',
                                 kwargs["gater_model_name"] + '.pkl')

        elif file_type == 'gating_summary':
            fname = os.path.join('output/COSMO', kwargs["dataset_name"], kwargs["expert_name"], 'gater',
                                 'summary.pkl')
        elif file_type == 'activations':
            fname = os.path.join('output/COSMO', kwargs["dataset_name"], kwargs["expert_name"], 'activations.pkl')
        else:
            raise ValueError(f'unknown file_type: {file_type}')

    return fname


def get_ood_metrics(Y_is_inD, activations, name):
    Y_pred_ad = (1-activations['Punseen_ad']) > 0.5
    TPR, FPR, TNR, FNR, ACC = ml_utils.binary_classification_metrics(Y_is_inD,
                                                                     Y_pred_ad)
    TPR, FPR, TNR, FNR, ACC = to_percent((TPR, FPR, TNR, FNR, ACC ))

    df_ood_metrics = pd.DataFrame(dict(TPRad=TPR, FPRad=FPR, TNRad=TNR, FNRad=FNR,
                                       ACCad=ACC),
                 index=[f'{name}'])
    #
    # df_ood_metrics = pd.DataFrame(columns='TPR,FPR,TNR,FNR,ACC'.split(','))
    # df_ood_metrics = pd.concat([df_ood_metrics,
    #            ], axis=1, sort=True)
    df_ood_metrics = df_ood_metrics.loc[:, 'TPRad,FPRad,TNRad,FNRad,ACCad'.split(',')]
    return df_ood_metrics

def get_predictions_of_best_cfg(best_complete_cfg, Y_GZSLval, Y_GZSLtest, current_model_GZSLval,
                                current_model_GZSLtest):
    cp = ml_utils.copy_np_arrays_in_dict

    # Get best config after selecting a threshold
    amax = current_model_GZSLval.df_sweep_results.Acc_H.idxmax()
    best_final_cfg = ml_utils.slice_dict(current_model_GZSLval.df_sweep_results.loc[amax,
                                         :].to_dict(),
                                         list(best_complete_cfg.keys()))
    best_final_cfg = configurations.cast_to_lists(best_final_cfg)

    # Rerun best config and save its activations
    current_model_GZSLval.save_activations = True
    _ = current_model_GZSLval.sweep_variables(Y_GZSLval, num_pool_workers=1,
                                              **best_final_cfg)
    activations_tr = cp(current_model_GZSLval.activations)
    current_model_GZSLval.save_activations = False

    current_model_GZSLtest.save_activations = True
    _ = current_model_GZSLtest.sweep_variables(Y_GZSLtest, num_pool_workers=1,
                                               **best_final_cfg)
    activations_ts = cp(current_model_GZSLtest.activations)
    current_model_GZSLtest.save_activations = False


    return activations_tr, activations_ts, current_model_GZSLval, current_model_GZSLtest


# noinspection PyUnresolvedReferences
def crossval_and_plot_sweep_results(sweep_key, data, expert_preds, current_model_GZSLval,
                                    current_model_GZSLtest, fig_list, fs_results, best_complete_cfg, dirname):
    dataset_name = data['dataset_name']
    metrics_GZSL_val = ZSL_Metrics(data['seen_classes_val'], data['unseen_classes_val'])
    metrics_GZSL_test = ZSL_Metrics(data['seen_classes_test'], data['unseen_classes_test'])

    baselines = get_baselines(dataset_name, metrics_GZSL_val, metrics_GZSL_test,
                              data['Y_GZSLval'], data['Y_GZSLtest'], expert_preds[
                                  'ZS__GZSLval'], expert_preds['ZS__GZSLtest'],
                              expert_preds['S__GZSLval'], expert_preds['S__GZSLtest'])
    df_res_test, df_res_val = summarize_results_and_cross_validate_by_acc_H(current_model_GZSLval,
                                                                            current_model_GZSLtest,
                                                                            baselines=baselines)
    xlim, ylim = get_lims_cvpr(dataset_name)
    fig = sweep_plot(current_model_GZSLval, current_model_GZSLtest, sweep_key, xlim=xlim,
                     ylim=ylim)
    fig_list.append(fig)

    amax = current_model_GZSLval.df_sweep_results.Acc_H.idxmax()
    print('\n*Best hyper params and val metrics:* \n',
          current_model_GZSLval.df_sweep_results.loc[amax, :])

    if fs_results is not None:
        best_final_cfg = ml_utils.slice_dict(current_model_GZSLval.df_sweep_results.loc[amax,
                                             :].to_dict(),
                                             list(best_complete_cfg.keys()))
        name = current_model_GZSLtest.name
        fname_curves = os.path.join(dataset_name, dirname, name, 'curves.pkl')
        with fs_results.new_file(filename=fname_curves) as fp:
            ml_utils.pickle_iffnn(dict(best_final_cfg=best_final_cfg,
                                       amax=amax,
                                       best_hp_results_tr=current_model_GZSLval.df_sweep_results.loc[amax, :],
                                       best_hp_results_ts=current_model_GZSLtest.df_sweep_results.loc[amax, :],
                                       current_model_GZSLval=current_model_GZSLval,
                                       current_model_GZSLtest=current_model_GZSLtest), fp)

    return df_res_test, df_res_val

def resample_sweepkey_by_curve(df_sweep_results, sweep_key, n_smp_rise=100,
                               n_smp_fall=100):
    # geometric mean of acc_tr, acc_ts
    # curve = np.sqrt(df_sweep_results.Acc_tr.values * df_sweep_results.Acc_ts.values)
    curve = 2*(df_sweep_results.Acc_tr.values * df_sweep_results.Acc_ts.values) / (
            df_sweep_results.Acc_tr.values + df_sweep_results.Acc_ts.values + \
            ml_utils.epsilon())
    amax = curve.argmax()
    if amax==0 or amax==len(curve)-1:
        tmin = df_sweep_results[sweep_key].min()
        tmax = df_sweep_results[sweep_key].max()

        warn(f'Argmax  ={amax} is on edge on thresholds range. Please '
             f'increase that range [{tmin}, {tmax}]')
        if amax == 0:
            amax += 1
        elif amax == len(curve)-1:
            amax -= 1


    x_rise = curve[:(amax + 1)]
    y_rise = df_sweep_results[sweep_key].iloc[:(amax + 1)]

    x_fall = curve[amax:]
    y_fall = df_sweep_results[sweep_key].iloc[amax:]

    f_th_vs_metric_rise = interpolate.interp1d(x_rise, y_rise)
    f_th_vs_metric_fall = interpolate.interp1d(x_fall, y_fall)

    # making most of the sampling around the peak
    rise_range = np.sort(np.unique(np.block([
        np.linspace(x_rise.min(), x_rise.min() + (x_rise.max() - x_rise.min()) * 0.9,
                                                  int(n_smp_rise * 0.25)),
        np.linspace(x_rise.min() + (x_rise.max() - x_rise.min()) * 0.9, x_rise.max(),
                    int(n_smp_rise * 0.75))])))
    fall_range = np.sort(np.unique(np.block([
        np.linspace(x_fall.min(), x_fall.min() + (x_fall.max() - x_fall.min()) * 0.9,
                    int(n_smp_fall * 0.25)),
        np.linspace(x_fall.min() + (x_fall.max() - x_fall.min()) * 0.75, x_fall.max(),
                    int(n_smp_fall * 0.9))])))
    # rise_range = []
    sweep_key_range_rise = [f_th_vs_metric_rise(x) for x in rise_range]
    sweep_key_range_fall = [f_th_vs_metric_fall(x) for x in fall_range]
    sweep_key_equi_dist = np.sort(
        np.unique(sweep_key_range_rise + sweep_key_range_fall))
    return sweep_key_equi_dist


def summarize_results_and_cross_validate_by_acc_H(current_model_GZSLval, current_model_GZSLtest, baselines=None):
    # cross validate by Acc_H metric
    amax = current_model_GZSLval.df_sweep_results.Acc_H.idxmax()

    AUSUC_ts = current_model_GZSLtest.AUSUC()
    current_title = current_model_GZSLtest.name

    df_list = []
    for model in [current_model_GZSLtest, current_model_GZSLval]:
        ats, atr, ah = model.df_sweep_results.loc[
            amax, ['Acc_ts', 'Acc_tr', 'Acc_H']]

        df_res = pd.DataFrame(columns='method,Acc_ts,Acc_tr,Acc_H,AUSUC'.split(','))
        values = np.round(100 * np.array((ats, atr, ah, AUSUC_ts)), 1)
        df_res = pd.concat([df_res,
                   pd.DataFrame(dict(method=f'{current_title}',
                                     Acc_ts=values[0],
                                     Acc_tr=values[1],
                                     Acc_H=values[2],
                                     AUSUC=values[3]), index=[0])],)# sort=False)

        df_list.append(df_res)

    df_res_test, df_res_val = df_list

    if baselines is not None:
        for i, (bl, bl_res) in enumerate(baselines.items()):
            df_res_test.loc[i+1, 'method,Acc_ts,Acc_tr,Acc_H,AUSUC'.split(',')] = \
                [bl] + np.round(100 * np.array(bl_res), 1).tolist() + [None]

    return df_res_test.loc[:, 'method,Acc_ts,Acc_tr,Acc_H,AUSUC'.split(',')], \
           df_res_val.loc[:, 'method,Acc_ts,Acc_tr,Acc_H,AUSUC'.split(',')]


def sweep_plot(current_model_GZSLval, current_model_GZSLtest, sweep_key, xlim=None, ylim=None):
    if xlim is None:
        xlim = (0, 0.8)
    if ylim is None:
        ylim = (0, 0.8)

    # Left plot:
    fig = plt.figure(figsize=(13.5, 4.7))
    ax = fig.add_subplot(1, 2, 1)
    current_model_GZSLval.df_sweep_results.plot(sweep_key, 'Acc_ts,Acc_tr,Acc_H'.split(','),
                                                ax=ax)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(f'Validation Accuracies vs {sweep_key}')
    ax = fig.add_subplot(1, 2, 2)

    # Right plot
    current_model_GZSLtest.plot_tstr_curve_cvpr(ax=ax, xlim=xlim, ylim=ylim,
                                                is_first=True)
    ax.set_title('Test Acc_ts vs Acc_tr')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    return fig

def get_baselines(dataset_name, metrics_GZSL_val=None, metrics_GZSL_test=None, Y_GZSLval=None, Y_GZSLtest=None,
                  pred_ZS__GZSLval=None, pred_ZS__GZSLtest=None, pred_S__GZSLval=None, pred_S__GZSLtest=None, only_sota=True):
    if metrics_GZSL_val is None:
        ZSmodel = (-1, -1, -1)
    else:
        ZSmodel = metrics_GZSL_test.generlized_scores(Y_GZSLtest, pred_ZS__GZSLtest)
    if dataset_name == 'CUB':
        # Generalized ZS baseline
        CUB_baselines = dict(
            ZSmodel=ZSmodel,
            # Dmodel=Dmodel,
            DCN_NIPS18=(28.4 / 100, 60.7 / 100, 38.7 / 100),  # NIPS18
            #     DIPL=(41.7/100, 44.8/100, 43.2/100), # Transductive, NIPS18
            #         CDL=(23.5/100, 55.2/100, 32.9/100) # transductive, ECCV18,
            GAN_CVPR18=(43.7 / 100, 57.7 / 100, 49.7 / 100),
            CCGAN_ECCV18=(47.9 / 100, 59.3 / 100, 53.0 / 100),
            CVAE2_CVPR18=(41.5/100, 53.3/100, 46.7/100),
            Kernel_CVPR18=(19.9/100, 52.5/100, 28.9/100),
            ICINESS_IJCAI18=(-1, -1, 41.8/100),
            RELATIONET_CVPR18=(38.1/100, 61.1/100, 47.0/100),
            TRIPLE_TIP19=(26.5 / 100, 62.3 / 100, 37.2 / 100),# Triple Verification Network.., 2019: IEEE T. on Image Processing 2019
            DEM_CVPR17=(19.6/100, 57.9/100, 29.2/100), #  Learning a deep embedding
            # model for zero-shot learning
            # #Rahman17=(44.9, 41.7, 43.3), # (2017 only @ arxiv) A Unified approach ..
            DAP=(1.7/100, 67.9/100, 3.3/100,),
            ALE=(23.7/100, 62.8/100, 34.4/100,),
            ESZSL=(12.6/100, 63.8/100, 21.0/100,),
            SYNC=(11.5/100, 70.9/100, 19.8/100,),
            SJE=(23.5/100, 59.2/100, 33.6/100, ),
            DEVISE=(23.8/100, 53.0/100, 32.8/100,),
            CMT_SOCHER=(4.7/100, 60.1/100, 8.7/100),
        )

        # Regular ZS baseline
        CUB_ZSbaselines = dict(
            ZSmodel=ZSmodel,
            DCN_NIPS18=58.2 / 100,  # NIPS18
            #         CDL=54.5/100,# transductive, ECCV18,
            #         GAN=57.3/100, # as reported in CCGAN (Xian report used different semantic represenatation)
            #         CCGAN=58.4/100,
        )


    elif dataset_name == 'AWA1':
        # Generalized ZS baseline
        AWA1_baselines = dict(
            ZSmodel=ZSmodel,
            # Dmodel=Dmodel,
            DCN_NIPS18=(25.5 / 100, 84.2 / 100, 39.1 / 100),
            GAN_CVPR18=(59.7 / 100, 61.4 / 100, 59.6 / 100),
            CCGAN_ECCV18=(59.6 / 100, 63.4 / 100, 59.8 / 100),
            #         CDL=(28.1/100, 73.5/100, 40.6/100) # transductive, ECCV18,
            CVAE2_CVPR18=(56.3/100, 67.8/100, 61.5/100,),
            Kernel_CVPR18=(18.3/100, 79.3/100, 29.8/100,),  # CVPR18
            RELATIONET_CVPR18=(31.4/100, 91.3/100, 46.7/100),
            DEM_CVPR17=(32.8/100, 84.7/100, 47.3/100),
            TRIPLE_TIP19=(27.0 / 100, 67.9 / 100, 38.6 / 100), # Triple Verification Network.., 2019: IEEE T. on Image Processing 2019
            # #Rahman17=(45.2, 68.6, 54.5), # (2017 only @ arxiv) A Unified approach ..
            DAP=(0.0/100, 88.7/100, 0.0/100,),
            ALE=(16.8/100, 76.1/100, 27.5/100,),
            ESZSL=(6.6/100, 75.6/100, 12.1/100,),
            SYNC=(8.9/100, 87.3/100, 16.2/100,),
            SJE=(11.3/100, 74.6/100, 19.6/100,),
            DEVISE=(13.4/100, 68.7/100, 22.4/100,),
            CMT_SOCHER=(8.4/100, 86.9/100, 15.3/100),

        )
        # Regular ZS baseline
        AWA1_ZSbaselines = dict(
            ZSmodel=ZSmodel,
            DCN_NIPS18=65.6 / 100,  # NIPS18
            #         CDL=69.9/100, # transductive, ECCV18,
            GAN=(57.9 / 100, 61.4 / 100, 59.6 / 100)
        )
    elif dataset_name == 'AWA2':
        # RELATIONET_CVPR18 = (30.0, 93.4, 45.3,),
        # DEM_CVPR17 = (30.5, 86.4, 45.1,),

        raise NotImplementedError()
    elif dataset_name == 'SUN':
        # Generalized ZS baseline
        SUN_baselines = dict(
            ZSmodel=ZSmodel,
            # Dmodel=Dmodel,
            DCN_NIPS18=(25.5 / 100, 37.0 / 100, 30.2 / 100),
            #         CDL=(21.5/100, 34.7/100, 26.5/100) # transductive, ECCV18,
            GAN_CVPR18=(42.6 / 100, 36.6 / 100, 39.4 / 100),
            CCGAN_ECCV18=(47.2 / 100, 33.8 / 100, 39.4 / 100),
            CVAE2_CVPR18=(40.9/100, 30.5/100, 34.9/100,),
            Kernel_CVPR18=(19.8/100, 29.1/100, 23.6/100),
            TRIPLE_TIP19=(22.2 / 100, 38.3 / 100, 28.1 / 100),# Triple Verification Network.., 2019: IEEE T. on Image Processing 2019
            # #Rahman17=(35.8, 27.8, 31.3),  # (2017 only @ arxiv) A Unified approach ..
            ICINESS_IJCAI18=(-1, -1, 30.3/100),
            DAP=(4.2/100, 25.1/100, 7.2/100,),
            ALE=(21.8/100, 33.1/100, 26.3/100,),
            ESZSL=(11.0/100, 27.9/100, 15.8/100,),
            SYNC=(7.9/100, 43.3/100, 13.4/100,),
            SJE=(14.7/100, 30.5/100, 19.8/100,),
            DEVISE=(16.9/100, 27.4/100, 20.9/100,),
            CMT_SOCHER = (8.7/100, 28.0/100, 13.3/100),
        )
        # Regular ZS baseline
        SUN_ZSbaselines = dict(
            ZSmodel=ZSmodel,
            DCN_NIPS18=62.4 / 100,  # NIPS18
            #         CDL=63.6/100, # transductive, ECCV18,
            GAN=60.8 / 100,
        )

    elif dataset_name == 'FLO':
        FLO_baselines = dict(
            ZSmodel=ZSmodel,
            GAN_CVPR18=(59.0 / 100, 73.8 / 100, 65.6 / 100),
            CCGAN_ECCV18=(61.6 / 100, 69.2 / 100, 65.2 / 100),

            DEVISE=(9.9 / 100, 44.2 / 100, 16.2 / 100,),
            SJE=(13.9 / 100,  47.6/ 100, 21.5 / 100,),
            ESZSL=(11.4 / 100, 56.8 / 100, 19.0 / 100,),
            ALE=(34.4 / 100, 13.3 / 100, 21.9 / 100,),
        )

    baselines = locals()[dataset_name + '_baselines']
    
    # oly return state-of-the-art methods
    if only_sota:
        SOTA_methods = 'ZSmodel, DCN_NIPS18, GAN_CVPR18, CCGAN_ECCV18'
        baselines = ml_utils.slice_dict(baselines, SOTA_methods)
    
    return baselines

def get_lims_cvpr(dataset_name):
    if dataset_name == 'CUB':
        xlim=(40, 75); ylim=(0., 65);
    elif dataset_name == 'SUN':
        xlim=(15, 50); ylim=(0., 65);
    elif dataset_name.startswith('AWA'):
        xlim=(60, 95); ylim=(0., 65);
    elif dataset_name == 'FLO':
        xlim=(30, 99); ylim=(5, 70);


    return xlim, ylim

def get_lims(dataset_name):
    if dataset_name == 'CUB':
        xlim=(0., 1); ylim=(0., 1); clim = (0, 0.55)
    elif dataset_name == 'SUN':
        xlim=(0., 1); ylim=(0., 1); clim = (0, 0.45)
    elif dataset_name.startswith('AWA'):
        xlim=(0., 1); ylim=(0., 1); clim = (0, 0.65)
    elif dataset_name.startswith('FLO'):
        xlim=(0., 1); ylim=(0., 1); clim = (0, 0.7)

    return xlim, ylim, clim

def get_zoom_lims(dataset_name):
    if dataset_name == 'CUB':
        xlim=(0.35, 0.7); ylim=(0.2, 0.55); clim = (0, 0.55)
    elif dataset_name == 'SUN':
        xlim=(0.1, 0.55); ylim=(0.1, 0.55); clim = (0, 0.45)
    elif dataset_name.startswith('AWA'):
        xlim=(0.55, 0.95); ylim=(0.2, 0.65); clim = (0, 0.65)

    return xlim, ylim, clim

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     as_plotly(fig=plt.gcf())


def my_outer_product(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0],
                                                   a.shape[1] * b.shape[1])



class skclf():
    """ Sklearn multiclass classifier """
    def __init__(self, model, num_class):
        self.model = model
        self.num_class = num_class
        self.T = 1

    def predict(self, X):
        pred = self.model.predict_proba(X)
        pred_padded = np.zeros((X.shape[0], self.num_class))
        pred_padded[:, self.model.classes_] = pred
        logits = np.log(pred_padded  + 1e-12)
        pred_padded = softmax(logits / self.T)

        return pred_padded

    def set_params(self, T=None):
        if T is not None:
            self.T = T
        return self


def train_or_load_logreg(fname, data, overwrite=False, **hyper_params):

    if os.path.exists(fname) and not overwrite:
        print(f'Loading {fname}')
        with open(fname, 'rb') as f:
            skclf_model = pickle.load(f)
            model = skclf(skclf_model, hyper_params['num_class'])
    else:
        print(f'Training {fname}')

        # seen_val_seed value is only used for setting the filename
        hyper_params.pop('seen_val_seed')

        model = train_logisitic_reg_model(data['X_seen_train'], data['Y_seen_train'],
                                          **hyper_params)
        print(f'Saving {fname}')
        ml_utils.my_mkdir(os.path.dirname(fname))
        with open(fname, 'wb') as f:
            pickle.dump(model.model, f)

    return model


def train_logisitic_reg_model(X_seen_train, Y_seen_train, num_class, C=1,
                              max_iter=1000):
    clf = LogisticRegression(random_state=0, C=C, solver='lbfgs', n_jobs=-1,
                             multi_class='multinomial', verbose=1, max_iter=max_iter,
                             ).fit(X_seen_train, Y_seen_train)

    return skclf(clf, num_class)


def softmax(z):
    """https://stackoverflow.com/a/39558290/2476373"""
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def entropy2(X):
    return entropy(X.T)/np.log(2)

