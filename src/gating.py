import re
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.cosmo_misc import get_filename, softmax
from src.utils import ml_utils
from src.utils.ml_utils import to_percent, fpr_at_tpr


class gater_model():
    def __init__(self, data):
        self._data = data
        self._dataset_name = data['dataset_name']
        self._Y_GZSLval = data['Y_GZSLval']
        self._Y_GZSLtest = data['Y_GZSLtest']
        self._seen_classes_val = data['seen_classes_val']
        self._seen_classes_test = data['seen_classes_test']
        self._unseen_classes_val = data['unseen_classes_val']
        self._unseen_classes_test = data['unseen_classes_test']


        # Generate the Gating-Train / Gating-Val split (Figure 3 in paper)
        # by a simple 2:1 split
        Y_is_Seen__GZSLval = ml_utils.is_member(data['Y_GZSLval'], data['seen_classes_val'])
        Y_is_Seen__GZSLtest = ml_utils.is_member(data['Y_GZSLtest'], data['seen_classes_test'])
        self.ix_val = np.zeros_like(Y_is_Seen__GZSLval)
        self.ix_val[0::3] = 1
        self.ix_train = ~self.ix_val
        self.Y_Gating_Train = Y_is_Seen__GZSLval[self.ix_train]
        self.Y_Gating_Val = Y_is_Seen__GZSLval[self.ix_val]
        self.Y_Gating_Test = Y_is_Seen__GZSLtest

        # Predictions on GZSLval and GZSLtest sets, updated by self.eval()
        self.pred_GZSLval = None
        self.pred_GZSLtest = None
        self._expert_name = None

        self.name = None
        self._set_basic_name()

    def _set_basic_name(self):
        name = str(self.__class__).split('.')[-1][:-2]
        # Add spaces
        if '_' in name:
            # by replacing '_' with ' '
            name = name.replace('_', ' ')
        else:
            # by adding ' ' before Capital letters
            # https://stackoverflow.com/a/2277363/2476373
            name= ' '.join(re.findall('[a-zA-Z][^A-Z]*', name))
        self.name = name
        return name

    def name_to_filename(self):
        """ trim ':'   '('   ')'   ','  and replace ' ' with '_'"""
        fname = re.sub('[:\(\),]', '', self.name)
        fname = fname.replace(' ', '_')
        return fname




    def eval(self,  expert_preds, expert_name, save_to_file=False, show_figure=False,
             show_title=True, show_legend=True,):

        self._expert_name = expert_name

        unseen_classes_val = self._unseen_classes_val
        unseen_classes_test = self._unseen_classes_test
        pred_S__GZSLval = expert_preds['S__GZSLval']
        pred_ZS__GZSLval = expert_preds['ZS__GZSLval']
        pred_S__GZSLtest = expert_preds['S__GZSLtest']
        pred_ZS__GZSLtest = expert_preds['ZS__GZSLtest']

        # predict p^{gating} for GZSLval and for GZSLtest
        pred_G__GZSLval = self.predict(pred_S__GZSLval, pred_ZS__GZSLval, unseen_classes_val)
        pred_G__GZSLtest = self.predict(pred_S__GZSLtest, pred_ZS__GZSLtest, unseen_classes_test)

        self.pred_GZSLval = pred_G__GZSLval
        self.pred_GZSLtest = pred_G__GZSLtest

        # map predictions on GZSLval to Gating_Train and Gating_Val sets
        pred_Gating_Train = pred_G__GZSLval[self.ix_train]
        pred_Gating_Val = pred_G__GZSLval[self.ix_val]
        pred_Gating_Test = pred_G__GZSLtest

        # For evaluation purpose, split to predictions of images from S vs ZS domains
        self.pred_on_S__Gating_Train = pred_Gating_Train[self.Y_Gating_Train]
        self.pred_on_S__Gating_Val = pred_Gating_Val[self.Y_Gating_Val]
        self.pred_on_S__Gating_Test = pred_Gating_Test[self.Y_Gating_Test]
        self.pred_on_ZS__Gating_Train = pred_Gating_Train[~self.Y_Gating_Train]
        self.pred_on_ZS__Gating_Val = pred_Gating_Val[~self.Y_Gating_Val]
        self.pred_on_ZS__Gating_Test = pred_Gating_Test[~self.Y_Gating_Test]

        pred_domain_tuples = {}
        pred_domain_tuples['train'] = (self.pred_on_S__Gating_Train, self.pred_on_ZS__Gating_Train)
        pred_domain_tuples['val'] = (self.pred_on_S__Gating_Val, self.pred_on_ZS__Gating_Val)
        pred_domain_tuples['test'] = (self.pred_on_S__Gating_Test, self.pred_on_ZS__Gating_Test)


        # Evaluate the gating performance
        dict_df_xvset = {}
        for xvset in ['train', 'val', 'test']:
            df = self.scores_metrics(pred_domain_tuples[xvset][0],
                                     pred_domain_tuples[xvset][1],
                                     save_to_file=save_to_file and (xvset == 'test'),
                                     show_figure=show_figure and (xvset == 'test'),
                                     show_legend=show_legend)
            dict_df_xvset[xvset.capitalize()] = df

            if xvset == 'test' and show_title and show_figure:
                plt.title(f'Distributions of AD scores on {xvset} set')
        df_results = pd.concat(dict_df_xvset, axis=1)
        df_results = df_results[['Train', 'Val', 'Test']].reset_index()
        df_results = df_results.rename(columns=dict(index='method'))
        return df_results

    def scores_metrics(self, scores_on_S, scores_on_ZS, show_figure=True,
                       show_legend=True, save_to_file=False):
        """ Evaluate out-of-distribution metrics and the distributions of prediction
        scores for S vs ZS domains.
        """


        # df = pd.DataFrame(columns='AUC,FPR@95TPR,FNR@95TNR'.split(','))
        df = pd.DataFrame(columns='AUC,FPR@95TPR'.split(','))

        mx = max(scores_on_S.max(), scores_on_ZS.max())
        mn = min(scores_on_S.min(), scores_on_ZS.min())
        bins = np.linspace(mn, mx, 40)

        inP, bin_edges_in = np.histogram(scores_on_S, bins, density=True)
        outP, bin_edges_out = np.histogram(scores_on_ZS, bins, density=True)

        if show_figure:
            plt.plot(bins[:-1], outP)
            plt.fill_between(bins[:-1], outP)
            plt.plot(bins[:-1], inP)
            plt.fill_between(bins[:-1], inP, alpha=0.5)

            if show_legend:
                plt.legend('Unseen,Seen'.split(','), bbox_to_anchor=(1, 0.7))

        # Write score distributions to pickle
        if save_to_file:
            fname = get_filename('gater_scores_dist',
                                 gater_model_name=self.name_to_filename(),
                                 expert_name=self._expert_name,
                                 dataset_name=self._dataset_name)
            ml_utils.my_mkdir(os.path.dirname(fname))
            with open(fname, 'wb') as fp:
                ml_utils.pickle_iffnn(dict(bins=bins, outP=outP, inP=inP, inD=scores_on_S,
                                           outD=scores_on_ZS), fp)



        Y = np.block([np.ones_like(scores_on_S), np.zeros_like(scores_on_ZS)])
        scores = np.block([scores_on_S, scores_on_ZS])

        df = pd.concat([df,
                   pd.DataFrame({'AUC': to_percent(roc_auc_score(Y, scores)),
                                 'FPR@95TPR':
                                     to_percent(fpr_at_tpr(Y, scores, tpr_val=0.95,
                                                         pos_label=1)),
                                 # 'FNR@95TNR':
                                 #     to_percent(fpr_at_tpr(1-Y, -scores, tpr_val=0.95,
                                 #                         pos_label=1)),
                                 }
                                , index=[f'{self.name}'])],
                       )#sort=False)

        return df



    ####################################################
    """ Below API to be implemented by child classes """
    ####################################################

    def train(self, pred_S__GZSLval, pred_ZS__GZSLval):
        """
        To implement by child class
        """
        raise NotImplementedError()

    def predict(self, pred_S, pred_ZS, unseen_classes):
        """
        To implement by child class.
        Used only in case the same model is used for train and test data
        """
        raise NotImplementedError()


class maxP(gater_model):
    """

    """

    def __init__(self, data, T=1):
        self.T = T
        super(maxP, self).__init__(data)

    def _set_basic_name(self):
        self.name = super(maxP, self)._set_basic_name() + f': T = {self.T}'

    def predict(self, pred_S, _, __):
        pred_S_at_T = adjust_pdist_temperature(pred_S, self.T)
        return pred_S_at_T.max(axis=1)


class ConfidenceBasedGater(gater_model):
    """ Confidence Based gating model (See paper). Implemented with logistic regression,
    where input features are Top-K (sorted) predictions on different temperatures.
    """

    def __init__(self, data, numK=5, temperatures=(3,), ignore_zs_pred=False):
        self._numK = numK
        self._temperatures = temperatures
        self._model = None
        self._ignore_zs_pred = ignore_zs_pred
        super(ConfidenceBasedGater, self).__init__(data)
    def _set_basic_name(self):
        super(ConfidenceBasedGater, self)._set_basic_name()
        self.name += f': T = {self._temperatures}'
        if self._ignore_zs_pred:
            self.name += ', only Seen'

    def train(self, pred_S__GZSLval, pred_ZS__GZSLval):
        pred_S__Gating_Train = pred_S__GZSLval[self.ix_train, :]
        pred_ZS__Gating_Train = pred_ZS__GZSLval[self.ix_train, :]
        feat = self._feat_gen(pred_S__Gating_Train, pred_ZS__Gating_Train, self._unseen_classes_val)
        self._model = LogisticRegression(random_state=0, C=1, solver='lbfgs', n_jobs=-1,
                                 multi_class='multinomial', verbose=1, max_iter=1000,
                                 ).fit(feat, self.Y_Gating_Train)

    def predict(self, pred_S, pred_ZS, unseen_classes):
        feat = self._feat_gen(pred_S, pred_ZS, unseen_classes)
        return self._model.predict_proba(feat)[:, 1]

    def _feat_gen(self, pred_S, pred_ZS, unseen_classes):
        numK = self._numK

        feat_blocks = []
        if not self._ignore_zs_pred:
            feat_blocks.append(np.sort(pred_ZS[:, unseen_classes], axis=1)[:,::-1][:,:numK])
        for T in self._temperatures:
            feat_blocks.append(np.sort(
                adjust_pdist_temperature(pred_S, T), axis=1)[:, ::-1][:, :numK])

        feat = np.block(feat_blocks)
        return feat


def gating_model_selection_experiment(data, expert_preds, zs_expert_name):
    """ Reproducing Table 3 in paper """

    df_gating_summary = pd.DataFrame()
    gating_models = {}

    CB_Gating_Teq3 = ConfidenceBasedGater(data, numK=5, temperatures=(3,))
    gating_models['CB_Gating_Teq3'] = CB_Gating_Teq3
    CB_Gating_Teq3.train(expert_preds['S__GZSLval'], expert_preds['ZS__GZSLval'])
    df_res = CB_Gating_Teq3.eval(expert_preds, zs_expert_name, save_to_file=True)
    df_gating_summary = df_gating_summary.append(df_res)

    CB_Gating_Teq1 = ConfidenceBasedGater(data, numK=5, temperatures=(1,))
    gating_models['CB_Gating_Teq1'] = CB_Gating_Teq1
    CB_Gating_Teq1.train(expert_preds['S__GZSLval'], expert_preds['ZS__GZSLval'])
    df_res = CB_Gating_Teq1.eval(expert_preds, zs_expert_name, save_to_file=True)
    df_gating_summary = df_gating_summary.append(df_res)

    CB_Gating_Teq3_noZS = ConfidenceBasedGater(data, numK=5, temperatures=(3,),
                                               ignore_zs_pred=True)
    gating_models['CB_Gating_Teq3_noZS'] = CB_Gating_Teq3_noZS
    CB_Gating_Teq3_noZS.train(expert_preds['S__GZSLval'], expert_preds['ZS__GZSLval'])
    df_res = CB_Gating_Teq3_noZS.eval(expert_preds, zs_expert_name, save_to_file=True)
    df_gating_summary = df_gating_summary.append(df_res)

    maxP3 = maxP(data, T=3)
    gating_models['maxP3'] = maxP3
    df_res = maxP3.eval(expert_preds, zs_expert_name, save_to_file=True)
    df_gating_summary = df_gating_summary.append(df_res)

    maxP1 = maxP(data, T=1)
    gating_models['maxP1'] = maxP1
    df_res = maxP1.eval(expert_preds, zs_expert_name, save_to_file=True)
    df_gating_summary = df_gating_summary.append(df_res)

    fname = get_filename('gating_summary', dataset_name=data['dataset_name'], expert_name=zs_expert_name)
    ml_utils.my_mkdir(os.path.dirname(fname))
    with open(fname, 'wb') as fp:
        ml_utils.pickle_iffnn(dict(df_gating_summary=df_gating_summary), fp)

    print(df_gating_summary.drop_duplicates().set_index('method').iloc[::-1, :])

    return gating_models


def adjust_pdist_temperature(pred_at_T_eq_1, T):
    """ Adjust the temperature of a probability distribution """
    if T == 1:
        return pred_at_T_eq_1
    else:
        logits = np.log(pred_at_T_eq_1 + 1e-12)
        return softmax(1e-12 + logits / T)