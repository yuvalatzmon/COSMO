import numpy as np
import pandas as pd
from sklearn.metrics import auc


def AUSUC(Acc_tr, Acc_ts):

    """ Calc area under seen-unseen curve """

    # Sort by X axis
    X_sorted_arg = np.argsort(Acc_tr)
    sorted_X = np.array(Acc_tr)[X_sorted_arg]
    sorted_Y = np.array(Acc_ts)[X_sorted_arg]

    # zero pad
    leftmost_X, leftmost_Y = 0, sorted_Y[0]
    rightmost_X, rightmost_Y = sorted_X[-1], 0
    sorted_X = np.block([np.array([leftmost_X]), sorted_X, np.array([rightmost_X])])
    sorted_Y = np.block([np.array([leftmost_Y]), sorted_Y, np.array([rightmost_Y])])

    # eval AUC
    AUSUC = auc(sorted_X, sorted_Y)

    return AUSUC

def calc_cs_ausuc(pred, y_gt, seen_classses, unseen_classses, gamma_range=None, verbose=True):
    if gamma_range is None:
        mx = pred.max()
        delta = mx / 100
        gamma_range = np.arange(-mx, mx, delta)

    zs_metrics = ZSL_Metrics(seen_classses, unseen_classses)
    Acc_tr_values = []
    Acc_ts_values = []

    for gamma in gamma_range:
        cs_pred = pred.copy()
        cs_pred[:, seen_classses] -= gamma

        Acc_tr, Acc_ts, H = zs_metrics.generlized_scores(y_gt, cs_pred)
        Acc_tr_values.append(Acc_tr)
        Acc_ts_values.append(Acc_ts)

    cs_ausuc = AUSUC(Acc_tr=Acc_tr_values, Acc_ts=Acc_ts_values)
    if min(Acc_tr_values) > 0.01:
        raise RuntimeError(f'CS AUSUC: Increase gamma range (add low values), because min(Acc_tr_values) equals {min(Acc_tr_values)}')
    if min(Acc_ts_values) > 0.01:
        raise RuntimeError(f'CS AUSUC: Increase gamma range (add high values), because min(Acc_ts_values) equals {min(Acc_ts_values)}')
    if verbose:
        print(f'AUSUC (by Calibrated Stacking) = {100*cs_ausuc:.1f}')
    return cs_ausuc


class ZSL_Metrics():
    def __init__(self, seen_classes, unseen_classes, report_entropy=False):
        self._seen_classes = np.sort(seen_classes)
        self._unseen_classes = np.sort(unseen_classes)
        self._n_seen = len(seen_classes)
        self._n_unseen = len(unseen_classes)
        self._report_entropy = report_entropy

        assert(self._n_seen == len(np.unique(seen_classes))) # sanity check
        assert(self._n_unseen == len(np.unique(unseen_classes))) # sanity check


    def unseen_balanced_accuracy(self, y_true, pred_softmax):
        Acc_zs, Ent_zs =  self._subset_classes_balanced_accuracy(y_true, pred_softmax,
                                                      self._unseen_classes)
        if self._report_entropy:
            return Acc_zs, Ent_zs
        else:
            return Acc_zs

    def seen_balanced_accuracy(self, y_true, pred_softmax):
        Acc_seen, Ent_seen = self._subset_classes_balanced_accuracy(y_true,
                                                                    pred_softmax,
                                                      self._seen_classes)
        if self._report_entropy:
            return Acc_seen, Ent_seen
        else:
            return Acc_seen

    def generlized_scores(self, y_true, pred_softmax):

        Acc_ts, Ent_ts = self._generalized_unseen_balanced_accuracy(y_true,
                                                                    pred_softmax)
        Acc_tr, Ent_tr = self._generalized_seen_balanced_accuracy(y_true, pred_softmax)
        H = 2*Acc_tr*Acc_ts/(Acc_tr + Acc_ts)
        Ent_H = 2*Ent_tr*Ent_ts/(Ent_tr + Ent_ts)

        if self._report_entropy:
            return Acc_ts, Acc_tr, H, Ent_ts, Ent_tr, Ent_H
        else:
            return Acc_ts, Acc_tr, H

    def _generalized_unseen_balanced_accuracy(self, y_true, pred_softmax):
        return self._generalized_subset_balanced_accuracy(y_true, pred_softmax,
                                                      self._unseen_classes)

    def _generalized_seen_balanced_accuracy(self, y_true, pred_softmax):
        return self._generalized_subset_balanced_accuracy(y_true, pred_softmax,
                                                          self._seen_classes)

    def _generalized_subset_balanced_accuracy(self, y_true, pred_softmax, subset_classes):
        is_member = np.in1d # np.in1d is like MATLAB's ismember
        ix_subset_samples = is_member(y_true, subset_classes)

        y_true_subset = y_true[ix_subset_samples]
        all_classes = np.sort(np.block([self._seen_classes, self._unseen_classes]))
        y_pred = all_classes[(pred_softmax[:, all_classes]).argmax(axis=1)]
        y_pred_subset = y_pred[ix_subset_samples]

        Acc = float(xian_per_class_accuracy(y_true_subset, y_pred_subset,
                                        len(subset_classes)))
        # Ent = float(entropy2(pred_softmax[ix_subset_samples, :][:, all_classes]).mean())
        Ent = 0*Acc + 1e-3 # disabled because its too slow
        return Acc, Ent

    def _subset_classes_balanced_accuracy(self, y_true, pred_softmax, subset_classes):
        is_member = np.in1d # np.in1d is like MATLAB's ismember
        ix_subset_samples = is_member(y_true, subset_classes)

        y_true_zs = y_true[ix_subset_samples]
        y_pred = subset_classes[(pred_softmax[:, subset_classes]).argmax(axis=1)]
        y_pred_zs = y_pred[ix_subset_samples]

        Acc = float(xian_per_class_accuracy(y_true_zs, y_pred_zs, len(subset_classes)))
        # Ent = float(entropy2(pred_softmax[:, subset_classes]).mean())
        Ent = 0*Acc + 1e-3 # disabled because its too slow
        return Acc, Ent


def xian_per_class_accuracy(y_true, y_pred, num_class=None):
    """ A balanced accuracy metric as in Xian (CVPR 2017). Accuracy is
        evaluated individually per class, and then uniformly averaged between
        classes.
    """

    y_true = y_true.flatten().astype('int32')
    # # if num_class is None:
    # #     num_class = len(np.unique(np.block([y_true, y_pred])))
    # print(num_class)
    ## my method is faster
    # return balanced_accuracy_score(y_true, y_pred, num_class=num_class)

    if num_class is None:
        num_class = len(np.unique(np.block([y_true, y_pred])))
        # num_class = len(counts_per_class)  # e.g. @CUB: 50, 100, 150
        # num_class = len(np.unique(y_true))

    max_class_id = 1+max([num_class, y_true.max(), y_pred.max()])

    counts_per_class_s = pd.Series(y_true).value_counts()
    counts_per_class = np.zeros((max_class_id,))
    counts_per_class[counts_per_class_s.index] = counts_per_class_s.values

    # accuracy = ((y_pred == y_true) / np.array(
    #     [counts_per_class[y] for y in y_true])).sum() / num_class

    accuracy = (1.*(y_pred == y_true) / counts_per_class[y_true]).sum() / num_class
    return accuracy.astype('float32')

