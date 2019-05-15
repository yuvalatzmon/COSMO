"""
A collection of general purpose utility procedures for machine-learning.
NOTE: Most procedures here are NOT used in LAGO project

Author: Yuval Atzmon
"""

import functools
import re
import collections
import subprocess
import time
import os

import pickle
from contextlib import contextmanager
import json
from copy import deepcopy
from math import log10, floor  # for round_sig
import sys
import glob
import argparse
import bz2

import numpy as np
import pandas as pd

from scipy.interpolate import interpolate
from sklearn.metrics import roc_curve, confusion_matrix

import itertools
import random

def copy_np_arrays_in_dict(d):
    d_copy = {}
    for k,v in d.items():
        d_copy[k] = v.copy()
    return d_copy

def cycle_and_reshuffle(iterable, reshuffle=True):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        if reshuffle:
            random.shuffle(saved)
        for element in saved:
            yield element

def pretty_dict_str(d):
    return json.dumps(d, indent=4)

def grouper_and_reshuffle(batch_size, iterable, cycle=True, reshuffle=True):
    """
    Useful for building a generator for iterating over dataset batches
    Example:
        def LOO_generator(X1, Y1, X2, batch_size, reshuffle=True):
            'iterating on two inputs. Each of different length, and concat them for'
            'every batch'
            n1 = X1.shape[0]
            n2 = X2.shape[0]
            batch_getter1 = grouper_and_reshuffle(batch_size, list(range(n1)), cycle=True, reshuffle=reshuffle)
            batch_getter2 = grouper_and_reshuffle(batch_size, list(range(n2)), cycle=True, reshuffle=reshuffle)
            while True:
                ix1 = batch_getter1.__next__()
                ix2 = batch_getter2.__next__()
                X_batch = np.block([X1[ix1, :].T, X2[ix2, :].T]).T
                Y_batch = np.block([Y1[ix1,], np.zeros((batch_size,))])
                yield X_batch, Y_batch
    """
    "grouper_and_reshuffle(3, 'ABCDEFG') --> ABC DEF"

    if cycle:
        iterable = cycle_and_reshuffle(iterable, reshuffle=reshuffle)
    args = [iter(iterable)] * batch_size
    return zip(*args)

def fpr_at_tpr(y_true, y_pred, tpr_val=0.95, pos_label=1):
    """ Approximate (by interpolating the ROC curve) the False Positive Rate at a
    specific True-Positive Rate.
    """
    fpr, tpr, _= roc_curve(y_true, y_pred, pos_label=pos_label)
    fpr_vs_tpr = interpolate.interp1d(tpr, fpr)
    return float(fpr_vs_tpr(tpr_val))



def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    source https://stackoverflow.com/a/25588487
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import six
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def if_none_get_default(a, default_value):
    return default_value if a is None else a

def pfloor(x, precision):
    """ Floor to precision
    Example:
    >>> pfloor([0.014999, 0.016878, 2.437], precision=2)
    [0.01 0.01 2.43]
    """
    x = np.array(x)
    return np.floor(x*10**precision)/10**precision
def pceil(x, precision):
    """ Ceil to precision
    Example:
    >>> pceil([0.014999, 0.016878, 2.437], precision=2)
    [0.02 0.02 2.44]
    """
    x = np.array(x)
    return np.ceil(x*10**precision)/10**precision

class HiddenPrints:
    """ Source: https://stackoverflow.com/a/45669280

    Example:
        with HiddenPrints():
            print("This will not be printed")

        print("This will be printed as before")
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def dict_to_sorted_str(d):
    """ dict to string, sorted by keys """
    def default(o):
        # a workaround dealing with numpy.int64
        # see https://stackoverflow.com/a/50577730/2476373
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    return json.dumps(d, sort_keys=True, default=default)

def dict_to_pd_query(d):
    items = []
    for k, v in d.items():
        assert (isinstance(k, str))

        if isinstance(v, str):
            s = f'{k}=="{v}"'
        else:
            s = f'{k}=={v}'
        items.append(s)
    return ' and '.join(items)

def parametrized_decorator(dec):
    """a meta-decorator, (a decorator for decorators) for allowing decorators to
    accept arguments """
    @functools.wraps(dec)
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized_decorator
def tf_to_np(f_tf, sess):
    """ A decorator for converting tensorflow functions to numpy
            Example:
                @tf_to_np(K.get_session())
                def my_add(x,y):
                    return tf.add(x,y)

                print(my_add([0,1,2], [1,1,1]))
    """

    @functools.wraps(f_tf)
    def f_wrapper(*args, **kwargs):
        return sess.run(f_tf(*args, **kwargs))

    return f_wrapper


def find(condition):
    """ Similar to MATLAB's find() API: Return the indices where condition is true.
    Source: matplotlib.mlab v2.2 , will be deprecated on v3.1+
    """
    res, = np.nonzero(np.ravel(condition))
    return res

def get_latest_file(fullpath):
    """Returns the name of the latest (most recent) file
    of the joined path(s)
    Modified from: https://codereview.stackexchange.com/a/120500
    """
    list_of_files = glob.glob(fullpath + '/*')  # You may use iglob in Python3
    if not list_of_files:  # I prefer using the negation
        return None  # because it behaves like a shortcut
    latest_full_file = max(list_of_files, key=os.path.getctime)
    return latest_full_file


def last_modification_time(full_path_name):
    file_mod_time = os.stat(full_path_name).st_mtime
    # Time in minutes since last modification of file
    return (time.time() - file_mod_time) / 60.


def safe_path_delete(path_to_del, base_path):
    """ Raise an error if path_to_del is not under base_path"""
    path_to_del = os.path.expanduser(path_to_del)
    if path_to_del.startswith(os.path.expanduser(base_path)):
        print('Deleting %s' % path_to_del)
        print(run_bash('rm -r %s' % path_to_del))
    else:
        raise ValueError('path_to_del %s is not under base path %s' % (
            path_to_del, base_path))

def safe_path_content_delete(path_to_del, base_path, except_list):
    """ Delete the content of a path.
        Raise an error if path_to_del is not under base_path"""
    path_to_del = os.path.expanduser(path_to_del)
    if path_to_del.startswith(os.path.expanduser(base_path)):

        if except_list:
            except_str = ' '.join([f'"{os.path.basename(fname)}"' for fname in except_list])
            print(f'Deleting {path_to_del}/*, except {except_str}')
            print(run_bash(f'find {path_to_del} -type f -not -name {except_str} -delete'))
        else:
            print('Deleting %s/*' % path_to_del)
            print(run_bash('rm -r %s/*' % path_to_del))
    else:
        raise ValueError('path_to_del %s is not under base path %s' % (
            path_to_del, base_path))

def path_exists_with_wildcard(full_path_filename, depth=1):
    dirname = os.path.dirname(os.path.expanduser(full_path_filename))
    filename = os.path.basename(os.path.expanduser(full_path_filename))

    print(f'find {dirname} -maxdepth {depth} -name {filename} ')
    find_results = run_bash(f'find {dirname} -maxdepth {depth} -name {filename} ')
    path_exists = len(find_results) >0

    return path_exists


def touch(full_name):
    # Python 3
    from pathlib import Path

    Path(full_name).touch()


def path_reduce_user(pathname, begins_with_path=True):
    """The opposite of os.path.expanduser()
        Reduces full path names to ~/... when possible """
    if begins_with_path:
        if pathname.startswith(os.path.expanduser('~')):
            pathname = pathname.replace(os.path.expanduser('~'), '~', 1)
    else:
        pathname = pathname.replace(os.path.expanduser('~'), '~')

    return pathname


def get_current_git_hash(dir='./', shorten=True):
    """ Returns a (hex) string with the current git commit hash.
    """
    short = ''
    if shorten:
        short = '--short'

    dir = os.path.expanduser(dir)
    git_hash = run_bash('cd %s && git rev-parse %s HEAD' % (dir, short))
    return git_hash


def get_username():
    """ 
    :return: username by last part of os.path.expanduser('~')
    """""
    return os.path.basename(os.path.expanduser('~'))


def slice_dict_to_tuple(d, keys, none_as_default=False):
    """ Returns a tuple from dictionary values, ordered and slice by given keys
        keys can be a list, or a CSV string
    """
    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == ',' else keys
        keys = re.split(', |[, ]', keys)

    if none_as_default:
        out = [d.get(k, None) for k in keys]
    else:
        out = [d[k] for k in keys]
    return out

def replace_filename_in_fullname(fullname, new_name):
    return os.path.join(os.path.dirname(fullname), new_name)

def replace_file_extension(fname, new_ext):
    """ replace filename extension """
    if new_ext[0] != '.':
        new_ext = '.' + new_ext
    return os.path.splitext(fname)[0] + new_ext


def show_redundant_command_line_args():
    import tensorflow as tf

    FLAGS = tf.flags.FLAGS
    _, parsed_flags = argparse.ArgumentParser().parse_known_args()
    redundant_args = set(
        map(lambda x: re.split('[ =]', x[2:])[0], parsed_flags)) - set(
        vars(FLAGS)['__flags'])
    redundant_args = list(redundant_args)

    msg = 'Passed redundant command line arguments: {}'.format(redundant_args)
    eprint(msg)
    tf.logging.info(msg)


def normalize_data_per_sample(DATA_samples_x_features):
    """ Normalize each sample to zero mean and unit variance
        input data dimensions is samples_x_features
    """
    X = DATA_samples_x_features
    X_zero_mean = ((X.T - X.mean(axis=1)).T)
    X_L2_normed = ((X_zero_mean.T / np.linalg.norm(X_zero_mean, axis=1)).T)
    return X_L2_normed

def join_strings_left_right_align(lhs_str, rhs_str, min_str_length):
    output = lhs_str + ' ' * (min_str_length - len(lhs_str) - len(rhs_str))
    output += rhs_str
    return output

def get_gpus_stat():
    try:
        gpus_stat = run_bash('src/utils/gpustat.py --no-color').split('\n')[1:]
    except RuntimeError:
        gpus_stat = []
    return gpus_stat

def all_gpus_ids():
    return list(range(len(get_gpus_stat())))

def find_available_gpus(mem_threshold=1000):
    """ Iterate on stats per gpu. Return only GPUs that
        used memory is below threshold

        THIS PROCEDURE IS BUGGY when multiple instantiations run in parallel.
        To fix add mutex mechanism
        """

    # Call a script that nicely parse nvidia-smi to GPUs statistics
    gpus_stat = get_gpus_stat()
    gpus_list = []
    # Iterate on stats per gpu.
    for gpu_id, gpu in enumerate(gpus_stat):
        used_mem = int(re.findall('(\d+) / (\d+) MB', gpu)[0][0])
        # Take only GPUs that used memory is below threshold
        print('used_mem=', used_mem)
        if used_mem < mem_threshold:
            gpus_list.append(gpu_id)

    if len(gpus_list) == 0 and len(gpus_stat) > 0:
        raise RuntimeError('No GPUs are available\n'
                           'Uses memory threshold=%d\n'
                           'GPUs stats:\n%s'%(mem_threshold, '\n'.join(gpus_stat)))
    return gpus_list


def sigmoid(x):
    """ Numerically stable numpy sigmoid
        https://stackoverflow.com/a/29863846/2476373
    """
    return np.exp(-np.logaddexp(0, -np.array(x)))



def binary_classification_metrics(y_true, y_pred):
    # Source https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    CM = confusion_matrix(y_true, y_pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return TPR, FPR, TNR, FNR, ACC

def to_percent(val, precision=1):
    if isinstance(val, list) or isinstance(val, tuple):
        val = np.array(val)
    return np.round(100 * val, precision)

def unique_rows_np(a):
    # http://stackoverflow.com/a/31097277
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def is_member(ar1, ar2, assume_unique=False, invert=False, keepdims=False):
    # np.in1d is like MATLAB's ismember
    result = np.in1d(ar1, ar2, assume_unique=assume_unique, invert=invert)
    if keepdims:
        result = result.reshape([-1, 1])

    return result

def is_PID_running(PID, hostname=None):
    # Adpated from http://stackoverflow.com/a/15774758/2476373

    if hostname is None:
        pid_cmd = """ps -p {PID}""".format(PID=PID)
    else:
        pid_cmd = """ssh {hostname} "ps -p {PID}" """.format(PID=PID,
                                                             hostname=hostname)

    return len(run_bash(pid_cmd).split('\n')) > 1


def eprint(*args, **kwargs):
    """ Print to stderr
        http://stackoverflow.com/a/14981125/2476373
    """
    print(*args, file=sys.stderr, **kwargs)


def write_hostname_pid_to_filesystem(filename):
    my_mkdir(os.path.dirname(filename))

    with open(filename, 'w') as f:
        json.dump(dict(hostname=run_bash('hostname'), pid=os.getpid()), f,
                  indent=4)


def read_hostname_pid_from_filesystem(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
    return d['hostname'], d['pid']


def grep(s, pattern):
    """ grep
    """
    # Adapted from http://stackoverflow.com/a/25181706
    return '\n'.join(
        [line for line in s.split('\n') if re.search(pattern, line)])

def epsilon(float_type=np.float64):
    return np.finfo(float_type).eps

def epsilon_smooth_distribution(P, axis=1, sanity_checks=True, float_type=np.float64,
                                atol=1e-6):
    if sanity_checks:
        assert((P<=1).all())
        assert((P >=0).all())

        if not (np.isclose(P.sum(axis=axis), 1, atol=atol)).all():
            print(P.sum(axis=axis))
            raise ValueError('Distribution should be close to 1')

    P_original = P.copy()
    P += epsilon(float_type)
    P /= P.sum(axis=axis, keepdims=1)

    if sanity_checks:
        assert((P<=1).all())
        assert((P>=0).all())
        assert(np.isclose(P, P_original).all())

    return P

def add_epsilon_to_P(P,  axis=None, sanity_checks=True, float_type=np.float64):
    # add / subtract epsilon s.t. 0<P<1
    # set sanity_checks=False for faster exec.

    # check that each value hold 0<=p<=1, (i.e. a legit probability value)
    if sanity_checks:
        assert((P<=1).all())
        assert((P>=0).all())

    # Add epsilon
    P_original = P.copy()
    P += epsilon(float_type)

    # saturate at 1-eps
    P[P >= 1] = 1-epsilon(float_type)

    # make sure above didn't exceed 1 for sum reduction vs some axis
    if axis is not None:
        greater_1_flag = (P.sum(axis=axis) > 1)
        if axis==0:
            P = P.T
        P[greater_1_flag, :] /= P[greater_1_flag, :].sum(axis=axis, keepdims=1)

        if axis==0:
            P = P.T

    if sanity_checks:
        assert(np.isclose(P, P_original).all())
        assert((P<=1).all())
        assert((P>=0).all())

    return P

def pickle_iffnn(data, fp, verbose=1):
    """
    Write data to pickle if fp is not None
    """
    if fp is not None:
        if verbose:
            if '_file' in vars(fp):
                # getting the filename of MongoDB gridfs file
                fname = vars(fp)['_file']['filename']
            else:
                # getting the filename of a os file
                fname = fp.name
            print(f'Writing {fname}')

        fp.write(pickle.dumps(data))
        return True
    return False


def data_to_pkl(data, fname, compress=False):
    # pickle data
    if compress:
        with bz2.open(fname + '.bz2', 'wb') as f:
            pickle.dump(data, f)

    else:
        with open(fname, 'wb') as f:
            pickle.dump(data, f)


def dict_to_arg_flags_str(flags_dict):
    """
    Converts a dictionary to a commandline arguments string
    in the format '--<key0>=value0 --<key1>=value1 ...'
    """
    return ' '.join(
        ['--{}={}'.format(k, flags_dict[k]) for k in flags_dict.keys()])


def tensorflow_flags_to_str(tf_flags):
    """ Convert tensorflow FLAGS to a commandline arguments string
    in the format '--<flag0>=value0 --<flag1>=value1 ...' """
    flags_dict = vars(tf_flags)['__flags']
    return dict_to_arg_flags_str(flags_dict)


def generate_split_indices(samples_ids, seed, split_ratios):
    """
    Generate indices for a cross validation (XV) split, given ids of samples, or just number of samples

    :param samples_ids:  given ids of samples (list or ndarray), or a scalar indicating number of samples
    :param seed: random seed
    :param split_ratios: a list of ratios for XV sets
    :return: a list of lists of indices per XV set

    Example:
        train_ids, val_ids, test_ids = generate_split_indices(Nimages, seed=111, split_ratios=[0.6, 0.2, 0.2])

    """
    samples_ids = np.array([samples_ids]).astype(int).flatten()

    if samples_ids.size == 1:
        samples_ids = np.array(range(samples_ids[0])).astype(int).flatten()

    nsamples = samples_ids.size

    # Set a seed for the split, with a temporary context
    with temporary_random_seed(seed):
        # Calc number of samples per set
        n_indices = np.round(nsamples * np.array(split_ratios)).astype(int)
        n_indices[-1] = nsamples - np.sum(n_indices[0:-1])

        # Draw a random permutation of the samples IX
        perm = np.random.permutation(nsamples)
        # Split the random permutation IX sequentially according to num of samples per set
        ix0 = 0
        ix_set = []
        for N_ix in n_indices.tolist():
            ix1 = ix0 + int(N_ix)
            ix_set += [perm[ix0:ix1].tolist()]
            ix0 = ix1

    # Assign ids according to the split IX
    ids_set = []
    for ix in ix_set:
        ids_set += [np.take(samples_ids, ix).tolist()]

    return ids_set


def my_mkdir(dir_name):
    # mkdir if not exist
    return os.makedirs(dir_name, exist_ok=True)

def bring_list_element_to_front(mylist, targetvalue, inplace=False):
    if not inplace:
        mylist = deepcopy(mylist)
    mylist.insert(0, mylist.pop(mylist.index(targetvalue)))
    return mylist

def load_dict(fname, var_names, load_func=pickle.load):
    """ Loads specific keys from a dictionary that was to a file
    :type fname: file name
    :type var_names: variables to retrieve. Can be a list or comma seperated string
          e.g. 'a, b,c' or ['a', 'b', 'c']
    :param load_func: default: pickle.load
    """
    if type(var_names) == str:
        var_names = re.split(', ?[, ]?', var_names)
    with open(fname, "rb") as f:
        data_dict = load_func(f)
    assert isinstance(data_dict, dict)
    return tuple([data_dict[var] for var in var_names])


def cond_load_dict(fname, var_names, do_force=False, load_func=pickle.load):
    """
    usage:
    data_dict, do_stage = cond_load_dict(fname, 'x,y,z', do_force):
       if do_stage:
           data_dict = <calculate the data>
           <save the data to fname>

       return data_dict, do_stage
    """
    do_stage = True
    if type(var_names) == str:
        var_names = re.split(', ?[, ]?', var_names)

    # Python 2 to 3 compatibility
    if sys.version_info[0] > 2 and load_func == pickle.load:
        def load_func23(f):
            return load_func(f, encoding='latin1')
    else:
        load_func23 = load_func

    # noinspection PyBroadException
    try:
        with open(fname, "rb") as f:
            data_dict = load_func23(f)
        assert isinstance(data_dict, dict)

        # check if all required var names are members of loaded data
        if np.in1d(var_names, list(data_dict.keys())).all():
            do_stage = False
    except:
        do_stage = True

    if do_stage or do_force:
        return tuple([True] + [None for _ in var_names])
    else:
        # noinspection PyUnboundLocalVariable
        return tuple([False] + [data_dict[var] for var in var_names])


@contextmanager
def temporary_random_seed(seed):
    """ A context manager for a temporary random seed (only within context)
        When leaving the context the numpy random state is restored
        Inspired by http://stackoverflow.com/q/32679403
    """
    state = np.random.get_state()
    np.random.seed(seed)
    yield None
    np.random.set_state(state)


def remove_dict_key(d, key):
    """remove_dict_key(d, key)
       Removes a key-value from a dict.
       Do nothing if key does not exist
       :param d: dictionary
       :type key:
       """

    if key in d:
        d.pop(key)

    return d


# noinspection PyPep8Naming
def ismatrix(M):
    if len(M.shape) == 1 or any(np.array(M.shape) == 1):
        return False
    else:
        return True


# noinspection PyPep8Naming
def get_prec_at_k(a_ind):
    # evaluate 'precision at k' for all k's (number of positive indications (of higest k scores) / k)
    """

    :type a_ind: matrix
    """
    assert (ismatrix(a_ind))

    K, N = a_ind.shape
    a_ind = a_ind.astype('float32')

    precision_at_all_ks = (a_ind.cumsum(0).T / range(1, K + 1)).mean(
        0)  # used .T for broadcasting
    # alternative: a_ind.sum(1).cumsum()/range(1,K+1)/N

    return precision_at_all_ks


def run_bash(cmd, raise_on_err=True, raise_on_warning=False, versbose=True):
    """ This function takes Bash commands and return their stdout
    Returns: string (stdout)
    :type cmd: string
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, executable='/bin/bash')
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out = p.stdout.read().strip().decode('utf-8')
    err = p.stderr.read().strip().decode('utf-8')
    if err and raise_on_err:
        do_raise = True
        if 'warning' in err.lower():
            do_raise = raise_on_warning
            if versbose:
                print('command was: {}'.format(cmd))
            eprint(err)

        if do_raise:
            if versbose:
                print('command was: {}'.format(cmd))
            raise RuntimeError(err)

    return out  # This is the stdout from the shell command


def build_string_from_dict(d, sep='__'):
    """
     Builds a string from a dictionary.
     Mainly used for formatting hyper-params to file names.
     Key-Value(s) are sorted by the key, and dictionaries with
     nested structure are flattened.

    Args:
        d: dictionary

    Returns: string
    :param d: input dictionary
    :param sep:

    """
    fd = _flatten_dict(d)
    return sep.join(
        ['{}={}'.format(k, _value2str(fd[k])) for k in sorted(fd.keys())])


def slice_dict(d, keys_list):
    return {k: v for k, v in d.items() if k in keys_list}


def grouped(iterable, n, incomplete_tuple_ok=True):
    """ http://stackoverflow.com/a/5389547/2476373and http://stackoverflow.com/a/38059462
    s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...\

    Usage example:
    for x, y in grouped(range(10:21), 2):
        print "%d + %d = %d" % (x, y, x + y)

    """

    if incomplete_tuple_ok:
        def zip_discard_compr(*iterables):
            """izip_longest which discards missing values and produce incomplete tuples"""
            sentinel = object()
            return [[entry for entry in iterable if entry is not sentinel]
                    for iterable in
                    itertools.izip_longest(*iterables, fillvalue=sentinel)]

        return zip_discard_compr(*[iter(iterable)] * n)
    else:
        return itertools.izip(*[iter(iterable)] * n)


def join_path_with_extension(path_parts, extension=None):
    """ Join path parts and safely adding extension

    path_parts: list of parts of path
    extension: file extension, if set to None, just calls os.path.join(*path_parts)

    returns full path with extension

    Examples:

    >>> join_path_with_extension(['a', 'b', 'c'], 'jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c'], '.jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c.jpg'], 'jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c.jpg'], '.jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'cjpg'], '.jpg')
    'a/b/cjpg.jpg'
    >>> join_path_with_extension(['a', 'b', 'c'])
    'a/b/c'
    """
    full_path = os.path.join(*path_parts)

    if extension is not None:
        if extension[0] != '.':
            extension = '.' + extension

        if not full_path.endswith(extension):
            full_path += extension

    return full_path


def round_sig(x, sig=2):
    # http://stackoverflow.com/a/3413529/2476373
    if x == 0:
        return 0
    else:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)


def dataframe_from_json(fname, columns):
    if os.path.exists(fname):
        df = pd.read_json(fname)
    else:
        df = pd.DataFrame(None, columns=columns)

    return df


def dataframe_from_csv(fname, columns, sep=','):
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep=sep)
    else:
        df = pd.DataFrame(None, columns=columns)

    return df


def read_json(fname, return_None_if_not_exist=True):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)

    else:
        if not return_None_if_not_exist:
            raise Exception('Error, no such file: %s' % fname)
        else:
            data = None
    return data


def merge_dict_list(dict_list):
    # http://stackoverflow.com/a/3495415/2476373
    return dict(kv for d in dict_list for kv in d.iteritems())


def read_modify_write_json(fname, update_dict, create_if_not_exist=True):
    """
    Read-Modify-Write a JSON file
    NOTE that it only modify on 1 level. Nested dicts are over written.
    Args:
      fname: filename
      update_dict: update values
      create_if_not_exist: default = True

    Returns:

    """
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)
    else:
        if not create_if_not_exist:
            raise Exception('Error, no such file: %s' % fname)
        else:
            data = {}
            with open(fname, 'w') as f:
                f.write(json.dumps(data, indent=4))

    data.update(update_dict)
    with open(fname, 'w') as f:
        f.write(json.dumps(data, indent=4))


# Homemade version of matlab tic and toc functions
# from http://stackoverflow.com/a/18903019/2476373
def tic():
    # noinspection PyGlobalUndefined
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(
            time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def save_notebook():
    from IPython.core.display import Javascript, display

    return display(Javascript("IPython.notebook.save_notebook()"),
                   include=['application/javascript'])

def display_side_by_side(*args, escape=False):
    from IPython.display import display_html

    html_str=''
    for df in args:
        html_str+=df.to_html(escape=escape)
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def output_HTML(read_file, output_file):
    from nbconvert import HTMLExporter
    import codecs
    import nbformat
    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)


# --- Auxilary functions ---


def _flatten_dict(d, parent_key='', sep='_'):
    # from http://stackoverflow.com/a/6027615/2476373
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _value2str(val):
    if isinstance(val, float):  # and not 1e-3<val<1e3:
        # %g means: "Floating point format.
        # Uses lowercase exponential format if exponent is less than -4 or not less than precision,
        # decimal format otherwise."
        val = '%g' % val
    else:
        val = '{}'.format(val)
    val = re.sub('\.', '_', val)
    return val


