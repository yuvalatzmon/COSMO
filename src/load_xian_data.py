"""
A python module to load zero-shot dataset of Xian (CVPR 2017).

Note that it requires altering the original directory structure of Xian.
See README.md for details.

Author: Yuval Atzmon
"""
import struct

import h5py
import pandas as pd
import numpy as np
import scipy.io

from src.utils import ml_utils
from src import definitions

# Get values for the relevant command line arguments
common_args, unknown_args = definitions.get_common_commandline_args()



import os



class XianDataset(definitions.ZSLData):
    def __init__(self, use_trainval_set=common_args.use_trainval_set,
                 transfer_task=common_args.transfer_task, val_fold=0,
                 use_xian_normed_class_description=common_args.use_xian_normed_class_description):
        self.use_xian_normed_class_description = use_xian_normed_class_description

        self._df_class_descriptions_by_attributes = None
        self._data_dir = os.path.expanduser(common_args.data_dir)
        self._get_dir = self._init_dirs()


        data, self._att = self._load_raw_xian2017()
        self._all_X = data['features'].T
        self._all_Y = data['labels']
        self._all_image_files = np.array(data['image_files'])

        self._classname_to_id = self._att['classname_to_id']
        self.id_to_classname = self._att['id_to_classname']

        # ( train / val ) split id
        self._val_fold = val_fold

        # With this flag we use the *test set* to judge the performance
        # Use the test set split by :
        # (1) Replacing train indices with (train bitwiseOR val) indices
        # (2) Replacing validation set indices with the test set indices
        self._use_trainval_set = use_trainval_set
        self._transfer_task = transfer_task
        if not transfer_task in  ['ZSL', 'GZSL']:
            raise ValueError(f'--transfer_task={self._transfer_task}')

        self._attributes_name = None

    def dataset_name(self):
        raise NotImplementedError('To implement by child class')

    def _init_dirs(self):
        get_dir = dict(meta=os.path.join(self._data_dir, 'meta'),
                       xian_raw_data=os.path.join(self._data_dir, 'xian2017'))
        return get_dir

    @staticmethod
    def _matlab_ids_to_bool(matlab_ids, bool_vec_length):
        """ Converts a list of ids in MATLAB format (indexing starts from 1) to a
            boolean numpy vector (indexing starts from 0).
        """
        matlab_ids = np.array(matlab_ids) # cast to np array

        # init indices bool vector
        ix = np.zeros(bool_vec_length).astype(bool)
        # set relevant indices to True
        ix[matlab_ids - 1] = True # -1 because MATLAB indexing starts from 1
        return ix

    def get_data(self):
        # region: Get split indices of Xian
        data_dir = self._get_dir['xian_raw_data']
        val_fold_id = 1 + self._val_fold

        if self.dataset_name() != 'FLO':
            train_class_names = pd.read_csv(
                os.path.join(data_dir, 'trainclasses%d.txt' % val_fold_id),
                names=[]).index.tolist()
            val_class_names = pd.read_csv(
                os.path.join(data_dir, 'valclasses%d.txt' % val_fold_id),
                names=[]).index.tolist()
            test_class_names = pd.read_csv(
                os.path.join(data_dir, 'testclasses.txt'), names=[]).index.tolist()

            def class_names_to_ids(class_names_list):
                return set([self._classname_to_id[class_name] for class_name in
                            class_names_list])

            train_classes_ids = class_names_to_ids(train_class_names)
            val_classes_ids = class_names_to_ids(val_class_names)
            test_classes_ids = class_names_to_ids(test_class_names)

            #### Sanity check
            tvc_names = pd.read_csv(os.path.join(data_dir, 'trainvalclasses.txt'),
                                    names=[]).index.tolist()
            assert (class_names_to_ids(tvc_names) == train_classes_ids.union(
                val_classes_ids))

            ### Sanity check end

            def get_boolean_indices(set_ids):
                return np.array(list(label in set_ids for label in self._all_Y))

            ix_train = get_boolean_indices(train_classes_ids)
            ix_val = get_boolean_indices(val_classes_ids)
            ix_test = get_boolean_indices(test_classes_ids)

            ### Sanity check: No overlap between sets
            assert (np.dot(ix_train, ix_val) == 0)
            assert (np.dot(ix_train, ix_test) == 0)
            assert (np.dot(ix_val, ix_test) == 0)

            # Sanity Check: Verify correspondence to train_loc, val_loc, ..

            if self._val_fold == 0:
                assert ((ml_utils.find(ix_train) + 1 == self._att['train_loc']).all())
                assert ((ml_utils.find(ix_val) + 1 == self._att['val_loc']).all())

            assert ((ml_utils.find(ix_test) + 1 == self._att['test_unseen_loc']).all())
            ### End sanity checks

        # endregion: Get split indices of Xian

        self._get_official_class_descriptions_by_attributes()

        # Split the data according to the transfer task
        if self._transfer_task == 'ZSL':
            # classic zero-shot task

            # With _use_trainval_set flag as True, we use the *test set* to judge the
            # performance. Therefore, here we do the following:
            # (1) Join the train+val sets, to be the train set
            # (2) Replace the validation set indices to be the test set indices,
            #     because the evaluations are always performed on what are set to be
            #     the validation set indices. With this setting we will run the
            #     evaluations on the test set.
            if self._use_trainval_set:
                # Replacing train indices with (train bitwiseOR val) indices
                ix_train = np.bitwise_or(ix_train, ix_val)
                # Replacing validation set indices with the test set indices
                ix_val = ix_test

                ### Sanity check: No overlap between sets
                assert (np.dot(ix_train, ix_val) == 0)
                ### End Sanity Check

            # Seen-train data is all seen data
            X_seen_train = self._all_X[ix_train, :]
            Y_seen_train = np.array(self._all_Y)[ix_train]
            F_seen_train = np.array(self._all_image_files)[ix_train]

            # empty arrays for Seen-Val set
            X_seen_val = self._all_X[0:0, :]
            Y_seen_val = np.array([])
            F_seen_val = np.array([])

            # empty arrays for Unseen-train set
            X_unseen_train = self._all_X[0:0, :]
            Y_unseen_train = np.array([])
            F_unseen_train = np.array([])


            # unseen-val data is all unseen data
            X_unseen_val = self._all_X[ix_val, :]
            Y_unseen_val = np.array(self._all_Y)[ix_val]
            F_unseen_val = np.array(self._all_image_files)[ix_val]

        elif self._transfer_task == 'GZSL':
            # Generalized zero-shot task
            num_samples = len(self._att['trainval_loc']) + \
                          len(self._att['test_seen_loc']) + \
                          len(self._att['test_unseen_loc'])
            if self.dataset_name() != 'FLO':
                assert(num_samples == ix_train.shape[0])

            if self._use_trainval_set:
                # With _use_trainval_set flag as True, we use the *test set* to judge the
                # performance.

                # NOTE: ix_train OR ix_val = ix_seen_trainval OR ix_seen_test
                ix_seen_trainval = self._matlab_ids_to_bool(self._att['trainval_loc'], num_samples)
                ix_seen_test = self._matlab_ids_to_bool(self._att['test_seen_loc'], num_samples)


                ix_unseen_test = self._matlab_ids_to_bool(self._att['test_unseen_loc'],
                                                     num_samples)
                ### Sanity checks: No overlap between sets
                assert (np.dot(ix_seen_trainval, ix_seen_test) == 0)
                assert(np.dot(ix_unseen_test, np.bitwise_or(ix_seen_trainval,
                                                           ix_seen_test)) == 0)
                ### Sanity check: ix_train OR ix_val == ix_seen_trainval OR ix_seen_test
                if self.dataset_name() != 'FLO':
                    assert((ml_utils.find(np.bitwise_or(ix_seen_trainval, ix_seen_test)) ==
                            ml_utils.find(np.bitwise_or(ix_train, ix_val))).all())

                # Seen-train data takes seen_trainval indices
                X_seen_train = self._all_X[ix_seen_trainval, :]
                Y_seen_train = np.array(self._all_Y)[ix_seen_trainval]
                F_seen_train = np.array(self._all_image_files)[ix_seen_trainval]


                # Seen-val data takes seen_test indices
                X_seen_val = self._all_X[ix_seen_test, :]
                Y_seen_val = np.array(self._all_Y)[ix_seen_test]
                F_seen_val = np.array(self._all_image_files)[ix_seen_test]

                # empty arrays for Unseen-train set (for future compatibility with FSL)
                X_unseen_train = self._all_X[0:0, :]
                Y_unseen_train = np.array([])
                F_unseen_train = np.array([])

                # Unseen-val data takes unseen_test indices
                X_unseen_val = self._all_X[ix_unseen_test, :]
                Y_unseen_val = np.array(self._all_Y)[ix_unseen_test]
                F_unseen_val = np.array(self._all_image_files)[ix_unseen_test]

            else:
                # NOTE: ix_train OR ix_val = ix_seen_trainval OR ix_seen_test
                ids_seen_trainval = self._att['trainval_loc'] # ids in MATLAB format
                ids_seen_test = self._att['test_seen_loc']  # ids in MATLAB format
                ids_all_seen_train = self._att['train_loc']  # in MATLAB format
                ids_all_unseen_val = self._att['val_loc']  # in MATLAB format
                num_seen_test_samples = len(ids_seen_test)

                # Draw seen val set (in MATLAB format)
                with ml_utils.temporary_random_seed(common_args.seen_val_seed):
                    # draw from ids_seen_trainval, to ensure ids_seen_val does *not*
                    # overlaps ids_seen_test

                    ids_seen_trainval_train = list(set(ids_seen_trainval).intersection(
                        ids_all_seen_train))
                    ids_seen_val = np.random.choice(ids_seen_trainval_train,
                                                    num_seen_test_samples,
                                                    replace=False)
                ix_seen_val = self._matlab_ids_to_bool(ids_seen_val, num_samples)


                # ids_seen_train = set(ids_seen_trainval_train) - set(ids_seen_val)
                ids_seen_train  = list( set(ids_seen_trainval_train).difference(ids_seen_val) )
                ix_seen_train = self._matlab_ids_to_bool(ids_seen_train, num_samples)

                ids_unseen_val = list(set(ids_all_unseen_val).difference(ids_seen_test))
                ix_unseen_val = self._matlab_ids_to_bool(ids_unseen_val, num_samples)

                ### Sanity checks: No overlap between sets
                ix_seen_test = self._matlab_ids_to_bool(self._att['test_seen_loc'],
                                                        num_samples)
                assert (np.dot(ix_seen_test, ix_seen_val) == 0)
                assert (np.dot(ix_seen_train, ix_seen_val) == 0)
                assert (np.dot(ix_seen_train, ix_seen_test) == 0)
                assert (np.dot(ix_unseen_val, ix_seen_test) == 0)
                assert (len(set(ids_seen_trainval_train).difference(ids_seen_val))
                        + len(ids_seen_val)
                        + len(set(ids_seen_test).intersection(ids_all_seen_train))
                        == len(ids_all_seen_train))
                assert(len(ids_seen_trainval_train) +
                       len(set(ids_all_unseen_val).difference(set(ids_seen_test)))
                       == len(ids_seen_trainval))
                if self.dataset_name() != 'FLO':
                    assert(np.dot(ix_val, np.bitwise_or(ix_seen_train, ix_seen_val)) == 0)
                    ### Sanity check:
                    # ix_train == ix_seen_train OR ix_seen_val OR (ix_seen_test w/o unseen_val)
                    assert(((ml_utils.find(np.bitwise_or(
                        np.bitwise_or(ix_seen_train, ix_seen_val),
                        self._matlab_ids_to_bool(
                            list(set(ids_seen_test).difference(ids_all_unseen_val)),
                            num_samples))) ==
                             ml_utils.find(ix_train)).all()))
                    # ix_train == ix_seen_train OR ix_seen_val OR (ix_seen_test
                    # intersect all_seen_train)
                    assert((ml_utils.find(np.bitwise_or(
                        np.bitwise_or(ix_seen_train, ix_seen_val),
                        self._matlab_ids_to_bool(
                            list(set(ids_seen_test).intersection(ids_all_seen_train)),
                            num_samples))) ==
                            ml_utils.find(ix_train)).all())


                    assert((ml_utils.find(np.bitwise_or(ix_unseen_val,
                                                        self._matlab_ids_to_bool(
                            list(set(ids_seen_test).intersection(ids_all_unseen_val)),
                            num_samples))) ==
                            ml_utils.find(ix_val)).all())
                    # assert ((ix_val == ix_unseen_val).all())

                # Seen-train data takes seen_trainval indices
                X_seen_train = self._all_X[ix_seen_train, :]
                Y_seen_train = np.array(self._all_Y)[ix_seen_train]
                F_seen_train = np.array(self._all_image_files)[ix_seen_train]

                # Seen-val data takes seen_test indices
                X_seen_val = self._all_X[ix_seen_val, :]
                Y_seen_val = np.array(self._all_Y)[ix_seen_val]
                F_seen_val = np.array(self._all_image_files)[ix_seen_val]

                # empty arrays for Unseen-train set (for future compatibility with FSL)
                X_unseen_train = self._all_X[0:0, :]
                Y_unseen_train = np.array([])
                F_unseen_train = np.array([])

                # Unseen-val data takes unseen_test indices
                X_unseen_val = self._all_X[ix_unseen_val, :]
                Y_unseen_val = np.array(self._all_Y)[ix_unseen_val]
                F_unseen_val = np.array(self._all_image_files)[ix_unseen_val]

        elif self._transfer_task == 'FSL':
            raise NotImplementedError()


        assert(len(Y_seen_train)==len(F_seen_train))
        assert(len(Y_seen_val)==len(F_seen_val))
        assert(len(Y_unseen_train)==len(F_unseen_train)) # for future FSL task
        assert(len(Y_unseen_val)==len(F_unseen_val))
        data = dict(X_seen_train=X_seen_train, Y_seen_train=Y_seen_train,
                    F_seen_train=F_seen_train,
                    X_seen_val=X_seen_val, Y_seen_val=Y_seen_val, F_seen_val=F_seen_val,
                    X_unseen_train=X_unseen_train, Y_unseen_train=Y_unseen_train, # for FSL task
                    F_unseen_train=F_unseen_train, # for FSL task
                    X_unseen_val=X_unseen_val, Y_unseen_val=Y_unseen_val,
                    F_unseen_val=F_unseen_val,
                    df_class_descriptions_by_attributes=
                    self._df_class_descriptions_by_attributes,
                    attributes_name=self._attributes_name,
                    )

        return data


    def _load_raw_xian2017(self):
        """
        load data as in Xian 2017 (images as ResNet101 vectors, different labels indexing of classes and unique dataset splits)
        :return:
            df_xian ?
        """
        """ 
            resNet101.mat includes the following fields:
            -features: columns correspond to image instances
            -labels: label number of a class is its row number in allclasses.txt
            -image_files: image sources  


            att_splits.mat includes the following fields:
            -att: columns correpond to class attributes vectors, following the classes order in allclasses.txt 
            -trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
            -test_seen_loc: instances indexes of test set features for seen classes
            -test_unseen_loc: instances indexes of test set features for unseen classes
            -train_loc: instances indexes of train set features (subset of trainval_loc)
            -val_loc: instances indexes of val set features (subset of trainval_loc)        
        """

        data_dir = self._get_dir['xian_raw_data']
        data, att = get_xian2017_data(data_dir)
        return data, att

    def _prepare_classes(self):
        """
        Returns a dataframe with meta-data with the following columns:
        class_id (index) | name | clean_name

        for clean_name column: removing numbers, lower case, replace '_' with ' ', remove trailing spaces
        """
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir, 'classes.txt')
        classes_df = pd.read_csv(fname, sep='[\s]',
                                 names=['class_id', 'name'],
                                 engine='python')
        self._classes_df = classes_df.set_index('class_id')

        return self._classes_df.name.tolist()

    def _load_attribute_names(self):
        # Load attribute names. Format is: id   'group_name:attribute_name'
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir,
                             'attribute_names_with_semantic_group.txt')
        df_attributes_list = \
            pd.read_csv(fname, delim_whitespace=True,
                        names=['attribute_id',
                               'attribute_name']).set_index('attribute_id')
        return df_attributes_list

    def _get_official_class_descriptions_by_attributes(self):
        data_dir = self._get_dir['meta']
        class_names_and_order = self._prepare_classes()


        # Load class descriptions
        fname = os.path.join(data_dir, 'class_descriptions_by_attributes.txt')
        df_class_descriptions_by_attributes = pd.read_csv(fname, header=None,
                                                          delim_whitespace=True,
                                                          error_bad_lines=False)

        df_attributes_list = self._load_attribute_names()

        # casting from percent to [0,1]
        df_class_descriptions_by_attributes /= 100.

        # Set its columns to attribute names
        df_class_descriptions_by_attributes.columns = \
            df_attributes_list.attribute_name.tolist()

        # Setting class id according to Xian order
        df_class_descriptions_by_attributes.index = \
            [self._classname_to_id[class_name]
             for class_name in class_names_and_order]

        # Sort according to Xian order
        df_class_descriptions_by_attributes = \
            df_class_descriptions_by_attributes.sort_index(axis=0)

        # xian_class_names_and_order = self._att['allclasses_names']
        # df_class_descriptions_by_attributes = \
        #     df_class_descriptions_by_attributes.loc[xian_class_names_and_order]


        ### Sanity check:
        # Make sure that when L2 normalizing official class results with Xian provided description

        # Extract only matrix values
        official_values = \
            df_class_descriptions_by_attributes.copy().values.T
        # L2 norm
        official_values_l2 = official_values / np.linalg.norm(official_values,
                                                              axis=0),
        # Compare to xian provided description
        check_official_class_description_l2_norm_equals_xian2017 = \
            np.isclose( official_values_l2, self._att['att'], rtol=1e-4).all()
        assert (check_official_class_description_l2_norm_equals_xian2017)
        ### End sanity check

        # If use_xian_normed_class_description=True, then replace official
        # values with Xian L2 normalized class description.
        #
        # NOTE: This is only provided to support other ZSL methods within this
        # framework. LAGO can not allow using Xian (CVPR 2017) class description,
        # because this description is a **L2 normalized** version of the mean
        # attribute values. Such normalization removes the probabilistic meaning
        # of the attribute-class description, which is a key ingredient of LAGO.
        if self.use_xian_normed_class_description:
            df_class_descriptions_by_attributes.iloc[:, :] = np.array(self._att['att']).T

        # Sorting class description and attributes by attribute names,
        # in order to cluster them by semantic group names.
        # (because a group name is the prefix for each attribute name)
        if common_args.sort_attr_by_names:
            df_class_descriptions_by_attributes = \
                df_class_descriptions_by_attributes.sort_index(axis=1)
        self._attributes_name = df_class_descriptions_by_attributes.columns

        self._df_class_descriptions_by_attributes = \
            df_class_descriptions_by_attributes

class AWA1_Xian(XianDataset):
    def dataset_name(self):
        return 'AWA1'

class AWA2_Xian(XianDataset):
    def dataset_name(self):
        return 'AWA2'

class CUB_Xian(XianDataset):
    def dataset_name(self):
        return 'CUB'

class FLO_Xian(XianDataset):
    def dataset_name(self):
        return 'FLO'

    def _load_attribute_names(self):
        """ FLO isn't based on attributes. Therefore, this method is empty. """
        pass

    def _get_official_class_descriptions_by_attributes(self):
        """ FLO isn't based on attributes. Therefore, we take an embedding based
        description (from Xian 2018)
         """
        df_class_descriptions_by_attributes = pd.DataFrame(np.array(self._att['att']).T)
        emb_dim = df_class_descriptions_by_attributes.shape[1]
        df_class_descriptions_by_attributes.columns = [f'g{i}::a{i}' for i in range(emb_dim)]
        df_class_descriptions_by_attributes.index += 1
        self._df_class_descriptions_by_attributes = df_class_descriptions_by_attributes



class SUN_Xian(XianDataset):
    def dataset_name(self):
        return 'SUN'
    def _prepare_classes(self):
        return self._att['allclasses_names']

    def _load_attribute_names(self):
        # Load attribute names. Format is: 'group_name:attribute_name'
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir,
                             'attribute_names_with_semantic_group.txt')
        df_attributes_list = \
            pd.read_csv(fname, delim_whitespace=True,
                        names=['attribute_name'])

        df_attributes_list.index += 1
        df_attributes_list.index.name = 'attribute_id'
        return df_attributes_list



def get_xian2017_data(data_dir):
    """
    load data as in Xian 2017 (images as ResNet101 vectors, different labels indexing of classes and unique dataset splits)
    :return:
        data, att (dictionaries)
    """

    """ From Xian2017 README:
        resNet101.mat includes the following fields:
        -features: columns correspond to image instances
        -labels: label number of a class is its row number in allclasses.txt
        -image_files: image sources


        att_splits.mat includes the following fields:
        -att: columns correpond to class attributes vectors, following the classes order in allclasses.txt
        -trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
        -test_seen_loc: instances indexes of test set features for seen classes
        -test_unseen_loc: instances indexes of test set features for unseen classes
        -train_loc: instances indexes of train set features 
        -val_loc: instances indexes of val set features 
    """
    att_file = 'att_splits.mat'
    feat_file = 'res101.mat'

    att_mat = scipy.io.loadmat(os.path.join(data_dir, att_file))
    data_mat = scipy.io.loadmat(os.path.join(data_dir, feat_file))

    data = _data_mat_to_py(data_mat)

    att = _att_mat_to_py(att_mat)

    # Add a mapping from classname to (xian) class_id
    if 'allclasses_names' in att:
        id_to_classname = {(k + 1): v for k, v in
                           enumerate(att['allclasses_names'])}
        classname_to_id =  {v: (k + 1) for k, v in enumerate(att['allclasses_names'])}
    else:
        offset = 0
        if np.unique(data['labels']).min() == 0:
            offset = 1
        id_to_classname = {(id+offset):str(id) for id in np.unique(data['labels'])}
        classname_to_id = {v: k for k, v in id_to_classname.items()}

    if 'CUB' in data_dir:
        assert (
            (np.array([id_to_classname[k].lower() for k in data['labels']]) == np.array(
                [k.split('/')[0].lower() for k in data['image_files']])).all())
    if 'SUN' in data_dir:
        assert (
            (np.array([id_to_classname[k].lower() for k in data['labels']]) == np.array(
                ['_'.join(k.split('/')[1:-1]).lower() for k in
                 data['image_files']])).all())
    att['id_to_classname'] = id_to_classname
    att['classname_to_id'] = classname_to_id

    """ Properties of CUB att (for reference): 
        print [len(att[k]) for k in 
                ['train_loc', 'val_loc', 'trainval_loc', 'test_seen_loc', 'test_unseen_loc', 'att', ]]
        >>> [5875, 2946, 7057, 1764, 2967, 312]
        print 5875 + 2946
        >>> 8821
        print 7057 + 1764
        >>> 8821
        print 7057 + 1764 + 2967
        >>> 11788
        print len(set(att['train_loc']).intersection(att['trainval_loc']))
        >>> 4702
        print len(set(att['val_loc']).intersection(att['trainval_loc']))
        >>> 2355
        print 4702 + 2355
        >>> 7057      
        print len(set(att['test_seen_loc']).intersection(att['train_loc'] + att['val_loc']))
        >>> 1764
    """

    return data, att


def _att_mat_to_py(att_mat):
    att_py = {}
    if 'allclasses_names' in att_mat:
        att_py['allclasses_names'] = [val[0][0] for val in
                                      att_mat['allclasses_names']]
    att_py['train_loc'] = att_mat['train_loc'].astype(int).flatten().tolist()
    att_py['trainval_loc'] = att_mat['trainval_loc'].astype(
        int).flatten().tolist()
    att_py['val_loc'] = att_mat['val_loc'].astype(int).flatten().tolist()
    att_py['test_seen_loc'] = att_mat['test_seen_loc'].astype(
        int).flatten().tolist()
    att_py['test_unseen_loc'] = att_mat['test_unseen_loc'].astype(
        int).flatten().tolist()
    att_py['att'] = att_mat['att']

    return att_py


def _data_mat_to_py(data_mat):
    data_py = {}
    data_py['features'] = data_mat['features']
    if 'image_files' in data_mat:
        first_file_name = data_mat['image_files'][0][0][0]
        if '/Flowers/' in first_file_name:
            # Handle filenames for FLO dataset
            data_py['image_files'] = [
                fname[0][0].lower().split('/jpg/')[1].split('.jpg')[0] for fname in
                data_mat['image_files']]
        elif 'JPEGImages' in first_file_name:
            # this relates to AWA1&2 @ Xian
            data_py['image_files'] = [
                fname[0][0].lower().split('images/')[1].split('.jpg')[0] for fname in
                data_mat['image_files']]
        else:
            data_py['image_files'] = [
                fname[0][0].split('images/')[1].split('.jpg')[0] for fname in
                data_mat['image_files']]
    data_py['labels'] = data_mat['labels'].astype(int).flatten().tolist()
    return data_py

