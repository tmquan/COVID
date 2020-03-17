import os

import cv2

import numpy as np

import pandas as pd

import tensorpack.dataflow as df
from tensorpack.utils import get_rng
from tensorpack.utils.argtools import shape2d
from sklearn.model_selection import KFold

class KFoldCovidDataset(df.RNGDataFlow):
    # https://github.com/tensorpack/tensorpack/blob/master/tensorpack/dataflow/image.py
    """ Produce images read from a list of files as (h, w, c) arrays. """

    def __init__(self, folder, types=3, is_train='train', channel=1,
                 resize=None, debug=False, shuffle=False, pathology=None, fname='train.csv', 
                 fold_idx=0, n_folds=5):
        """[summary]
        [description
        Arguments:
            folder {[type]} -- [description]
        Keyword Arguments:
            types {number} -- [description] (default: {14})
            is_train {str} -- [description] (default: {'train'}, {'valid'} or {'test'})
            channel {number} -- [description] (default: {1})
            resize {[type]} -- [description] (default: {None})
            debug {bool} -- [description] (default: {False})
            shuffle {bool} -- [description] (default: {False})
            fname {str} -- [description] (default: {"train.csv"})
        """
        self.version = "1.0.0"
        self.description = "KFoldCovidDataset is a large dataset of chest X-rays\n",
        self.citation = "\n"
        self.folder = folder
        self.types = types
        self.is_train = is_train
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if self.channel == 1:
            self.imread_mode = cv2.IMREAD_GRAYSCALE
        else:
            self.imread_mode = cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.debug = debug
        self.shuffle = shuffle
        self.csvfile = os.path.join(self.folder, fname)
        print(self.folder)
        # Read the csv
        self.df = pd.read_csv(self.csvfile)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        print(self.df.info())
        self.pathology = pathology
        self.indices = list(range(len(self.df)))


        if n_folds>1:
            kfs = KFold(n_splits=n_folds)
            index = 0
            for train_indices, valid_indices in kfs.split(self.indices):
                if index != fold_idx:
                    index += 1
                    continue
                self.indices = train_indices if self.is_train == 'train' else valid_indices
                break
        # else:
        #     self.indices = range(len(self.df))
        else: 
            self.indices = list(range(self.__len__()))

    def reset_state(self):
        self.rng = get_rng(self)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        # indices = list(range(self.__len__()))
        # if self.is_train == 'train':
        #     self.rng.shuffle(indices)
        self.rng.shuffle(self.indices)

        for idx in self.indices:
            fpath = os.path.join(self.folder, 'data') #(os.path.dirname(self.folder), 'data')
            fname = os.path.join(fpath, self.df.iloc[idx]['Images'])
            image = cv2.imread(fname, self.imread_mode)
            assert image is not None, fname
            # print('File {}, shape {}'.format(fname, image.shape))
            if self.channel == 3:
                image = image[:, :, ::-1]
            if self.resize is not None:
                image = cv2.resize(image, tuple(self.resize[::-1]))
            if self.channel == 1:
                image = image[:, :, np.newaxis]

            # Process the label
            if self.is_train == 'train' or self.is_train == 'valid':
                label = []
                if self.types == 3:
                    label.append(self.df.iloc[idx]['Covid'])
                    label.append(self.df.iloc[idx]['Pneumonia'])
                    label.append(self.df.iloc[idx]['No_Disease'])
                elif self.types == 1:
                    assert self.pathology is not None
                    label.append(self.df.iloc[idx][self.pathology])
                else:
                    pass
                # Try catch exception
                label = np.nan_to_num(label, copy=True, nan=0)
                label = np.array(label, dtype=np.float32)
                types = label.copy()
                yield [image, types]
            elif self.is_train == 'test':
                yield [image]  # , np.array([-1, -1, -1, -1, -1])
            else:
                pass


if __name__ == '__main__':
    ds = KFoldCovidDataset(folder='/data//data/COVID_Data/',
                train_or_valid='train',
                resize=256)
    ds.reset_state()
    # ds = df.MultiProcessRunnerZMQ(ds, num_proc=8)
    ds = df.BatchData(ds, 32)
    # ds = df.PrintData(ds)
    df.TestDataSpeed(ds).start()