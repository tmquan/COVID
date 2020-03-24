
import os

import cv2

import numpy as np

import pandas as pd

import tensorpack.dataflow as df
from tensorpack.utils import get_rng
from tensorpack.utils.argtools import shape2d


class Vinmec(df.RNGDataFlow):
    # https://github.com/tensorpack/tensorpack/blob/master/tensorpack/dataflow/image.py
    """ Produce images read from a list of files as (h, w, c) arrays. """

    def __init__(self, folder, types=14, is_train='train', channel=1,
                 resize=None, debug=False, shuffle=False, pathology=None, fname='train.csv'):
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
        self.description = "Vinmec is a large dataset of chest X-rays\n",
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

    def reset_state(self):
        self.rng = get_rng(self)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.is_train == 'train':
            self.rng.shuffle(indices)

        for idx in indices:
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
                if self.types == 5:
                    label.append(self.df.iloc[idx]['Atelectasis'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Consolidation'])
                    label.append(self.df.iloc[idx]['Edema'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    # label.append(self.df.iloc[idx]['Pneumonia/infection'])
                if self.types == 6:
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Fracture'])
                    label.append(self.df.iloc[idx]['Lung_Lesion'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    label.append(self.df.iloc[idx]['Pneumothorax'])
                if self.types == 16:
                    label.append(self.df.iloc[idx]['Atelectasis'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Consolidation'])
                    label.append(self.df.iloc[idx]['Edema'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    label.append(self.df.iloc[idx]['Pneumothorax'])
                    label.append(self.df.iloc[idx]['Pleural_Other'])
                    label.append(self.df.iloc[idx]['Lung_Lesion'])
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Pneumonia/infection'])
                    label.append(self.df.iloc[idx]['Cavitation'])
                    label.append(self.df.iloc[idx]['Fibrosis'])
                    label.append(self.df.iloc[idx]['Widening_Mediastinum'])
                    label.append(self.df.iloc[idx]['Medical_device'])
                    label.append(self.df.iloc[idx]['Fracture'])
                    label.append(self.df.iloc[idx]['No_Finding'])
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
    ds = Vinmec(folder='/u01/data/Vimmec_Data_small/',
                train_or_valid='train',
                resize=256)
    ds.reset_state()
    # ds = df.MultiProcessRunnerZMQ(ds, num_proc=8)
    ds = df.BatchData(ds, 32)
    # ds = df.PrintData(ds)
    df.TestDataSpeed(ds).start()