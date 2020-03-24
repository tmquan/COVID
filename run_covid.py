# coding=utf-8
# Author: Tran Minh Quan
import cv2
import random
import numpy as np
import pandas as pd
from datetime import datetime

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.predict import FeedfreePredictor, PredictConfig
from tensorpack.utils import logger, fix_rng_seed
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.stats import BinaryStatistics
import albumentations as AB
import argparse
import sklearn.metrics 
import sys
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
tf = tf.compat.v1
# tf.disable_v2_behavior()
# from tensorlayer.cost import dice_coe
from covid import KFoldCovidDataset
from models.inceptionbn import InceptionBN
from models.shufflenet import ShuffleNet
from models.densenet import DenseNet121, DenseNet169, DenseNet201
from models.resnet import ResNet101
from models.vgg16 import VGG16


def visualize_tensors(name, imgs, scale_func=lambda x: (x + 1.) * 128., max_outputs=1):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualization operation
        *imgs: multiple tensors as list
        scale_func: scale input tensors to fit range [0, 255]
    Example:
        visualize_tensors('viz1', [img1])
        visualize_tensors('viz2', [img1, img2, img3], max_outputs=max(30, BATCH))
    """
    xy = scale_func(tf.concat(imgs, axis=2))
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)

class CustomBinaryStatistics(object):
    """
    Statistics for binary decision,
    including precision, recall, false positive, false negative
    """

    def __init__(self, threshold=0.5, types=6):
        self.reset()
        self.threshold = threshold
        self.types = types

    def reset(self):
        self._f1_score = -1 
        self._f2_score = -1 

        self._precision = -1 
        self._recall = -1 

        self.total_label = []
        self.total_estim = []

    def feed(self, estim, label):
        """
        Args:
            estim (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert estim.shape == label.shape, "{} != {}".format(estim.shape, label.shape)
    
        self.total_estim.append(estim >= self.threshold)
        self.total_label.append(label)
    @property
    def estim(self):
        return np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
   
    @property
    def label(self):
        return np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
   
    @property
    def precision(self):
        np_label = np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
        np_estim = np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
        # print(np_label.shape, np_estim.shape, np_label.dtype, np_estim.dtype)
        return sklearn.metrics.precision_score(np_label, np_estim, average='binary')

    @property
    def recall(self):
        np_label = np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
        np_estim = np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
        return sklearn.metrics.recall_score(np_label, np_estim, average='binary')

    @property
    def roc_auc(self):
        np_label = np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
        np_estim = np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
        return sklearn.metrics.roc_auc_score(np_label, np_estim, average='binary')

    @property
    def f1_score(self):
        np_label = np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
        np_estim = np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
        return sklearn.metrics.f1_score(np_label, np_estim, average='binary')

    @property
    def f2_score(self):
        np_label = np.array(self.total_label).astype(np.float32).reshape(-1, self.types)
        np_estim = np.array(self.total_estim).astype(np.float32).reshape(-1, self.types)
        return sklearn.metrics.fbeta_score(np_label, np_estim, beta=2, average='binary')

class CustomBinaryClassificationStats(Inferencer):
    """
    Compute precision / recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_tensor_name, label_tensor_name, args=None, prefix='valid'):
        """
        Args:
            pred_tensor_name(str): name of the 0/1 prediction tensor.
            label_tensor_name(str): name of the 0/1 label tensor.
        """
        self.pred_tensor_name = pred_tensor_name
        self.label_tensor_name = label_tensor_name
        self.prefix = prefix
        self.args = args

    def _before_inference(self):
        self.stat = CustomBinaryStatistics(threshold=args.threshold, types=args.types)

    def _get_fetches(self):
        return [self.pred_tensor_name, self.label_tensor_name]

    def _on_fetches(self, outputs):
        estim, label = outputs
        self.stat.feed(estim, label)

    def _after_inference(self):
        np_estim = self.stat.estim
        np_label = self.stat.label
        result = {}
       
        if args.types==1:
            f1_score = sklearn.metrics.fbeta_score(np_label, np_estim, beta=1, average='binary')
            f2_score = sklearn.metrics.fbeta_score(np_label, np_estim, beta=2, average='binary')
            precision_score = sklearn.metrics.precision_score(np_label, np_estim, average='binary')
            recall_score = sklearn.metrics.recall_score(np_label, np_estim, average='binary')

            result[f'{self.prefix}_f1_score_{args.pathology}'] = f1_score
            result[f'{self.prefix}_f2_score_{args.pathology}'] = f2_score
            result[f'{self.prefix}_precision_score_{args.pathology}'] = precision_score
            result[f'{self.prefix}_recall_score_{args.pathology}'] = recall_score
        elif args.types==6:
            pathologies = ["Airspace_Opacity", "Cardiomegaly", "Fracture", 
                       "Lung_Lesion", "Pleural_Effusion", "Pneumothorax"]
            for pathology in range(self.args.types):
                f1_score = sklearn.metrics.fbeta_score(np_label[:,pathology], np_estim[:,pathology], beta=1, average='binary')
                f2_score = sklearn.metrics.fbeta_score(np_label[:,pathology], np_estim[:,pathology], beta=2, average='binary')
                precision_score = sklearn.metrics.precision_score(np_label[:,pathology], np_estim[:,pathology], average='binary')
                recall_score = sklearn.metrics.recall_score(np_label[:,pathology], np_estim[:,pathology], average='binary')
            result[f'{self.prefix}_f1_score_{pathologies[pathology]}'] = f1_score
            result[f'{self.prefix}_f2_score_{pathologies[pathology]}'] = f2_score
            result[f'{self.prefix}_precision_score_{pathologies[pathology]}'] = precision_score
            result[f'{self.prefix}_recall_score_{pathologies[pathology]}'] = recall_score

        return result

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos + 1e-6)

        pos_weight = beta / (1 - beta + 1e-6)
        cost = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)

class Model(ModelDesc):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

    def inputs(self):
        return [tf.TensorSpec([None, self.args.shape, self.args.shape, 1], tf.float32, 'image'),
                tf.TensorSpec([None, self.args.types], tf.float32, 'label')
                ]

    def build_graph(self, image, label):
        image = image / 128.0 - 1.0

        if self.args.name == 'VGG16':
            logit, recon = VGG16(image, classes=self.args.types)
        elif self.args.name == 'ShuffleNet':
            logit = ShuffleNet(image, classes=self.args.types)
        elif self.args.name == 'ResNet101':
            logit, recon = ResNet101(image, mode=self.args.mode, classes=self.args.types)
        elif self.args.name == 'DenseNet121':
            logit, recon = DenseNet121(image, classes=self.args.types) 
        elif self.args.name == 'DenseNet169':
            logit, recon = DenseNet169(image, classes=self.args.types)
        elif self.args.name == 'DenseNet201':
            logit, recon = DenseNet201(image, classes=self.args.types)
        elif self.args.name == 'InceptionBN':
            logit = InceptionBN(image, classes=self.args.types)
        else:
            pass

        estim = tf.sigmoid(logit, name='estim')
        loss_xent = class_balanced_sigmoid_cross_entropy(logit, label, name='loss_xent')
        # loss_dice = tf.identity(1.0 - dice_coe(estim, label, axis=[0,1], loss_type='jaccard'), 
        #                          name='loss_dice') 
        # # Reconstruction
        # with argscope([Conv2D, Conv2DTranspose], use_bias=False,
        #               kernel_initializer=tf.random_normal_initializer(stddev=0.02)), \
        #         argscope([Conv2D, Conv2DTranspose, InstanceNorm], data_format='channels_first'):
        #     recon = (LinearWrap(recon)
        #              .Conv2DTranspose('deconv0', 64 * 8, 3, strides=2)
        #              .Conv2DTranspose('deconv1', 64 * 8, 3, strides=2)
        #              .Conv2DTranspose('deconv2', 64 * 4, 3, strides=2)
        #              .Conv2DTranspose('deconv3', 64 * 2, 3, strides=2)
        #              .Conv2DTranspose('deconv4', 64 * 1, 3, strides=2)
        #              .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]], mode='SYMMETRIC')
        #              .Conv2D('recon', 1, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        #     recon = tf.transpose(recon, [0, 2, 3, 1])
        # loss_mae = tf.reduce_mean(tf.abs(recon-image), name='loss_mae')
        # Visualization
        visualize_tensors('image', [image], scale_func=lambda x: x * 128.0 + 128.0, 
                          max_outputs=max(64, self.args.batch))
        # Regularize the weight of model 
        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        cost = tf.add_n([loss_xent, wd_cost], name='cost')
        add_moving_summary(loss_xent)
        add_moving_summary(wd_cost)
        add_moving_summary(cost)
        return cost

    def optimizer(self):
        lrate = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        add_moving_summary(lrate)
        optim = tf.train.AdamOptimizer(lrate, beta1=0.5, epsilon=1e-3)
        return optim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--name', help='Model name', default='DenseNet121')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--pred', action='store_true', help='run prediction')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data', default='/data//data/COVID_Data/', help='Data directory')
    parser.add_argument('--save', default='train_log/', help='Saving directory')
    parser.add_argument('--mode', default='none', help='Additional mode of resnet')
    
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--types', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pathology', default='All')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--shape', type=int, default=256)

    args = parser.parse_args()

    if args.seed:
        os.environ['PYTHONHASHSEED']=str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        fix_rng_seed(args.seed)
        tf.random.set_random_seed(args.seed)

    if args.gpus:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # model = Model(args=args)

    if args.eval:
        # To be implemented
        sys.exit(0)

    elif args.pred:
        # To be implemented
        sys.exit(0)

    else:
        for valid_fold_index in range(args.folds):
            # with tf.Graph().as_default():
            tf.reset_default_graph()
            logger.set_logger_dir(os.path.join(
                args.save, args.name, args.pathology, 
                os.path.basename(args.load) if args.load else "Scratch", 
                str(valid_fold_index), 
                str(args.shape), str(args.types), ), 'd')

            model = Model(args=args)
            # Setup the dataset for training
            ds_train = KFoldCovidDataset(folder=args.data,
                                         is_train='train',
                                         fname='train_ncov.csv',
                                         types=args.types,
                                         pathology=args.pathology,
                                         resize=int(args.shape), 
                                         fold_idx=valid_fold_index,
                                         n_folds=args.folds)
           
            ag_train = [
                imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
                imgaug.Affine(shear=10),
                imgaug.RandomChooseAug([
                    imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),  
                    imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),  
                    imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),  
                ]),
                imgaug.RandomOrderAug(
                    [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                     imgaug.Contrast((0.6, 1.4), clip=False),
                     imgaug.Saturation(0.4, rgb=False),
                     # rgb-bgr conversion for the constants copied from
                     # fb.resnet.torch
                     imgaug.Lighting(0.1,
                                     eigval=np.asarray(
                                         [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                     eigvec=np.array(
                                         [[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]],
                                         dtype='float32')[::-1, ::-1]
                )]),
                imgaug.Albumentations(AB.CLAHE(p=0.5)),
                imgaug.RotationAndCropValid(max_deg=25),
                imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0), 
                        aspect_ratio_range=(0.8, 1.2),
                        interp=cv2.INTER_AREA, 
                        target_shape=args.shape),
                imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
                imgaug.ToFloat32(),
            ]
            ag_label = [ # Label smoothing
                imgaug.BrightnessScale((0.8, 1.2), clip=False),
            ]
            ds_train.reset_state()
            # ds_train = FixedSizeData(ds_train, 128)
            ds_train = AugmentImageComponent(ds_train, ag_train, 0)
            # ds_train = AugmentImageComponent(ds_train, ag_label, 1)
            ds_train = BatchData(ds_train, args.batch)
            ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=2)
            ds_train = PrintData(ds_train)

            # Setup the dataset for validating
            ds_valid = KFoldCovidDataset(folder=args.data,
                                         is_train='valid',
                                         fname='train_ncov.csv',
                                         types=args.types,
                                         pathology=args.pathology,
                                         resize=int(args.shape), 
                                         fold_idx=valid_fold_index,
                                         n_folds=args.folds)

            ag_valid = [
                imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
                imgaug.Albumentations(AB.CLAHE(p=1)),
                imgaug.Albumentations(AB.SmallestMaxSize(args.shape, p=1.0)),  
                imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
                imgaug.ToFloat32(),
            ]
            ds_valid.reset_state()
            # ds_valid = FixedSizeData(ds_valid, 128)
            ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
            ds_valid = BatchData(ds_valid, 1)
            # ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=1)
            ds_valid = PrintData(ds_valid)


            # Setup the dataset for validating
            ds_test2 = KFoldCovidDataset(folder=args.data,
                                         is_train='valid',
                                         fname='test_ncov.csv',
                                         types=args.types,
                                         pathology=args.pathology,
                                         resize=int(args.shape), 
                                         fold_idx=None,
                                         n_folds=1)

            ag_test2 = [
                imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
                imgaug.Albumentations(AB.CLAHE(p=1)),
                imgaug.Albumentations(AB.SmallestMaxSize(args.shape, p=1.0)),  
                imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
                imgaug.ToFloat32(),
            ]
            ds_test2.reset_state()
            # ds_test2 = FixedSizeData(ds_test2, 128)
            ds_test2 = AugmentImageComponent(ds_test2, ag_test2, 0)
            ds_test2 = BatchData(ds_test2, 1)
            # ds_test2 = MultiProcessRunnerZMQ(ds_test2, num_proc=1)
            ds_test2 = PrintData(ds_test2)


            # Setup the config
            config = TrainConfig(
                model=model,
                dataflow=ds_train,
                callbacks=[
                    ModelSaver(),
                    MinSaver('cost'),
                    ScheduledHyperParamSetter('learning_rate',
                                              [(0, 1e-2), (50, 1e-3), (100, 1e-4), (150, 1e-5), (200, 1e-6)]),
                    InferenceRunner(ds_valid, [CustomBinaryClassificationStats('estim', 'label', args, prefix='valid'),
                                               ScalarStats(['loss_xent', 'cost'], prefix='valid'),
                                               ], tower_name='ValidTower'),
                    InferenceRunner(ds_test2, [CustomBinaryClassificationStats('estim', 'label', args, prefix='test2'),
                                               ScalarStats(['loss_xent', 'cost'], prefix='test2'),
                                               ], tower_name='Test2Tower'),
                ],
                max_epoch=250,
                session_init=SmartInit(args.load),
            )


            trainer = SyncMultiGPUTrainerParameterServer(max(get_num_gpu(), 1))

            launch_train_with_config(config, trainer)
