#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/2 10:54
# @Author  : ZWP
# @Desc    : 
# @File    : MQRNN.py
import numpy as np
import pandas as pd

from Decoder import GlobalDecoder, LocalDecoder
import tensorflow as tf
from sklearn.utils import shuffle
import tensorflow_probability as tfp

from utils import data_process, build_dataset, preprocessing


class MQRNN():
    def __init__(self,
                 horizon_size: int,  # 预测步长
                 hidden_size: int,  # 隐藏层单元大小
                 quantiles: list,  # 计算多少分位数
                 columns: list,
                 dropout: float,
                 layer_size: int,
                 n_step: int,
                 lr: float,
                 batch_size: int,
                 num_epochs: int,
                 context_size: int,
                 covariate_size: int,
                 feature_covariate_size: int,
                 device):
        # 协变量
        self.covariate_size = covariate_size
        self.feature_covariate_size = feature_covariate_size
        self.horizon_size = horizon_size
        self.quantiles = quantiles
        # 初始化
        self.encoder = tf.keras.layers.LSTM(100,
                                            input_shape=(n_step, covariate_size + 1),
                                            dropout=dropout)
        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                      horizon_size=horizon_size,
                                      context_size=context_size,
                                      covariate_size=covariate_size)

        self.ldecoder = LocalDecoder(covariate_size=feature_covariate_size,
                                     context_size=context_size,
                                     horizon_size=horizon_size,
                                     quantiles=quantiles)
        # 协变量输入
        covariate_input = tf.keras.layers.Input(shape=(n_step, covariate_size + 1))
        enc_output = self.encoder(covariate_input)

        feature_covariate_input = tf.keras.layers.Input(shape=(horizon_size, feature_covariate_size))
        feature_covariate_input_reshape = tf.reshape(feature_covariate_input,
                                                     (-1, tf.shape(feature_covariate_input)[1] *
                                                      tf.shape(feature_covariate_input)[2]))
        gdecoder_input = tf.keras.layers.Concatenate()([enc_output, feature_covariate_input_reshape])
        gdecoder_output = self.gdecoder(gdecoder_input)
        ldecoder_input = tf.keras.layers.Concatenate()([gdecoder_output, feature_covariate_input_reshape])
        ldecoder_output = self.ldecoder(ldecoder_input)
        self.model = tf.keras.Model(inputs=[covariate_input, feature_covariate_input], outputs=ldecoder_output)
        self.model.compile(loss=QuantileLoss(quantiles=quantiles), optimizer="adam")

    def train(self, x_train_shuffle, x_train_hour_shuffle, y_train_shuffle, x_test_shuffle, x_test_hour_shuffle,
              y_test):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.h5',
                                                                 monitor='val_loss',
                                                                 save_best_only=True,
                                                                 save_weights_only=True)
        self.model.fit([x_train_shuffle, x_train_hour_shuffle], y_train_shuffle, epochs=3,
                       validation_data=([x_test_shuffle, x_test_hour_shuffle], y_test), callbacks=[checkpoint_callback])

    def test(self, x_test_shuffle, x_test_hour_shuffle,
             y_test):
        self.model.load_weights("./best_weights.h5")
        y_pred = self.model.predict([x_test_shuffle, x_test_hour_shuffle])
        y_pred = tf.reshape(y_pred, (-1, self.horizon_size, len(self.quantiles)))
        y_pred = y_pred[:, :, (int)(y_pred.shape[2] / 2)]
        y_test = tf.reshape(y_test, (y_pred.shape[0], y_pred.shape[1]))
        mae = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred, y_test))
        print(mae)
        return mae

    def predict(self, x_test_shuffle, x_test_hour_shuffle):
        y_pred = self.model.predict([x_test_shuffle, x_test_hour_shuffle])
        y_pred = tf.reshape(y_pred, (-1, self.horizon_size, len(self.quantiles)))
        y_pred = y_pred[:, :y_pred.shape[2] / 2]
        return y_pred


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # 将真实结果张量转换为分位数张量，形状为(batch_size, num_timesteps, num_quantiles)
        y_true_quantiles = tf.tile(y_true, [1, 1, len(self.quantiles)])
        y_pred = tf.reshape(y_pred, (-1, y_true_quantiles.shape[1], len(self.quantiles)))
        # 计算各分位数损失
        low = tf.multiply((y_true_quantiles - y_pred), tf.convert_to_tensor(self.quantiles)),
        high = tf.multiply((y_pred - y_true_quantiles), tf.convert_to_tensor([1 - _ for _ in self.quantiles]))
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.maximum(low, high, ), axis=-1))
        print(loss)
        return loss