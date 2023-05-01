#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/1 17:51
# @Author  : ZWP
# @Desc    : 
# @File    : utils.py
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import Sequential, Input, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Attention, Concatenate, Input, Lambda
from keras.optimizers.legacy_learning_rate_decay import exponential_decay
from sklearn.preprocessing import MinMaxScaler



def data_process(dataset: pd.DataFrame):
    # 增加一些拓展属性
    # 增加是否关井的判断
    def isClose(row):
        if row["DailyHour"] == 0:
            return 1
        else:
            return 0

    def isAuto(row):
        if row["DailyHour"] == 24:
            return 1
        else:
            return 0

    # 独热编码
    dataset = pd.get_dummies(dataset, columns=["Station"])
    dataset = dataset.assign(isClose=dataset.apply(
        isClose, axis=1), isAuto=dataset.apply(isAuto, axis=1))
    return dataset


def split_dataset(dataset: pd.DataFrame, test_size=0.2):
    print(type(dataset))
    size = dataset.shape[0]
    return dataset[:int(size * (1 - test_size))][:], dataset[int(size * (1 - test_size)):][:]


def build_dataset(dataset: pd.DataFrame, input_size=30, output_size=7,
                  test_size=0.2, stride=1):
    train_set = []
    test_set = []
    for i, well in enumerate(dataset["WellNo"].unique()):
        df_well = dataset[dataset["WellNo"] == well]
        df_well.sort_values("Date", inplace=True)
        train_set_well, test_set_well = split_dataset(
            df_well, test_size=test_size)
        train_set.append(train_set_well)
        test_set.append(test_set_well)
    train_set = pd.concat(train_set, axis=0)
    test_set = pd.concat(test_set, axis=0)
    return train_set, test_set


class preprocessing():
    def __init__(self, input_feature, input_size=30, output_size=7):
        self.minMax = MinMaxScaler()
        self.feature_list = ["DailyHour", "Allocation", "WellHeadPressure", "CasingHeadPressure", "WellHeadTemperature",
                             "ElapsedProduction", "DailyProduction"]
        self.input_feature = input_feature
        self.input_size = 30
        self.output_size = 7

    def fit(self, dataset):
        self.minMax.fit(dataset[self.feature_list].values)

    def fit_transform(self, dataset):
        self.minMax.fit(dataset[self.feature_list].values)
        return self.transform(dataset=dataset)

    def transform(self, dataset):
        input_total = {"input": [], "hour": []}
        output_total = []

        for i, well in enumerate(dataset["WellNo"].unique()):
            df_well = dataset[dataset["WellNo"] == well]
            df_well_dp = df_well["DailyProduction"].copy(deep=True)
            if i >= 1:
                break
            df_well.loc[:, self.feature_list] = self.minMax.transform(
                df_well[self.feature_list].values)
            df_well.loc[:, 'ElapsedProduction'] = df_well.loc[:, 'ElapsedProduction'].round(
                2)
            if (df_well.shape[0] > self.input_size + self.output_size):
                input, output = self.data_modeling(
                    df_well, df_well_dp)
                input_total["input"].append(input[0])
                input_total["hour"].append(input[1])
                output_total.append(output)
        input_total["input"] = np.concatenate(input_total["input"], axis=0)
        input_total["hour"] = np.concatenate(input_total["hour"], axis=0)
        output_total = np.concatenate(output_total, axis=0)

        return input_total, output_total

    def data_modeling(self, X, y, input_size=30, output_size=7, stride=1):

        if X.shape[0] > input_size + output_size:
            target_multi_step = []
            target = np.expand_dims(
                y.to_numpy(), 1)
            for i in range(output_size):
                target_multi_step.append(np.roll(target, -i, 0)[:-output_size])
            target_multi_step = np.concatenate(target_multi_step, axis=1)

            # 进行数据建模
            input = [[], []]
            output = []
            for i in range(0, len(X) - input_size - output_size, stride):
                input[0].append(X[i:i + input_size]
                                [self.input_feature].to_numpy())
                input[1].append(X[i + input_size:i + input_size +
                                                 output_size]["DailyHour"].to_numpy().reshape(-1, 1))
                output.append(
                    target_multi_step[i + input_size:i + input_size + 1])
            input[0] = np.array(input[0])
            input[1] = np.array(input[1])
            output = np.array(output)
            return input, output