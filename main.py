import numpy as np
import pandas as pd

from MQRNN import MQRNN
from utils import data_process, build_dataset, preprocessing


def train():
    df = pd.read_csv("GasProductionOri2.csv")
    # 数据预处理，特征构建，分类数据转换为独热编码
    df = data_process(df)
    train_set, test_set = build_dataset(df)

    input_feature = ['DailyHour',
                     'Allocation', 'WellHeadPressure', 'CasingHeadPressure',
                     'WellHeadTemperature', 'DailyProduction', 'ElapsedProduction',
                     'isClose', 'isAuto', 'Station_C1', 'Station_C2', 'Station_C3']
    dp = preprocessing(input_feature=input_feature)

    x_train, y_train = dp.fit_transform(train_set)
    x_test, y_test = dp.transform(test_set)

    bad_row = []
    for i, data in enumerate(y_train):
        if 0.0 in data:
            bad_row.append(i)
    x_train["input"] = np.delete(x_train["input"], bad_row, axis=0)
    x_train["hour"] = np.delete(x_train["hour"], bad_row, axis=0)
    y_train = np.delete(y_train, bad_row, axis=0)
    x_train_shuffle, x_train_hour_shuffle, y_train_shuffle = shuffle(
        x_train["input"], x_train["hour"], y_train)
    bad_row = []
    for i, data in enumerate(y_test):
        if 0.0 in data:
            bad_row.append(i)

    x_test["input"] = np.delete(x_test["input"], bad_row, axis=0)
    x_test["hour"] = np.delete(x_test["hour"], bad_row, axis=0)
    y_test = np.delete(y_test, bad_row, axis=0)
    x_test_shuffle, x_test_hour_shuffle = x_test["input"], x_test["hour"]

    np.savez("data.npz", x_train_shuffle=x_train_shuffle, x_train_hour_shuffle=x_train_hour_shuffle,
             y_train_shuffle=y_train_shuffle,
             x_test_shuffle=x_test_shuffle, x_test_hour_shuffle=x_test_hour_shuffle, y_test=y_test)
    data = np.load('data.npz')
    x_train_shuffle, x_train_hour_shuffle, y_train_shuffle, x_test_shuffle, x_test_hour_shuffle, y_test = data[
                                                                                                              "x_train_shuffle"], \
                                                                                                          data[
                                                                                                              "x_train_hour_shuffle"], \
                                                                                                          data[
                                                                                                              "y_train_shuffle"], \
                                                                                                          data[
                                                                                                              "x_test_shuffle"], \
                                                                                                          data[
                                                                                                              "x_test_hour_shuffle"], \
                                                                                                          data[
                                                                                                              "y_test"]

    covariate_size = x_train_shuffle.shape[-1] - 1
    model = MQRNN(horizon_size,
                  hidden_size,
                  quantiles,
                  columns,
                  dropout,
                  layer_size,
                  n_step,
                  lr,
                  batch_size,
                  num_epochs,
                  context_size,
                  covariate_size,
                  feature_covariate_size, "")
    model.train(x_train_shuffle, x_train_hour_shuffle,
                y_train_shuffle.reshape(y_train_shuffle.shape[0], horizon_size, 1),
                x_test_shuffle,
                x_test_hour_shuffle, y_test.reshape(y_test.shape[0], horizon_size, 1))


def test():
    data = np.load('data.npz')
    x_train_shuffle, x_train_hour_shuffle, y_train_shuffle, x_test_shuffle, x_test_hour_shuffle, y_test = data[
                                                                                                              "x_train_shuffle"], \
                                                                                                          data[
                                                                                                              "x_train_hour_shuffle"], \
                                                                                                          data[
                                                                                                              "y_train_shuffle"], \
                                                                                                          data[
                                                                                                              "x_test_shuffle"], \
                                                                                                          data[
                                                                                                              "x_test_hour_shuffle"], \
                                                                                                          data[
                                                                                                              "y_test"]
    covariate_size = x_train_shuffle.shape[-1] - 1
    model = MQRNN(horizon_size,
                  hidden_size,
                  quantiles,
                  columns,
                  dropout,
                  layer_size,
                  n_step,
                  lr,
                  batch_size,
                  num_epochs,
                  context_size,
                  covariate_size,
                  feature_covariate_size, "")
    model.test(x_test_shuffle,
               x_test_hour_shuffle, y_test.reshape(y_test.shape[0], horizon_size, 1))


# if __name__ == '__main__':
#     # 定义输出形状
#     batch_size = 32
#     time_steps = 5
#     num_quantiles = 5
#
#     # 生成随机的预测结果
#     for i in range(1000):
#         y_pred = tf.random.normal(shape=(batch_size, time_steps, num_quantiles))
#         y_true = tf.random.normal(shape=(batch_size, time_steps, 1))
#         loss = QuantileLoss([0.1, 0.2, 0.3, 0.4, 0.5])
#         loss(y_pred=y_pred, y_true=y_true)

if __name__ == '__main__':
    horizon_size = 7
    hidden_size = 100
    columns = []
    quantiles = [0.1, 0.5, 0.9]
    dropout = 0.5
    layer_size = 1
    n_step = 30
    lr = 0.001
    batch_size = 64
    num_epochs = 3
    context_size = 15
    feature_covariate_size = 1
    test()
