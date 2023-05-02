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
