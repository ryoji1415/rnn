from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.utils import to_categorical

import xlrd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


wb = xlrd.open_workbook('/media/ryoji/ボリューム1/simulation_data/rnn_simulation_step2.xlsx')
sheet = wb.sheet_by_name('Sheet1')

#エクセルから値を抽出して配列に格納
values = [[int(sheet.cell_value(row, 4)), int(sheet.cell_value(row, 6))] for row in range(6, sheet.nrows)] #特徴量
signals = [[int(sheet.cell_value(row, 7))] for row in range(6, sheet.nrows)] #信号データ

# データの標準化
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaler = scaler.fit_transform(values)

# 調整パラメータ
TRAIN_RATE = 70 # %換算
TRAIN_DATA_SIZE = 100000 + 5
time_step = 5
BATCH_SIZE = 100
LSTM_units = 10
EPOCH = 10

# データを訓練用、テスト用に分割
train_x = values_scaler[:TRAIN_DATA_SIZE]
train_y = signals[:TRAIN_DATA_SIZE]
test_x = values_scaler[TRAIN_DATA_SIZE:]
test_y = signals[TRAIN_DATA_SIZE:]

save_test_y = []
for i in range(len(test_y)):
    value = test_y[i]
    save_test_y.append(value[0])

# テストデータをcsvに一時保存
x_sig_path = './test_x_sig.csv'
with open(x_sig_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(test_x)

y_sig_path = './test_y_sig.csv'
with open(y_sig_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(save_test_y)


input_train_x = []
input_train_y = []
for i in range(len(train_x) - time_step):
    input_train_x.append(train_x[i:i+time_step])
    y_array = train_y[i+time_step]
    if y_array[0] == -1:
        input_train_y.append(2)
    else:
        input_train_y.append(y_array[0])

train_x = np.array(input_train_x)

np_train_y = np.array(input_train_y)
train_y = np.reshape(np.array(to_categorical(np_train_y)), [np_train_y.shape[0], -1, 3])


# モデルの作成
model = Sequential()
model.add(LSTM(LSTM_units, return_sequences=True,
      batch_input_shape=(BATCH_SIZE, None, 2)))
model.add(Dense(3))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "RMSprop", metrics = ['accuracy'])
model.summary()

# チェックポイントが保存されるディレクトリ
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    )
history = model.fit(train_x, train_y, epochs=EPOCH, callbacks=[checkpoint_callback], shuffle=True)

model_name = 'step2_BATCH' + str(BATCH_SIZE) + 'epoch' + str(EPOCH) + 'LSTM' + str(LSTM_units) + '.h5'
model.save(model_name)


# Lossをグラフ表示
loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()
# Accuracyをグラフ表示
accuracy = history.history['acc']
plt.plot(np.arange(len(accuracy)), accuracy)
plt.show()
