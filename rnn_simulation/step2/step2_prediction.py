from __future__ import absolute_import, division, print_function, unicode_literals

import os
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import csv
import numpy as np
import matplotlib.pyplot as plt


data_length = 500
time_step = 5

new_model = keras.models.load_model('step2_BATCH100epoch10LSTM10.h5')
new_model.summary()

path_list = ['./test_x_sig.csv', './test_y_sig.csv']

test_x_sig, test_y_sig = [], []
csv_path = path_list[0]
with open(csv_path) as f_r:
    reader = csv.reader(f_r)
    logs = [log for log in reader]
    for index in range(len(logs)):
        log = logs[index]
        int_log = [s for s in log]
        test_x_sig.append(int_log)

csv_path = path_list[1]
with open(csv_path) as f_r:
    reader = csv.reader(f_r)
    logs = [log for log in reader][0]
    test_y_sig = [int(s) for s in logs]

print(len(test_x_sig), len(test_y_sig))

input_x = []
target_y = []
for i in range(data_length):
    input_x.append(test_x_sig[i+100:i+time_step+100])
    target_y.append(test_y_sig[i+time_step+100])


input = np.array(input_x)

output = new_model.predict(input)
print(output)
print(output.shape)

plot_outs = []
for i in range(output.shape[0]):
    timestep_value = output[i]
    output_array = timestep_value[4]
    print(output_array)
    out = np.where(output_array > 0.5, 1, 0)
    print(out)
    if np.argmax(out) == 0:
        plot_out = 0
    elif np.argmax(out) == 2:
        plot_out = -1
    else:
        plot_out = 1
    plot_outs.append(plot_out)

# for index in range(len(plot_outs)):
#     print(plot_target[index], plot_outs[index])

t = np.linspace(0, data_length-1, data_length)
fig, ax = plt.subplots()
y1 = np.array([target_y[i] for i in range(data_length)])
y2 = np.array([plot_outs[i] for i in range(data_length)])

ax.set_xlabel('t')
ax.set_ylim(-1.2, 1.2)
ax.set_yticks([])
ax.grid()


ax.plot(t, y1, color='blue', label='Target')
ax.plot(t, y2, color='red', label='Prediction')
ax.legend()
plt.show()
