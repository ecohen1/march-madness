from __future__ import division, print_function, absolute_import

import tflearn

# Data loading and preprocessing
# X, Y, testX, testY = mnist.load_data(one_hot=True)
import tensorflow as tf
import numpy as np
import pandas as pd

orig_data = pd.read_csv('data/processed/shuffled.csv')

del orig_data['team_1']
del orig_data['team_2']
del orig_data['seed_1']
del orig_data['seed_2']

data = orig_data.sample(frac=1).reset_index(drop=True)

allx = data.copy()
del allx['result']
allx = allx.values.tolist()

ally = data['result'].values.tolist()
ally = [ [1,0] if y == 'win' else [0,1] for y in ally ]

index1 = int(len(allx)*.70)
index2 = int(len(allx)*.90)

trainx = allx[:index1]
trainy = ally[:index1]
validx = allx[index1:index2]
validy = ally[index1:index2]
testx = allx[index2:]
testy = ally[index2:]
# Building deep neural network
input_layer = tflearn.input_data(shape=[None, len(trainx[0])])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, .9)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, .9)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainx, trainy, n_epoch=20, validation_set=(validx, validy),
          show_metric=True, run_id="dense_model")

probabilities = model.predict(testx)
num_correct = 0.0
num_class_1 = 0
for i,p in enumerate(probabilities):
	if p.index(max(p)) == testy[i].index(max(testy[i])):
		num_correct += 1
	if testy[i].index(max(testy[i])) == 0:
		num_class_1 += 1
print(num_correct/len(testx),'-',abs(0.5-num_class_1/len(testy)))
