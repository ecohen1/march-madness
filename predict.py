from __future__ import division, print_function, absolute_import

import tflearn

# Data loading and preprocessing
# X, Y, testX, testY = mnist.load_data(one_hot=True)
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint

orig_data = pd.read_csv('data/processed/shuffled.csv')
del orig_data['team_1']
del orig_data['team_2']
del orig_data['history']

test_data = pd.read_csv('data/2017/64_s.csv')
team_1 = test_data['team_1'].values.tolist()
team_2 = test_data['team_2'].values.tolist()
del test_data['team_1']
del test_data['team_2']
del test_data['history']

testx = test_data.copy()
del testx['result']
testx = testx.values.tolist()

data = orig_data.sample(frac=1).reset_index(drop=True)

allx = data.copy()
del allx['result']
allx = allx.values.tolist()

ally = data['result'].values.tolist()
ally = [ [1,0] if y == 'win' else [0,1] for y in ally ]

index1 = int(len(allx)*.75)
index2 = int(len(allx)*1.0)

trainx = allx[:index1]
trainy = ally[:index1]
validx = allx[index1:index2]
validy = ally[index1:index2]

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
team_results = {}
for i in range(len(team_1)):
	print(team_1[i],'vs',team_2[i],probabilities[i])
	
	winner = team_1[i]
	if probabilities[i][0] < probabilities[i][1]:
		winner = team_2[i]
	
	if winner in team_results.keys():
		team_results[winner] += 1
	else:
		team_results[winner] = 1
pprint(team_results)